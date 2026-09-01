"""NanoLLM training entrypoint — single-GPU, multi-GPU (DDP) or FSDP.

Single GPU:
    python main.py

Two GPUs on one node:
    torchrun --standalone --nproc_per_node=2 main.py

Multi-node:
    torchrun --nnodes=2 --node_rank=0 --master_addr=... --nproc_per_node=2 main.py

Everything in `config.py` is overridable on the command line, e.g.
    torchrun --standalone --nproc_per_node=2 main.py --num_epochs=5 --batch_size=64

Notable differences from the original single-GPU loop:

* Distributed data-parallel training, with gradient accumulation that skips
  the all-reduce on every micro-step but the last (`no_sync`), so accumulation
  does not multiply communication volume.
* Per-*step* cosine LR with linear warmup. The original stepped the scheduler
  once per epoch, giving a 50-point schedule across the whole run.
* Weights & Biases tracking replaces the ad-hoc TensorBoard writes; GPU memory
  and utilisation now land on the same dashboard as loss and throughput.
* Checkpoints save every epoch *and* on best val loss, and training can resume
  from either — the original only ever wrote `best_model.pt`, so a crash lost
  all progress made since the last improvement.
"""

import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import fused_ln
from config import Config
from CrossEntropyLoss import CrossEntropyLoss
from data import TokenStream, prepare_bin
from NanoLLM import GPT2, Block


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

class DistInfo:
    def __init__(self, cfg):
        self.enabled = int(os.environ.get("RANK", -1)) != -1 and cfg.strategy != "none"
        if self.enabled:
            backend = cfg.backend
            if backend == "nccl" and not torch.cuda.is_available():
                # Lets the distributed code path be exercised on a CPU box.
                backend = "gloo"
            dist.init_process_group(backend=backend)
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            if torch.cuda.is_available():
                self.device = f"cuda:{self.local_rank}"
                torch.cuda.set_device(self.local_rank)
            else:
                self.device = "cpu"
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_master(self):
        return self.rank == 0

    @property
    def device_type(self):
        return "cuda" if self.device.startswith("cuda") else "cpu"

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def all_reduce_mean(self, value):
        """Average a python float across ranks (for logging only)."""
        if not self.enabled:
            return value
        t = torch.tensor([value], device=self.device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t / self.world_size).item()

    def cleanup(self):
        if self.enabled:
            dist.destroy_process_group()


def log(msg, dist_info):
    if dist_info.is_master:
        print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Schedule
# ─────────────────────────────────────────────────────────────────────────────

def lr_at(step, cfg, total_steps):
    """Linear warmup, then cosine decay to `min_lr`."""
    if not cfg.decay_lr:
        return cfg.learning_rate
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    if step >= total_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# GPU telemetry
# ─────────────────────────────────────────────────────────────────────────────

def gpu_stats():
    if not torch.cuda.is_available():
        return {}
    stats = {
        "gpu/mem_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "gpu/mem_peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu/mem_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
    }
    try:
        stats["gpu/utilization_pct"] = torch.cuda.utilization()
    except Exception:
        pass  # pynvml unavailable — memory numbers are still useful
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def gather_state(base_model, model, optimizer, cfg, dinfo):
    """Collect model + optimizer state for saving.

    Under FSDP the parameters are sharded across ranks, so the state has to be
    gathered through FSDP's own APIs. Both calls are *collective*: every rank
    must reach them, even though only rank 0 ends up holding the full tensors.
    """
    if dinfo.enabled and cfg.strategy == "fsdp":
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)
        return model_state, optim_state

    return base_model.state_dict(), optimizer.state_dict()


def save_checkpoint(path, model_state, optim_state, cfg, epoch, step, best_val, val_loss):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "config": asdict(cfg),
            "epoch": epoch,
            "step": step,
            "best_val_loss": best_val,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(path, base_model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("best_val_loss", float("inf"))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = Config.from_cli()
    dinfo = DistInfo(cfg)

    torch.manual_seed(cfg.seed + dinfo.rank)
    np.random.seed(cfg.seed + dinfo.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    log("=" * 72, dinfo)
    log(f"NanoLLM training | world_size={dinfo.world_size} | strategy={cfg.strategy}", dinfo)
    if dinfo.device_type == "cuda":
        log(f"Device: {torch.cuda.get_device_name(dinfo.local_rank)}", dinfo)
    log("=" * 72, dinfo)

    # ── Tokenizer + data ────────────────────────────────────────────────────
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    train_txt = os.path.join(cfg.data_dir, "train.txt")
    val_txt = os.path.join(cfg.data_dir, "val.txt")
    train_bin = os.path.join(cfg.data_dir, "train.bin")
    val_bin = os.path.join(cfg.data_dir, "val.bin")

    # Only rank 0 writes the .bin files; the others wait, or they would race
    # on the same output path.
    if dinfo.is_master:
        prepare_bin(train_txt, train_bin, tokenizer)
        prepare_bin(val_txt, val_bin, tokenizer)
    dinfo.barrier()

    train_stream = TokenStream(train_bin, cfg.context_length, dinfo.device,
                               seed=cfg.seed, rank=dinfo.rank)
    val_stream = TokenStream(val_bin, cfg.context_length, dinfo.device,
                             seed=cfg.seed, rank=dinfo.rank)
    log(f"[data] train tokens: {len(train_stream):,} | val tokens: {len(val_stream):,}", dinfo)

    # ── Model ───────────────────────────────────────────────────────────────
    base_model = GPT2(
        context_size=cfg.context_length,
        vocab_size=vocab_size,
        num_embeddings=cfg.num_embeddings,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
        tie_weights=cfg.tie_weights,
        use_flash=cfg.use_flash,
        fused_ln_fallback=cfg.fused_ln_fallback,
    ).to(dinfo.device)

    if dinfo.device_type == "cuda" and not cfg.fused_ln_fallback:
        fused_ln.load_extension(verbose=False)
    log(f"[model] parameters: {base_model.num_parameters():,}", dinfo)
    log(f"[model] LayerNorm path: {fused_ln.kernel_status()}", dinfo)

    optimizer, n_decay, n_nodecay = base_model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), dinfo.device_type
    )
    log(f"[optim] decayed tensors: {n_decay} | undecayed: {n_nodecay}", dinfo)

    start_epoch, global_step, best_val_loss = 0, 0, float("inf")
    resume_path = cfg.resume
    if resume_path == "auto":
        candidate = os.path.join(cfg.out_dir, "last.pt")
        resume_path = candidate if os.path.exists(candidate) else ""
    if resume_path:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_path, base_model, optimizer, dinfo.device
        )
        log(f"[resume] {resume_path} @ epoch {start_epoch}, step {global_step}", dinfo)

    # Wrap for distribution first, then compile the wrapper. FSDP in particular
    # expects to see the un-compiled module so it can place its sharding hooks
    # on the real submodules.
    model = base_model
    if dinfo.enabled:
        if cfg.strategy == "fsdp":
            import functools

            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            policy = functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls={Block}
            )
            model = FSDP(model, auto_wrap_policy=policy, device_id=dinfo.local_rank)
        else:
            # device_ids is only valid for single-CUDA-device processes.
            device_ids = [dinfo.local_rank] if dinfo.device_type == "cuda" else None
            model = DDP(model, device_ids=device_ids)

    if cfg.use_compile:
        log("[model] compiling (first step will be slow) ...", dinfo)
        model = torch.compile(model)

    criterion = CrossEntropyLoss()

    # ── Precision ───────────────────────────────────────────────────────────
    ptdtype = {"float32": torch.float32,
               "bfloat16": torch.bfloat16,
               "float16": torch.float16}[cfg.dtype]

    # bf16 needs compute capability 8.0+ (Ampere). On Turing (T4, sm_75) it is
    # either rejected or emulated in software at a large slowdown, so fall back
    # to fp16 — which Turing does support in hardware, via its tensor cores.
    # fp16 then needs loss scaling, handled by the GradScaler below.
    if ptdtype == torch.bfloat16 and dinfo.device_type == "cuda":
        major, minor = torch.cuda.get_device_capability(dinfo.local_rank)
        if major < 8:
            name = torch.cuda.get_device_name(dinfo.local_rank)
            log(f"[precision] {name} is sm_{major}{minor}: no hardware bf16. "
                f"Falling back to float16 + GradScaler.", dinfo)
            ptdtype = torch.float16

    use_amp = dinfo.device_type == "cuda" and ptdtype != torch.float32
    amp_ctx = (torch.autocast(device_type="cuda", dtype=ptdtype)
               if use_amp else nullcontext())
    # bf16 shares fp32's exponent range, so it needs no loss scaling; fp16 does.
    scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16 and use_amp))

    total_steps = cfg.num_epochs * cfg.steps_per_epoch
    tokens_per_step = (cfg.batch_size * cfg.grad_accum_steps
                       * cfg.context_length * dinfo.world_size)
    log(f"[train] {total_steps} steps | {tokens_per_step:,} tokens/step", dinfo)

    # ── W&B ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if dinfo.is_master and cfg.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity or None,
                name=cfg.wandb_run_name or None,
                mode=cfg.wandb_mode,
                config={**asdict(cfg),
                        "world_size": dinfo.world_size,
                        "vocab_size": vocab_size,
                        "parameters": base_model.num_parameters(),
                        "tokens_per_step": tokens_per_step},
            )
            wandb.watch(base_model, log="all", log_freq=cfg.hist_interval)
            log(f"[wandb] logging to {wandb_run.url}", dinfo)
        except Exception as exc:  # noqa: BLE001
            log(f"[wandb] disabled ({exc.__class__.__name__}: {exc})", dinfo)

    writer = None
    if dinfo.is_master and cfg.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter("runs/training_logs")

    # ── Loop ────────────────────────────────────────────────────────────────
    history = {"train": [], "val": [], "mem": [], "util": []}
    log("Starting training ...", dinfo)

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        if dinfo.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        running_loss, t_epoch = 0.0, time.time()
        t_step = time.time()
        # torch.cuda.utilization() is an *instantaneous* sample. Reading it once
        # at the end of an epoch can land in a quiet moment (checkpoint save,
        # validation boundary) and report ~0% for an otherwise saturated GPU, so
        # sample it throughout the epoch and average.
        util_samples = []

        for step in range(cfg.steps_per_epoch):
            lr = lr_at(global_step, cfg, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            micro_loss = 0.0

            for micro in range(cfg.grad_accum_steps):
                inputs, targets = train_stream.batch(cfg.batch_size)
                is_last_micro = micro == cfg.grad_accum_steps - 1

                # Only all-reduce gradients on the final micro-step; otherwise
                # accumulation would cost one full sync per micro-batch.
                if dinfo.enabled and cfg.strategy == "ddp":
                    model.require_backward_grad_sync = is_last_micro

                with amp_ctx:
                    logits = model(inputs)
                    loss = criterion(logits, targets) / cfg.grad_accum_steps

                scaler.scale(loss).backward()
                micro_loss += loss.item()

            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                if dinfo.enabled and cfg.strategy == "fsdp":
                    # Gradients are sharded, so the norm has to be computed
                    # collectively; the free function would clip per-shard.
                    grad_norm = model.clip_grad_norm_(cfg.grad_clip)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=cfg.grad_clip
                    )
            else:
                grad_norm = torch.tensor(0.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += micro_loss
            global_step += 1

            if dinfo.is_master and global_step % cfg.log_interval == 0:
                dt = time.time() - t_step
                t_step = time.time()
                tok_per_sec = tokens_per_step * cfg.log_interval / max(dt, 1e-9)
                step_stats = gpu_stats()
                if "gpu/utilization_pct" in step_stats:
                    util_samples.append(step_stats["gpu/utilization_pct"])
                metrics = {
                    "train/loss": micro_loss,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "perf/tokens_per_sec": tok_per_sec,
                    "perf/ms_per_step": dt * 1000 / cfg.log_interval,
                    "epoch": epoch + step / cfg.steps_per_epoch,
                    **step_stats,
                }
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)
                if writer is not None:
                    for k, v in metrics.items():
                        writer.add_scalar(k, v, global_step)
                if global_step % (cfg.log_interval * 20) == 0:
                    log(f"  step {global_step}/{total_steps} "
                        f"loss {micro_loss:.4f} lr {lr:.2e} "
                        f"{tok_per_sec/1e3:.1f}k tok/s", dinfo)

        avg_train = dinfo.all_reduce_mean(running_loss / cfg.steps_per_epoch)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for vx, vy in val_stream.fixed_batches(cfg.batch_size, cfg.val_steps,
                                                   seed=cfg.seed):
                with amp_ctx:
                    val_total += criterion(model(vx), vy).item()
        avg_val = dinfo.all_reduce_mean(val_total / cfg.val_steps)

        stats = gpu_stats()
        history["train"].append(avg_train)
        history["val"].append(avg_val)
        history["mem"].append(stats.get("gpu/mem_peak_gb", 0.0))
        # Mean over the epoch's samples, not the single end-of-epoch reading.
        mean_util = (sum(util_samples) / len(util_samples)) if util_samples \
            else stats.get("gpu/utilization_pct", 0.0)
        history["util"].append(mean_util)
        stats["gpu/utilization_mean_pct"] = mean_util

        log(f"Epoch {epoch+1}/{cfg.num_epochs} | train {avg_train:.4f} | "
            f"val {avg_val:.4f} | ppl {math.exp(min(avg_val, 20)):.1f} | "
            f"peak {stats.get('gpu/mem_peak_gb', 0):.2f}GB | "
            f"{time.time()-t_epoch:.0f}s", dinfo)

        if wandb_run is not None:
            wandb_run.log({"train/epoch_loss": avg_train,
                           "val/loss": avg_val,
                           "val/perplexity": math.exp(min(avg_val, 20)),
                           "epoch": epoch + 1,
                           **stats}, step=global_step)
        if writer is not None:
            writer.add_scalar("val/loss", avg_val, epoch)

        # ── Checkpoints ─────────────────────────────────────────────────────
        # gather_state must run on every rank (it is collective under FSDP);
        # only rank 0 writes the file.
        is_best = avg_val < best_val_loss
        if cfg.save_every_epoch or is_best:
            model_state, optim_state = gather_state(base_model, model, optimizer, cfg, dinfo)
            if dinfo.is_master:
                if is_best:
                    best_val_loss = avg_val
                if cfg.save_every_epoch:
                    save_checkpoint(os.path.join(cfg.out_dir, "last.pt"),
                                    model_state, optim_state, cfg, epoch + 1,
                                    global_step, best_val_loss, avg_val)
                if is_best:
                    save_checkpoint(os.path.join(cfg.out_dir, "best_model.pt"),
                                    model_state, optim_state, cfg, epoch + 1,
                                    global_step, best_val_loss, avg_val)
                    log(f"  new best val loss {best_val_loss:.4f} -> best_model.pt", dinfo)
            elif is_best:
                best_val_loss = avg_val
            del model_state, optim_state
        dinfo.barrier()

    log("Training complete.", dinfo)

    # ── Post-training artefacts (rank 0) ────────────────────────────────────
    if dinfo.is_master:
        _plot_history(history, wandb_run)

        if cfg.generate_after_training:
            log("\n--- sample ---", dinfo)
            start = torch.tensor(
                tokenizer.encode(cfg.generate_prompt), dtype=torch.long
            ).unsqueeze(0).to(dinfo.device)
            with amp_ctx:
                out = base_model.generate(start, max_new_tokens=cfg.generate_tokens,
                                          temperature=0.8, top_k=200)
            text = tokenizer.decode(out[0].tolist())
            log(text, dinfo)
            if wandb_run is not None:
                import wandb

                wandb_run.log({"samples": wandb.Table(columns=["text"], data=[[text]])})

        if writer is not None:
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()

    dinfo.cleanup()


def _plot_history(history, wandb_run):
    if not history["train"]:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(epochs, history["train"], label="Train")
    axes[0].plot(epochs, history["val"], label="Val")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training and Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["mem"], label="Peak (GB)", marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Memory (GB)", title="GPU Memory (peak per epoch)")
    # Peak memory is essentially constant across epochs. Left to autoscale,
    # matplotlib renders a ~0.0001 GB window with a "+5.079" offset label, which
    # makes flat memory look like runaway growth. Anchor at 0 and drop the
    # offset so the shape of the line reflects reality.
    axes[1].set_ylim(0, max(history["mem"] + [1e-6]) * 1.3)
    axes[1].ticklabel_format(useOffset=False, axis="y")
    axes[1].legend()

    axes[2].plot(epochs, history["util"], color="green", label="GPU util % (epoch mean)",
                 marker="o")
    axes[2].set(xlabel="Epoch", ylabel="Utilization (%)", title="GPU Utilization")
    axes[2].set_ylim(0, 105)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[plot] wrote training_stats.png")


if __name__ == "__main__":
    main()
