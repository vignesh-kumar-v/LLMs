"""Training configuration.

Previously a flat module of globals. It is now a dataclass so that every value
can be overridden from the command line without editing the file — which
matters once runs are launched on a remote GPU VM through `torchrun`, where
editing source between runs is awkward.

    python main.py --num_epochs=5 --batch_size=64 --wandb_mode=offline

Model defaults are larger than the original 128-dim/4-layer network. At that
size a multi-GPU run is dominated by launch and gradient-sync overhead, so DDP
would show *negative* scaling and prove nothing. 384-dim/6-layer is still small
(~30M params, minutes per epoch on an L4) but large enough that the GPUs are
actually the bottleneck. Set them back with flags if you want the tiny model.
"""

import argparse
from dataclasses import dataclass, fields


@dataclass
class Config:
    # ── Data ────────────────────────────────────────────────────────────────
    data_dir: str = "."
    context_length: int = 256
    batch_size: int = 32          # per-device batch size
    grad_accum_steps: int = 1     # effective batch = batch_size * accum * world
    steps_per_epoch: int = 1000
    val_steps: int = 100
    num_epochs: int = 20

    # ── Model ───────────────────────────────────────────────────────────────
    num_embeddings: int = 384
    num_heads: int = 6
    num_blocks: int = 6
    dropout: float = 0.1
    tie_weights: bool = True
    use_flash: bool = True        # F.scaled_dot_product_attention
    fused_ln_fallback: bool = False  # True = use F.layer_norm, for A/B testing

    # ── Optimisation ────────────────────────────────────────────────────────
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    decay_lr: bool = True

    # ── Precision / compilation ─────────────────────────────────────────────
    dtype: str = "bfloat16"       # bfloat16 | float16 | float32
    use_compile: bool = True      # set False when profiling with ncu

    # ── Distributed ─────────────────────────────────────────────────────────
    # "ddp" is right for this model size. FSDP shards params/optimiser state and
    # only pays off when the model no longer fits comfortably on one device.
    strategy: str = "ddp"         # ddp | fsdp | none
    backend: str = "nccl"

    # ── Logging ─────────────────────────────────────────────────────────────
    wandb_project: str = "nanollm"
    wandb_entity: str = ""        # empty = W&B default entity
    wandb_run_name: str = ""
    wandb_mode: str = "online"    # online | offline | disabled
    use_tensorboard: bool = False
    log_interval: int = 10        # steps between scalar logs
    hist_interval: int = 500      # steps between weight/grad histograms

    # ── Checkpointing ───────────────────────────────────────────────────────
    out_dir: str = "checkpoints"
    resume: str = ""              # path to a checkpoint, or "auto"
    save_every_epoch: bool = True

    # ── Misc ────────────────────────────────────────────────────────────────
    seed: int = 1337
    generate_after_training: bool = True
    generate_tokens: int = 500
    generate_prompt: str = "Once upon a time"

    @property
    def head_size(self) -> int:
        return self.num_embeddings // self.num_heads

    @classmethod
    def from_cli(cls, argv=None) -> "Config":
        parser = argparse.ArgumentParser(
            description="NanoLLM training configuration",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        for f in fields(cls):
            if f.type is bool or f.type == "bool":
                # Accept --flag, --flag=true, --flag=false uniformly.
                parser.add_argument(
                    f"--{f.name}",
                    type=_str2bool,
                    nargs="?",
                    const=True,
                    default=f.default,
                )
            else:
                parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            print(f"[config] ignoring unrecognised args: {unknown}")
        return cls(**vars(args))

    def describe(self) -> str:
        return "\n".join(f"  {f.name:24s} = {getattr(self, f.name)}" for f in fields(self))


def _str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


#: Module-level default, so `import config; config.defaults.batch_size` works
#: for quick scripts that do not want CLI parsing.
defaults = Config()
