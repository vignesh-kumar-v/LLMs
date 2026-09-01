"""Kaggle kernel entrypoint — runs NanoLLM DDP training on 2x NVIDIA T4.

This file is what Kaggle executes. It is deliberately self-contained and does
NOT import the project, because Kaggle script kernels run a single file; the
rest of the source arrives as an attached Dataset and is staged onto disk here.

Layout on Kaggle:
    /kaggle/input/<src-dataset>/   source, read-only
    /kaggle/temp/                  scratch (NOT saved) — dataset + .bin files
    /kaggle/working/               kernel output (downloadable) — ckpts, logs

Data goes to /kaggle/temp on purpose: TinyStories is ~2 GB of text plus ~1 GB
of tokens, and anything left in /kaggle/working has to be uploaded as kernel
output at the end of the session.

Precision: T4 is Turing (sm_75) and has **no hardware bf16**. `main.py` detects
this and falls back to fp16 + GradScaler automatically, but we pass
--dtype=float16 explicitly so the intent is visible in the logs.
"""

import os
import shutil
import subprocess
import sys
import time

SRC_INPUT_ROOT = "/kaggle/input"
WORK_DIR = "/kaggle/working"
TEMP_DIR = "/kaggle/temp"
STAGE_DIR = os.path.join(TEMP_DIR, "nanollm")

# Overridable by editing this block before pushing the kernel.
TRAIN_ARGS = os.environ.get("NANOLLM_ARGS", "").split() or [
    # ~8 x 4000 x (24 x 256 x 2 GPUs) = ~390M tokens, roughly one pass over
    # TinyStories. Sized to finish inside Kaggle's 9h session cap with room for
    # the ~30 min download + tokenisation.
    "--num_epochs=8",
    "--steps_per_epoch=4000",
    "--val_steps=100",
    "--batch_size=24",
    "--context_length=256",
    "--num_embeddings=384",
    "--num_heads=6",
    "--num_blocks=6",
    "--dtype=float16",
    "--use_compile=false",
]


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    result = subprocess.run(cmd, shell=isinstance(cmd, str), **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"command failed with code {result.returncode}")
    return result


def find_source_dir():
    """Locate the attached source dataset by walking /kaggle/input for main.py.

    Kaggle's mount layout is not stable across dataset types — it has been
    observed as /kaggle/input/<slug>/ and as /kaggle/input/datasets/<owner>/
    <slug>/ — so search rather than assume a fixed depth.
    """
    if not os.path.isdir(SRC_INPUT_ROOT):
        raise SystemExit(f"{SRC_INPUT_ROOT} missing - is a Dataset attached?")
    for root, dirs, files in os.walk(SRC_INPUT_ROOT):
        if "main.py" in files and "NanoLLM.py" in files:
            return root
    listing = []
    for root, dirs, files in os.walk(SRC_INPUT_ROOT):
        listing.append(f"{root}: {files[:6]}")
    raise SystemExit(
        "No attached dataset contains main.py.\n" + "\n".join(listing[:25])
    )


def main():
    t0 = time.time()
    print("=" * 70)
    print("NanoLLM on Kaggle")
    print("=" * 70)

    run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
         "--format=csv"])

    import torch

    n_gpu = torch.cuda.device_count()
    print(f"\ntorch {torch.__version__} | CUDA {torch.version.cuda} | GPUs: {n_gpu}")
    if n_gpu:
        cap = torch.cuda.get_device_capability(0)
        print(f"compute capability sm_{cap[0]}{cap[1]} "
              f"(bf16 requires sm_80+, so fp16 is used here)")
    if n_gpu < 2:
        print(f"\nWARNING: only {n_gpu} GPU visible. For 2x T4 set the notebook "
              f"accelerator to 'GPU T4 x2'. Continuing on {n_gpu}.")

    # ── Stage source ────────────────────────────────────────────────────────
    src = find_source_dir()
    print(f"\nSource dataset: {src}")
    if os.path.isdir(STAGE_DIR):
        shutil.rmtree(STAGE_DIR)
    shutil.copytree(src, STAGE_DIR)
    os.chdir(STAGE_DIR)
    print(f"Staged to {STAGE_DIR}: {sorted(os.listdir('.'))[:12]}")

    # ── Dependencies (torch is preinstalled on Kaggle) ──────────────────────
    run([sys.executable, "-m", "pip", "install", "-q",
         "tiktoken", "datasets", "wandb", "ninja"])

    # ── W&B key from Kaggle Secrets, if present ─────────────────────────────
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        wandb_mode = "online"
        print("W&B: using key from Kaggle Secrets")
    except Exception as exc:  # noqa: BLE001
        wandb_mode = "offline"
        print(f"W&B: offline ({exc.__class__.__name__}) — "
              "add a WANDB_API_KEY secret to log online")
    os.environ.setdefault("WANDB_DIR", WORK_DIR)

    # ── Dataset ─────────────────────────────────────────────────────────────
    os.makedirs(TEMP_DIR, exist_ok=True)
    data_dir = os.path.join(TEMP_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(os.path.join(data_dir, "train.txt")):
        print("\nDownloading TinyStories (needs Internet enabled on the kernel) ...")
        run([sys.executable, "TinyStories.py"], cwd=STAGE_DIR)
        for name in ("train.txt", "val.txt"):
            shutil.move(os.path.join(STAGE_DIR, name), os.path.join(data_dir, name))
    else:
        print("\nDataset already staged.")

    # ── Train ───────────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(WORK_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    nproc = max(1, n_gpu)
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={nproc}", "main.py",
        f"--data_dir={data_dir}", f"--out_dir={ckpt_dir}",
        f"--wandb_mode={wandb_mode}", "--resume=auto",
        *TRAIN_ARGS,
    ]
    env = dict(os.environ, TORCH_NCCL_ASYNC_ERROR_HANDLING="1", OMP_NUM_THREADS="2")
    run(cmd, cwd=STAGE_DIR, env=env)

    # ── Collect artefacts into the kernel output ────────────────────────────
    # main.py writes the plot into out_dir, which already lives under
    # /kaggle/working, so nothing needs copying; kept as a fallback for older
    # runs that wrote it beside the source.
    legacy = os.path.join(STAGE_DIR, "training_stats.png")
    if os.path.exists(legacy):
        shutil.copy(legacy, os.path.join(WORK_DIR, "training_stats.png"))

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print("Output files:")
    for root, _, files in os.walk(WORK_DIR):
        for f in files:
            p = os.path.join(root, f)
            print(f"  {p}  ({os.path.getsize(p)/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
