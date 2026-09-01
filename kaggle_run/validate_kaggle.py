"""Kaggle validation kernel — verify the CUDA kernels and NCCL DDP on 2x T4.

Run this before any long training job. It is cheap (a few minutes of GPU quota)
and checks the things that have never touched a GPU:

  1. `fused_layernorm_train.cu` actually compiles with nvcc for sm_75.
  2. Forward/backward numerics vs. an fp32 reference, in fp32 and fp16.
  3. The V1/V2/V3 benchmark kernels still build and run.
  4. Real NCCL DDP across 2 GPUs, with the fused LayerNorm on the hot path.
  5. torch.compile tracing through the registered custom ops.

Uses a tiny synthetic corpus rather than TinyStories so nothing waits on a
~10 minute dataset download.
"""

import os
import random
import shutil
import subprocess
import sys

SRC_INPUT_ROOT = "/kaggle/input"
WORK_DIR = "/kaggle/working"
TEMP_DIR = "/kaggle/temp"
STAGE_DIR = os.path.join(TEMP_DIR, "nanollm")


def run(cmd, check=True, **kwargs):
    printable = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n$ {printable}", flush=True)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), **kwargs)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED (exit {r.returncode}): {printable}")
    return r.returncode


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


def make_corpus(path, n_docs, seed=0):
    random.seed(seed)
    words = ("once upon a time there was a little girl named lily she liked to "
             "play in the park with her dog and they were very happy").split()
    with open(path, "w") as f:
        for _ in range(n_docs):
            f.write(" ".join(random.choice(words) for _ in range(60)))
            f.write("\n<|endoftext|>\n")


def main():
    print("=" * 70)
    print("NanoLLM validation on Kaggle")
    print("=" * 70)

    run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total",
         "--format=csv"])

    import torch

    n_gpu = torch.cuda.device_count()
    cap = torch.cuda.get_device_capability(0) if n_gpu else (0, 0)
    print(f"\ntorch {torch.__version__} | GPUs={n_gpu} | sm_{cap[0]}{cap[1]}")
    if n_gpu < 2:
        print(f"WARNING: {n_gpu} GPU visible — set accelerator to 'GPU T4 x2'")

    src = find_source_dir()
    if os.path.isdir(STAGE_DIR):
        shutil.rmtree(STAGE_DIR)
    shutil.copytree(src, STAGE_DIR)
    os.chdir(STAGE_DIR)
    print(f"Staged source from {src}")

    run([sys.executable, "-m", "pip", "install", "-q", "tiktoken", "ninja"])

    results = {}

    # ── 1-3. Kernel compilation + correctness + benchmarks ──────────────────
    print("\n" + "=" * 70)
    print("STEP 1 — CUDA kernel compile, correctness and benchmarks")
    print("=" * 70)
    results["kernels"] = run([sys.executable, "test_layernorm.py"],
                             cwd=STAGE_DIR, check=False)

    # ── 4-5. Real DDP with the fused kernel on the hot path ─────────────────
    print("\n" + "=" * 70)
    print("STEP 2 — NCCL DDP across 2 GPUs (fused LayerNorm live)")
    print("=" * 70)
    data_dir = os.path.join(TEMP_DIR, "smoke")
    os.makedirs(data_dir, exist_ok=True)
    make_corpus(os.path.join(data_dir, "train.txt"), 4000)
    make_corpus(os.path.join(data_dir, "val.txt"), 400, seed=1)

    nproc = max(1, n_gpu)
    env = dict(os.environ, TORCH_NCCL_ASYNC_ERROR_HANDLING="1", OMP_NUM_THREADS="2")
    results["ddp"] = run([
        "torchrun", "--standalone", f"--nproc_per_node={nproc}", "main.py",
        f"--data_dir={data_dir}", f"--out_dir={os.path.join(TEMP_DIR, 'smoke_ckpt')}",
        "--context_length=256", "--batch_size=16",
        "--num_embeddings=384", "--num_heads=6", "--num_blocks=6",
        "--steps_per_epoch=30", "--val_steps=5", "--num_epochs=1",
        "--warmup_steps=5", "--dtype=bfloat16",  # exercises the sm_75 fallback
        "--use_compile=false", "--wandb_mode=disabled",
        "--generate_after_training=false", "--log_interval=10",
    ], cwd=STAGE_DIR, env=env, check=False)

    # ── torch.compile is the riskiest interaction; test it separately so a
    #    failure here is distinguishable from a plain DDP failure.
    print("\n" + "=" * 70)
    print("STEP 3 — torch.compile tracing through the custom ops")
    print("=" * 70)
    results["compile"] = run([
        "torchrun", "--standalone", f"--nproc_per_node={nproc}", "main.py",
        f"--data_dir={data_dir}", f"--out_dir={os.path.join(TEMP_DIR, 'compile_ckpt')}",
        "--context_length=256", "--batch_size=16",
        "--num_embeddings=384", "--num_heads=6", "--num_blocks=6",
        "--steps_per_epoch=15", "--val_steps=3", "--num_epochs=1",
        "--warmup_steps=3", "--dtype=bfloat16", "--use_compile=true",
        "--wandb_mode=disabled", "--generate_after_training=false",
        "--log_interval=5",
    ], cwd=STAGE_DIR, env=env, check=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, code in results.items():
        print(f"  {name:10s} {'PASS' if code == 0 else f'FAIL (exit {code})'}")
    failed = [k for k, v in results.items() if v != 0]
    print("\nVERDICT:", "ALL PASS" if not failed else f"FAILURES: {failed}")


if __name__ == "__main__":
    main()
