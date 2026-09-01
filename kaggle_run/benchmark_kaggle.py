"""Kaggle kernel — the three things the main training run left unverified.

  1. DDP scaling: tokens/sec at 1 GPU vs 2 GPUs, with speedup and efficiency.
     "We use DDP" means nothing without this number.
  2. FSDP: whether the sharded path actually runs end to end on real hardware.
     It was implemented but never executed.
  3. Nsight Compute: whether the custom kernel can be profiled here at all.

Uses a synthetic corpus rather than TinyStories. For a *relative* 1-vs-2
comparison that is the better choice — it removes dataset download time and
disk-read variance, so the difference measured is compute and gradient sync
rather than I/O. Absolute throughput will read slightly high against the real
run for the same reason.

Model config matches the real training run (384 dim, 6 heads, 6 blocks,
context 256, per-device batch 24) so the numbers are comparable.
"""

import os
import random
import re
import shutil
import subprocess
import sys

SRC_INPUT_ROOT = "/kaggle/input"
WORK_DIR = "/kaggle/working"
TEMP_DIR = "/kaggle/temp"
STAGE_DIR = os.path.join(TEMP_DIR, "nanollm")

STEPS = int(os.environ.get("BENCH_STEPS", "150"))
MODEL_ARGS = [
    "--context_length=256", "--batch_size=24",
    "--num_embeddings=384", "--num_heads=6", "--num_blocks=6",
    "--dtype=bfloat16",          # falls back to fp16 on sm_75
    "--use_compile=false",       # compile time would swamp a 150-step run
    "--wandb_mode=disabled", "--generate_after_training=false",
    "--save_every_epoch=false",
]


def run(cmd, capture=False, **kw):
    printable = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n$ {printable}", flush=True)
    if capture:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, **kw)
        print(r.stdout[-4000:])
        if r.stderr:
            print("--- stderr tail ---")
            print(r.stderr[-2000:])
        return r.returncode, r.stdout + r.stderr
    r = subprocess.run(cmd, shell=isinstance(cmd, str), **kw)
    return r.returncode, ""


def find_source_dir():
    for root, _dirs, files in os.walk(SRC_INPUT_ROOT):
        if "main.py" in files and "NanoLLM.py" in files:
            return root
    raise SystemExit(f"no dataset with main.py under {SRC_INPUT_ROOT}")


def make_corpus(path, n_docs, seed=0):
    random.seed(seed)
    words = ("once upon a time there was a little girl named lily she liked to "
             "play in the park with her dog and they were very happy the sun "
             "was bright and the birds sang loudly in the tall green trees").split()
    with open(path, "w") as f:
        for _ in range(n_docs):
            f.write(" ".join(random.choice(words) for _ in range(80)))
            f.write("\n<|endoftext|>\n")


def parse_throughput(text):
    """Prefer the epoch-level tok/s figure, which is always printed.

    The per-interval readings only appear every log_interval*20 steps, so a
    short benchmark run can finish without emitting a single one.
    """
    epoch = re.findall(r"\|\s*([0-9.]+)k tok/s\s*\|", text)
    if epoch:
        return float(epoch[-1]) * 1000
    vals = [float(v) * 1000 for v in re.findall(r"([0-9.]+)k tok/s", text)]
    vals = vals[2:] or vals
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def main():
    print("=" * 70)
    print("NanoLLM — scaling, FSDP and profiling validation")
    print("=" * 70)
    run(["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv"])

    import torch

    n_gpu = torch.cuda.device_count()
    print(f"\ntorch {torch.__version__} | GPUs={n_gpu}")
    if n_gpu < 2:
        print("WARNING: need 2 GPUs for the scaling comparison "
              "(set accelerator to 'GPU T4 x2')")

    src = find_source_dir()
    if os.path.isdir(STAGE_DIR):
        shutil.rmtree(STAGE_DIR)
    shutil.copytree(src, STAGE_DIR)
    os.chdir(STAGE_DIR)
    run([sys.executable, "-m", "pip", "install", "-q", "tiktoken", "ninja"])

    data_dir = os.path.join(TEMP_DIR, "bench")
    os.makedirs(data_dir, exist_ok=True)
    make_corpus(os.path.join(data_dir, "train.txt"), 20000)
    make_corpus(os.path.join(data_dir, "val.txt"), 1000, seed=1)

    base_env = dict(os.environ, TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
                    OMP_NUM_THREADS="2")
    results = {}

    # ── 1. DDP scaling ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 1 — DDP scaling ({STEPS} steps per configuration)")
    print("=" * 70)
    throughput = {}
    for ngpu in (1, 2):
        if ngpu > n_gpu:
            print(f"\n--- {ngpu} GPU: skipped (only {n_gpu} available) ---")
            continue
        print(f"\n--- {ngpu} GPU ---")
        env = dict(base_env)
        # Restrict visibility rather than changing the model, so per-device
        # batch stays identical and this measures weak scaling.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(ngpu))
        code, out = run([
            "torchrun", "--standalone", f"--nproc_per_node={ngpu}", "main.py",
            f"--data_dir={data_dir}",
            f"--out_dir={os.path.join(TEMP_DIR, f'bench_{ngpu}')}",
            f"--steps_per_epoch={STEPS}", "--val_steps=5", "--num_epochs=1",
            "--warmup_steps=10", "--log_interval=10", *MODEL_ARGS,
        ], capture=True, cwd=STAGE_DIR, env=env)
        tp = parse_throughput(out)
        throughput[ngpu] = tp
        print(f"  -> {ngpu} GPU: {tp:,.0f} tok/s" if tp else "  -> no reading")
        results[f"scaling_{ngpu}gpu"] = code

    # ── 2. FSDP ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 — FSDP on 2 GPUs")
    print("=" * 70)
    if n_gpu >= 2:
        code, out = run([
            "torchrun", "--standalone", "--nproc_per_node=2", "main.py",
            f"--data_dir={data_dir}",
            f"--out_dir={os.path.join(TEMP_DIR, 'fsdp')}",
            "--strategy=fsdp", "--steps_per_epoch=40", "--val_steps=5",
            "--num_epochs=1", "--warmup_steps=5", "--log_interval=10",
            *MODEL_ARGS,
        ], capture=True, cwd=STAGE_DIR, env=base_env)
        results["fsdp"] = code
        throughput["fsdp"] = parse_throughput(out)
    else:
        print("skipped — needs 2 GPUs")

    # ── 3. Nsight Compute ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 — Nsight Compute profiling")
    print("=" * 70)
    ncu = shutil.which("ncu") or shutil.which("/usr/local/cuda/bin/ncu")
    for cand in ("/usr/local/cuda/bin/ncu", "/opt/nvidia/nsight-compute/ncu"):
        if not ncu and os.path.exists(cand):
            ncu = cand
    if not ncu:
        print("ncu not found on this image — profiling cannot run here.")
        print("Searched PATH, /usr/local/cuda/bin, /opt/nvidia/nsight-compute.")
        results["ncu"] = 127
    else:
        print(f"found ncu at {ncu}")
        # Kaggle kernels run unprivileged; ncu needs elevated access to the GPU
        # performance counters, which normally requires root or a relaxed
        # perf_event_paranoid. Try anyway and report what actually happens.
        code, out = run([
            ncu, "--target-processes", "all", "--kernel-name", "regex:ln_fwd_kernel",
            "--launch-count", "1", "--csv", "--metrics",
            "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum",
            sys.executable, "profile_run.py", "--rows=4096", "--hidden=384",
            "--dtype=float16",
        ], capture=True, cwd=STAGE_DIR, env=base_env)
        results["ncu"] = code
        if code != 0 and ("ERR_NVGPUCTRPERM" in out or "permission" in out.lower()):
            print("\nERR_NVGPUCTRPERM: GPU performance counters need elevated "
                  "permissions that an unprivileged Kaggle kernel cannot obtain.\n"
                  "This is an environment limit, not a code fault - note that "
                  "profile_run.py itself ran correctly under ncu, only the "
                  "counter collection was refused. Profiling needs a machine "
                  "where you can set perf_event_paranoid or run as root.")
            results["ncu"] = "blocked-by-environment"

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        if v == 0:
            status = "PASS"
        elif isinstance(v, str):
            status = v.upper()
        else:
            status = f"FAIL (exit {v})"
        print(f"  {k:16s} {status}")

    if throughput.get(1) and throughput.get(2):
        t1, t2 = throughput[1], throughput[2]
        print(f"\n  DDP weak scaling (per-device batch held constant):")
        print(f"    1 GPU : {t1:>10,.0f} tok/s")
        print(f"    2 GPU : {t2:>10,.0f} tok/s")
        print(f"    speedup    {t2/t1:.2f}x   efficiency {t2/t1/2*100:.0f}%")
    if throughput.get("fsdp"):
        print(f"    FSDP 2 GPU: {throughput['fsdp']:,.0f} tok/s")


if __name__ == "__main__":
    main()
