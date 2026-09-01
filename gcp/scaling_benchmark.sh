#!/usr/bin/env bash
# Measure DDP throughput scaling across 1, 2 and 4 GPUs on the training VM.
#
#   ./gcp/scaling_benchmark.sh            # 1,2,4 GPUs
#   GPU_COUNTS="1 2" ./gcp/scaling_benchmark.sh
#
# Runs a short, fixed workload at each GPU count and reports tokens/sec plus
# scaling efficiency relative to a single GPU. This is the number that makes a
# multi-GPU claim concrete — "we use DDP" says nothing without it.
#
# Per-device batch size is held constant, so the *global* batch grows with GPU
# count. That is weak scaling: the right measurement for data parallelism,
# where the point is to process more tokens per unit time.
set -euo pipefail
source "$(dirname "$0")/env.sh"

GPU_COUNTS="${GPU_COUNTS:-1 2 4}"
STEPS="${STEPS:-60}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Scaling benchmark on ${INSTANCE} — GPU counts: ${GPU_COUNTS}"
echo

gcloud_ compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
set -euo pipefail
cd ${REMOTE_DIR}
AVAIL=\$(nvidia-smi --list-gpus | wc -l)
echo \"GPUs available on this VM: \${AVAIL}\"
echo

for N in ${GPU_COUNTS}; do
  if [ \"\${N}\" -gt \"\${AVAIL}\" ]; then
    echo \"--- \${N} GPU(s): skipped (only \${AVAIL} available) ---\"
    continue
  fi
  echo \"--- \${N} GPU(s) ---\"
  # compile disabled: a 60-step run would be dominated by compilation time,
  # which would understate throughput and differ between GPU counts.
  WANDB_MODE=disabled torchrun --standalone --nproc_per_node=\${N} main.py \
      --steps_per_epoch=${STEPS} --val_steps=5 --num_epochs=1 \
      --use_compile=false --wandb_mode=disabled \
      --save_every_epoch=false --generate_after_training=false \
      --log_interval=10 --out_dir=/tmp/scaling_\${N} ${EXTRA_ARGS} \
    2>&1 | grep -E 'step |Epoch|tok/s' | tail -5
  echo
done
" | tee /tmp/nanollm_scaling.txt

echo
echo "── Summary ──────────────────────────────────────────────────────"
python3 - /tmp/nanollm_scaling.txt <<'PY'
import re, sys

text = open(sys.argv[1]).read()
results = {}
current = None
for line in text.splitlines():
    m = re.search(r"---\s*(\d+) GPU\(s\)\s*---", line)
    if m:
        current = int(m.group(1))
        continue
    m = re.search(r"([\d.]+)k tok/s", line)
    if m and current:
        # keep the last (warmest) reading for this GPU count
        results[current] = float(m.group(1)) * 1000

if not results:
    print("  No throughput readings parsed — check /tmp/nanollm_scaling.txt")
    raise SystemExit(0)

base_n = min(results)
base = results[base_n]
print(f"  {'GPUs':>5}  {'tokens/sec':>12}  {'speedup':>8}  {'efficiency':>10}")
for n in sorted(results):
    speedup = results[n] / base
    ideal = n / base_n
    print(f"  {n:>5}  {results[n]:>12,.0f}  {speedup:>7.2f}x  {speedup/ideal*100:>9.0f}%")
print()
print("  Efficiency = achieved speedup / ideal linear speedup.")
print("  80%+ at 2 GPUs is healthy for a model this size; gradient all-reduce")
print("  is a fixed cost per step, so efficiency falls as GPU count rises.")
PY
