#!/usr/bin/env bash
# Push the working tree to the VM, install deps, and launch DDP training.
#
#   ./gcp/sync_and_train.sh                      # full run, defaults from config.py
#   ./gcp/sync_and_train.sh --num_epochs=2       # extra flags go to main.py
#   NUM_GPUS=1 ./gcp/sync_and_train.sh           # single-GPU baseline
#   SYNC_ONLY=1 ./gcp/sync_and_train.sh          # push code, don't train
#
# Training runs under `nohup` inside a detached session, so closing your laptop
# does not kill the run. Follow along with ./gcp/logs.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SYNC_ONLY="${SYNC_ONLY:-0}"
TRAIN_ARGS=("$@")

echo "Syncing source to ${INSTANCE}:${REMOTE_DIR} ..."
gcloud_ compute ssh "${INSTANCE}" --zone="${ZONE}" --command="mkdir -p ${REMOTE_DIR}"

# Only source is copied. Datasets and checkpoints live on the VM / in GCS —
# they are far too large to push from a laptop.
tar --exclude='.git' --exclude='__pycache__' --exclude='*.bin' \
    --exclude='*.pt' --exclude='*.txt' --exclude='checkpoints' \
    --exclude='wandb' --exclude='runs' --exclude='*.ncu-rep' \
    -czf - . \
  | gcloud_ compute ssh "${INSTANCE}" --zone="${ZONE}" \
      --command="tar -xzf - -C ${REMOTE_DIR}"

echo "Installing dependencies ..."
gcloud_ compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  cd ${REMOTE_DIR}
  sudo apt-get install -y -q ninja-build >/dev/null 2>&1 || true
  pip install -q --upgrade pip
  # torch ships with the DLVM image; do not let pip swap it for a CPU build.
  pip install -q tiktoken datasets wandb matplotlib tqdm tensorboard ninja nvidia-ml-py
  python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"gpus\", torch.cuda.device_count())'
"

if [ "${SYNC_ONLY}" = "1" ]; then
  echo "SYNC_ONLY=1 — stopping here."
  exit 0
fi

# W&B needs a key on the VM. Forward it from the local environment if set,
# otherwise fall back to offline mode so a run is never lost to a missing key.
WANDB_SETUP="export WANDB_MODE=offline"
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_SETUP="export WANDB_API_KEY=${WANDB_API_KEY}"
fi

echo "Preparing dataset (first run downloads TinyStories, ~10 min) ..."
gcloud_ compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  cd ${REMOTE_DIR}
  if [ ! -f train.txt ]; then python TinyStories.py; else echo 'train.txt present'; fi
"

echo "Launching training on ${NUM_GPUS} GPU(s) ..."
gcloud_ compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  cd ${REMOTE_DIR}
  ${WANDB_SETUP}
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  nohup torchrun --standalone --nproc_per_node=${NUM_GPUS} main.py \
      --out_dir=checkpoints --resume=auto ${TRAIN_ARGS[*]:-} \
      > train.log 2>&1 &
  echo \"started pid \$!\"
  sleep 5
  tail -n 20 train.log
"

echo
echo "Training launched. Follow it with:  ./gcp/logs.sh"
echo "Fetch checkpoints with:            ./gcp/fetch_results.sh"
