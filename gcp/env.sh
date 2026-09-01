#!/usr/bin/env bash
# Shared settings for the GCP scripts. Override any of these in your shell
# before running, e.g.  ZONE=us-west1-b ./gcp/create_vm.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-nanollm-507220}"
ZONE="${ZONE:-us-central1-a}"
REGION="${REGION:-${ZONE%-*}}"
INSTANCE="${INSTANCE:-nanollm-train}"

# L4 rather than T4 on purpose: training runs in bfloat16, and T4 (Turing,
# sm_75) has no hardware bf16. L4 is Ada (sm_89) and is the cheapest GCP GPU
# with native bf16 support.
#
# G2 machine types have a *fixed* GPU count baked into the shape, so the count
# is chosen by picking the machine type — never with --accelerator, which is an
# error on G2. NUM_GPUS is the knob; MACHINE_TYPE follows unless overridden.
NUM_GPUS="${NUM_GPUS:-4}"

case "${NUM_GPUS}" in
  1) DEFAULT_MACHINE="g2-standard-8"  ;;
  2) DEFAULT_MACHINE="g2-standard-24" ;;
  4) DEFAULT_MACHINE="g2-standard-48" ;;
  8) DEFAULT_MACHINE="g2-standard-96" ;;
  *) echo "NUM_GPUS=${NUM_GPUS} has no G2 shape (use 1, 2, 4 or 8)" >&2; exit 1 ;;
esac
MACHINE_TYPE="${MACHINE_TYPE:-${DEFAULT_MACHINE}}"

BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-300GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"

# Deep Learning VM: CUDA toolkit, driver and PyTorch preinstalled.
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-latest-gpu}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

# Spot instances are ~60-70% cheaper but can be reclaimed at any time. The
# training loop checkpoints every epoch and supports --resume=auto, so the cost
# of reclamation is at most one epoch.
#
# Defaults to OFF because Spot draws on a *separate* quota
# (PREEMPTIBLE_NVIDIA_L4_GPUS) — enabling it means one more quota request.
# Set SPOT=1 once that quota is granted.
SPOT="${SPOT:-0}"

BUCKET="${BUCKET:-gs://${PROJECT_ID}-nanollm}"
REMOTE_DIR="${REMOTE_DIR:-~/LLMs}"

gcloud_() { gcloud --project="${PROJECT_ID}" "$@"; }
