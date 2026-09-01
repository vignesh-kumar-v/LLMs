#!/usr/bin/env bash
# Create the GPU training VM (and the checkpoint bucket, if missing).
#
#   ./gcp/create_vm.sh
#
# Requires GPU quota. Two quotas matter and BOTH must be raised — the global
# one is the usual blocker because it defaults to 0:
#   * GPUS_ALL_REGIONS  (global)          >= NUM_GPUS
#   * NVIDIA_L4_GPUS    (per region)      >= NUM_GPUS
# Check with: ./gcp/check_quota.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "Project      : ${PROJECT_ID}"
echo "Zone         : ${ZONE}"
echo "Instance     : ${INSTANCE}"
echo "Machine type : ${MACHINE_TYPE}  (${NUM_GPUS} x L4)"
echo "Provisioning : $([ "${SPOT}" = "1" ] && echo SPOT || echo on-demand)"
echo

if gcloud_ compute instances describe "${INSTANCE}" --zone="${ZONE}" >/dev/null 2>&1; then
  echo "Instance ${INSTANCE} already exists. Start it with:"
  echo "  gcloud compute instances start ${INSTANCE} --zone=${ZONE} --project=${PROJECT_ID}"
  exit 0
fi

# Durable storage for checkpoints — matters most on spot VMs, which can be
# reclaimed mid-run.
if ! gcloud_ storage buckets describe "${BUCKET}" >/dev/null 2>&1; then
  echo "Creating bucket ${BUCKET} ..."
  gcloud_ storage buckets create "${BUCKET}" --location="${REGION}"
fi

SPOT_FLAGS=()
if [ "${SPOT}" = "1" ]; then
  # STOP (not DELETE) on reclaim, so the boot disk and any local state survive
  # and the VM can simply be restarted.
  SPOT_FLAGS=(--provisioning-model=SPOT --instance-termination-action=STOP)
fi

gcloud_ compute instances create "${INSTANCE}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --image-family="${IMAGE_FAMILY}" \
  --image-project="${IMAGE_PROJECT}" \
  --boot-disk-size="${BOOT_DISK_SIZE}" \
  --boot-disk-type="${BOOT_DISK_TYPE}" \
  --maintenance-policy=TERMINATE \
  --scopes=cloud-platform \
  --metadata="install-nvidia-driver=True" \
  "${SPOT_FLAGS[@]}"

echo
echo "Waiting for SSH to come up (the driver install on first boot takes a few minutes) ..."
for i in $(seq 1 40); do
  if gcloud_ compute ssh "${INSTANCE}" --zone="${ZONE}" --command="true" >/dev/null 2>&1; then
    echo "SSH is up."
    break
  fi
  sleep 15
done

echo
echo "Verifying GPUs ..."
gcloud_ compute ssh "${INSTANCE}" --zone="${ZONE}" \
  --command="nvidia-smi --query-gpu=index,name,memory.total --format=csv || echo 'driver still installing — retry in a minute'"

echo
echo "Next: ./gcp/sync_and_train.sh"
