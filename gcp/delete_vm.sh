#!/usr/bin/env bash
# Delete the VM. GPUs bill by the second while the instance is RUNNING, so
# stop or delete it as soon as a run finishes.
set -euo pipefail
source "$(dirname "$0")/env.sh"
read -r -p "Delete instance ${INSTANCE} in ${ZONE}? [y/N] " ans
[[ "${ans}" == "y" || "${ans}" == "Y" ]] || { echo "aborted"; exit 0; }
gcloud_ compute instances delete "${INSTANCE}" --zone="${ZONE}" --quiet
echo "Deleted. Bucket ${BUCKET} was left in place."
