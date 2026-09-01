#!/usr/bin/env bash
# Tail the training log on the VM.
set -euo pipefail
source "$(dirname "$0")/env.sh"
gcloud_ compute ssh "${INSTANCE}" --zone="${ZONE}" \
  --command="tail -n ${LINES:-80} -f ${REMOTE_DIR}/train.log"
