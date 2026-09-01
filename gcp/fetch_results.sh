#!/usr/bin/env bash
# Copy checkpoints, plots and logs back from the VM into ./artifacts.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p artifacts
for f in checkpoints/best_model.pt checkpoints/last.pt training_stats.png train.log; do
  echo "fetching ${f} ..."
  gcloud_ compute scp --zone="${ZONE}" \
    "${INSTANCE}:${REMOTE_DIR}/${f}" "artifacts/$(basename "${f}")" 2>/dev/null \
    || echo "  (not present yet)"
done
echo "Done -> ./artifacts"
