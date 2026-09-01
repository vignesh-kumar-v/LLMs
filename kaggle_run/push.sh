#!/usr/bin/env bash
# Package the repo as a private Kaggle Dataset, then push + run the training
# kernel on 2x T4.
#
#   ./kaggle_run/push.sh              # upload source, push kernel, start run
#   ./kaggle_run/push.sh --status     # check the current run
#   ./kaggle_run/push.sh --logs       # print kernel logs
#   ./kaggle_run/push.sh --fetch      # download output into ./artifacts
#
# Auth first (one time):
#   kaggle auth login          # OAuth, or
#   export KAGGLE_API_TOKEN=...  # from https://www.kaggle.com/settings/api
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${TMPDIR:-/tmp}/nanollm-kaggle-src"
KAGGLE="${KAGGLE:-kaggle}"

MODE="${MODE:-train}"
DATASET_SLUG="${DATASET_SLUG:-nanollm-src}"
# Kaggle's machine-shape enum is not published in the SDK. "NvidiaTeslaT4" is
# the value that yields 2x T4 (Kaggle allocates T4s in pairs) — discovered by
# pulling metadata from public T4x2 kernels. Anything unrecognised is silently
# normalised by the server to "Gpu", which is a single P100 — and Kaggle's
# PyTorch build does not even support the P100's sm_60. So this matters.
ACCELERATOR="${ACCELERATOR:-NvidiaTeslaT4}"
require_auth() {
  if ! ${KAGGLE} config view >/dev/null 2>&1; then
    cat <<'EOF'
Kaggle CLI is not authenticated.

  Run ONE of these, then re-run this script:
    kaggle auth login
    export KAGGLE_API_TOKEN=<token from https://www.kaggle.com/settings/api>

EOF
    exit 1
  fi
}

kaggle_username() {
  ${KAGGLE} config view 2>/dev/null | sed -n 's/^- username: //p' | head -1
}

case "${1:-}" in
  --validate) MODE=validate ;;
  --train)    MODE=train ;;
esac

# Validation and training are separate kernels so a long training run is never
# clobbered by a quick validation push, and their logs stay distinct.
if [ "${MODE}" = "validate" ]; then
  KERNEL_SLUG="${KERNEL_SLUG:-nanollm-validate-t4x2}"
  KERNEL_SCRIPT="validate_kaggle.py"
  KERNEL_TITLE="NanoLLM validate t4x2"
  TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
else
  KERNEL_SLUG="${KERNEL_SLUG:-nanollm-ddp-t4x2}"
  KERNEL_SCRIPT="run_kaggle.py"
  KERNEL_TITLE="NanoLLM ddp t4x2"
  TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-32400}"
fi

case "${1:-}" in
  --status)
    require_auth
    ${KAGGLE} kernels status "$(kaggle_username)/${KERNEL_SLUG}"
    exit 0 ;;
  --logs)
    require_auth
    ${KAGGLE} kernels output "$(kaggle_username)/${KERNEL_SLUG}" -p /tmp/nanollm-logs
    cat /tmp/nanollm-logs/*.log 2>/dev/null || echo "(no log file yet)"
    exit 0 ;;
  --fetch)
    require_auth
    mkdir -p "${REPO_ROOT}/artifacts"
    ${KAGGLE} kernels output "$(kaggle_username)/${KERNEL_SLUG}" -p "${REPO_ROOT}/artifacts"
    echo "-> ${REPO_ROOT}/artifacts"
    exit 0 ;;
esac

require_auth
USER_NAME="$(kaggle_username)"
[ -n "${USER_NAME}" ] || { echo "Could not determine Kaggle username"; exit 1; }
echo "Kaggle user: ${USER_NAME}   mode: ${MODE}"

echo
echo "── Accelerator quota ────────────────────────────────────────────"
${KAGGLE} quota 2>/dev/null || echo "(quota unavailable)"

# ── 1. Stage source ───────────────────────────────────────────────────────
echo
echo "── Packaging source ─────────────────────────────────────────────"
rm -rf "${STAGE}"; mkdir -p "${STAGE}"
# Only what the trainer needs. No data, checkpoints, git history or venv.
for f in main.py config.py data.py NanoLLM.py fused_ln.py CrossEntropyLoss.py \
         TinyStories.py test_layernorm.py profile_run.py requirements.txt \
         fused_layernorm_train.cu fused_layernorm.cu fused_layernorm_v3.cu; do
  cp "${REPO_ROOT}/${f}" "${STAGE}/"
done

cat > "${STAGE}/dataset-metadata.json" <<EOF
{
  "title": "NanoLLM source",
  "id": "${USER_NAME}/${DATASET_SLUG}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo "Files: $(ls "${STAGE}" | tr '\n' ' ')"

# ── 2. Create or version the dataset ──────────────────────────────────────
echo
echo "── Uploading source dataset ─────────────────────────────────────"
if ${KAGGLE} datasets status "${USER_NAME}/${DATASET_SLUG}" >/dev/null 2>&1; then
  ${KAGGLE} datasets version -p "${STAGE}" -m "update $(date -u +%Y-%m-%dT%H:%M:%SZ)" --dir-mode zip
else
  ${KAGGLE} datasets create -p "${STAGE}" --dir-mode zip
fi

# Kaggle needs a moment before a fresh dataset version is attachable.
echo "Waiting 20s for the dataset version to become available ..."
sleep 20

# ── 3. Push the kernel ────────────────────────────────────────────────────
echo
echo "── Pushing kernel ───────────────────────────────────────────────"
KSTAGE="${TMPDIR:-/tmp}/nanollm-kaggle-kernel"
rm -rf "${KSTAGE}"; mkdir -p "${KSTAGE}"
cp "${REPO_ROOT}/kaggle_run/${KERNEL_SCRIPT}" "${KSTAGE}/"

# Kaggle derives a NEW kernel's slug from the TITLE, not from "id" — so the
# title must slugify to exactly KERNEL_SLUG or the kernel lands at a different
# URL and every later status/output call misses it.
cat > "${KSTAGE}/kernel-metadata.json" <<EOF
{
  "id": "${USER_NAME}/${KERNEL_SLUG}",
  "title": "${KERNEL_TITLE}",
  "code_file": "${KERNEL_SCRIPT}",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["${USER_NAME}/${DATASET_SLUG}"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
EOF

PUSH_ARGS=(-p "${KSTAGE}" -t "${TIMEOUT_SECONDS}")
[ -n "${ACCELERATOR}" ] && PUSH_ARGS+=(--accelerator "${ACCELERATOR}")

${KAGGLE} kernels push "${PUSH_ARGS[@]}"

cat <<EOF

── Next ────────────────────────────────────────────────────────────
Kernel: https://www.kaggle.com/code/${USER_NAME}/${KERNEL_SLUG}
Accelerator requested: ${ACCELERATOR}

  ./kaggle_run/push.sh --status     check run state
  ./kaggle_run/push.sh --logs       print logs
  ./kaggle_run/push.sh --fetch      download checkpoints into ./artifacts
EOF
