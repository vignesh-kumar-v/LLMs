#!/usr/bin/env bash
# Report the GPU quotas that gate this project, plus the status of any quota
# increase requests you have filed.
#
# Two Compute Engine quotas must BOTH be raised:
#   GPUS_ALL_REGIONS  (global)      -- defaults to 0; nothing launches without it
#   NVIDIA_L4_GPUS    (per region)  -- defaults to 1 in every region
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "Project: ${PROJECT_ID}   Region: ${REGION}   Need: ${NUM_GPUS} GPU(s)"
echo

echo "── Effective quota ──────────────────────────────────────────────"
# The check below exits non-zero when quota is short; that is a reported
# result, not a script failure, so errexit is lifted around it.
set +e
python3 - "$PROJECT_ID" "$REGION" "$NUM_GPUS" <<'PY'
import json, subprocess, sys

project, region, need = sys.argv[1], sys.argv[2], int(sys.argv[3])

def run(cmd):
    return json.loads(subprocess.check_output(cmd, text=True))

glob = run(["gcloud", "compute", "project-info", "describe",
            f"--project={project}", "--format=json"])
reg = run(["gcloud", "compute", "regions", "describe", region,
           f"--project={project}", "--format=json"])

checks = [
    ("GPUS_ALL_REGIONS", glob.get("quotas", []), "global", True),
    ("NVIDIA_L4_GPUS", reg.get("quotas", []), region, True),
    ("PREEMPTIBLE_NVIDIA_L4_GPUS", reg.get("quotas", []), region, False),
]

blocking = False
for metric, quotas, scope, required in checks:
    entry = next((q for q in quotas if q["metric"] == metric), None)
    if entry is None:
        print(f"  {metric:30s} {scope:12s} not reported")
        continue
    limit, usage = int(entry["limit"]), int(entry["usage"])
    ok = limit >= need
    tag = "OK" if ok else ("BLOCKING" if required else "spot only")
    if required and not ok:
        blocking = True
    print(f"  {metric:30s} {scope:12s} limit={limit:<4d} usage={usage:<4d} {tag}")

print()
sys.exit(1 if blocking else 0)
PY
QUOTA_OK=$?
set -e

echo "── Quota increase requests ──────────────────────────────────────"
gcloud quotas preferences list --project="${PROJECT_ID}" --format=json 2>/dev/null \
  | python3 -c "
import json, sys
try:
    prefs = json.load(sys.stdin)
except Exception:
    prefs = []
if not prefs:
    print('  (none filed)')
for p in prefs:
    qc = p.get('quotaConfig', {})
    state = qc.get('stateDetail') or ('IN PROGRESS' if p.get('reconciling') else 'unknown')
    dims = (p.get('dimensions') or {}).get('region', 'global')
    print(f\"  {p.get('quotaId','?'):52s} {dims:12s} \"
          f\"req={qc.get('preferredValue')} granted={qc.get('grantedValue')}  {state}\")
" || echo "  (could not read quota preferences)"

echo
if [ "${QUOTA_OK}" -eq 0 ]; then
  echo "Quota is sufficient. Run ./gcp/create_vm.sh"
else
  cat <<EOF
Quota is NOT sufficient. Request increases here:
  https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}

Request BOTH of these (a denial on either one blocks everything):
  1. Service 'Compute Engine API', metric 'GPUS_ALL_REGIONS'
     scope: global          -> ${NUM_GPUS}
  2. Service 'Compute Engine API', metric 'NVIDIA_L4_GPUS'
     scope: ${REGION}  -> ${NUM_GPUS}

Justification text is in gcp/QUOTA_REQUEST.md — a specific, technical
justification materially improves the approval odds over a blank one.
EOF
fi
