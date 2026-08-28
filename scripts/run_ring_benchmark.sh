#!/usr/bin/env bash
#
# run_ring_benchmark.sh — Phase 6 custom Ring AllReduce runner.
#
# Builds the custom implementation, captures environment and topology, runs the
# benchmark (which verifies correctness before timing every configuration), and
# writes a manifest — reusing the Phase 1-5 experiment layout.
#
# Usage:
#   scripts/run_ring_benchmark.sh [-g NGPUS] [-w WARMUP] [-n ITERS] [-i EXPERIMENT_ID]
#                                 [--allow-host-staged]
#
# Env:
#   CUDA_HOME, NCCL_HOME, RESULTS_ROOT
#
# The benchmark refuses to run unless every ring edge supports direct GPU peer
# access, unless --allow-host-staged is passed. A host-staged run is recorded as
# a distinct transport, never as P2P.

set -uo pipefail

NGPUS=4; WARMUP=5; ITERS=20; EXPERIMENT_ID=""; EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    -g) NGPUS="$2"; shift 2 ;;
    -w) WARMUP="$2"; shift 2 ;;
    -n) ITERS="$2"; shift 2 ;;
    -i) EXPERIMENT_ID="$2"; shift 2 ;;
    --allow-host-staged) EXTRA="--allow-host-staged"; shift ;;
    -h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/src/ring_allreduce"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

GIT_SHORT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
GIT_FULL="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo '')"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
[ -n "$EXPERIMENT_ID" ] || EXPERIMENT_ID="p6-ring-allreduce-${STAMP}-${GIT_SHORT}"

RAW_DIR="$RESULTS_ROOT/raw/$EXPERIMENT_ID"
[ -e "$RAW_DIR" ] && { echo "refusing to overwrite $RAW_DIR" >&2; exit 1; }
mkdir -p "$RAW_DIR"

echo "experiment id : $EXPERIMENT_ID"
echo "gpus          : $NGPUS"
echo

"$SCRIPT_DIR/collect_env.sh" -o "$RAW_DIR"

# Target the architecture actually present, rather than relying on defaults.
SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
echo "building custom ring (sm_${SM:-default})"
make -C "$SRC_DIR" clean >/dev/null 2>&1
if ! make -C "$SRC_DIR" CUDA_HOME="$CUDA_HOME" ${NCCL_HOME:+NCCL_HOME="$NCCL_HOME"} \
        ${SM:+SM="$SM"} > "$RAW_DIR/build.log" 2>&1; then
  echo "BUILD FAILED — see $RAW_DIR/build.log" >&2
  tail -20 "$RAW_DIR/build.log" >&2
  exit 1
fi
echo "  ok -> $SRC_DIR/build/ring_allreduce"

CMD="$SRC_DIR/build/ring_allreduce -g $NGPUS -w $WARMUP -n $ITERS $EXTRA"
echo "running: $CMD"
# shellcheck disable=SC2086
NCCL_DEBUG=VERSION $SRC_DIR/build/ring_allreduce -g "$NGPUS" -w "$WARMUP" -n "$ITERS" $EXTRA \
  > "$RAW_DIR/ring_allreduce.stdout.txt" 2> "$RAW_DIR/ring_allreduce.stderr.txt"
RC=$?
echo "exit=$RC"

# Correctness gate: any mismatch, or a refusal to run without P2P, stops here.
BAD=$(awk -F, 'NR>1 && $1!~/^#/ && $15!="" && $15+0!=0' "$RAW_DIR/ring_allreduce.stdout.txt" 2>/dev/null | wc -l | tr -d ' ')
ROWS=$(grep -vc '^#' "$RAW_DIR/ring_allreduce.stdout.txt" 2>/dev/null || echo 0)
echo "data rows=$ROWS  rows with mismatches=$BAD"

jstr() {
  local v="${1:-}"; [ -z "$v" ] && { printf 'null'; return; }
  printf '%s' "$v" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' \
    | awk 'BEGIN{printf "\""} {if(NR>1) printf "\\n"; printf "%s", $0} END{printf "\""}'
}
TRANSPORT="$(grep -o 'transport=[a-z-]*' "$RAW_DIR/ring_allreduce.stdout.txt" | head -1 | cut -d= -f2)"
RINGORDER="$(grep '^# ring order:' "$RAW_DIR/ring_allreduce.stdout.txt" | head -1)"

cat > "$RAW_DIR/run_manifest.json" <<EOF
{
  "schema_version": 1,
  "experiment_id": $(jstr "$EXPERIMENT_ID"),
  "phase": "phase6",
  "created_at_utc": $(jstr "$(date -u +%Y-%m-%dT%H:%M:%SZ)"),
  "value_kind": "measured",
  "benchmark_tool": "custom-ring-allreduce",
  "git_commit": $(jstr "$GIT_FULL"),
  "node_count": 1,
  "gpu_count": $NGPUS,
  "network": "none-single-node",
  "transport": $(jstr "$TRANSPORT"),
  "ring_order": $(jstr "$RINGORDER"),
  "warmup_iterations": $WARMUP,
  "measured_iterations": $ITERS,
  "command": $(jstr "$CMD"),
  "exit_code": $RC,
  "rows_with_mismatches": ${BAD:-0},
  "runner": "scripts/run_ring_benchmark.sh"
}
EOF
echo "manifest -> $RAW_DIR/run_manifest.json"
echo
echo "Next: copy $RAW_DIR back, TERMINATE the pod, then parse locally:"
echo "  python3 scripts/parse_ring_output.py --raw-dir results/raw/$EXPERIMENT_ID"

[ "$RC" -eq 0 ] && [ "${BAD:-0}" -eq 0 ] || exit 1
