#!/usr/bin/env bash
#
# run_nccl_baseline.sh — Phase 1 NCCL baseline runner.
#
# Runs AllReduce, AllGather and ReduceScatter through nccl-tests on a single
# node, captures verbatim output plus full environment metadata, and writes a
# manifest describing exactly what was executed.
#
# It measures and collects only. Parsing, analysis and plotting happen locally
# after the GPU node is terminated, so we never pay for GPU time while
# thinking about the numbers.
#
# Usage:
#   scripts/run_nccl_baseline.sh [-g NGPUS] [-t smoke|full|both] [-i EXPERIMENT_ID]
#
# Env:
#   NCCL_TESTS_DIR  nccl-tests checkout (default: $HOME/nccl-tests)
#   RESULTS_ROOT    results directory   (default: <repo>/results)
#   FULL_REPEATS    whole-sweep repeats for the full tier (default: 3)
#
# Design notes:
#   * Single-process multi-GPU mode (-g N). No MPI in Phase 1.
#   * The smoke tier gates the full tier: if the 3-point sweep fails its
#     correctness check, the expensive sweep is not attempted.
#   * One NCCL_DEBUG=INFO run is captured for algorithm/topology diagnosis;
#     the measured runs use NCCL_DEBUG=VERSION so the version banner is
#     recorded without INFO's logging overhead perturbing timings.

set -uo pipefail

NGPUS=2
TIER="both"
EXPERIMENT_ID=""

while getopts ":g:t:i:h" opt; do
  case "$opt" in
    g) NGPUS="$OPTARG" ;;
    t) TIER="$OPTARG" ;;
    i) EXPERIMENT_ID="$OPTARG" ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done

case "$TIER" in
  smoke|full|both) ;;
  *) echo "-t must be smoke, full or both" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-$HOME/nccl-tests}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
FULL_REPEATS="${FULL_REPEATS:-3}"

# --- sweep parameters (see docs/experiments/phase1_nccl_baseline.md) --------
SMOKE_RANGE=(-b 8 -e 128M -f 4096)   # 8 B, 32 KiB, 128 MiB
SMOKE_WARMUP=5
SMOKE_ITERS=20

FULL_RANGE=(-b 8 -e 128M -f 2)       # 25 points, powers of two
FULL_WARMUP=20
FULL_ITERS=50

# --- preflight --------------------------------------------------------------
BINARIES=(all_reduce_perf all_gather_perf reduce_scatter_perf)
missing=0
for b in "${BINARIES[@]}"; do
  [ -x "$NCCL_TESTS_DIR/build/$b" ] || { echo "missing binary: $NCCL_TESTS_DIR/build/$b" >&2; missing=1; }
done
if [ "$missing" -ne 0 ]; then
  echo "run scripts/setup_nccl_tests.sh first (or set NCCL_TESTS_DIR)" >&2
  exit 1
fi

VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c .)"
if [ -z "$VISIBLE_GPUS" ] || [ "$VISIBLE_GPUS" -lt "$NGPUS" ]; then
  echo "requested -g $NGPUS but only ${VISIBLE_GPUS:-0} GPU(s) visible" >&2
  exit 1
fi

GIT_SHORT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
GIT_FULL="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo '')"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
[ -n "$EXPERIMENT_ID" ] || EXPERIMENT_ID="p1-nccl-baseline-${STAMP}-${GIT_SHORT}"

RAW_DIR="$RESULTS_ROOT/raw/$EXPERIMENT_ID"
if [ -e "$RAW_DIR" ]; then
  echo "refusing to overwrite existing experiment: $RAW_DIR" >&2
  exit 1
fi
mkdir -p "$RAW_DIR"

echo "experiment id : $EXPERIMENT_ID"
echo "raw dir       : $RAW_DIR"
echo "gpus          : $NGPUS"
echo "tier          : $TIER"
echo

# --- environment metadata ---------------------------------------------------
"$SCRIPT_DIR/collect_env.sh" -o "$RAW_DIR"

NCCL_TESTS_COMMIT="$(git -C "$NCCL_TESTS_DIR" rev-parse HEAD 2>/dev/null || echo '')"

# --- manifest helpers -------------------------------------------------------
jstr() {
  local v="${1:-}"
  if [ -z "$v" ]; then printf 'null'; return; fi
  printf '%s' "$v" \
    | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' \
    | awk 'BEGIN{printf "\""} {if(NR>1) printf "\\n"; printf "%s", $0} END{printf "\""}'
}

MANIFEST="$RAW_DIR/run_manifest.json"
RUN_ENTRIES=""
RUN_COUNT=0
FAILED_RUNS=0

collective_name() {
  case "$1" in
    all_reduce_perf) echo all_reduce ;;
    all_gather_perf) echo all_gather ;;
    reduce_scatter_perf) echo reduce_scatter ;;
    *) echo "${1%_perf}" ;;
  esac
}

# run_one <binary> <tier> <repeat_index> <warmup> <iters> <range...>
run_one() {
  local bin="$1" tier="$2" rep="$3" warmup="$4" iters="$5"
  shift 5
  local range=("$@")

  local coll; coll="$(collective_name "$bin")"
  local base="${coll}.${tier}.r${rep}"
  local out="$RAW_DIR/${base}.stdout.txt"
  local err="$RAW_DIR/${base}.stderr.txt"

  local cmd="$NCCL_TESTS_DIR/build/$bin ${range[*]} -g $NGPUS -w $warmup -n $iters -c 1 -d float"
  local started; started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local t0; t0="$(date +%s)"

  echo "[$tier r$rep] $coll"
  NCCL_DEBUG=VERSION \
    "$NCCL_TESTS_DIR/build/$bin" "${range[@]}" -g "$NGPUS" -w "$warmup" -n "$iters" -c 1 -d float \
    >"$out" 2>"$err"
  local rc=$?
  local t1; t1="$(date +%s)"

  # Gate: nccl-tests prints this trailer only when every validation passed.
  local ok=false
  if [ "$rc" -eq 0 ] && grep -q 'Out of bounds values *: *0 *OK' "$out"; then
    ok=true
  else
    FAILED_RUNS=$((FAILED_RUNS + 1))
    echo "  !! FAILED (exit=$rc) — see $err" >&2
  fi
  echo "  exit=$rc correctness_gate=$ok duration=$((t1 - t0))s"

  [ "$RUN_COUNT" -gt 0 ] && RUN_ENTRIES="$RUN_ENTRIES,"
  RUN_COUNT=$((RUN_COUNT + 1))
  RUN_ENTRIES="$RUN_ENTRIES
    {
      \"collective\": $(jstr "$coll"),
      \"binary\": $(jstr "$bin"),
      \"tier\": $(jstr "$tier"),
      \"repeat_index\": $rep,
      \"warmup_iterations\": $warmup,
      \"measured_iterations\": $iters,
      \"datatype\": \"float\",
      \"gpu_count\": $NGPUS,
      \"command\": $(jstr "$cmd"),
      \"exit_code\": $rc,
      \"correctness_gate_passed\": $ok,
      \"started_at_utc\": $(jstr "$started"),
      \"duration_s\": $((t1 - t0)),
      \"stdout_file\": $(jstr "${base}.stdout.txt"),
      \"stderr_file\": $(jstr "${base}.stderr.txt")
    }"

  return 0
}

# --- NCCL_DEBUG=INFO diagnostic (one small run) -----------------------------
# Reveals the ring/tree topology, channel count, algorithm and protocol NCCL
# selected. Logging overhead makes it unsuitable for timing, so it is kept
# out of the measured set and stored separately.
echo "[diag] NCCL_DEBUG=INFO probe"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV \
  "$NCCL_TESTS_DIR/build/all_reduce_perf" -b 8 -e 8 -f 2 -g "$NGPUS" -w 1 -n 1 -c 1 -d float \
  >"$RAW_DIR/nccl_debug_info.txt" 2>&1
echo "  -> nccl_debug_info.txt"
echo

# --- smoke tier -------------------------------------------------------------
SMOKE_OK=1
if [ "$TIER" = "smoke" ] || [ "$TIER" = "both" ]; then
  echo "=== tier: smoke (${SMOKE_RANGE[*]}, -w $SMOKE_WARMUP -n $SMOKE_ITERS) ==="
  before_failures=$FAILED_RUNS
  for b in "${BINARIES[@]}"; do
    run_one "$b" smoke 0 "$SMOKE_WARMUP" "$SMOKE_ITERS" "${SMOKE_RANGE[@]}"
  done
  [ "$FAILED_RUNS" -gt "$before_failures" ] && SMOKE_OK=0
  echo
fi

# --- full tier (gated on smoke) ---------------------------------------------
if [ "$TIER" = "full" ] || [ "$TIER" = "both" ]; then
  if [ "$TIER" = "both" ] && [ "$SMOKE_OK" -ne 1 ]; then
    echo "smoke tier failed its correctness gate — skipping the full sweep." >&2
    echo "Diagnose before spending more GPU time." >&2
  else
    echo "=== tier: full (${FULL_RANGE[*]}, -w $FULL_WARMUP -n $FULL_ITERS, ${FULL_REPEATS} repeats) ==="
    rep=0
    while [ "$rep" -lt "$FULL_REPEATS" ]; do
      for b in "${BINARIES[@]}"; do
        run_one "$b" full "$rep" "$FULL_WARMUP" "$FULL_ITERS" "${FULL_RANGE[@]}"
      done
      rep=$((rep + 1))
    done
    echo
  fi
fi

# --- manifest ---------------------------------------------------------------
{
  printf '{\n'
  printf '  "schema_version": 1,\n'
  printf '  "experiment_id": %s,\n' "$(jstr "$EXPERIMENT_ID")"
  printf '  "phase": "phase1",\n'
  printf '  "created_at_utc": %s,\n' "$(jstr "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
  printf '  "value_kind": "measured",\n'
  printf '  "benchmark_tool": "nccl-tests",\n'
  printf '  "nccl_tests_dir": %s,\n' "$(jstr "$NCCL_TESTS_DIR")"
  printf '  "nccl_tests_commit": %s,\n' "$(jstr "$NCCL_TESTS_COMMIT")"
  printf '  "git_commit": %s,\n' "$(jstr "$GIT_FULL")"
  printf '  "node_count": 1,\n'
  printf '  "gpu_count": %s,\n' "$NGPUS"
  printf '  "network": "none-single-node",\n'
  printf '  "runner": "scripts/run_nccl_baseline.sh",\n'
  printf '  "failed_runs": %s,\n' "$FAILED_RUNS"
  printf '  "runs": [%s\n  ]\n' "$RUN_ENTRIES"
  printf '}\n'
} > "$MANIFEST"

echo "manifest -> $MANIFEST"
echo
echo "runs: $RUN_COUNT   failed: $FAILED_RUNS"
echo
echo "Next: copy $RAW_DIR back, TERMINATE the pod, then parse locally:"
echo "  python3 scripts/parse_nccl_output.py --raw-dir results/raw/$EXPERIMENT_ID"

[ "$FAILED_RUNS" -eq 0 ] || exit 1
