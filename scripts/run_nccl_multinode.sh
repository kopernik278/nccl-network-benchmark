#!/usr/bin/env bash
#
# run_nccl_multinode.sh — Phase 3 multi-node NCCL Socket/TCP baseline runner.
#
# Extends the single-node Phase 1/2 workflow across hosts using nccl-tests'
# normal MPI multi-process mode. No custom distributed runtime.
#
# Usage:
#   scripts/run_nccl_multinode.sh -H host1,host2 -I <iface> \
#       [-p RANKS_PER_NODE] [-t smoke|full|both] [-c COLLECTIVES] [-i EXPERIMENT_ID]
#
#   -H  comma-separated hostnames/IPs, at least 2 (required)
#   -I  cluster-internal network interface for NCCL (required — never guessed)
#   -p  ranks (GPUs) per node, default 1
#   -t  tier, default both
#   -c  comma-separated collectives, default all_reduce
#   -i  explicit experiment id
#
# Env:
#   NCCL_TESTS_DIR  nccl-tests checkout built with MPI=1 (default: $HOME/nccl-tests)
#   RESULTS_ROOT    results directory (default: <repo>/results)
#   FULL_REPEATS    whole-sweep repeats for the full tier (default: 3)
#
# TRANSPORT CONTRACT
#
# Phase 3 measures NCCL over TCP sockets so it can later be compared against
# RDMA. That comparison is meaningless unless the socket run really used
# sockets, so this script:
#
#   1. sets NCCL_IB_DISABLE=1 and an EXPLICIT NCCL_SOCKET_IFNAME;
#   2. runs a NCCL_DEBUG=INFO diagnostic;
#   3. PROVES from that output that the collective used NET/Socket, on the
#      requested interface, across the expected number of hosts;
#   4. refuses to run any timed measurement if that proof fails.
#
# It never falls back to another interface or transport. A wrong-fabric result
# is worse than no result.
#
# The INFO diagnostic is kept separate from the timed runs, which use
# NCCL_DEBUG=VERSION, so debug logging cannot perturb the measurements.

set -uo pipefail

HOSTS=""; IFACE=""; RANKS_PER_NODE=1; TIER="both"; COLLECTIVES="all_reduce"; EXPERIMENT_ID=""

while getopts ":H:I:p:t:c:i:h" opt; do
  case "$opt" in
    H) HOSTS="$OPTARG" ;;
    I) IFACE="$OPTARG" ;;
    p) RANKS_PER_NODE="$OPTARG" ;;
    t) TIER="$OPTARG" ;;
    c) COLLECTIVES="$OPTARG" ;;
    i) EXPERIMENT_ID="$OPTARG" ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

# --- mandatory arguments: refuse to guess ----------------------------------
[ -n "$HOSTS" ]  || die "-H is required (comma-separated hosts, at least 2)"
[ -n "$IFACE" ]  || die "-I is required. Run scripts/collect_net_env.sh on the
       nodes first and choose the cluster-internal interface from its output.
       This script will not guess an interface and will not fall back."
case "$TIER" in smoke|full|both) ;; *) die "-t must be smoke, full or both" ;; esac

IFS=',' read -r -a HOST_ARR <<< "$HOSTS"
NNODES=${#HOST_ARR[@]}
[ "$NNODES" -ge 2 ] || die "-H needs at least 2 hosts for a multi-node run (got $NNODES)"
TOTAL_RANKS=$(( NNODES * RANKS_PER_NODE ))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-$HOME/nccl-tests}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
FULL_REPEATS="${FULL_REPEATS:-3}"

# --- sweep parameters: identical to Phase 1/2 so results stay comparable ----
SMOKE_RANGE=(-b 8 -e 128M -f 4096)   # 8 B, 32 KiB, 128 MiB
SMOKE_WARMUP=5;  SMOKE_ITERS=20
FULL_RANGE=(-b 8 -e 128M -f 2)       # 25 points
FULL_WARMUP=20;  FULL_ITERS=50

# --- preflight --------------------------------------------------------------
command -v mpirun >/dev/null 2>&1 || die "mpirun not found; Phase 3 needs an MPI launcher"

MPI_RAW="$(mpirun --version 2>&1)"
if printf '%s' "$MPI_RAW" | grep -qi 'open mpi'; then MPI_IMPL="OpenMPI"
elif printf '%s' "$MPI_RAW" | grep -qi 'hydra\|mpich'; then MPI_IMPL="MPICH"
elif printf '%s' "$MPI_RAW" | grep -qi 'mvapich'; then MPI_IMPL="MVAPICH"
elif printf '%s' "$MPI_RAW" | grep -qi 'intel'; then MPI_IMPL="IntelMPI"
else MPI_IMPL="unknown"; fi
MPI_VERSION="$(printf '%s\n' "$MPI_RAW" | grep -iE 'open mpi|mvapich|intel\(r\) mpi' | head -1)"
[ -z "$MPI_VERSION" ] && MPI_VERSION="$(printf '%s\n' "$MPI_RAW" | sed -n 's/^[[:space:]]*Version:[[:space:]]*\(.*\)$/MPICH \1/p' | head -1)"
[ -z "$MPI_VERSION" ] && MPI_VERSION="$(printf '%s\n' "$MPI_RAW" | head -1)"
MPI_VERSION="$(printf '%s' "$MPI_VERSION" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

for b in all_reduce_perf all_gather_perf reduce_scatter_perf; do
  [ -x "$NCCL_TESTS_DIR/build/$b" ] || die "missing $NCCL_TESTS_DIR/build/$b — run scripts/setup_nccl_tests.sh -m"
done
# A non-MPI binary under mpirun silently becomes N independent 1-rank jobs,
# each "succeeding". That would look like a multi-node run and be nonsense.
if command -v ldd >/dev/null 2>&1; then
  ldd "$NCCL_TESTS_DIR/build/all_reduce_perf" 2>/dev/null | grep -qi libmpi \
    || die "nccl-tests was not built with MPI=1 (binary does not link libmpi).
       Rebuild with: scripts/setup_nccl_tests.sh -m"
fi

GIT_SHORT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
GIT_FULL="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo '')"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
[ -n "$EXPERIMENT_ID" ] || EXPERIMENT_ID="p3-nccl-socket-${NNODES}n${TOTAL_RANKS}r-${STAMP}-${GIT_SHORT}"

RAW_DIR="$RESULTS_ROOT/raw/$EXPERIMENT_ID"
[ -e "$RAW_DIR" ] && die "refusing to overwrite existing experiment: $RAW_DIR"
mkdir -p "$RAW_DIR" || die "cannot create $RAW_DIR"

# --- hostfile (format differs by MPI implementation) ------------------------
HOSTFILE="$RAW_DIR/hostfile"
: > "$HOSTFILE"
for h in "${HOST_ARR[@]}"; do
  case "$MPI_IMPL" in
    OpenMPI) echo "$h slots=$RANKS_PER_NODE" >> "$HOSTFILE" ;;
    *)       echo "$h:$RANKS_PER_NODE"       >> "$HOSTFILE" ;;
  esac
done

# --- environment propagation (flag differs by MPI implementation) ----------
# Every rank must receive the transport configuration; a rank that misses
# NCCL_IB_DISABLE could select a different transport than its peers.
NCCL_ENV_KV=(
  "NCCL_IB_DISABLE=1"
  "NCCL_SOCKET_IFNAME=$IFACE"
)
build_env_args() {   # $1 = extra "K=V" entries, space separated
  ENV_ARGS=()
  local kv
  for kv in "${NCCL_ENV_KV[@]}" "$@"; do
    case "$MPI_IMPL" in
      OpenMPI) ENV_ARGS+=(-x "$kv") ;;
      *)       ENV_ARGS+=(-genv "${kv%%=*}" "${kv#*=}") ;;
    esac
  done
  # Pass through the library path so a locally-installed NCCL is found on every node.
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    case "$MPI_IMPL" in
      OpenMPI) ENV_ARGS+=(-x "LD_LIBRARY_PATH") ;;
      *)       ENV_ARGS+=(-genv LD_LIBRARY_PATH "$LD_LIBRARY_PATH") ;;
    esac
  fi
}

echo "experiment id  : $EXPERIMENT_ID"
echo "hosts          : ${HOST_ARR[*]}  ($NNODES nodes)"
echo "ranks          : $TOTAL_RANKS ($RANKS_PER_NODE per node)"
echo "interface      : $IFACE  (explicitly selected)"
echo "mpi            : $MPI_IMPL / $MPI_VERSION"
echo "transport      : NCCL_IB_DISABLE=1, NET/Socket expected"
echo "raw dir        : $RAW_DIR"
echo

# --- per-node environment capture -------------------------------------------
# One process per node so each host reports its own view.
echo "[env] capturing per-node environment"
build_env_args
mpirun --hostfile "$HOSTFILE" -np "$NNODES" \
  $( [ "$MPI_IMPL" = "OpenMPI" ] && echo "--map-by ppr:1:node" ) \
  "${ENV_ARGS[@]}" \
  bash -c "H=\$(hostname); mkdir -p '$RAW_DIR'/nodes/\$H; \
           '$SCRIPT_DIR/collect_env.sh'     -o '$RAW_DIR'/nodes/\$H >/dev/null 2>&1; \
           '$SCRIPT_DIR/collect_net_env.sh' -o '$RAW_DIR'/nodes/\$H >/dev/null 2>&1" \
  > "$RAW_DIR/env_capture.log" 2>&1
echo "  -> $RAW_DIR/nodes/<hostname>/{env,netenv}.{json,txt}"
# Keep a top-level env.json so the existing parser finds one where it expects.
_first_node="$(ls "$RAW_DIR/nodes" 2>/dev/null | head -1)"
if [ -n "$_first_node" ] && [ -f "$RAW_DIR/nodes/$_first_node/env.json" ]; then
  cp "$RAW_DIR/nodes/$_first_node/env.json" "$RAW_DIR/env.json"
  cp "$RAW_DIR/nodes/$_first_node/env.txt"  "$RAW_DIR/env.txt" 2>/dev/null || true
fi
echo

# --- GATE 1: transport verification (diagnostic run, NOT a measurement) -----
echo "[gate 1] NCCL_DEBUG=INFO transport diagnostic"
build_env_args "NCCL_DEBUG=INFO" "NCCL_DEBUG_SUBSYS=INIT,NET,ENV,GRAPH"
DIAG_CMD="mpirun --hostfile $HOSTFILE -np $TOTAL_RANKS ${ENV_ARGS[*]} \
  $NCCL_TESTS_DIR/build/all_reduce_perf -b 8 -e 8 -f 2 -g 1 -w 1 -n 1 -c 1 -d float"
mpirun --hostfile "$HOSTFILE" -np "$TOTAL_RANKS" "${ENV_ARGS[@]}" \
  "$NCCL_TESTS_DIR/build/all_reduce_perf" -b 8 -e 8 -f 2 -g 1 -w 1 -n 1 -c 1 -d float \
  > "$RAW_DIR/nccl_debug_info.txt" 2>&1
DIAG_RC=$?
echo "  diagnostic exit=$DIAG_RC -> nccl_debug_info.txt"

if [ "$DIAG_RC" -ne 0 ]; then
  echo >&2
  echo "ABORT: the multi-node launch itself failed (exit $DIAG_RC)." >&2
  echo "No sweep will be attempted. See $RAW_DIR/nccl_debug_info.txt" >&2
  exit 1
fi

python3 "$SCRIPT_DIR/verify_nccl_transport.py" \
  --log "$RAW_DIR/nccl_debug_info.txt" \
  --expect-transport socket \
  --expect-iface "$IFACE" \
  --min-hosts "$NNODES" \
  --json-out "$RAW_DIR/transport_verification.json"
TRANSPORT_RC=$?
if [ "$TRANSPORT_RC" -ne 0 ]; then
  echo >&2
  echo "ABORT: transport could not be verified as NET/Socket on $IFACE." >&2
  echo "A performance result from an unverified transport is not a valid" >&2
  echo "TCP baseline, so no timed run will be attempted." >&2
  echo "Evidence: $RAW_DIR/transport_verification.json" >&2
  exit 1
fi
echo

# --- run bookkeeping --------------------------------------------------------
jstr() {
  local v="${1:-}"
  if [ -z "$v" ]; then printf 'null'; return; fi
  printf '%s' "$v" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' \
    | awk 'BEGIN{printf "\""} {if(NR>1) printf "\\n"; printf "%s", $0} END{printf "\""}'
}
RUN_ENTRIES=""; RUN_COUNT=0; FAILED_RUNS=0

collective_binary() {
  case "$1" in
    all_reduce) echo all_reduce_perf ;;
    all_gather) echo all_gather_perf ;;
    reduce_scatter) echo reduce_scatter_perf ;;
    *) echo "" ;;
  esac
}

run_one() {   # <collective> <tier> <repeat> <warmup> <iters> <range...>
  local coll="$1" tier="$2" rep="$3" warmup="$4" iters="$5"; shift 5
  local range=("$@")
  local bin; bin="$(collective_binary "$coll")"
  [ -n "$bin" ] || { echo "unknown collective: $coll" >&2; FAILED_RUNS=$((FAILED_RUNS+1)); return 0; }

  local base="${coll}.${tier}.r${rep}"
  local out="$RAW_DIR/${base}.stdout.txt" err="$RAW_DIR/${base}.stderr.txt"

  # Timed runs use VERSION, never INFO: debug logging would perturb timings.
  build_env_args "NCCL_DEBUG=VERSION"
  local cmd="mpirun --hostfile $HOSTFILE -np $TOTAL_RANKS ${ENV_ARGS[*]} $NCCL_TESTS_DIR/build/$bin ${range[*]} -g 1 -w $warmup -n $iters -c 1 -d float"
  local started; started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local t0; t0="$(date +%s)"

  echo "[$tier r$rep] $coll  ($TOTAL_RANKS ranks over $NNODES nodes)"
  mpirun --hostfile "$HOSTFILE" -np "$TOTAL_RANKS" "${ENV_ARGS[@]}" \
    "$NCCL_TESTS_DIR/build/$bin" "${range[@]}" -g 1 -w "$warmup" -n "$iters" -c 1 -d float \
    >"$out" 2>"$err"
  local rc=$?
  local t1; t1="$(date +%s)"

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
      \"collective\": $(jstr "$coll"), \"binary\": $(jstr "$bin"),
      \"tier\": $(jstr "$tier"), \"repeat_index\": $rep,
      \"warmup_iterations\": $warmup, \"measured_iterations\": $iters,
      \"datatype\": \"float\",
      \"node_count\": $NNODES, \"ranks_per_node\": $RANKS_PER_NODE, \"gpu_count\": $TOTAL_RANKS,
      \"command\": $(jstr "$cmd"),
      \"exit_code\": $rc, \"correctness_gate_passed\": $ok,
      \"started_at_utc\": $(jstr "$started"), \"duration_s\": $((t1 - t0)),
      \"stdout_file\": $(jstr "${base}.stdout.txt"), \"stderr_file\": $(jstr "${base}.stderr.txt")
    }"
  return 0
}

IFS=',' read -r -a COLL_ARR <<< "$COLLECTIVES"

# --- GATE 2: smoke tier ------------------------------------------------------
SMOKE_OK=1
if [ "$TIER" = "smoke" ] || [ "$TIER" = "both" ]; then
  echo "[gate 2] smoke tier (${SMOKE_RANGE[*]}, -w $SMOKE_WARMUP -n $SMOKE_ITERS)"
  before=$FAILED_RUNS
  for c in "${COLL_ARR[@]}"; do
    run_one "$c" smoke 0 "$SMOKE_WARMUP" "$SMOKE_ITERS" "${SMOKE_RANGE[@]}"
  done
  [ "$FAILED_RUNS" -gt "$before" ] && SMOKE_OK=0
  echo
fi

# --- full tier (gated on smoke) ---------------------------------------------
if [ "$TIER" = "full" ] || [ "$TIER" = "both" ]; then
  if [ "$TIER" = "both" ] && [ "$SMOKE_OK" -ne 1 ]; then
    echo "smoke tier failed its correctness gate — skipping the full sweep." >&2
    echo "Diagnose before spending more cluster time." >&2
  else
    echo "full tier (${FULL_RANGE[*]}, -w $FULL_WARMUP -n $FULL_ITERS, $FULL_REPEATS repeats)"
    rep=0
    while [ "$rep" -lt "$FULL_REPEATS" ]; do
      for c in "${COLL_ARR[@]}"; do
        run_one "$c" full "$rep" "$FULL_WARMUP" "$FULL_ITERS" "${FULL_RANGE[@]}"
      done
      rep=$((rep + 1))
    done
    echo
  fi
fi

# --- manifest ---------------------------------------------------------------
HOSTS_JSON="["; _f=1
for h in "${HOST_ARR[@]}"; do [ $_f -eq 0 ] && HOSTS_JSON="$HOSTS_JSON,"; _f=0; HOSTS_JSON="$HOSTS_JSON$(jstr "$h")"; done
HOSTS_JSON="$HOSTS_JSON]"

RANK_MAP_JSON="null"
if [ -f "$RAW_DIR/transport_verification.json" ] && command -v python3 >/dev/null 2>&1; then
  RANK_MAP_JSON="$(python3 -c '
import json,sys
try:
    d=json.load(open(sys.argv[1])); print(json.dumps(d.get("rank_to_host")))
except Exception: print("null")' "$RAW_DIR/transport_verification.json" 2>/dev/null)"
  [ -z "$RANK_MAP_JSON" ] && RANK_MAP_JSON="null"
fi

{
  printf '{\n'
  printf '  "schema_version": 1,\n'
  printf '  "experiment_id": %s,\n' "$(jstr "$EXPERIMENT_ID")"
  printf '  "phase": "phase3",\n'
  printf '  "created_at_utc": %s,\n' "$(jstr "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
  printf '  "value_kind": "measured",\n'
  printf '  "benchmark_tool": "nccl-tests",\n'
  printf '  "nccl_tests_dir": %s,\n' "$(jstr "$NCCL_TESTS_DIR")"
  printf '  "nccl_tests_commit": %s,\n' "$(jstr "$(git -C "$NCCL_TESTS_DIR" rev-parse HEAD 2>/dev/null || echo '')")"
  printf '  "git_commit": %s,\n' "$(jstr "$GIT_FULL")"
  printf '  "node_count": %s,\n' "$NNODES"
  printf '  "ranks_per_node": %s,\n' "$RANKS_PER_NODE"
  printf '  "gpu_count": %s,\n' "$TOTAL_RANKS"
  printf '  "hosts": %s,\n' "$HOSTS_JSON"
  printf '  "rank_to_host": %s,\n' "$RANK_MAP_JSON"
  printf '  "network": "tcp",\n'
  printf '  "transport": "NET/Socket",\n'
  printf '  "transport_verified": true,\n'
  printf '  "net_interface": %s,\n' "$(jstr "$IFACE")"
  printf '  "mpi_implementation": %s,\n' "$(jstr "$MPI_IMPL")"
  printf '  "mpi_version": %s,\n' "$(jstr "$MPI_VERSION")"
  printf '  "launcher": "mpirun",\n'
  printf '  "hostfile": "hostfile",\n'
  printf '  "nccl_env": { "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": %s },\n' "$(jstr "$IFACE")"
  printf '  "runner": "scripts/run_nccl_multinode.sh",\n'
  printf '  "failed_runs": %s,\n' "$FAILED_RUNS"
  printf '  "runs": [%s\n  ]\n' "$RUN_ENTRIES"
  printf '}\n'
} > "$RAW_DIR/run_manifest.json"

echo "manifest -> $RAW_DIR/run_manifest.json"
echo "runs: $RUN_COUNT   failed: $FAILED_RUNS"
echo
echo "Next: copy $RAW_DIR back, TERMINATE the cluster, then parse locally:"
echo "  python3 scripts/parse_nccl_output.py --raw-dir results/raw/$EXPERIMENT_ID --phase phase3"

[ "$FAILED_RUNS" -eq 0 ] || exit 1
