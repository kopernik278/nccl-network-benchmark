#!/usr/bin/env bash
# Phase 10 final benchmark: one baseline, one evidence-based optimization, and
# one negative control, on one host, in one session.
#
#   25 MiB  reference        — PyTorch's conventional default-style capacity
#    4 MiB  optimized        — chosen because Phase 9 measured earlier
#                              communication, a shorter exposed tail and a
#                              lower step time there. Not a grid search.
#   64 MiB  negative control — kept deliberately: it produces fewer and
#                              individually more efficient collectives and is
#                              still slower, which is the whole point.
#   serial  non-overlapped   — one flat AllReduce after backward completes, so
#                              the overlap benefit is anchored on this host.
#
# Repeats are INDEPENDENT process launches, not repeats inside one process:
# three in-process repeats share a warm communicator and allocator and
# understate run-to-run variance (Phase 9 §5).
#
# Usage: run_final_ddp.sh <out-dir> "<transport env or empty>" [model args...]
set -uo pipefail

OUT=${1:?usage: run_final_ddp.sh <out-dir> "<transport env>" [model args]}
TRANSPORT_ENV=${2:-}
shift 2 || true
MODEL=${*:-"--layers 8 --heads 12 --embd 768 --vocab 16384 --seq 1024 --batch 16 --pool 4 --precision bf16"}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
T="$REPO/src/ddp/train_ddp.py"
mkdir -p "$OUT"

WARMUP=${WARMUP:-12}      # DDP rebuilds its buckets in the first steps
STEPS=${STEPS:-30}
LAUNCHES=${LAUNCHES:-3}

# shellcheck disable=SC2086
export ${TRANSPORT_ENV:-DUMMY_UNUSED=1}

echo "# model: $MODEL"
echo "# transport env: ${TRANSPORT_ENV:-<NCCL defaults>}"
echo "# warmup=$WARMUP steps=$STEPS independent_launches=$LAUNCHES"

launch() {  # launch <tag> <port> <args...>
  local tag=$1 port=$2; shift 2
  # shellcheck disable=SC2086
  timeout -k 20 900 "$@" > "$OUT/$tag.stdout.txt" 2>&1
  local rc=$?
  local step
  step=$(grep -h '^train' "$OUT/$tag.stdout.txt" | head -1 | cut -d, -f7)
  printf '  %-22s rc=%-3s step=%s ms\n' "$tag" "$rc" "${step:-<none>}"
  [ "$rc" -eq 0 ] || echo "     !! non-zero exit; this configuration is not reportable"
}

echo
echo "===== single-GPU reference (compute floor) ====="
for L in $(seq 0 $((LAUNCHES - 1))); do
  # shellcheck disable=SC2086
  launch "single-L$L" 0 python3 "$T" --mode single --label single \
    --warmup $WARMUP --steps $STEPS --repeats 1 $MODEL
done

echo
echo "===== DDP: reference, optimized, negative control ====="
for MB in 25 4 64; do
  for L in $(seq 0 $((LAUNCHES - 1))); do
    # shellcheck disable=SC2086
    launch "final-${MB}mb-L$L" 0 torchrun --nproc_per_node=4 \
      --master_port=$((29700 + MB * 10 + L)) "$T" --mode ddp --bucket-mb "$MB" \
      --label "final-${MB}mb" --warmup $WARMUP --steps $STEPS --repeats 1 \
      --nosync-probe $MODEL
  done
done

echo
echo "===== non-overlapped control (one flat AllReduce after backward) ====="
for L in $(seq 0 $((LAUNCHES - 1))); do
  # shellcheck disable=SC2086
  launch "serial-L$L" 0 torchrun --nproc_per_node=4 \
    --master_port=$((29790 + L)) "$T" --mode serial --label serial \
    --warmup $WARMUP --steps $STEPS --repeats 1 $MODEL
done

echo
echo "===== transport evidence for the benchmarked configuration ====="
# shellcheck disable=SC2086
NCCL_DEBUG=INFO timeout -k 20 600 torchrun --nproc_per_node=4 --master_port=29799 \
  "$T" --mode ddp --bucket-mb 4 --warmup 3 --steps 3 --repeats 1 $MODEL \
  > "$OUT/nccl_debug_transport_final.txt" 2>&1
echo "  rc=$?"
grep -oE 'Channel [0-9]+[^:]*: [0-9]+\[[0-9]+\] -> [0-9]+\[[0-9]+\] via .*' \
  "$OUT/nccl_debug_transport_final.txt" | sort -u | sed 's/^/  /'

echo
echo "===== FINAL BENCHMARK DONE ====="
