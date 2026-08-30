#!/usr/bin/env bash
# Preflight for a multi-GPU DDP benchmark run.
#
# This encodes the reliability lesson this project learned three times over:
# a CUDA peer-capability query is NOT evidence that the P2P path works. In
# Phase 6 the capability bit was set and peer copies returned NaN. In Phases 8
# and 9, on three different cloud hosts, the bit was set and NCCL deadlocked
# after successfully building its rings.
#
# So the gate is functional, not declarative. It runs a real DDP step under a
# timeout and checks an invariant that silent corruption would break:
# param_sync_max_abs_diff, the maximum |p_rank - p_rank0| after real gradient
# AllReduces. Only a configuration that passes is allowed to be benchmarked.
#
# Order:  topology  ->  capability  ->  functional collective  ->  correctness
#         gate      ->  transport decision
#
# Usage:  preflight_ddp.sh <out-dir> [gpus]
# Exit:   0 with TRANSPORT_ENV written to <out-dir>/transport_env.txt
#         1 if no transport passes the gate — in which case nothing is
#           benchmarked, because a wrong number is worse than no number.
set -uo pipefail

OUT=${1:?usage: preflight_ddp.sh <out-dir> [gpus]}
NGPU=${2:-4}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
T="$REPO/src/ddp/train_ddp.py"
mkdir -p "$OUT"

# Small enough to fail fast, large enough that every rank really communicates.
TINY="--layers 2 --heads 2 --embd 128 --vocab 1024 --seq 128 --batch 2 --pool 2"
GATE_TIMEOUT=${GATE_TIMEOUT:-180}

echo "===== 1. hardware and topology ====="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
  | tee "$OUT/gpus.txt"
nvidia-smi topo -m > "$OUT/topology.txt" 2>&1
sed -n '1,12p' "$OUT/topology.txt"

echo
echo "===== 2. software versions ====="
python3 - <<'PY' | tee "$OUT/versions.txt"
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} "
      f"nccl={'.'.join(map(str, torch.cuda.nccl.version()))} "
      f"bf16={torch.cuda.is_bf16_supported()} devices={torch.cuda.device_count()}")
PY

echo
echo "===== 3. peer capability (declarative — NOT trusted on its own) ====="
python3 - "$NGPU" <<'PY' | tee "$OUT/peer_capability.txt"
import sys, torch
n = int(sys.argv[1])
for i in range(n):
    row = [("-" if i == j else ("yes" if torch.cuda.can_device_access_peer(i, j) else "no"))
           for j in range(n)]
    print(f"GPU{i}: " + " ".join(f"{v:>3}" for v in row))
print("# cudaDeviceCanAccessPeer only. Phases 6/8/9 each found this bit set on a "
      "path that did not work. It is recorded, not believed.")
PY

echo
echo "===== 4. functional collective validation + correctness gate ====="
# Each candidate runs a real DDP training step. A hang is caught by the
# timeout; silent corruption is caught by param_sync_max_abs_diff.
CHOSEN=""; CHOSEN_LOG=""; ACCEPTED=0
for CAND in "" "NCCL_P2P_DISABLE=1"; do
  NAME=${CAND:-p2p-enabled}
  NAME=${NAME//=/_}
  LOG="$OUT/gate_${NAME}.txt"
  echo "--- candidate: ${CAND:-<NCCL default, P2P enabled>} ---"
  # shellcheck disable=SC2086
  env ${CAND:-DUMMY_UNUSED=1} NCCL_DEBUG=INFO \
    timeout -k 15 "$GATE_TIMEOUT" torchrun --nproc_per_node="$NGPU" \
      --master_port=29401 "$T" --mode ddp --bucket-mb 25 \
      --warmup 3 --steps 3 --repeats 1 --precision fp32 $TINY \
      > "$LOG" 2>&1
  RC=$?
  LINE=$(grep -m1 '^# CORRECTNESS' "$LOG" || true)
  TRANSPORT=$(grep -oE 'via (P2P|SHM|NET)[^ ]*' "$LOG" | sort -u | tr '\n' ' ')
  echo "  rc=$RC  transport: ${TRANSPORT:-<none observed>}"
  echo "  ${LINE:-# CORRECTNESS <absent — run did not complete>}"

  if [ "$RC" -ne 0 ]; then
    echo "  VERDICT: rejected (rc=$RC; $( [ "$RC" -eq 124 ] && echo 'timed out — hang' || echo 'failed'))"
    continue
  fi
  if ! grep -q 'loss_finite=True grads_finite=True' "$LOG"; then
    echo "  VERDICT: rejected (loss or gradients not finite)"
    continue
  fi
  DIFF=$(sed -n 's/.*param_sync_max_abs_diff=\([^ ]*\).*/\1/p' "$LOG" | head -1)
  if [ -z "$DIFF" ] || ! python3 -c "import sys; sys.exit(0 if float('$DIFF') == 0.0 else 1)"; then
    echo "  VERDICT: rejected (param_sync_max_abs_diff=${DIFF:-<missing>}, ranks diverged)"
    continue
  fi
  echo "  VERDICT: accepted (param_sync_max_abs_diff=$DIFF)"
  CHOSEN=$CAND; CHOSEN_LOG=$LOG; ACCEPTED=1
  break
done

echo
echo "===== 5. transport decision ====="
if [ "$ACCEPTED" -ne 1 ]; then
  echo "PREFLIGHT FAILED: no transport passed the functional gate."
  echo "Nothing is benchmarked. A wrong number is worse than no number."
  exit 1
fi
printf '%s\n' "$CHOSEN" > "$OUT/transport_env.txt"
if [ -z "$CHOSEN" ]; then
  echo "chosen: NCCL defaults (P2P functional on this host)"
else
  echo "chosen: $CHOSEN"
  echo "note: NCCL's own transport choice deadlocked here despite the capability"
  echo "      bit; falling back to the transport that passed the functional gate."
fi
grep -oE 'via (P2P|SHM|NET)[^ ]*' "$CHOSEN_LOG" 2>/dev/null \
  | sort -u | sed 's/^/  measured transport: /' || true
echo "PREFLIGHT OK"
