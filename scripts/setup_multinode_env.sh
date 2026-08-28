#!/usr/bin/env bash
#
# setup_multinode_env.sh — make a node ready to take part in an MPI launch.
#
# Two gaps stand between a fresh RunPod pod and a working multi-node run, both
# observed rather than assumed: the runpod/pytorch image ships NO MPI (Phase 1B
# environment capture recorded "mpi: <not installed>"), and mpirun needs
# passwordless SSH from the launching node to every other node.
#
# Usage, on EVERY node:
#     scripts/setup_multinode_env.sh -i
#
# then once, on the launching (primary) node only:
#     scripts/setup_multinode_env.sh -k -H node-a,node-b   # distribute a key
#     scripts/setup_multinode_env.sh -v -H node-a,node-b   # verify reachability
#
#   -i  install OpenMPI if absent
#   -k  create an inter-node SSH key and print what to install on the peers
#   -v  verify passwordless SSH and mpirun reachability for -H hosts
#   -H  comma-separated hosts (for -k / -v)
#
# This script only prepares; it measures nothing and starts no benchmark.

set -uo pipefail

DO_INSTALL=0; DO_KEY=0; DO_VERIFY=0; HOSTS=""
while getopts ":ikvH:h" opt; do
  case "$opt" in
    i) DO_INSTALL=1 ;;
    k) DO_KEY=1 ;;
    v) DO_VERIFY=1 ;;
    H) HOSTS="$OPTARG" ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done
[ $((DO_INSTALL + DO_KEY + DO_VERIFY)) -gt 0 ] || { echo "nothing to do; see -h" >&2; exit 2; }

die() { echo "ERROR: $*" >&2; exit 1; }

# --- install MPI ------------------------------------------------------------
if [ "$DO_INSTALL" -eq 1 ]; then
  if command -v mpirun >/dev/null 2>&1; then
    echo "[install] mpirun already present: $(mpirun --version 2>&1 | head -1)"
  else
    echo "[install] installing OpenMPI"
    if command -v apt-get >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq >/dev/null 2>&1
      apt-get install -y -qq openmpi-bin libopenmpi-dev openssh-client >/dev/null 2>&1 \
        || die "apt-get install of openmpi failed"
    else
      die "no apt-get; install an MPI manually and re-run"
    fi
    command -v mpirun >/dev/null 2>&1 || die "mpirun still not on PATH after install"
    echo "[install] ok: $(mpirun --version 2>&1 | head -1)"
    command -v mpicc >/dev/null 2>&1 \
      && echo "[install] mpicc present — nccl-tests can be built with -m" \
      || echo "[install] WARNING: mpicc missing; nccl-tests cannot be built with MPI=1" >&2
  fi
fi

# --- inter-node SSH key -----------------------------------------------------
KEY=~/.ssh/id_multinode
if [ "$DO_KEY" -eq 1 ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  if [ ! -f "$KEY" ]; then
    ssh-keygen -t ed25519 -N '' -f "$KEY" -C "nccl-multinode" >/dev/null \
      || die "ssh-keygen failed"
    echo "[key] created $KEY"
  else
    echo "[key] reusing existing $KEY"
  fi
  # Authorise on this node too: mpirun may place a rank locally via ssh.
  touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
  grep -qxF "$(cat "$KEY".pub)" ~/.ssh/authorized_keys 2>/dev/null \
    || cat "$KEY".pub >> ~/.ssh/authorized_keys
  cat > ~/.ssh/config <<EOF
Host *
  IdentityFile $KEY
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
EOF
  chmod 600 ~/.ssh/config
  echo "[key] local authorized_keys and ssh config updated"
  echo
  echo "Install this public key in ~/.ssh/authorized_keys on EVERY other node:"
  echo "--------8<--------"
  cat "$KEY".pub
  echo "-------->8--------"
fi

# --- verify -----------------------------------------------------------------
if [ "$DO_VERIFY" -eq 1 ]; then
  [ -n "$HOSTS" ] || die "-v requires -H host1,host2"
  IFS=',' read -r -a HOST_ARR <<< "$HOSTS"
  fail=0
  echo "[verify] passwordless SSH"
  for h in "${HOST_ARR[@]}"; do
    if ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
           -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
           "$h" "hostname" >/dev/null 2>&1; then
      echo "  ok    $h"
    else
      echo "  FAIL  $h  (no passwordless SSH)" >&2; fail=1
    fi
  done

  echo "[verify] mpirun reachability"
  HF="$(mktemp)"
  for h in "${HOST_ARR[@]}"; do echo "$h slots=1" >> "$HF"; done
  if mpirun --hostfile "$HF" -np "${#HOST_ARR[@]}" hostname 2>/dev/null | sort -u; then
    echo "  ok    mpirun reached all hosts"
  else
    echo "  FAIL  mpirun could not launch across the hostfile" >&2; fail=1
  fi
  rm -f "$HF"

  if [ "$fail" -ne 0 ]; then
    echo >&2
    echo "Multi-node prerequisites are NOT satisfied. Fix these before spending" >&2
    echo "cluster time — run_nccl_multinode.sh will refuse to produce a result" >&2
    echo "it cannot verify." >&2
    exit 1
  fi
  echo "[verify] all prerequisites satisfied"
fi
