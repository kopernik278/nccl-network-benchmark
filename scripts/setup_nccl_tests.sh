#!/usr/bin/env bash
#
# setup_nccl_tests.sh — build nccl-tests on a GPU node.
#
# Kept separate from the runner so that build failures surface immediately,
# before any measurement is attempted, and so the build step can be timed and
# charged separately from measurement time.
#
# Usage:
#   scripts/setup_nccl_tests.sh [-d DEST_DIR] [-r GIT_REF]
#
# Env:
#   CUDA_HOME   CUDA toolkit root (default: /usr/local/cuda)
#   NCCL_HOME   NCCL root, only needed if NCCL is not in the default paths
#   BUILD_JOBS  parallel make jobs (default: nproc)
#
# MPI is intentionally NOT enabled: Phase 1 runs nccl-tests in single-process
# multi-GPU mode (-g N), which removes MPI as a dependency and as a variable.
# Multi-node phases will rebuild with MPI=1.

set -euo pipefail

DEST="${HOME}/nccl-tests"
GIT_REF=""

while getopts ":d:r:h" opt; do
  case "$opt" in
    d) DEST="$OPTARG" ;;
    r) GIT_REF="$OPTARG" ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
[ -x "$CUDA_HOME/bin/nvcc" ] || command -v nvcc >/dev/null 2>&1 || {
  echo "nvcc not found (looked in \$CUDA_HOME=$CUDA_HOME and \$PATH)" >&2
  exit 1
}

if [ ! -d "$DEST/.git" ]; then
  echo "cloning nccl-tests into $DEST"
  git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git "$DEST"
fi

cd "$DEST"
if [ -n "$GIT_REF" ]; then
  echo "checking out pinned ref $GIT_REF"
  git fetch --depth 1 origin "$GIT_REF"
  git checkout --detach FETCH_HEAD
fi

MAKE_ARGS=("CUDA_HOME=$CUDA_HOME")
if [ -n "${NCCL_HOME:-}" ]; then
  MAKE_ARGS+=("NCCL_HOME=$NCCL_HOME")
fi

echo "building nccl-tests (jobs=$BUILD_JOBS, MPI disabled)"
make -j"$BUILD_JOBS" "${MAKE_ARGS[@]}"

# The runner needs exactly these three for Phase 1.
missing=0
for b in all_reduce_perf all_gather_perf reduce_scatter_perf; do
  if [ -x "$DEST/build/$b" ]; then
    echo "  ok: build/$b"
  else
    echo "  MISSING: build/$b" >&2
    missing=1
  fi
done
[ "$missing" -eq 0 ] || { echo "build incomplete" >&2; exit 1; }

echo
echo "nccl-tests ready at: $DEST"
echo "commit: $(git rev-parse HEAD)"
echo
echo "Export this for the runner:"
echo "  export NCCL_TESTS_DIR=$DEST"
