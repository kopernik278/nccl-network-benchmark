#!/usr/bin/env bash
#
# setup_nccl_tests.sh — build nccl-tests on a GPU node.
#
# Kept separate from the runner so that build failures surface immediately,
# before any measurement is attempted, and so the build step can be timed and
# charged separately from measurement time.
#
# Usage:
#   scripts/setup_nccl_tests.sh [-d DEST_DIR] [-r GIT_REF] [-m]
#
# Env:
#   CUDA_HOME   CUDA toolkit root (default: /usr/local/cuda)
#   NCCL_HOME   NCCL root, only needed if NCCL is not in the default paths
#   MPI_HOME    MPI root, only needed with -m if mpicc is not on PATH
#   BUILD_JOBS  parallel make jobs (default: nproc)
#
# Single-node phases (1 and 2) build WITHOUT MPI and run nccl-tests in
# single-process multi-GPU mode (-g N), which removes MPI as a dependency and
# as a variable.
#
# -m builds with MPI=1 for the multi-node phases (3 onward), where ranks must
# span hosts and a launcher is unavoidable. The two builds are otherwise
# identical, so a multi-node binary can still run single-node.

set -euo pipefail

DEST="${HOME}/nccl-tests"
GIT_REF=""
WITH_MPI=0

while getopts ":d:r:mh" opt; do
  case "$opt" in
    d) DEST="$OPTARG" ;;
    r) GIT_REF="$OPTARG" ;;
    m) WITH_MPI=1 ;;
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

MPI_DESC="MPI disabled"
if [ "$WITH_MPI" -eq 1 ]; then
  # Fail loudly here rather than producing a non-MPI binary that would later
  # run N independent single-rank jobs and look like a successful launch.
  if [ -n "${MPI_HOME:-}" ]; then
    [ -x "$MPI_HOME/bin/mpicc" ] || { echo "MPI_HOME=$MPI_HOME has no bin/mpicc" >&2; exit 1; }
  elif ! command -v mpicc >/dev/null 2>&1; then
    echo "-m requested but mpicc was not found; install an MPI or set MPI_HOME" >&2
    exit 1
  fi
  MAKE_ARGS+=("MPI=1")
  if [ -n "${MPI_HOME:-}" ]; then MAKE_ARGS+=("MPI_HOME=$MPI_HOME"); fi
  MPI_DESC="MPI enabled"
fi

echo "building nccl-tests (jobs=$BUILD_JOBS, $MPI_DESC)"
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

if [ "$WITH_MPI" -eq 1 ]; then
  # A binary built without MPI still links and runs, so verify the MPI symbols
  # are really there instead of trusting the make flag.
  if command -v ldd >/dev/null 2>&1 && ldd "$DEST/build/all_reduce_perf" 2>/dev/null | grep -qi 'libmpi'; then
    echo "  ok: binary links libmpi"
  else
    echo "  WARNING: -m was requested but the binary does not link libmpi." >&2
    echo "  A non-MPI binary under mpirun runs N independent single-rank jobs" >&2
    echo "  that can look like a successful multi-node launch. Refusing." >&2
    exit 1
  fi
fi

echo
echo "nccl-tests ready at: $DEST"
echo "commit: $(git rev-parse HEAD)"
if [ "$WITH_MPI" -eq 1 ]; then echo "built with MPI=1"; fi
echo
echo "Export this for the runner:"
echo "  export NCCL_TESTS_DIR=$DEST"
