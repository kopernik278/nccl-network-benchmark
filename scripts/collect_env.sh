#!/usr/bin/env bash
#
# collect_env.sh — capture reproducibility metadata for an NCCL benchmark run.
#
# Designed to run on a freshly provisioned GPU node where we cannot assume any
# particular tool exists. Every probe degrades gracefully: a missing tool
# yields JSON null plus a note, never a wrong value and never a hard failure.
#
# Usage:
#   scripts/collect_env.sh [-o OUTPUT_DIR]
#
# Writes OUTPUT_DIR/env.json  (machine-readable, consumed by the parser)
#        OUTPUT_DIR/env.txt   (human-readable)
#
# Security: environment variables are captured through a prefix allowlist and
# then filtered by a name-based redaction rule. A blanket `env` dump is never
# written, so credentials cannot reach results/ or git.

# Deliberately no `set -e`: probing for absent tools is normal operation here.
set -uo pipefail

OUT_DIR="."
while getopts ":o:h" opt; do
  case "$opt" in
    o) OUT_DIR="$OPTARG" ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR" || { echo "cannot create $OUT_DIR" >&2; exit 1; }
JSON_OUT="$OUT_DIR/env.json"
TEXT_OUT="$OUT_DIR/env.txt"

have() { command -v "$1" >/dev/null 2>&1; }

# Run a command, returning its stdout, or the empty string if it is missing
# or fails. Callers treat empty as "unavailable".
try() {
  if have "$1"; then "$@" 2>/dev/null || true; fi
}

HAVE_PYTHON3=0
have python3 && HAVE_PYTHON3=1

# Emit a JSON string literal (or bare null when the value is empty).
json_str() {
  local v="${1:-}"
  if [ -z "$v" ]; then printf 'null'; return; fi
  if [ "$HAVE_PYTHON3" = "1" ]; then
    printf '%s' "$v" | python3 -c 'import json,sys; sys.stdout.write(json.dumps(sys.stdin.read()))'
  else
    printf '%s' "$v" \
      | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/\\t/g' -e 's/\r//g' \
      | awk 'BEGIN{printf "\""} {if(NR>1) printf "\\n"; printf "%s", $0} END{printf "\""}'
  fi
}

# Emit a bare JSON number, or null if not numeric.
json_num() {
  local v="${1:-}"
  case "$v" in
    ''|*[!0-9]*) printf 'null' ;;
    *) printf '%s' "$v" ;;
  esac
}

# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------
CAPTURED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOSTNAME_V="$(try hostname)"
[ -z "$HOSTNAME_V" ] && HOSTNAME_V="${HOSTNAME:-}"
KERNEL_V="$(try uname -r)"
ARCH_V="$(try uname -m)"

OS_V=""
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  OS_V="$(. /etc/os-release 2>/dev/null && printf '%s' "${PRETTY_NAME:-${NAME:-}}")"
fi
[ -z "$OS_V" ] && OS_V="$(try uname -s)"

CPU_MODEL=""
CPU_CORES=""
if have lscpu; then
  CPU_MODEL="$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
  CPU_CORES="$(lscpu -p=CPU 2>/dev/null | grep -vc '^#')"
fi
if [ -z "$CPU_MODEL" ] && [ -r /proc/cpuinfo ]; then
  CPU_MODEL="$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo | head -1)"
  CPU_CORES="$(grep -c '^processor' /proc/cpuinfo)"
fi
if [ -z "$CPU_MODEL" ] && have sysctl; then   # macOS, for local-development capture
  CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  CPU_CORES="$(sysctl -n hw.ncpu 2>/dev/null)"
fi

MEM_TOTAL_KB=""
[ -r /proc/meminfo ] && MEM_TOTAL_KB="$(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p' /proc/meminfo)"

# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
PROVIDER="${BENCH_PROVIDER:-}"
PROVIDER_INSTANCE="${RUNPOD_POD_ID:-}"
if [ -z "$PROVIDER" ]; then
  if [ -n "$PROVIDER_INSTANCE" ]; then PROVIDER="runpod"; else PROVIDER="unknown"; fi
fi

# ---------------------------------------------------------------------------
# GPU / driver / CUDA
# ---------------------------------------------------------------------------
GPU_COUNT=""
GPU_MODEL=""
DRIVER_VERSION=""
CUDA_VERSION_SMI=""
GPUS_JSON="null"
TOPOLOGY=""
NVLINK_STATUS=""

if have nvidia-smi; then
  # PCIe link generation/width is required to interpret bandwidth on a
  # PCIe-connected (non-NVLink) system: without it a measured GB/s figure
  # cannot be compared against the link it actually crossed.
  GPU_QUERY="$(nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max \
                          --format=csv,noheader,nounits 2>/dev/null)"
  if [ -n "$GPU_QUERY" ]; then
    GPU_COUNT="$(printf '%s\n' "$GPU_QUERY" | grep -c .)"
    GPU_MODEL="$(printf '%s\n' "$GPU_QUERY" | head -1 | awk -F', *' '{print $2}')"
    DRIVER_VERSION="$(printf '%s\n' "$GPU_QUERY" | head -1 | awk -F', *' '{print $5}')"

    GPUS_JSON="["
    first=1
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      idx="$(printf '%s' "$line" | awk -F', *' '{print $1}')"
      nm="$(printf '%s' "$line" | awk -F', *' '{print $2}')"
      bus="$(printf '%s' "$line" | awk -F', *' '{print $3}')"
      mem="$(printf '%s' "$line" | awk -F', *' '{print $4}')"
      lgen="$(printf '%s' "$line" | awk -F', *' '{print $6}')"
      lwid="$(printf '%s' "$line" | awk -F', *' '{print $7}')"
      mgen="$(printf '%s' "$line" | awk -F', *' '{print $8}')"
      mwid="$(printf '%s' "$line" | awk -F', *' '{print $9}')"
      [ $first -eq 0 ] && GPUS_JSON="$GPUS_JSON,"
      first=0
      GPUS_JSON="$GPUS_JSON{\"index\":$(json_num "$idx"),\"name\":$(json_str "$nm"),\"pci_bus_id\":$(json_str "$bus"),\"memory_total_mib\":$(json_num "$mem"),\"pcie_link_gen_current\":$(json_num "$lgen"),\"pcie_link_width_current\":$(json_num "$lwid"),\"pcie_link_gen_max\":$(json_num "$mgen"),\"pcie_link_width_max\":$(json_num "$mwid")}"
    done <<EOF
$GPU_QUERY
EOF
    GPUS_JSON="$GPUS_JSON]"
  fi

  # Driver-reported CUDA runtime ceiling, e.g. "CUDA Version: 12.4"
  CUDA_VERSION_SMI="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9.]*\).*/\1/p' | head -1)"
  TOPOLOGY="$(nvidia-smi topo -m 2>/dev/null)"
  NVLINK_STATUS="$(nvidia-smi nvlink --status 2>/dev/null)"
fi

# Toolkit CUDA version (may differ from the driver ceiling above).
CUDA_VERSION_NVCC=""
NVCC_BIN=""
if have nvcc; then
  NVCC_BIN="nvcc"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  NVCC_BIN="/usr/local/cuda/bin/nvcc"
fi
if [ -n "$NVCC_BIN" ]; then
  CUDA_VERSION_NVCC="$("$NVCC_BIN" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
fi

# NVLink presence: infer from the topology matrix, which is the same source a
# reader of the report would check. Unknown if nvidia-smi is absent.
NVLINK_PRESENT="null"
if [ -n "$TOPOLOGY" ]; then
  if printf '%s' "$TOPOLOGY" | grep -qE '(^|[[:space:]])NV[0-9]+([[:space:]]|$)'; then
    NVLINK_PRESENT="true"
  else
    NVLINK_PRESENT="false"
  fi
fi

# Dominant inter-GPU link class, best effort.
TOPOLOGY_SUMMARY=""
if [ -n "$TOPOLOGY" ]; then
  TOPOLOGY_SUMMARY="$(printf '%s\n' "$TOPOLOGY" \
    | grep -E '^GPU[0-9]+' \
    | grep -oE '(NV[0-9]+|PIX|PXB|PHB|NODE|SYS)' \
    | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')"
fi

PCIE_SUMMARY=""
if [ -n "${GPU_QUERY:-}" ]; then
  _l="$(printf '%s\n' "$GPU_QUERY" | head -1)"
  _g="$(printf '%s' "$_l" | awk -F', *' '{print $6}')"
  _w="$(printf '%s' "$_l" | awk -F', *' '{print $7}')"
  [ -n "$_g" ] && PCIE_SUMMARY="Gen${_g} x${_w}"
fi

# ---------------------------------------------------------------------------
# NCCL version — several detection strategies, most reliable first.
# ---------------------------------------------------------------------------
NCCL_VERSION=""
NCCL_VERSION_SOURCE=""

# 1. nccl.h macros: authoritative for the headers we would compile against.
for h in /usr/include/nccl.h /usr/local/cuda/include/nccl.h \
         /usr/local/include/nccl.h /opt/nccl/include/nccl.h; do
  if [ -r "$h" ]; then
    _maj="$(awk '/^#define NCCL_MAJOR/{print $3}' "$h" 2>/dev/null | head -1)"
    _min="$(awk '/^#define NCCL_MINOR/{print $3}' "$h" 2>/dev/null | head -1)"
    _pat="$(awk '/^#define NCCL_PATCH/{print $3}' "$h" 2>/dev/null | head -1)"
    if [ -n "$_maj" ]; then
      NCCL_VERSION="${_maj}.${_min:-0}.${_pat:-0}"
      NCCL_VERSION_SOURCE="nccl.h ($h)"
      break
    fi
  fi
done

# 2. Shared library soname, e.g. libnccl.so.2.21.5
if [ -z "$NCCL_VERSION" ]; then
  _lib="$(try ldconfig -p | grep -oE 'libnccl\.so\.[0-9]+(\.[0-9]+)*' | sort -V | tail -1)"
  if [ -z "$_lib" ]; then
    for d in /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/local/cuda/lib64 /opt/nccl/lib; do
      _cand="$(ls -1 "$d"/libnccl.so.[0-9]* 2>/dev/null | sort -V | tail -1)"
      if [ -n "$_cand" ]; then _lib="$(basename "$_cand")"; break; fi
    done
  fi
  if [ -n "$_lib" ]; then
    _v="${_lib#libnccl.so.}"
    # A bare soname like "libnccl.so.2" carries no useful patch level.
    case "$_v" in
      *.*) NCCL_VERSION="$_v"; NCCL_VERSION_SOURCE="shared library soname" ;;
    esac
  fi
fi

# 3. PyTorch's bundled NCCL, if torch happens to be installed.
if [ -z "$NCCL_VERSION" ] && [ "$HAVE_PYTHON3" = "1" ]; then
  _v="$(python3 -c 'import torch;print(".".join(map(str,torch.cuda.nccl.version())))' 2>/dev/null)"
  if [ -n "$_v" ]; then
    NCCL_VERSION="$_v"
    NCCL_VERSION_SOURCE="torch.cuda.nccl.version()"
  fi
fi

# The runner additionally records the NCCL banner from NCCL_DEBUG=VERSION at
# actual run time, which is the definitive value for a given measurement.

# ---------------------------------------------------------------------------
# MPI / compiler / python / git
# ---------------------------------------------------------------------------
# Implementations disagree on --version layout: Open MPI puts the version on
# line 1, while MPICH/Hydra leads with "HYDRA build details:" and reports the
# version several lines down. Match on content rather than trusting line 1.
MPI_VERSION=""
_mpi_raw=""
if have mpirun; then
  _mpi_raw="$(mpirun --version 2>&1)"
elif have mpiexec; then
  _mpi_raw="$(mpiexec --version 2>&1)"
fi
if [ -n "$_mpi_raw" ]; then
  MPI_VERSION="$(printf '%s\n' "$_mpi_raw" | grep -iE 'open mpi|mvapich|intel\(r\) mpi' | head -1)"
  [ -z "$MPI_VERSION" ] && MPI_VERSION="$(printf '%s\n' "$_mpi_raw" \
    | sed -n 's/^[[:space:]]*Version:[[:space:]]*\(.*\)$/MPICH \1/p' | head -1)"
  [ -z "$MPI_VERSION" ] && MPI_VERSION="$(printf '%s\n' "$_mpi_raw" | head -1)"
  # Trim leading/trailing whitespace.
  MPI_VERSION="$(printf '%s' "$MPI_VERSION" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
fi

COMPILER_VERSION="$(try gcc --version | head -1)"
[ -z "$COMPILER_VERSION" ] && COMPILER_VERSION="$(try cc --version | head -1)"

PYTHON_VERSION="$(try python3 --version)"

GIT_COMMIT=""
GIT_BRANCH=""
GIT_DIRTY="null"
if have git && git rev-parse --git-dir >/dev/null 2>&1; then
  GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null)"
  GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then GIT_DIRTY="true"; else GIT_DIRTY="false"; fi
fi

# ---------------------------------------------------------------------------
# Environment variables: prefix allowlist, then name-based redaction.
# ---------------------------------------------------------------------------
ENV_JSON="{"
_first_env=1
while IFS= read -r kv; do
  k="${kv%%=*}"
  v="${kv#*=}"
  case "$k" in
    NCCL_*|CUDA_*|NVIDIA_*|UCX_*|OMPI_*|RUNPOD_*) ;;
    *) continue ;;
  esac
  # Never emit anything whose name suggests a credential.
  case "$k" in
    *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*PASSWD*|*CREDENTIAL*|*AUTH*|*PRIVATE*)
      v="<redacted>" ;;
  esac
  [ $_first_env -eq 0 ] && ENV_JSON="$ENV_JSON,"
  _first_env=0
  ENV_JSON="$ENV_JSON$(json_str "$k"):$(json_str "$v")"
done <<EOF
$(env 2>/dev/null | sort)
EOF
ENV_JSON="$ENV_JSON}"

# ---------------------------------------------------------------------------
# Tool availability, so a null above can be explained.
# ---------------------------------------------------------------------------
TOOLS_JSON="{"
_first_tool=1
for t in nvidia-smi nvcc gcc git python3 mpirun ibstat ibv_devinfo lspci lscpu; do
  if have "$t"; then _av=true; else _av=false; fi
  [ $_first_tool -eq 0 ] && TOOLS_JSON="$TOOLS_JSON,"
  _first_tool=0
  TOOLS_JSON="$TOOLS_JSON$(json_str "$t"):$_av"
done
TOOLS_JSON="$TOOLS_JSON}"

# RDMA presence, so later phases can assert it and Phase 1 can deny it.
RDMA_DEVICES="$(try ibv_devinfo -l)"
[ -z "$RDMA_DEVICES" ] && RDMA_DEVICES="$(try ibstat -l)"

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
{
  printf '{\n'
  printf '  "schema_version": 1,\n'
  printf '  "captured_at_utc": %s,\n'   "$(json_str "$CAPTURED_AT")"
  printf '  "provider": %s,\n'          "$(json_str "$PROVIDER")"
  printf '  "provider_instance_id": %s,\n' "$(json_str "$PROVIDER_INSTANCE")"
  printf '  "hostname": %s,\n'          "$(json_str "$HOSTNAME_V")"
  printf '  "os": %s,\n'                "$(json_str "$OS_V")"
  printf '  "kernel": %s,\n'            "$(json_str "$KERNEL_V")"
  printf '  "arch": %s,\n'              "$(json_str "$ARCH_V")"
  printf '  "cpu_model": %s,\n'         "$(json_str "$CPU_MODEL")"
  printf '  "cpu_cores": %s,\n'         "$(json_num "$CPU_CORES")"
  printf '  "memory_total_kb": %s,\n'   "$(json_num "$MEM_TOTAL_KB")"
  printf '  "node_count": 1,\n'
  printf '  "gpu_count": %s,\n'         "$(json_num "$GPU_COUNT")"
  printf '  "gpu_model": %s,\n'         "$(json_str "$GPU_MODEL")"
  printf '  "gpus": %s,\n'              "$GPUS_JSON"
  printf '  "driver_version": %s,\n'    "$(json_str "$DRIVER_VERSION")"
  printf '  "cuda_version": %s,\n'      "$(json_str "${CUDA_VERSION_NVCC:-$CUDA_VERSION_SMI}")"
  printf '  "cuda_version_nvcc": %s,\n' "$(json_str "$CUDA_VERSION_NVCC")"
  printf '  "cuda_version_driver": %s,\n' "$(json_str "$CUDA_VERSION_SMI")"
  printf '  "nccl_version": %s,\n'      "$(json_str "$NCCL_VERSION")"
  printf '  "nccl_version_source": %s,\n' "$(json_str "$NCCL_VERSION_SOURCE")"
  printf '  "mpi_version": %s,\n'       "$(json_str "$MPI_VERSION")"
  printf '  "compiler_version": %s,\n'  "$(json_str "$COMPILER_VERSION")"
  printf '  "python_version": %s,\n'    "$(json_str "$PYTHON_VERSION")"
  printf '  "topology": %s,\n'          "$(json_str "$TOPOLOGY")"
  printf '  "topology_summary": %s,\n'  "$(json_str "$TOPOLOGY_SUMMARY")"
  printf '  "nvlink_present": %s,\n'    "$NVLINK_PRESENT"
  printf '  "nvlink_status": %s,\n'     "$(json_str "$NVLINK_STATUS")"
  printf '  "rdma_devices": %s,\n'      "$(json_str "$RDMA_DEVICES")"
  printf '  "git_commit": %s,\n'        "$(json_str "$GIT_COMMIT")"
  printf '  "git_branch": %s,\n'        "$(json_str "$GIT_BRANCH")"
  printf '  "git_dirty": %s,\n'         "$GIT_DIRTY"
  printf '  "env": %s,\n'               "$ENV_JSON"
  printf '  "tool_availability": %s\n'  "$TOOLS_JSON"
  printf '}\n'
} > "$JSON_OUT"

{
  echo "=== NCCL benchmark environment capture ==="
  echo "captured_at_utc : $CAPTURED_AT"
  echo "provider        : $PROVIDER ${PROVIDER_INSTANCE:+($PROVIDER_INSTANCE)}"
  echo "hostname        : ${HOSTNAME_V:-<unavailable>}"
  echo "os / kernel     : ${OS_V:-<unavailable>} / ${KERNEL_V:-<unavailable>} (${ARCH_V:-?})"
  echo "cpu             : ${CPU_MODEL:-<unavailable>} (${CPU_CORES:-?} logical cores)"
  echo "gpu             : ${GPU_MODEL:-<unavailable>} x ${GPU_COUNT:-?}"
  echo "driver          : ${DRIVER_VERSION:-<unavailable>}"
  echo "cuda (nvcc)     : ${CUDA_VERSION_NVCC:-<unavailable>}"
  echo "cuda (driver)   : ${CUDA_VERSION_SMI:-<unavailable>}"
  echo "nccl            : ${NCCL_VERSION:-<unavailable>} [${NCCL_VERSION_SOURCE:-not detected}]"
  echo "mpi             : ${MPI_VERSION:-<not installed>}"
  echo "compiler        : ${COMPILER_VERSION:-<unavailable>}"
  echo "git commit      : ${GIT_COMMIT:-<unavailable>} (${GIT_BRANCH:-?}, dirty=$GIT_DIRTY)"
  echo "nvlink present  : $NVLINK_PRESENT"
  echo "pcie link (gpu0): ${PCIE_SUMMARY:-<unavailable>}"
  echo "topology class  : ${TOPOLOGY_SUMMARY:-<unavailable>}"
  echo "rdma devices    : ${RDMA_DEVICES:-<none detected>}"
  echo
  echo "--- nvidia-smi topo -m ---"
  echo "${TOPOLOGY:-<unavailable>}"
  echo
  echo "--- nvidia-smi nvlink --status ---"
  echo "${NVLINK_STATUS:-<unavailable>}"
  echo
  echo "--- allowlisted environment (credentials redacted) ---"
  echo "$ENV_JSON"
} > "$TEXT_OUT"

echo "environment captured -> $JSON_OUT, $TEXT_OUT"
