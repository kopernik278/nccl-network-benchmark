#!/usr/bin/env bash
#
# collect_net_env.sh — inspect a node's network so an interface can be CHOSEN,
# never guessed.
#
# Phase 3 crosses nodes, so NCCL must be pointed at a specific interface via
# NCCL_SOCKET_IFNAME. Hard-coding "eth0" is exactly the mistake that produces a
# run on the wrong fabric. This script reports what is actually present; the
# operator (or run_nccl_multinode.sh) selects from that evidence.
#
# Usage:
#   scripts/collect_net_env.sh [-o OUTPUT_DIR]
#
# Writes OUTPUT_DIR/netenv.json  (machine-readable)
#        OUTPUT_DIR/netenv.txt   (human-readable)
#
# Degrades gracefully: a missing tool yields null, never a wrong value.
# Security: environment capture uses the same allowlist + redaction rule as
# collect_env.sh, so credentials cannot reach results/ or git.

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
JSON_OUT="$OUT_DIR/netenv.json"; TEXT_OUT="$OUT_DIR/netenv.txt"

have() { command -v "$1" >/dev/null 2>&1; }
try()  { if have "$1"; then "$@" 2>/dev/null || true; fi; }
HAVE_PYTHON3=0; have python3 && HAVE_PYTHON3=1

json_str() {
  local v="${1:-}"
  if [ -z "$v" ]; then printf 'null'; return; fi
  if [ "$HAVE_PYTHON3" = "1" ]; then
    printf '%s' "$v" | python3 -c 'import json,sys; sys.stdout.write(json.dumps(sys.stdin.read()))'
  else
    printf '%s' "$v" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/\\t/g' -e 's/\r//g' \
      | awk 'BEGIN{printf "\""} {if(NR>1) printf "\\n"; printf "%s", $0} END{printf "\""}'
  fi
}
json_num() { case "${1:-}" in ''|*[!0-9]*) printf 'null' ;; *) printf '%s' "$1" ;; esac; }

HOSTNAME_V="$(try hostname)"; [ -z "$HOSTNAME_V" ] && HOSTNAME_V="${HOSTNAME:-}"
FQDN_V="$(try hostname -f)"

# ---------------------------------------------------------------------------
# Interfaces: name, state, MTU, addresses. `ip -j` gives JSON directly when
# available; otherwise fall back to parsing `ip addr`, then ifconfig.
# ---------------------------------------------------------------------------
IFACES_JSON="null"
IFACE_SUMMARY=""
if have ip; then
  _ipj="$(ip -j addr show 2>/dev/null)"
  if [ -n "$_ipj" ] && [ "$HAVE_PYTHON3" = "1" ]; then
    IFACES_JSON="$(printf '%s' "$_ipj" | python3 -c '
import json,sys
try: data=json.load(sys.stdin)
except Exception: print("null"); raise SystemExit
out=[]
for l in data:
    v4=[a["local"] for a in l.get("addr_info",[]) if a.get("family")=="inet"]
    v6=[a["local"] for a in l.get("addr_info",[]) if a.get("family")=="inet6"]
    out.append({"name":l.get("ifname"),"state":l.get("operstate"),
                "mtu":l.get("mtu"),"mac":l.get("address"),
                "flags":l.get("flags",[]),"ipv4":v4,"ipv6":v6,
                "link_type":l.get("link_type")})
print(json.dumps(out))' 2>/dev/null)"
    [ -z "$IFACES_JSON" ] && IFACES_JSON="null"
  fi
  IFACE_SUMMARY="$(ip -o addr show 2>/dev/null)"
  [ -z "$IFACE_SUMMARY" ] && IFACE_SUMMARY="$(ip addr show 2>/dev/null)"
fi
[ -z "$IFACE_SUMMARY" ] && IFACE_SUMMARY="$(try ifconfig -a)"

ROUTES="$(try ip route show)"; [ -z "$ROUTES" ] && ROUTES="$(try route -n)"
DEFAULT_IFACE="$(printf '%s\n' "$ROUTES" | sed -n 's/^default .*dev \([^ ]*\).*/\1/p' | head -1)"

# ---------------------------------------------------------------------------
# Candidate cluster-internal interfaces.
#
# RunPod Instant Clusters place member pods on a VXLAN overlay (the API's
# ClusterNetwork carries cidr / vxlanId / vxlanPort), so the inter-node
# interface is typically an overlay device rather than the default route's.
# We rank candidates but never auto-commit: the runner requires an explicit
# choice.
# ---------------------------------------------------------------------------
# Interface NAMES must come from a link listing, not from parsing address
# output positionally: `ip -o addr` and `ifconfig -a` disagree on column
# layout, and a positional parse silently yields MAC addresses and flag words
# instead of interface names.
_IF_NAMES=""
if have ip; then
  _IF_NAMES="$(ip -o link show 2>/dev/null | awk -F': ' '{print $2}' | sed 's/@.*//')"
fi
if [ -z "$_IF_NAMES" ] && have ifconfig; then
  _IF_NAMES="$(ifconfig -a 2>/dev/null | grep -E '^[a-zA-Z][a-zA-Z0-9._-]*:' | cut -d: -f1)"
fi
CANDIDATES=""
if [ -n "$_IF_NAMES" ]; then
  CANDIDATES="$(printf '%s\n' "$_IF_NAMES" | sort -u \
    | grep -vE '^(lo|lo[0-9]+|docker[0-9]*|veth|br-|virbr|tun[0-9]*|tap[0-9]*|utun[0-9]*|awdl[0-9]*|llw[0-9]*|bridge[0-9]*|gif[0-9]*|stf[0-9]*|anpi[0-9]*|ap[0-9]+)$' \
    | grep -v '^$' | tr '\n' ' ')"
fi
OVERLAY_HINTS="$(printf '%s\n' "$CANDIDATES" | tr ' ' '\n' | grep -iE 'vxlan|overlay|wg|tail|net[0-9]' | tr '\n' ' ')"

# ---------------------------------------------------------------------------
# GPU + NIC + RDMA visibility
# ---------------------------------------------------------------------------
GPU_TOPO="$(try nvidia-smi topo -m)"
GPU_LIST="$(try nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader)"
NIC_PCI="$(try lspci | grep -iE 'ethernet|infiniband|network')"

RDMA_DEVS="$(try ibv_devinfo -l)"; [ -z "$RDMA_DEVS" ] && RDMA_DEVS="$(try ibstat -l)"
RDMA_LINK="$(try ibv_devinfo)"
RDMA_PRESENT="false"; [ -n "$RDMA_DEVS" ] && RDMA_PRESENT="true"
[ -d /sys/class/infiniband ] && [ -n "$(ls -A /sys/class/infiniband 2>/dev/null)" ] && RDMA_PRESENT="true"

# ---------------------------------------------------------------------------
# MPI
# ---------------------------------------------------------------------------
MPI_VERSION=""; MPI_IMPL=""
_mpi_raw=""
if have mpirun; then _mpi_raw="$(mpirun --version 2>&1)"; elif have mpiexec; then _mpi_raw="$(mpiexec --version 2>&1)"; fi
if [ -n "$_mpi_raw" ]; then
  if printf '%s' "$_mpi_raw" | grep -qi 'open mpi'; then MPI_IMPL="OpenMPI"
  elif printf '%s' "$_mpi_raw" | grep -qi 'hydra\|mpich'; then MPI_IMPL="MPICH"
  elif printf '%s' "$_mpi_raw" | grep -qi 'mvapich'; then MPI_IMPL="MVAPICH"
  elif printf '%s' "$_mpi_raw" | grep -qi 'intel'; then MPI_IMPL="IntelMPI"; fi
  MPI_VERSION="$(printf '%s\n' "$_mpi_raw" | grep -iE 'open mpi|mvapich|intel\(r\) mpi' | head -1)"
  [ -z "$MPI_VERSION" ] && MPI_VERSION="$(printf '%s\n' "$_mpi_raw" | sed -n 's/^[[:space:]]*Version:[[:space:]]*\(.*\)$/MPICH \1/p' | head -1)"
  [ -z "$MPI_VERSION" ] && MPI_VERSION="$(printf '%s\n' "$_mpi_raw" | head -1)"
  MPI_VERSION="$(printf '%s' "$MPI_VERSION" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
fi

# ---------------------------------------------------------------------------
# NCCL-relevant environment: same allowlist + redaction rule as collect_env.sh
# ---------------------------------------------------------------------------
ENV_JSON="{"; _first=1
while IFS= read -r kv; do
  k="${kv%%=*}"; v="${kv#*=}"
  case "$k" in NCCL_*|UCX_*|OMPI_*|CUDA_*|NVIDIA_*|RUNPOD_*) ;; *) continue ;; esac
  case "$k" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*PASSWD*|*CREDENTIAL*|*AUTH*|*PRIVATE*) v="<redacted>" ;; esac
  [ $_first -eq 0 ] && ENV_JSON="$ENV_JSON,"; _first=0
  ENV_JSON="$ENV_JSON$(json_str "$k"):$(json_str "$v")"
done <<EOF
$(env 2>/dev/null | sort)
EOF
ENV_JSON="$ENV_JSON}"

TOOLS_JSON="{"; _ft=1
for t in ip ifconfig route nvidia-smi lspci ibv_devinfo ibstat mpirun python3 ss netstat; do
  if have "$t"; then _a=true; else _a=false; fi
  [ $_ft -eq 0 ] && TOOLS_JSON="$TOOLS_JSON,"; _ft=0
  TOOLS_JSON="$TOOLS_JSON$(json_str "$t"):$_a"
done
TOOLS_JSON="$TOOLS_JSON}"

{
  printf '{\n'
  printf '  "schema_version": 1,\n'
  printf '  "captured_at_utc": %s,\n' "$(json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
  printf '  "hostname": %s,\n'        "$(json_str "$HOSTNAME_V")"
  printf '  "fqdn": %s,\n'            "$(json_str "$FQDN_V")"
  printf '  "interfaces": %s,\n'      "$IFACES_JSON"
  printf '  "interfaces_raw": %s,\n'  "$(json_str "$IFACE_SUMMARY")"
  printf '  "routes": %s,\n'          "$(json_str "$ROUTES")"
  printf '  "default_route_iface": %s,\n' "$(json_str "$DEFAULT_IFACE")"
  printf '  "candidate_interfaces": %s,\n' "$(json_str "$CANDIDATES")"
  printf '  "overlay_interface_hints": %s,\n' "$(json_str "$OVERLAY_HINTS")"
  printf '  "gpu_list": %s,\n'        "$(json_str "$GPU_LIST")"
  printf '  "gpu_topology": %s,\n'    "$(json_str "$GPU_TOPO")"
  printf '  "nic_pci": %s,\n'         "$(json_str "$NIC_PCI")"
  printf '  "rdma_present": %s,\n'    "$RDMA_PRESENT"
  printf '  "rdma_devices": %s,\n'    "$(json_str "$RDMA_DEVS")"
  printf '  "rdma_devinfo": %s,\n'    "$(json_str "$RDMA_LINK")"
  printf '  "mpi_implementation": %s,\n' "$(json_str "$MPI_IMPL")"
  printf '  "mpi_version": %s,\n'     "$(json_str "$MPI_VERSION")"
  printf '  "env": %s,\n'             "$ENV_JSON"
  printf '  "tool_availability": %s\n' "$TOOLS_JSON"
  printf '}\n'
} > "$JSON_OUT"

{
  echo "=== node network inspection ==="
  echo "hostname            : ${HOSTNAME_V:-<unavailable>}"
  echo "default route iface : ${DEFAULT_IFACE:-<unavailable>}"
  echo "candidate ifaces    : ${CANDIDATES:-<none>}"
  echo "overlay hints       : ${OVERLAY_HINTS:-<none>}"
  echo "rdma present        : $RDMA_PRESENT  (${RDMA_DEVS:-none})"
  echo "mpi                 : ${MPI_IMPL:-<none>} ${MPI_VERSION:-}"
  echo
  echo "--- interfaces ---"; echo "${IFACE_SUMMARY:-<unavailable>}"
  echo; echo "--- routes ---";  echo "${ROUTES:-<unavailable>}"
  echo; echo "--- gpu topology ---"; echo "${GPU_TOPO:-<unavailable>}"
  echo; echo "--- nic (pci) ---"; echo "${NIC_PCI:-<unavailable>}"
  echo; echo "--- nccl env (redacted) ---"; echo "$ENV_JSON"
  echo
  echo "NOTE: no interface is selected here. run_nccl_multinode.sh requires an"
  echo "explicit -I <iface>; it will not guess and will not fall back."
} > "$TEXT_OUT"

echo "network environment captured -> $JSON_OUT, $TEXT_OUT"
