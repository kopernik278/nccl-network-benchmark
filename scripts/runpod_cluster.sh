#!/usr/bin/env bash
#
# runpod_cluster.sh — minimal wrapper over the RunPod Cluster REST API.
#
# The Cluster endpoints (/v2/clusters) are NOT exposed by the RunPod MCP server
# or by runpodctl, so Phase 3 must call the REST API directly.
#
# Usage:
#   scripts/runpod_cluster.sh list
#   scripts/runpod_cluster.sh create -n NAME -g GPU_TYPE_ID [-p PODS] [-c GPUS_PER_POD] [-d DC]
#   scripts/runpod_cluster.sh get   -i CLUSTER_ID
#   scripts/runpod_cluster.sh pods  -i CLUSTER_ID
#   scripts/runpod_cluster.sh delete -i CLUSTER_ID
#   scripts/runpod_cluster.sh price -g GPU_TYPE_ID -p PODS -c GPUS_PER_POD
#
# CREDENTIAL HANDLING
#
# The API key is read from the macOS Keychain, never from a repository file,
# never from a shell rc file, and never from an argument (which would land in
# shell history and in `ps`). It is passed to curl through a header file with
# 0600 permissions rather than on the command line.
#
# Store it once, interactively, with:
#   security add-generic-password -a "$USER" -s runpod-project3-nccl-agent -w
#
# The key is never echoed by this script. Diagnostics print only whether a
# credential was found.
#
# COST GUARD
#
# `create` refuses configurations whose TOTAL hourly cost exceeds the project
# threshold (default $3.00/hour) unless -f is passed. The threshold applies to
# the whole cluster, not the per-GPU price.

set -uo pipefail

KEYCHAIN_SERVICE="${RUNPOD_KEYCHAIN_SERVICE:-runpod-project3-nccl-agent}"
API="https://api.runpod.io/v2"
COST_LIMIT="${RUNPOD_COST_LIMIT:-3.00}"

die() { echo "ERROR: $*" >&2; exit 1; }

ACTION="${1:-}"; shift 2>/dev/null || true
[ -n "$ACTION" ] || { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

NAME=""; GPU_TYPE=""; PODS=2; GPUS_PER_POD=1; DC=""; CLUSTER_ID=""; FORCE=0; CTYPE="APPLICATION"
IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"; DISK=20
while getopts ":n:g:p:c:d:i:t:m:k:fh" opt; do
  case "$opt" in
    n) NAME="$OPTARG" ;;      g) GPU_TYPE="$OPTARG" ;;
    p) PODS="$OPTARG" ;;      c) GPUS_PER_POD="$OPTARG" ;;
    d) DC="$OPTARG" ;;        i) CLUSTER_ID="$OPTARG" ;;
    t) CTYPE="$OPTARG" ;;     m) IMAGE="$OPTARG" ;;
    k) DISK="$OPTARG" ;;      f) FORCE=1 ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) die "unknown option: -$OPTARG" ;;
  esac
done

# --- credential -------------------------------------------------------------
get_key() {
  command -v security >/dev/null 2>&1 || die "macOS 'security' not available; this wrapper expects the Keychain"
  local k rc
  k="$(security find-generic-password -a "$USER" -s "$KEYCHAIN_SERVICE" -w 2>/dev/null)"; rc=$?
  # Distinguish "no item" from "item exists but is empty". The second happens
  # when the interactive -w prompt runs somewhere that cannot supply stdin, and
  # it silently stores a zero-length secret that would otherwise look present.
  if [ $rc -ne 0 ]; then
    die "no RunPod API key in the Keychain under service '$KEYCHAIN_SERVICE'.
       Store it from a REAL interactive terminal (the value is not echoed and
       does not enter shell history):
         security add-generic-password -U -a \"\$USER\" -s $KEYCHAIN_SERVICE -w"
  fi
  if [ -z "$k" ]; then
    die "the Keychain entry '$KEYCHAIN_SERVICE' exists but is EMPTY.
       The interactive prompt captured nothing — this happens when it runs
       without a real tty. Re-store it from your own terminal:
         security delete-generic-password -a \"\$USER\" -s $KEYCHAIN_SERVICE
         security add-generic-password -U -a \"\$USER\" -s $KEYCHAIN_SERVICE -w"
  fi
  printf '%s' "$k"
}

# Build a 0600 header file so the key never appears in argv or in `ps` output.
HEADER_FILE=""
cleanup() { [ -n "$HEADER_FILE" ] && rm -f "$HEADER_FILE"; }
trap cleanup EXIT INT TERM

api() {  # api <METHOD> <PATH> [JSON_BODY]
  local method="$1" path="$2" body="${3:-}"
  # Resolve the credential BEFORE building the request. get_key's die() runs in
  # a subshell under command substitution, so its exit cannot stop the caller —
  # without this guard the script would go on to make an unauthenticated call
  # and report a confusing 401 instead of the real problem.
  local key
  key="$(get_key)" || exit 1
  [ -n "$key" ] || die "empty credential in the Keychain"
  HEADER_FILE="$(mktemp)"; chmod 600 "$HEADER_FILE"
  printf 'Authorization: Bearer %s\n' "$key" > "$HEADER_FILE"
  unset key
  local args=(-sS -X "$method" -H @"$HEADER_FILE" -H "Content-Type: application/json"
              -w '\n__HTTP_STATUS__:%{http_code}\n' --max-time 120)
  [ -n "$body" ] && args+=(-d "$body")
  curl "${args[@]}" "$API$path"
  local rc=$?
  rm -f "$HEADER_FILE"; HEADER_FILE=""
  return $rc
}

show() {  # print body, surface non-2xx clearly
  local out="$1"
  local status; status="$(printf '%s' "$out" | sed -n 's/^__HTTP_STATUS__:\([0-9]*\)$/\1/p' | tail -1)"
  printf '%s\n' "$out" | grep -v '^__HTTP_STATUS__:'
  case "$status" in
    2*) return 0 ;;
    401) echo "HTTP 401 — the stored API key was rejected." >&2; return 1 ;;
    403) echo "HTTP 403 — the key is valid but lacks permission for this action. RunPod
       'Read Only' keys cannot create clusters; a key with write access is required." >&2; return 1 ;;
    *)  echo "HTTP ${status:-?} — request failed." >&2; return 1 ;;
  esac
}

# --- per-GPU price lookup (unauthenticated catalog is not available; use MCP
#     output or pass the price in). Kept explicit so cost is never guessed. ---
price_check() {
  local per_gpu="${RUNPOD_PRICE_PER_GPU:-}"
  [ -n "$per_gpu" ] || die "set RUNPOD_PRICE_PER_GPU to the CURRENT secure-cloud
       per-GPU hourly price for $GPU_TYPE (check the live catalog first; never
       reuse a stale figure)."
  local total
  total="$(python3 -c "print(f'{float('$per_gpu') * $PODS * $GPUS_PER_POD:.2f}')")"
  echo "cost: $PODS pods x $GPUS_PER_POD GPU x \$$per_gpu/GPU/hr = \$$total/hour"
  local over
  over="$(python3 -c "print(1 if float('$total') > float('$COST_LIMIT') else 0)")"
  if [ "$over" -eq 1 ] && [ "$FORCE" -ne 1 ]; then
    die "TOTAL \$$total/hour exceeds the \$$COST_LIMIT/hour project threshold.
       Per project policy this requires USER APPROVAL before provisioning.
       Re-run with -f only after the user has approved."
  fi
  echo "$total"
}

# Every action below talks to the API. Resolve the credential HERE, in the main
# shell, so a missing or empty key exits cleanly with one message — inside
# api() the failure happens in a command-substitution subshell and cannot stop
# the script, which produced a confusing "HTTP ? — request failed." after the
# real error.
get_key >/dev/null || exit 1

case "$ACTION" in
  list)   show "$(api GET /clusters)" ;;
  sshkeys) show "$(api GET /account/ssh-keys)" ;;
  get)    [ -n "$CLUSTER_ID" ] || die "get needs -i CLUSTER_ID";  show "$(api GET "/clusters/$CLUSTER_ID")" ;;
  pods)   [ -n "$CLUSTER_ID" ] || die "pods needs -i CLUSTER_ID"; show "$(api GET "/clusters/$CLUSTER_ID/pods")" ;;
  delete) [ -n "$CLUSTER_ID" ] || die "delete needs -i CLUSTER_ID"
          show "$(api DELETE "/clusters/$CLUSTER_ID")"
          echo "verify with: $0 get -i $CLUSTER_ID   (expect 404) and $0 list" ;;
  price)  [ -n "$GPU_TYPE" ] || die "price needs -g GPU_TYPE_ID"; price_check >/dev/null ;;
  create)
    [ -n "$NAME" ]     || die "create needs -n NAME"
    [ -n "$GPU_TYPE" ] || die "create needs -g GPU_TYPE_ID"
    [ "$PODS" -ge 2 ]  || die "podCount minimum is 2 (API schema constraint)"
    price_check >/dev/null
    BODY="$(python3 - "$NAME" "$CTYPE" "$GPU_TYPE" "$PODS" "$GPUS_PER_POD" "$DC" "$IMAGE" "$DISK" <<'PY'
import json,sys
name,ctype,gpu,pods,gpp,dc,image,disk = sys.argv[1:9]
body={"name":name,"type":ctype,
      "compute":{"gpuTypeId":gpu,"podCount":int(pods),"gpuCountPerPod":int(gpp)},
      "image":image,"disk":int(disk),"ports":["22/tcp"],
      "startSsh":True}
if dc: body["dataCenterIds"]=[dc]
print(json.dumps(body))
PY
)"
    echo "request body: $BODY"
    show "$(api POST /clusters "$BODY")" ;;
  *) die "unknown action: $ACTION" ;;
esac
