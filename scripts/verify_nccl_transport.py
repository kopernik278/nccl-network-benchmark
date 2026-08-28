#!/usr/bin/env python3
"""Prove, from NCCL runtime evidence, which transport a run actually used.

Phase 3 compares NCCL Socket/TCP against RDMA later. That comparison is
worthless if we cannot show the socket run really used sockets. Setting
NCCL_IB_DISABLE=1 states an *intent*; this script checks the *outcome*.

    python3 scripts/verify_nccl_transport.py --log run.log \
        --expect-transport socket --expect-iface eth0 --min-hosts 2

Exit status is 0 only when every requested check passes. A run whose transport
cannot be verified must not be accepted as a measurement.

What it reads (NCCL_DEBUG=INFO output):

    NCCL INFO NET/Socket : Using [0]eth0:10.0.0.2<0>
    NCCL INFO Channel 00/0 : 0[0] -> 1[0] [receive] via NET/Socket/0
    NCCL INFO Channel 00/0 : 1[0] -> 0[0] [send] via NET/IB/0        <- would FAIL

Intra-node hops legitimately use P2P or SHM; only the inter-node hops go over
NET/*. The check therefore asserts on the NET/* evidence and independently
confirms that more than one host took part.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# "via NET/Socket/0", "via NET/IB/0", "[send] via NET/Socket/1"
RE_VIA_NET = re.compile(r"via\s+NET/(\w+)")
# "NET/Socket : Using [0]eth0:10.0.0.2<0>"  /  "NET/IB : Using [0]mlx5_0:1/RoCE"
RE_NET_USING = re.compile(r"NET/(\w+)\s*:\s*Using\s+(.+)")
RE_IFACE = re.compile(r"\[\d+\]([A-Za-z0-9._-]+):")
# "# Rank  0 Group  0 Pid 123 on hostA device  0 [0x07] NVIDIA L4"
RE_RANK = re.compile(r"Rank\s+(\d+)\s+.*?\bon\s+(\S+)\s+device\s+(\d+)")
RE_NCCL_WARN = re.compile(r"NCCL WARN (.+)")

TRANSPORT_ALIASES = {
    "socket": {"Socket"},
    "ib": {"IB", "IBext", "IBext_v8", "IBext_v9"},
}


def analyse(text: str) -> dict:
    via = {}
    using = {}
    ifaces = set()
    ranks = {}
    warns = []

    for line in text.splitlines():
        for m in RE_VIA_NET.finditer(line):
            via[m.group(1)] = via.get(m.group(1), 0) + 1
        m = RE_NET_USING.search(line)
        if m:
            using.setdefault(m.group(1), []).append(m.group(2).strip())
            im = RE_IFACE.search(m.group(2))
            if im:
                ifaces.add(im.group(1))
        m = RE_RANK.search(line)
        if m:
            ranks[int(m.group(1))] = m.group(2)
        m = RE_NCCL_WARN.search(line)
        if m:
            warns.append(m.group(1).strip())

    return {
        "via_net_counts": via,
        "net_using": using,
        "socket_interfaces": sorted(ifaces),
        "rank_to_host": {str(k): v for k, v in sorted(ranks.items())},
        "hosts": sorted(set(ranks.values())),
        "host_count": len(set(ranks.values())),
        "nccl_warnings": warns,
    }


def verify(facts: dict, expect_transport: str, expect_iface: str | None,
           min_hosts: int) -> tuple[bool, list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []

    wanted = TRANSPORT_ALIASES.get(expect_transport.lower())
    if wanted is None:
        failed.append(f"unknown --expect-transport {expect_transport!r}")
        return False, passed, failed

    seen = set(facts["via_net_counts"]) | set(facts["net_using"])

    # 1. the wanted transport must actually appear
    if seen & wanted:
        n = sum(facts["via_net_counts"].get(w, 0) for w in wanted)
        passed.append(f"NET/{'|'.join(sorted(seen & wanted))} present ({n} channel lines)")
    else:
        failed.append(
            f"no NET/{expect_transport} evidence found "
            f"(saw: {sorted(seen) or 'no NET/* lines at all'})")

    # 2. no other NET transport may appear — this is what catches a silent
    #    fallback to IB/RDMA despite NCCL_IB_DISABLE=1
    others = seen - wanted
    if others:
        failed.append(f"unexpected transport(s) also used: {sorted(others)}")
    else:
        passed.append("no competing NET/* transport present")

    # 3. the run must genuinely span nodes
    if facts["host_count"] >= min_hosts:
        passed.append(f"{facts['host_count']} distinct hosts: {facts['hosts']}")
    else:
        failed.append(
            f"expected >= {min_hosts} hosts, found {facts['host_count']} "
            f"({facts['hosts'] or 'none parsed'})")

    # 4. the interface must be the one we chose, not one NCCL picked for us
    if expect_iface:
        if expect_iface in facts["socket_interfaces"]:
            passed.append(f"interface {expect_iface} confirmed in NET/* banner")
        elif not facts["socket_interfaces"]:
            failed.append(
                f"expected interface {expect_iface} but no 'NET/* : Using [n]<iface>' "
                "banner was found (run with NCCL_DEBUG=INFO)")
        else:
            failed.append(
                f"expected interface {expect_iface}, NCCL reported "
                f"{facts['socket_interfaces']}")

    return not failed, passed, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, help="NCCL_DEBUG=INFO log (default: stdin)")
    ap.add_argument("--expect-transport", default="socket", choices=["socket", "ib"])
    ap.add_argument("--expect-iface", default=None)
    ap.add_argument("--min-hosts", type=int, default=2)
    ap.add_argument("--json-out", type=Path, help="write the facts + verdict here")
    args = ap.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace") if args.log else sys.stdin.read()
    facts = analyse(text)
    ok, passed, failed = verify(facts, args.expect_transport, args.expect_iface, args.min_hosts)

    print(f"transport verification: {'PASS' if ok else 'FAIL'}")
    for p in passed:
        print(f"  [ok]   {p}")
    for f in failed:
        print(f"  [FAIL] {f}")
    if facts["nccl_warnings"]:
        print(f"  note: {len(facts['nccl_warnings'])} NCCL WARN line(s):")
        for w in facts["nccl_warnings"][:5]:
            print(f"         {w}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"verdict": "pass" if ok else "fail",
             "expect_transport": args.expect_transport,
             "expect_iface": args.expect_iface,
             "checks_passed": passed, "checks_failed": failed, **facts},
            indent=2, sort_keys=True), encoding="utf-8")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
