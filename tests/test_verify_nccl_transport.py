#!/usr/bin/env python3
"""Unit tests for scripts/verify_nccl_transport.py.

Zero-cost, no cluster: all input is hand-written SYNTHETIC NCCL log text in
tests/fixtures/. These tests exist so the Phase 3 transport gate is known to
work *before* paid multi-node time is spent relying on it.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import verify_nccl_transport as V  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


class TestAnalyse(unittest.TestCase):
    def test_socket_log_facts(self):
        f = V.analyse(load("synthetic_nccl_multinode_socket.txt"))
        self.assertEqual(f["via_net_counts"], {"Socket": 4})
        self.assertEqual(f["socket_interfaces"], ["vxlan0"])
        self.assertEqual(f["hosts"], ["node-a", "node-b"])
        self.assertEqual(f["host_count"], 2)
        self.assertEqual(f["rank_to_host"], {"0": "node-a", "1": "node-b"})

    def test_ib_log_facts(self):
        f = V.analyse(load("synthetic_nccl_multinode_ib.txt"))
        self.assertEqual(f["via_net_counts"], {"IB": 2})
        self.assertEqual(f["host_count"], 2)

    def test_single_node_log_has_no_net_transport(self):
        f = V.analyse(load("synthetic_nccl_singlenode.txt"))
        self.assertEqual(f["via_net_counts"], {})
        self.assertEqual(f["host_count"], 1)

    def test_empty_input_does_not_crash(self):
        f = V.analyse("")
        self.assertEqual(f["host_count"], 0)
        self.assertEqual(f["via_net_counts"], {})


class TestVerify(unittest.TestCase):
    def _v(self, fixture, **kw):
        opts = dict(expect_transport="socket", expect_iface=None, min_hosts=2)
        opts.update(kw)
        return V.verify(V.analyse(load(fixture)), **opts)

    def test_socket_run_passes(self):
        ok, passed, failed = self._v("synthetic_nccl_multinode_socket.txt")
        self.assertTrue(ok, failed)
        self.assertFalse(failed)

    def test_socket_run_with_matching_interface_passes(self):
        ok, _, failed = self._v("synthetic_nccl_multinode_socket.txt",
                                expect_iface="vxlan0")
        self.assertTrue(ok, failed)

    def test_wrong_interface_fails(self):
        """Guards against NCCL silently choosing an interface we did not pick."""
        ok, _, failed = self._v("synthetic_nccl_multinode_socket.txt",
                                expect_iface="eth0")
        self.assertFalse(ok)
        self.assertTrue(any("expected interface eth0" in f for f in failed))

    def test_ib_run_rejected_when_socket_expected(self):
        """The central Phase 3 guarantee: RDMA cannot pass as a TCP baseline."""
        ok, _, failed = self._v("synthetic_nccl_multinode_ib.txt")
        self.assertFalse(ok)
        self.assertTrue(any("no NET/socket evidence" in f for f in failed))

    def test_single_node_run_rejected(self):
        ok, _, failed = self._v("synthetic_nccl_singlenode.txt")
        self.assertFalse(ok)
        self.assertTrue(any("no NET/socket evidence" in f for f in failed))
        self.assertTrue(any("expected >= 2 hosts" in f for f in failed))

    def test_ib_run_passes_when_ib_expected(self):
        ok, _, failed = self._v("synthetic_nccl_multinode_ib.txt",
                                expect_transport="ib")
        self.assertTrue(ok, failed)

    def test_mixed_transport_is_rejected(self):
        """A run that used both must not pass as a clean socket baseline."""
        mixed = (load("synthetic_nccl_multinode_socket.txt")
                 + "\nnode-a:1101:1120 [0] NCCL INFO Channel 02/0 : 0[0] -> 1[0] [send] via NET/IB/0\n")
        ok, _, failed = V.verify(V.analyse(mixed), "socket", None, 2)
        self.assertFalse(ok)
        self.assertTrue(any("unexpected transport" in f for f in failed))

    def test_unknown_transport_name_is_rejected(self):
        ok, _, failed = V.verify(V.analyse(load("synthetic_nccl_multinode_socket.txt")),
                                 "rdma-over-carrier-pigeon", None, 2)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
