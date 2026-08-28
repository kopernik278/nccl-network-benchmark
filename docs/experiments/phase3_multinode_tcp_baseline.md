# Phase 3 — Multi-Node NCCL Socket/TCP Baseline

Status: **designed (Phase 3A complete); Phase 3B deferred on cost — see §9.2b**
Design date: 2026-08-28
Repo commit at design time: `47e6f34`
Previous: [Phase 1B](p1b-first-2gpu-nccl-baseline.md) · [Phase 2](p2-multigpu-scaling.md)

> This document contains **no measured results**. Every number is a
> configuration parameter, a catalog price, or an explicitly labelled
> expectation. **No RoCE or InfiniBand measurement is planned in Phase 3.**

---

## 1. Motivation

Phases 1 and 2 measured collectives inside a single node. Phase 3 crosses the
node boundary for the first time and establishes the **TCP/socket baseline**
against which RDMA, RoCE and InfiniBand will later be compared.

That future comparison is the whole point, and it imposes an unusual
requirement on this phase: it is not enough to run NCCL across nodes with
RDMA disabled and assume sockets were used. **Phase 3 must prove from runtime
evidence that the collective actually travelled over NET/Socket.** A number
measured on an unverified transport is worse than no number, because it would
silently corrupt the RDMA speed-up figure later.

Phase 3 therefore treats transport verification as a gate on par with
correctness.

### What Phase 2 leaves behind

Phase 2 found that stepping 2→4 GPUs on one node retained only 24–29% of bus
bandwidth. That result is **confounded**: the 4-GPU configuration crossed two
PCIe/P2P domains, so rank count and topology moved together.

**That ~75% loss must not be generalised to arbitrary 4-GPU systems.** It
describes one two-P2P-island PCIe host. Phase 3 does not resolve that
confound and does not attempt to.

---

## 2. Hypotheses

Stated so the data can refute them.

| ID | Hypothesis | Test |
|----|------------|------|
| H1 | NCCL initialises across 2 nodes under MPI and all ranks complete with zero validation errors. | `#wrong == 0`, exit 0 |
| H2 | With `NCCL_IB_DISABLE=1` and an explicit `NCCL_SOCKET_IFNAME`, NCCL reports `via NET/Socket` for every inter-node channel and no `NET/IB` anywhere. | transport verifier |
| H3 | The inter-node latency floor is **substantially higher** than the ~7–21 µs intra-node floors of Phases 1–2, because a TCP round trip through the kernel network stack replaces a PCIe P2P write. | latency at the smallest sizes |
| H4 | Large-message bus bandwidth is bounded by the node's network link and will be **well below** the ~10–20 GB/s intra-node PCIe figures. | bandwidth plateau |
| H5 | The bus/algorithmic bandwidth ratio holds at the new rank count, as it did at n=2 and n=4. | parser self-check |

H3 and H4 are directional, not quantitative: no specific µs or GB/s figure is
predicted, because the cluster's network type is not yet known (§9).

---

## 3. Minimum Infrastructure

| Requirement | Value |
|-------------|-------|
| Nodes | **2** (the minimum that makes "multi-node" meaningful) |
| GPUs per node | **1** |
| Total ranks | **2** |
| Interconnect between nodes | whatever the cluster provides — TCP over it |
| RDMA / RoCE / InfiniBand | **not required, and deliberately disabled** |
| Shared filesystem | not required |
| Persistent storage | not required |

One GPU per node is deliberate. Phase 3's question is *what does crossing a
node boundary cost*, and 2 ranks on 2 nodes isolates exactly that: every hop in
the ring is an inter-node hop. Adding intra-node ranks would reintroduce the
mixed-topology confound that made Phase 2 hard to interpret.

Multi-GPU-per-node configurations are a Phase 3 follow-up, not the baseline.

---

## 4. Node and Rank Topology

```text
  node-a                        node-b
  ┌──────────────┐              ┌──────────────┐
  │ rank 0       │              │ rank 1       │
  │ GPU 0        │◄────────────►│ GPU 0        │
  └──────────────┘   TCP over   └──────────────┘
                   <iface chosen
                    at runtime>
```

- One MPI rank per GPU; `-g 1` passed to nccl-tests so MPI, not the binary,
  supplies the ranks.
- Rank→host mapping is **recorded, not assumed**: it is parsed from the
  runtime output and stored in the manifest and in every result row.
- Ring hops: `0 → 1` and `1 → 0`, both inter-node.

---

## 5. Software Requirements

| Component | Requirement | Status on the RunPod pytorch image |
|-----------|-------------|------------------------------------|
| CUDA toolkit | `nvcc` to build nccl-tests | present (12.8) |
| NCCL | must support the GPU's architecture | present, but see below |
| **MPI** | **required** — `mpirun` + `mpicc` | **ABSENT** — installed by `setup_multinode_env.sh -i` |
| nccl-tests | built with `MPI=1` | `setup_nccl_tests.sh -m` |
| passwordless SSH between nodes | required by mpirun | **not present by default** |
| Python 3 | transport verifier | present |

Two of these are observed facts from earlier phases rather than assumptions:

- **MPI is absent.** Phase 1B environment capture recorded
  `mpi: <not installed>` on `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- **NCCL may be too old for the GPU.** Phase 2 hit
  `NCCL 2.25.1` failing at kernel launch on sm_120. The same rebuild-with-newer-NCCL
  path may be needed again depending on the GPU chosen.

---

## 6. MPI Design

Deliberately minimal: **nccl-tests' normal multi-process workflow**, no custom
distributed runtime, no orchestration layer.

```bash
mpirun --hostfile <hostfile> -np <total_ranks> \
       <env args> \
       $NCCL_TESTS_DIR/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 -w 20 -n 50 -c 1 -d float
```

Implemented in `scripts/run_nccl_multinode.sh`. Design points:

**Implementation detection.** OpenMPI and MPICH disagree on both hostfile
format (`host slots=N` vs `host:N`) and environment propagation (`-x VAR` vs
`-genv VAR val`). The runner detects the implementation and emits the correct
form rather than assuming OpenMPI.

**Environment reaches every rank.** Transport variables are passed through the
launcher explicitly. A rank that missed `NCCL_IB_DISABLE` could negotiate a
different transport than its peers — the verifier would catch it, but the
launcher is built not to cause it.

**A non-MPI binary is refused.** nccl-tests built without `MPI=1` still runs
under `mpirun`; it just becomes N independent single-rank jobs, each of which
"succeeds". That failure mode looks like a successful multi-node launch and
produces meaningless single-rank numbers. Both `setup_nccl_tests.sh -m` and the
runner check that the binary links `libmpi` and refuse otherwise.

**Recorded per run:** MPI implementation, MPI version, the hostfile, the exact
`mpirun` command line, rank→host mapping, per-run exit codes, and per-run
stdout/stderr.

**Per-node failures are preserved.** Each node writes its own environment
capture under `nodes/<hostname>/`, so a failure localised to one host stays
diagnosable.

---

## 7. NCCL Socket Configuration

```bash
NCCL_IB_DISABLE=1                 # refuse IB/RoCE verbs transport
NCCL_SOCKET_IFNAME=<iface>        # explicitly chosen; never guessed
NCCL_DEBUG=INFO                   # diagnostic runs only
NCCL_DEBUG=VERSION                # timed runs
```

`NCCL_IB_DISABLE=1` expresses intent. It does **not** constitute evidence, and
Phase 3 does not treat it as such — see §12.

`NCCL_DEBUG=INFO` is used **only** for the diagnostic run. Timed measurements
run at `VERSION` so that debug logging cannot perturb the timings it is meant
to explain. This mirrors the Phase 1/2 practice of keeping the INFO probe out
of the measured set.

---

## 8. Network-Interface Discovery

**No interface name is hard-coded anywhere.** Not `eth0`, not `ens1`, not the
default route's device.

`scripts/collect_net_env.sh` runs on each node and reports hostname and FQDN,
every interface with state / MTU / MAC / addresses, routing table and default
route device, ranked candidate interfaces (loopback, docker, veth, bridges and
tunnels filtered out), overlay-interface hints, GPU topology, NIC PCI devices,
RDMA devices if any, MPI implementation and version, and the allowlisted
NCCL-relevant environment.

It **selects nothing**. Selection is a human/runner decision made *after*
looking at real output, and `run_nccl_multinode.sh` requires `-I <iface>`
explicitly:

```
ERROR: -I is required. Run scripts/collect_net_env.sh on the
       nodes first and choose the cluster-internal interface from its output.
       This script will not guess an interface and will not fall back.
```

**Why this matters here specifically.** RunPod's cluster API describes the
inter-node network as a **VXLAN overlay** (`ClusterNetwork` carries `cidr`,
`vxlanId`, `vxlanPort` — see §9). On such a node the cluster-internal device is
an overlay interface, and the default route almost certainly is not it. A
hard-coded `eth0` would plausibly still *work* while measuring the wrong path.

Interface-name extraction is taken from a link listing (`ip -o link`), never
from positionally parsing address output — the two disagree on column layout,
and a positional parse silently yields MAC addresses and flag words instead of
interface names.

---

## 9. RunPod Cluster Feasibility (read-only discovery, 2026-08-28)

Three tiers of confidence, kept separate on purpose.

### 9.1 Verified API evidence

From the live REST v2 contract at `https://api.runpod.io/v2/openapi.json`,
fetched 2026-08-28:

| Endpoint | Methods |
|----------|---------|
| `/v2/clusters` | GET, **POST** |
| `/v2/clusters/{id}` | GET, PATCH, DELETE |
| `/v2/clusters/{id}/pods` | GET |
| `/v2/billing/clusters` | GET |

`ClusterCompute` (required fields):

| Field | Constraint |
|-------|-----------|
| `gpuTypeId` | string |
| `gpuCountPerPod` | integer, **minimum 1** |
| `podCount` | integer, **minimum 2**, maximum 250 |

`CreateClusterRequest`: `name`, `type`, `compute` required; optional
`dataCenterIds`, `mounts`, `startSsh`, `startJupyter`.
`type` ∈ `APPLICATION | TRAINING | SLURM | RAY`.

`ClusterNetwork`: `cidr` (required), `vxlanId`, `vxlanPort` — **the cluster
network is a VXLAN overlay**.

`ClusterPrimary`: `podId`, `status`, `sshEndpoint` (`host:port`) — a primary
node is designated and is the natural place to launch `mpirun` from.

A cluster is always placed **within a single data center**.

`startSsh` "injects a PUBLIC_KEY environment variable carrying your account's
registered SSH public key".

**So the minimum viable cluster is `podCount: 2, gpuCountPerPod: 1` — 2 GPUs
total.** That is permitted by the schema.

### 9.2 Catalog pricing (live, 2026-08-28, `product: CLUSTER`)

Cluster-context availability and per-GPU secure-cloud price:

| GPU | Availability | $/GPU/hr | 2 nodes × 1 GPU |
|-----|--------------|---------:|----------------:|
| **RTX A5000** | **HIGH** | **0.27** | **$0.54/hr** |
| A40 | HIGH | 0.44 | $0.88/hr |
| RTX PRO 4500 | HIGH | 0.72 | $1.44/hr |
| RTX 4090 | HIGH | 0.74 | $1.48/hr |
| RTX 5090 | HIGH | 0.99 | $1.98/hr |
| H200 SXM | HIGH | 4.59 | $9.18/hr — **over threshold** |

**Planned minimum viable cluster: 2 × 1 × RTX A5000 = $0.54/hour.**
Even 2 nodes × 2 GPUs would be $1.08/hour. Both are **under the $3.00/hour
threshold, so Phase 3B may proceed autonomously** — provided the price is
re-checked at provisioning time rather than taken from this table.

### 9.2b VERIFIED 2026-08-28: the minimum cluster is far larger and costlier
### than §9.1 permits — Phase 3B is deferred

The §9.4 assumption *"that a 2 × 1 cluster is actually schedulable"* was
**tested against the live API and is false.** This section records the
outcome, because it invalidates the plan in §9.2 and §17.

**Attempted and refused.** With a working scoped API key, `POST /v2/clusters`
was called for every configuration below. All returned
`400 {"detail":"Insufficient resources"}` — **no resources were provisioned and
nothing was billed**:

| GPU | Shape | Total | Data centre |
|-----|-------|------:|-------------|
| RTX A5000 | 2 × 1 | $0.54/hr | scheduler's choice |
| RTX A5000 | 2 × 1 | $0.54/hr | CA-MTL-1, EU-SE-1, US-IL-1 (each explicitly) |
| RTX A5000 | 2 × 2 | $1.08/hr | CA-MTL-1 |
| RTX A5000 | 2 × 4 | $2.16/hr | CA-MTL-1 |
| A40 | 2 × 1 | $0.88/hr | scheduler's choice |
| A40 | 2 × 2 | $1.76/hr | CA-MTL-1 |
| RTX 4090 | 2 × 1 | $1.48/hr | scheduler's choice |
| L40S | 2 × 1 | $1.98/hr | scheduler's choice |

**Cause, from RunPod's Instant Clusters documentation.** Instant Clusters are
offered on **only four GPU types — B200, H200, H100, A100** — in a range of
"2-8 nodes (16-64 GPUs)". Sixteen GPUs across two nodes means **8 GPUs per
node**, and none of the workstation/consumer types tried above are cluster
GPUs at all.

**The catalog's CLUSTER availability figure is not cluster schedulability.**
`list-gpu-types --product CLUSTER` reported RTX A5000 as HIGH, and
`get-gpu-type` broke that down per data centre (CA-MTL-1 MEDIUM, EU-SE-1 LOW,
US-IL-1 LOW) — yet every create was refused. The field reflects GPU stock in a
cluster context, not whether a cluster of the requested shape can be placed.
**Do not use it as a feasibility signal.**

**Resulting minimum cost**, at live prices on 2026-08-28:

| GPU | Cluster availability | $/GPU/hr | 2 nodes × 8 = 16 GPUs |
|-----|----------------------|---------:|----------------------:|
| **A100 SXM 80GB** | LOW (US-KS-2, US-MD-1, US-WA-1) | 1.59 | **$25.44/hr** |
| A100 PCIe | **NONE** | 1.39 | unavailable |
| H100 SXM | MEDIUM | 3.29 | $52.64/hr |
| H200 SXM | HIGH | 4.59 | $73.44/hr |
| B200 | LOW | 6.79 | $108.64/hr |

**$25.44/hour is 8.5× the $3.00/hour project threshold**, so provisioning
requires explicit user approval. It was put to the user on 2026-08-28 and
**Phase 3B was deferred**. The Phase 3A tooling and this design are retained
unchanged for whenever budget or a smaller cluster shape becomes available.

A second cost consideration weighed in that decision: at the minimum shape,
**16 GPUs are billed while only 2 participate** (one rank per node, by design
— see §3). Fourteen idle paid GPUs is a poor ratio for a baseline whose whole
purpose is to isolate the cost of one inter-node hop.

**Also confirmed by the documentation**, and worth carrying into a future
Phase 3B: cluster nodes expose **dedicated high-speed interfaces `ens1`–`ens8`**
for inter-node traffic (A100: 1600 Gbps; H100/H200/B200: 3200 Gbps). This
vindicates §8's refusal to hard-code an interface — the data plane is those
dedicated NICs, and is neither `eth0` nor the VXLAN control-plane metadata.

### 9.3 Cluster tooling: resolved

Neither the RunPod MCP server nor `runpodctl` exposes any cluster endpoint, so
Phase 3 calls `POST /v2/clusters` directly. `scripts/runpod_cluster.sh` wraps
that API; the key lives in the macOS Keychain and never appears in a file, an
argument, or shell history.

RunPod API keys **cannot be created programmatically** — not via REST v2 (the
34-path contract has no key endpoint), not via MCP, not via `runpodctl`, whose
own help points at the console. RunPod's documentation confirms console-only
creation. A scoped key named `project3-nccl-agent` was therefore created
manually by the user and stored in the Keychain; `GET /v2/clusters` then
authenticated successfully.

One honest limitation on least privilege: RunPod's key model offers only
**All / Restricted / Read Only**, where *Restricted* granularises Serverless
endpoints alone. Cluster-scoped keys are not expressible, and *Read Only*
cannot create a cluster. The mitigation is a dedicated project key that can be
revoked when Phase 3 ends, not a narrower scope.

### 9.4 Assumptions still requiring runtime verification

These are **not** established and must not be treated as facts:

1. ~~That a 2 × 1 cluster is actually schedulable.~~ **RESOLVED — false.**
   See §9.2b: the minimum is 2 nodes × 8 GPUs on one of four supported GPU
   types.
2. **That cluster GPU pricing equals pod pricing.** The catalog price is a pod
   price, and `/v2/billing/clusters` is a separate billing scope. Still
   unverified — no cluster was ever billed.
3. **Which interface carries cluster-internal traffic**, and what its
   name, MTU and speed are. The VXLAN evidence says "an overlay device",
   not which one. This is exactly why §8 refuses to guess.
4. **The advertised inter-node network type and bandwidth.** Nothing in the
   catalog states it. It will be recorded from the nodes, not assumed.
5. **Whether passwordless SSH exists between member nodes.** `startSsh` covers
   ingress from outside using the account key; node-to-node reachability for
   mpirun is a different thing.
6. **Whether MPI can be installed on the member image** (needs apt access).

---

## 10. Smoke-First Experiment Strategy

Phase 3B does **not** begin with a sweep. Gates run in order, and each one can
stop the experiment before it costs more.

```text
  Step 1  multi-node init        mpirun launches; all ranks join a communicator
             │  fail ⇒ ABORT (launch broken; a sweep would only repeat it)
             ▼
  Step 2  transport verification NCCL_DEBUG=INFO run, proven NET/Socket
             │  fail ⇒ ABORT (an unverified transport is not a TCP baseline)
             ▼
  Step 3  minimal AllReduce      3 sizes: 8 B · 32 KiB · 128 MiB
             │  fail ⇒ ABORT (correctness gate)
             ▼
  Step 4  baseline sweep         25-point AllReduce sweep × 3 repeats
             │
             ▼
  Step 5  (only once stable)     add AllGather and ReduceScatter
```

**AllReduce first, alone.** It is the collective that matters most for
data-parallel training and the one whose behaviour Phases 1–2 characterised
best, so it gives the cleanest read on what crossing a node boundary costs.
AllGather and ReduceScatter are added only after the multi-node path is proven
stable — they are a `-c` flag on the runner, not new code.

**A distributed launch failure stops everything.** Implemented, not just
documented: the runner aborts before the smoke tier if the diagnostic run
fails, and skips the full tier if the smoke tier fails its correctness gate.

---

## 11. Benchmark Matrix

Identical sweep parameters to Phases 1 and 2, so results stay comparable.

| Tier | Sizes | Warmup | Iters | Repeats | Collectives |
|------|-------|-------:|------:|--------:|-------------|
| Diagnostic | 8 B | 1 | 1 | 1 | AllReduce (INFO, **not** a measurement) |
| Smoke | `-b 8 -e 128M -f 4096` → 8 B, 32 KiB, 128 MiB | 5 | 20 | 1 | AllReduce |
| Full | `-b 8 -e 128M -f 2` → 25 points | 20 | 50 | 3 | AllReduce |
| Extended | as above | 20 | 50 | 3 | + AllGather, ReduceScatter |

fp32, out-of-place and in-place both recorded, `-c 1` validation on.

---

## 12. Correctness

Unchanged from Phase 1 and enforced by the same code:

1. process exit code 0;
2. `#wrong == 0` on every row, both placements;
3. `# Out of bounds values : 0 OK` trailer present;
4. no CUDA/NCCL error output;
5. rank count matches the request;
6. bus/algorithmic bandwidth ratio matches the expected factor for the rank
   count.

Rows failing any check are marked `correctness_ok = false` and excluded from
analysis. Failed runs are preserved, not deleted.

---

## 13. Transport Verification

**This is the gate that makes Phase 3 worth running.**

`scripts/verify_nccl_transport.py` reads the `NCCL_DEBUG=INFO` output and
requires **all** of:

| Check | Requirement |
|-------|-------------|
| Wanted transport present | at least one `via NET/Socket` channel line |
| No competing transport | **zero** `NET/IB*` lines — catches a silent RDMA fallback |
| Genuinely multi-node | ranks span ≥ 2 distinct hosts |
| Correct interface | the `NET/Socket : Using [n]<iface>` banner names the interface we chose |

Exit status is non-zero if any check fails, and the runner then **refuses to
run a single timed measurement**. Evidence is written to
`transport_verification.json` alongside the raw output.

The verifier is covered by **12 unit tests** against synthetic NCCL logs,
including the two cases that matter most: an InfiniBand log must be *rejected*
when sockets were requested, and a log mixing both transports must also be
rejected. These tests exist so the gate is known to work before any paid
multi-node time depends on it.

A run whose transport cannot be verified produces no accepted performance
number. `transport_verified` is carried on every result row, and
`parse_nccl_output.py --strict` fails if any multi-node row lacks it.

---

## 14. Environment Metadata

Reuses the Phase 1/2 capture, extended per node:

```text
results/raw/<experiment-id>/
    env.json / env.txt            primary node (parser entry point)
    nodes/<hostname>/env.json     per-node hardware/software capture
    nodes/<hostname>/netenv.json  per-node network capture
    hostfile                      exact MPI hostfile used
    transport_verification.json   the transport proof
    nccl_debug_info.txt           INFO diagnostic
    run_manifest.json             every command, exit code, rank→host map
    <collective>.<tier>.r<N>.std{out,err}.txt
```

The same allowlist + credential-redaction rule applies to both capture scripts,
so cluster credentials cannot reach `results/` or git.

---

## 15. Result Schema Changes

Eight **additive, optional** fields. `schema_version` stays **1** — RFC-001
states additive optional fields do not bump it, and single-node rows simply
carry `null`.

| Field | Purpose |
|-------|---------|
| `hosts` | hostnames taking part |
| `rank_to_host` | rank index → hostname, parsed from runtime output |
| `ranks_per_node` | ranks per host |
| `net_interface` | the interface NCCL was pointed at |
| `transport` | transport **proven** from runtime evidence |
| `transport_verified` | whether the check actually passed |
| `mpi_implementation` | OpenMPI / MPICH / … |
| `launcher` | `mpirun`, or null for single-process runs |

`transport_verified` is the important one: a row without it must never be used
in a transport comparison, and `--strict` enforces that.

Verified by test that Phase 1 and Phase 2 data reparses **identically** and
that single-node rows leave every new field `null` rather than acquiring
invented multi-node metadata.

---

## 16. Failure Policy

Cluster time is more expensive than a single pod, and multi-node failures are
easier to repeat expensively.

1. capture the failure and preserve all logs, per node;
2. **stop** — do not retry the same failing launch;
3. diagnose from the preserved evidence;
4. reproduce at the lowest layer that still shows it (single node first, then
   two nodes) — Phase 2 showed the value of this, where a failed gencode
   hypothesis correctly redirected the diagnosis;
5. fix locally or on cheap hardware;
6. only then re-provision.

Specific stop conditions, all implemented:

- MPI launch fails → abort before the smoke tier
- transport not verified as NET/Socket → abort before any timed run
- smoke tier fails correctness → skip the full sweep
- binary does not link libmpi → refuse to start at all

---

## 17. Cost Policy

Project rule: **total** compute ≤ $3.00/hour is autonomous; above it needs
approval; the threshold applies to the whole cluster, not per GPU.

| Configuration | Total | Decision |
|---------------|------:|----------|
| 2 nodes × 1 × RTX A5000 | **$0.54/hr** | autonomous |
| 2 nodes × 2 × RTX A5000 | $1.08/hr | autonomous |
| 2 nodes × 1 × H200 | $9.18/hr | **approval required** |

Price must be re-queried at provisioning time; the §9.2 table is a snapshot and
availability has already changed materially between phases.

Budget target for Phase 3B: **≤ $2 total**, expected well under $1 based on
Phase 1B and 2 wall-clock times (11–13 min of pod life each).

Cost discipline carried over: everything buildable locally is built locally;
the cluster compiles and measures only; no interactive debugging on cluster
hardware; terminate immediately after collection; analyse locally.

---

## 18. Cleanup

1. save results, logs and per-node metadata;
2. verify the raw directory is complete locally;
3. **delete the cluster** (`DELETE /v2/clusters/{id}`);
4. confirm no member pods survive the cluster deletion — a leaked member pod
   would keep billing;
5. confirm no orphaned network volumes;
6. report the final infrastructure state.

Analysis, plotting and documentation happen **after** the cluster is gone.

---

## 19. Success Criteria

Phase 3B is complete when:

1. NCCL initialises across ≥ 2 nodes and all ranks join one communicator;
2. **transport is verified as NET/Socket** on the explicitly chosen interface,
   with zero NET/IB evidence;
3. all runs pass the correctness gate with zero validation errors;
4. rows validate against the schema, with `transport_verified: true`;
5. latency, algbw and busbw are characterised across 8 B – 128 MiB for
   AllReduce, with run-to-run spread reported;
6. per-node environment and network metadata are captured and preserved;
7. the inter-node result is compared against the Phase 1/2 intra-node figures
   with the hardware differences stated explicitly;
8. spend is within budget, the cluster is deleted, and deletion is verified;
9. limitations are documented — including that this is TCP only, that no RDMA
   comparison exists yet, and whatever the cluster's actual network turns out
   to be.

Phase 3B is **not** complete because numbers were produced. It is complete when
those numbers are provably from the transport they claim.

---

## 20. What Phase 3A Delivered

| Artifact | Purpose |
|----------|---------|
| `scripts/collect_net_env.sh` | per-node network discovery; selects nothing |
| `scripts/setup_multinode_env.sh` | installs MPI, sets up inter-node SSH, verifies both |
| `scripts/setup_nccl_tests.sh -m` | MPI build, refuses a binary that does not link libmpi |
| `scripts/run_nccl_multinode.sh` | mpirun launcher, transport gate, smoke gate, manifest |
| `scripts/verify_nccl_transport.py` | the NET/Socket proof, 12 unit tests |
| schema + parser | 8 additive multi-node fields, `--strict` enforces verification |
| this document | the design |

All of it built and tested locally. **Phase 3A used $0.00 of RunPod compute.**
