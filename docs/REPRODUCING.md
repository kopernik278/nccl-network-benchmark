# Reproducing this work

Every command below is a script that exists in this repository. Nothing here
requires a new orchestration framework, and you do **not** need to rerun every
historical experiment — §2 is a six-step path that reproduces the project's main
results.

The analysis layer is GPU-free. If you only want to check that the published
numbers follow from the published raw data, skip to §5.

---

## 0. Environment assumptions

| | Assumption |
|---|---|
| OS | Linux x86-64. All GPU work in this project ran on Ubuntu 22.04 / 24.04 containers. |
| GPUs | 2–4 NVIDIA GPUs on **one node**. No multi-node path is exercised. Compute capability 8.6 (A40) and 8.9 (L4) were used; sm_120 needs NCCL ≥ 2.26. |
| Driver / CUDA | Driver 550+ with CUDA 12.4–12.8. `nvcc` is required only for the CUDA benchmarks (§2 steps 3–5). |
| NCCL | 2.25.1 or 2.27.3 were used. Ships with the PyTorch container; `nccl.h` must be on the include path for the CUDA benchmarks. |
| PyTorch | 2.8.0+cu128 for the DDP work. The training script also runs on CPU/gloo, which is how it was validated before any GPU was rented. |
| Python | 3.10+. **The parsers, analysis and tests use the standard library only**; `matplotlib` is needed only to regenerate figures. |
| Profiler | Nsight Systems (optional). `apt-get install cuda-nsight-systems-12-8`. Nsight **Compute** needs privileges a container tenant usually lacks — see the note in §4. |

Nothing in this repository requires root beyond installing packages, and no step
needs GPU performance counters.

### Dependencies

```bash
# analysis / tests — no GPU, no third-party packages
python3 -m pytest tests/ -q

# figures only
pip install matplotlib

# DDP benchmark
pip install torch          # or use a runpod/pytorch container
```

---

## 1. Where things live

| Path | Contents |
|---|---|
| `results/raw/<experiment-id>/` | Verbatim tool output, `env.json`, `run_manifest.json`, NCCL debug logs, profiler summaries. **Append-only; never edited.** |
| `results/summary/<experiment-id>/` | `results.jsonl` + `results.csv` produced by a parser from the directory above. **Derived; regenerable.** |
| `results/plots/` | Generated figures. |
| `schemas/nccl_result.schema.json` | The result schema every row validates against. |
| `docs/experiments/` | One report per phase — the primary record. |

Experiment IDs are `p<phase>-<slug>-<UTC timestamp>-<git short sha>`, so every
summary row traces back to a raw directory, a commit, and a wall-clock time.
Each row also carries `raw_output_path`, pointing at the exact file it came from.

---

## 2. Minimal reproduction path

Six steps. Steps 1–2 need any GPU node; 3–6 need 2–4 GPUs.

### Step 1 — environment and topology

```bash
bash scripts/collect_env.sh -o .
cat env.txt
```

Writes `env.json` (machine-readable, consumed by every parser) and `env.txt`.
Environment variables pass through an allowlist with credential redaction, so
this is safe to commit.

### Step 2 — NCCL AllReduce smoke / baseline

```bash
bash scripts/setup_nccl_tests.sh                 # builds nccl-tests
bash scripts/run_nccl_baseline.sh -g 2 -t smoke  # or -t full for the sweep
```

`-t smoke` is a short correctness-gated run; `-t full` is the Phase 1/2 sweep.
Add `-e "NCCL_ALGO=Tree,NCCL_PROTO=LL128"` and `-l tree-ll128` to reproduce a
Phase 5 configuration.

### Step 3 — custom Ring AllReduce correctness

```bash
make -C src/ring_allreduce SM=86        # match your compute capability
./src/ring_allreduce/build/ring_allreduce -g 2 -w 5 -n 20
```

Correctness is not a separate mode: **every configuration is checked against an
exact fp32 oracle before it is timed**, and a configuration that fails is not
timed at all. The binary also runs a *functional* peer test and refuses to use
a P2P path that does not actually work — pass `--allow-host-staged` to fall back
to host staging instead of aborting.

### Step 4 — Ring V1 vs V2: the synchronization result

```bash
bash scripts/run_ring_benchmark.sh -g 4
python3 scripts/parse_ring_output.py --raw-dir results/raw/<new-experiment-id>
```

V1 is the naive ring with device-wide barriers; V2 replaces them with per-peer
events. The published gain is **3.28–3.35× at ≥ 1 MiB**
([Phase 7A](experiments/p7a-harness-validation.md)) — use that figure, not the
uncorrected one in the Phase 6 report.

### Step 5 — communication/compute overlap

```bash
make -C src/overlap SM=86
./src/overlap/build/overlap_bench -g 4 -w 5 -n 3 --repeats 3 --only-micro \
    --sizes 1M,16M,128M --ratios 1.0 --workloads compute,memory,mixed
python3 scripts/parse_overlap_output.py --raw-dir results/raw/<id> --phase phase8
```

If the run hangs with all GPUs at 100 %, you have hit the P2P problem described
in §4 — rerun with `NCCL_P2P_DISABLE=1`.

### Step 6 — real DDP bucket comparison

```bash
bash scripts/preflight_ddp.sh /tmp/preflight 4      # must print PREFLIGHT OK
bash scripts/run_final_ddp.sh /tmp/p10 "$(cat /tmp/preflight/transport_env.txt)"
python3 scripts/parse_ddp_output.py --raw-dir /tmp/p10 --phase phase10
python3 scripts/analyze_ddp.py results/summary/<id>/results.jsonl \
        --prefix final- --baseline 25 --optimized 4
```

`run_final_ddp.sh` runs the 25 / 4 / 64 MiB configurations as **independent
process launches** plus a single-GPU floor and a non-overlapped control. Useful
knobs: `WARMUP`, `STEPS`, `LAUNCHES`, `SKIP_REF=1`, `TAG=<name>`.

To reproduce the Phase 9/10 SHM anchor on a host where P2P works:

```bash
TAG=shm SKIP_REF=1 bash scripts/run_final_ddp.sh /tmp/p10 "NCCL_P2P_DISABLE=1"
```

---

## 3. Running the DDP workload without a GPU

The training benchmark falls back to gloo and host timers, which is how it was
debugged before any GPU was rented:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 src/ddp/train_ddp.py \
  --mode ddp --bucket-mb 1 --warmup 2 --steps 4 --repeats 2 \
  --batch 2 --seq 64 --layers 2 --heads 2 --embd 64 --vocab 512 \
  --precision fp32 --pool 2 --nosync-probe
```

The timings are meaningless on CPU. The point is that the correctness gate, the
DDP bucket reporting, the `no_sync` probe and the CSV output are all exercised.

---

## 4. Profiling, and the two things that will bite you

**Nsight Systems** works fine as a tenant:

```bash
NS=$(ls /opt/nvidia/nsight-systems/*/target-linux-x64/nsys | head -1)
$NS profile -t cuda,nvtx --force-overwrite true -o trace <your command> --nvtx
$NS stats --force-export=true --report cuda_gpu_kern_sum --format csv trace.nsys-rep
python3 results/raw/p10-*/extract_timeline.py.txt trace.sqlite 0
```

Build the CUDA benchmarks with `USE_NVTX=1 BUILD=build_nvtx` so the profiled
binary is never the one that produced reported timings. `.nsys-rep` and
`.sqlite` files are deliberately **not** committed — only their derived CSV and
timeline JSON are.

**Nsight Compute will probably fail.** Hardware counters need `CAP_SYS_ADMIN` or
`NVreg_RestrictProfilingToAdminUsers=0`, neither of which a container tenant
controls:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device
```

This is why [Phase 8](experiments/p8-contention.md) contains no counter-verified
arithmetic intensity. Attempt it once, record the outcome, move on.

**The P2P path may be broken even when the capability bit says otherwise.** On
three of four cloud hosts in this project, `cudaDeviceCanAccessPeer` returned
*yes* and the path then corrupted data or deadlocked. `scripts/preflight_ddp.sh`
tests it functionally and picks a transport that passes; run it first.

---

## 5. Verifying the published results without a GPU

Every number in the reports is derived from committed raw data by a committed
parser. To check that end to end:

```bash
# 1. all local tests (parsers, schema conformance, integrity interlocks)
python3 -m pytest tests/ -q                       # 72 passed

# 2. re-derive a summary from its raw directory and diff it
python3 scripts/parse_ddp_output.py \
    --raw-dir results/raw/p10-final-ddp-20260830T0741Z-502811c \
    --out-dir /tmp/recheck --phase phase10
diff <(sort /tmp/recheck/results.jsonl) \
     <(sort results/summary/p10-final-ddp-20260830T0741Z-502811c/results.jsonl) \
  && echo "summary reproduces from raw"

# 3. re-derive the headline analysis
python3 scripts/analyze_ddp.py \
    results/summary/p10-final-ddp-20260830T0741Z-502811c/results.jsonl \
    --prefix final- --baseline 25 --optimized 4

# 4. validate every row in the repository against the schema
python3 - <<'EOF'
import json, pathlib, re
s = json.load(open("schemas/nccl_result.schema.json"))
req, allowed = set(s["required"]), set(s["properties"])
pat = re.compile(s["properties"]["phase"]["pattern"])
n = bad = 0
for f in pathlib.Path("results/summary").glob("*/results.jsonl"):
    for line in f.read_text().splitlines():
        r = json.loads(line); n += 1
        if (req - set(r)) or (set(r) - allowed) or not pat.match(r["phase"]):
            bad += 1; print("PROBLEM", f)
print(f"{n} rows, {bad} problems")
EOF
```

The other parsers follow the same shape:

```bash
python3 scripts/parse_nccl_output.py    --raw-dir results/raw/p2-scaling-g4-*
python3 scripts/parse_ring_output.py    --raw-dir results/raw/p6-ring-allreduce-*
python3 scripts/parse_overlap_output.py --raw-dir results/raw/p8-interference-matrix-* --phase phase8
```

Add `--strict` to make any parse warning a non-zero exit.

### Regenerating figures

```bash
cd scripts
python3 plot_contention.py ../results/summary/p8-*/results.jsonl -o ../results/plots/p8-contention.png
python3 plot_ddp.py        ../results/summary/p9-*/results.jsonl \
        --timelines ../results/raw/p9-*/profile-ddp-*.timeline.json -o ../results/plots/p9-ddp.png
python3 plot_final_ddp.py  ../results/summary/p10-*/results.jsonl \
        --timelines ../results/raw/p10-*/profile-final-*.timeline.json -o ../results/plots/p10-final.png
```

---

## 6. What you cannot reproduce from this repository

- **The Phase 6 host anomaly.** The ~4.4 ms floor followed that specific
  machine; the pod is gone and the cause was never recovered. Phase 7A documents
  what it was *not*.
- **Multi-node, RDMA, RoCE, InfiniBand or NVLink results.** None exist. Phase 3
  contains a prepared design and a cost justification for deferring it.
- **Hardware-counter measurements.** See §4.
- **Absolute timings on different hardware.** Every phase reports its GPU model,
  topology and transport for exactly this reason; the qualitative conclusions are
  what transfer.
