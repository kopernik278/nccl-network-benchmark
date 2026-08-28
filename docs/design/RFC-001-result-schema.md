# RFC-001: NCCL Benchmark Result Schema

Status: accepted (v1)
Date: 2026-08-26
Related: `docs/experiments/phase1_nccl_baseline.md`, `schemas/nccl_result.schema.json`

---

## Motivation

Benchmark numbers without provenance are unusable. A result that cannot be
traced back to a specific commit, GPU, topology, NCCL version, and command
line cannot be reproduced, compared, or defended.

This RFC defines one machine-readable row format used by every phase of the
project, so that a Phase 1 PCIe AllReduce point and a Phase 5 InfiniBand
AllReduce point are directly comparable and unambiguously distinguishable.

---

## Design Principles

1. **One row = one measurement point.** A row is uniquely identified by
   (`experiment_id`, `collective`, `message_size_bytes`, `placement`,
   `repeat_index`).
2. **Self-contained rows.** Every row carries its full environment. Rows can
   be concatenated across experiments and phases and remain interpretable
   without a side lookup.
3. **Raw is separate from parsed.** `results/raw/` holds verbatim tool output
   and is never edited. `results/summary/` is derived and regenerable.
4. **Absent ≠ zero.** Anything not detected is `null`, never `0`, `""`, or a
   guess.
5. **Provenance of the number itself is a field.** `value_kind` distinguishes
   a real measurement from anything else, and the pipeline enforces it.
6. **No secrets.** Environment capture is allowlist-based with a redaction
   filter.

---

## Storage Format

| File | Format | Role |
|------|--------|------|
| `results/summary/<id>/results.jsonl` | JSON Lines | canonical; one JSON object per line |
| `results/summary/<id>/results.csv` | CSV | same rows, flat, for spreadsheets/plots |

JSONL is canonical because rows are nested (GPU list, env vars) and because
appending is safe. CSV is a lossy convenience view: nested fields are
serialized as compact JSON strings in a single column.

The JSON Schema in `schemas/nccl_result.schema.json` validates a single row.

---

## Field Reference

### Identity and provenance

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | int | `1` |
| `experiment_id` | string | `p1-nccl-baseline-20260826T141530Z-b1f095c` |
| `phase` | string | `phase1` … `phase9` |
| `timestamp` | string | UTC ISO 8601, e.g. `2026-08-26T14:15:30Z` |
| `git_commit` | string\|null | full SHA of this repo at run time |
| `git_dirty` | bool\|null | true if the working tree had uncommitted changes |
| `value_kind` | enum | `measured` \| `estimated` \| `theoretical` \| `synthetic` |

`value_kind` is the integrity interlock. `measured` is only written for rows
parsed from real tool output. Test fixtures carry a `SYNTHETIC` marker that
the parser detects and downgrades to `synthetic`, so fabricated numbers
cannot silently enter a results file. Vendor peak figures, if ever recorded,
must be `theoretical`.

### Infrastructure

| Field | Type | Notes |
|-------|------|-------|
| `provider` | string\|null | `runpod`, `local`, … |
| `provider_instance_id` | string\|null | e.g. RunPod pod ID |
| `hostname` | string\|null | |
| `os` | string\|null | distro pretty name |
| `kernel` | string\|null | `uname -r` |
| `cpu_model` | string\|null | |
| `cpu_cores` | int\|null | |
| `node_count` | int | 1 for single-node |
| `gpu_model` | string\|null | e.g. `NVIDIA RTX A5000` |
| `gpu_count` | int | GPUs participating (= rank count in Phase 1) |
| `gpus` | array\|null | per-GPU index, name, PCI bus ID, memory |
| `topology` | string\|null | verbatim `nvidia-smi topo -m` |
| `topology_summary` | string\|null | e.g. `PIX`, `NV12`, `SYS` — dominant link |
| `network` | string\|null | `none-single-node`, `tcp`, `roce`, `infiniband` |
| `nvlink_present` | bool\|null | |
| `p2p_enabled` | bool\|null | |

`network` is `none-single-node` for Phase 1 — an honest value, not a blank.
It must only say `roce`/`infiniband` when the run genuinely used that fabric.

### Multi-node (Phase 3+)

Additive and optional: single-node rows carry `null` rather than invented
values. Because they are additive optional fields, `schema_version` stays `1`.

| Field | Type | Notes |
|-------|------|-------|
| `hosts` | array\|null | hostnames taking part in the run |
| `rank_to_host` | object\|null | rank index → hostname, parsed from runtime output |
| `ranks_per_node` | int\|null | |
| `net_interface` | string\|null | the interface NCCL was pointed at (`NCCL_SOCKET_IFNAME`) — chosen from runtime inspection, never guessed |
| `transport` | string\|null | transport **proven** from runtime evidence: `NET/Socket`, `NET/IB`, `P2P/direct` |
| `transport_verified` | bool\|null | whether a transport check actually passed |
| `mpi_implementation` | string\|null | OpenMPI, MPICH, … |
| `launcher` | string\|null | `mpirun`, or null for single-process runs |

`transport_verified` is an integrity field of the same kind as `value_kind`.
Setting `NCCL_IB_DISABLE=1` states an intent; only a check against
`NCCL_DEBUG=INFO` output establishes the outcome. **A row with
`transport_verified` false or null must not be used in a transport
comparison**, and `parse_nccl_output.py --strict` fails when a multi-node row
lacks it. Without this, a silent fallback to RDMA could be reported as a TCP
baseline and would corrupt every later speed-up figure.

### Software versions

| Field | Type |
|-------|------|
| `cuda_version` | string\|null |
| `driver_version` | string\|null |
| `nccl_version` | string\|null |
| `nccl_version_source` | string\|null (how it was detected) |
| `mpi_version` | string\|null (`null` in Phase 1 — MPI intentionally unused) |
| `compiler_version` | string\|null |
| `nccl_tests_commit` | string\|null |

### Benchmark configuration

| Field | Type | Notes |
|-------|------|-------|
| `benchmark_tool` | string | `nccl-tests` (later: `custom-ring-allreduce`) |
| `collective` | enum | `all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`, `p2p` |
| `datatype` | string | `float` |
| `redop` | string\|null | `sum` |
| `root` | int\|null | `-1` when not applicable |
| `message_size_bytes` | int | size as reported by the tool |
| `count_elements` | int\|null | element count |
| `placement` | enum | `out_of_place` \| `in_place` |
| `warmup_iterations` | int | |
| `measured_iterations` | int | |
| `repeat_index` | int | 0-based; distinguishes whole-sweep repeats |
| `tier` | string\|null | `smoke` \| `full` |

### Measurements

| Field | Type | Unit |
|-------|------|------|
| `latency_us` | float | microseconds, mean over `measured_iterations` |
| `algorithmic_bandwidth_gbps` | float | **GB/s = 10⁹ bytes/s** |
| `bus_bandwidth_gbps` | float | **GB/s = 10⁹ bytes/s** |

> **Unit warning.** Despite the `_gbps` suffix (kept for schema stability),
> these are gigaBYTES per second, decimal, matching nccl-tests output. They
> are not gigabits per second. Confusing the two is an 8× error.

### Correctness

| Field | Type | Notes |
|-------|------|-------|
| `wrong_count` | int\|null | nccl-tests `#wrong` for this row |
| `correctness_ok` | bool | false ⇒ excluded from performance analysis |
| `out_of_bounds_ok` | bool\|null | from the `Out of bounds values` trailer |
| `exit_code` | int\|null | of the producing process |

### Reproduction

| Field | Type | Notes |
|-------|------|-------|
| `command` | string | exact command line executed |
| `env` | object | allowlisted, redacted environment variables |
| `raw_output_path` | string\|null | repo-relative path to the verbatim output |
| `notes` | string\|null | free text |

---

## Environment Capture Policy

Captured by prefix allowlist: `NCCL_`, `CUDA_`, `NVIDIA_`, `UCX_`, `OMPI_`,
`RUNPOD_`.

Then filtered: any variable whose **name** contains `KEY`, `TOKEN`, `SECRET`,
`PASSWORD`, `PASSWD`, `CREDENTIAL`, `AUTH`, or `PRIVATE` is dropped and
replaced by a `<redacted>` marker recording only that it existed.

A full `env` dump is never written. This keeps RunPod and GitHub credentials
out of `results/`, out of git history, and out of any published report.

---

## CSV Mapping

Column order is fixed (see `scripts/parse_nccl_output.py::CSV_COLUMNS`).
Nested fields (`gpus`, `env`) and the multi-line `topology` field are
serialized as compact JSON strings so the CSV stays one-row-per-record.
CSV is a view; JSONL is the source of truth.

---

## Versioning

`schema_version` starts at `1`. Additive optional fields do not bump it.
Removing a field, renaming one, or changing a unit does — and requires a note
here plus a parser that can still read older rows.

---

## Worked Example (field shape only — values are illustrative, not measured)

```json
{
  "schema_version": 1,
  "experiment_id": "p1-nccl-baseline-20260826T141530Z-b1f095c",
  "phase": "phase1",
  "value_kind": "measured",
  "collective": "all_reduce",
  "placement": "out_of_place",
  "message_size_bytes": 134217728,
  "latency_us": null,
  "algorithmic_bandwidth_gbps": null,
  "bus_bandwidth_gbps": null,
  "correctness_ok": true,
  "network": "none-single-node",
  "command": "./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -w 20 -n 50"
}
```

Measurement fields are shown as `null` on purpose: this repository contains no
benchmark results yet, and this RFC will not be the place where the first
fake ones appear.
