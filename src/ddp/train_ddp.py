#!/usr/bin/env python3
"""Phase 9 — real PyTorch DDP training benchmark.

Runs a compact GPT-style model as an actual training step (forward, loss,
backward, optimizer) under DistributedDataParallel, and measures where the time
goes as a function of DDP's gradient bucket capacity.

Nothing here simulates backward. The compute is the model's own autograd graph
and the communication is DDP's own reducer.

Timing method
-------------
CUDA events bracket each region and are read only *after* the measured window
closes, so no synchronisation is injected into the steps being measured. Three
regions are timed per step:

    fwd  : forward + loss
    bwd  : loss.backward() — in DDP mode this includes the gradient AllReduce
           tail, because DDP makes the default stream wait on its comm stream
           before backward returns
    opt  : optimizer.step()

The exposed communication tail is obtained by differencing, not by guessing:
the same loop is re-run inside `model.no_sync()`, which suppresses the reducer
entirely and leaves pure backward compute.

    exposed_tail = bwd(with gradient sync) - bwd(no_sync)

That difference bundles the genuine post-backward tail together with any
slowdown the concurrent collective inflicts on backward itself. Phases 7B and 8
showed that second term is real, so this is reported as a combined
"synchronisation cost", never as proof that communication was free.

Modes
-----
    --mode single    one GPU, no distributed anything — the compute reference
    --mode ddp       DistributedDataParallel with --bucket-mb
    --mode serial    no_sync backward, then one flat AllReduce afterwards —
                     a deliberately serialised reduction for comparison

Launch (DDP):
    torchrun --nproc_per_node=4 src/ddp/train_ddp.py --mode ddp --bucket-mb 25
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig, flops_per_step  # noqa: E402

CSV_FIELDS = [
    "experiment", "mode", "label", "bucket_mb", "world_size", "repeat",
    "step_ms", "fwd_ms", "bwd_ms", "opt_ms", "step_wall_ms",
    "tokens_per_s", "grad_bytes", "param_count", "loss",
]


# ---------------------------------------------------------------------------
# distributed setup
# ---------------------------------------------------------------------------
def dist_init() -> tuple[int, int, int]:
    """torchrun supplies RANK/WORLD_SIZE/LOCAL_RANK; single-GPU mode needs none."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    # gloo only ever runs on this project's laptop, where the script is
    # exercised for correctness before any GPU is paid for.
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def peak_mem_gib() -> float:
    return (torch.cuda.max_memory_allocated() / (1 << 30)) if torch.cuda.is_available() else 0.0


def log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# data — synthetic, deterministic, and resident on the GPU
# ---------------------------------------------------------------------------
def make_batches(cfg: GPTConfig, batch: int, pool: int, device, seed: int):
    """Pre-generate a small pool of batches so no I/O enters the timed path.

    A pool rather than a single batch, so a configuration cannot accidentally
    benefit from operating on one perfectly cached tensor forever.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(pool):
        idx = torch.randint(0, cfg.vocab_size, (batch, cfg.seq_len), generator=g)
        tgt = torch.randint(0, cfg.vocab_size, (batch, cfg.seq_len), generator=g)
        out.append((idx.to(device, non_blocking=True), tgt.to(device, non_blocking=True)))
    return out


# ---------------------------------------------------------------------------
# one measured window
# ---------------------------------------------------------------------------
class _HostMark:
    """CPU stand-in for a CUDA event, so the script runs without a GPU."""

    __slots__ = ("t",)

    def __init__(self) -> None:
        self.t = 0.0

    def record(self) -> None:
        self.t = time.perf_counter()

    def elapsed_time(self, other: "_HostMark") -> float:
        return (other.t - self.t) * 1e3


class Timers:
    """Event pairs recorded during the window, read only after it closes.

    Reading during the window would insert a synchronisation into exactly the
    region whose overlap is being measured, so every pair is buffered and the
    single synchronise happens at the end.
    """

    def __init__(self) -> None:
        self.ev = []
        self.cuda = torch.cuda.is_available()

    def new_step(self):
        mk = ((lambda: torch.cuda.Event(enable_timing=True)) if self.cuda else _HostMark)
        e = [mk() for _ in range(4)]
        self.ev.append(e)
        return e

    def read(self) -> dict[str, list[float]]:
        if self.cuda:
            torch.cuda.synchronize()
        out = {"step": [], "fwd": [], "bwd": [], "opt": []}
        for a, b, c, d in self.ev:
            out["fwd"].append(a.elapsed_time(b))
            out["bwd"].append(b.elapsed_time(c))
            out["opt"].append(c.elapsed_time(d))
            out["step"].append(a.elapsed_time(d))
        return out


def run_window(model, raw_model, opt, batches, steps, *, sync_grads: bool,
               serial: bool, autocast_ctx, nvtx: bool, world: int):
    """Run `steps` training steps and return per-step GPU timings.

    sync_grads=False wraps the step in model.no_sync(), which stops DDP's
    reducer from launching any collective — the pure-compute backward.
    serial=True instead performs one flat AllReduce after backward has
    completely finished, the deliberately non-overlapped configuration.
    """
    t = Timers()
    params = [p for p in raw_model.parameters() if p.requires_grad]
    nvtx_range = torch.cuda.nvtx.range if (nvtx and torch.cuda.is_available()) else None
    losses = []

    host_t0 = time.perf_counter()
    for i in range(steps):
        idx, tgt = batches[i % len(batches)]
        a, b, c, d = t.new_step()

        no_sync = (model.no_sync() if (isinstance(model, DDP) and not sync_grads)
                   else nullcontext())
        with no_sync:
            a.record()
            if nvtx_range:
                with nvtx_range("forward"):
                    with autocast_ctx():
                        loss = model(idx, tgt)
            else:
                with autocast_ctx():
                    loss = model(idx, tgt)
            b.record()

            if nvtx_range:
                with nvtx_range("backward"):
                    loss.backward()
            else:
                loss.backward()

            if serial:
                # One flat AllReduce, issued only after backward is complete.
                # No DDP internals are touched; this is the honest serialised
                # baseline the overlap is being compared against.
                grads = [p.grad for p in params]
                flat = torch._utils._flatten_dense_tensors(grads)
                if nvtx_range:
                    with nvtx_range("serial_allreduce"):
                        dist.all_reduce(flat)
                else:
                    dist.all_reduce(flat)
                flat.div_(world)
                for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
                    g.copy_(s)
            c.record()

            if nvtx_range:
                with nvtx_range("optimizer"):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            else:
                opt.step()
                opt.zero_grad(set_to_none=True)
            d.record()
        losses.append(loss.detach())

    per = t.read()
    host_ms = (time.perf_counter() - host_t0) * 1e3 / steps
    return per, host_ms, [float(x) for x in losses]


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------
def check_param_sync(raw_model, rank: int, world: int) -> float:
    """Max |p_rank - p_rank0| over every parameter. 0.0 on one process."""
    if world == 1:
        return 0.0
    dev = next(raw_model.parameters()).device
    worst = torch.zeros((), device=dev)
    for p in raw_model.parameters():
        ref = p.detach().clone()
        dist.broadcast(ref, src=0)
        worst = torch.maximum(worst, (p.detach() - ref).abs().max())
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    return float(worst)


def resync_params(raw_model, world: int) -> None:
    """Restore the cross-rank invariant after a no_sync probe."""
    if world == 1:
        return
    for p in raw_model.parameters():
        dist.broadcast(p.data, src=0)


def grads_finite(raw_model) -> bool:
    for p in raw_model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["single", "ddp", "serial"], default="ddp")
    ap.add_argument("--bucket-mb", type=float, default=25.0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--embd", type=int, default=768)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--pool", type=int, default=4, help="pre-generated batches")
    ap.add_argument("--nvtx", action="store_true", help="profiling build; never a benchmark sample")
    ap.add_argument("--nosync-probe", action="store_true",
                    help="also measure backward with the reducer suppressed")
    ap.add_argument("--calibrate", action="store_true",
                    help="print model size and one timing estimate, then exit")
    a = ap.parse_args()

    rank, world, local = dist_init()
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(a.seed)
    # Repeatability over peak throughput: no autotuning that could pick a
    # different algorithm between bucket configurations.
    torch.backends.cudnn.benchmark = False

    cfg = GPTConfig(vocab_size=a.vocab, n_layer=a.layers, n_head=a.heads,
                    n_embd=a.embd, seq_len=a.seq)
    raw_model = GPT(cfg).to(device)
    grad_bytes = raw_model.grad_bytes()
    n_params = raw_model.num_params()

    if a.precision == "bf16" and device.type == "cuda":
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    else:
        autocast_ctx = nullcontext

    label = a.label or (a.mode if a.mode != "ddp" else f"ddp-{a.bucket_mb:g}mb")

    if a.mode == "single" or world == 1:
        model = raw_model
        bucket_mb = None
    else:
        model = DDP(raw_model, device_ids=[local] if device.type == "cuda" else None,
                    bucket_cap_mb=a.bucket_mb, broadcast_buffers=False,
                    gradient_as_bucket_view=True)
        # In serial mode the reducer never runs, so reporting a bucket capacity
        # would invite the reader to treat it as a bucket-size data point.
        bucket_mb = a.bucket_mb if a.mode == "ddp" else None

    opt = torch.optim.AdamW(raw_model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=(device.type == "cuda"))
    batches = make_batches(cfg, a.batch, a.pool, device, a.seed + rank)
    tokens_per_step = a.batch * a.seq * world

    if rank == 0:
        print(f"# phase9 ddp training benchmark")
        print(f"# mode={a.mode} label={label} world_size={world} bucket_mb={bucket_mb}")
        print(f"# model: layers={cfg.n_layer} heads={cfg.n_head} embd={cfg.n_embd} "
              f"vocab={cfg.vocab_size} seq={cfg.seq_len} tie={cfg.tie_embeddings}")
        print(f"# params={n_params} grad_bytes={grad_bytes} "
              f"({grad_bytes / (1 << 20):.1f} MiB, dtype=float32)")
        print(f"# batch_per_gpu={a.batch} tokens_per_step={tokens_per_step} "
              f"precision={a.precision} optimizer=AdamW(fused)")
        print(f"# warmup={a.warmup} steps={a.steps} repeats={a.repeats} "
              f"pool={a.pool} seed={a.seed} nvtx={a.nvtx}")
        est = flops_per_step(cfg, a.batch * a.seq)
        print(f"# ESTIMATED fwd+bwd flops per GPU per step = {est / 1e12:.2f} TFLOP "
              f"(analytic, not measured)")
        if bucket_mb:
            print(f"# ESTIMATED buckets = ceil({grad_bytes / (1 << 20):.1f} MiB / "
                  f"{bucket_mb:g} MiB) ~= {int(-(-grad_bytes // (bucket_mb * (1 << 20))))} "
                  f"(DDP packs by parameter order; actual count reported below)")

    # ---- warmup: DDP builds and may rebuild its buckets in the first steps ---
    warm_ctx = model
    run_window(warm_ctx, raw_model, opt, batches, a.warmup,
               sync_grads=True, serial=(a.mode == "serial"),
               autocast_ctx=autocast_ctx, nvtx=False, world=world)

    if a.calibrate:
        per, host_ms, losses = run_window(
            model, raw_model, opt, batches, max(5, a.steps // 3),
            sync_grads=True, serial=(a.mode == "serial"),
            autocast_ctx=autocast_ctx, nvtx=False, world=world)
        if rank == 0:
            print(f"# CALIBRATION step={statistics.median(per['step']):.2f} ms "
                  f"fwd={statistics.median(per['fwd']):.2f} "
                  f"bwd={statistics.median(per['bwd']):.2f} "
                  f"opt={statistics.median(per['opt']):.2f} "
                  f"wall={host_ms:.2f} ms "
                  f"tok/s={tokens_per_step / (host_ms / 1e3):,.0f} "
                  f"mem={peak_mem_gib():.2f} GiB")
        if world > 1:
            dist.destroy_process_group()
        return 0

    # ---- correctness, once, before any performance number is accepted -------
    per, _, losses = run_window(model, raw_model, opt, batches, 3,
                                sync_grads=True, serial=(a.mode == "serial"),
                                autocast_ctx=autocast_ctx, nvtx=False, world=world)
    loss_finite = all(torch.isfinite(torch.tensor(x)) for x in losses)
    # zero_grad(set_to_none=True) ran last, so re-populate grads to inspect them
    idx, tgt = batches[0]
    with autocast_ctx():
        probe = model(idx, tgt)
    probe.backward()
    g_finite = grads_finite(raw_model)
    opt.zero_grad(set_to_none=True)
    sync_err = check_param_sync(raw_model, rank, world)
    if rank == 0:
        print(f"# CORRECTNESS loss_finite={loss_finite} grads_finite={g_finite} "
              f"param_sync_max_abs_diff={sync_err:.3e} steps_completed={a.warmup + 3}")
    if not (loss_finite and g_finite):
        if rank == 0:
            print("# ABORT: correctness check failed; no performance data emitted")
        if world > 1:
            dist.destroy_process_group()
        return 1

    # ---- DDP's own view of its buckets -------------------------------------
    if isinstance(model, DDP) and rank == 0:
        try:
            d = model._get_ddp_logging_data()
            keep = {k: v for k, v in d.items() if k in (
                "bucket_cap_bytes", "bucket_sizes", "num_buckets",
                "total_parameter_size_bytes", "num_parameter_tensors",
                "rebuilt_bucket_sizes", "avg_forward_compute_time",
                "avg_backward_compute_time", "avg_backward_comm_time",
                "avg_backward_compute_comm_overlap_time")}
            print("# DDP_LOGGING " + json.dumps(keep, sort_keys=True, default=str))
        except Exception as e:  # noqa: BLE001 - diagnostic only, never fatal
            print(f"# DDP_LOGGING unavailable: {type(e).__name__}: {e}")

    # ---- measured repeats ---------------------------------------------------
    if rank == 0:
        print(",".join(CSV_FIELDS))
    step_medians = []
    for rep in range(a.repeats):
        per, host_ms, losses = run_window(
            model, raw_model, opt, batches, a.steps,
            sync_grads=True, serial=(a.mode == "serial"),
            autocast_ctx=autocast_ctx, nvtx=a.nvtx, world=world)
        med = {k: statistics.median(v) for k, v in per.items()}
        step_medians.append(med["step"])
        if rank == 0:
            print(",".join(str(x) for x in [
                "train", a.mode, label,
                f"{bucket_mb:g}" if bucket_mb else "",
                world, rep,
                f"{med['step']:.4f}", f"{med['fwd']:.4f}",
                f"{med['bwd']:.4f}", f"{med['opt']:.4f}",
                f"{host_ms:.4f}",
                f"{tokens_per_step / (host_ms / 1e3):.2f}",
                grad_bytes, n_params, f"{losses[-1]:.6f}",
            ]))

    # ---- optional: the same backward with the reducer suppressed ------------
    if a.nosync_probe and isinstance(model, DDP):
        per, host_ms, _ = run_window(
            model, raw_model, opt, batches, a.steps,
            sync_grads=False, serial=False,
            autocast_ctx=autocast_ctx, nvtx=False, world=world)
        med = {k: statistics.median(v) for k, v in per.items()}
        if rank == 0:
            print(",".join(str(x) for x in [
                "train", "nosync", f"{label}-nosync",
                f"{bucket_mb:g}" if bucket_mb else "",
                world, 0,
                f"{med['step']:.4f}", f"{med['fwd']:.4f}",
                f"{med['bwd']:.4f}", f"{med['opt']:.4f}",
                f"{host_ms:.4f}",
                f"{tokens_per_step / (host_ms / 1e3):.2f}",
                grad_bytes, n_params, "",
            ]))
        # the probe let the ranks drift; put them back before anything else
        resync_params(raw_model, world)
        if rank == 0:
            print(f"# NOSYNC_PROBE done; params re-broadcast from rank 0")

    if world > 1:
        worst = torch.tensor([max(step_medians)], device=device)
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"# SLOWEST_RANK_STEP_MS {float(worst):.4f}")
        print(f"# PEAK_MEM_GIB rank{rank} {peak_mem_gib():.3f}", flush=True)
        dist.destroy_process_group()
    else:
        print(f"# PEAK_MEM_GIB rank0 {peak_mem_gib():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
