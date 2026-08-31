# Portfolio material

Summary, resume bullets and an interview guide for the
**NCCL Communication Performance Lab**. Every number below is measured and
traceable to a report in `docs/experiments/` and raw output in `results/raw/`.

---

## 1. Project summary

**Problem.** In data-parallel training, gradient AllReduce competes with the
backward pass for the same GPU. Standard advice — overlap communication with
compute, tune DDP's bucket size — is widely repeated and rarely measured against
end-to-end step time on the hardware in front of you. I wanted to know, from
measurement, which parts of that advice survive contact with a real training
loop and which do not.

**Implementation.** Ten phases on rented single-node cloud GPUs (2–4 × L4 or
A40). NCCL collective baselines and a 2 vs 4 GPU topology study; an
algorithm × protocol sweep (Ring/Tree × Simple/LL/LL128); a Ring AllReduce
written from first principles as ReduceScatter + AllGather, validated against an
exact fp32 oracle; a communication/compute overlap benchmark with per-stream CUDA
events; and a compact 82.6 M-parameter GPT trained under PyTorch DDP. Results go
through a JSON Schema with an explicit `measured` / `estimated` / `theoretical`
interlock; 72 local tests cover the parsers.

**Profiling.** Nsight Systems with NVTX. Its most valuable output was negative:
it showed that a "~4.4 ms harness floor" I had attributed to my own benchmark was
a property of one *host* — the harness costs ≈ 27 µs — which invalidated two of
my earlier performance conclusions and revised three more. It also showed that
**91–93 % of the naive ring's runtime was device-wide barriers**, which turned a
vague "V2 is faster" into a mechanism.

**Optimization.** Removing those barriers gave **3.28–3.35×** at ≥ 1 MiB. In
real DDP, communication/compute overlap is worth **20–30 %** of step time against
an otherwise identical serialised reduction, and scaling efficiency reaches
**93.1 %** on 4 GPUs with per-GPU batch held constant.

**Real-training validation.** The microbenchmarks predicted the real gradient
AllReduce within **7.8 %** across different hosts and NCCL versions. Two results
did not survive: DDP's *requested* bucket capacity is not the collective size
(one indivisible 50 MiB parameter put a third of the gradient traffic out of the
knob's reach), and the bucket optimization the microbenchmarks implied is worth
only **0.48 %** of step time on a host with a healthy P2P transport — **below
run-to-run noise**. What did reproduce everywhere is the negative control: 64 MiB
buckets produce the fewest, largest, individually fastest collectives and **22 %
less total NCCL kernel time**, and are **3.6–6.1 % slower to train**.

The engineering point of the project is that last sentence: isolated
communication efficiency is not the optimization objective, and the only way to
know that is to measure end-to-end step time.

---

## 2. Resume bullets

> Pick 3; the fourth is the reliability angle, which matters most for
> infrastructure roles.

- **Implemented a Ring AllReduce from first principles** (ReduceScatter +
  AllGather) in CUDA/NCCL, validated against an exact fp32 oracle and confirmed
  empirically to move `2(N−1)/N·M` bytes per rank; **profiled with Nsight Systems
  to find that 91–93 % of runtime was device-wide barriers, and removing them
  gave a 3.3× speedup at ≥ 1 MiB** on 4 GPUs.

- **Measured communication/compute overlap in real PyTorch DDP training**
  (82.6 M-parameter GPT, 4 GPUs, BF16, 315 MiB of gradients per step) using
  per-stream CUDA events and Nsight timelines: **overlap is worth 20–30 % of
  end-to-end step time** against an identical serialised reduction, reaching
  **93.1 % scaling efficiency**, while measurably slowing the backward pass —
  quantifying the interference rather than assuming overlap is free.

- **Ran a controlled bucket-size study that produced a decisive negative
  result**: the configuration with the fewest, largest and individually fastest
  collectives — **22 % less total NCCL kernel time per step** — was **3.6–6.1 %
  slower to train**, because gradients became communicable later and the exposed
  communication tail grew from 6.5 ms to 13.2 ms. Established that isolated
  collective efficiency is not the training optimization objective.

- **Built a functional transport preflight after three of four cloud hosts
  silently failed a P2P path that `cudaDeviceCanAccessPeer` reported as
  available** (one corrupted data, two deadlocked). The gate runs a real
  collective under a hang timeout and requires cross-rank parameter identity
  (`max |p_rank − p_rank0| == 0`) before any benchmark is allowed to run,
  turning a class of silent-wrong-result failures into a fast, loud one.

---

## 3. Interview guide

### 30 seconds

> I built a communication performance lab for distributed training. It goes from
> NCCL collective baselines, through a Ring AllReduce I wrote and profiled
> myself, to communication/compute overlap measured inside a real PyTorch DDP
> training loop. The headline is a negative result: the DDP configuration that
> produced the most efficient collectives — 22 % less total NCCL kernel time —
> was the slowest to actually train, because the communication started later and
> left a bigger exposed tail. It's a concrete demonstration that end-to-end step
> time, not collective bandwidth, is the thing to optimize.

### 3 minutes — baseline → bottleneck → implementation → profiling → DDP result

**Baseline.** I started with NCCL collectives across 8 B to 128 MiB on 2 and 4
GPUs, under a correctness gate: a result whose validation didn't pass isn't a
result. Going 2→4 GPUs, small-message latency grew ×1.9 — matching a tree, not
the ×3 a ring would predict — and large messages kept only ~28 % of their bus
bandwidth, because the 4-GPU ring crosses a PCIe host bridge twice. Topology, not
rank count, was governing performance.

**Implementation.** To understand what NCCL is doing, I wrote a Ring AllReduce
myself: ReduceScatter then AllGather, checked against an exact fp32 oracle. Two
things came out of that. First, a write-after-read race in my double-buffering,
which I found because the oracle caught nondeterministic mismatches. Second, and
more interesting: `cudaDeviceCanAccessPeer` said peer access was available, and
peer copies returned NaN. So the benchmark now runs a *functional* peer test and
refuses to use a path that doesn't work.

**Profiling.** My first ring was slow and I didn't know why. Nsight Systems
showed 91–93 % of its runtime was `cudaDeviceSynchronize` barriers. Replacing
them with per-peer events gave 3.3× at ≥ 1 MiB. The same profiling pass also
caught a mistake of my own: I'd reported a "4.4 ms harness floor" as a property
of my benchmark. It was a property of that one host — the harness costs about
27 µs. That invalidated two of my earlier conclusions, and I wrote the
correction rather than quietly fixing the number.

**DDP result.** Then I asked whether any of this survives in real training: a
compact GPT under DDP on 4 GPUs. It does — a collective is resident for 50–83 %
of backward, and overlap is worth 20–30 % of step time. But two things didn't.
DDP's `bucket_cap_mb` isn't the collective size — it can't split a parameter, and
one 50 MiB output projection put a third of the gradient traffic beyond the
knob's reach. And the bucket size my microbenchmarks pointed at improved step
time by 0.48 %, which was below run-to-run noise on that host. The result that
*did* reproduce everywhere was the negative control at 64 MiB.

### 15 minutes — technical walkthrough

**NCCL algorithm and protocol.** NCCL picks an algorithm (Ring, Tree) and a
protocol (Simple, LL, LL128). I forced all combinations across message sizes.
Below ~4 KiB the protocol is worth 45–85 % and the algorithm is nearly
irrelevant — you're latency-bound and LL's inline flags avoid a synchronisation.
Above ~2 MiB it inverts: Tree vs Ring is worth up to 53 %, because you're
bandwidth-bound and the algorithm decides how many bytes cross the slowest link.
NCCL's automatic choice was good but missed systematically at large sizes. I
labelled that as *inferred from timings*, because I read it from timing
correlation, not from a log line stating the chosen algorithm.

**Ring ReduceScatter + AllGather.** Each of N ranks ends up with the full sum.
In the reduce-scatter half, over N−1 steps each rank forwards and accumulates a
1/N chunk, so it owns one fully reduced chunk; in the all-gather half, N−1 more
steps circulate those chunks. Each rank sends `(N−1)/N · M` in each half, so
`2(N−1)/N · M` total — which is exactly the bus-bandwidth correction factor. I
instrumented the byte counter and confirmed it at all five sizes; it's a nice
check because it links the implementation to the metric definition.

**The synchronisation race.** My V2 used parity double-buffering: step s writes
`stage[s & 1]` while step s−1 reads the other. That's a write-after-read hazard
across ranks — a fast neighbour can overwrite a buffer the slow one hasn't
finished reading. V1 was correct, V2 and V3 produced varying mismatch counts run
to run. Per-step staging buffers fixed it. The general lesson: the oracle catches
what a "it ran without errors" check doesn't.

**Measurement correction.** Phase 6 reported that every implementation including
NCCL bottomed out at ~4.4 ms, and I attributed it to fixed harness cost. In
Phase 7A I instrumented the harness with NVTX and measured it at ~27 µs, and the
same binary on another host gave 42 µs. So the floor followed the machine, not
the code. I published a reconciliation table marking each earlier conclusion
STILL VALID, REVISED or INVALIDATED, and left the original report intact. Two
were invalidated — including a V3 pipelining gain that simply didn't reproduce.

**Overlap and interference.** I measured compute and communication on separate
streams with CUDA events recorded *on each stream*, buffered and read only after
the window closed so no synchronisation entered the measured region. Overlap is
real, and the compute stream is slowed in every case — 1.03–2.09×. Then I asked
which resource they fight over. The natural guess is DRAM bandwidth. It isn't:
the collective's own device-memory traffic is bounded at ~2 % of achievable
bandwidth, and the workload consuming ~460 GB/s disturbed the collective *less*
than one consuming almost none. What the traces show is that per-kernel median
durations barely move while tails inflate up to 13× — delayed scheduling, not a
uniform slowdown. I stated that as *consistent with* execution-resource
contention rather than proven, because Nsight Compute's counters weren't
available to me as a container tenant, and because kernel granularity and
arithmetic intensity covary in my design.

**DDP bucket behaviour.** DDP groups gradients into buckets and fires an
AllReduce when a bucket fills. Small buckets communicate earlier but pay more
per-collective overhead; large buckets are individually more efficient but become
ready later. I measured both arms. From the traces, collective count exactly
matched DDP's reported bucket count (26 / 10 / 4 per step), the first AllReduce
started at −5 % to +39 % of backward depending on capacity, and the exposed tail
ran 6.5 to 22.5 ms.

**Exposed tail.** I define it as the gradient communication that hasn't finished
when backward compute has. Measuring it needs care: `loss.backward()` in DDP
already joins the communication stream, so its duration includes the tail. I got
the compute-only backward by re-running the identical loop inside
`model.no_sync()`, which suppresses the reducer entirely, and differenced. That
difference bundles the tail with the interference from the previous paragraph, so
I called it "synchronisation cost", never "the tail", and separated the two with
the trace — they differ by 3–4 ms in the expected direction.

**Transport-dependent optimization.** On an SHM-only host, small buckets
measurably help: there's a lot of exposed tail to reclaim. On a host where NCCL's
P2P path actually worked, the collective is fast enough that 4 MiB and 25 MiB
start communicating at the same point and the difference falls below noise. Same
code, same model, same measurement — different transport, different answer. So I
report the bucket recommendation with its transport attached, and the only
universal claim I make is the negative one about very large buckets.

### Likely follow-up questions

| Question | Short answer |
|---|---|
| *How do you know overlap actually happened rather than just being enqueued?* | Nsight timeline: NCCL kernel intervals intersected with the backward NVTX range, 50–83 % residency. Configuration alone proves nothing. |
| *Why is `sync_cost` not just the tail?* | Because a resident collective also slows backward. The trace-derived tail is 3–4 ms below the event-derived cost, and that gap is the interference. |
| *Why not just trust `bucket_cap_mb`?* | DDP can't split a parameter. With a 16 K vocabulary the output projection alone is a 50 MiB bucket, so no cap below ~48 MiB affects a third of the traffic. Read `_get_ddp_logging_data()['rebuilt_bucket_sizes']`. |
| *You claim 0.48 % — is that real?* | No, and I say so. The noise floor from independent relaunches was 0.81 ms and the gap was 0.73 ms. The reportable result is the 64 MiB penalty at 3.6–6.1 %. |
| *Why did you rule out DRAM instead of proving the mechanism?* | Counters were unavailable (`ERR_NVGPUCTRPERM`). I could bound the collective's memory traffic arithmetically and contrast workload classes, which is enough to falsify the bandwidth hypothesis but not to confirm a specific scheduler mechanism. |
| *What would you do differently with more budget?* | Kernel-granularity isolation — the one confound I couldn't remove — and repeating the bucket study on NVLink, where I predict the optimization shrinks further while the large-bucket cliff persists. |
| *How did the P2P failures actually present?* | One host: `canAccessPeer` true, peer copies returned NaN, caught by the oracle. Two hosts: NCCL logged "Connected all rings", then the first collective never returned, GPUs at 100 %. Caught by a timeout, not by an error code. |
| *Is your 3.3× a real speedup or a fix to a bad baseline?* | Both, and I'd describe it that way. The baseline was a straightforward correct implementation; profiling showed most of its time was barriers I had put there. It's the honest number for "what does removing global synchronisation buy", not a claim of beating NCCL. |

---

## 4. One-paragraph version

> **NCCL Communication Performance Lab** — a ten-phase distributed-training
> performance study on 2–4 GPU cloud nodes. Benchmarked NCCL collectives and
> algorithm/protocol behaviour; implemented and profiled a Ring AllReduce from
> first principles, finding via Nsight Systems that 91–93 % of its runtime was
> device-wide barriers and gaining 3.3× by removing them; measured
> communication/compute overlap and interference with per-stream CUDA events; and
> validated the findings in real PyTorch DDP training, where overlap is worth
> 20–30 % of step time at 93.1 % scaling efficiency. The central result is
> negative: the bucket configuration producing 22 % less total NCCL kernel time
> trained 3.6–6.1 % slower, because communication started later and left a larger
> exposed tail. All 2 469 result rows are schema-validated and regenerate
> byte-for-byte from committed raw output.
