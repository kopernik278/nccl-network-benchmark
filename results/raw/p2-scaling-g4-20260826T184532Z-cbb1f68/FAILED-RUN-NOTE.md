# Failed run — preserved as evidence, not a result

This run failed its correctness gate and produced **no measurements**. It is kept
because the failure is itself a finding.

Cause: the container's **NCCL 2.25.1 does not support sm_120 (Blackwell)**. NCCL
initialised fully — bootstrap, topology detection and P2P channel setup all
succeeded — then failed at collective kernel launch:

    enqueue.cc:1500 NCCL WARN Cuda failure 1 'invalid argument'

Rebuilding nccl-tests with `-gencode=arch=compute_120,code=sm_120` did NOT fix
it, which correctly located the problem in NCCL rather than the test harness.

Fix: NCCL 2.31.2 installed into an isolated prefix, nccl-tests rebuilt against
it. The successful runs are the `...185443Z...` experiment IDs.

See `docs/experiments/p2-multigpu-scaling.md` section 4.
