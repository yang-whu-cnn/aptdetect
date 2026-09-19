# Formal v3 execution readiness

## Current machine audit (2026-09-19)

The machine has an NVIDIA GeForce RTX 4070 Laptop GPU with 8,188 MiB VRAM and a
610.62 driver. The dedicated `.venv_cc4` starts successfully with Python 3.11.9
and PyTorch 2.14.0+cu130; `torch.cuda.is_available()` is true. This is the sole
runtime used for CC4 execution. A previous restricted subagent result that
reported a CPU-only runtime was a sandbox-accessibility error and is superseded
by the host-verified manifest captured by the 500-tick DCA benchmark.

## Method split and bottlenecks

- DCA is CPU work. Its CC4 environment loop and five-agent observation handling
  dominate; moving its small clustering/tabular calculations to GPU would add
  overhead. It currently has a development pilot, not a formal raw-metric runner.
- RSMBRL accepts both devices, but the projection-aware matched benchmark shows
  CPU median 0.09636 s/call versus CUDA median 0.23656 s/call (CUDA is about
  2.46x slower for this small batched workload). A local CPU call
  with population 200, five iterations and H=4 took about 0.10 seconds. The earlier
  2x5x100 pilot spent 276.57 seconds planning for 3,978 decisions. Scaling that
  pilot from 1,000 to 250,000 environment ticks gives roughly 19.2 hours of CPU
  planning and 1.3 hours of other episode work. Scaling the isolated 0.103-second
  call by the same decision density gives about 28.5 hours, so a prudent CPU
  estimate is 20-30 hours. These are extrapolations, not measured formal runtimes. Benchmark CUDA after
  repairing the environment before scheduling all repeats.
- PriorRL's small PPO network benefits modestly from GPU training, while CC4
  collection stays CPU-bound. Its current `forward` creates an input tensor
  without the model device, so a CUDA model with a CPU/numpy state will fail.
  More importantly, exact offline cache coverage is the formal blocker; no
  training should start until scheme B produces and validates the frozen cache.

## Fastest protocol-complete order after scheme B

1. Use the verified CUDA-enabled `.venv_cc4`; retain the recorded Python,
   PyTorch, CUDA, driver, and CPU/CUDA matched benchmark reports. Route PPO to
   CUDA and RSMBRL planning to CPU.
2. Finish scheme B cache generation and its hash/schema/coverage gate. In
   parallel, build formal raw-event runners for DCA and RSMBRL against W0's
   `episodes.jsonl` schema. The current `run_formal_suite` validates formal
   outputs but does not execute 5x100x500 runs.
3. Run DCA on CPU first because it needs no training and can expose environment
   or evaluator defects cheaply. Run one formal-shaped dry episode, validate it,
   then schedule its five repeats.
4. Run RSMBRL planning on CPU; the matched projection-aware benchmark rejects
   CUDA for this workload. Parallelize at the repeat/episode level while keeping
   each episode's environment transitions sequential.
5. Fix PriorRL state-device placement, train five independent PPO seeds on GPU,
   validate each checkpoint using validation only, then evaluate shared test
   seeds. Cache miss remains fail closed and online LLM remains forbidden.
6. Validate every repeat directory independently, confirm all five frozen
   repeat/seed identities, and only then invoke the five-repeat aggregator.

## Blocking gaps

- The CC4 CUDA runtime is verified. GPU availability is no longer a blocker.
- The unified suite has formal validation but no formal execution command or
  method adapter that writes W0 raw incident/LWF/recovery events for these three
  baselines.
- DCA and RSMBRL pilots are development-only summaries and cannot be promoted.
- PriorRL cache coverage and CUDA input-device placement block training.
- Runtime estimates for DCA and PriorRL require a formal-shaped one-episode
  benchmark after their runners exist; inventing estimates before that would be
  misleading.
