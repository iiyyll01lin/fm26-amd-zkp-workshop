# Part B — HIP witgen + accumulation: a measured, deliberate CPU delegation

**Status (2026-08-29): the accepted prover keeps the default CPU-witgen /
CPU-accum hybrid, and that is a design decision taken against measurement, not
a blocker.** Both HIP paths exist, both pass their correctness gates on both
validated architectures, and both are default-off because on the architecture
where the A/B was run with energy and per-receipt verification (`gfx1201`)
turning them on does not pay. An older `gfx1151` suite disagrees on the sign;
that is recorded below and is **not** resolved.

## What upstream CUDA does, and where we deliberately differ

This is the single most misreadable thing about the port, so it is stated first.

Upstream risc0 v2.3.2 runs **witgen and accumulation on the GPU** in its CUDA
prover — there is no CPU delegation and no switch:

- `vendor/risc0/risc0/circuit/rv32im/src/prove/hal/cuda.rs:100` calls
  `risc0_circuit_rv32im_cuda_witgen(...)` unconditionally.
- The same file, `:155`, calls `risc0_circuit_rv32im_cuda_accum(...)`
  unconditionally.
- The kernels are real device work in
  `vendor/risc0/risc0/circuit/rv32im-sys/kernels/cuda/ffi.cu`: witgen is
  `par_stepExec<<<...>>>` (`:448`, `:453`), accumulation is
  `stepAccum<<<...>>>` (`:484`) → `thrust::inclusive_scan(thrust::device, ...)`
  (`:495`) → `finalizeAccum<<<...>>>` (`:502`).

Our HIP backend **departs from that on purpose**:

- `rocm-port/files/risc0/circuit/rv32im/src/prove/hal/hip.rs:75` reads
  `RISC0_ROCM_WITGEN` and, when it is unset, calls
  `risc0_circuit_rv32im_cpu_witgen` at `:110` instead of the HIP kernel.
- The same file, `:125`, reads `RISC0_ROCM_ACCUM` and defaults to
  `risc0_circuit_rv32im_cpu_accum` at `:172`.
- `rocm_runtime_switch` (`hip.rs:56-65`) returns **`false` when the variable is
  absent**, so both switches are off unless a run opts in.

So "witgen + accum on the CPU" is **not** parity with the reference
implementation and must never be described as such. It is a divergence, and the
two reasons for it are below — one an estimate, one a measurement.

## Reason 1 (estimate): the Amdahl ceiling is 1.41×

The measured CPU-side witgen + accumulation share is **28.87%** of the prove
([`phase-breakdown.md`](phase-breakdown.md)), which caps a *perfect* port at
**~1.41×**. Accumulation, the larger half at 20.0%, is a sequential grand
product — the least GPU-friendly phase in the pipeline.

## Reason 2 (measurement): on gfx1201, turning them on is wall-neutral and costs energy

`witgen-gate-gfx1201-20260731T024319Z/`
is the direct A/B the ceiling argument was standing in for. Discrete `gfx1201`,
the Demo B Cartesi step ELF, **6 runs per arm × 4 arms = 24 runs**, rep 0 in each
arm discarded as warm-up so the medians below are over **5 timed reps**, the card
pinned by bus id and asserted idle before each run, and **every one of the 24
receipts re-verified with stock `cargo risczero verify`**:

| witgen | accum | median wall | vs default | median energy |
|---|---|---:|---:|---:|
| CPU | CPU (shipped default) | 26.861 s | 1.000× | 5,579.2 J |
| CPU | GPU | 26.956 s | 0.996× | 6,303.8 J |
| GPU | CPU | 27.389 s | 0.981× | 6,437.6 J |
| GPU | GPU | 27.314 s | 0.983× | 6,440.4 J |

Read that as: **no arm moves the wall by more than 2%, and every arm that puts
work on the GPU burns 13–15% more energy** (+13.0% / +15.4% / +15.4% against the
default). The energy column is the card's `power1_average` integrated over the
run — socket-wide on this part, so it is a same-basis comparison between arms
and not a GPU-only or whole-system figure
([`run-witgen-gate-sweep.sh:64-65`](../scripts/run-witgen-gate-sweep.sh)).

This is strictly stronger evidence than the Amdahl estimate: it is a measured
"moving it does not pay", not a projected one.

## The blocker this file used to describe was fixed on 2026-07-22

Earlier revisions of this file said the GPU witgen was *"blocked at runtime on a
parallel cross-cycle dependency"* needing a dependency-aware wavefront
scheduler. **That diagnosis was withdrawn by the author on 2026-07-22 and this
file's later reinstatement of it was a documentation regression.** The trail:

1. `5507680` (2026-07-22 03:50:35) — *"enable independent HIP witgen and
   accum"*. The real bug was a device-wide `INVALID` fill inside
   `steps_ffi.hip` that ran **after** Rust had already scattered cycle/PC and
   ECall/Poseidon/SHA/BigInt state into `data`, destroying valid inputs.
   Initialization moved to `HipHal::alloc_elem_init`, before any scatter. The
   fix is still visible in the source:
   `steps_ffi.hip:11-14`
   — *"data/global/accum are initialized by `HipHal::alloc_elem_init` before
   Rust scatters preflight state into them. Never fill those complete buffers
   here."*
2. `0983a14` (2026-07-22 03:51:28) — this file was rewritten to **PASS on real
   gfx1151**, with ordered gates: `rocm_seqforward_cpu_trace_raw_words` (raw
   Montgomery words equal), parallel-HIP vs sequential-CPU golden on basic /
   loop / guest-SHA / Poseidon-paging / BigInt2 `modmul_256`,
   `rocm_accum_switch_matrix` across all four arms, DualHal 15/15, an all-HIP
   receipt accepted by the stock verifier, and a clean-room replay. The tests
   are committed at
   `witgen/tests.rs:257-310`.
   That revision stated plainly: *"There is no remaining evidence of a
   scheduling race, so no scheduling rewrite was made."*
3. `b782c51` (2026-07-22 08:46:22) — reinstated the pre-fix "blocked" text while
   demoting the all-HIP path to research-only. Scoping it out of release
   acceptance is defensible; **restating the withdrawn technical diagnosis as a
   live blocker is not**, and that is what propagated into the reading notes and
   the deck.

No commit has touched the overlay since `b4880f5` (2026-07-23), and the live
`vendor/risc0` tree is byte-identical to
`rocm-port/files/` for both `hip.rs` and `steps_ffi.hip`
— so the code that passed on 2026-07-22 is the code the 2026-07-31 sweep ran.
The independent confirmation is the sweep itself: **12 of its 24 runs have
`witgen=1` and every one of them produced a stock-verifier-accepted receipt.**

One measure of how detached the stale text had become:
[`run-witgen-gate-sweep.sh:5-8`](../scripts/run-witgen-gate-sweep.sh), written by
the same author nine days later, describes the switches as merely unmeasured —
*"Nobody had measured what turning them on costs or saves, so the Amdahl ceiling
of the whole port was unknown"* — with no hint that a blocker was still on the
books. The person running the experiment did not believe the document.

## Unresolved: gfx1151 and gfx1201 disagree on the sign, not on correctness

`benchmark-suite-20260722/summary.md`
(generated 2026-07-22T11:12:33Z on the `gfx1151` APU host, base `44c5183`, which
already contains the `5507680` fix) ran the same four arms across ten workloads
and found the opposite of wall-neutral. At `cartesi p20/m1`: hybrid **15.145 s**,
hip-witgen **14.525 s**, hip-accum **12.787 s**, all-hip **12.204 s** — all-HIP
**1.24× faster** than the hybrid, and the file's own summary attributes
**1.15–1.21×** to HIP accum and **1.01–1.05×** to HIP witgen. Every row passed.

The two sweeps are not reconciled here, and the cause is **not established**.
They differ in host (32-thread Zen 5 APU vs a 128-core WRX90 box), in GPU
(20-CU integrated gfx1151 vs discrete gfx1201), in workload parameterisation,
and in how the CPU-side C++ golden was compiled (the suite pins
`portable-generic` for its ROCm arms; the sweep used a separately built
`r0vm`). Any of those could carry the sign flip, and picking one without
measuring would be a guess. What both agree on is the part that matters for
this file: **the GPU witgen and accum paths are correct on both architectures**.

Consequences for how the default is justified:

- On **gfx1201** the default is right on the measurement: same wall, less
  energy.
- On **gfx1151** the default may be leaving ~1.2× on the table, and the honest
  statement is that the paired A/B which would settle it has not been re-run
  since 2026-07-22.
- The Amdahl ceiling (~1.41×) bounds both and is contradicted by neither.

## Runtime switches

- both unset: CPU C++ witness, CPU C++ accumulation — **accepted default**
- `RISC0_ROCM_WITGEN=1`: HIP witness path
- `RISC0_ROCM_ACCUM=1`: HIP accumulation path

Values `0`, `false`, `no`, `off`, and the empty string are disabled
(`hip.rs:56-65`). The release and clean-room gates explicitly unset both
(`scripts/run-clean-room.sh:559`, `:568`, `:578`, `:715`), so every published
acceptance number is a default-hybrid number.

## Durable correctness boundary

- `HipHal` STARK kernels and rv32im `eval_check` run on the selected AMD GPU.
- Witgen and accumulation are **CPU-delegated by default** over managed memory,
  diverging from upstream CUDA for the reasons above.
- DualHal 15/15 and `eval_check` GPU==CPU are exact equality gates.
- The end-to-end receipt must be non-empty, stock-verifier accepted, carry
  HipHal plus CPU-delegation markers, and overlap non-zero GPU telemetry.
- This default hybrid passed clean-room validation on `gfx1151` and `gfx1201`.

The all-HIP arms remain outside workshop, release, and upstream acceptance
criteria — because on the architecture with the strongest evidence they buy
nothing, and because the one result that says otherwise has not been reproduced.
**Not because they fail.**
