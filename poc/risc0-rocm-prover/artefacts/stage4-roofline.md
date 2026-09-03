# Stage 4 roofline — the single-point speedup becomes an honest where-wins/loses curve

> **2026-07-20 re-measure:** fresh po2=20 headline is **5.46×** (iGPU 26.1 s vs same-code
> 32t CPU 142.4 s), consistent with the flat ~5.3–5.5× curve below. The "~6.6–6.8×"
> resolves as 5.46× **×** a **1.25× codegen gap** vs the installed rzup r0vm (177.6 s) —
> see [`a3-speed.md`](a3-speed.md). Per-phase attribution (why the iGPU wins ~60 % of the
> prove) is in [`phase-breakdown.md`](phase-breakdown.md); the CPU↔GPU handoff is 0-copy
> ([`handoff-cost.md`](handoff-cost.md)).

**What this adds over `stage4-gate.md`:** the Stage-4 gate proved *one* point (a
~4-segment `po2=20` prove, iGPU vs 32t CPU). This sweep turns that single number
into a **defensible curve** across the two knobs the r0vm CLI can vary, under the
repo solo-guard, every row verified by the **stock** `cargo risczero verify`.
Source data: [`stage4-sweep.csv`](stage4-sweep.csv). Reproduce:
`R0VM_CPU=… R0VM_GPU=… SWEEP_PO2S="16 18 20 21" scripts/run-stage4-sweep.sh`.

## Method (honest by construction)

- **Same fork code, two builds:** `r0vm` built with `--features rocm` (iGPU HipHal:
  STARK math + `eval_check` on gfx1151, witgen/accum CPU-delegated over unified
  memory) vs no-rocm (32-thread Zen5). Identical guest ELF + session.
- **Segment-po2 axis:** the fork r0vm was given a tiny env knob
  `RISC0_SEGMENT_LIMIT_PO2` (wired to `ExecutorEnvBuilder::segment_limit_po2`; no
  effect unless set). It re-partitions the **same** Cartesi-step session into
  segments of `2^po2` cycles — po2=16 ⇒ ~56 tiny segments, po2=21 ⇒ ~2 big ones.
- **Solo discipline:** solo-guard runs **before every row** (iGPU-idle + loadavg),
  with a cooldown that waits out our *own* residual load/GPU-busy between rows, so
  no row is recorded under contention. GPU=min-of-2, CPU=min-of-1. The guard
  **refused to record twice** during development (loadavg 26 after a CPU run; iGPU
  36% right after a GPU run) — it never fabricates.

## The curve (2026-07-20, gfx1151, ROCm 7.2.3, solo)

| segment po2 | ~segments | CPU wall (32t) | iGPU wall | **speedup** | iGPU peak RSS | receipt |
| ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| 16 | ~56 | 340.2 s | 64.2 s | **5.30×** | 0.83 GB | 30.9 MB |
| 18 | ~14 | 157.4 s | 30.0 s | **5.25×** | 2.6 GB | 4.1 MB |
| 20 (default) | ~4 | 144.2 s | 26.1 s | **5.52×** | 9.5 GB | 1.1 MB |
| 21 | ~2 | 168.7 s | 30.6 s | **5.51×** | 18.9 GB | 0.6 MB |

Every row: `verify=valid` (stock verifier), GPU rows carry HipHal markers
(140/16/4/2 ≈ per-segment, proving the iGPU path, not a CPU fallback).

## What the curve says (and does not)

1. **Flat ~5.3–5.5×, no crossover.** The hybrid iGPU prover wins at **every**
   segment size from po2=16→21 by a similar margin. It does **not** reverse at
   small segments, nor spike at large ones — there is no CPU-win region in range.
2. **The GPU wall is stable (~26–31 s); the CPU baseline carries the variance.**
   po2=16 is slower on **both** sides (56-segment launch/overhead: 64 s GPU / 340 s
   CPU) but the *ratio* holds. For po2≥18 the iGPU is a rock-steady ~26–31 s.
3. **The "6.6×" folded in a 1.25× codegen gap, not random noise.** This sweep's
   **cleanest** solo CPU baseline (po2=20, loadavg **0.07**) is **144.2 s ⇒ 5.52×**,
   matching the fresh same-code **5.46×**. The earlier single point / independent
   audit divided by a slower **~178 s** baseline — but that is the **installed rzup
   r0vm's CI/generic codegen** (177.6 s) vs the local same-code fork build (142.4 s),
   a **systematic 1.25× local-vs-shipped codegen gap**, *not* CPU-baseline noise. The
   **iGPU wall is unchanged (~26 s)**, so `~6.6–6.8× = 5.46× × 1.25× codegen gap`; the
   honest same-code figure is **~5.5×**, and we say so. (This mirrors the Demo C
   correction discipline in `docs/INTEGRITY-REPORT.md`: report the fair same-code
   baseline, which *lowers* the GPU speedup.)
4. **hashfn axis collapses to poseidon2.** risc0 2.3.2's rv32im composite prover
   **rejects** `--hashfn sha-256` ("supported hashfn values are: poseidon2") on
   **both** CPU and GPU. So poseidon2 is not a *choice* here — it is the only
   supported hash, and it happens to be the CPU-expensive / GPU-friendly one. There
   is no hashfn crossover to measure; this **sharpens** rather than weakens the
   caveat.
5. **OOM-ceiling axis.** iGPU peak RSS ≈ doubles per po2 step
   (0.83→2.6→9.5→**18.9 GB**). At po2=21, **18.9 GB peak RSS crosses the 16 GB wall**
   a 16 GB discrete card / box cannot hold — the **94 GB unified LPDDR5X** carries
   it (po2=22 extrapolates to ~37 GB, still inside the 94 GB pool). This ties the
   speed story to the repo's unified-memory thesis.

## Verdict (unchanged hard guarantee; sharpened speed claim)

- **Correctness is the hard guarantee:** every row is bit-for-bit accepted by the
  stock `cargo risczero verify`; the HAL is `CpuHal == HipHal` (DualHal 15/15).
- **Speed is a scoped, honest curve:** **~5.3–5.5× across po2 16–21** on this
  ~4-segment poseidon2 Cartesi-step workload with the **hybrid** design
  (witgen/accum CPU; recursion + Groth16 wrap CPU). Workload-/poseidon2-/
  hybrid-specific and CPU-baseline-sensitive — **not** a general "iGPU is faster"
  claim. The **stock r0vm stays CPU-only.**
