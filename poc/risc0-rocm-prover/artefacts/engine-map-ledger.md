# RISC0 rv32im segment-STARK — engine → role → evidence ledger

The per-engine map for the **scoped v2.3.2 fork** hybrid prover on Strix Halo
(`gfx1151`, ROCm 7.2.3), with every role tied to a committed evidence file. This
fills the plan's remaining evidence gap: the **quantified CPU witgen/accum share**
(A1). Distinction preserved throughout: **stock `r0vm 2.3.2` is CPU-only**; this map
describes the **fork built with `--features rocm`**.

## Phase → engine → wall share → evidence (measured 2026-07-20, solo)

Per-phase wall from A1 (CPU timers + rocprofv3 cross-check), as a share of the
25 687.9 ms segment-prove (Demo B ~4-seg Cartesi-step, poseidon2):

| STARK phase | engine (this fork) | wall % of prove | evidence file |
| --- | --- | ---: | --- |
| `eval_check` (26k-LOC gen. poly) | **iGPU** | **29.2 %** | [`phase-breakdown.csv`](phase-breakdown.csv), [`stage3-gate.md`](stage3-gate.md) |
| accum (grand-product) | **CPU**-delegated¹ | **20.0 %** | [`phase-breakdown.csv`](phase-breakdown.csv) |
| NTT (expand/eval/interp/bit-rev) | **iGPU** | **20.2 %** | [`phase-breakdown.csv`](phase-breakdown.csv), [`stage2-gate.md`](stage2-gate.md) |
| witgen (`steps.cu`) | **CPU**-delegated¹ | **8.8 %** | [`phase-breakdown.csv`](phase-breakdown.csv) |
| Merkle commit (poseidon2) | **iGPU** | **8.0 %** | [`phase-breakdown.csv`](phase-breakdown.csv), [`stage3-gate.md`](stage3-gate.md) |
| combos (`poly_divide` DEEP) | **CPU** (default) | 1.6 % | [`phase-breakdown.csv`](phase-breakdown.csv) |
| FRI + DEEP query | **iGPU** | 1.2 % | [`phase-breakdown.csv`](phase-breakdown.csv), [`stage2-gate.md`](stage2-gate.md) |
| Fiat-Shamir / IOP / alloc glue | **CPU** | 9.6 % | [`phase-breakdown.csv`](phase-breakdown.csv) |
| recursion (lift/join/succinct) + Groth16 wrap | separate stages, **now on the iGPU** (hybrid; out of this segment-prove breakdown) | — | [`recursion-validation.md`](recursion-validation.md), [`groth16-seam-validation.md`](groth16-seam-validation.md) |

**iGPU total ≈ 59.4 % of the prove (rocprofv3 15 266.8 ms); CPU ≈ 40.6 %.**

¹ **witgen + accum = 28.87 % of the prove** — the CPU-delegated share and the Part B
target. **Part B status:** `steps.cu` (30310 LOC) now **compiles to a native-HIP
witgen/accum** (`kernels/hip/steps_ffi.hip`, exit 0, 8.6 MB obj) — the field ABI is
native `fp.h` (portable, **not** sppark), so it hipifies by the same field-shim route
as `eval_check`. It is **wired but not yet bit-for-bit correct** at runtime (see
[`partb-status.md`](partb-status.md)); the shipped hybrid keeps witgen+accum on CPU.

## Correctness + capability evidence (the hard guarantees)

| claim | evidence file / command |
| --- | --- |
| every HAL op **bit-for-bit == CpuHal** (DualHal 15/15) | `cargo test -p risc0-zkp --features rocm hal::hip` ([`stage4-gate.md`](stage4-gate.md)) |
| `eval_check` GPU == CPU-C++ golden | [`stage3-gate.md`](stage3-gate.md) |
| GPU-produced seal accepted by **stock** verifier | `step.rocm.proof.bin` + `cargo risczero verify` → `✅ Receipt is valid!` |
| GPU path real (not CPU fallback) | 4 HipHal markers + 95 % iGPU busy (audit differential) ([`stage4-gate.md`](stage4-gate.md)) |
| `steps.cu` witgen mechanically portable to HIP | [`b-raw-steps-compile.log`](b-raw-steps-compile.log) (compiles clean) |

## Speed + memory evidence (scoped secondary)

| metric | value | evidence |
| --- | --- | --- |
| iGPU vs same-code 32t CPU (po2=20) | **5.46×** (26.1 s vs 142.4 s) | [`a3-speed.csv`](a3-speed.csv) |
| fork-CPU vs installed rzup r0vm | 1.25× (codegen gap; explains prior ~6.8×) | [`a3-speed.csv`](a3-speed.csv) |
| segment-po2 sweep | flat ~5.3–5.5× (po2 16→21) | [`stage4-sweep.csv`](stage4-sweep.csv), [`stage4-roofline.md`](stage4-roofline.md) |
| hashfn axis | poseidon2-only (sha-256 rejected) | [`a3-speed.md`](a3-speed.md) |
| unified-memory handoff | **0 hipMemcpy** (vs ≤0.68 % PCIe tax counterfactual) | [`handoff-cost.csv`](handoff-cost.csv) |
| unified-memory capacity | peak RSS 9.8 GB (po2=20) → 18.9 GB (po2=21), past 16 GB wall | [`stage4-roofline.md`](stage4-roofline.md) |

## Amdahl framing for the CPU share (why the hybrid is the sweet spot)

Moving witgen+accum (28.87 %) to the iGPU has an **Amdahl ceiling of ~1.41×** on the
prove (~1.40× end-to-end) — and accum (the larger 20 % half) is a **sequential
grand-product**, the least GPU-friendly phase. So the hybrid's CPU/GPU split is a
deliberate, measured optimum, not a limitation: see [`phase-breakdown.md`](phase-breakdown.md).
