# A1 — per-phase wall breakdown of the hybrid rv32im segment-STARK (gfx1151)

**What & why.** Before porting `steps.cu` (Part B), measure where the hybrid
prover's wall actually goes, so the **witgen+accum Amdahl ceiling** (the most Part B
could ever buy) is a number, not a guess. All figures are from real, **solo**
(`solo-guard`: iGPU 0 %, loadavg < 1) runs on the reconstructed known-good hybrid
build (Strix Halo `gfx1151`, ROCm 7.2.3, `RAYON_NUM_THREADS=32`, release).

**Workload.** The Demo B `~4-segment` Cartesi-step composite (`3×po2=20 + 1×po2=19`),
poseidon2 hashfn — the same session the Stage-4 gate + roofline use. The
GPU-produced seal for this exact run **stock-verifies** (`✅ Receipt is valid!`).

**Method (two independent instruments, cross-checked).**
1. **CPU wall timers** (`RISC0_PHASE_PROFILE=1`, env-gated, measurement-only — the
   seal is byte-identical with the flag off). A `std::time::Instant` wraps every
   `HipHal`/`HipCircuitHal` op. Because every HipHal FFI wrapper ends in
   `launch_done() → hipDeviceSynchronize()`, the CPU timer around a GPU op captures
   that op's **full GPU execution**, so the timers form a complete wall decomposition.
2. **`rocprofv3 --kernel-trace --memory-copy-trace`** (independent, 2026-07-20 solo):
   authoritative GPU-side kernel durations, aggregated by kernel.

## The breakdown (sum over the 4 segments; `prove_total = 25 687.9 ms`)

| phase | engine | wall ms | % of prove | rocprofv3 GPU ms | notes |
| --- | --- | ---: | ---: | ---: | --- |
| **eval_check** | **iGPU** | 7 502.7 | **29.2 %** | 7 452.6 | generated 26k-LOC rv32im check-poly |
| **accum** | **CPU** (delegated) | 5 149.1 | **20.0 %** | — | grand-product permutation (`cpu_accum`); GPU idle |
| **NTT** | **iGPU** | 5 199.8 | **20.2 %** | 5 205.0 | expand+evaluate+interpolate+bit-reverse |
| **witgen** | **CPU** (delegated) | 2 265.8 | **8.8 %** | — | witness generation (`cpu_witgen`); GPU idle |
| **Merkle (poseidon2)** | **iGPU** | 2 042.4 | **8.0 %** | 2 083.8 | `hash_rows`+`hash_fold` commit |
| combos | CPU (default) | 409.1 | 1.6 % | — | `poly_divide` DEEP combos (rayon) |
| other poly | iGPU | 342.5 | 1.3 % | 224.0 | zk_shift, scatter, eltwise |
| FRI+DEEP | iGPU | 307.5 | 1.2 % | 301.4 | fri_fold, mix_poly_coeffs, batch_evaluate_any |
| CPU glue | CPU | 2 469.0 | 9.6 % | — | Fiat-Shamir hashing + IOP + alloc + Merkle-tree glue (residual) |
| **prove_total** | mixed | **25 687.9** | 100 % | 15 266.8 | — |

- **End-to-end `r0vm` wall = 26 110 ms.** Σ(4 segment proves) = 25 687.9 ms ⇒
  execution/preflight + CLI setup is only **~422 ms (1.6 %)** — proving dominates.
- **GPU busy = 15 266.8 ms (rocprofv3) = 59.4 %** of the prove; CPU = 40.6 %. The
  two instruments agree per-phase to within launch/sync overhead
  (eval_check 7 502.7 vs 7 452.6; NTT 5 199.8 vs 5 205.0; Merkle 2 042.4 vs 2 083.8).

## The number Part B hinges on: witgen + accum

> **witgen + accum = 7 414.8 ms = 28.87 % of the prove (28.40 % of end-to-end).**

These two phases are exactly what Part B moves from CPU to the iGPU. While they run,
the **iGPU is idle** (they are CPU-delegated over unified memory); while the iGPU runs
the STARK math, the CPU is idle. The hybrid trades the machine back and forth.

**Amdahl ceiling for Part B** (the *best case*, if GPU witgen+accum were instant):

| basis | fraction | max speedup `1/(1-f)` |
| --- | ---: | ---: |
| of the prove | 0.2887 | **1.41×** |
| of end-to-end wall | 0.2840 | **1.40×** |

So even a *perfect* Part B cannot make the hybrid more than **~1.4× faster** — and
that is the unreachable ceiling, not a forecast.

**Why the realistic gain is smaller (and could be negative):**
- **accum (20.0 %) is bigger than witgen (8.8 %)**, and accum is a **sequential
  grand-product** (prefix-product data dependence). It is the *hardest* phase to
  parallelize on a GPU — the biggest chunk is the least GPU-friendly.
- Moving witgen+accum onto the **already-busy iGPU** removes the CPU/GPU division of
  labour that is the hybrid's entire point. Within a segment the phases are
  data-dependent (witgen → NTT/hash → accum → eval_check), so they cannot overlap;
  Part B mostly **relocates** ~7.4 s of work rather than hiding it.
- Rough envelope, moving witgen+accum to GPU at speedup *s* on that slice:
  `s=∞ → 1.41×`, `s=5× → 1.30×`, `s=2× → 1.17×`, `s=1× (same) → 1.00×`,
  `s=0.5× (GPU slower, plausible for the sequential accum) → 0.78× (slower)`.

**A1 verdict.** The hybrid already puts **~60 % of the prove on the iGPU** and the
two GPU-idle CPU phases are only **~29 %**, capped at a **~1.4× Amdahl ceiling** with
the larger half (accum) being GPU-hostile. This frames Part B as a **bounded,
likely-modest** win and pre-registers "B done but small/negative" as a valid honest
outcome that would validate the hybrid split as the sweet spot.

## Reproduce

```bash
# instrumented build (env-gated timers; seal identical with flag off):
( cd vendor/risc0 && RISC0_HIP_OFFLOAD_ARCH=gfx1151 cargo build -p risc0-r0vm --features rocm --release )
RISC0_PHASE_PROFILE=1 RISC0_PHASE_PROFILE_OUT=/tmp/phase.csv RAYON_NUM_THREADS=32 \
  target/release/r0vm --elf <demoB.elf> --initial-input <step.bin> --receipt /tmp/p.bin
# GPU-side cross-check:
rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d /tmp/rp -o prove -- \
  target/release/r0vm --elf <demoB.elf> --initial-input <step.bin> --receipt /tmp/p.bin
```

Raw evidence: `a1-raw/phase-prof-final.txt`,
`a1-raw/rocprofv3-kernel-agg.csv`,
`a1-raw/rocprofv3_kernel_trace.csv`.
