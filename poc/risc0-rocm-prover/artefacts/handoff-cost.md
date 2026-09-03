# A2 — unified-memory CPU↔GPU handoff cost + the hipMemcpy/PCIe counterfactual

**What & why.** The hybrid runs witgen+accum on the **CPU** and the STARK math on the
**iGPU**, both over the *same* HIP **managed** (unified) buffers. This turns "the APU
makes the hybrid cheap" from a slogan into numbers: the per-segment CPU→GPU handoff
volume, the copy it **avoids**, and what a discrete GPU would pay to move the same data.
All measured **solo** on `gfx1151` (ROCm 7.2.3).

## What crosses the CPU→GPU boundary each segment

The CPU witgen writes `data`+`global`, the CPU accum writes `accum`; the iGPU STARK
then reads them (commit/NTT/hash/eval_check). With `Val = 4 B` and the rv32im v2
trace widths (`data=211`, `accum=103`, `code=1` columns):

| segment | data | accum | code | **boundary total** |
| --- | ---: | ---: | ---: | ---: |
| po2=20 (`2²⁰` cyc) | 844 MB | 412 MB | 4 MB | **1.321 GB** |
| po2=19 (`2¹⁹` cyc) | 422 MB | 206 MB | 2 MB | 0.661 GB |
| **session** (3×po2=20 + 1×po2=19) | | | | **4.624 GB** |

## Measured: the handoff is genuinely free on the APU

- **`rocprofv3 --memory-copy-trace` recorded ZERO memory-copy operations** for the
  entire prove — no trace file is even emitted (only a single 0.031 ms internal
  `__amd_rocclr_copyBuffer`). The CPU-written managed buffers are read **in place** by
  the iGPU; there is no H2D/D2H step to time.
- Microbenchmark (`hip/a2_handoff.hip`, best of 6):

  | op (844 MB `data` buffer) | time | bandwidth |
  | --- | ---: | ---: |
  | `hipMemcpy` H2D (discrete-style) | 11.07 ms | 80.2 GB/s |
  | `hipMemcpy` D2H | 12.83 ms | 69.0 GB/s |
  | **managed: GPU reads CPU-written buf** | **4.59 ms** | **193.0 GB/s** |

  The managed read is **2.4× faster** than an explicit `hipMemcpy` *and* needs **no
  second buffer** — because on the APU there is one physical LPDDR5X pool, so the GPU
  reads the CPU's bytes directly (no migration penalty: a resident-vs-fresh read shows
  no slowdown). `hipMemcpy` is slower precisely because it *reads + writes* a duplicate.

## Counterfactual: what a discrete GPU pays to move 4.624 GB/session

| path | H2D bandwidth | session copy | % of 26.11 s prove |
| --- | ---: | ---: | ---: |
| **unified managed (this build)** | — (0-copy) | **0.0 ms** | **0.00 %** |
| APU `hipMemcpy` (measured) | 80 GB/s | 57.8 ms | 0.22 % |
| PCIe 5.0 ×16 (~55 GB/s eff.) | 55 GB/s | 84.1 ms | 0.32 % |
| PCIe 4.0 ×16 (~26 GB/s eff.) | 26 GB/s | 177.9 ms | 0.68 % |

## Honest reading

- **For this compute-bound prove the raw copy *time* is small** (≤0.68 % even at
  PCIe4). The handoff is not where a discrete GPU would mainly lose — proving is 26 s of
  BabyBear arithmetic, not data movement. We state this plainly rather than inflate it.
- **The decisive unified-memory wins are structural, not copy-time:**
  1. **No duplicate RAM.** A discrete hybrid must hold the trace in host RAM **and**
     device VRAM at once (≥4.6 GB duplicated, plus the expanded STARK working set). The
     APU keeps **one** copy in the 94 GB pool.
  2. **Capacity.** Peak working set is **9.78 GB at po2=20 and 18.9 GB at po2=21** — the
     latter **exceeds a 16 GB discrete card's VRAM entirely** (the roofline OOM ceiling,
     `stage4-roofline.md`). The unified pool is what lets the hybrid run at all at scale.
  3. **Zero-copy convenience** is what makes the *CPU-witgen / GPU-STARK split itself
     practical*: the CPU's writes are instantly GPU-visible, so delegating witgen+accum
     to the CPU (the hybrid design A1 shows is the sweet spot) costs nothing to wire.

**A2 verdict.** The unified-memory handoff is **measured at 0 copies / 0 ms**; a
discrete GPU would pay a modest PCIe tax (≤0.7 %) *and*, more importantly, need duplicate
RAM it doesn't have at po2=21. The APU advantage for this workload is **capacity + no
duplication**, with copy-time savings a real-but-minor bonus.

Raw: [`a2-raw-handoff-bench.csv`](a2-raw-handoff-bench.csv), `hip/a2_handoff.hip`;
zero-memcpy evidence in `a1-raw/` (rocprofv3 emitted no memory-copy trace).
