# A3 — honest speed range (segment/po2, hashfn, codegen fairness)

Re-measured fresh on the reconstructed known-good hybrid build (2026-07-20, Strix
Halo gfx1151, ROCm 7.2.3), **solo-guarded**, every seal **stock-verified**. Prior
numbers were not trusted; these supersede them.

## Headline (po2=20 default, poseidon2, RAYON_NUM_THREADS=32, best-of-N, solo)

| config | backend | wall | verify | vs fork-CPU |
| --- | --- | ---: | :---: | ---: |
| **fork-GPU** | rocm gfx1151 (+CPU witgen/accum) | **26.1 s** | ✅ valid | **5.46×** |
| **fork-CPU** | no-rocm, 32t Zen5 (same code) | **142.4 s** | ✅ valid | 1.00× |
| installed `r0vm 2.3.2` | 32t Zen5 (rzup binary) | 177.6 s | ✅ valid | 0.80× |

- **The honest headline is 5.46×** — iGPU vs the **same fork code** built no-rocm on
  the same box. Apples-to-apples (identical source, toolchain, generic codegen).
- Matches the solo po2-sweep's cleanest baseline (5.52× at po2=20) within noise; the
  iGPU wall is a stable ~26 s.

## The fork-CPU vs installed-r0vm codegen gap (nailed)

`installed / fork-CPU = 177.6 / 142.4 = 1.25×`. Same source (`v2.3.2`), but the fork
is compiled with the **local toolchain** (rustc 1.85 + gcc 13, `-O3`) while the rzup
`r0vm` ships a CI-built binary. So **iGPU-vs-installed = 177.6/26.1 = 6.80×**, which is
where the earlier "~6.6–6.8× / audited 6.79×" came from — it was partly measuring the
**1.25× codegen gap** on top of the true same-code **5.46×**. We report **5.46× as the
honest figure** and attribute the rest to codegen. (My fresh installed run, 177.6 s at
loadavg 4.02, is already *faster* than the 254.3 s in older notes; the gap is smaller
and more conservative than previously stated.)

## Hashfn axis collapses (poseidon2 only)

`r0vm --hashfn sha-256` is **rejected** by the rv32im 2.3.2 prover:

```
unsupported `hashfn` value of "sha-256"; supported `hashfn` values are: "poseidon2".
```

So there is **no poseidon2-vs-sha256 crossover to measure** — poseidon2 is forced, not
chosen, and it is the CPU-costly / GPU-friendly hash. This *sharpens* the caveat rather
than widening the speed range.

## Fairness knobs — confirmed, with the honest SIMD caveat

- **release**: yes — all binaries are `--release` (`-O3` on the C++ kernels).
- **RAYON_NUM_THREADS=32**: yes — set for every CPU run (32 physical Zen5 cores).
- **SIMD**: the build uses risc0's **stock, generic codegen** — the C++ kernels compile
  `-march=x86-64 -mtune=generic` and the Rust side has **no `-C target-cpu=native`**
  (`.cargo/config.toml` sets only `-Dwarnings`). The Zen5 CPU **has AVX-512**
  (`avx512f/bw/cd/dq/vl/ifma/vbmi`, confirmed via `/proc/cpuinfo`) but the build does
  **not** target it. So the fork-CPU baseline is the *stock risc0 CPU path* — a
  `-march=native` / AVX-512-tuned CPU build would speed up the CPU and **narrow** the
  5.46×. The fork-GPU-vs-fork-CPU ratio is still apples-to-apples (both generic
  codegen); the caveat only bounds generalisation to a maximally-tuned CPU.

## po2 curve (segment-size sweep)

The default po2=20 point re-measured here (5.46×) is consistent with the solo
segment-po2 sweep (`stage4-sweep.csv` / `stage4-roofline.md`): a **flat ~5.3–5.5×**
across po2 16→21 with **no crossover** (iGPU wins at every segment size), the iGPU wall
stable ~26–31 s, and peak RSS reaching **18.9 GB at po2=21** (past the 16 GB discrete
wall; held by the 94 GB unified pool). hashfn stays poseidon2-only throughout.

## A3 verdict (honest range)

The honest speed result is a **workload-/poseidon2-/hybrid-specific ~5.5×** vs the
same-code 32t CPU (range 5.3–5.5× across po2), **not** a general "iGPU is faster" claim.
The larger "~6.8×" figures fold in a **1.25× local-vs-shipped codegen gap**; correctness
(stock-verifier seal + bit-for-bit kernels) remains the hard guarantee, speed the scoped
secondary. Raw: [`a3-speed.csv`](a3-speed.csv), [`a3-raw-headline.log`](a3-raw-headline.log).
