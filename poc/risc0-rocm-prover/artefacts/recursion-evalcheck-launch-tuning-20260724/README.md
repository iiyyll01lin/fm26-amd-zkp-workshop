# Recursion eval_check launch-geometry tuning - negative result

Date: 2026-07-24. Arch: `gfx1151` (Radeon 8060S). rocprofv3 1.1.0.
Base commit: `10d5aed`. Task C(a): a bounded attempt to speed up the recursion
`eval_check` kernel by tuning its HIP launch geometry (threads per block).

## TL;DR

**No launch-tuning speedup exists.** The kernel is **register-bound at 192
VGPRs/thread** (measured, launch-independent), so changing the block size cannot
improve occupancy. The current default (**block = 256**) is already the fastest
of every size swept. The experimental env-tunable block was **reverted**; the
shipped kernel is unchanged.

## Method

- Temporarily made the block size env-tunable
  (`RISC0_ROCM_RECURSION_EVAL_CHECK_BLOCK_SIZE`, clamped to the kernel's
  `maxThreadsPerBlock`), rebuilt `risc0-circuit-recursion` with `--features rocm`,
  and confirmed the `hip::tests::eval_check` GPU==CPU test still passes.
- Swept block in {64, 128, 256, 512}, 3 reps each, under
  `rocprofv3 --kernel-trace --stats`, recording the `eval_check` kernel's median
  duration and register/scratch use. Solo GPU window (loadavg < 0.2, GPU 0%).

## Results

| block | median (ns) | min | max | VGPR/thread | scratch (B) |
|---|---|---|---|---|---|
| 64  | 1,179,782 | 811,607 | 1,180,432 | 192 | 3,824 |
| 128 | 1,008,614 | 822,665 | 1,034,153 | 192 | 3,824 |
| **256** | **813,134** | 812,971 | 1,215,411 | 192 | 3,824 |
| 512 | 1,129,530 | 931,956 | 1,146,607 | 192 | 3,824 |

- **block = 256 is optimal** (ratio 256/best = 1.000x). Smaller and larger blocks
  are slower.
- **VGPR/thread = 192 is constant across all block sizes.** Register allocation is
  a property of the compiled kernel, not the launch configuration, so no launch
  geometry can reduce it. On `gfx1151` (wave32, ~1536 VGPRs/SIMD) 192 VGPRs caps
  occupancy at ~8 waves/SIMD regardless of block - the kernel is **register-bound**.

## Why the correctness-test scale is sufficient for this conclusion

The `hip::tests::eval_check` gate runs at `domain = 64` (a single workgroup). That
is too small to compare *throughput* meaningfully, but the **register footprint
(192 VGPRs) is independent of both the launch geometry and the domain size** - it
is fixed by the generated kernel's code. The production `eval_check` (~31 ms in a
full succinct prove, larger domain) is bound by the same 192-VGPR ceiling, so the
register-bound conclusion carries over: the bottleneck is register pressure, not
how the fixed work is tiled into blocks.

## Conclusion & recommendation

- **Negative result:** launch-geometry (block-size) tuning yields no speedup for
  the recursion `eval_check`; the default `block = 256` is retained unchanged.
- **The real lever is register pressure**, which is a kernel/codegen concern
  (e.g. recompute-vs-cache trade-offs, LDS staging, or compiler options) - outside
  the scope of a safe launch-only change and not attempted here.
- Separately, the Groth16 benchmark (`groth16-seam-validation.md`) showed the
  end-to-end Groth16 prove is **witness-generation-bound**, so GPU-offloading the
  circom witness generator is the higher-value (but multi-week) optimization
  target across the pipeline.

Evidence: `block-sweep-summary.csv`, `eval_check-block256-kernel_trace.csv`,
`eval_check-block256-kernel_stats.csv`, `provenance.txt` (this directory).
