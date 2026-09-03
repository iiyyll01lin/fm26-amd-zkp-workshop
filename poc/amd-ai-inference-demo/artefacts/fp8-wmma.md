# FP8 native-WMMA vs convert+upcast — verdict (gfx1151 / MIGraphX)

_Source: `artefacts/fp8-wmma.txt` (raw evidence) — produced by
`poc/amd-ai-inference-demo/scripts/fp8-wmma-probe.sh`._

- **Hardware:** AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151)
- **Stack:** ROCm 7.2.3, MIGraphX python binding (ROCm tree); profiler `rocprofv3`
- **Probe:** MiniLM ONNX → `migraphx.quantize_fp8` / `quantize_fp16` → `gpu` target
- **Date:** 2026-06-22 · solo=false loadavg=5.27

## Verdict: **CONVERT + UPCAST + FP32 FMA  (NOT native WMMA FP8)**

every migraphx.dot in the FP8 program consumes f32 operands (the fp8 is upcast to f32 BEFORE the matmul); the FP16 control instead feeds f16 operands straight into the dot. gfx1151 WMMA has no f32xf32 input mode, so an f32 dot cannot use WMMA — it runs on v_fma_f32 VALU.

This confirms the FP8-INT8-SCOPE.md §5 hypothesis and explains the measured
perf (FP8 is no win — 0.43× at large batch, §7): FP8 pays quantize/dequantize/
convert overhead **and** still runs the GEMM in FP32, so it is strictly worse
than plain FP32 on this RDNA 3.5 iGPU.

## Evidence

### 1. MLIR `dot` operand types (the decisive signal)

`MIGRAPHX_MLIR_DUMP` dumps every `migraphx.dot` module MIGraphX hands to
rocMLIR. The operand **element type** is the smoking gun:

| precision | `migraphx.dot` operand element type | reading |
|---|---|---|
| **FP8** | **f32** (fp8 tokens on dot lines: 0; f32: 30) | fp8 is upcast to f32 **before** the matmul (`dequantizelinear` brackets the dot) → no fp8 matrix op |
| **FP16** (control) | **f16** (f16 tokens: 18) | f16 fed straight into the dot — f16 **is** a native gfx1151 WMMA input type |

gfx1151 WMMA has no f32×f32 input mode, so an f32 `dot` **cannot** use WMMA
at all — it lowers to `v_fma_f32` VALU. The FP16 control proves the pipeline
*does* keep the matrix type when the hardware supports it (f16); FP8 does not.

### 2. JIT convert-kernel ISA (`AMD_COMGR_SAVE_TEMPS` + `llvm-objdump -d`)

- The FP8 program compiles explicit **8** `quantizelinear` /
  `dequantizelinear` / `convert` pointwise kernels, full of `v_cvt_*` (66)
  + `v_fma_f32` (609) — the literal convert+FMA round-trip.
- `v_wmma_*` count in the comgr-captured objects: FP8 0 / FP16 0.
  **Caveat:** the dot/GEMM is compiled by rocMLIR *in-process* and is not
  emitted to a separately disassemblable code object, so neither precision's
  comgr objects contain the gemm — the 0/0 `v_wmma` is expected and is **not**
  evidence about the gemm. The gemm signal is the MLIR operand type in (1).

### 3. `rocprofv3 --kernel-trace` (dispatched GPU kernels)

- kernel-trace: SKIPPED — no clean solo window (solo=false, loadavg=5.27) or rocprofv3 absent; verdict rests on the compile-time MLIR + ISA evidence.

## Honesty notes

- Demo F weights are **random-init** → this is a **capability / instruction**
  probe, **not** an accuracy claim.
- The iGPU accelerates the **AI model** forward only; EZKL Halo2 / RISC0 STARK
  proving stay **CPU-only** on AMD (the repo's standing rule).
- The gemm code object is rocMLIR-in-process / opaque to file-level disasm;
  the verdict rests on the MLIR `dot` operand types (1) + the convert-kernel
  ISA (2) + the kernel trace (3) + the measured §7 perf, which all agree.
