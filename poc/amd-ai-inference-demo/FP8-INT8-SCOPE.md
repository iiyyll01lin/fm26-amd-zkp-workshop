# B2 quantization scope — FP8 / INT8 usability for the MIGraphX MiniLM path

> **Question (plan todo `scope-fp8`).** Are FP8 and INT8 quantization *actually
> usable* on this box for the Demo F MIGraphX MiniLM path, so we know how deep
> the B2 (MIGraphX inference deepening) track can go?
>
> **Short answer.** Yes — `fp16`, `bf16`, `int8`, and `fp8` quantization all
> exist in this box's MIGraphX python bindings **and all four compile + lower to
> the `gfx1151` GPU target** on the real MiniLM ONNX. INT8 is the deepest *safe*
> rung (native RDNA 3.5 packed-dot WMMA); FP16/BF16 is the zero-cost first rung;
> FP8 is functional but a research-grade rung on this architecture. Concrete
> recommendation in §6.

Everything in §1–§4 is grounded in command output captured on this box on
2026-06-18. The FP8-hardware nuance in §5 that I did **not** measure directly is
labelled as such and cited.

---

## 0. Honesty caveats (read first)

- **The probe box was NOT solo.** During this scoping run `rocm-smi` reported the
  iGPU at **100 %** busy and 1-min loadavg ≈ **14.7** (ceiling 8.0), i.e. a
  concurrent track owned the `gfx1151`. So the compile times in §4 are **inflated
  upper bounds**, not benchmark numbers, and **no perf/throughput numbers were
  recorded** during that scoping probe (that would violate
  `docs/gpu-bench-discipline.md`). The perf/watt comparison of the quantized
  variants was **captured later, in a clean solo window** (iGPU ≤ 4 %, loadavg
  2.85, `solo=true`) — see the measured results in §7.
- **Demo F weights are random-init** (capability bench, per `README.md`), so this
  note scopes quantization as a **throughput / memory-footprint / perf-per-watt**
  capability — it does **not** and cannot claim post-quant *accuracy* retention
  on a real checkpoint.
- This accelerates the **AI model** forward pass only. EZKL Halo2 / RISC0 STARK
  proving stay CPU-only on AMD (the repo's standing honesty rule).

---

## 1. Probe environment (observed)

| Item | Value | Source |
|---|---|---|
| GPU arch | `gfx1151` (RDNA 3.5, Strix Halo, Wave32) | `rocminfo \| grep -i gfx` |
| Marketing name | AMD Radeon 8060S (Ryzen AI MAX+ PRO 395) | `rocminfo` Marketing Name |
| ROCm | `7.2.3` | `/opt/rocm-*/.info/version` |
| MIGraphX python | `/opt/rocm/lib/migraphx.cpython-313-*.so` | `import migraphx` (`PYTHONPATH=/opt/rocm-*/lib`) |
| SMI tools | `rocm-smi`, `amd-smi` on PATH | `which` |

---

## 2. Quantization APIs available (observed)

`dir(migraphx)` exposes the full quantization surface; the pybind signatures are:

```text
quantize_fp16(prog, ins_names=['all']) -> None
quantize_bf16(prog, ins_names=['all']) -> None
quantize_int8(prog, t: target, calibration=[], ins_names={'convolution','dot'}) -> None
quantize_fp8 (prog, t: target, calibration=[]) -> None
autocast_fp8(...)                # also present
```

Takeaways:

- **FP16 / BF16**: take only the program (+ optional op filter). **No target, no
  calibration** — a one-liner, accuracy-neutral-ish reduced precision.
- **INT8 / FP8**: take a **compile `target`** and an **optional `calibration`**
  list. INT8 additionally restricts to `{'convolution','dot'}` ops by default
  (i.e. the GEMMs — exactly the MiniLM hot path).

## 3. FP8 encodings exposed (observed)

`migraphx.shape.type_t` enumerates these low-precision types:

```text
int8_type,
fp8e4m3fn_type, fp8e4m3fnuz_type, fp8e5m2_type, fp8e5m2fnuz_type
```

- Both **e4m3** and **e5m2** FP8 encodings are present, each in IEEE-ish `fn` and
  AMD `fnuz` (no-Inf/NaN-unsigned-zero) variants. `quantize_fp8` defaults to the
  e4m3 family (the accuracy-oriented FP8 used for inference).
- **No standalone `e8m0` type** is exposed. `e8m0` is the OCP **MX block-scale**
  exponent (MXFP8); this MIGraphX build offers per-tensor/per-channel FP8, not a
  first-class MXFP8 block-scaled path.

## 4. Functional compile probe (observed)

Parsed the real `artefacts/minilm-l6.onnx` (shape `1×32`), applied each transform,
and compiled to `migraphx.get_target("gpu")` on `gfx1151`:

| Variant | Result | Compile time* |
|---|---|---|
| FP32 baseline | **COMPILE OK** | 6.6 s |
| `quantize_fp16` | **COMPILE OK** | 9.9 s |
| `quantize_int8` (no calibration) | **COMPILE OK** | 11.8 s |
| `quantize_fp8` (no calibration) | **COMPILE OK** | 21.5 s |

\* *Compile times are under heavy GPU contention (iGPU 100 %); they are lowering
times, **not** perf numbers, and only show each path is functionally viable.*

**All four quantization paths are functionally usable** end-to-end (parse →
quantize → lower to the gfx1151 GPU target) in this MIGraphX build. FP8 lowering
is the slowest, consistent with it being the least mature path (§5).

## 5. FP8 / INT8 *hardware* on gfx1151 — RESOLVED: FP8 is convert+upcast, not native WMMA

The §4 result proves the **software** path compiles; the open question was whether
`quantize_fp8` lowers to **native WMMA FP8** matrix instructions on `gfx1151` or to
**convert+upcast (fp8 → f32) + FP32 FMA** GEMM. This is now **measured and
resolved** (2026-06-22) by a dedicated instruction probe —
[`scripts/fp8-wmma-probe.sh`](scripts/fp8-wmma-probe.sh) → verdict in
[`artefacts/fp8-wmma.md`](artefacts/fp8-wmma.md) (raw evidence
[`fp8-wmma.txt`](artefacts/fp8-wmma.txt)):

> **VERDICT: convert+upcast+FP32 FMA — NOT native WMMA FP8.**

The decisive signal is the **MLIR `dot` operand element type** that MIGraphX hands
to rocMLIR (`MIGRAPHX_MLIR_DUMP`): in the FP8-quantized program **every
`migraphx.dot` consumes `f32` operands** (30/30 dot tokens f32, **0** fp8) — the
fp8 is `dequantizelinear`-upcast to `f32` *before* the matmul. The **FP16 control**
instead feeds `f16` straight into the dot (18/18 f16). Since `gfx1151` WMMA has **no
`f32×f32` input mode**, an `f32` dot **cannot** use WMMA at all — it runs on
`v_fma_f32` VALU. Corroborating ISA evidence (`AMD_COMGR_SAVE_TEMPS` +
`llvm-objdump`): the FP8 path compiles **8** explicit
`quantizelinear`/`dequantizelinear`/`convert` JIT kernels (66 `v_cvt_*` + 609
`v_fma_f32`), the literal convert+FMA round-trip. (The gemm code object itself is
compiled by rocMLIR *in-process* and is not separately disassemblable; the verdict
rests on the MLIR operand types + the convert-kernel ISA + the measured §7 perf,
which all agree. The `rocprofv3 --kernel-trace` step was solo-gated and skipped on
the contended probe run — it is corroborating, not load-bearing.)

This **explains the measured §7 perf**: FP8 pays the quantize/dequantize/convert
overhead **and** still runs the GEMM in FP32, so it is strictly *worse* than plain
FP32 (0.43× at large batch). Architectural context from public RDNA 3.5 sources:

- **INT8 — native, mature.** gfx1151 WMMA has packed INT8 dot
  (`amd_mixed_dot` / `dot4add_i8packed`); INT8 is the well-trodden quantized path
  on RDNA 3.5. ([fsr4-rdna3-optimization](https://github.com/lhl/fsr4-rdna3-optimization),
  [chipsandcheese RDNA 3.5](https://chipsandcheese.com/p/amd-rdna-3-5s-llvm-changes))
- **FP8 (e4m3) — the *hardware* exists, but this MIGraphX build does not target
  it.** RDNA 3.5 WMMA *can* do FP8 (`AmdWaveMatrixMultiply`, 16×16 tiles), and
  community microkernels report FP8 conv+FMA ≈ **3.7× slower than INT8** on a
  non-WMMA harness ([fsr4-rdna3-optimization](https://github.com/lhl/fsr4-rdna3-optimization)).
  Our probe shows that on **this ROCm 7.2.3 / MIGraphX build the `quantize_fp8`
  lowering does not emit the FP8 WMMA path at all** — it upcasts to f32. So the
  no-win is not a tuning gap; it is the lowering choosing convert+upcast.
- Peak FP16/BF16 ≈ **59.4 TFLOPS** on the 8060S *only* with WMMA / wave32 VOPD
  (else halved). ([llm-tracker Strix Halo](https://llm-tracker.info/AMD-Strix-Halo-(Ryzen-AI-Max%2B-395)-GPU-Performance))

Net: on `gfx1151`, **INT8 is the precision with the strongest native-hardware
story**, **FP16 keeps the native f16 WMMA path**, and **FP8 (in this build) is
convert+upcast+FP32 — measured, not assumed**.

## 6. Recommended B2 depth

Ordered by ROI / risk, and wired to the new per-shape **perf/watt** columns
(`power_w` / `joules` / `embeddings_per_joule`) that `embed_bench.py` now emits —
so each rung can be judged on throughput **and** energy, solo, in one sweep:

1. **Rung 1 — FP16 (and/or BF16). GREEN, do first.** One-liner
   (`quantize_fp16(prog)`), no calibration, no target plumbing. A compute-bound
   transformer forward is exactly what halved-precision WMMA eats; expect a clean
   throughput + footprint + perf/watt win. Lowest cost, lowest risk; add it as a
   `rocm-fp16` backend row alongside the existing `rocm` (fp32) rows.
2. **Rung 2 — INT8 with a real calibration set. GREEN/AMBER, the depth target.**
   Native packed-dot on gfx1151, applied to `{dot, convolution}` (the GEMMs).
   **Calibration requirement:** `quantize_int8` wants a `list[dict[str,
   migraphx.argument]]` of *representative activation* feeds — for MiniLM that's a
   handful of `{input_ids, token_type_ids, position_ids}` batches across the swept
   shapes. With an **empty** calibration it still compiles (default scales) but is
   accuracy-blind; since Demo F is random-init, scope INT8 as a
   **throughput / perf-watt capability** (optionally add a fp32-vs-int8 cosine
   check on the same random model to show numerical drift, not "accuracy").
3. **Rung 3 — FP8 (e4m3). RED, research bullet only — and now we know *why*.** It
   compiles (§4) and the encodings exist (§3), but the §5 probe **measured** that
   on this build `quantize_fp8` lowers to **convert+upcast+FP32 FMA, not native
   WMMA FP8** ([`artefacts/fp8-wmma.md`](artefacts/fp8-wmma.md)). So FP8 is
   strictly slower than FP32 here (it adds convert overhead on top of an f32 GEMM)
   — **never an FP8 headline on `gfx1151`/ROCm 7.2.3**. MXFP8/`e8m0` block-scaling
   is **out of scope** (not exposed here).

**Prerequisite for any rung's published numbers:** a **solo** window
(`solo-guard` green, iGPU ≤ 25 %, loadavg ≤ 8.0). The box was contended during
this *scoping* probe; the quantized-variant perf/watt sweep was subsequently run
in a clean solo window (iGPU ≤ 4 %, loadavg 2.85). **Measured numbers are now in
§7** — and they confirm this ordering: FP16 is the clean all-round win, INT8 is
the perf/watt leader at low latency, and FP8 is *not* a win on this build (often
at/below FP32), exactly the "research rung" prediction of §5.

## 7. Implementation status + MEASURED perf/watt (clean solo, 2026-06-18)

The three rungs above are **now implemented**, not just recommended:

- `src/embed_bench.py` exposes backends `rocm-fp16` / `rocm-int8` / `rocm-fp8`
  via a generalized `run_migraphx(..., quant=...)`: it applies
  `quantize_fp16(prog)` / `quantize_int8(prog, target, calibration)` /
  `quantize_fp8(prog, target, calibration)` **before** `compile`. The INT8/FP8
  calibration is a per-shape list of representative `make_inputs()` feeds
  (`AI_BENCH_CALIB_BATCHES`, default 4) whose shapes match the parsed program.
- `scripts/run-all.sh` now (a) defaults the rocm backend set to
  `cpu,rocm,rocm-fp16,rocm-int8,rocm-fp8`, and (b) is **solo-guarded** — it
  refuses (exit 42) under contention and stamps every CSV row with
  `solo`/`loadavg`, exactly like the other timed benches.

**Status of the perf/watt rows: CAPTURED (clean solo, 2026-06-18).** An earlier
execution attempt was abandoned because the `gfx1151` was contended by a sibling
project's ROCm job (iGPU 90–100 %); recording then would have violated
`docs/gpu-bench-discipline.md` and stamped a false `solo=true`. A clean window
later opened (`solo-guard` green at start: iGPU 2 %, loadavg 2.85; iGPU stayed
≤ 4 % between our own GPU phases throughout), so `make demo-f-embed` swept all five
backends and stamped every row `solo=true, loadavg=2.85`. The numbers below come
from that run ([`artefacts/ai-inference.csv`](artefacts/ai-inference.csv),
best-of-20 `fwd_ms`; `power_w` = whole-SoC package draw via `rocm-smi`).

**Measured — low-latency shape (batch 1, seq 32):**

| backend | fwd_ms | embeddings/s | tokens/s | emb/J | vs FP32 (fwd) |
|---|---|---|---|---|---|
| `cpu` (Zen 5, 32T) | 2.382 | 419.8 | 13,434 | 3.21 | 0.38× |
| `rocm` (FP32) | 0.897 | 1,115 | 35,694 | 13.24 | 1.00× |
| `rocm-fp16` | 0.357 | 2,799 | 89,574 | **31.35** | **2.51×** |
| `rocm-int8` | 0.351 | 2,851 | 91,238 | **43.50** | **2.56×** |
| `rocm-fp8` | 0.960 | 1,042 | 33,331 | 14.26 | 0.93× |

**Measured — large-batch shape (batch 32, seq 256):**

| backend | fwd_ms | embeddings/s | tokens/s | emb/J | vs FP32 (fwd) |
|---|---|---|---|---|---|
| `cpu` (Zen 5, 32T) | 198.592 | 161.1 | 41,250 | 1.06 | 0.23× |
| `rocm` (FP32) | 46.321 | 690.8 | 176,854 | 6.72 | 1.00× |
| `rocm-fp16` | 13.784 | 2,321 | 594,298 | **19.88** | **3.36×** |
| `rocm-int8` | 25.019 | 1,279 | 327,432 | 9.86 | 1.85× |
| `rocm-fp8` | 108.004 | 296.3 | 75,849 | 3.40 | 0.43× |

Peak across the full 9-shape grid: **FP16 hits 1.02 M tokens/s** (b32/s128) and
28,766 embeddings/s (b32/s32); INT8 peaks at 613 k tokens/s; FP8 peaks at 238 k.

**Reading of the three rungs (it matches §5/§6 exactly):**

1. **FP16 — the clean all-round win.** ~2.5–3.8× faster than FP32 across the grid,
   the highest absolute throughput (>1 M tok/s), and the best large-batch perf/watt
   (19.88 emb/J at b32/s256, ~3× the FP32 6.72 and ~19× the CPU 1.06). This is the
   "do first, GREEN" rung, confirmed.
2. **INT8 — the perf/watt leader at low latency, but shape-dependent.** Best
   energy efficiency on the box at small shapes (**43.5 emb/J** at b1/s32 — 3.3× FP32,
   13.6× CPU) and 2.5× faster than FP32 there. At large batch the per-call
   quant/dequant overhead eats into it (1.85× FP32 at b32/s256, and it can even fall
   *behind* FP32 at b32/s32 where the GEMMs are short) — so INT8 is the depth target
   for **latency-bound / small-batch** serving, not a blanket win.
3. **FP8 — research rung, NOT a win on this build (now confirmed by the ISA probe).**
   It compiles and runs end-to-end, but it never beats FP32 in this sweep: roughly
   FP32-parity at tiny shapes (0.93× at b1/s32, 14.26 vs 13.24 emb/J) and **markedly
   slower at large batch** (0.43× / 108 ms vs 46 ms at b32/s256). The §5 probe
   (2026-06-22) **resolved why**: `quantize_fp8` lowers to **convert+upcast+FP32 FMA,
   not native WMMA FP8** ([`artefacts/fp8-wmma.md`](artefacts/fp8-wmma.md)) — every
   `migraphx.dot` consumes f32 operands (fp8 dequantized first), while the FP16
   control keeps f16 in the dot. So FP8 pays convert overhead on top of an f32 GEMM,
   strictly worse than FP32. **No FP8 headline on `gfx1151`/ROCm 7.2.3** — not a
   tuning gap, the lowering simply does not target the FP8 WMMA path.

**Honesty caveats (unchanged):** Demo F uses **random-init** weights, so these are
**throughput / footprint / perf-per-watt capability** numbers, **not** accuracy.
`power_w` is whole-SoC package draw (the iGPU shares the LPDDR5X budget with the
Zen 5 cores), i.e. a package-level efficiency view, not a per-engine power split.
And this accelerates the **AI model** forward pass only — EZKL Halo2 / RISC0 STARK
proving stay CPU-only on AMD.
