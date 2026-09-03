# Demo F — iGPU AI-inference on the AMD Radeon (MiniLM forward, MIGraphX/ROCm)

> Path F. The honest, **measured** answer to "does the iGPU actually accelerate
> the AI model on this box?" — turning a claim that lived only in the docs into a
> committed artefact.
>
> It runs the **`all-MiniLM-L6-v2`** sentence-embedding forward pass (the same
> model family Demo A's `384 -> 64` head emulates) through the **same ONNX
> graph** on two runtimes and sweeps batch size × sequence length:
>
> - **CPU baseline**: `onnxruntime` CPUExecutionProvider, all 32 Zen 5 threads.
> - **iGPU**: AMD **MIGraphX** (ROCm 7.2.3, native gfx1151) on the Radeon 8060S.

## What it proves (and does not)

- **Does**: the AMD iGPU (Radeon 8060S / `gfx1151`) genuinely runs a real
  transformer-encoder forward pass through AMD's native ROCm inference engine
  (MIGraphX), and we measure the speedup over the 32-thread Zen 5 baseline on
  the identical ONNX graph.
- **Does NOT**: accelerate any ZK proof. The iGPU here accelerates the **AI
  MODEL** (the embedding / prefill step the DEAAP pipeline runs *before*
  proving). EZKL Halo2 (Demo A) and the RISC0 `r0vm` STARK (Demo B) proving stay
  **CPU-only on AMD** — see [`../../docs/amd-strix-halo-acceleration.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/amd-strix-halo-acceleration.md).
- The weights are **random-init** (like Demo A's `01_make_model.py`): this is a
  forward-latency / throughput **capability** measurement with the published
  model's exact architecture and FLOPs (22.6M params, hidden 384, 6 layers, 12
  heads, intermediate 1536), not an accuracy benchmark, so no checkpoint
  download is needed.

## Result (measured on AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S, ROCm 7.2.3, kernel 6.17)

`fwd_ms` = **best (min) of 20 timed runs** after 5 warmups (per-shape MIGraphX
compile excluded; it lands in the warmup). `speedup = cpu_ms / gpu_ms`
(>1 ⇒ the iGPU forward beats the 32-thread CPU forward):

| batch × seq | CPU (ms) | iGPU (ms) | speedup |
|---|---:|---:|---:|
| 1 × 32 | 1.94 | 1.02 | **1.90x** |
| 1 × 256 | 6.93 | 1.75 | **3.95x** |
| 8 × 128 | 22.82 | 3.77 | **6.05x** |
| 32 × 32 | 40.02 | 3.79 | **10.56x** |
| 32 × 128 | 71.88 | 15.89 | **4.52x** |
| 32 × 256 | 208.03 | 42.41 | **4.91x** |

- The iGPU is **~1.9–10.6x faster** across the whole batch × seq grid (low end
  1×32 1.90x, peak 32×32 10.56x). Peak iGPU throughput hit **~270k tokens/s** and
  **~8,446 embeddings/s** (batch 32, seq 32). Unlike Path E's MSM (size-gated on
  the ec-gpu OpenCL path, only wins at large sizes), a transformer forward is
  **compute-bound dense GEMM** — exactly what the iGPU's shaders eat, so it wins
  from the smallest workload up.
- **Clean solo re-bench (2026-06-18)**: the previously committed sweep had a
  contention-**inflated** CPU baseline (iGPU/CPU were concurrently occupied),
  making the speedup look as high as 9–25x. A solo-guard-gated clean window
  (iGPU **0% busy**, loadavg **1.59**; peak loadavg 3.70) re-ran the sweep and the
  whole artefact was **replaced** with clean solo numbers — e.g. CPU `1×32`
  10.93 → **1.94 ms**, `32×256` 411.73 → **208.03 ms**. The iGPU side was already
  solo and barely moved, so this is a **pure CPU-baseline correction** (9–25x →
  **~1.9–10.6x**); still a clean win, just smaller. Same class of honest correction
  as [`docs/INTEGRITY-REPORT.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/INTEGRITY-REPORT.md)'s two-step 1.34x→0.70x→0.994x
  folding retraction. Durable backup:
  [`artefacts/ai-inference.solo-rebench-2026-06-18.{csv,log}`](artefacts/).

Full data: [`artefacts/ai-inference.csv`](artefacts/ai-inference.csv) /
[`artefacts/ai-inference.md`](artefacts/ai-inference.md);
chart: `artefacts/ai-inference.png`;
run log (with median/stdev spread): [`artefacts/ai-inference.log`](artefacts/ai-inference.log).

### Per-shape perf/watt (appended, backward-compatible)

`embed_bench.py` now wires labkit's `gpu_telemetry` (rocm-smi) sampler into the
sweep, so each `(backend, batch, seq_len)` row also carries **`power_w`**,
**`joules`** (energy per forward), and **`embeddings_per_joule`** — perf/watt is
per-shape, not a single decoupled telemetry track. The columns are appended after
the existing schema (older CSVs/readers are unaffected) and are emitted **empty**
when no SMI tool is available or a shape's power window captures < 2 samples
(never fabricated). `power_w` is the **whole-SoC package** draw (the iGPU shares
the LPDDR5X budget with the Zen 5 cores), so this is a GPU-side / whole-package
efficiency view, not a per-engine split. Set `AI_BENCH_ENERGY=0` to skip the
energy windows. **The committed `ai-inference.csv` predates this wiring** and its
energy cells are populated on the next **solo** (`solo-guard`-green) refresh —
the box was contended at wiring time, so a live perf/watt sweep is pending idle
hardware.

### Quantization depth (B2)

Whether to push the iGPU path to FP16 / INT8 / FP8 is scoped, with grounded probe
output, in [`FP8-INT8-SCOPE.md`](FP8-INT8-SCOPE.md): all four
(`fp16`/`bf16`/`int8`/`fp8`) MIGraphX quantizers exist here and compile to
`gfx1151`; **INT8** is the recommended depth (native RDNA 3.5 packed-dot),
**FP16/BF16** the zero-calibration first step, **FP8** a research rung.

## Backend selection

The backend set follows `AI_GPU_BACKEND` exported by
[`../risc0-cartesi-step-demo/scripts/amd-accel-detect.sh`](../risc0-cartesi-step-demo/scripts/amd-accel-detect.sh):

| `AI_GPU_BACKEND` | backends run | notes |
|---|---|---|
| `rocm` (this box) | `cpu` + `rocm` | iGPU via MIGraphX gpu target (gfx1151) |
| `vulkan` | `cpu` + `vulkan` | documented skip: no general transformer runtime on the Vulkan ICD here (onnxruntime has no Vulkan EP); recorded, never crashes |
| `cpu` | `cpu` | CPU-only host (laptop) — still produces a baseline CSV |

Override explicitly with `AI_BENCH_BACKENDS=cpu,rocm` or `--backends cpu`.

## Prerequisites

- ROCm 7.2.x with the **MIGraphX** python bindings present at
  `/opt/rocm-*/lib/migraphx.cpython-3XX-*.so` (ships with ROCm; `run-all.sh`
  adds that dir to `PYTHONPATH`). `rocminfo` must enumerate `gfx1151`.
- Python 3.13 for the self-contained venv (matches the ROCm `migraphx` cpython
  ABI on this box). The CPU baseline + ONNX export need no GPU.

## Run

```bash
bash scripts/run-all.sh
# tune the sweep grid:
BATCHES=1,8,32 SEQ_LENS=32,128,256 REPS=20 WARMUP=5 bash scripts/run-all.sh
```

`run-all.sh` sources `amd-accel-detect.sh`, creates a **self-contained venv
inside this demo dir** (NOT the repo `.venv` — keeps the heavy torch/onnx deps
isolated from the lab), builds the MiniLM ONNX, runs the iGPU-vs-CPU sweep
(CSV + log), then plots (PNG via matplotlib, else a markdown table).

## Layout

```
amd-ai-inference-demo/
├── README.md
├── INTEGRATION-SPEC.md       # exact labkit/Makefile/run-on-halo/notebook/STATUS wiring (closeout agent applies)
├── requirements.txt          # onnx + onnxscript + onnxruntime + numpy/pandas/matplotlib (torch installed from CPU index)
├── .gitignore                # ignores .venv/ + the regenerable *.onnx{,.data}
├── src/
│   ├── build_model.py        # define all-MiniLM-L6-v2 encoder (torch) -> export ONNX
│   └── embed_bench.py        # iGPU(MIGraphX) vs CPU(onnxruntime) forward sweep -> CSV
├── scripts/
│   ├── run-all.sh            # detect -> venv -> build ONNX -> sweep -> plot
│   └── plot-ai-inference.py  # 2-panel latency + speedup chart, markdown fallback
└── artefacts/                # ai-inference.{csv,md,png,log} committed as evidence
                              # (minilm-l6.onnx{,.data} are regenerated, gitignored)
```

See [`../../reading-notes/path-f-amd-ai-inference.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/reading-notes/path-f-amd-ai-inference.md)
for the full positioning, landscape, and where this sits in the engine map, and
[`../amd-gpu-zk-primitive-demo/`](../amd-gpu-zk-primitive-demo/) (Path E) for the
complementary "iGPU accelerates SNARK *primitives*" track.
