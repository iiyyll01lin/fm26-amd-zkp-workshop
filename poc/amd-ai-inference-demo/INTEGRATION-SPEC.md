# Demo F / Path F — INTEGRATION SPEC (for the closeout agent)

This file specifies the EXACT edits to the **shared** files that Phase 1 (G1)
must NOT touch directly (to keep parallel work conflict-free). The closeout
agent owns `lab/labkit.py`, `Makefile`, `scripts/run-on-halo.sh`,
`docs/IMPLEMENTATION-STATUS.md`, `README.md`, `docs/amd-strix-halo-acceleration.md`,
and `lab/*.ipynb`. Everything below is drop-in and matches the existing Path E
pattern.

Measured headline (committed in `artefacts/ai-inference.{csv,md,png}`; best-of-20,
iGPU was 98–100% contended by a concurrent track so these are lower bounds):

- iGPU `gfx1151` (MIGraphX/ROCm 7.2.3) runs the `all-MiniLM-L6-v2` forward
  **~1.9–10.6x faster** than the 32-thread Zen 5 CPU (onnxruntime, same ONNX)
  (2026-06-18 clean-solo re-bench; the old 9–25x was a contention-inflated CPU baseline).
- best speedup **10.56x** (batch 32 × seq 32); low end **1.90x** (1×32).
- peak iGPU throughput **270,285 tokens/s** / **8,446 embeddings/s**.
- iGPU accelerates the **AI MODEL forward**, NOT any ZK proof (EZKL/RISC0 stay CPU-only).

---

## (a) `lab/labkit.py` additions

**A1. New path constant** — add next to the other artefact-path constants
(after the `GPU_GROTH16_CSV` block, ~line 113):

```python
_AI_DEMO = REPO_ROOT / "poc" / "amd-ai-inference-demo"
#: Path F — MiniLM all-MiniLM-L6-v2 forward, iGPU(MIGraphX) vs CPU(onnxruntime).
AI_INFER_CSV: Path = _AI_DEMO / "artefacts" / "ai-inference.csv"
```

**A2. `__all__`** — add `"AI_INFER_CSV"` to the paths group, and
`"load_ai_inference"` to the csv-loaders group, and `"plot_ai_inference"` to the
plotters group.

**A3. Loader** — add after `load_gpu_groth16` (~line 534). Returned DataFrame
columns: `backend, batch, seq_len, fwd_ms, tokens_per_s, embeddings_per_s,
device, cpu_threads` (numeric coerced where applicable).

```python
def load_ai_inference(path: Optional[Path] = None):
    """Load the Path F MiniLM forward iGPU-vs-CPU sweep as a ``DataFrame``.

    Columns: ``backend, batch, seq_len, fwd_ms, tokens_per_s, embeddings_per_s,
    device, cpu_threads``. ``backend`` is ``cpu`` (onnxruntime, all Zen 5
    threads) or ``rocm`` (MIGraphX gfx1151). ``fwd_ms`` is the best-of-N forward
    latency. Defaults to :data:`AI_INFER_CSV`.
    """
    return _read_csv(
        path or AI_INFER_CSV,
        numeric=("batch", "seq_len", "fwd_ms", "tokens_per_s",
                 "embeddings_per_s", "cpu_threads"),
    )
```

**A4. Plotter** — add after `plot_speedup` (~line 750). Signature
`plot_ai_inference(df=None) -> Figure`. Draws TWO panels: (1) MiniLM forward
latency bars, CPU vs iGPU per `batch×seq` workload (lower = better); (2) speedup
`cpu_ms/gpu_ms` per workload with a break-even line at 1.0 and a note that the
iGPU wins from the smallest workload because a transformer forward is
compute-bound (unlike Path E's size-gated ec-gpu OpenCL MSM). Headless-safe via `_get_plt()`.

```python
def plot_ai_inference(df=None):
    """Plot the Path F MiniLM forward iGPU-vs-CPU story; return the ``Figure``.

    Two panels: (1) forward latency (ms) bars, CPU(onnxruntime, all Zen 5
    threads) vs iGPU(MIGraphX gfx1151), one bar pair per ``batch×seq`` workload;
    (2) speedup ``cpu_ms/gpu_ms`` per workload with a break-even line at 1.0. The
    iGPU wins from the smallest workload (a transformer forward is compute-bound
    dense GEMM — unlike Path E's size-gated ec-gpu OpenCL MSM). Defaults to
    :func:`load_ai_inference`. HONESTY: this accelerates the AI MODEL forward,
    not the proof.
    """
    if df is None:
        df = load_ai_inference()
    plt = _get_plt()
    import numpy as np

    cpu = df[df["backend"] == "cpu"].set_index(["batch", "seq_len"])["fwd_ms"]
    gpu = df[df["backend"] == "rocm"].set_index(["batch", "seq_len"])["fwd_ms"]
    keys = sorted(set(cpu.index) & set(gpu.index))
    labels = [f"b{b}\u00b7s{s}" for b, s in keys]
    cpu_ms = [float(cpu[k]) for k in keys]
    gpu_ms = [float(gpu[k]) for k in keys]
    speed = [c / g if g else 0.0 for c, g in zip(cpu_ms, gpu_ms)]
    dev = next((d for d in df[df["backend"] == "rocm"]["device"]
                if isinstance(d, str) and d), "gfx1151")
    threads = next((int(t) for t in df[df["backend"] == "cpu"]["cpu_threads"]
                    if t == t), "?")
    x = np.arange(len(labels))
    w = 0.38

    fig, (ax_t, ax_sp) = plt.subplots(1, 2, figsize=(14, 5))
    ax_t.bar(x - w / 2, cpu_ms, w, label=f"CPU {threads}t (onnxruntime)",
             color="tab:orange")
    ax_t.bar(x + w / 2, gpu_ms, w, label=f"iGPU {dev} (MIGraphX)", color="tab:blue")
    ax_t.set_title("MiniLM forward latency (lower = better)")
    ax_t.set_ylabel("forward wall time (ms)")
    ax_t.set_xticks(x); ax_t.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_t.grid(True, axis="y", ls=":", alpha=0.5)
    ax_t.legend(fontsize=9)

    ax_sp.plot(x, speed, marker="o", color="tab:green")
    ax_sp.axhline(1.0, color="gray", ls="--", alpha=0.7)
    ax_sp.text(0, 1.02, "break-even (iGPU == CPU)", fontsize=8, color="gray")
    ax_sp.annotate("compute-bound dense GEMM:\niGPU wins from the smallest "
                   "workload\n(contrast Path E size-gated OpenCL MSM)",
                   xy=(0.5, 0.1), xycoords="axes fraction", fontsize=8, color="#444")
    ax_sp.set_title("Speedup = CPU / iGPU  (>1 \u21d2 iGPU forward wins)")
    ax_sp.set_ylabel("speedup x")
    ax_sp.set_xticks(x); ax_sp.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_sp.set_ylim(bottom=0)
    ax_sp.grid(True, axis="y", ls=":", alpha=0.5)

    fig.suptitle(
        f"Path F — all-MiniLM-L6-v2 forward on AMD {dev} (MIGraphX/ROCm) vs "
        f"{threads}-thread Zen 5 — iGPU accelerates the AI MODEL, not the proof",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
```

---

## (b) `Makefile` target `demo-f-embed`

**B1.** Add `demo-f-embed` to the `.PHONY` list (the line that already has
`demo-e-msm demo-e-groth16`).

**B2.** Add a variable near `GPU_ZK_DEMO` (~line 22):

```make
AI_INFER_DEMO := $(REPO_ROOT)/poc/amd-ai-inference-demo
```

**B3.** Add the help line (in the `help:` block, after the `demo-e-groth16` line):

```make
	@echo "  make demo-f-embed        # Path F: MiniLM all-MiniLM-L6-v2 forward on the iGPU (MIGraphX/ROCm) vs CPU -> ai-inference.{csv,png} (iGPU accelerates the AI MODEL, not the proof)"
```

**B4.** Add the target (next to the Path E `demo-e-*` recipes, ~line 166).
**Tier-0 readiness note**: `demo-f-embed` reuses the **same** read-only ROCm
readiness gate as Path E — `make gpu-zk-probe` (`GPU-ZK-READY` ⇒ ROCm + `gfx1151`
enumerated). `run-all.sh` additionally needs the ROCm **MIGraphX** python
bindings at `/opt/rocm-*/lib` (it adds them to `PYTHONPATH` itself). No new
probe target is required; `gpu-zk-probe` is the Tier-0 gate.

```make
# Path F: the all-MiniLM-L6-v2 forward on the Radeon iGPU (MIGraphX/ROCm 7.2.3,
# gfx1151) vs the 32-thread CPU (onnxruntime, same ONNX) -> ai-inference.{csv,png}.
# Tier-0 readiness: `make gpu-zk-probe` (GPU-ZK-READY). This accelerates the AI
# MODEL forward pass (embedding/prefill), NOT any ZK proof — EZKL/RISC0 proving
# stays CPU-only on AMD. Tune with e.g. `BATCHES=1,8,32 SEQ_LENS=32,128,256`.
demo-f-embed:
	bash $(AI_INFER_DEMO)/scripts/run-all.sh
```

---

## (c) `scripts/run-on-halo.sh` opt-in `--ai-infer`

**C1. Flag var** (with the other opt-in vars, ~line 49):
`RUN_BENCH=0 RUN_FOLDING=0 RUN_NPU=0 RUN_AI=0`

**C2. Arg parse** (in the `case` block, with `--npu-probe)`):
```bash
        --ai-infer)     RUN_AI=1 ;;
```

**C3. Help/comment**: add `--ai-infer  make demo-f-embed (iGPU MiniLM forward vs CPU; AI model, not the proof)` to the opt-in flags list in the header comment (~line 18 and the `--bench --folding --npu-probe` summary at ~line 33).

**C4. Stage** (with the other opt-in stages, ~line 168, after the `--npu-probe` line):
```bash
[[ "${RUN_AI}"     == "1" ]] && timed_target "demo-f-embed" make -C "${REPO_ROOT}" demo-f-embed
```

**C5. full-run.info fields.** Add this metrics block in step 4 (alongside the
other opt-in stage metrics, ~line 209, after the folding/npu blocks). Every read
is guarded so a not-run stage prints `n/a` and never errors:
```bash
AI_CSV="${REPO_ROOT}/poc/amd-ai-inference-demo/artefacts/ai-inference.csv"
if [[ -f "${AI_CSV}" ]]; then
    AI_ROWS=$(( $(wc -l < "${AI_CSV}" 2>/dev/null || echo 1) - 1 )); (( AI_ROWS < 0 )) && AI_ROWS=0
    AI_PEAK_TPS="$(awk -F, 'NR>1 && $1=="rocm"{if($5+0>m)m=$5+0} END{printf "%.0f", m}' "${AI_CSV}")"
    AI_BEST_SPD="$(awk -F, 'NR>1{if($1=="cpu")c[$2","$3]=$4; if($1=="rocm")g[$2","$3]=$4} END{m=0; for(k in g) if(c[k]>0 && g[k]>0){s=c[k]/g[k]; if(s>m)m=s} printf "%.1f", m}' "${AI_CSV}")"
else
    AI_ROWS="n/a"; AI_PEAK_TPS="n/a"; AI_BEST_SPD="n/a"
fi
```
And add these lines to the `## Next-step stages` heredoc block (after the npu lines):
```bash
    echo "ai_infer.csv_rows=${AI_ROWS}"
    echo "ai_infer.backend=${AI_GPU_BACKEND:-n/a}"
    echo "ai_infer.best_speedup_igpu_vs_cpu=${AI_BEST_SPD}"
    echo "ai_infer.peak_tokens_per_s=${AI_PEAK_TPS}"
```

**C6. Summary + Targets loops.** Add `demo-f-embed` to BOTH `for t in ...` lists
(the `## Targets` writer ~line 237 and the final summary ~line 305) so it reads
`skipped` when `--ai-infer` was not passed:
```
for t in demo-b-full demo-b-full-rootfs demo-b-groth16 integration-full bench demo-c-fold npu-probe demo-f-embed; do
```
Keep `demo-f-embed` OUT of the exit gate (only `demo-b-full` gates the exit code),
exactly like `bench`/`demo-c-fold`/`npu-probe`.

---

## (d) Notebook integration (live-or-replay; same verdict on a laptop)

### `lab/01_zkml_embedding_ezkl.ipynb`

Replace the cell that *asserts* "the model would run on the iGPU/NPU" with a
real measured **live-or-replay** cell. The verdict string MUST be computed from
the DataFrame so it is identical live and replay (committed CSV is the source of
truth). Drop-in cell:

```python
import labkit as lk
lk.capability_badge()

def _ai_live():
    # LIVE: a tiny solo grid so the notebook stays quick; full grid is `make demo-f-embed`.
    import subprocess, os
    demo = lk.repo_path("poc", "amd-ai-inference-demo")
    env = dict(os.environ, BATCHES="1,8", SEQ_LENS="32,128", REPS="10", WARMUP="3")
    subprocess.run(["bash", str(demo / "scripts" / "run-all.sh")], check=True, env=env)
    return lk.load_ai_inference()

def _ai_replay():
    # REPLAY: the committed full sweep (source of truth).
    return lk.load_ai_inference()

df, mode = lk.live_or_replay(_ai_live, _ai_replay,
                             requires=[lk.has_rocm, lk.is_strix_halo],
                             label="Path F MiniLM forward iGPU-vs-CPU")

# Same verdict either way:
cpu = df[df.backend == "cpu"].set_index(["batch", "seq_len"]).fwd_ms
gpu = df[df.backend == "rocm"].set_index(["batch", "seq_len"]).fwd_ms
keys = sorted(set(cpu.index) & set(gpu.index))
speeds = [float(cpu[k]) / float(gpu[k]) for k in keys if float(gpu[k]) > 0]
peak_tps = float(df[df.backend == "rocm"].tokens_per_s.max())
lo, hi = (min(speeds), max(speeds)) if speeds else (0, 0)
print(f"[{mode}] VERDICT: iGPU gfx1151 (MIGraphX/ROCm) runs the all-MiniLM-L6-v2 "
      f"forward {lo:.1f}x–{hi:.1f}x faster than the {int(df[df.backend=='cpu'].cpu_threads.iloc[0])}-thread "
      f"Zen 5 CPU (peak {peak_tps:,.0f} tokens/s). This accelerates the AI MODEL "
      f"forward, NOT the proof (EZKL/RISC0 proving stays CPU-only on AMD).")
lk.plot_ai_inference(df);
```

Notes: `requires=[lk.has_rocm, lk.is_strix_halo]` ⇒ a laptop (no ROCm) takes the
REPLAY branch automatically; `make lab-replay` sets `LAB_FORCE_REPLAY=1` so even
the Strix Halo replays the committed CSV (no heavy live run in CI). The verdict
text and the plot are byte-for-byte determined by the DataFrame, so they read
the same in both modes.

### `lab/00_amd_engine_map.ipynb`

Add the iGPU **AI-inference** row to the engine map as *measured* (it currently
only has CPU proving + Path E primitives). Replay-safe drop-in cell:

```python
import labkit as lk
df = lk.load_ai_inference()
cpu = df[df.backend == "cpu"].set_index(["batch", "seq_len"]).fwd_ms
gpu = df[df.backend == "rocm"].set_index(["batch", "seq_len"]).fwd_ms
keys = sorted(set(cpu.index) & set(gpu.index))
spd = [float(cpu[k]) / float(gpu[k]) for k in keys if float(gpu[k]) > 0]
print("AMD Strix Halo engine map — measured:")
print(f"  Zen5 CPU (32t)   : ZK proving (STARK/Groth16/folding) — the proof path")
print(f"  iGPU gfx1151     : AI-model forward  {min(spd):.1f}x–{max(spd):.1f}x vs CPU "
      f"(MiniLM, MIGraphX)  [Path F]")
print(f"  iGPU gfx1151     : SNARK primitives  NTT — BLS12-381 FFT vs blstrs parallel_fft: "
      f"above parity across the sweep, peak 5.553x@2^22, NO bound claimed "
      f"(the same iGPU's BN254 Fr NTT vs arkworks loses at 2^18, 0.963x) / "
      f"MSM size-gated by the ec-gpu OpenCL kernel  [Path E]")
print(f"  XDNA2 NPU        : DRIVER-READY, no dispatch yet  [Path D]")
print("  honesty: iGPU/NPU accelerate the AI MODEL; the stock r0vm STARK has no upstream AMD GPU prover (scoped v2.3.2 fork: iGPU segment-STARK, path-i).")
lk.plot_ai_inference(df);
```

---

## (e) `docs/IMPLEMENTATION-STATUS.md` — exact section to paste

**E1.** Add a row to the §0 TL;DR table (after the Path E row):

```markdown
| **Path F — AMD iGPU 加速 AI 推論**（2026-06-11 新增；2026-06-18 clean-solo 重測校正） | ✅ **實跑通過** | iGPU(gfx1151) 經 MIGraphX/ROCm 跑 all-MiniLM-L6-v2 前向:**~1.9–10.6x** 快過 32-thread CPU(同一份 ONNX;舊 9–25x 是 contention-inflated CPU baseline);加速的是 AI 模型前向、**非** proof（見 §11) |
```

**E2.** Append this new section at the end (after §10 Path E):

```markdown
---

## 11. Path F — AMD iGPU 加速 AI 模型推論（2026-06-11 新增,實跑通過)

坐實 repo 一直宣稱、卻從未在本機量過的 G1:「iGPU 加速 AI 模型」。用 **MIGraphX**（ROCm 7.2.3 隨附、原生認得 gfx1151）跑 `all-MiniLM-L6-v2`（22.6M params、hidden 384/6 層/12 heads，random-init,量延遲/吞吐非精度)前向,對照 32-thread Zen 5（onnxruntime CPU EP,**同一份 ONNX**)。`fwd_ms`=best-of-20。

| backend | batch×seq | fwd_ms | speedup(iGPU vs CPU) | iGPU tokens/s |
|---|---|---:|---:|---:|
| iGPU gfx1151 (MIGraphX) | 1×32 | 1.02 | **1.90x** | 31,240 |
| iGPU gfx1151 (MIGraphX) | 32×32 | 3.79 | **10.56x** | 270,285 |
| iGPU gfx1151 (MIGraphX) | 32×128 | 15.89 | **4.52x** | 257,707 |
| iGPU gfx1151 (MIGraphX) | 32×256 | 42.41 | 4.91x | 193,175 |
| CPU Zen5 32t (onnxruntime) | 32×128 | 71.88 | — (baseline) | — |

- **iGPU 全程 ~1.9–10.6x 領先**(低端 1×32 1.90x、峰值 32×32 10.56x),峰值 **270,285 tokens/s** / **8,446 embeddings/s**。transformer 前向是 compute-bound dense GEMM,所以從最小 workload 就贏(對照 Path E 的 MSM 在 ec-gpu OpenCL 路徑上 size-gated，只在大 size parity)。
- **誠實 note**:這是 **2026-06-18 乾淨 solo 重測**(iGPU 0% busy、loadavg 1.59)。先前的 9–25x 是 CPU baseline 被併發干擾**膨脹**的錯誤值,現校正回乾淨 solo——與 `docs/INTEGRITY-REPORT.md` 的 `1.34x→0.70x→0.994x` folding 兩段更正同一類(folding 第二段是**往上**修回打平);durable 備份 `ai-inference.solo-rebench-2026-06-18`。
- **honesty rule 不變**:iGPU 加速的是 **AI 模型前向**(DEAAP 的 embedding/prefill),**不是** proof——prover time / proof size / on-chain gas 仍由 32 Zen 5 threads + unified RAM 扛。

| Tier | 狀態 | 實測 evidence |
|---|---|---|
| Tier 0 `make gpu-zk-probe` | ✅ READY | 沿用 Path E 的 ROCm/gfx1151 readiness gate |
| `make demo-f-embed` | ✅ PASS | MiniLM 前向 iGPU **~1.9–10.6x** vs CPU;`poc/amd-ai-inference-demo/artefacts/ai-inference.{csv,md,png,log}` |

完整寫法見 [`../reading-notes/path-f-amd-ai-inference.md`](../../reading-notes/path-f-amd-ai-inference.md) 與 demo [`../poc/amd-ai-inference-demo/`](.)。Commits(本地,未 push):`feat(path-f)` — demo harness + artefacts + reading-note。
```

**E3. (optional) `docs/amd-strix-halo-acceleration.md` engine matrix** — flip the
iGPU AI-inference cell from "claimed/experimental" to **measured**: "iGPU
`gfx1151` accelerates the AI-model forward (all-MiniLM-L6-v2) **~1.9–10.6x** vs the
32-thread CPU via MIGraphX/ROCm 7.2.3 — Path F; the proof path stays CPU-only."
And the `README.md` workload→engine table: add "AI-model embedding forward →
iGPU (measured, Path F, ~1.9–10.6x)".

---

## Reproduce / sanity check before wiring

```bash
make demo-f-embed          # after E-applying the Makefile target, or:
bash poc/amd-ai-inference-demo/scripts/run-all.sh
# labkit smoke (after A-applying):
.venv/bin/python -c "import sys; sys.path.insert(0,'lab'); import labkit as lk; print(lk.load_ai_inference().head()); lk.plot_ai_inference()"
```
