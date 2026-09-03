# INTEGRATION-SPEC — zkLLM AMD engine-split demo

> Paste-ready wiring for `lab/labkit.py`, the top-level `Makefile`, the lab
> notebook, and the lab nav/engine-map. All artefact paths, CSV columns and JSON
> keys below are real and committed under `poc/zkllm-amd-split-demo/artefacts/`.
> Measured on the AMD Strix Halo (Ryzen AI MAX+ 395, 94 GB, kernel 6.17.0-29),
> CPU-only proving, 2026-06-11.

This PoC **reuses, does not duplicate**: the proven attention head + EZKL proof
come from `poc/zkml-faithful-demo`; the iGPU forward pattern from
`poc/amd-ai-inference-demo`; the BN254 MSM evidence + EZKL-on-AMD blocker from
`poc/amd-gpu-zk-primitive-demo`.

---

## (a) Artefact schema (the contract that threads through every component)

### `artefacts/attention-forward.csv` (Stage 1 — produced here)

```
backend,batch,seq_len,fwd_ms,tokens_per_s,device,cpu_threads
```

`backend` ∈ {`cpu` (onnxruntime, all Zen 5 threads), `rocm` (MIGraphX gfx1151)};
`fwd_ms` is the best-of-N forward latency; the iGPU rows have an empty
`cpu_threads`. Same column family as Path F's `ai-inference.csv` (minus
`embeddings_per_s`, which is meaningless for a sub-block).

### `artefacts/split.json` (the synthesised 3-engine story)

Flat contract keys (read directly): `forward_ms_igpu`, `forward_ms_cpu`,
`forward_speedup`, `prove_seconds`, `verify_seconds`, `verify_status`,
`proof_bytes`, `msm_speedup_min`, `msm_speedup_max`, `msm_blocker`.
Structured detail: `stage1_forward` (incl. `grid[]` + `best_igpu_speedup`),
`stage2_prove`, `stage3_msm_frontier` (incl. `sizes[]`), `timeline[]`,
`verdict{}` (per engine), `honesty`, `captured{}`, `host{}`.

---

## (b) labkit additions (already applied — listed for review)

Path constants (near `FULL_RUN_INFO`):

```python
_ZKLLM_SPLIT_DEMO = REPO_ROOT / "poc" / "zkllm-amd-split-demo"
ATTENTION_FORWARD_CSV: Path = _ZKLLM_SPLIT_DEMO / "artefacts" / "attention-forward.csv"
ZKLLM_SPLIT_JSON:      Path = _ZKLLM_SPLIT_DEMO / "artefacts" / "split.json"
```

Loaders + plotter (added to `__all__`):

```python
load_attention_forward(path=None) -> pandas.DataFrame   # the Stage-1 CSV
load_zkllm_split(path=None)        -> dict               # split.json
plot_zkllm_amd_split(split=None, save=None) -> Figure    # timeline + iGPU MSM frontier inset
```

`plot_zkllm_amd_split` is headless-safe (uses labkit's `_get_plt()` + reuses
`load_gpu_bn254()` for the inset). No new predicates needed — the notebook reuses
`has_rocm` (forward live) and `has_docker` (delegated prove).

---

## (c) Makefile targets (already applied)

```makefile
ZKLLM_SPLIT_DEMO := $(REPO_ROOT)/poc/zkllm-amd-split-demo

demo-zkllm-split:          ## zkLLM engine-split: forward (iGPU vs CPU) + synth + plot
	cd $(ZKLLM_SPLIT_DEMO) && ./scripts/run-all.sh forward && ./scripts/run-all.sh synth && ./scripts/run-all.sh plot

demo-zkllm-split-replay:   ## zkLLM engine-split: reproduce split.json/png from committed artefacts (no GPU/Docker)
	cd $(ZKLLM_SPLIT_DEMO) && ./scripts/run-all.sh replay
```

`demo-zkllm-split` needs ROCm/MIGraphX for the iGPU forward row (CPU baseline
otherwise); `demo-zkllm-split-replay` is CPU-only and Docker-free.

---

## (d) Notebook (already applied) — `lab/10_zkllm_amd_split.ipynb`

> Numbered **10** (07 = zkLLM attention, 08 = zkRAG HNSW, 09 = zkRAG e2e already
> exist). Footer cross-links thread `08 → 09 zkRAG e2e → 10 zkLLM engine-split`;
> default under `make lab-replay` (`LAB_FORCE_REPLAY=1`) is **pure replay** — no
> Docker, no GPU, no torch.

Cell skeleton: story + honesty → `lk.capability_badge()` → `live_or_replay`
forward (live `run-all.sh forward` gated by `[_heavy, has_rocm]`; replay
`load_attention_forward()`) → `live_or_replay` proof (live `run-all.sh prove`
gated by `[_heavy, has_docker]`; replay `load_zkllm_prove_info()`) →
`load_zkllm_split()` + `plot_zkllm_amd_split()` + per-engine verdict →
`engine → stage → evidence` table → "What this shows about AMD" + Sources footer.

Engine-map / nav (`lab/00_amd_engine_map.ipynb`, `lab/README.md`): notebook 10
added as one row — *"zkLLM engine-split — iGPU forward (size-gated) · CPU Halo2
proof · iGPU MSM frontier"*.

---

## (e) Three-engine zkLLM panorama — paste-ready (DEFERRED: wire when the lab track settles)

> **Not applied yet — intentionally.** The `lab/*` notebooks + `lab/labkit.py` are
> in flight on another track, so this PoC does **not** edit them now (collision).
> The synth + figure already ship here (`bash scripts/run-all.sh three-engine` →
> `artefacts/three-engine.{json,png,md}`); below is the exact, copy-paste wiring
> for the lab track to drop in **LATER**.

### `artefacts/three-engine.json` schema (the contract)

Top-level: `schema_version`, `title`, `kind="capability-map"`, `claim`,
`representative_workload{model,d_model,seq_len}`, `host{}`, `sources{}`,
`stages[]`, `honesty`, `honesty_boundary[]`. Each `stages[]` entry has
`stage` (1|2|3), `name`, `engine`, `role`, `headline_number{}`, a per-stage
detail block, and an explicit `honesty` string. Stage-specific keys:

- **Stage 1** — `npu_int8{label,shape,peak_gflops,avg_gflops,min_us,result,precision}`,
  `cross_engine_forward{igpu,cpu,igpu_vs_cpu_speedup}`,
  `utilization{npu_rep_peak_gflops,npu_advertised_int8_tops,pct_of_advertised_ceiling_min/max}`.
- **Stage 2** — `representative{}`, `scale_curve[]` (head/mha/layer),
  `memory_ceiling_gb`, `cap{config,logrows,status,note}`, `proof_system`.
- **Stage 3** — `representative{}`, `crossover[]` (rows with `prove_speedup`,
  `bn254_msm_speedup`), `best{}`, `prove_speedup_min/max`, `crossover_pow`,
  `blocker`, `proof_system`.

### labkit path constant + loaders (paste near the existing `ZKLLM_SPLIT_JSON`)

```python
#: Three-engine zkLLM panorama (NPU int8 forward · CPU+94GB Halo2 proof · iGPU
#: OpenCL Groth16 MSM offload) — capability map, NOT one monolithic proof.
THREE_ENGINE_JSON: Path = _ZKLLM_SPLIT_DEMO / "artefacts" / "three-engine.json"


def load_three_engine(path: Path | None = None) -> dict:
    """The committed three-engine panorama (capability map). Stdlib-only."""
    import json
    return json.loads(Path(path or THREE_ENGINE_JSON).read_text())


def plot_three_engine(panorama: dict | None = None, save: Path | None = None):
    """Headless-safe render of the panorama (engine cards + iGPU proof crossover).

    Delegates to the PoC's own ``scripts/plot-three-engine.py`` so the figure
    stays identical to the committed artefact; falls back to ``three-engine.md``
    when matplotlib is absent.
    """
    import json
    import runpy
    import sys
    if panorama is None:
        panorama = load_three_engine()
    plotter = _ZKLLM_SPLIT_DEMO / "scripts" / "plot-three-engine.py"
    argv = sys.argv
    try:
        sys.argv = [str(plotter)]
        runpy.run_path(str(plotter), run_name="__main__")
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    return _ZKLLM_SPLIT_DEMO / "artefacts" / ("three-engine.png" if save is None else save)
```

Add `THREE_ENGINE_JSON`, `load_three_engine`, `plot_three_engine` to `__all__`.
No new predicate is needed — the panorama is pure replay from committed sources
(no `has_rocm`/`has_docker` gate).

### Notebook cell (paste into `lab/10_zkllm_amd_split.ipynb` — a NEW cell, replay-only)

```python
# --- Three-engine zkLLM panorama: NPU int8 forward · CPU Halo2 proof · iGPU Groth16 MSM ---
# Capability MAP of each Strix Halo engine's measured best contribution — NOT one
# monolithic proof. Pure replay from committed artefacts (no GPU/Docker/re-run).
pano = lk.load_three_engine()
rep = pano["representative_workload"]
print(f"Anchor workload: MiniLM d_model={rep['d_model']}, T={rep['seq_len']}  ·  {pano['kind']}")

import pandas as pd
rows = []
for s in pano["stages"]:
    hn = s.get("headline_number") or {}
    measure = (
        f"{hn.get('peak_gflops')} GFLOPs @ {hn.get('min_us')}us" if s["stage"] == 1 else
        f"{hn.get('prove_s')} s @ {hn.get('peak_rss_gb', '?')} GB" if s["stage"] == 2 else
        f"{s.get('prove_speedup_min')}->{s.get('prove_speedup_max')}x @ 2^{s.get('crossover_pow')}"
    )
    rows.append({"stage": s["stage"], "engine": s["engine"], "measure": measure})
display(pd.DataFrame(rows).set_index("stage"))

lk.plot_three_engine(pano)   # engine cards + iGPU proof crossover; -> three-engine.png

print("\nHonesty boundary:")
for b in pano["honesty_boundary"]:
    print(" -", b)
```

Engine-map / nav (when the lab track settles): extend notebook 10's row with the
panorama line — *"three-engine capability map: NPU int8 forward · CPU Halo2
proof (shipping) · iGPU OpenCL Groth16 MSM (re-cast, size-gated, never EZKL)"*.

---

## Files this PoC created (all under `poc/zkllm-amd-split-demo/`)

- `README.md`, `INTEGRATION-SPEC.md`, `requirements.txt`, `.gitignore`
- `scripts/{run-all.sh, plot-split.py, plot-three-engine.py}`
- `src/{01_attention_forward_bench.py, 02_synthesize.py, 05_three_engine.py}`
- committed `artefacts/{attention-forward.csv, attention-forward.log, split.json, split.png, split.md}`
- committed `artefacts/{three-engine.json, three-engine.png, three-engine.md}` (the NPU+CPU+iGPU panorama)

Shared files touched (small, additive): `lab/labkit.py` (constants + 2 loaders +
1 plotter + `__all__`), `lab/00_amd_engine_map.ipynb` (nav row + walk-the-map
link), `lab/README.md` (nav row + file tree), `Makefile` (2 targets + help +
`.PHONY`), `lab/10_zkllm_amd_split.ipynb` (new), and a cross-link in
`lab/08_zkrag_hnsw_zkvm.ipynb`.
