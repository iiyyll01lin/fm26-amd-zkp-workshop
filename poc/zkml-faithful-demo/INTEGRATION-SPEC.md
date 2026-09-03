# INTEGRATION-SPEC — Demo G (G4 faithful zkLLM / zkRAG)

> For the **closeout agent** that owns the shared files. This Phase-4 agent did
> **not** touch `lab/labkit.py`, `Makefile`, `scripts/run-on-halo.sh`,
> `docs/IMPLEMENTATION-STATUS.md`, `README.md`, `docs/amd-strix-halo-acceleration.md`,
> or any `lab/*.ipynb`. Everything below is ready to paste; all artefact paths,
> JSON keys and numbers are real and committed under `poc/zkml-faithful-demo/`.

All measured numbers were produced on the AMD Strix Halo (Ryzen AI MAX+ 395,
94 GB, kernel 6.17.0-29), CPU-only proving, 2026-06-11.

---

## (a) labkit additions (keep minimal)

The Demo-G artefacts are **JSON**, not CSV — so add lightweight JSON loaders
(no pandas required). Suggested placement: alongside the other path constants
and after the CSV loaders.

### Path constants (add near `GPU_GROTH16_CSV`)

```python
_ZKML_DEMO = REPO_ROOT / "poc" / "zkml-faithful-demo"

#: G4 zkLLM — EZKL attention-block prove metrics (logrows, prove_seconds, proof_bytes).
ZKLLM_PROVE_INFO: Path = _ZKML_DEMO / "zkllm" / "artefacts" / "prove.info"
#: G4 zkLLM — EZKL circuit settings (run_args.logrows, input/param scale).
ZKLLM_SETTINGS:  Path = _ZKML_DEMO / "zkllm" / "artefacts" / "settings.json"
#: G4 zkLLM — base-b exp (tlookup) decomposition prototype metrics.
ZKLLM_TLOOKUP:   Path = _ZKML_DEMO / "zkllm" / "artefacts" / "tlookup.json"
#: G4 zkRAG — committed receipt journal (top-k ids/dists, recall, digests).
ZKRAG_JOURNAL:   Path = _ZKML_DEMO / "zkrag" / "artefacts" / "zkrag.journal.json"
#: G4 zkRAG — prove/verify metrics (cycles, prove_seconds, receipt_bytes, dev_mode).
ZKRAG_PROOF_INFO: Path = _ZKML_DEMO / "zkrag" / "artefacts" / "zkrag.proof.info"
```

Add these names to `__all__`.

### Loaders (add after `load_gpu_groth16`)

```python
import json as _json

def _read_json(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"labkit: expected committed JSON at {path}")
    return _json.loads(path.read_text())

def load_zkllm_prove_info(path: Optional[Path] = None) -> dict:
    """G4 zkLLM EZKL metrics: keys logrows, witness_seconds, prove_seconds, proof_bytes."""
    return _read_json(path or ZKLLM_PROVE_INFO)

def load_zkllm_tlookup(path: Optional[Path] = None) -> dict:
    """G4 zkLLM tlookup prototype: keys params, table_sizes, reconstruction."""
    return _read_json(path or ZKLLM_TLOOKUP)

def load_zkrag_journal(path: Optional[Path] = None) -> dict:
    """G4 zkRAG receipt journal: keys n, d, k, ef, num_visited, top_ids, top_dists, recall, pq_monotone."""
    return _read_json(path or ZKRAG_JOURNAL)

def load_zkrag_proof_info(path: Optional[Path] = None) -> dict:
    """G4 zkRAG metrics: keys total_cycles, user_cycles, segments, prove_seconds, verify_seconds, receipt_bytes, dev_mode."""
    return _read_json(path or ZKRAG_PROOF_INFO)
```

Add the four loader names to `__all__`.

### Predicates (reuse existing — none new needed)

- zkLLM live needs: `has_docker()` (or `has_ezkl()` if running bare-metal).
- zkRAG live needs: `has_r0vm()` (already present). For the zkRAG `--verify-only`
  replay there is no extra predicate — it only reads the committed receipt.

### Optional plotter (only if you want a figure; otherwise a table is fine)

A single horizontal-bar/summary is enough. Minimal signature, matching the
other plotters' headless-safe style:

```python
def plot_zkml_faithful_summary(save: Optional[Path] = None):
    """One figure summarising G4: zkLLM prove_s/proof_KB + zkRAG prove_s/visited.
    Reads load_zkllm_prove_info() + load_zkrag_journal()/load_zkrag_proof_info()."""
```

There is **no** new CSV; do not add a CSV loader for Demo G.

---

## (b) Makefile targets

Add a Demo-G block. Exact commands (run from repo root):

```makefile
# ---- Demo G (G4): faithful zkLLM / zkRAG step-up -------------------------
.PHONY: demo-g-zkllm demo-g-zkrag demo-g demo-g-replay

demo-g-zkllm:           ## G4 zkLLM: EZKL Halo2 proof of a MiniLM attention sub-block
	cd poc/zkml-faithful-demo && ./scripts/run-all.sh zkllm

demo-g-zkrag:           ## G4 zkRAG: RISC0 zkVM HNSW top-k/membership proof (real STARK)
	cd poc/zkml-faithful-demo && ./scripts/run-all.sh zkrag

demo-g: demo-g-zkllm demo-g-zkrag  ## G4: run both faithful zkML demos

demo-g-replay:          ## G4: verify committed proofs only (fast; no heavy prove)
	cd poc/zkml-faithful-demo && ./scripts/run-all.sh replay
```

Notes for the closeout agent:
- `demo-g-zkllm` builds the `zkml-faithful-zkllm` Docker image on first run
  (~2-3 min) then runs the EZKL pipeline (~25 s). Override the image name with
  `ZKLLM_IMAGE=...`. An already-built image with the identical stack
  (`deaap-embedding-zkp:fixed`, EZKL 22.3.0) also works.
- `demo-g-zkrag` needs the RISC0 toolchain + a Rust toolchain on PATH; real
  STARK prove is ~137 s. The env preamble is the repo standard:
  `. "$HOME/.cargo/env"; export PATH="$HOME/.risc0/bin:$PATH"`.

### `scripts/run-on-halo.sh` opt-in stage (optional, mirrors `--ai-infer` style)

Add an opt-in `--zkml-faithful` flag that runs `make demo-g` and records, in
`full-run.info`, the fields: `zkllm_logrows`, `zkllm_prove_s`,
`zkllm_proof_bytes`, `zkrag_prove_s`, `zkrag_recall`, `zkrag_visited`,
`zkrag_receipt_bytes`. These are all readable from the two `*.info`/journal JSONs
above. Keep it off the default path (research-grade, ~3 min).

---

## (c) Notebook integration

Recommended: **two new notebooks** (keeps notebook 01 focused on the Demo-A
linear proxy; these are the explicit step-up).

- `lab/07_zkllm_attention_ezkl.ipynb`
- `lab/08_zkrag_hnsw_zkvm.ipynb`

Each follows the established live-or-replay pattern. **Default behaviour under
`make lab-replay` (which sets `LAB_FORCE_REPLAY=1`) must be replay**, reading the
committed JSON — never building Docker or proving a STARK in CI.

### `07_zkllm_attention_ezkl.ipynb` — core cell

```python
import labkit as lk

def live():   # heavy: rebuild + EZKL prove (Docker). Gated off in lab-replay.
    import subprocess
    subprocess.run(["bash", "scripts/run-all.sh", "zkllm"],
                   cwd=lk.repo_path("poc/zkml-faithful-demo"), check=True)
    return lk.load_zkllm_prove_info()

def replay():  # committed artefacts — the source of truth
    return lk.load_zkllm_prove_info()

info, mode = lk.live_or_replay(live, replay,
                               requires=[lk.has_docker],
                               label="zkLLM attention-block EZKL proof")
tl = lk.load_zkllm_tlookup()
print(f"[{mode}] logrows={info['logrows']} prove={info['prove_seconds']}s "
      f"proof={info['proof_bytes']:,} B")
print(f"tlookup table {tl['table_sizes']['tlookup_total_entries']} vs naive "
      f"{tl['table_sizes']['naive_flat_exp_table_entries']} "
      f"({tl['table_sizes']['compression_ratio']}x smaller)")
```

> Markdown narration to include: this proves a **real attention head incl.
> softmax** (not the linear proxy); softmax → Halo2 lookup tables (same family as
> zkLLM `tlookup`); see the README "gap to full paper".

### `08_zkrag_hnsw_zkvm.ipynb` — core cell

```python
import labkit as lk

def live():   # heavy: real STARK prove (~137 s). Gated off in lab-replay.
    import subprocess
    subprocess.run(["bash", "zkrag/scripts/run-all.sh"],
                   cwd=lk.repo_path("poc/zkml-faithful-demo"), check=True)
    return lk.load_zkrag_journal(), lk.load_zkrag_proof_info()

def replay():  # committed receipt-derived JSON
    return lk.load_zkrag_journal(), lk.load_zkrag_proof_info()

(journal, pinfo), mode = lk.live_or_replay(
    live, replay, requires=[lk.has_r0vm],
    label="zkRAG HNSW top-k/membership STARK")
print(f"[{mode}] n={journal['n']} visited={journal['num_visited']}/{journal['n']} "
      f"recall={journal['recall']}/{journal['k']} pq_monotone={journal['pq_monotone']}")
print(f"prove={pinfo['prove_seconds']:.1f}s verify={pinfo['verify_seconds']:.3f}s "
      f"cycles={pinfo['total_cycles']} receipt={pinfo['receipt_bytes']:,} B "
      f"dev_mode={pinfo['dev_mode']}")
```

> Optional stronger live replay (re-runs the actual zkVM verifier on the
> committed receipt in ~30 ms instead of just reading JSON):
> `bash zkrag/scripts/run-all.sh --verify-only`.

**Committed-artefact replay paths** (what every replay cell reads):
`poc/zkml-faithful-demo/zkllm/artefacts/{prove.info,tlookup.json,settings.json,proof.json}`
and `poc/zkml-faithful-demo/zkrag/artefacts/{zkrag.journal.json,zkrag.proof.info}`.

If you would rather **extend notebook 01** than add new ones: append a single
"Step-up: attention + HNSW (G4)" section at the end using the two cells above;
do not modify the existing Demo-A linear-proxy cells.

### `lab/00_amd_engine_map.ipynb`

Add one row to the engine/workload map: `zkLLM attention sub-block (EZKL Halo2)`
and `zkRAG HNSW search (RISC0 STARK)` → both **CPU-only proving** on AMD (matches
the honesty rule; iGPU/NPU do not prove these).

---

## (d) IMPLEMENTATION-STATUS.md — section to paste

Add as a new section (e.g. **§11**). Numbers are real (2026-06-11, Strix Halo).

```markdown
## 11. Demo G — Faithful zkLLM / zkRAG step-up (G4, 2026-06-11, research-grade)

把 G4 從「reading-note + 384→64 linear proxy」推進到 **真實、端到端、reduced-scale** 的兩份證明（皆 CPU-only proving，符合 honesty rule）。產物在 `poc/zkml-faithful-demo/`。

| 項目 | 狀態 | 實測 evidence |
|---|---|---|
| zkLLM — MiniLM **attention sub-block（含 softmax）** through EZKL Halo2 | ✅ **prove+verify PASS** | seq=8 / d_model=384 / d_head=32 單頭 attention；logrows=16；calibrate `target=accuracy` 量化誤差 **mean abs % = 0.037%**；witness 0.27 s、**prove 7.85 s**、`PROOF VERIFIED`、proof **545,679 B**；artefacts: `zkllm/artefacts/{attention.onnx,settings.json,vk.key,proof.json,prove.info,witness.json}` |
| zkLLM — `tlookup` base-b exp 分解 prototype（plaintext） | ✅ | exp 用 per-digit 小表乘積重建：表大小 **48 vs 598（12.5× 壓縮）**、softmax 重建誤差 ≤1.2e-4；`zkllm/artefacts/tlookup.json` |
| zkRAG — **HNSW top-k / membership** in RISC0 zkVM（真 STARK） | ✅ **prove+verify PASS** | n=256 / d=16 / k=5 / ef=20；navigation **visited 133/256（~48% pruned）**、**recall=5/5**、`pq_monotone=true`；cycles total 2,097,152（2 segments）；**prove 137.3 s**、**verify 0.026 s**、receipt **563,340 B**、journal 360 B；committed receipt 重驗 31 ms（`--verify-only`）；artefacts: `zkrag/artefacts/{zkrag.receipt.bin,zkrag.journal.json,zkrag.proof.info,zkrag.index.json}` |

**zkRAG 的四個 in-circuit check 對應論文四個 PIOP component**：(1) priority-queue checker = beam-worst 單調非增；(2) hybrid lookup = 展開的邊必為 committed graph 合法 neighbor；(3) distance check = 整數 L2 重算；(4) membership + top-k = 回傳 id 為合法 member 且 `recall==k`（對 brute-force 真值）。

**誠實的 gap-to-paper（務必照講，勿 over-claim）**：
- **zkLLM**：證的是「**單一** attention head + softmax」，softmax 走 EZKL **generic per-op Halo2 lookup**；**未**實作 zkLLM 的 `tlookup`（tensorised logUp/cq 批次化）與 `zkAttn` sumcheck/GKR（只在 plaintext prototype 了 base-b 表壓縮 trick），也**未**證整顆多層 LLM。weights 為 seeded（架構/scaling/softmax 與 MiniLM head-0 一致，換成 trained-HF 權重是 one-liner，不改電路）。
- **zkRAG**：用 **general zkVM（RISC0 STARK）重跑 search trace** 來證 top-k/membership 關係——這正是論文要超越的 baseline；**未**實作論文的 custom HNSW PIOP（polynomial heap-invariant、cq/logUp hybrid lookup、membership selector vector、batched sumcheck），scale 為 256×16-dim（非 1M×128-dim）。

> 完整方法、layout、重現指令與 gap 表見 `poc/zkml-faithful-demo/README.md` 與
> `poc/zkml-faithful-demo/INTEGRATION-SPEC.md`；reading-note 更新見
> `reading-notes/zkllm-summary.md` / `reading-notes/zkrag-summary.md` 的「G4」尾段與
> `reading-notes/path-g-zkml-faithful.md`。
```

Also update the §0 TL;DR table with one row:

```markdown
| **Demo G — faithful zkLLM/zkRAG（G4）** | ✅ **reduced-scale 端到端通過** | EZKL 證 MiniLM attention+softmax（prove 7.85 s / `PROOF VERIFIED`）；RISC0 zkVM 證 HNSW top-k/membership（真 STARK，recall 5/5，verify 26 ms）；gap-to-paper 已明列 |
```

---

## Files this agent created (all under `poc/zkml-faithful-demo/`, no shared files touched)

- `README.md`, `INTEGRATION-SPEC.md`, `.gitignore`, `scripts/run-all.sh`
- `zkllm/`: `Dockerfile`, `.dockerignore`, `requirements.txt`,
  `src/{01_make_attention,02_setup,03_prove,04_verify,tlookup_prototype}.py`,
  `scripts/{run-all,clean}.sh`, committed `artefacts/*` (no `pk.key`/`*.srs`).
- `zkrag/`: `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`,
  `core/`, `methods/`, `host/`, `scripts/run-all.sh`, committed `artefacts/*`.
- Reading notes (not shared/owned files): appended a "G4" section + status
  comment to `reading-notes/zkllm-summary.md` and `reading-notes/zkrag-summary.md`,
  and added `reading-notes/path-g-zkml-faithful.md`.
