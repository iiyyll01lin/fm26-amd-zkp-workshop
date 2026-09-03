# Demo G — Faithful zkLLM / zkRAG step-up (research-grade)

> Plan gap **G4** (`phase4-zkllm-zkrag`). Goes beyond Demo A's `384 -> 64`
> *linear* proxy to (1) a real **MiniLM self-attention sub-block — including the
> softmax nonlinearity** — proved end-to-end through EZKL Halo2, and (2) a
> minimal **HNSW vector search** whose **top-k / membership-by-distance**
> relation is proved end-to-end in the **RISC0 zkVM** (real STARK).
>
> **Honesty up front.** This is the *largest* research gap in the repo. Full
> zkLLM (`tlookup` + `zkAttn` sumcheck/GKR over a 13B transformer) and full
> zkRAG (a custom HNSW PIOP) are multi-week constructions. What ships here is a
> **faithful, real, end-to-end proof at reduced scale**, with an explicit
> [gap-to-paper](#gap-to-the-full-papers) section. Nothing here claims to be the
> paper construction.

All proving here is **CPU-only on the AMD Strix Halo** (Ryzen AI MAX+ 395,
94 GB, kernel 6.17). EZKL Halo2 and RISC0 STARK proving are CPU-only on AMD —
that is expected and consistent with the rest of the repo's honesty rule.

---

## What is actually proved on this box

### zkLLM — MiniLM attention sub-block through EZKL Halo2

A single scaled-dot-product self-attention head with the **softmax
nonlinearity**, matching `all-MiniLM-L6-v2` head dims (`d_model=384`,
`d_head=32`, `seq=8`):

```
Q,K,V = X·Wq, X·Wk, X·Wv          S = (Q·Kᵀ)/√d_head
A = softmax(S)   <-- the nonlinear op a linear proxy skips
Y = A·V
```

Measured (EZKL 22.3.0, `target=accuracy`, scales `[7,11]`):

| metric | value |
|--------|-------|
| circuit size | **logrows = 16** (k=16) |
| quant fidelity | **mean abs % error = 0.037%** (max abs err 6.0e-4) |
| witness | 0.27 s |
| **prove** | **7.85 s** |
| **verify** | `PROOF VERIFIED` (exit 0) |
| proof size | **545,679 bytes** (~533 KB) |

The softmax is what makes this a *faithful* step-up: EZKL lowers it to Halo2
**lookup tables** (exp) + range checks — the same lookup-argument family zkLLM's
`tlookup` belongs to. `src/tlookup_prototype.py` additionally shows zkLLM's
**base-b exp decomposition** in the clear: it reconstructs `exp` as a product of
per-digit small-table lookups, achieving a **12.5× smaller table** (48 vs 598
entries) at ≤1.2e-4 softmax error.

### zkRAG — HNSW search top-k/membership in the RISC0 zkVM

A minimal HNSW-style navigable-graph search proved as a **real STARK**. The
guest re-executes greedy best-first navigation over a committed index and, in
the same circuit, checks the four relations that mirror zkRAG's four PIOP
components, then commits a public journal.

Measured (RISC0 2.3.2 / r0vm, `n=256`, `d=16`, `k=5`, `ef=20`, real STARK):

| metric | value |
|--------|-------|
| index | 256 vectors × 16-dim, graph avg-degree 9.3 |
| navigation | **visited 133 / 256 nodes** (~48% pruned) |
| correctness | **recall = 5/5** (exact true top-k), `pq_monotone = true` |
| zkVM cycles | total 2,097,152 (2 segments), user 1,504,786 |
| **prove** | **137.3 s** (real STARK, CPU, 32 threads) |
| **verify** | **0.026 s** |
| receipt size | 563,340 bytes; journal 360 bytes |

The committed receipt re-verifies in **31 ms** via `--verify-only` (no re-prove).

The journal publicly commits: `index_digest`, `query_digest`, `n/d/m/k/ef`,
`entry`, `num_visited`, `top_ids`, `top_dists`, `recall`, `pq_monotone`. A valid
receipt therefore attests: *"for the index committed by `index_digest` and the
query committed by `query_digest`, greedy HNSW navigation from `entry` returned
exactly these top-k (id, distance) pairs, which were checked in-circuit to be
valid index members at the recomputed L2 distances and identical to the true
brute-force nearest neighbours."*

The four in-circuit checks (`zkrag/methods/guest/src/main.rs`):

| zkRAG PIOP component | in-circuit check here |
|----------------------|------------------------|
| (1) Priority-Queue Checker | once the beam is full, its worst distance is monotone non-increasing |
| (2) Hybrid Lookup (adjacency) | every expanded edge is a legal neighbour in the committed graph |
| (3) Distance Computation Check | every returned/visited distance is recomputed as integer L2 |
| (4) Membership + top-k | returned ids are valid members; `recall == k` vs brute-force truth |

---

## Layout

```
poc/zkml-faithful-demo/       # as carried in this trimmed repo
├── README.md                 # this file
├── INTEGRATION-SPEC.md       # exact labkit / Makefile / notebook / status wiring for the closeout agent
├── zkllm/                    # EZKL Halo2 attention sub-block
│   ├── requirements.txt
│   └── artefacts/            # committed here: prove.info, tlookup.json
└── zkrag/                    # RISC0 zkVM HNSW search
    └── artefacts/            # committed here: zkrag.corpus.json, zkrag.index.json,
                              #                 zkrag.journal.json, zkrag.proof.info
```

**That tree is what this export ships, not the whole demo.** The sources that
produce those artefacts are **not carried in this trimmed repo**: the top-level
and per-side `scripts/run-all.sh` drivers, `zkllm/`'s `Dockerfile` and
`src/{01_make_attention,02_setup,03_prove,04_verify,tlookup_prototype}.py`, and
zkRAG's Rust crates (`Cargo.toml`, `rust-toolchain.toml`, `Cargo.lock`, `core/`
shared serde types, `methods/guest/` — the in-circuit HNSW search + 4 checks —
and `host/`). Upstream also commits a larger `zkllm/artefacts/`
(`attention.onnx`, `settings.json`, `vk.key`, `proof.json`) and a
`zkrag/artefacts/zkrag.receipt.bin`. The full demo lives upstream at
[`poc/zkml-faithful-demo/`](https://github.com/iiyyll01lin/zkp-final/tree/main/poc/zkml-faithful-demo)
— so **Run it** below describes the upstream tree, not this one.

Committed artefacts are the source of truth (live-or-replay). The 1.1 GB EZKL
`pk.key` and the KZG `*.srs` are **not** committed (regenerable in seconds — see
`.gitignore`); the receipt's `pk` is embedded, so zkRAG needs no extra key.

---

## Run it

```bash
cd poc/zkml-faithful-demo

# Everything (zkLLM in Docker, zkRAG native):
./scripts/run-all.sh all

# Just one side:
./scripts/run-all.sh zkllm        # ~3 min build (first time) + ~25 s pipeline
./scripts/run-all.sh zkrag        # ~2 min build + ~137 s real STARK prove

# Fast replay of committed proofs (no heavy prove):
./scripts/run-all.sh replay
```

zkRAG needs the RISC0 toolchain (`curl -L https://risczero.com/install | bash && rzup install`)
and a Rust toolchain; zkLLM needs Docker (or a venv from `zkllm/requirements.txt`).

---

## Gap to the full papers

**This is the honest boundary. Read this before quoting any number above.**

### zkLLM (vs CCS '24, arXiv:2404.16109)

| Aspect | This demo | Full zkLLM |
|--------|-----------|------------|
| Scope | **one** attention head, `seq=8` | **all** heads/layers of a 7B–13B transformer |
| Softmax | EZKL **generic per-op Halo2 lookups** (exp + recip range checks) | **`tlookup`**: tensorised logUp/cq batching *all* digit lookups into one PIOP |
| exp table | EZKL's internal lookup (per op) | base-b decomposition `exp(x)=∏ₖexp(dₖbᵏ)` → `O(b·log_b M)` table (prototyped here in plaintext only) |
| matmul | dense Halo2 advice columns | **GKR sumcheck** over multilinear extensions |
| Proof system | Halo2 + KZG, monolithic circuit | sumcheck/GKR composed → single KZG/IPA open |
| Weights | seeded (architecture/scaling/softmax identical to MiniLM head-0; trained-HF swap is a one-liner) | the actual committed model weights |

**Net:** we faithfully prove a *real attention sub-block including the softmax
nonlinearity*, the exact part the linear proxy skipped. We do **not** implement
the `tlookup`/`zkAttn` sumcheck machinery (only a plaintext prototype of its
table-compression trick) and do not prove a full multi-layer LLM.

### zkRAG (vs ePrint 2026/709)

| Aspect | This demo | Full zkRAG |
|--------|-----------|------------|
| Proof system | **general zkVM (RISC0 STARK)** re-executing the search | a **custom HNSW PIOP** (the thing zkVM baselines are ~1000× slower than) |
| Scale | 256 × 16-dim, single layer, top-5 | 1M × 128-dim, multi-layer HNSW, top-10 |
| PQ checker | in-circuit beam-worst monotonicity | polynomial heap-invariant `P_pq(x,t)=0` via sumcheck |
| Adjacency | direct in-circuit neighbour check | cq/logUp **hybrid lookup** over the static edge table |
| Membership | in-circuit recall vs brute force | **membership selector vector** + lookup |
| Cost | 137 s prove for 256 vectors | ~50 s prove for **1M** vectors (the PIOP's whole point) |

**Net:** we prove the *same end-to-end relation* zkRAG targets ("retrieved docs
are members of the committed index and are the true top-k, reached by legal HNSW
navigation"), but via the **general-zkVM baseline** at small scale — explicitly
the baseline zkRAG's custom PIOP is designed to beat. We do **not** implement
the polynomial PIOP, the batched sumcheck, or million-scale indices.

---

## References

- zkLLM — CCS '24: <https://arxiv.org/abs/2404.16109> (`reading-notes/zkllm-summary.md`)
- zkRAG — ePrint 2026/709: <https://eprint.iacr.org/2026/709> (`reading-notes/zkrag-summary.md`)
- EZKL: <https://docs.ezkl.xyz/> (`reading-notes/ezkl-overview.md`)
- RISC0 zkVM: <https://dev.risczero.com/>
- Flow diagrams: `diagrams/zkllm-zkattn-flow.md`, `diagrams/zkrag-piop-flow.md`

<!-- demo-G-status: PASS zkllm_logrows=16 zkllm_prove_s=7.85 zkllm_proof_bytes=545679 zkllm_verify="PROOF VERIFIED" zkllm_quant_abs_pct_err=0.037 tlookup_compression=12.5x zkrag_n=256 zkrag_visited=133 zkrag_recall="5/5" zkrag_prove_s=137.3 zkrag_verify_s=0.026 zkrag_receipt_bytes=563340 zkrag_real_stark=true -->
