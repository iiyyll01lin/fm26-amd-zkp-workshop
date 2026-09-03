# Recursion ROCm validation ledger

Status: **release evidence complete; ready for integration** on
`feature/risc0-rocm-recursion`.

The complete seven-stage gate passed on `gfx1151` at commit `57d218d` on
2026-07-23. The first attempt stopped only after its successful release build
because the gate referenced an ignored `dist/` path; the corrected gate uses
tracked `artefacts/dist/` inputs and preserves its receipt and verifier logs.

## Implemented boundary

- `risc0-circuit-recursion-sys` has a `rocm` feature and HIP build path.
- Two HIP unity translation units compile the upstream generated
  `eval_check.cu`, `step_compute_accum.cu`, and `step_verify_accum.cu`
  unchanged.
- `HipCircuitHal` uses the common managed-memory `HipHal` / `HipBuffer`.
- Recursion witness generation remains in the existing CPU
  `RecursionExecutor`.
- `step_compute_accum` and `step_verify_accum` execute on HIP; the WOM
  grand-product prefix scan remains CPU-delegated over managed memory.
- Poseidon2 and SHA-256 recursion select the HIP backend through
  `risc0-zkvm/rocm`. Poseidon254 identity remains CPU in this stage.
- Keccak and Groth16 are unchanged.

## Static checks completed without GPU execution

- [x] Overlay manifests parse and every recursion entry exists.
- [x] Overlay applies byte-for-byte to pinned RISC0 v2.3.2
  (`218e3bc4a8ffcd203a9cd4e46f921bf60aa7e2bd`).
- [x] `cargo metadata --no-deps` accepts the overlaid workspace.
- [x] Feature graph confirms `risc0-zkvm/rocm` enables
  `risc0-circuit-recursion/rocm`.
- [x] `RISC0_SKIP_BUILD_KERNELS=1 cargo check` type-checks the recursion
  library and its ROCm tests.
- [x] `hipcc -fsyntax-only --offload-arch=gfx1151` accepts both recursion HIP
  wrappers.
- [x] `rustfmt --check`, TOML parsing, shell syntax, and `git diff --check`.

Re-run the GPU-free checks after any edit:

```bash
bash poc/risc0-rocm-prover/scripts/check-recursion-port.sh
```

## Deferred correctness gates

Wait until vLLM and every other GPU workload have stopped. The runtime script
has both an explicit acknowledgement and the repository solo guard; overrides
are rejected.

```bash
RISC0_RECURSION_RUNTIME_GATES=I_UNDERSTAND_GPU_MUST_BE_IDLE \
RISC0_HIP_OFFLOAD_ARCH=gfx1151 \
bash poc/risc0-rocm-prover/scripts/run-recursion-gates.sh
```

The script is stop-on-first-failure and covers, in order:

1. recursion `eval_check` HIP == CPU;
2. generated compute/prefix/verify accumulation output buffers HIP == CPU;
3. Poseidon2 recursion proof plus native verification;
4. SHA-256 recursion identity proof plus native verification;
5. existing lift + join + CPU `identity_p254` end-to-end test;
6. release `r0vm --features rocm --receipt-kind succinct`;
7. non-empty succinct receipt, recursion HIP marker, and stock
   `cargo risczero verify`.

## Runtime evidence (2026-07-23)

Canonical run: `artefacts/recursion-runtime-20260723T003616Z/`.

- [x] recursion `eval_check` HIP == CPU;
- [x] compute/prefix/verify accumulation buffers HIP == CPU;
- [x] Poseidon2 and SHA-256 recursion prove + verify;
- [x] lift + join + CPU `identity_p254` end to end;
- [x] `r0vm --receipt-kind succinct` emitted the recursion HIP marker;
- [x] the 223,418-byte succinct receipt passed stock `cargo risczero verify`;
- [x] resolve and union tests passed with recursion HIP markers;
- [x] a `prove`-only CPU control passed with zero recursion HIP markers.

Control logs: `artefacts/recursion-controls-20260723/`. The malformed
`rocm_markers=1\n1` line in that ad-hoc run's provenance is a reporting-only
issue: both individual logs contain one marker and the committed control script
now sums per-file counts numerically.

Collected and retained before integration:

- [x] `rocprofv3` dispatch evidence for `eval_check`, `step_compute_accum`, and
  `step_verify_accum` (`artefacts/recursion-profile-20260723/`);
- [x] receipt/journal equality against a same-input CPU backend
  (`artefacts/recursion-receipt-equality-20260723/`);
- [x] clean-clone overlay replay from the pinned tag
  (`artefacts/recursion-clean-room-20260723/`);
- [x] 3-repetition lift/join/succinct benchmark after all correctness gates
  (`artefacts/recursion-bench-20260723T090710Z/`).

## Release evidence (2026-07-23)

All four deferred gates completed on `gfx1151` and are aggregated by
`scripts/summarize-recursion-evidence.py` into
`artefacts/recursion-release-summary.json` (verdict **PASS**) with an
`evidence.sha256` manifest in each directory.

### Kernel dispatch profile

`rocprofv3` recorded the three recursion kernels
(`artefacts/recursion-profile-20260723/`):

| kernel | dispatch duration |
|---|---:|
| `risc0::circuit::recursion::eval_check` | 31,465,590 ns |
| `risc0::circuit::recursion::step_compute_accum` | 254,117 ns |
| `risc0::circuit::recursion::step_verify_accum` | 134,893 ns |

### Receipt / journal equality

The ROCm and same-input CPU succinct receipts share a byte-identical journal
(`artefacts/recursion-receipt-equality-20260723/summary.json`):

- kind `succinct`, journal 96 bytes;
- journal digest
  `6ffd15b4566971e82332ddfb05c81c7a95f484bd0621bebcc1aaa000769ab50d`;
- both receipts 223,418 bytes and accepted by stock `cargo risczero verify`.

### Clean-room overlay replay

A fresh clone of pinned RISC0 `v2.3.2`
(`218e3bc4a8ffcd203a9cd4e46f921bf60aa7e2bd`) with the 56-file overlay rebuilt
`r0vm` in a private target in 20 m 06 s and reproduced a valid succinct receipt
(`artefacts/recursion-clean-room-20260723/`, `verdict=PASS`).

### Stage benchmark (3 repetitions, solo)

Median wall-clock over three solo repetitions
(`artefacts/recursion-bench-20260723T090710Z/`); ROCm rows carry a recursion
HIP marker and CPU rows carry none:

| stage | ROCm ms | native CPU ms | portable CPU ms | ROCm speedup |
|---|---:|---:|---:|---:|
| lift | 762.8 | 3140.2 | 3959.4 | 4.12x / 5.19x |
| join | 1001.9 | 3350.5 | 4102.0 | 3.34x / 4.09x |
| succinct | 3349.9 | 11542.0 | 13532.4 | 3.45x / 4.04x |

Regenerate the aggregate verdict and manifests after any re-run:

```bash
python3 poc/risc0-rocm-prover/scripts/summarize-recursion-evidence.py \
  poc/risc0-rocm-prover/artefacts
```

## Deferred performance gate

Only after every correctness gate passes, run 3-5 serialized repetitions of
lift, join, and succinct for ROCm, portable CPU, and native CPU. The script
records raw JSON/logs plus median and range CSVs and refuses contended data.

```bash
RISC0_RECURSION_BENCH_GATES=I_UNDERSTAND_GPU_MUST_BE_IDLE \
RECURSION_BENCH_REPS=3 \
RISC0_HIP_OFFLOAD_ARCH=gfx1151 \
bash poc/risc0-rocm-prover/scripts/bench-recursion-stages.sh
```

Do not run or report Groth16, `stark-to-snark`, or on-chain gates from this
branch. Those remain a separate project after recursion integration.
