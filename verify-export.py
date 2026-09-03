#!/usr/bin/env python3
"""Check this export is self-sufficient before you trust it in front of a room.

    python3 verify-export.py

`poc/` in this export is pruned to what the six curated notebooks actually need,
while `labkit.py` retains the broad upstream API and therefore carries constants
that point at artefacts this export does not ship. Asserting *every* labkit
constant would fail by construction. Instead this script asserts the five things
that actually have to hold:

  1. `import labkit` works and resolves REPO_ROOT to *this* directory (the
     poc/ + Makefile marker pair), not to some parent that happens to match.
  2. Every labkit Path constant REACHABLE from the six curated notebooks exists.
     REQUIRED below is that reachable set: the constants the notebooks name
     directly, plus the ones the labkit loaders/plotters they call read.
  3. Every path the notebooks hard-code themselves exists — the paths built in
     notebook code cells rather than via a labkit constant. The old
     all-constants check never covered these, so this is a stronger gate on the
     files that matter, not a weaker one.
  4. The six curated notebooks are present.
  5. Every measurement file the prose cites by name exists. Notebook
     reachability does not cover these: a number can live only in README /
     lab/README / an artefact write-up, and pruning by "what the notebooks
     read" would orphan it while the sentence quoting it stays. EVIDENCE_PATHS
     below is that set.

Constants outside REQUIRED are reported as informational: they are upstream
constants whose artefacts were pruned. Anything in REQUIRED that labkit does not
even define is a hard failure, so a labkit rename cannot silently pass.

Exits non-zero on the first category that fails, listing what is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CURATED = (
    "00_amd_engine_map.ipynb",
    "01_zkml_embedding_ezkl.ipynb",
    "16_verifiable_rag_e2e.ipynb",
    "23_risc0_rocm_stark.ipynb",
    "24_risc0_rocm_bottleneck_lab.ipynb",
    "14_unified_memory_bigmodel.ipynb",
)

#: labkit Path constants reachable from the six curated notebooks, via the
#: transitive closure of the labkit functions those notebooks call.
REQUIRED = (
    # repo anchor + the demo roots the reachable loaders resolve against
    "REPO_ROOT",
    "_AI_DEMO", "_BIGMODEL_DEMO", "_E2E_DEMO", "_FOLD_DEMO",
    "_RISC0_DEMO", "_RISC0_ROCM_PROVER", "_UMA_DEMO", "_ZKML_DEMO",
    # nb00 / nb01 — Path F AI-inference row
    "AI_INFER_CSV", "DETECT_SCRIPT",
    # nb14 — unified-memory flagship
    "BIGMODEL_CSV", "BIGMODEL_JSON", "UMA_BANDWIDTH_CSV",
    # nb16 — verifiable-RAG capstone
    "E2E_ANSWER", "E2E_TIMELINE",
    "ZKRAG_CORPUS", "ZKRAG_JOURNAL", "ZKRAG_PROOF_INFO", "ZKRAG_BN254_PROOF",
    # nb23 — Path I Stage-4 seal
    "RISC0_ROCM_BENCH_CSV", "RISC0_ROCM_PHASE_CSV",
    "RISC0_ROCM_GATE_MD", "RISC0_ROCM_LEDGER_MD", "RISC0_ROCM_CORRECTNESS_MD",
    # nb24 — Groth16 bottleneck forensics (directory-granular evidence bundles)
    "RISC0_ROCM_GROTH16_BENCH_DIR", "RISC0_ROCM_GROTH16_WITNESS_DIR",
    "RISC0_ROCM_GROTH16_COMPARE_DIR", "RISC0_ROCM_EVAL_TUNING_DIR",
)

#: Paths the notebooks build in their own code cells instead of reading a labkit
#: constant. ``(relative path, which notebook needs it)``. nb24 reads named files
#: out of the four directory constants above, so those files are listed too.
NOTEBOOK_PATHS = (
    # nb01 replay branch reads these three directly off repo_path(...)
    ("poc/ezkl-embedding-demo/artefacts/proof.json", "01"),
    ("poc/ezkl-embedding-demo/artefacts/settings.json", "01"),
    ("poc/ezkl-embedding-demo/artefacts/vk.key", "01"),
    # nb14 gated appendix loops over these two CSV names
    ("poc/amd-bigmodel-demo/artefacts/bigmodel-xl.csv", "14"),
    ("poc/amd-bigmodel-demo/artefacts/bigmodel-xl-gemma-bf16.csv", "14"),
    # nb24 reads these by name from the evidence bundles
    ("poc/risc0-rocm-prover/artefacts/groth16-seam-evidence-20260724T060845Z/"
     "benchmark/groth16-benchmark-summary.json", "24"),
    ("poc/risc0-rocm-prover/artefacts/groth16-seam-evidence-20260724T060845Z/"
     "benchmark/groth16-benchmark.csv", "24"),
    ("poc/risc0-rocm-prover/artefacts/groth16-witness-fifo-gfx1151-20260729T050555Z/"
     "summary.json", "24"),
    ("poc/risc0-rocm-prover/artefacts/groth16-witness-fifo-gfx1151-20260729T050555Z/"
     "witness-baseline.csv", "24"),
    ("poc/risc0-rocm-prover/artefacts/groth16-container-compare-gfx1151-20260729T104140Z/"
     "summary.json", "24"),
    ("poc/risc0-rocm-prover/artefacts/groth16-container-compare-gfx1151-20260729T104140Z/"
     "container-compare.csv", "24"),
    ("poc/risc0-rocm-prover/artefacts/groth16-container-compare-gfx1151-20260729T104140Z/"
     "rocm-witness-1/run.log", "24"),
    ("poc/risc0-rocm-prover/artefacts/recursion-evalcheck-launch-tuning-20260724/"
     "block-sweep-summary.csv", "24"),
)

#: Measurement files no notebook reads but the prose cites by name, so the
#: notebook-reachability rule that produced this export does not protect them.
#: Keep this list in step with any number quoted outside a notebook.
#: ``(relative path, the claim it backs)``.
EVIDENCE_PATHS = (
    # Root README honesty layer + poc/zkllm-amd-split-demo/README.md —
    # BN254 G1 MSM crossover (0.654 / 0.814 / 0.845x -> 1.091x at 2^22)
    ("poc/amd-gpu-zk-primitive-demo/artefacts/gpu-bn254.csv", "BN254 G1 MSM crossover"),
    ("poc/amd-gpu-zk-primitive-demo/artefacts/gpu-bn254.log", "BN254 G1 MSM crossover, raw"),
    ("poc/amd-gpu-zk-primitive-demo/artefacts/gpu-bn254.md", "BN254 G1 MSM crossover, write-up"),
    ("poc/amd-gpu-zk-primitive-demo/artefacts/gpu-bn254.solo-rebench-2026-06-18.csv",
     "BN254 G1 MSM crossover, solo re-bench"),
    ("poc/amd-gpu-zk-primitive-demo/artefacts/gpu-bn254.solo-rebench-2026-06-18.log",
     "BN254 G1 MSM crossover, solo re-bench raw"),
    # folding headline 0.994x (OpenCL G1-only paired check-off) and the
    # correction chain 1.34x -> 0.70x -> 0.994x that lab prose walks through
    ("poc/folding-step-demo/artefacts/demo-c-paired-checkoff-opencl-gfx1151.csv",
     "folding 0.994x paired check-off"),
    ("poc/folding-step-demo/artefacts/demo-c-paired-checkoff-opencl-gfx1151.log",
     "folding 0.994x paired check-off, raw"),
    ("poc/folding-step-demo/artefacts/demo-c-paired-checkoff-gfx1151.log",
     "folding 1.048x native-HIP arm, raw"),
    ("poc/folding-step-demo/artefacts/demo-c-gpu.md",
     "folding correction chain write-up + scope notes"),
    # nb24 prose blocks: numbers narrated in markdown cells and in the lab
    # write-ups, read from these bundles by eye rather than by loader.
    # Each bundle's provenance.txt is asserted alongside its data, because the
    # image digest and input hash in it *are* the scope note for those numbers
    # — a figure and the conditions it was measured under travel together.
    ("poc/risc0-rocm-prover/artefacts/groth16-stage-split-gfx1201-20260730T110524Z/"
     "stage-split.txt", "MSM stage-split ceiling"),
    ("poc/risc0-rocm-prover/artefacts/groth16-stage-split-gfx1201-20260730T110524Z/"
     "provenance.txt", "MSM stage-split ceiling, scope note"),
    # the cpu-scaling bundle carries no provenance.txt upstream
    ("poc/risc0-rocm-prover/artefacts/groth16-cpu-scaling-gfx1151-20260731T022020Z/"
     "cpu-scaling.csv", "CPU thread-scaling conversion"),
    ("poc/risc0-rocm-prover/artefacts/groth16-power-gfx1151-20260731T011902Z/tuned/"
     "rocm-tuning.csv", "tuned ROCm 5221.7 ms"),
    ("poc/risc0-rocm-prover/artefacts/groth16-power-gfx1151-20260731T011902Z/tuned/"
     "provenance.txt", "tuned ROCm 5221.7 ms, scope note"),
    ("poc/risc0-rocm-prover/artefacts/groth16-percu-verify-gfx1151-20260730T150717Z/"
     "rocm-tuning.csv", "per-CU work-unit re-check"),
    ("poc/risc0-rocm-prover/artefacts/groth16-percu-verify-gfx1151-20260730T150717Z/"
     "summary.json", "per-CU work-unit re-check, summary"),
    ("poc/risc0-rocm-prover/artefacts/groth16-percu-verify-gfx1151-20260730T150717Z/"
     "summary.md", "per-CU work-unit re-check, write-up"),
    ("poc/risc0-rocm-prover/artefacts/groth16-percu-verify-gfx1151-20260730T150717Z/"
     "provenance.txt", "per-CU work-unit re-check, scope note"),
    ("poc/risc0-rocm-prover/artefacts/groth16-tuning-ab-gfx1151-20260730T052934Z/"
     "tuning-ab.csv", "8-run tuning A/B span"),
    ("poc/risc0-rocm-prover/artefacts/groth16-tuning-ab-gfx1151-20260730T052934Z/"
     "provenance.txt", "8-run tuning A/B span, scope note"),
    # nb24's W0 negative result names its evidence file in prose. Both build
    # logs are asserted, not just the `-O2` one, because the claim *is* the
    # comparison: keeping "O2 ran 106,093 s and was killed" without "O0 takes
    # 245.5 s" leaves a figure with nothing to measure it against.
    ("poc/risc0-rocm-prover/artefacts/groth16-witness-gfx1151-20260727T130143Z/"
     "build-o2.log", "-O2 witness build measured infeasible, killed at 106,093 s"),
    ("poc/risc0-rocm-prover/artefacts/groth16-witness-gfx1151-20260727T130143Z/"
     "build-o0.log", "-O0 witness build 245.5 s, the arm -O2 is compared against"),
    ("poc/risc0-rocm-prover/artefacts/groth16-witness-gfx1151-20260727T130143Z/"
     "provenance.txt", "-O2 vs -O0 build comparison, scope note"),
)

#: The binned end-to-end block and the image-provenance A/B block are asserted
#: per run rather than per file, because each claim rests on the spread across
#: its runs — losing one run silently narrows the span the prose quotes.
EVIDENCE_PATHS += tuple(
    (f"poc/risc0-rocm-prover/artefacts/{bundle}/{rel}", claim)
    for bundle in ("groth16-ab-gfx1201-tuned-binned-20260730T112308Z",
                   "groth16-ab-gfx1201-binned2-20260730T112803Z")
    for rel, claim in [(f"rocm-witness-{i}/run.log", "binned end-to-end run")
                       for i in (1, 2, 3)]
                      + [("provenance.txt", "binned end-to-end run, scope note")]
)

#: The workunits-plus-w8 arm has no summary.{json,md} upstream — the summary
#: step produced no output for it — so only its two files are asserted.
EVIDENCE_PATHS += tuple(
    (f"poc/risc0-rocm-prover/artefacts/"
     f"groth16-image-provenance-gfx1151-20260806T083000Z/{arm}/{name}",
     "ROCm tuning A/B arm")
    for arm in ("default", "workunits-only", "workunits-plus-w8", "window8")
    for name in ("container-compare.csv", "provenance.txt", "summary.json", "summary.md")
    if not (arm == "workunits-plus-w8" and name.startswith("summary."))
)

#: nb24 globs this pattern; assert the glob still matches something.
NOTEBOOK_GLOBS = (
    ("poc/risc0-rocm-prover/artefacts/groth16-seam-evidence-20260724T060845Z/benchmark",
     "bench-gpu-rocm-*.log", "24"),
)

sys.path.insert(0, str(HERE / "lab"))
import labkit as lk  # noqa: E402

failures = []

resolved = Path(lk.REPO_ROOT).resolve()
if resolved != HERE:
    failures.append(f"labkit.REPO_ROOT is {resolved}, expected {HERE}")
print(f"[verify] labkit.REPO_ROOT -> {resolved}")


def walk(name, value, depth=0):
    if depth > 4:
        return
    if isinstance(value, Path):
        yield name, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from walk(f"{name}[{k!r}]", v, depth + 1)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for i, v in enumerate(value):
            yield from walk(f"{name}[{i}]", v, depth + 1)


# --- 2. every constant the six notebooks reach must exist -------------------
undefined = [c for c in REQUIRED if not hasattr(lk, c)]
if undefined:
    failures.append("labkit no longer defines: " + ", ".join(undefined))

checked = 0
missing = []
for attr in REQUIRED:
    if not hasattr(lk, attr):
        continue
    for name, path in walk(attr, getattr(lk, attr)):
        checked += 1
        if not path.exists():
            missing.append(f"{name} -> {path}")
print(f"[verify] required labkit constants: {checked} checked, {len(missing)} missing")
if missing:
    failures.append("missing REQUIRED labkit artefacts:\n  " + "\n  ".join(sorted(missing)))

# --- informational: upstream constants whose artefacts were pruned ----------
pruned = []
for attr, value in vars(lk).items():
    if attr.startswith("__") or attr in REQUIRED:
        continue
    for name, path in walk(attr, value):
        if not path.exists():
            pruned.append(name)
print(f"[verify] upstream constants pointing at pruned artefacts: {len(pruned)} "
      f"(expected — labkit.py retains the broad API; see README.md)")

# --- 3. notebook-hardcoded paths -------------------------------------------
nb_missing = [f"{rel}  (needed by nb{who})"
              for rel, who in NOTEBOOK_PATHS if not (HERE / rel).exists()]
for d, pattern, who in NOTEBOOK_GLOBS:
    if not list((HERE / d).glob(pattern)):
        nb_missing.append(f"{d}/{pattern}  (nb{who} glob matched nothing)")
total_nb = len(NOTEBOOK_PATHS) + len(NOTEBOOK_GLOBS)
print(f"[verify] notebook-hardcoded paths: {total_nb - len(nb_missing)}/{total_nb} present")
if nb_missing:
    failures.append("missing notebook-hardcoded paths:\n  " + "\n  ".join(nb_missing))

# --- 4. the curated notebooks ----------------------------------------------
absent = [nb for nb in CURATED if not (HERE / "lab" / nb).is_file()]
print(f"[verify] curated notebooks: {len(CURATED) - len(absent)}/{len(CURATED)} present")
if absent:
    failures.append("missing curated notebooks: " + ", ".join(absent))

# --- 5. measurement files the prose cites by name ---------------------------
ev_missing = [f"{rel}  ({claim})"
              for rel, claim in EVIDENCE_PATHS if not (HERE / rel).exists()]
print(f"[verify] prose-cited evidence files: "
      f"{len(EVIDENCE_PATHS) - len(ev_missing)}/{len(EVIDENCE_PATHS)} present")
if ev_missing:
    failures.append("missing prose-cited evidence:\n  " + "\n  ".join(ev_missing))

if failures:
    print("\n[verify] FAILED")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("\n[verify] OK — this export is self-sufficient.")
