"""labkit — the shared hybrid toolkit for the AMD Strix Halo ZK lab.

Every notebook in ``lab/`` does ``import labkit as lk`` and builds on the public
API documented below. The design goal is **one notebook, two hosts**: a cell
runs *live* on the AMD Strix Halo (Ryzen AI MAX+ 395 / ROCm) when the hardware
is present, and *replays* the repo's committed CSV/PNG/artefacts on any laptop —
drawing the **same** plot / table / verdict either way.

Three guarantees this module upholds:

* **CWD-independent** — all paths resolve from this file's location, so a
  notebook works regardless of the directory Jupyter was launched from.
* **Never explodes on a laptop** — :func:`detect` and every predicate degrade
  gracefully (``rocm=False``, ``backend="cpu"``) instead of raising when the AMD
  stack is absent.
* **Headless-safe plotting** — plotters pick a non-interactive matplotlib
  backend outside Jupyter, so ``make lab-replay`` (nbconvert, no display) and a
  plain ``python`` smoke test both render and save figures.

Honesty rule (baked into the badge and the plot titles): the iGPU/NPU accelerate
the **AI model**, and the iGPU OpenCL path accelerates SNARK **primitives** +
Groth16 (size-gated), but the **stock** RISC0 ``r0vm`` STARK is CPU-only on AMD.
Frontier (2026-07, scoped): a pinned v2.3.2 research fork runs the rv32im
segment-STARK core + the generated ``eval_check`` on the gfx1151 iGPU — a hybrid
prover (witgen/accum CPU-delegated over unified memory; recursion + Groth16 wrap
still CPU), bit-for-bit == risc0's own ``CpuHal`` and accepted by the stock
``cargo risczero verify`` (independently audited); ~5.46x (flat ~5.3-5.5x; the old
~6.6-6.8x = 5.46x x a 1.25x local-vs-shipped codegen gap) on one ~4-segment
poseidon2 Cartesi-step prove, workload-specific. The stock ``r0vm``
stays CPU-only. See reading-notes/path-i-risc0-rocm-stark.md.

Public API
----------
Paths            : ``REPO_ROOT``, ``repo_path()`` + artefact path constants
Detection        : ``detect()``, ``capability_badge()``
Predicates       : ``has_rocm()``, ``is_strix_halo()``, ``has_ezkl()``,
                   ``has_docker()``, ``has_r0vm()``
Hybrid driver    : ``live_or_replay()``
CSV loaders      : ``load_throughput()``, ``load_gpu_primitive()``,
                   ``load_gpu_groth16()``, ``load_gpu_bn254()``,
                   ``load_gpu_halo2_hotspot()``, ``load_demo_c_gpu()``,
                   ``load_ai_inference()``, ``load_attention_forward()``,
                   ``load_zkllm_scale_sweep()``, ``load_zkllm_msm()``,
                   ``load_gpu_g2_fft()``, ``load_zkrag_msm()``
JSON loaders     : ``load_zkllm_prove_info()``, ``load_zkllm_tlookup()``,
                   ``load_zkrag_journal()``, ``load_zkrag_proof_info()``,
                   ``load_zkrag_piop_info()``, ``load_zkllm_split()``
Plotters/summary : ``plot_thread_scaling()``, ``plot_speedup()``,
                   ``plot_ai_inference()``, ``plot_zkml_faithful_summary()``,
                   ``plot_zkllm_amd_split()``, ``plot_zkllm_scale_sweep()``,
                   ``plot_zkllm_msm()``, ``plot_zkrag_scale()``,
                   ``plot_zkrag_msm_speedup()``, ``summary_from_full_run_info()``
Path I (frontier): RISC0 rv32im segment-STARK on the gfx1151 iGPU (scoped v2.3.2
                   fork, hybrid; nb23) — ``load_risc0_rocm_bench()``,
                   ``load_risc0_rocm_stage4_bench()``, ``load_risc0_rocm_phases()``,
                   ``load_risc0_rocm_correctness()``, ``load_risc0_rocm_gate()``,
                   ``risc0_rocm_amdahl()``, ``risc0_rocm_scope()`` +
                   ``plot_risc0_rocm_gpu_evidence()``, ``plot_risc0_rocm_correctness()``,
                   ``plot_risc0_rocm_bench()``, ``plot_risc0_rocm_phases()``
Telemetry        : ``gpu_telemetry()`` (rocm-smi background sampler),
                   ``load_gpu_telemetry()``, ``plot_gpu_telemetry()`` +
                   ``TELEMETRY_DIR`` / ``TELEMETRY_*`` snapshot path constants
Scorecard        : ``scorecard_table()`` (dynamic, drift-free cross-engine
                   summary), ``plot_scorecard()``
Efficiency/NPU   : ``telemetry_energy()`` + ``plot_energy()`` (GPU-side
                   energy-per-run / perf-per-watt from committed telemetry),
                   ``crossover_table()`` + ``plot_roofline()`` (above-parity vs
                   size-gated-on-OpenCL, ~2²⁰–2²² crossover band:
                   BLS12-381 G1 MSM crosses at 2²⁰, Groth16/retrieval/BN254
                   MSM at 2²²), ``NPU_DISPATCH_JSON`` +
                   ``load_npu_dispatch()`` + ``plot_npu_gemm()`` (measured NPU
                   attention GEMM, peak 1491 GFLOPs — AI model, never the proof)
"""
from __future__ import annotations

import csv
import math
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

__all__ = [
    # paths
    "REPO_ROOT",
    "repo_path",
    "DETECT_SCRIPT",
    "THROUGHPUT_CSV",
    "GPU_PRIMITIVE_CSV",
    "MSM_NTT_BACKEND_CSV",
    "GPU_GROTH16_CSV",
    "GPU_BN254_CSV",
    "GPU_HALO2_HOTSPOT_CSV",
    "HALO2_GPU_PROVE_CSV",
    "HALO2_GPU_PROVE_HIP_CSV",
    "BIGMODEL_CSV",
    "BIGMODEL_JSON",
    "BIGMODEL_XL_JSON",
    "BIGMODEL_XL_GEMMA_JSON",
    "DEMO_C_GPU_CSV",
    "DEMO_C_GPU_HIP_CSV",
    "DEMO_C_PHASE_SPLIT_CSV",
    "DEMO_C_PAIRED_CHECKOFF_CSV",
    "DEMO_C_PAIRED_CHECKOFF_OPENCL_CSV",
    "AI_INFER_CSV",
    "ZKLLM_PROVE_INFO",
    "ZKLLM_SETTINGS",
    "ZKLLM_TLOOKUP",
    "ZKRAG_JOURNAL",
    "ZKRAG_PROOF_INFO",
    "ZKRAG_CORPUS",
    "ZKRAG_SCALE_CSV",
    "ZKRAG_MEM_CSV",
    "ZKRAG_DEAAP_CROSSCHECK",
    "ZKRAG_DEAAP_PROOF_INFO",
    "ZKRAG_MSM_CSV",
    "ZKRAG_PIOP_INFO",
    "ZKRAG_BN254_PROOF",
    "E2E_TIMELINE",
    "E2E_ANSWER",
    "FULL_RUN_INFO",
    "ATTENTION_FORWARD_CSV",
    "ZKLLM_SPLIT_JSON",
    "ZKLLM_SCALE_SWEEP_CSV",
    "ZKLLM_MSM_CSV",
    "ZKRAG_BN254_PROVE_CSV",
    "ZKLLM_MSM_HIP_CSV",
    "GPU_G2_FFT_CSV",
    "HIP_MSM_CSV",
    "HIP_MSM_G2_CSV",
    "HIP_NTT_CSV",
    "HIP_MSM_ROCPRIM_CSV",
    "HIP_MSM_TUNE_CSV",
    "ZEROCOPY_HEADROOM_CSV",
    "NPU_DISPATCH_JSON",
    # Path I (RISC0 rv32im segment-STARK on the gfx1151 iGPU)
    "RISC0_ROCM_BENCH_CSV",
    "RISC0_ROCM_STAGE4_BENCH_CSV",
    "RISC0_ROCM_PHASE_CSV",
    "RISC0_ROCM_GATE_MD",
    "RISC0_ROCM_LEDGER_MD",
    "RISC0_ROCM_PATH_I_MD",
    "TELEMETRY_DIR",
    "TELEMETRY_COLUMNS",
    "TELEMETRY_DEMO_F_EMBED",
    "TELEMETRY_DEMO_E_MSM",
    "TELEMETRY_DEMO_C_FOLD",
    "TELEMETRY_DEMO_G2_MSM",
    # detection
    "detect",
    "capability_badge",
    # predicates
    "has_rocm",
    "is_strix_halo",
    "has_ezkl",
    "has_docker",
    "has_r0vm",
    # hybrid driver
    "live_or_replay",
    # csv loaders
    "load_throughput",
    "load_gpu_primitive",
    "load_gpu_groth16",
    "load_gpu_bn254",
    "load_gpu_halo2_hotspot",
    "load_halo2_gpu_prove",
    "load_demo_c_gpu",
    "demo_c_opencl_g1_paired_speedup",
    "load_ai_inference",
    "path_f_forward_range",
    "load_attention_forward",
    "load_zkllm_scale_sweep",
    "load_zkllm_msm",
    "load_gpu_g2_fft",
    "load_zkrag_msm",
    "load_hip_msm",
    "load_hip_msm_g2",
    "load_hip_ntt",
    "load_hip_msm_rocprim",
    "load_zerocopy_headroom",
    "load_hip_msm_tune",
    "load_bigmodel",
    "load_bigmodel_info",
    # Path I (RISC0 rv32im segment-STARK on the gfx1151 iGPU) loaders + summaries
    "load_risc0_rocm_bench",
    "load_risc0_rocm_stage4_bench",
    "load_risc0_rocm_phases",
    "load_risc0_rocm_correctness",
    "load_risc0_rocm_gate",
    "risc0_rocm_amdahl",
    "risc0_rocm_scope",
    # hardware telemetry
    "gpu_telemetry",
    "load_gpu_telemetry",
    "plot_gpu_telemetry",
    # json loaders
    "load_zkllm_prove_info",
    "load_zkllm_tlookup",
    "load_zkrag_journal",
    "load_zkrag_proof_info",
    "load_zkrag_bn254_onchain",
    "load_zkrag_corpus",
    "load_zkrag_scale",
    "load_zkrag_mem",
    "load_zkrag_deaap_crosscheck",
    "load_zkrag_deaap_proof_info",
    "load_zkrag_piop_info",
    "load_zkllm_split",
    "load_e2e_timeline",
    "load_e2e_answer",
    # plotters / summary
    "plot_thread_scaling",
    "plot_speedup",
    "plot_ai_inference",
    "plot_zkml_faithful_summary",
    "plot_zkllm_amd_split",
    "plot_zkllm_scale_sweep",
    "plot_zkllm_msm",
    "plot_zkrag_scale",
    "plot_zkrag_mem",
    "plot_zkrag_msm_speedup",
    "plot_hip_primitives",
    "plot_bigmodel_memory",
    "plot_e2e_pipeline",
    # Path I (RISC0 rv32im segment-STARK on the gfx1151 iGPU) plotters
    "plot_risc0_rocm_gpu_evidence",
    "plot_risc0_rocm_correctness",
    "plot_risc0_rocm_bench",
    "plot_risc0_rocm_phases",
    "summary_from_full_run_info",
    # cross-engine scorecard
    "scorecard_table",
    "plot_scorecard",
    # efficiency / roofline / NPU GEMM (nb15)
    "telemetry_energy",
    "plot_energy",
    "crossover_table",
    "plot_roofline",
    "load_npu_dispatch",
    "plot_npu_gemm",
]


# ---------------------------------------------------------------------------
# Paths — everything resolves from THIS file, never from the notebook's CWD.
# ---------------------------------------------------------------------------
def _find_repo_root() -> Path:
    """Resolve the repo root from labkit.py's own location.

    ``lab/labkit.py`` lives one level under the repo root, but we still walk up
    looking for the repo markers (``poc/`` + ``Makefile``) so the lab keeps
    working if it is ever vendored a level deeper.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "poc").is_dir() and (candidate / "Makefile").is_file():
            return candidate
    return here.parent  # sensible default: lab/ -> repo root


#: Absolute path to the repository root (resolved from this file).
REPO_ROOT: Path = _find_repo_root()


def repo_path(*parts: str) -> Path:
    """Join ``parts`` onto :data:`REPO_ROOT` and return an absolute ``Path``."""
    return REPO_ROOT.joinpath(*parts)


_RISC0_DEMO = REPO_ROOT / "poc" / "risc0-cartesi-step-demo"
_GPU_DEMO = REPO_ROOT / "poc" / "amd-gpu-zk-primitive-demo"
_EZKL_HALO2_GPU_DEMO = REPO_ROOT / "poc" / "ezkl-halo2-gpu-demo"
_AI_DEMO = REPO_ROOT / "poc" / "amd-ai-inference-demo"
_ZKML_DEMO = REPO_ROOT / "poc" / "zkml-faithful-demo"
_ZKLLM_SPLIT_DEMO = REPO_ROOT / "poc" / "zkllm-amd-split-demo"
_FOLD_DEMO = REPO_ROOT / "poc" / "folding-step-demo"
_BIGMODEL_DEMO = REPO_ROOT / "poc" / "amd-bigmodel-demo"
_E2E_DEMO = REPO_ROOT / "poc" / "verifiable-rag-e2e"
_BRINGUP_DEMO = REPO_ROOT / "poc" / "amd-rocm-bringup"
_UMA_DEMO = REPO_ROOT / "poc" / "amd-unified-memory-demo"
_LIBS_DEMO = REPO_ROOT / "poc" / "amd-rocm-libs-demo"

#: Read-only AMD capability probe (safe to bash; never hard-fails).
DETECT_SCRIPT: Path = _RISC0_DEMO / "scripts" / "amd-accel-detect.sh"
#: CPU STARK throughput sweep (mcycle x RAYON_NUM_THREADS).
THROUGHPUT_CSV: Path = _RISC0_DEMO / "artefacts" / "bench" / "throughput.csv"
#: Path E Tier 1 — MSM/NTT iGPU-OpenCL vs CPU sweep.
GPU_PRIMITIVE_CSV: Path = _GPU_DEMO / "artefacts" / "gpu-primitive.csv"
#: Path E Tier 2 — full Groth16 prove, iGPU-OpenCL vs CPU.
GPU_GROTH16_CSV: Path = _GPU_DEMO / "artefacts" / "gpu-groth16.csv"
#: Path E Phase 3 — BN254 G1 MSM, iGPU-OpenCL vs arkworks CPU.
GPU_BN254_CSV: Path = _GPU_DEMO / "artefacts" / "gpu-bn254.csv"
#: Path E Track 3 — halo2 hotspot (BN254 G1 MSM + Fr NTT), iGPU-OpenCL vs CPU.
GPU_HALO2_HOTSPOT_CSV: Path = _GPU_DEMO / "artefacts" / "halo2-hotspot.csv"
#: W3 Milestone A — FULL halo2 KZG prove w/ iGPU MSM+NTT (vendored fork + seam),
#: bit-for-bit == CPU. OpenCL backend (canonical); the `-hip` sibling is the
#: native-HIP backend. Produced live on the Halo by ``make demo-e-halo2-gpu-prove``.
HALO2_GPU_PROVE_CSV: Path = _EZKL_HALO2_GPU_DEMO / "artefacts" / "halo2-gpu-prove.csv"
#: W3 Milestone A — same, native-HIP backend (``FOLD_GPU_BACKEND=hip``).
HALO2_GPU_PROVE_HIP_CSV: Path = _EZKL_HALO2_GPU_DEMO / "artefacts" / "halo2-gpu-prove-hip.csv"
#: Demo H — unified-memory flagship: >16GB GGUF LLM on the iGPU (llama.cpp/HIP).
BIGMODEL_CSV: Path = _BIGMODEL_DEMO / "artefacts" / "bigmodel.csv"
#: Demo H — flagship run summary (weights/peak-VRAM/throughput + honesty keys).
BIGMODEL_JSON: Path = _BIGMODEL_DEMO / "artefacts" / "bigmodel.json"
#: Demo H XL — Llama-3.3-70B full-offload versus a 32 GB budget on one APU.
BIGMODEL_XL_JSON: Path = _BIGMODEL_DEMO / "artefacts" / "bigmodel-xl.json"
#: Demo H XL — Gemma-3-27B bf16, measured on the 96 GB-carveout Halo B.
BIGMODEL_XL_GEMMA_JSON: Path = _BIGMODEL_DEMO / "artefacts" / "bigmodel-xl-gemma-bf16.json"
#: Track 2 / Demo C — GPU-MSM-driven DeciderEth Groth16 prove, CPU vs iGPU.
DEMO_C_GPU_CSV: Path = REPO_ROOT / "poc" / "folding-step-demo" / "artefacts" / "demo-c-gpu.csv"
#: Track 2 / Demo C — NATIVE-HIP backend DeciderEth prove (FOLD_GPU_BACKEND=hip).
DEMO_C_GPU_HIP_CSV: Path = REPO_ROOT / "poc" / "folding-step-demo" / "artefacts" / "demo-c-gpu-hip.csv"
#: Track 2 / Demo C — where the DeciderEth prove's wall actually goes. Three
#: rows (Nova ``prove_step`` / the offloaded G1 MSMs / everything else) from one
#: clean solo check-off run, so the Amdahl denominator can be read off an
#: artefact instead of asserted. See the sibling ``.md`` for provenance.
DEMO_C_PHASE_SPLIT_CSV: Path = REPO_ROOT / "poc" / "folding-step-demo" / "artefacts" / "demo-c-phase-split-gfx1151.csv"
#: Track 2 / Demo C — the PAIRED native-HIP re-bench: both arms check-off, three
#: interleaved reps each, one session, one binary. Supersedes the ``hip-gpu`` row
#: of :data:`DEMO_C_GPU_HIP_CSV`, whose GPU arm paid a CPU recomputation of every
#: offloaded MSM inside the timed region and divided by a two-month-old baseline.
#: Carries its own untimed check-on control row.
DEMO_C_PAIRED_CHECKOFF_CSV: Path = REPO_ROOT / "poc" / "folding-step-demo" / "artefacts" / "demo-c-paired-checkoff-gfx1151.csv"
#: Track 2 / Demo C — the PAIRED **OpenCL** re-bench: one session, one binary,
#: both arms check-off, arms interleaved, three reps each, median, solo
#: re-verified before every run. Supersedes the ``gpu`` (G1-only) row of
#: :data:`DEMO_C_GPU_CSV`, whose GPU arm was ``n=1``, ``solo=false`` and
#: un-interleaved and was published as a *lower bound*. It does **NOT**
#: re-measure ``gpu-wide``, whose 0.74x keeps those defects uncorrected and
#: stays a floor.
DEMO_C_PAIRED_CHECKOFF_OPENCL_CSV: Path = REPO_ROOT / "poc" / "folding-step-demo" / "artefacts" / "demo-c-paired-checkoff-opencl-gfx1151.csv"
#: Path F — MiniLM all-MiniLM-L6-v2 forward, iGPU(MIGraphX) vs CPU(onnxruntime).
AI_INFER_CSV: Path = _AI_DEMO / "artefacts" / "ai-inference.csv"
#: G4 zkLLM — EZKL attention-block prove metrics (logrows, prove_seconds, proof_bytes).
ZKLLM_PROVE_INFO: Path = _ZKML_DEMO / "zkllm" / "artefacts" / "prove.info"
#: G4 zkLLM — EZKL circuit settings (run_args.logrows, input/param scale).
ZKLLM_SETTINGS: Path = _ZKML_DEMO / "zkllm" / "artefacts" / "settings.json"
#: G4 zkLLM — base-b exp (tlookup) decomposition prototype metrics.
ZKLLM_TLOOKUP: Path = _ZKML_DEMO / "zkllm" / "artefacts" / "tlookup.json"
#: G4 zkRAG — committed receipt journal (top-k ids/dists, recall, digests).
ZKRAG_JOURNAL: Path = _ZKML_DEMO / "zkrag" / "artefacts" / "zkrag.journal.json"
#: G4 zkRAG — prove/verify metrics (cycles, prove_seconds, receipt_bytes, dev_mode).
ZKRAG_PROOF_INFO: Path = _ZKML_DEMO / "zkrag" / "artefacts" / "zkrag.proof.info"
#: G4 zkRAG e2e — embedded corpus + query texts (Path F producer output).
ZKRAG_CORPUS: Path = _ZKML_DEMO / "zkrag" / "artefacts" / "zkrag.corpus.json"
#: G4 zkRAG e2e — Phase 2 unified-memory scale-up sweep (n, prove_seconds, cycles, peak RSS).
ZKRAG_SCALE_CSV: Path = _ZKML_DEMO / "zkrag" / "artefacts" / "zkrag-scale.csv"
#: Group B — segment-size mem-wall sweep (segment_po2, peak RSS crosses 16 GB on real STARK).
ZKRAG_MEM_CSV: Path = _ZKML_DEMO / "zkrag" / "artefacts" / "zkrag-mem.csv"
#: A3 DeAAP capstone — proof <-> qdrant cross-check (real all-mpnet-base-v2, d=768, recall_vs_qdrant).
ZKRAG_DEAAP_CROSSCHECK: Path = (
    _ZKML_DEMO / "zkrag" / "deaap-bridge" / "artefacts" / "zkrag-deaap.crosscheck.json"
)
#: A3 DeAAP capstone — real-STARK prove/verify metrics for the live mpnet bridge (12 segments).
ZKRAG_DEAAP_PROOF_INFO: Path = (
    _ZKML_DEMO / "zkrag" / "deaap-bridge" / "artefacts" / "zkrag-deaap.proof.info"
)
#: Phase 3 — iGPU-vs-CPU Groth16/BLS12-381 zkRAG-retrieval MSM-SNARK sweep
#: (m, gpu/cpu prove seconds, speedup, verified — parity to the STARK journal).
ZKRAG_MSM_CSV: Path = _GPU_DEMO / "artefacts" / "zkrag-msm.csv"
#: Phase 4 — faithful HNSW Priority-Queue-Checker PIOP prototype metrics (JSON).
ZKRAG_PIOP_INFO: Path = _ZKML_DEMO / "zkrag-piop" / "artefacts" / "zkrag-piop.info"
#: STEP 3 — zkRAG retrieval relation re-cast to arkworks BN254 R1CS, proven via the
#: Demo C ark-groth16 seam, with a Groth16 Solidity verifier + anvil replay
#: (VERIFIED ON-CHAIN). proof.json: a/b/c + 13 public inputs, constraints,
#: native_verify, gpu_msm, prove_s.
ZKRAG_BN254_PROOF: Path = _FOLD_DEMO / "artefacts" / "zkrag-bn254" / "proof.json"
#: Capstone — verifiable-RAG e2e per-stage timeline (engine/status/metric/artefact/
#: seconds for all 5 engines + the shared query, host, honesty note).
E2E_TIMELINE: Path = _E2E_DEMO / "artefacts" / "e2e-timeline.json"
#: Capstone — the verifiable-RAG grounded answer (query + retrieved doc texts +
#: the LLM's grounded answer + honesty block).
E2E_ANSWER: Path = _E2E_DEMO / "artefacts" / "e2e-answer.md"
#: Demo B maximal full-run capability/proof summary (key=value lines).
FULL_RUN_INFO: Path = _RISC0_DEMO / "artefacts" / "full-run.info"
#: zkLLM engine-split — attention sub-block forward, iGPU(MIGraphX) vs CPU(onnxruntime).
ATTENTION_FORWARD_CSV: Path = _ZKLLM_SPLIT_DEMO / "artefacts" / "attention-forward.csv"
#: zkLLM engine-split — synthesised 3-engine timeline + honest verdict (JSON).
ZKLLM_SPLIT_JSON: Path = _ZKLLM_SPLIT_DEMO / "artefacts" / "split.json"
#: zkLLM engine-split (A) — EZKL prove scale-up sweep to the 94 GB cap (CSV).
ZKLLM_SCALE_SWEEP_CSV: Path = _ZKLLM_SPLIT_DEMO / "artefacts" / "scale-sweep.csv"
#: zkLLM engine-split (C) — bellperson/opencl Groth16 of the attention matmuls,
#: iGPU vs CPU on BLS12-381 + standalone BN254 MSM at matching m (CSV).
ZKLLM_MSM_CSV: Path = _GPU_DEMO / "artefacts" / "zkllm-msm.csv"
#: Native-HIP end-to-end BN254 Groth16 prove (zkRAG retrieval relation padded to
#: an MSM/NTT-dominated size) — 3-way CPU/OpenCL/HIP, the curve where the iGPU
#: wins the WHOLE prove (1.51x@2^20, holds 1.31x@2^22, 512·CU OOM-safe) (CSV).
ZKRAG_BN254_PROVE_CSV: Path = REPO_ROOT / "poc" / "folding-step-demo" / "artefacts" / "zkrag-bn254-prove.csv"
#: Native-HIP zkLLM attention matmul R1CS re-cast to BN254 ark-groth16 —
#: MEASURED 3-way CPU/OpenCL/HIP end-to-end prove (replaces the projection) (CSV).
ZKLLM_MSM_HIP_CSV: Path = _GPU_DEMO / "artefacts" / "zkllm-msm-hip.csv"
#: Step 4a — BN254 G2 (Fq2) MSM + QAP radix-2 FFT, iGPU-OpenCL vs CPU (CSV). The
#: two Groth16 prover hotspots Demo C leaves on the CPU after the G1-only seam.
GPU_G2_FFT_CSV: Path = _GPU_DEMO / "artefacts" / "g2-fft.csv"

# --- Part 1 native-HIP standalone SNARK primitives (hip/*.hip, hipcc gfx1151) ---
# The ec-gpu OpenCL kernels ported to native HIP: each is bit-for-bit gated vs the
# arkworks reference BEFORE any timing, solo-guarded, widened to 2^22. Produced by
# scripts/run-hip-*.sh (Makefile ``demo-e-hip-*``). HONESTY: these are native-port
# vs generated-OpenCL comparisons on the SAME iGPU, not vs-CPU speed claims — the
# CPU crossover is size-gated (see load_gpu_bn254). The durable claim is
# bit-for-bit GPU == arkworks; the NTT zero-copy Δ is an honest negative at ≤2^20.
#: Native-HIP BN254 G1 *direct* MSM (Pippenger) — min-wall @ tuned 512·CU to 2^22.
#: (Produced live on the Halo by ``make demo-e-hip-msm``; absent until then.)
HIP_MSM_CSV: Path = _GPU_DEMO / "artefacts" / "hip-msm.csv"
#: Native-HIP BN254 G2 (Fq2) MSM — min-wall to 2^22.
HIP_MSM_G2_CSV: Path = _GPU_DEMO / "artefacts" / "hip-msm-g2.csv"
#: Native-HIP BN254 Fr NTT — copy vs APU zero-copy (hipHostMallocMapped) min-wall.
HIP_NTT_CSV: Path = _GPU_DEMO / "artefacts" / "hip-ntt.csv"
#: Native-HIP G1 MSM 3-way: direct kernel vs rocPRIM bucket-reduction vs OpenCL.
HIP_MSM_ROCPRIM_CSV: Path = _GPU_DEMO / "artefacts" / "hip-msm-rocprim.csv"
#: Native-HIP direct MSM work_units (occupancy) sweep [2c] — committed dated solo
#: snapshot (512·CU sweet spot; the undated hip-msm-tune.csv is the re-run output).
HIP_MSM_TUNE_CSV: Path = _GPU_DEMO / "artefacts" / "hip-msm-tune.solo-2026-06-18.csv"
#: APU zero-copy headroom: per-phase wall breakdown of one OpenCL MSM (H2D fraction).
ZEROCOPY_HEADROOM_CSV: Path = _GPU_DEMO / "artefacts" / "zerocopy-headroom.csv"
#: Same-binary backend A/B on gfx1151 (only ``FOLD_GPU_BACKEND`` differs): BN254 G1
#: MSM + Fr NTT, OpenCL vs native HIP. This is the file that keeps "NTT wins" tied to
#: a *curve* rather than to the primitive: the BLS12-381-FFT-vs-blstrs sweep in
#: :data:`GPU_PRIMITIVE_CSV` stays above parity across sizes, but BN254 Fr NTT vs
#: arkworks on the SAME iGPU loses at 2^18 (0.963x, OpenCL).
MSM_NTT_BACKEND_CSV: Path = _GPU_DEMO / "artefacts" / "msm-ntt-backend-gfx1151.csv"

#: Track C — gfx1151 ROCm bring-up runbook probe verdict (JSON: checks + ready).
BRINGUP_REPORT_JSON: Path = _BRINGUP_DEMO / "artefacts" / "bringup-report.json"
#: Track D — APU unified-memory microbench (hipMalloc/hipHostMalloc/hipMallocManaged
#: bandwidth + page-migration, SAXPY GB/s). Columns: alloc_kind, bytes, op, reps,
#: ms_per_iter, gbytes_s, note, device, verify, solo, loadavg.
UMA_BANDWIDTH_CSV: Path = _UMA_DEMO / "artefacts" / "uma-bandwidth.csv"
#: Track A — ROCm library-ecosystem demo (SGEMM rocBLAS/hipBLASLt vs hand-tiled +
#: rocFFT complex sweep). Columns: workload, impl, n, ms_per_iter, gflops, note,
#: device, verify, solo, loadavg.
ROCM_LIBS_CSV: Path = _LIBS_DEMO / "artefacts" / "rocm-libs.csv"

# --- Path I (frontier, scoped) — RISC0 rv32im segment-STARK on the gfx1151 iGPU ---
# The pinned v2.3.2 research fork's HYBRID prover: STARK math + the 26k-LOC generated
# ``eval_check`` on the iGPU (HipHal), witgen + accum CPU-delegated over unified memory.
# Recursion eval/accum and the Groth16 wrap's BN254 MSM/NTT now run on the iGPU too;
# recursion witness/prefix-products and Groth16 witness generation remain CPU. The GPU-produced seal is accepted by the STOCK
# ``cargo risczero verify`` and is bit-for-bit == risc0's own ``CpuHal`` (DualHal 15/15).
# HONESTY: the stock ``r0vm`` stays CPU-only on AMD; this is the scoped fork, one
# workload (poseidon2 ~4-segment Cartesi-step), ~5.46x (flat ~5.3-5.5x; the old
# ~6.6-6.8x = 5.46x x a 1.25x local-vs-shipped codegen gap) — correctness is the hard
# guarantee, speed the scoped secondary. See reading-notes/path-i-risc0-rocm-stark.md.
_RISC0_ROCM_PROVER = REPO_ROOT / "poc" / "risc0-rocm-prover"
#: Path I A3 — the fresh, honest same-code speed headline: iGPU vs the SAME fork code
#: built no-rocm vs the installed rzup r0vm. Columns: ``config, backend, wall_s,
#: receipt_bytes, verify, solo, loadavg``. The 5.46x headline + the 1.25x codegen gap.
RISC0_ROCM_BENCH_CSV: Path = _RISC0_ROCM_PROVER / "artefacts" / "a3-speed.csv"
#: Path I Stage-4 — the point bench of the ~4-segment Cartesi-step composite prove:
#: fork-CPU vs fork-GPU vs installed r0vm. Columns: ``config, backend, threads,
#: workload, wall_s, receipt_bytes, verify, solo, loadavg, notes`` (all seal to the
#: SAME 1112064-byte receipt SIZE + stock-verify; the seal bytes differ per run).
RISC0_ROCM_STAGE4_BENCH_CSV: Path = _RISC0_ROCM_PROVER / "artefacts" / "stage4-bench.csv"
#: Path I A1 — per-phase wall breakdown of the hybrid segment-STARK. Columns:
#: ``phase, engine, cpu_timer_ms, pct_of_prove, hal_op_calls, rocprofv3_kernel_ms,
#: note`` (+ a ``prove_total`` row). Carries the witgen+accum share -> Amdahl ceiling.
RISC0_ROCM_PHASE_CSV: Path = _RISC0_ROCM_PROVER / "artefacts" / "phase-breakdown.csv"
#: Path I Stage-4 gate write-up (markdown): GPU seal + stock-verify + 95%-busy
#: differential (real HipHal path, not a CPU fallback) + DualHal 15/15. Parsed by
#: :func:`load_risc0_rocm_gate` for the seal-panel facts.
RISC0_ROCM_GATE_MD: Path = _RISC0_ROCM_PROVER / "artefacts" / "stage4-gate.md"
#: Path I engine->role->evidence ledger (markdown): per-phase engine map + the
#: correctness/capability evidence table.
RISC0_ROCM_LEDGER_MD: Path = _RISC0_ROCM_PROVER / "artefacts" / "engine-map-ledger.md"
#: Path I reading-note (markdown): the bit-for-bit "ported + proven" kernel table
#: (field/hash/poly/circuit/Merkle GPU==CPU check counts) + the honesty framing.
#: Parsed by :func:`load_risc0_rocm_correctness`.
RISC0_ROCM_PATH_I_MD: Path = REPO_ROOT / "reading-notes" / "path-i-risc0-rocm-stark.md"
#: Lab 24 — full Groth16 receipt benchmark (3 solo, verified reps per mode).
RISC0_ROCM_GROTH16_BENCH_DIR: Path = (
    _RISC0_ROCM_PROVER / "artefacts" /
    "groth16-seam-evidence-20260724T060845Z" / "benchmark"
)
#: Lab 24 — W0 exact Groth16 witness stage timing (production FIFO transport,
#: 3 solo verified reps, whole-pipeline attribution recorded in the same run).
RISC0_ROCM_GROTH16_WITNESS_DIR: Path = (
    _RISC0_ROCM_PROVER / "artefacts" /
    "groth16-witness-fifo-gfx1151-20260729T050555Z"
)
#: Lab 24 — W0 standalone container comparison: ROCm vs CPU gnark on one archived
#: seal, plus the replayed-witness control that measures the witness ceiling.
RISC0_ROCM_GROTH16_COMPARE_DIR: Path = (
    _RISC0_ROCM_PROVER / "artefacts" /
    "groth16-container-compare-gfx1151-20260729T104140Z"
)
#: Lab 24 — recursion eval_check block-size sweep and register-pressure evidence.
RISC0_ROCM_EVAL_TUNING_DIR: Path = (
    _RISC0_ROCM_PROVER / "artefacts" /
    "recursion-evalcheck-launch-tuning-20260724"
)


def _telemetry_csv(demo_dir: Path, name: str) -> Path:
    """Resolve a demo's committed iGPU telemetry snapshot path."""
    return demo_dir / "artefacts" / "telemetry" / f"{name}.telemetry.csv"


#: Path F MiniLM embedding (``make demo-f-embed``) committed iGPU telemetry snapshot.
TELEMETRY_DEMO_F_EMBED: Path = _telemetry_csv(_AI_DEMO, "demo-f-embed")
#: Path E MSM/NTT sweep (``make demo-e-msm``) committed iGPU telemetry snapshot.
TELEMETRY_DEMO_E_MSM: Path = _telemetry_csv(_GPU_DEMO, "demo-e-msm")
#: folding DeciderEth GPU fold (``make demo-c-fold-gpu``) committed iGPU telemetry snapshot.
TELEMETRY_DEMO_C_FOLD: Path = _telemetry_csv(_FOLD_DEMO, "demo-c-fold-gpu")
#: zkRAG MSM-SNARK sweep (``make demo-g2-msm``) committed iGPU telemetry snapshot.
TELEMETRY_DEMO_G2_MSM: Path = _telemetry_csv(_GPU_DEMO, "demo-g2-msm")

#: Committed iGPU telemetry snapshots keyed by demo label. Each lives under that
#: demo's ``artefacts/telemetry/`` directory (there is no single shared dir — the
#: snapshots sit beside the demo that produced them); the nb13 telemetry panel and
#: a glance loop iterate this mapping. Columns: :data:`TELEMETRY_COLUMNS`.
TELEMETRY_DIR = {
    "demo-f-embed": TELEMETRY_DEMO_F_EMBED,
    "demo-e-msm": TELEMETRY_DEMO_E_MSM,
    "demo-c-fold-gpu": TELEMETRY_DEMO_C_FOLD,
    "demo-g2-msm": TELEMETRY_DEMO_G2_MSM,
}

#: Columns every telemetry CSV (a live snapshot or a committed one) carries.
TELEMETRY_COLUMNS: Tuple[str, ...] = (
    "elapsed_s", "power_w", "gpu_use_pct", "vram_used_mb", "vram_total_mb"
)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def _in_ipython() -> bool:
    """True when running inside an IPython/Jupyter kernel (inline backend live)."""
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return False
    return get_ipython() is not None


def _env_flag(name: str) -> bool:
    """True if env var ``name`` is set to a truthy string (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _read_meminfo_gb() -> int:
    """Total RAM in GiB from /proc/meminfo, or 0 if unavailable."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return 0
    match = re.search(r"^MemTotal:\s*(\d+)\s*kB", text, re.MULTILINE)
    return int(match.group(1)) // 1024 // 1024 if match else 0


def _cpu_model_fallback() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        text = ""
    match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    import platform

    return platform.processor() or "unknown CPU"


# ---------------------------------------------------------------------------
# detect() — run + parse amd-accel-detect.sh into a typed, cached dict.
# ---------------------------------------------------------------------------
_DETECT_CACHE: Optional[dict] = None

#: Keys every :func:`detect` result is guaranteed to carry.
_DETECT_KEYS = (
    "cpu_model", "threads", "ram_gb", "kernel",
    "rocm", "gfx", "vulkan", "npu", "backend",
)

# Banner line patterns (the script prints "#  Label : value").
_BANNER_PATTERNS = {
    "cpu_model": r"^#\s*CPU\s*:\s*(.+?)\s*$",
    "threads": r"^#\s*Logical threads:\s*(\d+)",
    "ram_gb": r"^#\s*Unified RAM\s*:\s*(\d+)\s*GB",
    "kernel": r"^#\s*Kernel\s*:\s*(.+?)\s*$",
    "vulkan": r"^#\s*Vulkan\s*:\s*(\w+)",
    "npu": r"^#\s*NPU \(xrt-smi\)\s*:\s*(\w+)",
    "backend": r"^#\s*AI_GPU_BACKEND\s*:\s*(\w+)",
}


def _fallback_detect(banner: str = "", source: str = "fallback") -> dict:
    """Conservative, never-raising capability map for laptops / missing probe."""
    import platform

    info = {
        "cpu_model": _cpu_model_fallback(),
        "threads": os.cpu_count() or 1,
        "ram_gb": _read_meminfo_gb(),
        "kernel": platform.release(),
        "rocm": False,
        "gfx": "unknown",
        "vulkan": False,
        "npu": False,
        "backend": "cpu",
        "source": source,
        "banner": banner,
    }
    return info


def _parse_banner(banner: str) -> dict:
    """Parse the amd-accel-detect.sh ``#`` banner into the typed dict."""
    info = _fallback_detect(banner=banner, source="script")

    for key, pattern in _BANNER_PATTERNS.items():
        match = re.search(pattern, banner, re.MULTILINE)
        if not match:
            continue
        value = match.group(1).strip()
        if key == "threads":
            info[key] = int(value)
        elif key == "ram_gb":
            info[key] = int(value)
        elif key in ("vulkan", "npu"):
            info[key] = value.lower() == "yes"
        else:  # cpu_model, kernel, backend
            info[key] = value

    # ROCm + gfx share a line: "ROCm (rocminfo): yes  (gpu=gfx1151)".
    rocm_match = re.search(
        r"^#\s*ROCm \(rocminfo\):\s*(\w+)\s*\(gpu=([^)]*)\)", banner, re.MULTILINE
    )
    if rocm_match:
        info["rocm"] = rocm_match.group(1).strip().lower() == "yes"
        info["gfx"] = rocm_match.group(2).strip() or "unknown"
    return info


def detect(refresh: bool = False, timeout: float = 120.0) -> dict:
    """Run the read-only AMD probe and return a parsed capability map.

    Bashes :data:`DETECT_SCRIPT` (``amd-accel-detect.sh`` — read-only, never
    hard-fails) and parses its ``#`` banner. The result is **cached**; pass
    ``refresh=True`` to re-probe.

    Always returns a dict with these keys (and never raises on a laptop):

    ===========  =========================================================
    key          meaning
    ===========  =========================================================
    cpu_model    str   — e.g. ``"AMD RYZEN AI MAX+ PRO 395 ..."``
    threads      int   — logical CPUs (``RAYON_NUM_THREADS`` baseline)
    ram_gb       int   — total unified RAM in GiB
    kernel       str   — ``uname -r``
    rocm         bool  — a ``gfxNNNN`` device enumerated via rocminfo
    gfx          str   — gfx target id, e.g. ``"gfx1151"`` or ``"unknown"``
    vulkan       bool  — vulkaninfo reported a device
    npu          bool  — xrt-smi enumerated the XDNA2 NPU
    backend      str   — AI-inference backend: ``rocm`` | ``vulkan`` | ``cpu``
    ===========  =========================================================

    Two extra keys are included for debugging: ``source`` (``"script"`` or
    ``"fallback"``) and ``banner`` (the raw probe output).
    """
    global _DETECT_CACHE
    if _DETECT_CACHE is not None and not refresh:
        return _DETECT_CACHE

    banner = ""
    try:
        proc = subprocess.run(
            ["bash", str(DETECT_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        banner = proc.stdout or ""
        info = _parse_banner(banner) if "capability map" in banner else _fallback_detect(banner)
    except (OSError, subprocess.SubprocessError):
        info = _fallback_detect()

    _DETECT_CACHE = info
    return info


# ---------------------------------------------------------------------------
# capability_badge() — inline HTML in Jupyter, plain text everywhere else.
# ---------------------------------------------------------------------------
def _badge_rows(info: dict) -> Sequence[Tuple[str, str, bool]]:
    """(engine, detail, is_present) rows for the badge."""
    return (
        (
            "Zen5 CPU",
            f"{info['cpu_model']} · {info['threads']} threads · "
            f"{info['ram_gb']} GB unified",
            True,  # the CPU prover path is always available
        ),
        (
            "Radeon iGPU (ROCm)",
            f"gfx={info['gfx']} · backend={info['backend']}"
            if info["rocm"]
            else "not detected (AI-model + SNARK-primitive accel only)",
            info["rocm"],
        ),
        ("Vulkan", "available" if info["vulkan"] else "not detected", info["vulkan"]),
        (
            "XDNA2 NPU",
            "enumerated (research; accelerates AI model, not proofs)"
            if info["npu"]
            else "not detected",
            info["npu"],
        ),
    )


def capability_badge(info: Optional[dict] = None) -> dict:
    """Render an AMD capability badge inline in Jupyter; return the ``info`` map.

    Uses ``IPython.display`` to show an HTML card when running in a notebook, and
    degrades to a plain-text block otherwise (so ``make lab-replay`` / a bare
    ``python`` session still print something useful). Pass an ``info`` dict to
    render a specific map, else :func:`detect` is called.
    """
    if info is None:
        info = detect()

    mode = "LIVE on AMD" if info["rocm"] else "REPLAY-capable (CPU host)"
    rows = _badge_rows(info)

    if _in_ipython():
        try:
            from IPython.display import HTML, display  # type: ignore

            def pill(present: bool) -> str:
                color = "#1a7f37" if present else "#8a8a8a"
                mark = "✓" if present else "—"
                return (
                    f'<span style="background:{color};color:#fff;border-radius:'
                    f'10px;padding:1px 8px;font-size:11px">{mark}</span>'
                )

            body = "".join(
                f'<tr><td style="padding:2px 10px">{pill(present)}</td>'
                f'<td style="padding:2px 10px;font-weight:600">{eng}</td>'
                f'<td style="padding:2px 10px;color:#444">{detail}</td></tr>'
                for eng, detail, present in rows
            )
            html = (
                '<div style="border:1px solid #d0d7de;border-radius:8px;'
                'padding:10px 12px;font-family:system-ui,sans-serif;'
                'max-width:760px">'
                f'<div style="font-weight:700;margin-bottom:6px">'
                f'AMD Strix Halo capability — <span style="color:#0969da">'
                f"{mode}</span></div>"
                f"<table style=\"border-collapse:collapse\">{body}</table>"
                '<div style="margin-top:8px;font-size:11px;color:#57606a">'
                "Honesty rule: iGPU/NPU accelerate the <b>AI model</b>; iGPU "
                "OpenCL accelerates SNARK <b>primitives</b> + Groth16 "
                "(size-gated). The <b>stock</b> RISC0 r0vm <b>STARK is "
                "CPU-only</b> on AMD — frontier (scoped): a pinned v2.3.2 fork "
                "runs the rv32im segment-STARK + eval_check on the iGPU "
                "(hybrid; == CpuHal; stock-verify ✓; path-i)."
                "</div></div>"
            )
            display(HTML(html))
            return info
        except Exception:
            pass  # fall through to text

    lines = [f"AMD Strix Halo capability — {mode}"]
    for eng, detail, present in rows:
        lines.append(f"  [{'x' if present else ' '}] {eng}: {detail}")
    lines.append(
        "  honesty: iGPU/NPU accel the AI model; iGPU OpenCL accel SNARK "
        "primitives+Groth16 (size-gated); stock r0vm STARK is CPU-only on AMD "
        "(frontier scoped: v2.3.2 fork runs rv32im segment-STARK+eval_check on "
        "the iGPU — hybrid; == CpuHal; stock-verify OK; path-i)."
    )
    print("\n".join(lines))
    return info


# ---------------------------------------------------------------------------
# Capability predicates — cheap, cached via detect(), never raise.
# ---------------------------------------------------------------------------
def has_rocm() -> bool:
    """True if rocminfo enumerated a ``gfxNNNN`` device (ROCm usable)."""
    return bool(detect()["rocm"])


def is_strix_halo() -> bool:
    """True if the host looks like a Ryzen AI MAX+ (Strix Halo) / gfx1151 box."""
    info = detect()
    model = info["cpu_model"].lower()
    gfx = info["gfx"].lower()
    return (
        "max+" in model
        or "ryzen ai max" in model
        or "strix halo" in model
        or gfx.startswith("gfx115")
    )


def has_ezkl() -> bool:
    """True if the ``ezkl`` CLI is on PATH (Demo A live prove)."""
    return shutil.which("ezkl") is not None


def has_docker() -> bool:
    """True if the ``docker`` CLI is on PATH (containerised demos)."""
    return shutil.which("docker") is not None


def has_r0vm() -> bool:
    """True if the RISC0 ``r0vm`` prover is on PATH (Demo B live STARK)."""
    return shutil.which("r0vm") is not None


# ---------------------------------------------------------------------------
# live_or_replay() — the hybrid driver every heavy cell routes through.
# ---------------------------------------------------------------------------
Predicate = Callable[[], bool]
_RequiresArg = Union[None, Predicate, Iterable[Predicate]]


def _normalise_requires(requires: _RequiresArg) -> Sequence[Predicate]:
    if requires is None:
        return ()
    if callable(requires):
        return (requires,)
    return tuple(requires)


def live_or_replay(
    live_fn: Callable[[], object],
    replay_fn: Callable[[], object],
    requires: _RequiresArg = None,
    label: str = "",
) -> Tuple[object, str]:
    """Run ``live_fn`` when the host qualifies, else ``replay_fn`` — same output.

    ``requires`` is a predicate or an iterable of predicates (e.g.
    ``[has_r0vm, is_strix_halo]``). When *all* pass, the LIVE path runs;
    otherwise the REPLAY path runs against committed artefacts. A clear
    ``[LIVE]`` / ``[REPLAY]`` banner naming ``label`` and the deciding
    predicates is printed either way.

    Two env overrides take precedence over ``requires`` so the same notebook is
    safe everywhere: ``LAB_FORCE_REPLAY=1`` always replays (used by
    ``make lab-replay`` / CI so heavy live paths never fire even on the real
    Strix Halo), and ``LAB_FORCE_LIVE=1`` always attempts live. As a safety net,
    if the LIVE path raises, the failure is reported and ``replay_fn`` runs
    instead.

    Returns ``(result, mode)`` where ``mode`` is ``"live"`` or ``"replay"``.
    """
    preds = _normalise_requires(requires)
    tag = f" {label}" if label else ""

    if _env_flag("LAB_FORCE_REPLAY"):
        print(f"[REPLAY]{tag}  (LAB_FORCE_REPLAY set; using committed artefacts)")
        return replay_fn(), "replay"

    failed = [p.__name__ for p in preds if not p()]
    force_live = _env_flag("LAB_FORCE_LIVE")

    if force_live or not failed:
        names = ", ".join(p.__name__ for p in preds) or "no preconditions"
        reason = "LAB_FORCE_LIVE set" if force_live and failed else f"satisfied: {names}"
        print(f"[LIVE]{tag}  ({reason})")
        try:
            return live_fn(), "live"
        except Exception as exc:  # noqa: BLE001 — notebooks must keep going
            print(f"[LIVE FAILED]{tag}: {exc!r} -> falling back to REPLAY")
            return replay_fn(), "replay"

    print(f"[REPLAY]{tag}  (missing: {', '.join(failed)}; using committed artefacts)")
    return replay_fn(), "replay"


# ---------------------------------------------------------------------------
# Provenance — name the artefact behind every printed number and every figure.
# ---------------------------------------------------------------------------
# Display-only, and deliberately assembled from what this module already has:
# the ``[LIVE]``/``[REPLAY]`` vocabulary and env precedence of
# :func:`live_or_replay`, the repo-relative rendering of :func:`_rel_to_repo`
# (so a provenance string never carries an absolute path, a host name or an
# account name), and the "name the evidence file" idea already used by
# ``scorecard_table()``'s ``evidence_file`` column and by
# ``path_f_forward_range()``'s ``source`` key. Nothing here changes what a
# loader reads, what it computes, or what it returns.
def provenance_mode() -> str:
    """``"LIVE"`` or ``"REPLAY"`` for the current session — the provenance tag.

    Mirrors the two env overrides :func:`live_or_replay` honours, in the same
    precedence (``LAB_FORCE_REPLAY`` wins over ``LAB_FORCE_LIVE``), so a
    provenance line and a ``live_or_replay()`` banner in the same cell never
    disagree.

    Deliberately **env-only**: it must never call :func:`detect`, because that
    would give the CSV/JSON loaders a subprocess side effect they have never
    had. The default is therefore ``"REPLAY"``, which is also the honest reading
    for a loader — a committed artefact is being replayed off disk no matter
    what hardware happens to be under it.
    """
    if _env_flag("LAB_FORCE_REPLAY"):
        return "REPLAY"
    return "LIVE" if _env_flag("LAB_FORCE_LIVE") else "REPLAY"


def _provenance(path, rows: Optional[int] = None) -> None:
    """Print one ``[MODE] source: <repo-relative path>`` line. Never raises."""
    try:
        extra = f" ({rows} rows)" if rows is not None else ""
        print(f"[{provenance_mode()}] source: {_rel_to_repo(path)}{extra}")
    except Exception:  # noqa: BLE001 — provenance must never break a lab cell
        pass


def _fig_source_note(fig, *paths, y: float = 0.004) -> None:
    """Caption a figure with the repo-relative artefacts behind it. Never raises.

    A PNG that leaves the notebook (dragged into a slide, exported to PDF) loses
    every printed line around it, so the source file has to survive *inside* the
    image. Drawn in the bottom-left margin after ``tight_layout`` so it cannot
    shift any axes, i.e. no datum moves.

    ``y`` is the figure-fraction baseline. Raise it for a plotter that already
    reserves a bottom band via ``tight_layout(rect=...)``, otherwise the
    ``bbox_inches="tight"`` crop stretches down to this caption and leaves a
    stripe of empty canvas between it and the real content.
    """
    try:
        rels = [_rel_to_repo(p) for p in paths if p is not None]
        if not rels:
            return
        fig.text(0.004, y,
                 f"[{provenance_mode()}] source: " + "  \u00b7  ".join(rels),
                 fontsize=6.0, color="#6a737d", ha="left", va="bottom")
    except Exception:  # noqa: BLE001 — a caption must never break a figure
        pass


# ---------------------------------------------------------------------------
# CSV loaders — committed artefacts -> typed pandas DataFrames.
# ---------------------------------------------------------------------------
def _require_pandas():
    try:
        import pandas as pd  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "labkit CSV loaders need pandas — run `make lab-install`"
        ) from exc
    return pd


def _read_csv(path: Path, numeric: Sequence[str]):
    pd = _require_pandas()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"labkit: expected committed CSV at {path}")
    df = pd.read_csv(path)
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    _provenance(path, rows=len(df))
    return df


def load_throughput(path: Optional[Path] = None):
    """Load the CPU STARK throughput sweep as a ``DataFrame``.

    Columns: ``max_mcycle, rayon_threads, wall_seconds, peak_rss_kb,
    proof_bytes, mode, timestamp`` plus a derived ``peak_rss_gb``. Defaults to
    :data:`THROUGHPUT_CSV`.
    """
    df = _read_csv(
        path or THROUGHPUT_CSV,
        numeric=("max_mcycle", "rayon_threads", "wall_seconds",
                 "peak_rss_kb", "proof_bytes"),
    )
    if "peak_rss_kb" in df.columns:
        df["peak_rss_gb"] = df["peak_rss_kb"] / 1024.0 / 1024.0
    return df


def load_gpu_primitive(path: Optional[Path] = None):
    """Load the Path E Tier 1 MSM/NTT sweep as a ``DataFrame``.

    Columns: ``primitive, log_size, size, gpu_ms, cpu_ms, speedup, gpu_device,
    cpu_threads``. Defaults to :data:`GPU_PRIMITIVE_CSV`.
    """
    return _read_csv(
        path or GPU_PRIMITIVE_CSV,
        numeric=("log_size", "size", "gpu_ms", "cpu_ms", "speedup", "cpu_threads"),
    )


def load_gpu_groth16(path: Optional[Path] = None):
    """Load the Path E Tier 2 Groth16 sweep as a ``DataFrame``.

    Columns: ``constraints_pow, constraints, setup_ms, gpu_prove_ms,
    cpu_prove_ms, speedup, verify_ok, gpu_device, cpu_threads``. Defaults to
    :data:`GPU_GROTH16_CSV`.
    """
    return _read_csv(
        path or GPU_GROTH16_CSV,
        numeric=("constraints_pow", "constraints", "setup_ms", "gpu_prove_ms",
                 "cpu_prove_ms", "speedup", "cpu_threads"),
    )


def _require_single_primitive(df, who: str, expected: Optional[str] = None):
    """Raise unless ``df`` carries exactly one ``primitive`` value; return ``df``.

    A guard for figures that put ONE curve on ONE scale.
    :data:`GPU_BN254_CSV` mixes a ``0.654-1.091×`` MSM sweep (``2^16-2^22``) with
    a ``7.367-1469.749×`` NTT microbench (``2^8-2^12``); when those NTT rows
    reached a panel titled "KZG MSM" the y-axis went to ``0-1400`` and the real
    MSM curve became a flat line at zero. Nothing raised and nothing logged — the
    PNG was simply wrong, and raster text inside a PNG is invisible to ``grep`` /
    ``course-drift-check`` / ``pdftotext``, so it shipped. This turns that class
    of mistake into a traceback at the call site instead.

    ``expected`` additionally pins WHICH primitive the caller believes it has, so
    a frame of the wrong single primitive (an NTT-only frame handed to a figure
    titled MSM) is caught too. A frame with no ``primitive`` column is passed
    through untouched — this guard is about mixed scales, not about schema.
    """
    if "primitive" not in getattr(df, "columns", []):
        return df
    seen = sorted({str(p).strip() for p in df["primitive"] if str(p).strip()})
    if len(seen) > 1:
        raise ValueError(
            f"{who}: refusing to plot a MIXED-SCALE frame — 'primitive' holds "
            f"{seen}. Filter to one primitive first (e.g. "
            f"load_gpu_bn254(primitive='msm')): one axis cannot honestly carry "
            f"two scales, and gpu-bn254.csv's ntt rows peak at 1469.749× against "
            f"an msm sweep that never leaves 0.654-1.091×."
        )
    if expected is not None and seen and seen != [str(expected)]:
        raise ValueError(
            f"{who}: expected primitive=={str(expected)!r} rows, got {seen}."
        )
    return df


def load_gpu_bn254(path: Optional[Path] = None, *,
                   primitive: Optional[str] = "msm"):
    """Load the Path E Phase 3 BN254 G1 MSM sweep as a ``DataFrame``.

    Columns (identical to load_gpu_primitive): ``primitive, log_size, size,
    gpu_ms, cpu_ms, speedup, gpu_device, cpu_threads``. Defaults to
    :data:`GPU_BN254_CSV`.

    🔴 **This artefact is MIXED-SCALE, so filtering is this loader's job and not
    the caller's memory.** ``primitive`` selects rows and defaults to ``"msm"``:

    * ``primitive="msm"`` (the default) → the four MSM rows, ``0.654-1.091×``
      over ``2^16-2^22`` (crossover ~2²²). This is the sweep the function is
      named for, and the only one a "BN254 G1 MSM" figure may draw.
    * ``primitive="ntt"`` → the three tiny-size NTT microbench rows at
      ``2^8/2^10/2^12``. Their speedups (``7.367 / 114.698 / 1469.749``) are
      measured against a very slow CPU baseline (``292.208`` ms for a single
      4096-point NTT) and are **three orders of magnitude off the MSM scale**.
    * ``primitive=None`` → every row: an explicit, visible opt-in to the
      mixed-scale frame. Anything that then feeds a single-scale axis must go
      through :func:`_require_single_primitive` first.

    Raises ``ValueError`` if the artefact carries no ``primitive`` column, or if
    the requested ``primitive`` matches no row. An artefact reshaped underneath a
    caller should stop that caller, not silently hand it an empty plot.

    **Why this is a default filter and a raise, not a warning.** This docstring
    used to assert *"Only ``primitive == "msm"`` rows are present"*. That was
    false, and the false assertion is exactly why :func:`plot_zkllm_amd_split`
    plotted the NTT rows into a panel titled "KZG MSM" — the ``1469.749×`` row
    set a ``0-1400`` y-axis and flattened the real ``0.654-1.091×`` curve to an
    invisible line. The previous round downgraded the assertion to a ⚠️ warning,
    which documents the trap without disarming it: a docstring cannot fail a
    build. Forgetting to filter now yields the MSM sweep (correct by default),
    and asking for the mixed frame and then plotting it single-scale raises.
    """
    df = _read_csv(
        path or GPU_BN254_CSV,
        numeric=("log_size", "size", "gpu_ms", "cpu_ms", "speedup", "cpu_threads"),
    )
    if primitive is None:
        return df
    src = path or GPU_BN254_CSV
    if "primitive" not in getattr(df, "columns", []):
        raise ValueError(
            f"{src} has no 'primitive' column, so the primitive="
            f"{primitive!r} filter cannot be applied. This loader will not hand "
            "back an unfiltered mixed-scale frame — pass primitive=None if you "
            "really want every row. See the load_gpu_bn254 docstring."
        )
    want = str(primitive)
    out = df[df["primitive"].astype(str).str.strip() == want]
    if out.empty:
        avail = sorted({str(p).strip() for p in df["primitive"]})
        raise ValueError(
            f"{src} has no primitive=={want!r} rows (available: {avail}). "
            "Pass primitive=None for every row."
        )
    return out.reset_index(drop=True)


def load_gpu_halo2_hotspot(path: Optional[Path] = None):
    """Load the Path E Track 3 halo2-hotspot sweep as a ``DataFrame``.

    Columns (identical to load_gpu_bn254): ``primitive, log_size, size, gpu_ms,
    cpu_ms, speedup, gpu_device, cpu_threads``. Defaults to
    :data:`GPU_HALO2_HOTSPOT_CSV`. ``primitive`` is ``msm`` or ``ntt`` — the two
    halo2/KZG prover hotspots. NTT CPU baseline is a real ark-poly radix-2 FFT.
    """
    return _read_csv(
        path or GPU_HALO2_HOTSPOT_CSV,
        numeric=("log_size", "size", "gpu_ms", "cpu_ms", "speedup", "cpu_threads"),
    )


def load_halo2_gpu_prove(path: Optional[Path] = None, backend: str = "opencl"):
    """Load the W3 Milestone A **full halo2 KZG prove** GPU-vs-CPU sweep.

    Columns: ``stage, k, size, gpu_ms, cpu_ms, speedup, backend, device, verify,
    solo, loadavg``. ``gpu_ms``/``cpu_ms`` are ``na`` on the gate-verdict row(s)
    (bit-for-bit GPU==CPU proof, no timing) and min-wall on the bench rows.
    ``backend`` picks the artefact: ``"opencl"`` -> :data:`HALO2_GPU_PROVE_CSV`
    (canonical), ``"hip"`` -> :data:`HALO2_GPU_PROVE_HIP_CSV` (native-HIP).

    Produced **live on the Halo** by ``make demo-e-halo2-gpu-prove``
    (``scripts/run-halo2-gpu-prove.sh``): the GPU prove is BYTE-IDENTICAL to the
    CPU prove (``verify=OK``), and timing is recorded only in a clean solo window.
    HONESTY: this is portability/enablement at a size-gated ceiling — the
    full-prove wall is ~2x the 32t CPU at 2^16–2^20 (MSM crossover ~2^22, NTT
    parity-to-~2x), not a headline speedup. Callers should handle
    ``FileNotFoundError`` (the file does not exist until the Halo run).
    """
    default = HALO2_GPU_PROVE_HIP_CSV if backend == "hip" else HALO2_GPU_PROVE_CSV
    return _read_csv(
        path or default,
        numeric=("k", "size", "gpu_ms", "cpu_ms", "speedup", "loadavg"),
    )


def load_demo_c_gpu(path: Optional[Path] = None):
    """Load the Demo C DeciderEth CPU-vs-GPU-MSM prove comparison.

    Columns (clean solo re-bench schema, 2026-06-16): ``mode, fold_n, run,
    decider_prove_s, native_verify, onchain_verify, speedup_vs_cpu_median,
    g1_msm_offloaded, gpu_device, solo, loadavg, note``. Rows: three ``cpu`` runs
    + a ``cpu-median`` baseline row + ``gpu`` (G1-only) + ``gpu-wide``
    (G1+G2+FFT). Defaults to :data:`DEMO_C_GPU_CSV`.

    HONESTY: the speedup column here is vs the **clean solo CPU median (66.5 s)**
    of 2026-06-16, and **both of its GPU rows are superseded or provisional**:

    * ``gpu`` (G1-only) published **0.70×**. That row is ``n=1``, ``solo=false``
      and un-interleaved, and was published at the time as a *lower bound*. It
      is **SUPERSEDED** by the paired re-bench in
      :data:`DEMO_C_PAIRED_CHECKOFF_OPENCL_CSV` — CPU 60.847 s vs OpenCL-GPU
      61.206 s = **0.994×, PARITY** (the published figure was pessimistic by
      1.42×). Use :func:`demo_c_opencl_g1_paired_speedup` for the current
      number, not this column. 🔴 The only defensible word is **parity**: the
      0.59% arm gap is smaller than the 0.93% / 1.21% within-arm spreads, so at
      ``n=3`` the arms are indistinguishable — never "the iGPU wins", never
      "0.6% slower". Parity is **not** acceleration.
    * ``gpu-wide`` (G1+G2+FFT) published **0.74×** and was **NOT re-measured**.
      It keeps the same ``n=1`` + contention defects, so it remains a **floor**,
      not an estimate, and the G1 correction must **not** be extrapolated to it.

    The old 1.34×/1.64× headline was contention-inflated (see
    ``docs/INTEGRITY-REPORT.md``). The GPU proof is bit-for-bit equal to the CPU
    proof and verifies natively + on-chain regardless of the speed sign — a
    correctness/plumbing result, independent of any of the above.
    A back-compat ``speedup_vs_cpu`` alias is provided for older consumers.
    """
    df = _read_csv(
        path or DEMO_C_GPU_CSV,
        numeric=("fold_n", "decider_prove_s", "speedup_vs_cpu_median",
                 "speedup_vs_cpu", "g1_msm_offloaded"),
    )
    if "speedup_vs_cpu" not in df.columns and "speedup_vs_cpu_median" in df.columns:
        df["speedup_vs_cpu"] = df["speedup_vs_cpu_median"]
    return df


def demo_c_opencl_g1_paired_speedup(path: Optional[Path] = None) -> Optional[float]:
    """Current OpenCL **G1-only** Demo C speedup, from the paired artefact.

    Computed as ``cpu-median / opencl-gpu-median`` out of
    :data:`DEMO_C_PAIRED_CHECKOFF_OPENCL_CSV` (60.847 / 61.206 = **0.994**)
    rather than read from a note, so it cannot drift away from the artefact.
    Returns ``None`` if the file or either median row is unavailable.

    This **supersedes** the ``gpu`` row of :data:`DEMO_C_GPU_CSV` (0.70×, an
    ``n=1`` contended lower bound). 🔴 The only defensible word for the result
    is **parity**: the 0.59% arm-to-arm gap is smaller than the within-arm
    spreads (0.93% cpu / 1.21% gpu), so at ``n=3`` the two arms are
    statistically indistinguishable. Never render it as an iGPU win and never as
    "0.6% slower"; mean-based is 0.990× and gives the same verdict. **Parity is
    not acceleration.**

    SCOPE: G1-only, OpenCL. It says nothing about ``gpu-wide`` (0.74×, not
    re-measured, a floor) and nothing about the native-HIP arms
    (:data:`DEMO_C_PAIRED_CHECKOFF_CSV`, G1-only 1.048×; ``hip-wide`` 0.77×, not
    re-measured). Four arms, four states — never blended, never averaged.
    """
    try:
        pd = _require_pandas()
        df = pd.read_csv(path or DEMO_C_PAIRED_CHECKOFF_OPENCL_CSV, comment="#")
        med = {str(a): r for a, r in zip(df["arm"], df.to_dict("records"))}
        cpu = float(med["cpu-median"]["decider_prove_s"])
        gpu = float(med["opencl-gpu-median"]["decider_prove_s"])
        return cpu / gpu if cpu > 0 and gpu > 0 else None
    except Exception:  # noqa: BLE001 - callers fall back to an n/a rendering
        return None


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


def path_f_forward_range(path: Optional[Path] = None) -> dict:
    """Derive Path F's fp32 iGPU-vs-CPU FULL-MiniLM forward range from its sweep.

    Computed off :data:`AI_INFER_CSV` instead of written down, so the "the real
    iGPU forward win is the FULL model, not one attention sub-block"
    cross-reference cannot drift away from the artefact the way the retired
    "8-25x" range did. Mirrors ``read_path_f_range`` in
    ``poc/zkllm-amd-split-demo/src/02_synthesize.py``; two gates, both
    load-bearing:

      * only ``solo=true`` rows count — the retired range was measured in a
        window with no solo gate, which inflated the CPU baseline;
      * a cpu/rocm pair is used only when both rows carry the SAME ``loadavg``.
        ai-inference.csv is a splice of two capture windows (fp32 cpu/rocm at
        loadavg 1.59, the fp16/int8/fp8 rows at 2.85), so pairing across it
        would silently mix measurement conditions.

    Returns ``{"range_text", "speedup_min", "speedup_max", "configs", "source",
    "gates"}``. Raises ``FileNotFoundError`` if the sweep is absent and
    ``ValueError`` if it yields no clean-solo cpu/rocm pair — a quoted range
    must come from the artefact or not be quoted at all.
    """
    csv_path = Path(path or AI_INFER_CSV)
    df = _read_csv(csv_path, numeric=("batch", "seq_len", "fwd_ms", "loadavg"))
    solo = df["solo"].astype(str).str.strip().str.lower() == "true"
    backend = df["backend"].astype(str).str.strip()
    ok = df[solo & df["fwd_ms"].notna() & (df["fwd_ms"] > 0)]

    arms = {}
    for name in ("cpu", "rocm"):
        rows = ok[backend.reindex(ok.index) == name]
        arms[name] = {(int(r.batch), int(r.seq_len)): (float(r.fwd_ms), r.loadavg)
                      for r in rows.itertuples()}

    speeds = []
    for key in sorted(set(arms["cpu"]) & set(arms["rocm"])):
        cpu_ms, cpu_load = arms["cpu"][key]
        gpu_ms, gpu_load = arms["rocm"][key]
        if cpu_load != gpu_load:  # spliced capture windows never pair
            continue
        speeds.append(cpu_ms / gpu_ms)

    if not speeds:
        raise ValueError(
            f"labkit: {csv_path} yielded no clean-solo cpu/rocm forward pair — "
            "the Path F full-MiniLM range cannot be derived, so it must not be quoted"
        )

    lo, hi = min(speeds), max(speeds)
    return {
        "range_text": f"{lo:.1f}-{hi:.1f}x",
        "speedup_min": round(lo, 3),
        "speedup_max": round(hi, 3),
        "configs": len(speeds),
        "source": str(csv_path),
        "gates": "solo=true and matching loadavg on both rows of each pair",
    }


def load_attention_forward(path: Optional[Path] = None):
    """Load the zkLLM-split attention-block forward sweep as a ``DataFrame``.

    Stage 1 of the zkLLM engine-split demo: the SAME MiniLM attention head the
    EZKL proof attests, run forward through both engines over a ``batch×seq``
    grid. Columns: ``backend, batch, seq_len, fwd_ms, tokens_per_s, device,
    cpu_threads``. ``backend`` is ``cpu`` (onnxruntime, all Zen 5 threads) or
    ``rocm`` (MIGraphX gfx1151); ``fwd_ms`` is the best-of-N forward latency.
    Defaults to :data:`ATTENTION_FORWARD_CSV`. HONESTY: this accelerates the AI
    MODEL forward, not the proof — and a single sub-block is dispatch-bound, so
    the iGPU win is size-gated (see :func:`load_zkllm_split`).
    """
    return _read_csv(
        path or ATTENTION_FORWARD_CSV,
        numeric=("batch", "seq_len", "fwd_ms", "tokens_per_s", "cpu_threads"),
    )


def load_zkllm_scale_sweep(path: Optional[Path] = None):
    """Load the zkLLM-split (A) EZKL prove scale-up sweep as a ``DataFrame``.

    Track A pushes the proven attention unit up its size curve on the 94 GB box —
    ``head`` (1 head) -> ``mha`` (12-head attention) -> ``layer`` (full encoder
    layer: MHA + LayerNorm + FFN/GELU) -> stacked — running the FULL EZKL Halo2
    flow (CPU-only) at each point. Columns: ``config, seq, d_model, n_heads,
    n_layers, logrows, witness_s, prove_s, proof_bytes, peak_rss_gb, fwd_ms_cpu,
    fwd_ms_igpu, status``. ``status == "ok"`` rows completed; a non-``ok`` status
    (e.g. ``capped``/``prove-timeout``/``setup-failed``) marks where the prover
    hit the cap — those rows carry the calibrated ``logrows`` but blank
    prove/RSS/forward metrics. Defaults to :data:`ZKLLM_SCALE_SWEEP_CSV`. HONESTY:
    EZKL Halo2 proving is CPU-only on AMD; the Strix Halo enabler is 32 threads +
    94 GB system RAM removing a small-host OOM wall as ``logrows`` climbs (the cap on
    this box sits between the 12-head attention at ~82 GB and a full encoder
    layer). ``fwd_ms_*`` is the iGPU(MIGraphX)-vs-CPU(onnxruntime) forward.
    """
    return _read_csv(
        path or ZKLLM_SCALE_SWEEP_CSV,
        numeric=("seq", "d_model", "n_heads", "n_layers", "logrows", "witness_s",
                 "prove_s", "proof_bytes", "peak_rss_gb", "fwd_ms_cpu", "fwd_ms_igpu"),
    )


def load_zkllm_msm(path: Optional[Path] = None):
    """Load the zkLLM-split (C) iGPU-vs-CPU attention-matmul Groth16 sweep.

    Track C leaves EZKL (CUDA/Metal-only on the GPU) for a runnable
    ``bellperson``/``opencl`` Groth16 circuit of the attention **matmuls**
    (``Q,K,V = X·W``; ``S = Q·Kᵀ``; ``Y = A·V`` — softmax EXCLUDED), proved
    end-to-end on **BLS12-381**, iGPU vs CPU (``BELLMAN_NO_GPU``), plus the
    standalone **BN254 G1 MSM** sized to the circuit's ``m`` (EZKL-curve parity).
    Columns: ``config, seq, d_model, d_head, constraints, constraints_pow,
    setup_ms, gpu_prove_ms, cpu_prove_ms, prove_speedup, verify_ok, bn254_msm_m,
    bn254_gpu_msm_ms, bn254_cpu_msm_ms, bn254_msm_speedup, gpu_device,
    cpu_threads``. ``prove_speedup`` and ``bn254_msm_speedup`` are ``cpu/gpu``
    (>1 ⇒ iGPU wins). Defaults to :data:`ZKLLM_MSM_CSV`. HONESTY: size-gated —
    small attention loses on the iGPU (a property of the ec-gpu OpenCL kernel,
    not of MSM and not of the shared LPDDR5X); the end-to-end Groth16 crossover
    appears only at large padded ``m`` (~2²²). The
    BN254 column is the standalone MSM (host-contention-depressed here; the clean
    solo ceiling is Path E's 0.65-1.09×, crossover ~2²² — 2026-06-18 re-bench).
    Full BN254 Groth16 stays the documented
    ``ark-groth16`` injection blocker (see ``docs/zkllm-igpu-proof-scope.md``).
    """
    return _read_csv(
        path or ZKLLM_MSM_CSV,
        numeric=("seq", "d_model", "d_head", "constraints", "constraints_pow",
                 "setup_ms", "gpu_prove_ms", "cpu_prove_ms", "prove_speedup",
                 "bn254_msm_m", "bn254_gpu_msm_ms", "bn254_cpu_msm_ms",
                 "bn254_msm_speedup", "cpu_threads"),
    )


def load_gpu_g2_fft(path: Optional[Path] = None):
    """Load the Step 4a BN254 G2 (Fq2) MSM + QAP radix-2 FFT sweep as a ``DataFrame``.

    These are the two Groth16 prover hotspots Demo C leaves on the CPU after the
    G1-only seam: the **G2 (Fq2)** ``B``-query MSM and the **QAP radix-2 FFT**
    (``ark-poly`` (i)fft + coset variants). Columns are identical to
    :func:`load_gpu_bn254`: ``primitive, log_size, size, gpu_ms, cpu_ms, speedup,
    gpu_device, cpu_threads``. ``primitive`` ∈ {``msm_g2``, ``fft``, ``ifft``,
    ``coset_fft``, ``coset_ifft``}; ``speedup = cpu_ms / gpu_ms`` (>1 ⇒ iGPU wins).
    Every cell is verified bit-for-bit (GPU == arkworks/ark-poly) before timing.
    Defaults to :data:`GPU_G2_FFT_CSV`. HONESTY: at the microbench sizes (≤2^20)
    the G2 (Fq2) MSM (0.73–0.83×) and the coset-FFT variants lose to the 32-thread
    Zen 5; only the forward FFT consistently wins — and the real decider FFT runs
    at 2^24, where the *wider* Demo C fold (``FOLD_GPU_G2=1 FOLD_GPU_FFT=1``)
    published **0.74×** against the clean solo CPU median (66.5 s) — still a
    slowdown. The old 1.34×/1.64× "win" was contention-inflated and is retracted
    (see :func:`load_demo_c_gpu` and ``docs/INTEGRITY-REPORT.md``).

    🔴 **Do not compare that 0.74× against the G1-only arm.** The two are no
    longer measured the same way: G1-only got a paired, interleaved, check-off
    re-bench and now reads **0.994× (parity)**, while the wide arm was **NOT
    re-measured** and keeps its ``n=1`` + contention defects, so its 0.74× is a
    **floor**, not an estimate. The old "wide narrows the gap vs G1-only"
    reading divided two arms taken under different conditions; the honest
    statement is that the wide arm simply has no clean measurement yet, and the
    G1 correction must not be extrapolated to it.
    """
    return _read_csv(
        path or GPU_G2_FFT_CSV,
        numeric=("log_size", "size", "gpu_ms", "cpu_ms", "speedup", "cpu_threads"),
    )


# --- Part 1 native-HIP standalone SNARK primitives -------------------------
def load_hip_msm(path: Optional[Path] = None):
    """Load the native-HIP BN254 **G1 direct MSM** sweep as a ``DataFrame``.

    Columns: ``primitive, log_size, size, min_ms, work_units, device, verify,
    solo, loadavg``. ``min_ms`` is ``na`` on the small verify rows (bit-for-bit
    gate only) and a min-wall timing on the bench rows (tuned 512·CU, to 2^22);
    ``work_units`` records the Pippenger work-unit count that produced each timing.
    Defaults to :data:`HIP_MSM_CSV`. NOTE: this artefact is produced **live on the
    Halo** by ``make demo-e-hip-msm`` (``scripts/run-hip-msm.sh``) in a clean solo
    window — it does not exist until then, so callers should handle
    ``FileNotFoundError`` (the scorecard/plot degrade to the committed
    :func:`load_hip_msm_rocprim` / :func:`load_hip_msm_tune` evidence). HONESTY:
    the vs-OpenCL / vs-CPU framing lives in the tune + rocprim artefacts; this file
    is the standalone tuned-direct timing, bit-for-bit == arkworks.
    """
    return _read_csv(
        path or HIP_MSM_CSV,
        numeric=("log_size", "size", "min_ms", "work_units", "loadavg"),
    )


def load_hip_msm_g2(path: Optional[Path] = None):
    """Load the native-HIP BN254 **G2 (Fq2) MSM** sweep as a ``DataFrame``.

    Columns: ``primitive, log_size, size, min_ms, device, verify, host_xcheck,
    solo, loadavg``. ``min_ms`` = ``na`` on verify rows / min-wall on bench rows
    (to 2^22); ``host_xcheck`` is the CPU-only ported-Fq2/G2 recompute verdict at
    small sizes. Defaults to :data:`HIP_MSM_G2_CSV`. HONESTY: the native-HIP G2
    multiexp is bit-for-bit == arkworks ``cpu_msm_bn254_g2``; the OpenCL G2 path it
    is compared against (``g2-msm.solo-*``) is itself a 0.79–0.96× *slowdown* vs
    the 32-thread CPU — an honest negative, so this is enablement/parity, not a win.
    """
    return _read_csv(
        path or HIP_MSM_G2_CSV,
        numeric=("log_size", "size", "min_ms", "loadavg"),
    )


def load_hip_ntt(path: Optional[Path] = None):
    """Load the native-HIP BN254 **Fr NTT** copy-vs-zero-copy sweep as a ``DataFrame``.

    Columns: ``primitive, log_size, size, copy_ms, zerocopy_ms,
    zerocopy_delta_pct, device, verify, solo, loadavg``. ``copy_ms`` is the
    textbook ``hipMalloc``+``hipMemcpy`` path; ``zerocopy_ms`` uses
    ``hipHostMallocMapped`` so the iGPU reads/writes the shared LPDDR5X in place
    (no ``hipMemcpy``); ``zerocopy_delta_pct = (copy − zerocopy)/copy``. Defaults
    to :data:`HIP_NTT_CSV`. HONESTY: in the committed run the delta is **negative**
    at ≤2^20 (the mapped-buffer path is *slower* — an honest negative; the setup
    isn't amortized at these sizes), even though on this APU there is no PCIe to
    cross. Kernel math is byte-identical to the OpenCL NTT (bit-for-bit == arkworks).
    """
    return _read_csv(
        path or HIP_NTT_CSV,
        numeric=("log_size", "size", "copy_ms", "zerocopy_ms",
                 "zerocopy_delta_pct", "loadavg"),
    )


def load_hip_msm_rocprim(path: Optional[Path] = None):
    """Load the native-HIP G1 MSM **3-way** (direct / rocPRIM / OpenCL) sweep.

    Columns: ``primitive, log_size, size, direct_hip_ms, rocprim_hip_ms,
    opencl_ms, device, verify, solo, loadavg``. All three paths are gated
    bit-for-bit against arkworks before timing. Defaults to
    :data:`HIP_MSM_ROCPRIM_CSV`. HONESTY: the rocPRIM segmented-bucket-reduction
    variant is the *slowest* of the three (~3.2–4.4× slower than the direct
    kernel) — a data-driven honest negative that kept the direct ec-gpu→HIP kernel
    as the reference. ``direct_hip_ms`` here is the 128·CU baseline; the tuned
    512·CU direct timing is :func:`load_hip_msm` / :func:`load_hip_msm_tune`.
    """
    return _read_csv(
        path or HIP_MSM_ROCPRIM_CSV,
        numeric=("log_size", "size", "direct_hip_ms", "rocprim_hip_ms",
                 "opencl_ms", "loadavg"),
    )


def load_zerocopy_headroom(path: Optional[Path] = None):
    """Load the APU zero-copy-headroom per-phase MSM breakdown as a ``DataFrame``.

    A per-phase wall-time breakdown of one OpenCL BN254 G1 MSM (``msm_timed``),
    used to estimate the upper bound APU unified-memory zero-copy could reclaim
    WITHOUT writing HIP. Columns: ``primitive, log_size, size, total_ms, h2d_ms,
    alloc_ms, kernel_ms, d2h_ms, h2d_fraction, input_bytes, gpu_device, solo,
    loadavg``. ``h2d_fraction = h2d_ms / total_ms`` is the actionable headroom —
    the fraction of wall time spent copying inputs H2D that a pointer-passed APU
    buffer would remove. Defaults to :data:`ZEROCOPY_HEADROOM_CSV` (to 2^22).
    HONESTY: on the committed run ``h2d_fraction`` is only ~1.8–2.5% — the MSM wall
    is dominated by the kernel/readback, so zero-copy's *ceiling* on MSM is small.
    """
    return _read_csv(
        path or ZEROCOPY_HEADROOM_CSV,
        numeric=("log_size", "size", "total_ms", "h2d_ms", "alloc_ms",
                 "kernel_ms", "d2h_ms", "h2d_fraction", "input_bytes", "loadavg"),
    )


def load_hip_msm_tune(path: Optional[Path] = None):
    """Load the native-HIP direct-MSM **work_units (2c) tuning** summary as a ``DataFrame``.

    Parses the committed solo tune snapshot (:data:`HIP_MSM_TUNE_CSV`, a
    human-authored multi-section file) into just its two per-size summary tables —
    the ``msm_g1_hip`` and ``msm_g2_hip`` rows — normalised to columns
    ``primitive, log2n, baseline_ms, tuned_ms, tuned_vs_baseline, opencl_ms,
    tuned_vs_opencl`` (both tables share this positional layout: 128·CU baseline,
    512·CU tuned, their ratio, the OpenCL ec-gpu time, and tuned÷OpenCL). This is
    the source of the scorecard's "~2.0–2.2× vs OpenCL after 512·CU" figure.
    HONESTY: ``tuned_vs_opencl`` is native-HIP vs generated-OpenCL on the SAME
    iGPU (not vs CPU); 512·CU is a mid-size (≤2^22) micro-bench win, and the Rust
    ``Bn254Gpu`` backend keeps the memory-safe 128·CU default at 2^24.
    """
    pd = _require_pandas()
    path = Path(path or HIP_MSM_TUNE_CSV)
    if not path.is_file():
        raise FileNotFoundError(f"labkit: expected committed CSV at {path}")
    cols = ("primitive", "log2n", "baseline_ms", "tuned_ms",
            "tuned_vs_baseline", "opencl_ms", "tuned_vs_opencl")
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not (s.startswith("msm_g1_hip,") or s.startswith("msm_g2_hip,")):
            continue
        parts = s.split(",")
        if len(parts) != 7:
            continue  # skip the sweep/other rows; only the 7-col summary tables
        try:
            int(parts[1])  # log2n must parse (guards against header-ish lines)
        except ValueError:
            continue
        rows.append(dict(zip(cols, parts)))
    df = pd.DataFrame(rows, columns=list(cols))
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_zkrag_msm(path: Optional[Path] = None):
    """Load the Phase 3 zkRAG-retrieval iGPU-vs-CPU Groth16 sweep as a ``DataFrame``.

    Phase 3 re-casts the four zkRAG retrieval checks (squared-L2 with bit-range,
    PQ-monotone ``<`` chain, adjacency lookup, membership + brute-force top-k) as
    a runnable ``bellperson``/``opencl`` Groth16 circuit on **BLS12-381** with an
    algebraic (Horner/MSM) commitment, proved over the SAME committed
    ``zkrag.index.json`` instance the RISC0 STARK attests — a relation-parity gate
    asserts the circuit's ``top_ids``/``recall`` equal the STARK journal before any
    timing. The prove is then run iGPU (OpenCL) vs CPU (``BELLMAN_NO_GPU``) at
    increasing padded ``m`` (2^16 → 2^22). Columns: ``m, gpu_prove_s, cpu_prove_s,
    speedup, verified``; ``speedup = cpu/gpu`` (>1 ⇒ iGPU wins). Defaults to
    :data:`ZKRAG_MSM_CSV`. HONESTY: this is the MSM-SNARK re-cast where the iGPU
    genuinely touches the PROOF (size-gated — parity ~1.0× only at the largest
    ~2^22 instance — a property of the ec-gpu OpenCL kernel, not of MSM and not
    of the shared LPDDR5X); the RISC0 zkRAG STARK itself stays CPU-only on AMD.
    """
    return _read_csv(
        path or ZKRAG_MSM_CSV,
        numeric=("m", "gpu_prove_s", "cpu_prove_s", "speedup"),
    )


def load_bigmodel(path: Optional[Path] = None):
    """Load the Demo H unified-memory flagship sweep as a ``DataFrame``.

    Demo H runs a real >16GB GGUF instruct LLM (the RAG **generator**) on the
    Radeon 8060S iGPU (``gfx1151``) via llama.cpp built with HIP. Columns:
    ``condition`` (``full_igpu`` | ``cap_16gb`` | ``cpu``), ``ngl`` (offloaded
    layers), ``weights_gb`` (model file size), ``peak_vram_gb`` / ``peak_gtt_gb``
    / ``peak_gpu_gb`` (peak ``rocm-smi`` used VRAM carveout / GTT / their sum,
    sampled during the run), ``prefill_tps``, ``gen_tps``. On this APU
    llama.cpp/HIP maps the weights primarily into the GTT region of the unified
    LPDDR5X, so the honest GPU-resident footprint is ``peak_gpu_gb`` (VRAM+GTT).
    ``peak_gpu_gb`` may exceed a device pool; that marks where spilling begins,
    not where a discrete card becomes unable to run. Defaults to
    :data:`BIGMODEL_CSV`. HONESTY: this accelerates the AI MODEL (the generator),
    NOT the proof — the RISC0 STARK stays CPU-only; the iGPU VRAM is a 32GB
    carveout of the 94GB pool; the ``cap_16gb`` contrast is generous to a
    discrete card (the CPU spill here is the same LPDDR5X, not a PCIe spill).
    """
    return _read_csv(
        path or BIGMODEL_CSV,
        numeric=("ngl", "weights_gb", "peak_vram_gb", "peak_gtt_gb",
                 "peak_gpu_gb", "prefill_tps", "gen_tps"),
    )


def load_bigmodel_info(path: Optional[Path] = None) -> dict:
    """Demo H run summary: keys model, weights_gb, carveout_gb, unified_pool_gb,
    discrete_budget_gb, peak_vram_gb_full_igpu, exceeds_16gb, conditions, host,
    honesty, captured. Defaults to :data:`BIGMODEL_JSON`."""
    return _read_json(path or BIGMODEL_JSON)


# ---------------------------------------------------------------------------
# JSON loaders — committed Demo G (G4 zkLLM / zkRAG) artefacts (no pandas).
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    """Read a committed JSON artefact into a dict; raise if it is missing."""
    import json as _json

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"labkit: expected committed JSON at {path}")
    _provenance(path)
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


def load_zkrag_bn254_onchain(path: Optional[Path] = None) -> dict:
    """STEP 3 zkRAG BN254 on-chain proof artefact (VERIFIED ON-CHAIN).

    The zkRAG HNSW top-k / membership retrieval relation re-cast from the
    bellperson/BLS12-381 bench to an arkworks ``ConstraintSynthesizer<ark_bn254::Fr>``
    and proven through the Demo C vendored+patched ``ark-groth16`` (BN254) prover,
    then verified natively AND on-chain (anvil) via a snarkJS-style Groth16 Solidity
    verifier. Keys: ``a``/``b``/``c`` (Groth16 proof points as decimal strings),
    ``input`` (13 public inputs = top_ids ‖ top_dists ‖ recall ‖ g ‖ C),
    ``constraints`` (~10.8k), ``native_verify`` (bool), ``gpu_msm`` (bool — whether
    the iGPU MSM seam was used; a CPU proof verifies on-chain identically),
    ``prove_s``. Defaults to :data:`ZKRAG_BN254_PROOF`. HONESTY: the deliverable is
    the BN254 **on-chain verification capability** (top-k parity with the BLS12-381
    bench: ``top_ids=[15,14,20,7,28]``, recall 5/5), NOT a speed win — per the
    Demo C paired re-bench an iGPU offload at this 2^24 size reaches only
    **parity** (OpenCL G1-only 0.994×), so this STEP 3 result is a
    correctness/plumbing achievement. **Parity is not acceleration**: it removes
    the measured cost, it does not add a speed claim.
    """
    return _read_json(path or ZKRAG_BN254_PROOF)


def load_e2e_timeline(path: Optional[Path] = None) -> dict:
    """Capstone — the verifiable-RAG end-to-end per-stage timeline (committed JSON).

    The ``poc/verifiable-rag-e2e`` orchestrator threads ONE query through all five
    Strix Halo engines and records what each did. Top-level keys: ``title``,
    ``query`` (the shared question), ``stages`` (list), ``honesty``, ``host``,
    ``generated``. Each stage dict has ``stage`` (1-5), ``name``, ``engine``,
    ``status`` (``live`` ran here | ``replay`` read from the home demo's committed
    artefact), ``metric`` (the key number), ``artefact`` (repo-relative path),
    ``seconds`` (wall time of this stage in the orchestrator), and ``note`` (the
    honesty nuance). HONESTY: the iGPU does stages 1 & 5 (embed + generate — the AI
    model); the CPU does stage 3 (the RISC0 STARK); stage 4 verifies on-chain. The
    proof is CPU-only — the iGPU never proves. Defaults to :data:`E2E_TIMELINE`.
    """
    return _read_json(path or E2E_TIMELINE)


def load_e2e_answer(path: Optional[Path] = None) -> str:
    """Capstone — the verifiable-RAG grounded answer markdown (committed text).

    Returns the raw markdown of :data:`E2E_ANSWER`: the shared query, the
    proven-honest top-5 retrieved document texts, the LLM's grounded answer, and
    the per-stage honesty block. Render it in a notebook with
    ``IPython.display.Markdown``. HONESTY: the retrieval is proven correct (STARK
    + on-chain) and the answer is grounded in those proven docs — but the LLM
    output itself is NOT proven. Defaults to :data:`E2E_ANSWER`.
    """
    path = Path(path or E2E_ANSWER)
    if not path.is_file():
        raise FileNotFoundError(f"labkit: expected committed answer at {path}")
    _provenance(path)
    return path.read_text()


def load_zkrag_corpus(path: Optional[Path] = None) -> dict:
    """G4 zkRAG e2e corpus (Path F producer output): the embedded texts.

    Keys: ``docs`` (list[str], one per index id), ``query`` (str),
    ``embed_backend`` (``rocm`` | ``cpu`` | ``pseudo``), ``embed_fallback``
    (bool — True iff a deterministic pseudo-embedding was used because no
    onnxruntime/MIGraphX was available), ``embed_doc_ms``, ``embed_query_ms``,
    ``model``, ``n``, ``d``, ``val_range``. Lets a notebook print the retrieved
    document *text* for the journal's ``top_ids``. Defaults to :data:`ZKRAG_CORPUS`.
    """
    return _read_json(path or ZKRAG_CORPUS)


def load_zkrag_scale(path: Optional[Path] = None):
    """G4 zkRAG e2e Phase 2 unified-memory scale-up sweep as a ``DataFrame``.

    Columns: ``n, d, prove_seconds, total_cycles, peak_rss_mb, dev_mode`` plus a
    derived ``peak_rss_gb``. ``dev_mode == False`` rows are a **real STARK**
    prove (real ``prove_seconds`` + prover peak RSS); ``dev_mode == True`` rows
    are ``RISC0_DEV_MODE`` executor runs (real cycle counts at larger ``n``,
    executor-only RSS, no cryptographic seal — so the sweep finishes in
    minutes). Defaults to :data:`ZKRAG_SCALE_CSV`.
    """
    df = _read_csv(
        path or ZKRAG_SCALE_CSV,
        numeric=("n", "d", "prove_seconds", "total_cycles", "peak_rss_mb"),
    )
    if "dev_mode" in df.columns:
        df["dev_mode"] = (
            df["dev_mode"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
        )
    if "peak_rss_mb" in df.columns:
        df["peak_rss_gb"] = df["peak_rss_mb"] / 1024.0
    return df


def load_zkrag_mem(path: Optional[Path] = None):
    """Group B — segment-size mem-wall sweep as a ``DataFrame``.

    Columns: ``segment_po2, n, d, prove_seconds, total_cycles, peak_rss_mb,
    dev_mode`` plus a derived ``peak_rss_gb``. Each row is a **real STARK** prove
    (``dev_mode == False``) at a fixed ``n``/``d`` but an increasing r0vm segment
    size (``ZKRAG_SEGMENT_PO2``). r0vm proves fixed-size segments, so prover peak
    RSS is bounded by the SEGMENT size, not ``n`` — raising ``segment_po2`` is the
    genuine lever that pushes CPU-prover peak RSS past a **15 GB laptop's system
    RAM**. This is host RSS, not GPU VRAM: the iGPU embeds the query, while this
    STARK prove stays CPU-only. Defaults to :data:`ZKRAG_MEM_CSV`.
    """
    df = _read_csv(
        path or ZKRAG_MEM_CSV,
        numeric=("segment_po2", "n", "d", "prove_seconds", "total_cycles", "peak_rss_mb"),
    )
    if "dev_mode" in df.columns:
        df["dev_mode"] = (
            df["dev_mode"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
        )
    if "peak_rss_mb" in df.columns:
        df["peak_rss_gb"] = df["peak_rss_mb"] / 1024.0
    return df


def load_zkrag_deaap_crosscheck(path: Optional[Path] = None) -> dict:
    """A3 — DeAAP/mpnet capstone proof <-> qdrant cross-check (committed JSON).

    The strongest semantic capstone: the RISC0 STARK proves the HNSW top-k over
    REAL ``sentence-transformers/all-mpnet-base-v2`` 768-d embeddings pulled from
    a live DeAAP qdrant index, then the proven top-k chunk_ids are cross-checked
    against qdrant's own returned ids. Useful keys: ``model``, ``n``, ``d`` (768),
    ``k``, ``ef``, ``m``, ``metric``, ``query_text``, ``recall_vs_qdrant`` (5),
    ``k_total`` (5), ``exact_match`` (True), ``in_circuit_recall_vs_bruteforce_cosine``,
    ``pq_monotone``, ``zkrag_topk_chunk_ids`` / ``deaap_topk_chunk_ids``,
    ``zkrag_top_cosine`` / ``deaap_scores``, ``index_digest_hex`` /
    ``query_digest_hex``. Defaults to :data:`ZKRAG_DEAAP_CROSSCHECK`.
    """
    return _read_json(path or ZKRAG_DEAAP_CROSSCHECK)


def load_zkrag_deaap_proof_info(path: Optional[Path] = None) -> dict:
    """A3 — DeAAP/mpnet capstone real-STARK prove/verify metrics (committed JSON).

    Keys: ``image_id_hex``, ``dev_mode`` (False — a real seal), ``total_cycles``,
    ``user_cycles``, ``segments`` (12 — the 768-d witness spans 12 r0vm segments),
    ``prove_seconds`` (~602s real STARK), ``verify_seconds``, ``receipt_bytes``
    (~3.36 MB), ``journal_bytes``. Defaults to :data:`ZKRAG_DEAAP_PROOF_INFO`.
    """
    return _read_json(path or ZKRAG_DEAAP_PROOF_INFO)


def load_zkrag_piop_info(path: Optional[Path] = None) -> dict:
    """Phase 4 — faithful HNSW Priority-Queue-Checker PIOP prototype metrics (JSON).

    The first real component of the faithful-HNSW custom PIOP: the heap invariant
    (beam-worst distance trace monotone non-increasing) encoded as a polynomial
    zero-check and proved by a hand-rolled multilinear **sumcheck** over **BN254
    Fr** with a multilinear **KZG** (PST13) opening of the committed trace. Useful
    keys: ``component``, ``paper``, ``relation``, ``field``, ``commitment``,
    ``argument``, ``trace_T`` (2048), ``hypercube_mu``/``hypercube_N``,
    ``constraints`` (69,632), ``srs_setup_seconds``, ``prove_seconds`` (~0.32s),
    ``verify_seconds`` (~0.10s), ``proof_bytes`` (~28.9 KB), ``honest_verified``
    (True), ``tampered_rejected`` (True), ``open_scope``. Defaults to
    :data:`ZKRAG_PIOP_INFO`. HONESTY: a prototype on a toy deterministic SRS that
    proves ONE of the four PIOP components; the remaining three (membership
    selector, cq/logUp lookup, distance check) are designed, not yet built.
    """
    return _read_json(path or ZKRAG_PIOP_INFO)


def load_zkllm_split(path: Optional[Path] = None) -> dict:
    """zkLLM engine-split synthesis: the 3-engine timeline + honest verdict.

    Flat contract keys (read directly): ``forward_ms_igpu``, ``forward_ms_cpu``,
    ``forward_speedup``, ``prove_seconds``, ``verify_seconds``, ``verify_status``,
    ``proof_bytes``, ``msm_speedup_min``, ``msm_speedup_max``, ``msm_blocker``,
    plus the flattened (B) parity keys ``parity_igpu_vs_circuit_max_abs``,
    ``parity_cpu_vs_circuit_max_abs``, ``parity_within_tolerance``. Structured
    detail: ``stage1_forward`` (incl. ``grid`` + ``best_igpu_speedup``),
    ``stage2_prove``, ``stage3_msm_frontier``, ``parity`` (the full (B) block —
    ``errors`` per pair, ``tolerance_abs``, ``output_scale``, ``statement``),
    ``timeline``, ``verdict`` (per engine), ``honesty``. Defaults to
    :data:`ZKLLM_SPLIT_JSON`.
    """
    return _read_json(path or ZKLLM_SPLIT_JSON)


def load_bringup_report(path: Optional[Path] = None) -> dict:
    """Track C — gfx1151 ROCm bring-up probe verdict (JSON).

    Keys: ``schema, generated_utc, host, kernel, gfx, rocm`` (bool), ``hipcc``,
    ``unified_ram_gb``, ``ttm_pages_limit``, ``ttm_pages_limit_gb``,
    ``amdgpu_gttsize_mib``, ``needs_hsa_override`` (bool), ``ready`` (bool),
    ``n_fail``, ``n_warn``, ``checks`` (list of ``{name,status,detail,hint}``)
    and ``honesty``. Produced read-only by
    ``poc/amd-rocm-bringup/scripts/diagnose.sh``. Defaults to
    :data:`BRINGUP_REPORT_JSON`.
    """
    return _read_json(path or BRINGUP_REPORT_JSON)


def bringup_checks(path: Optional[Path] = None):
    """Track C — the bring-up probe checks as a tidy ``DataFrame``.

    Columns: ``name, status, detail, hint`` (one row per probe check). Handy for
    rendering the runbook ladder in ``lab/20``. Defaults to the committed
    :data:`BRINGUP_REPORT_JSON`.
    """
    pd = _require_pandas()
    rep = load_bringup_report(path)
    rows = rep.get("checks", []) if isinstance(rep, dict) else []
    df = pd.DataFrame(rows, columns=["name", "status", "detail", "hint"])
    return df


def load_uma_bandwidth(path: Optional[Path] = None):
    """Track D — APU unified-memory microbench as a ``DataFrame``.

    Columns: ``alloc_kind, bytes, op, reps, ms_per_iter, gbytes_s, note, device,
    verify, solo, loadavg`` plus a derived ``mib`` (buffer MiB). Each row is the
    measured SAXPY (memory-bound) bandwidth for a HIP allocation strategy:
    ``hipMalloc`` (explicit H2D/D2H staging copy), ``hipHostMalloc`` (zero-copy,
    no staging), ``hipMallocManaged`` (page-migrated cold vs warm). Defaults to
    :data:`UMA_BANDWIDTH_CSV`.
    """
    df = _read_csv(
        path or UMA_BANDWIDTH_CSV,
        numeric=("bytes", "reps", "ms_per_iter", "gbytes_s", "loadavg"),
    )
    if "bytes" in df.columns:
        df["mib"] = df["bytes"] / 1024.0 / 1024.0
    return df


def load_rocm_libs(path: Optional[Path] = None):
    """Track A — ROCm library-ecosystem demo as a ``DataFrame``.

    Columns: ``workload, impl, n, ms_per_iter, gflops, note, device, verify,
    solo, loadavg``. ``workload == "sgemm"`` rows compare ``rocblas`` /
    ``hipblaslt`` / ``hand_tiled``; ``workload == "fft_complex"`` rows are the
    rocFFT complex-C2C sweep (the WRONG tool for a ZK finite-field NTT). Defaults
    to :data:`ROCM_LIBS_CSV`.
    """
    return _read_csv(
        path or ROCM_LIBS_CSV,
        numeric=("n", "ms_per_iter", "gflops", "loadavg"),
    )


# ---------------------------------------------------------------------------
# Path I (frontier, scoped) — RISC0 rv32im segment-STARK on the gfx1151 iGPU.
# Loaders for the committed artefacts under poc/risc0-rocm-prover/artefacts/ (+ the
# path-i reading-note). CSV numbers are read straight from the artefacts; the two
# markdown parsers degrade to []/{} rather than raise, so every loader is laptop-safe
# (they read committed files — no ROCm / no hardware needed to replay).
# ---------------------------------------------------------------------------
def _read_csv_commented(path: Path, numeric: Sequence[str]):
    """Like :func:`_read_csv` but skips ``#`` provenance lines.

    The phase-breakdown / stage4-bench CSVs wrap their data in a header + footer of
    ``#`` comment lines (workload, method, cross-checks); ``comment="#"`` drops them so
    the first real row is the CSV header.
    """
    pd = _require_pandas()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"labkit: expected committed CSV at {path}")
    df = pd.read_csv(path, comment="#")
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    _provenance(path, rows=len(df))
    return df


def load_risc0_rocm_bench(path: Optional[Path] = None):
    """Path I A3 — the honest same-code speed headline as a ``DataFrame``.

    Columns: ``config, backend, wall_s, receipt_bytes, verify, solo, loadavg`` plus a
    derived ``speedup_vs_fork_cpu`` (fork-CPU wall / this wall). The ``fork-gpu`` row
    is the **5.46x** headline (iGPU vs the SAME fork code built no-rocm — apples to
    apples); the ``installed`` rzup ``r0vm`` row exposes the **1.25x** local-vs-shipped
    codegen gap that inflated the old ~6.6-6.8x. Every row seals to a receipt of the same
    1112064-byte **size** and stock-verifies (the seal bytes themselves differ per run).
    Defaults to :data:`RISC0_ROCM_BENCH_CSV`.
    """
    df = _read_csv_commented(
        path or RISC0_ROCM_BENCH_CSV,
        numeric=("wall_s", "receipt_bytes", "loadavg"),
    )
    try:
        base = float(df.loc[df["config"] == "fork-cpu", "wall_s"].iloc[0])
        df["speedup_vs_fork_cpu"] = base / df["wall_s"]
    except Exception:  # noqa: BLE001 - derived column is best-effort
        pass
    return df


def load_risc0_rocm_stage4_bench(path: Optional[Path] = None):
    """Path I Stage-4 — the point bench of the ~4-segment Cartesi-step prove.

    Columns: ``config, backend, threads, workload, wall_s, receipt_bytes, verify, solo,
    loadavg, notes``. The three configs (``fork-cpu`` / ``fork-gpu`` / ``installed-r0vm``)
    all seal to a receipt of the identical 1112064-byte **size** and stock-verify (the
    seal bytes themselves differ per run). This is the
    point-in-time Stage-4 headline; :func:`load_risc0_rocm_bench` is the fresh same-code
    re-measurement. Defaults to :data:`RISC0_ROCM_STAGE4_BENCH_CSV`.
    """
    return _read_csv_commented(
        path or RISC0_ROCM_STAGE4_BENCH_CSV,
        numeric=("threads", "wall_s", "receipt_bytes", "loadavg"),
    )


def load_risc0_rocm_phases(path: Optional[Path] = None):
    """Path I A1 — per-phase wall breakdown of the hybrid segment-STARK as a ``DataFrame``.

    Columns: ``phase, engine, cpu_timer_ms, pct_of_prove, hal_op_calls,
    rocprofv3_kernel_ms, note``. The ``engine`` field tags each phase **iGPU**
    (``GPU_iGPU``) vs **CPU-delegated** (``CPU_delegated`` witgen/accum, ``CPU``/
    ``CPU_default`` glue); the ``prove_total`` row carries the totals
    (``cpu_timer_ms`` = full prove wall, ``rocprofv3_kernel_ms`` = GPU-busy ms, so
    ~59.4% of the prove runs on the iGPU). Defaults to :data:`RISC0_ROCM_PHASE_CSV`.
    """
    return _read_csv_commented(
        path or RISC0_ROCM_PHASE_CSV,
        numeric=("cpu_timer_ms", "pct_of_prove", "hal_op_calls", "rocprofv3_kernel_ms"),
    )


def risc0_rocm_amdahl(df=None) -> dict:
    """Path I A1 — the Part-B Amdahl ceiling, **computed** from the phase breakdown.

    Sums the CPU-delegated ``witgen`` + ``accum`` share ``f`` and returns the best-case
    ceiling ``1/(1-f)`` (if that slice went to zero on the GPU) plus a realistic envelope
    for finite GPU speedups ``s`` on that slice (``1/((1-f) + f/s)``). Also reports the
    measured iGPU / CPU wall split from the ``prove_total`` row. Every number is derived
    from :data:`RISC0_ROCM_PHASE_CSV` at call time (never hard-coded), so it cannot drift.
    Returns ``{}`` if the artefact is unavailable (never raises). Keys: ``witgen_pct,
    accum_pct, witgen_accum_pct, ceiling, gpu_busy_pct, cpu_pct, envelope``.
    """
    try:
        if df is None:
            df = load_risc0_rocm_phases()
        pcts = df.set_index("phase")["pct_of_prove"]
        witgen = float(pcts.get("witgen"))
        accum = float(pcts.get("accum"))
        f = (witgen + accum) / 100.0
        total = df[df["phase"] == "prove_total"].iloc[0]
        prove_ms = float(total["cpu_timer_ms"])
        gpu_ms = float(total["rocprofv3_kernel_ms"])
        gpu_busy = 100.0 * gpu_ms / prove_ms
        envelope = {s: 1.0 / ((1.0 - f) + f / s) for s in (0.5, 1.0, 2.0, 5.0)}
        return {
            "witgen_pct": witgen,
            "accum_pct": accum,
            "witgen_accum_pct": witgen + accum,
            "ceiling": (1.0 / (1.0 - f)) if f < 1.0 else float("inf"),
            "gpu_busy_pct": gpu_busy,
            "cpu_pct": 100.0 - gpu_busy,
            "envelope": envelope,
        }
    except Exception:  # noqa: BLE001 - laptop-safe: missing/odd artefact -> {}
        return {}


def load_risc0_rocm_correctness(path: Optional[Path] = None):
    """Path I — the bit-for-bit "ported + proven" kernel table as a list of dicts.

    Parses the committed reading-note (:data:`RISC0_ROCM_PATH_I_MD`) "What was ported +
    proven" markdown table into rows ``{layer, kernels, golden, gate, checks}`` where
    ``checks`` is the integer GPU==CPU PASS count parsed from the gate cell (e.g. the
    26k-LOC generated ``eval_check`` -> 1024 checks; ``1768 + 400`` -> 2168). Every row is
    bit-for-bit == risc0's own ``CpuHal`` / CPU-C++ golden. Returns ``[]`` (never raises)
    if the artefact is unavailable. Defaults to :data:`RISC0_ROCM_PATH_I_MD`.
    """
    rows: list = []
    try:
        md_path = Path(path or RISC0_ROCM_PATH_I_MD)
        text = md_path.read_text()
        _provenance(md_path)
    except Exception:  # noqa: BLE001 - laptop-safe
        return rows
    lines = text.splitlines()
    header = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.count("|") >= 3 and "layer" in low and "kernels" in low and "gate" in low:
            header = i
            break
    if header is None:
        return rows

    def _clean(cell: str) -> str:
        return cell.replace("**", "").replace("`", "").strip()

    for ln in lines[header + 1:]:
        if "|" not in ln:
            break
        parts = [p.strip() for p in ln.strip().strip("|").split("|")]
        if len(parts) < 4 or set(parts[0]) <= {"-", " ", ":"}:  # skip the --- separator
            continue
        layer, kernels, golden, gate = (
            _clean(parts[0]), _clean(parts[1]), _clean(parts[2]), _clean(parts[3]),
        )
        nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", gate)]
        rows.append({
            "layer": layer, "kernels": kernels, "golden": golden,
            "gate": gate, "checks": (sum(nums) if nums else None),
        })
    return rows


#: The audit differential sentence in ``stage4-gate.md`` ("GPU path is real (not a
#: CPU fallback) — audit differential test"), matched against markdown-emphasis-
#: stripped text. Both arms are captured by ONE regex on purpose: the rocm and
#: non-rocm telemetry only form a controlled A/B because they come from this single
#: audited run, so they must be parsed together or not at all.
_RISC0_AUDIT_DIFFERENTIAL_RE = re.compile(
    r"(\d+)\s*%\s*busy\s+across\s+(\d+)\s+samples\s*/\s*(\d+)\s*s\s+with\s+"
    r"(\d+)\s+HipHal\s+markers\s*,\s*vs\s+the\s+rocm\s+binary['\u2019]?s\s+"
    r"(\d+)\s*%\s*busy\s*\(\s*(\d+)\s*/\s*(\d+)\s+samples\s*\)\s*\+\s*"
    r"(\d+)\s+markers"
)

#: The Stage-4 gate prove's own rocm-smi observation ("GPU busy peaked at 95%
#: (80/83 rocm-smi samples nonzero) during the prove"). A SINGLE condition — that
#: section has no non-rocm control — so its sample counts must never be paired
#: against the audit's non-rocm arm.
_RISC0_PROVE_BUSY_RE = re.compile(
    r"GPU\s+busy\s+peaked\s+at\s*(\d+)\s*%\s*\(\s*(\d+)\s*/\s*(\d+)\s+"
    r"rocm-smi\s+samples\s+nonzero\s*\)"
)


def load_risc0_rocm_gate(path: Optional[Path] = None,
                         ledger_path: Optional[Path] = None) -> dict:
    """Path I Stage-4 — the end-to-end gate facts as a dict (parsed from committed md).

    Reads the committed Stage-4 gate write-up (:data:`RISC0_ROCM_GATE_MD`) + engine-map
    ledger (:data:`RISC0_ROCM_LEDGER_MD`) and returns the correctness / GPU-path-real
    facts behind the seal panel. **Field names carry their run** because
    ``stage4-gate.md`` documents TWO different rocm-smi captures, and only one of them
    is a controlled comparison.

    **The audit differential (a controlled A/B — safe to compare).** The audit ran the
    non-rocm binary on the *identical session* as the rocm binary and reported both arms
    in one sentence, so these eight values are the only ones that may be plotted against
    each other:

    ==========================  ===============================================
    key                         meaning
    ==========================  ===============================================
    audit_paired                True iff BOTH arms parsed from that one sentence
    audit_gpu_busy_pct          rocm arm: iGPU busy (95)
    audit_gpu_busy_samples      rocm arm: nonzero rocm-smi samples (375)
    audit_gpu_total_samples     rocm arm: total rocm-smi samples (385)
    audit_gpu_markers           rocm arm: HipHal markers (4 = 4 segments)
    audit_nonrocm_busy_pct      non-rocm arm: iGPU busy (0)
    audit_nonrocm_samples       non-rocm arm: rocm-smi samples (156)
    audit_nonrocm_seconds       non-rocm arm: wall of the sampled run (178)
    audit_nonrocm_markers       non-rocm arm: HipHal markers (0)
    ==========================  ===============================================

    **The Stage-4 gate prove (a SINGLE condition — do NOT pair).** The "Gate: GPU seal
    verifies" section reports its own rocm-smi capture with no non-rocm control, so its
    sample counts are provenance for that run alone:

    ==========================  ===============================================
    prove_gpu_busy_pct          iGPU busy peak during the gate prove (95)
    prove_gpu_busy_samples      nonzero rocm-smi samples (80)
    prove_gpu_total_samples     total rocm-smi samples (83)
    ==========================  ===============================================

    **Run-independent facts.**

    ==========================  ===============================================
    verify_ok                   stock ``cargo risczero verify`` -> ``Receipt is valid!``
    receipt_bytes               seal size (== the golden CPU seal, 1112064)
    dualhal_pass/total          risc0's own DualHal CpuHal==HipHal tests (15/15)
    ==========================  ===============================================

    Back-compat: the older unqualified keys ``gpu_busy_pct``, ``gpu_busy_samples``,
    ``gpu_total_samples``, ``hal_markers``, ``fallback_busy_pct``, ``fallback_samples``
    and ``fallback_markers`` are still populated, but they now alias the **audit
    differential** fields so that any caller pairing a ``gpu_*`` against a ``fallback_*``
    gets the controlled comparison. Before 2026-08-28 the ``gpu_busy_samples`` /
    ``gpu_total_samples`` aliases carried the gate prove's 80/83 while
    ``fallback_samples`` carried the audit's 156 — two unrelated runs (see
    ``docs/validation-ledger.md``). Prefer the qualified names in new code.

    Every value degrades to ``None`` if unparseable and ``audit_paired`` to ``False``;
    the function never raises. Defaults to :data:`RISC0_ROCM_GATE_MD` /
    :data:`RISC0_ROCM_LEDGER_MD`.
    """
    try:
        gate_path = Path(path or RISC0_ROCM_GATE_MD)
        text = gate_path.read_text()
        _provenance(gate_path)
    except Exception:  # noqa: BLE001 - laptop-safe
        text = ""
    try:
        ledger_md = Path(ledger_path or RISC0_ROCM_LEDGER_MD)
        ledger = ledger_md.read_text()
        _provenance(ledger_md)
    except Exception:  # noqa: BLE001 - laptop-safe
        ledger = ""
    # Drop markdown emphasis so the prose regexes below see plain sentences
    # (the artefact bolds fragments mid-number, e.g. "**95% busy (375/385 ...").
    plain = re.sub(r"\*+", "", text)
    blob = f"{text}\n{ledger}"

    def _num(pattern: str, src: str = text):
        m = re.search(pattern, src)
        if not m:
            return None
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:  # noqa: BLE001
            return None

    facts: dict = {}
    facts["verify_ok"] = "Receipt is valid!" in text
    facts["receipt_bytes"] = _num(r"Receipt:\s*\*{0,2}(\d[\d,]*)\s*bytes")

    # --- the controlled A/B: both arms or neither -------------------------------
    m = _RISC0_AUDIT_DIFFERENTIAL_RE.search(plain)
    facts["audit_paired"] = m is not None
    for key in ("audit_gpu_busy_pct", "audit_gpu_busy_samples",
                "audit_gpu_total_samples", "audit_gpu_markers",
                "audit_nonrocm_busy_pct", "audit_nonrocm_samples",
                "audit_nonrocm_seconds", "audit_nonrocm_markers"):
        facts[key] = None
    if m:
        facts["audit_nonrocm_busy_pct"] = int(m.group(1))
        facts["audit_nonrocm_samples"] = int(m.group(2))
        facts["audit_nonrocm_seconds"] = int(m.group(3))
        facts["audit_nonrocm_markers"] = int(m.group(4))
        facts["audit_gpu_busy_pct"] = int(m.group(5))
        facts["audit_gpu_busy_samples"] = int(m.group(6))
        facts["audit_gpu_total_samples"] = int(m.group(7))
        facts["audit_gpu_markers"] = int(m.group(8))

    # --- the gate prove's own single-condition capture --------------------------
    facts["prove_gpu_busy_pct"] = None
    facts["prove_gpu_busy_samples"] = None
    facts["prove_gpu_total_samples"] = None
    p = _RISC0_PROVE_BUSY_RE.search(plain)
    if p:
        facts["prove_gpu_busy_pct"] = int(p.group(1))
        facts["prove_gpu_busy_samples"] = int(p.group(2))
        facts["prove_gpu_total_samples"] = int(p.group(3))

    m = (re.search(r"DualHal[^\d]{0,80}?(\d+)/(\d+)", blob)
         or re.search(r"(\d+)/(\d+)\s*(?:CpuHal==HipHal\s*)?equality tests PASS", blob))
    if m:
        facts["dualhal_pass"] = int(m.group(1))
        facts["dualhal_total"] = int(m.group(2))

    # --- back-compat aliases, deliberately pointing at the CONTROLLED pair -----
    facts["gpu_busy_pct"] = (facts["audit_gpu_busy_pct"]
                             if facts["audit_gpu_busy_pct"] is not None
                             else facts["prove_gpu_busy_pct"])
    facts["gpu_busy_samples"] = facts["audit_gpu_busy_samples"]
    facts["gpu_total_samples"] = facts["audit_gpu_total_samples"]
    facts["hal_markers"] = (facts["audit_gpu_markers"]
                            if facts["audit_gpu_markers"] is not None
                            else _num(r"\+\s*(\d+)\s*markers"))
    facts["fallback_busy_pct"] = facts["audit_nonrocm_busy_pct"]
    facts["fallback_samples"] = facts["audit_nonrocm_samples"]
    facts["fallback_markers"] = facts["audit_nonrocm_markers"]
    return facts


def risc0_rocm_scope(bench=None, phases=None) -> dict:
    """Path I — the honesty framing for the hybrid STARK, as a structured dict.

    Combines the verbatim **say / never-say** guardrail (policy text, the overclaim
    firewall) with the LIVE numbers computed from the committed artefacts
    (:func:`load_risc0_rocm_bench` for the 5.46x + 1.25x codegen gap,
    :func:`risc0_rocm_amdahl` for the <=1.41x ceiling) so the scope box can neither
    overclaim nor drift. Returns keys ``headline, say, never, speed, amdahl, scope_out``.
    Numbers degrade to ``None`` if an artefact is missing; never raises. This is the
    single source the deck / handout / talking-points should mirror.
    """
    speedup = codegen = None
    try:
        b = bench if bench is not None else load_risc0_rocm_bench()
        gpu = float(b.loc[b["config"] == "fork-gpu", "wall_s"].iloc[0])
        cpu = float(b.loc[b["config"] == "fork-cpu", "wall_s"].iloc[0])
        inst = float(b.loc[b["config"] == "installed", "wall_s"].iloc[0])
        speedup = cpu / gpu
        codegen = inst / cpu
    except Exception:  # noqa: BLE001
        pass
    am = risc0_rocm_amdahl(phases)
    ceiling = am.get("ceiling")
    wa = am.get("witgen_accum_pct")

    sx = f"{speedup:.2f}x" if speedup else "~5.46x"
    cx = f"{codegen:.2f}x" if codegen else "~1.25x"
    ceil = f"{ceiling:.2f}x" if ceiling else "~1.41x"
    wapct = f"{wa:.1f}%" if wa else "28.9%"

    return {
        "headline": (
            "RISC0 zkVM segment-STARK now produces a stock-verifier-accepted seal on "
            "gfx1151 (hybrid, path-i)."
        ),
        "say": [
            "a GPU-produced rv32im SEGMENT-STARK seal, accepted by the STOCK "
            "`cargo risczero verify` (Receipt is valid!)",
            "bit-for-bit == risc0's own CpuHal (DualHal 15/15); eval_check compiles "
            "native HIP + is bit-for-bit GPU==CPU",
            "HYBRID: STARK math + eval_check, recursion eval/accum, and Groth16 "
            "MSM/NTT on the iGPU; segment witgen/accum, recursion witness/prefix-products, "
            "and Groth16 witness generation remain CPU-delegated",
        ],
        "never": [
            "the iGPU proves the whole zkVM / a pure-GPU prover",
            "beats a 5090",
            "drop the stock r0vm (CPU-only) vs scoped v2.3.2 fork (hybrid) distinction",
        ],
        "speed": (
            f"{sx} on ONE ~4-segment poseidon2 Cartesi-step prove (flat ~5.3-5.5x; the "
            f"old ~6.6-6.8x = {sx} x a {cx} local-vs-shipped codegen gap) — "
            "workload-specific; correctness is the hard guarantee."
        ),
        "amdahl": (
            f"witgen + accum = {wapct} of the prove -> Amdahl ceiling <={ceil}; Part B "
            "empirically confirmed full-GPU witgen isn't worth it -> the hybrid split is "
            "the measured sweet spot."
        ),
        "scope_out": (
            "segment witgen/accum, recursion witness/prefix-products, and Groth16 witness "
            "generation remain CPU; stock r0vm stays CPU-only. Complete Groth16 receipt "
            "speed is 0.973x, measured as an accelerator regression at this circuit "
            "size (CPU gnark 5.764 s vs ROCm 11.102 s), not witness-bound."
        ),
    }


# ---------------------------------------------------------------------------
# Plotting — headless-safe; each plotter returns a matplotlib Figure.
# ---------------------------------------------------------------------------
# Backends that render without a display (safe for nbconvert / SSH / CI).
_NONINTERACTIVE_BACKENDS = frozenset(
    {"agg", "pdf", "svg", "ps", "cairo", "template", "pgf"}
)


def _get_plt():
    """Import pyplot with a headless-safe backend outside Jupyter.

    In a notebook IPython has already selected the inline (Agg-based) backend,
    so we leave it. Anywhere else (``make lab-replay`` via nbconvert, a plain
    ``python`` smoke test, SSH with no ``$DISPLAY``) we force the non-interactive
    ``Agg`` backend unless an explicitly headless backend is already active —
    note that GUI variants like ``TkAgg``/``QtAgg`` are NOT headless-safe.
    """
    import matplotlib

    if not _in_ipython():
        if matplotlib.get_backend().lower() not in _NONINTERACTIVE_BACKENDS:
            matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _ok_throughput(df):
    """Rows that produced a real seal (positive wall + proof_bytes)."""
    return df[(df["wall_seconds"] > 0) & (df["proof_bytes"] > 0)].copy()


def plot_thread_scaling(df=None):
    """Plot the CPU STARK Rayon thread-scaling story; return the ``Figure``.

    Three panels: (1) wall time vs ``RAYON_NUM_THREADS`` (one line per
    ``max_mcycle``), (2) speedup vs threads relative to the 1-thread baseline
    with a plateau annotation at the diminishing-returns knee, and (3) peak RSS
    vs threads showing it stays ~constant (so this is thread-scaling data, not
    GPU/NPU acceleration). Defaults to :func:`load_throughput`.
    """
    if df is None:
        df = load_throughput()
    plt = _get_plt()
    ok = _ok_throughput(df)
    mcycles = sorted(ok["max_mcycle"].dropna().unique())

    fig, (ax_wall, ax_sp, ax_rss) = plt.subplots(1, 3, figsize=(17, 5))

    plateau_thr = None
    for mc in mcycles:
        grp = ok[ok["max_mcycle"] == mc].sort_values("rayon_threads")
        if grp.empty:
            continue
        threads = grp["rayon_threads"].to_numpy()
        wall = grp["wall_seconds"].to_numpy()
        ax_wall.plot(threads, wall, marker="o", label=f"{int(mc)} mcycle")

        base = wall[0]
        speedup = base / wall
        ax_sp.plot(threads, speedup, marker="o", label=f"{int(mc)} mcycle")

        # plateau = first thread count where doubling threads buys < 10% more.
        for i in range(1, len(threads)):
            if threads[i] >= 2 * threads[i - 1] and wall[i] > 0.9 * wall[i - 1]:
                plateau_thr = int(threads[i - 1])
                break

        if "peak_rss_gb" in grp.columns and grp["peak_rss_gb"].notna().any():
            ax_rss.plot(threads, grp["peak_rss_gb"].to_numpy(), marker="o",
                        label=f"{int(mc)} mcycle")

    ax_wall.set_title("Wall time vs RAYON_NUM_THREADS")
    ax_wall.set_xlabel("RAYON_NUM_THREADS")
    ax_wall.set_ylabel("wall seconds")
    ax_wall.set_xscale("log", base=2)
    ax_wall.set_yscale("log")
    ax_wall.grid(True, which="both", ls=":", alpha=0.5)
    ax_wall.legend(title="step size", fontsize=8)

    ax_sp.set_title("Speedup vs 1 thread")
    ax_sp.set_xlabel("RAYON_NUM_THREADS")
    ax_sp.set_ylabel("speedup x")
    ax_sp.set_xscale("log", base=2)
    ax_sp.grid(True, which="both", ls=":", alpha=0.5)
    if plateau_thr:
        ax_sp.axvline(plateau_thr, color="crimson", ls="--", alpha=0.7)
        ax_sp.annotate(
            f"plateau ≈ {plateau_thr} threads\n(>{plateau_thr}→2x buys <10%)",
            xy=(plateau_thr, ax_sp.get_ylim()[1] * 0.5),
            fontsize=8, color="crimson",
        )
    ax_sp.legend(fontsize=8)

    if ax_rss.has_data():
        ax_rss.set_title("Peak RSS vs threads (≈ constant)")
        ax_rss.set_xlabel("RAYON_NUM_THREADS")
        ax_rss.set_ylabel("peak RSS (GB)")
        ax_rss.set_xscale("log", base=2)
        ax_rss.set_ylim(bottom=0)
        ax_rss.grid(True, which="both", ls=":", alpha=0.5)
        ax_rss.legend(title="step size", fontsize=8)
    else:
        ax_rss.set_title("Peak RSS (no data)")
        ax_rss.text(0.5, 0.5, "no peak_rss data", ha="center", va="center")

    fig.suptitle(
        "RISC0 r0vm CPU STARK thread-scaling on AMD Strix Halo "
        "(CPU-only — no GPU/NPU prover)", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _plot_primitive_speedup(df, plt, title=None):
    ok = df[df["gpu_ms"].notna() & df["cpu_ms"].notna()].copy()
    dev = next((d for d in ok["gpu_device"] if isinstance(d, str) and d), "iGPU")
    threads = next((t for t in ok["cpu_threads"] if t == t), "?")  # t==t skips NaN

    fig, (ax_t, ax_sp) = plt.subplots(1, 2, figsize=(13, 5))

    # NB: the third tuple slot must NOT be bound to `title` — that shadowed the
    # `title` parameter, so the kind="bn254" caller's BN254-G1-MSM suptitle was
    # silently replaced by "NTT/FFT (scalar field)" on a figure with no NTT curve.
    for prim, marker, _prim_desc in (("msm", "o", "MSM (G1 multiexp)"),
                                     ("fft", "s", "NTT/FFT (scalar field)")):
        grp = ok[ok["primitive"] == prim].sort_values("log_size")
        if grp.empty:
            continue
        ax_t.plot(grp["log_size"], grp["gpu_ms"], marker=marker,
                  label=f"{prim.upper()} GPU OpenCL")
        ax_t.plot(grp["log_size"], grp["cpu_ms"], marker=marker, ls="--",
                  label=f"{prim.upper()} CPU {threads}t")
        ax_sp.plot(grp["log_size"], grp["speedup"], marker=marker,
                   label=prim.upper())

    ax_t.set_title("Wall time vs size")
    ax_t.set_xlabel("log2(size)")
    ax_t.set_ylabel("wall time (ms)")
    ax_t.set_yscale("log")
    ax_t.grid(True, which="both", ls=":", alpha=0.5)
    ax_t.legend(fontsize=8)

    ax_sp.axhline(1.0, color="gray", ls="--", alpha=0.7)
    ax_sp.text(ax_sp.get_xlim()[0], 1.02, "break-even (GPU == CPU)",
               fontsize=8, color="gray")
    # NO BOUND IS CLAIMED for the NTT/FFT curve: the repo has no NTT roofline /
    # arithmetic-intensity or memory-side PMC (rocprof-ntt.csv collected only
    # VALU/wave/busy; omniperf refuses gfx1151). So the annotation states the
    # measured PHENOMENON and names the CURVE — this sweep is BLS12-381 FFT vs
    # blstrs parallel_fft, and "NTT wins" is a property of that curve, not of NTT
    # (the same iGPU's BN254 Fr NTT vs arkworks loses at 2^18, 0.963x — see
    # MSM_NTT_BACKEND_CSV).
    # Each line is emitted only if its curve is actually on this figure, so the
    # kind="bn254" call (MSM-only by the loader's default filter, enforced by
    # _require_single_primitive) can no longer annotate an absent NTT.
    notes = []
    _fft_sp = ok.loc[ok["primitive"] == "fft", "speedup"].dropna()
    if len(_fft_sp):
        notes.append(f"NTT/FFT above parity throughout, peak {_fft_sp.max():.2f}x "
                     f"@ 2^{int(ok.loc[_fft_sp.idxmax(), 'log_size'])}\n"
                     "(BLS12-381 FFT vs blstrs parallel_fft — no bound claimed)")
    # The MSM crossover is READ OFF THE PLOTTED CURVE, never a literal. This one
    # function draws two different sweeps: the BLS12-381 Tier 1 MSM (crosses at
    # 2^20, 1.059x, best 1.213x @2^22) and the kind="bn254" BN254 G1 MSM (crosses
    # at 2^22), so no single hardcoded size can be right for both. The old
    # hardcoded "MSM crossover ≈ 2^22" was the RETIRED best-of-N BN254 gate and
    # was simply wrong on the BLS figure it was drawn onto — and because it was
    # raster text inside a PNG, grep / course-drift-check / pdftotext could not
    # see it while it shipped into book.pdf and both dist kits.
    _msm = ok[(ok["primitive"] == "msm") & ok["speedup"].notna()].sort_values("log_size")
    if not _msm.empty:
        _best = _msm.loc[_msm["speedup"].idxmax()]
        _up = _msm[_msm["speedup"] >= 1.0]
        if _up.empty:
            _where = (f"never reaches parity in this sweep (best "
                      f"{float(_best['speedup']):.3f}x @ 2^{int(_best['log_size'])})")
        elif int(_up.iloc[0]["log_size"]) == int(_best["log_size"]):
            _where = (f"crossover 2^{int(_up.iloc[0]['log_size'])} "
                      f"({float(_up.iloc[0]['speedup']):.3f}x, also this sweep's best)")
        else:
            _where = (f"crossover 2^{int(_up.iloc[0]['log_size'])} "
                      f"({float(_up.iloc[0]['speedup']):.3f}x), best "
                      f"{float(_best['speedup']):.3f}x @ 2^{int(_best['log_size'])}")
        notes.append(
            f"MSM {_where} — size-gated by the ec-gpu OpenCL KERNEL,\n"
            "not by MSM and not by the shared LPDDR5X (a native HIP kernel on "
            "this same chip is 2.0-2.2x faster and wins from 2^16)")
    ax_sp.set_title("Speedup = CPU / GPU  (>1 ⇒ iGPU wins)")
    ax_sp.set_xlabel("log2(size)")
    ax_sp.set_ylabel("speedup x")
    ax_sp.grid(True, which="both", ls=":", alpha=0.5)
    ax_sp.legend(fontsize=9)

    fig.suptitle(
        title or (f"Path E Tier 1 — ZK primitives on AMD {dev} (ec-gpu OpenCL) vs "
                  f"{threads}-thread Zen5"), fontsize=12)
    # The notes are a FIGURE FOOTER, not an in-axes annotation. Carrying both the
    # measured crossover AND the kernel-not-bandwidth attribution makes the lines
    # too long for the old xy=(0.5, 0.95) axes-fraction anchor, which ran them
    # past the right spine and overprinted the FFT curve. tight_layout reserves
    # the strip below the panels; the reserve is sized from the actual line count
    # so a longer/shorter note can never be clipped.
    _foot = 0.0
    if notes:
        _nlines = sum(n.count("\n") + 1 for n in notes)
        _foot = min(0.30, 0.030 * _nlines + 0.035)
    fig.tight_layout(rect=(0, _foot, 1, 0.95))
    if notes:
        fig.text(0.01, 0.012, "\n".join(notes), fontsize=8, color="#444",
                 ha="left", va="bottom")
    return fig


def _plot_groth16_speedup(df, plt):
    ok = df[df["gpu_prove_ms"].notna() & df["cpu_prove_ms"].notna()].copy()
    ok = ok.sort_values("constraints_pow")
    dev = next((d for d in ok["gpu_device"] if isinstance(d, str) and d), "iGPU")
    threads = next((t for t in ok["cpu_threads"] if t == t), "?")
    xs = ok["constraints_pow"]

    fig, (ax_t, ax_sp) = plt.subplots(1, 2, figsize=(13, 5))

    ax_t.plot(xs, ok["gpu_prove_ms"] / 1000.0, marker="o", label="GPU iGPU OpenCL")
    ax_t.plot(xs, ok["cpu_prove_ms"] / 1000.0, marker="s",
              label=f"CPU {threads}t Zen5")
    ax_t.set_title("Groth16 prove time vs circuit size")
    ax_t.set_xlabel("log2(constraints)")
    ax_t.set_ylabel("prove time (s)")
    ax_t.set_yscale("log")
    ax_t.grid(True, which="both", ls=":", alpha=0.5)
    ax_t.legend(fontsize=9)

    ax_sp.plot(xs, ok["speedup"], marker="d", color="tab:green")
    ax_sp.axhline(1.0, color="gray", ls="--", alpha=0.7)
    ax_sp.text(xs.iloc[0] if len(xs) else 0, 1.01, "break-even (GPU == CPU)",
               fontsize=8, color="gray")
    ax_sp.annotate("iGPU reaches parity ≈ 2^22\n"
                   "(4M constraints; size-gated: ec-gpu OpenCL path)",
                   xy=(0.5, 0.1), xycoords="axes fraction", fontsize=8,
                   color="#444")
    ax_sp.set_title("Speedup = CPU / GPU")
    ax_sp.set_xlabel("log2(constraints)")
    ax_sp.set_ylabel("speedup x")
    ax_sp.grid(True, which="both", ls=":", alpha=0.5)

    fig.suptitle(
        f"Path E Tier 2 — full Groth16 prove on AMD {dev} (bellperson OpenCL) "
        f"vs {threads}-thread Zen5", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _plot_halo2_hotspot_speedup(df, plt):
    """Two-panel: wall time + speedup for BOTH halo2 hotspots (msm, ntt)."""
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(13, 5))
    labels = {"msm": "BN254 G1 MSM", "ntt": "BN254 Fr NTT"}
    for prim, (gm, cm) in {"msm": ("o", "s"), "ntt": ("^", "v")}.items():
        sub = df[df["primitive"] == prim].sort_values("log_size")
        if sub.empty:
            continue
        lbl = labels[prim]
        ax_t.plot(sub["log_size"], sub["gpu_ms"], marker=gm, label=f"GPU {lbl}")
        ax_t.plot(sub["log_size"], sub["cpu_ms"], marker=cm, ls="--",
                  label=f"CPU {lbl}")
        ax_s.plot(sub["log_size"], sub["speedup"], marker=gm, label=lbl)
    ax_t.set_yscale("log"); ax_t.set_xlabel("logrows = log2(size)")
    ax_t.set_ylabel("wall time (ms)"); ax_t.set_title("halo2 hotspots: wall time")
    ax_t.grid(True, which="both", ls=":", alpha=0.5); ax_t.legend(fontsize=8)
    ax_s.axhline(1.0, color="gray", ls="--", alpha=0.7)
    ax_s.set_xlabel("logrows = log2(size)"); ax_s.set_ylabel("speedup x")
    ax_s.set_title("Speedup = CPU / GPU (>1 => iGPU wins)")
    ax_s.grid(True, which="both", ls=":", alpha=0.5); ax_s.legend(fontsize=8)
    fig.suptitle("Path E Track 3 — halo2 hotspot ceiling on AMD iGPU "
                 "(ec-gpu OpenCL) vs Zen 5 (arkworks + ark-poly)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _plot_g2_fft_speedup(df, plt):
    """Two-panel wall-time + speedup for the Step 4a G2 (Fq2) MSM + QAP-FFT sweep."""
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(13, 5))
    series = {
        "msm_g2": ("BN254 G2 (Fq2) MSM", "o"),
        "fft": ("radix-2 fft", "s"),
        "ifft": ("radix-2 ifft", "^"),
        "coset_fft": ("coset fft", "v"),
        "coset_ifft": ("coset ifft", "D"),
    }
    for prim, (lbl, mk) in series.items():
        sub = df[df["primitive"] == prim].sort_values("log_size")
        if sub.empty:
            continue
        ax_t.plot(sub["log_size"], sub["gpu_ms"], marker=mk, label=f"GPU {lbl}")
        ax_s.plot(sub["log_size"], sub["speedup"], marker=mk, label=lbl)
    ax_t.set_yscale("log"); ax_t.set_xlabel("log2(size)")
    ax_t.set_ylabel("wall time (ms)"); ax_t.set_title("G2 MSM + QAP-FFT: GPU wall time")
    ax_t.grid(True, which="both", ls=":", alpha=0.5); ax_t.legend(fontsize=8)
    ax_s.axhline(1.0, color="gray", ls="--", alpha=0.7)
    ax_s.set_xlabel("log2(size)"); ax_s.set_ylabel("speedup x")
    ax_s.set_title("Speedup = CPU / GPU (>1 => iGPU wins)")
    ax_s.grid(True, which="both", ls=":", alpha=0.5); ax_s.legend(fontsize=8)
    fig.suptitle("Path E Step 4a — BN254 G2 (Fq2) MSM + QAP radix-2 FFT on AMD iGPU "
                 "(ec-gpu OpenCL) vs Zen 5 (arkworks + ark-poly)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_speedup(df=None, kind: str = "primitive"):
    """Plot the iGPU-OpenCL vs CPU speedup story; return the ``Figure``.

    ``kind="primitive"`` (default) draws MSM + NTT wall time and speedup, annotating
    the NTT/FFT above-parity plateau — that curve is BLS12-381 FFT vs blstrs
    ``parallel_fft``, and no bound is claimed for it — and the MSM crossover, which
    is **read off the plotted curve rather than hardcoded**: the BLS12-381 sweep
    (from :func:`load_gpu_primitive`) crosses at **2^20 (1.059x)** and peaks at
    1.213x @2^22, while the ``kind="bn254"`` sweep below crosses at 2^22, so no one
    literal is right for both figures. The footer also names the attribution: the
    gating is a property of the ec-gpu OpenCL **kernel**, not of MSM and not of the
    shared LPDDR5X.
    ``kind="groth16"`` draws full-Groth16 prove time and the speedup reaching
    CPU parity at ~2^22 (from :func:`load_gpu_groth16`). ``kind="bn254"`` draws
    the Path E Phase 3 **BN254 G1 MSM** (the curve Demo B/C prove over), iGPU
    OpenCL vs arkworks CPU (from :func:`load_gpu_bn254`; MSM-only rows).
    ``kind="halo2"`` draws BOTH Path E Track 3 halo2 prover hotspots — BN254 G1
    MSM + Fr NTT — wall time and speedup (from :func:`load_gpu_halo2_hotspot`).
    ``kind="g2fft"`` draws the Step 4a BN254 G2 (Fq2) MSM + QAP radix-2 FFT
    primitives (from :func:`load_gpu_g2_fft`). Pass ``df`` to override the source.
    """
    plt = _get_plt()
    kind = kind.lower()
    if kind == "primitive":
        return _plot_primitive_speedup(df if df is not None else load_gpu_primitive(), plt)
    if kind == "groth16":
        return _plot_groth16_speedup(df if df is not None else load_gpu_groth16(), plt)
    if kind == "bn254":
        # MSM-only BY CONSTRUCTION now, not by accident. This used to hold only
        # because gpu-bn254.csv labels its extra rows `ntt` while the loop below
        # looks for `fft`, so the 2^8-2^12 / up-to-1469.749x rows fell through the
        # gap silently. Rename either side and this figure would have grown an NTT
        # curve under an "BN254 G1 MSM" title. The loader now filters and the
        # guard now raises, so neither is left to a naming coincidence.
        return _plot_primitive_speedup(
            _require_single_primitive(
                df if df is not None else load_gpu_bn254(primitive="msm"),
                'plot_speedup(kind="bn254")', "msm"),
            plt,
            title="Path E Phase 3 — BN254 G1 MSM on AMD iGPU (ec-gpu OpenCL) vs "
                  "arkworks CPU (the Demo B/C proof-path curve)")
    if kind == "halo2":
        return _plot_halo2_hotspot_speedup(
            df if df is not None else load_gpu_halo2_hotspot(), plt)
    if kind == "g2fft":
        return _plot_g2_fft_speedup(
            df if df is not None else load_gpu_g2_fft(), plt)
    raise ValueError(
        f"plot_speedup: kind must be 'primitive', 'groth16', 'bn254', 'halo2' or "
        f"'g2fft', got {kind!r}")


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
    _fig_source_note(fig, AI_INFER_CSV)
    return fig


def plot_zkml_faithful_summary(save: Optional[Path] = None):
    """One figure summarising Demo G (G4): zkLLM prove_s/proof_KB + zkRAG prove_s/visited.

    Reads :func:`load_zkllm_prove_info` + :func:`load_zkrag_journal` /
    :func:`load_zkrag_proof_info` (all committed JSON; CPU-only proving). Two
    panels: (1) prove wall time (s) for the zkLLM EZKL attention proof and the
    zkRAG RISC0 STARK; (2) zkRAG HNSW navigation — nodes visited vs the index
    size ``n`` (the pruning that makes the search proof small). HONESTY: both
    proofs are CPU-only on AMD; iGPU/NPU do not prove these. Returns the
    ``Figure`` (and saves a PNG if ``save`` is given).
    """
    plt = _get_plt()
    import numpy as np

    zll = load_zkllm_prove_info()
    journal = load_zkrag_journal()
    pinfo = load_zkrag_proof_info()

    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(13, 5))

    names = ["zkLLM\nattention (EZKL)", "zkRAG\nHNSW (RISC0 STARK)"]
    proves = [float(zll.get("prove_seconds", 0)), float(pinfo.get("prove_seconds", 0))]
    x = np.arange(len(names))
    bars = ax_t.bar(x, proves, color=["tab:purple", "tab:red"], width=0.5)
    for b, v in zip(bars, proves):
        ax_t.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}s",
                  ha="center", va="bottom", fontsize=9)
    ax_t.set_title("Prove wall time (CPU-only on AMD)")
    ax_t.set_ylabel("prove seconds")
    ax_t.set_xticks(x); ax_t.set_xticklabels(names, fontsize=9)
    ax_t.grid(True, axis="y", ls=":", alpha=0.5)
    proof_kb = float(zll.get("proof_bytes", 0)) / 1024.0
    receipt_kb = float(pinfo.get("receipt_bytes", 0)) / 1024.0
    ax_t.annotate(f"zkLLM proof {proof_kb:.0f} KB · zkRAG receipt {receipt_kb:.0f} KB\n"
                  f"recall {journal.get('recall')}/{journal.get('k')} · "
                  f"verify {float(pinfo.get('verify_seconds', 0)) * 1000:.0f} ms",
                  xy=(0.5, 0.92), xycoords="axes fraction", ha="center", va="top",
                  fontsize=8, color="#444")

    n = int(journal.get("n", 0)); visited = int(journal.get("num_visited", 0))
    ax_v.bar([0, 1], [n, visited], color=["#bbbbbb", "tab:green"], width=0.5)
    ax_v.set_xticks([0, 1]); ax_v.set_xticklabels(["index n", "visited"], fontsize=9)
    ax_v.set_title("zkRAG HNSW navigation (pruning)")
    ax_v.set_ylabel("nodes")
    pruned = (1 - visited / n) * 100 if n else 0
    ax_v.text(1, visited, f"{visited}/{n}\n(~{pruned:.0f}% pruned)",
              ha="center", va="bottom", fontsize=9)
    ax_v.grid(True, axis="y", ls=":", alpha=0.5)

    fig.suptitle("Demo G (G4) — faithful zkLLM / zkRAG, reduced-scale, CPU-only proving",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save is not None:
        fig.savefig(str(save), dpi=110, bbox_inches="tight")
    return fig


def plot_zkllm_amd_split(split: Optional[dict] = None, save: Optional[Path] = None):
    """One figure for the zkLLM engine-split: timeline + iGPU MSM frontier.

    Left panel — the end-to-end timeline of the proven path on a log-seconds
    axis: iGPU forward | CPU forward | CPU EZKL prove | CPU verify, each bar
    coloured by the engine that runs it (the proof dwarfs the forward by orders
    of magnitude). Right panel — the Path E **BN254 G1 MSM** speedup
    (:func:`load_gpu_bn254`, filtered to ``primitive == "msm"``): the KZG MSM that
    dominates Halo2 proving *could* offload to the iGPU, annotated with the
    EZKL-on-AMD blocker. The panel's own title states the range **read off that
    plotted curve** (clean-solo 0.654-1.091x over 2^16-2^22), NOT split.json's
    ``msm_speedup_min``/``msm_speedup_max``. Those keys held the retired,
    contention-inflated 0.994-1.352x capture when this was written; split.json
    has since been regenerated and now agrees (0.654/1.091). Sourcing the title
    from the plotted curve stays anyway — that is what makes the title and the
    line it sits above structurally unable to disagree, rather than merely
    agreeing today. Reads
    :func:`load_zkllm_split` (pass ``split`` to override). HONESTY: iGPU
    accelerates the model FORWARD only (size-gated); EZKL Halo2 proving is
    CPU-only on AMD; the MSM offload is a documented FRONTIER EZKL does not wire
    (CUDA/Metal only) — no GPU-proving claim. Returns the ``Figure`` (saves a PNG
    if ``save`` is given).
    """
    if split is None:
        split = load_zkllm_split()
    plt = _get_plt()
    from matplotlib.patches import Patch

    cpu_fwd_s = (split.get("forward_ms_cpu") or 0) / 1000.0
    igpu_fwd_s = (split.get("forward_ms_igpu") or 0) / 1000.0
    prove_s = split.get("prove_seconds") or 0
    verify_s = split.get("verify_seconds")
    verify_plot = verify_s if isinstance(verify_s, (int, float)) and verify_s > 0 else 0.05

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(15, 5.6),
                                     gridspec_kw={"width_ratios": [1.7, 1.0]})

    verify_known = isinstance(verify_s, (int, float)) and verify_s > 0
    # 4th tuple element overrides the numeric label (None = auto-format).
    bars = [
        ("forward · iGPU\n(MIGraphX gfx1151)", igpu_fwd_s, "tab:blue", None),
        ("forward · CPU\n(onnxruntime 32t)", cpu_fwd_s, "tab:orange", None),
        ("prove · CPU\n(EZKL Halo2, 32t + 94GB)", prove_s, "tab:red", None),
        ("verify · CPU\n(EZKL — PROOF VERIFIED)", verify_plot, "tab:green",
         None if verify_known else "\u2713 verified (sub-s, not timed)"),
    ]
    bars = [b for b in bars if b[1] and b[1] > 0]
    vals = [b[1] for b in bars]
    y = list(range(len(bars)))[::-1]
    ax_t.barh(y, vals, color=[b[2] for b in bars], height=0.6)
    for yi, v, b in zip(y, vals, bars):
        if b[3] is not None:
            txt = b[3]
        elif v >= 1:
            txt = f"{v:.2f} s"
        elif v >= 0.001:
            txt = f"{v * 1000:.2f} ms"
        else:
            txt = f"{v * 1e6:.0f} \u00b5s"
        ax_t.text(v * 1.15, yi, txt, va="center", fontsize=9)
    ax_t.set_yticks(y)
    ax_t.set_yticklabels([b[0] for b in bars], fontsize=9)
    ax_t.set_xscale("log")
    ax_t.set_xlabel("wall time (seconds, log scale)")
    ax_t.set_xlim(right=max(vals) * 4)
    ax_t.grid(True, axis="x", ls=":", alpha=0.5)
    fs = split.get("forward_speedup")
    fs_txt = (f"forward(b1·s8): CPU {split.get('forward_ms_cpu')} ms vs iGPU "
              f"{split.get('forward_ms_igpu')} ms = {fs:.2f}x"
              if isinstance(fs, (int, float)) else
              "forward iGPU row not captured on this host (CPU baseline only)")
    ax_t.set_title("End-to-end: ONE attention block across AMD engines\n"
                   f"{fs_txt}  ·  proof CPU-only", fontsize=10)
    ax_t.legend(handles=[
        Patch(color="tab:blue", label="iGPU (model forward — size-gated)"),
        Patch(color="tab:orange", label="CPU forward (wins at single-block scale)"),
        Patch(color="tab:red", label="CPU proof (EZKL Halo2 — CPU-only)"),
        Patch(color="tab:green", label="CPU verify (PROOF VERIFIED)"),
    ], fontsize=7.5, loc="lower right")

    # MSM-ONLY, and the range is read off the curve actually drawn. Two bugs used
    # to meet in this panel: (a) no primitive filter, so gpu-bn254.csv's three
    # `ntt` microbench rows (2^8-2^12, up to 1469.749x) were plotted into a panel
    # titled "KZG MSM" and set a 0-1400 y-axis that flattened the real MSM curve
    # to an invisible line; (b) the title quoted split.json's msm_speedup_min /
    # msm_speedup_max, which at the time still held the RETIRED contention-inflated
    # 0.994-1.352x capture that the 2026-06-18 clean-solo re-bench superseded, so
    # the title stated numbers its own curve contradicted. Both were raster text
    # inside a PNG, invisible to grep / course-drift-check / pdftotext.
    #
    # (b)'s artefact side is now resolved too — split.json was regenerated and its
    # keys read 0.654/1.091 — but the title keeps reading off the curve: agreeing
    # by construction is the fix, agreeing by coincidence is what broke before.
    #
    # (a) is now defended THREE deep, because it is the one that shipped: the
    # loader filters by default, _require_single_primitive raises on a mixed
    # frame, and the `== "msm"` mask below stays as belt-and-braces.
    _msm_range = None
    try:
        bn = _require_single_primitive(load_gpu_bn254(primitive="msm"),
                                      "plot_zkllm_amd_split KZG-MSM panel", "msm")
        ok = bn[(bn["primitive"] == "msm") & bn["speedup"].notna()].sort_values("log_size")
        ax_m.plot(ok["log_size"], ok["speedup"], marker="o", color="tab:purple")
        ax_m.axhline(1.0, color="gray", ls="--", alpha=0.7)
        ax_m.text(ok["log_size"].iloc[0], 1.02, "break-even (iGPU == CPU)",
                  fontsize=8, color="gray")
        ax_m.set_xlabel("log2(MSM size)")
        ax_m.set_ylabel("speedup = CPU / iGPU")
        ax_m.set_ylim(bottom=0)
        ax_m.grid(True, ls=":", alpha=0.5)
        if not ok.empty:
            _msm_range = (float(ok["speedup"].min()), float(ok["speedup"].max()),
                          int(ok["log_size"].min()), int(ok["log_size"].max()))
    except Exception:  # noqa: BLE001 — inset is best-effort
        ax_m.text(0.5, 0.5, "gpu-bn254.csv unavailable", ha="center", va="center")
    # No range at all rather than a stale one if the CSV could not be read.
    _msm_sub = ("BN254 G1 MSM range unavailable" if _msm_range is None else
                f"Path E BN254 G1 — {_msm_range[0]:.3f}-{_msm_range[1]:.3f}x "
                f"over 2^{_msm_range[2]}-2^{_msm_range[3]} (clean-solo)")
    ax_m.set_title(f"Stage-3 FRONTIER: KZG MSM on the iGPU\n({_msm_sub})",
                   fontsize=10)
    ax_m.annotate(
        "The MSM dominating Halo2 proving\nCOULD offload to gfx1151 (OpenCL),\n"
        "but EZKL wires only CUDA/Metal\n"
        f"=> {split.get('msm_blocker')}\n(documented frontier, not claimed)",
        xy=(0.5, 0.04), xycoords="axes fraction", ha="center", va="bottom",
        fontsize=8, color="#700",
        bbox=dict(boxstyle="round", fc="#fff3f3", ec="#e0b0b0"))

    fig.suptitle(
        "zkLLM AMD engine-split — iGPU forward (size-gated) · CPU Halo2 proof · "
        "iGPU MSM frontier (EZKL CUDA/Metal-only)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save is not None:
        fig.savefig(str(save), dpi=120, bbox_inches="tight")
    return fig


def plot_zkllm_scale_sweep(df=None, save: Optional[Path] = None):
    """Plot the zkLLM-split (A) scale-up-to-the-94GB-cap story; return the ``Figure``.

    Two panels over the growing proven unit (``head`` → ``mha`` → ``layer`` → …):

    1. **EZKL Halo2 prove cost vs unit size** — peak prover RSS (GB) bars on the
       left axis and ``prove_seconds`` markers on a twin right axis, with the
       calibrated ``logrows`` on each x-tick and a dashed **94 GB unified-memory
       ceiling**. The first config whose prove did NOT complete (``status != ok``)
       is drawn as a hatched **CAP** bar to the ceiling at its calibrated logrows
       (no prove/RSS value), making the proving cap explicit: on this box it sits
       between the 12-head attention (~82 GB) and a full encoder layer.
    2. **iGPU forward speedup** (``fwd_ms_cpu / fwd_ms_igpu``) vs unit size with a
       break-even line — the forward win is size-gated, climbing toward parity as
       the unit grows.

    HONESTY: EZKL Halo2 proving is CPU-only on AMD; the Strix Halo enabler is 64
    threads + 94 GB system RAM removing the CPU-prover OOM wall as logrows climbs. The
    iGPU only accelerates the model FORWARD. Defaults to
    :func:`load_zkllm_scale_sweep`.
    """
    if df is None:
        df = load_zkllm_scale_sweep()
    plt = _get_plt()

    df = df.reset_index(drop=True)
    configs = df["config"].astype(str).tolist()
    statuses = (df["status"].astype(str).str.strip().str.lower().tolist()
                if "status" in df.columns else ["ok"] * len(configs))
    ram_gb = float(_read_meminfo_gb()) or 94.0

    def fnum(col, i):
        if col not in df.columns:
            return None
        try:
            v = float(df[col].iloc[i])
        except (TypeError, ValueError):
            return None
        return v if v == v else None  # NaN -> None

    fig, (ax_r, ax_sp) = plt.subplots(1, 2, figsize=(14, 5))
    axp = ax_r.twinx()

    xticklabels = []
    for i, cfg in enumerate(configs):
        capped = statuses[i] != "ok"
        rss, prove, lr = fnum("peak_rss_gb", i), fnum("prove_s", i), fnum("logrows", i)
        if capped:
            ax_r.bar(i, ram_gb, width=0.55, color="none", edgecolor="crimson",
                     hatch="//", linewidth=1.5, zorder=2)
            ax_r.annotate(
                f"CAP\nprove did not\ncomplete\n(RSS → {ram_gb:.0f} GB)",
                xy=(i, ram_gb * 0.50), ha="center", va="center",
                fontsize=7.5, color="crimson", fontweight="bold")
        else:
            if rss is not None:
                ax_r.bar(i, rss, width=0.55, color="tab:blue", zorder=2)
                ax_r.text(i, rss + ram_gb * 0.015, f"{rss:.1f} GB",
                          ha="center", va="bottom", fontsize=8, color="tab:blue")
            if prove is not None:
                axp.plot(i, prove, marker="o", ms=9, color="tab:red", zorder=3)
                axp.text(i, prove, f"  {prove:.0f} s", ha="left", va="center",
                         fontsize=8, color="tab:red")
        xticklabels.append(cfg if lr is None else f"{cfg}\nlogrows {int(lr)}")

    ok_x = [i for i in range(len(configs))
            if statuses[i] == "ok" and fnum("prove_s", i) is not None]
    if len(ok_x) > 1:
        axp.plot(ok_x, [fnum("prove_s", i) for i in ok_x],
                 color="tab:red", lw=1.3, zorder=1)

    ax_r.axhline(ram_gb, color="crimson", ls="--", alpha=0.8)
    ax_r.text(0, ram_gb * 1.005, f"{ram_gb:.0f} GB unified-memory ceiling",
              fontsize=8, color="crimson", va="bottom")
    ax_r.set_ylim(0, ram_gb * 1.15)
    ax_r.set_ylabel("peak prover RSS (GB)", color="tab:blue")
    ax_r.tick_params(axis="y", labelcolor="tab:blue")
    ax_r.set_xticks(range(len(configs)))
    ax_r.set_xticklabels(xticklabels, fontsize=8)
    ax_r.set_xlabel("proven unit (growing)")
    axp.set_ylabel("EZKL Halo2 prove time (s)", color="tab:red")
    axp.tick_params(axis="y", labelcolor="tab:red")
    axp.set_ylim(bottom=0)
    ax_r.set_title("EZKL prove cost vs unit size (CPU-only) — capped by 94 GB RSS")
    ax_r.grid(True, axis="y", ls=":", alpha=0.4)

    sp_x, sp_v = [], []
    for i in range(len(configs)):
        fc, fg = fnum("fwd_ms_cpu", i), fnum("fwd_ms_igpu", i)
        if fc is not None and fg:
            sp_x.append(i)
            sp_v.append(fc / fg)
    if sp_x:
        ax_sp.plot(sp_x, sp_v, marker="o", color="tab:green")
        for i, v in zip(sp_x, sp_v):
            ax_sp.text(i, v, f"  {v:.2f}x", fontsize=8, color="#333",
                       va="bottom")
    ax_sp.axhline(1.0, color="gray", ls="--", alpha=0.7)
    ax_sp.text(0, 1.02, "break-even (iGPU == CPU)", fontsize=8, color="gray")
    ax_sp.annotate("forward win is SIZE-GATED:\na single sub-block is dispatch-bound\n"
                   "(CPU wins), climbing toward parity\nas the unit grows",
                   xy=(0.5, 0.12), xycoords="axes fraction", ha="center",
                   fontsize=8, color="#444")
    ax_sp.set_xticks(range(len(configs)))
    ax_sp.set_xticklabels(configs, fontsize=8)
    ax_sp.set_xlabel("proven unit (growing)")
    ax_sp.set_ylabel("forward speedup = CPU / iGPU")
    ax_sp.set_ylim(bottom=0)
    ax_sp.set_title("iGPU forward speedup vs unit size (the AI model, not the proof)")
    ax_sp.grid(True, axis="y", ls=":", alpha=0.5)

    fig.suptitle(
        "zkLLM track A — scaling the proven unit up the EZKL Halo2 curve on AMD "
        "Strix Halo (CPU-only proof; 94 GB unified memory is the enabler & the cap)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save is not None:
        fig.savefig(str(save), dpi=120, bbox_inches="tight")
    return fig


def plot_zkllm_msm(df=None, save: Optional[Path] = None):
    """Plot the zkLLM-split (C) iGPU-vs-CPU attention-matmul Groth16 story.

    Two panels vs circuit size (log2 constraints ``m``):

    1. **End-to-end Groth16 prove time on BLS12-381** — iGPU (bellperson/opencl)
       vs CPU (``BELLMAN_NO_GPU``), log y.
    2. **Speedup = CPU/GPU** for BOTH the end-to-end Groth16 prove AND the
       standalone **BN254 G1 MSM** at matching ``m``, with a break-even line and
       the iGPU **crossover** marked (the smallest ``m`` where the end-to-end
       prove flips above 1).

    HONESTY: size-gated — small attention loses on the iGPU (a property of the
    ec-gpu OpenCL kernel, not of MSM and not of the shared LPDDR5X); the iGPU
    end-to-end win appears only at large padded ``m`` (~2²²).
    The BN254 MSM column is host-contention-depressed here (the clean solo ceiling
    is Path E's 0.65-1.09×, crossover ~2²² — 2026-06-18 re-bench); full BN254 Groth16
    stays the documented ``ark-groth16``
    injection blocker (``docs/zkllm-igpu-proof-scope.md``). Defaults to
    :func:`load_zkllm_msm`.
    """
    if df is None:
        df = load_zkllm_msm()
    plt = _get_plt()

    ok = df[df["gpu_prove_ms"].notna() & df["cpu_prove_ms"].notna()].sort_values(
        "constraints_pow").copy()
    dev = next((d for d in ok["gpu_device"] if isinstance(d, str) and d), "gfx1151")
    threads = next((int(t) for t in ok["cpu_threads"] if t == t), "?")
    xs = ok["constraints_pow"]

    fig, (ax_t, ax_sp) = plt.subplots(1, 2, figsize=(14, 5))

    ax_t.plot(xs, ok["gpu_prove_ms"] / 1000.0, marker="o",
              label=f"iGPU {dev} (bellperson OpenCL)")
    ax_t.plot(xs, ok["cpu_prove_ms"] / 1000.0, marker="s", ls="--",
              label=f"CPU {threads}t (BELLMAN_NO_GPU)")
    ax_t.set_yscale("log")
    ax_t.set_xlabel("log2(constraints m)")
    ax_t.set_ylabel("Groth16 prove time (s)")
    ax_t.set_title("Attention-matmul Groth16 on BLS12-381 — iGPU vs CPU")
    ax_t.grid(True, which="both", ls=":", alpha=0.5)
    ax_t.legend(fontsize=8)

    ax_sp.plot(xs, ok["prove_speedup"], marker="o", color="tab:green",
               label="end-to-end Groth16 (BLS12-381)")
    if "bn254_msm_speedup" in ok.columns and ok["bn254_msm_speedup"].notna().any():
        ax_sp.plot(xs, ok["bn254_msm_speedup"], marker="^", ls="--",
                   color="tab:purple", label="standalone BN254 G1 MSM (m-matched)")
    ax_sp.axhline(1.0, color="gray", ls="--", alpha=0.7)
    if len(xs):
        ax_sp.text(xs.iloc[0], 1.03, "break-even (iGPU == CPU)", fontsize=8,
                   color="gray")
    cross = ok[ok["prove_speedup"] > 1.0]
    if not cross.empty:
        cx = float(cross["constraints_pow"].iloc[0])
        cm = float(cross["constraints"].iloc[0])
        cs = float(cross["prove_speedup"].iloc[0])
        ax_sp.axvline(cx, color="crimson", ls=":", alpha=0.8)
        ax_sp.annotate(f"iGPU crossover\nm ≈ {cm / 1e6:.2f}M  ({cs:.2f}×)",
                       xy=(cx, cs), xytext=(cx - 1.0, cs * 0.55),
                       fontsize=8, color="crimson", ha="right",
                       arrowprops=dict(arrowstyle="->", color="crimson"))
    ax_sp.set_xlabel("log2(constraints m)")
    ax_sp.set_ylabel("speedup = CPU / iGPU")
    ax_sp.set_ylim(bottom=0)
    ax_sp.set_title("Speedup (>1 ⇒ iGPU wins) — size-gated")
    ax_sp.grid(True, which="both", ls=":", alpha=0.5)
    ax_sp.legend(fontsize=8)

    fig.suptitle(
        f"zkLLM track C — attention-matmul Groth16 on AMD {dev} (BLS12-381, "
        f"bellperson OpenCL) vs {threads}-thread Zen5 + standalone BN254 MSM",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save is not None:
        fig.savefig(str(save), dpi=120, bbox_inches="tight")
    return fig


def plot_zkrag_scale(df=None, save: Optional[Path] = None):
    """Plot the zkRAG Phase 2 unified-memory scale-up story; return the ``Figure``.

    Two panels, x = index size ``n`` (log2):

    1. **Prove cost vs n** — the real STARK ``prove_seconds`` (``dev_mode==False``
       points, solid) on the left axis, and ``total_cycles`` for *every* point
       (incl. the larger ``dev_mode`` executor runs, hollow) on a twin right
       axis, so the cycle growth is visible past the real-prove points.
    2. **Peak RSS vs n** — the real-STARK prover peak RSS (GB) with a dashed
    **15 GB laptop system-RAM** line. HONESTY: the STARK is CPU-only
    on AMD (no GPU prover); this is a Zen 5 + system-memory win, exactly
       like notebook 03.

    Defaults to :func:`load_zkrag_scale`.
    """
    if df is None:
        df = load_zkrag_scale()
    plt = _get_plt()

    df = df.sort_values("n")
    real = df[~df["dev_mode"]] if "dev_mode" in df.columns else df
    ram_gb = float(_read_meminfo_gb()) or 94.0

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: prove_seconds (real) + total_cycles (all) on a twin axis.
    if not real.empty:
        ax_t.plot(real["n"], real["prove_seconds"], marker="o", color="tab:red",
                  label="real STARK prove (CPU-only)")
    ax_t.set_xscale("log", base=2)
    ax_t.set_xlabel("index size n (vectors)")
    ax_t.set_ylabel("real STARK prove time (s)", color="tab:red")
    ax_t.tick_params(axis="y", labelcolor="tab:red")
    ax_t.grid(True, which="both", ls=":", alpha=0.5)

    ax_c = ax_t.twinx()
    ax_c.plot(df["n"], df["total_cycles"] / 1e6, marker="s", ls="--",
              color="tab:gray", label="total cycles (incl. dev-mode points)")
    ax_c.set_ylabel("zkVM cycles (millions)", color="tab:gray")
    ax_c.tick_params(axis="y", labelcolor="tab:gray")
    if "dev_mode" in df.columns and df["dev_mode"].any():
        dev_min = df[df["dev_mode"]]["n"].min()
        ax_t.axvspan(dev_min / 1.4, df["n"].max() * 1.2, color="#f0f0f0", zorder=0)
        ax_t.annotate("dev-mode\n(cycles only,\nno seal)", xy=(dev_min, ax_t.get_ylim()[1] * 0.6),
                      fontsize=8, color="#888", ha="left")
    ax_t.set_title("Prove cost vs index size (CPU-only STARK)")
    lines1, labels1 = ax_t.get_legend_handles_labels()
    lines2, labels2 = ax_c.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    # Panel 2: real CPU-prover peak RSS + a small-laptop system-RAM reference.
    rss = real[real["peak_rss_gb"].notna()] if "peak_rss_gb" in real.columns else real.iloc[0:0]
    if not rss.empty:
        ax_r.plot(rss["n"], rss["peak_rss_gb"], marker="o", color="tab:blue",
                  label="real prover peak RSS")
    ax_r.axhline(15.0, color="crimson", ls="--", alpha=0.8)
    ax_r.text(ax_r.get_xlim()[0] if not rss.empty else 0, 15.2,
              "15 GB laptop system-RAM ceiling", fontsize=8, color="crimson")
    ax_r.axhline(ram_gb, color="tab:green", ls=":", alpha=0.7)
    ax_r.text(ax_r.get_xlim()[0] if not rss.empty else 0, ram_gb - 6,
              f"{ram_gb:.0f} GB unified LPDDR5X", fontsize=8, color="tab:green")
    ax_r.set_xscale("log", base=2)
    ax_r.set_xlabel("index size n (vectors)")
    ax_r.set_ylabel("peak RSS (GB)")
    ax_r.set_ylim(bottom=0, top=max(ram_gb * 1.05, 18))
    ax_r.grid(True, which="both", ls=":", alpha=0.5)
    ax_r.set_title("Prover peak RSS vs index size")
    ax_r.annotate("94 GB system RAM holds this CPU-only prover\n"
                  "where a 15 GB laptop cannot",
                  xy=(0.5, 0.12), xycoords="axes fraction", ha="center",
                  fontsize=8, color="#444")
    if not rss.empty:
        ax_r.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "zkRAG Phase 2 — unified-memory scale-up (RISC0 STARK is CPU-only on AMD; "
        "94 GB system RAM + 64 Zen 5 threads lift the laptop RAM wall)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save is not None:
        fig.savefig(str(save), dpi=110, bbox_inches="tight")
    return fig


def plot_zkrag_mem(df=None, save: Optional[Path] = None):
    """Plot the Group B CPU-prover host-RSS crossing; return the ``Figure``.

    x = ``segment_po2`` and y = real-STARK prover peak **host RSS (GB)**. The
    dashed 15 GB line is a small laptop's system-RAM ceiling, not GPU VRAM.
    r0vm proves fixed-size segments, so raising ``segment_po2`` is the genuine
    memory lever. HONESTY: the iGPU embeds the query; this prove is CPU-only.
    """
    if df is None:
        df = load_zkrag_mem()
    plt = _get_plt()

    df = df.sort_values("segment_po2")
    real = df[~df["dev_mode"]] if "dev_mode" in df.columns else df
    ram_gb = float(_read_meminfo_gb()) or 94.0

    fig, ax = plt.subplots(figsize=(9, 5.5))

    rss = real[real["peak_rss_gb"].notna()] if "peak_rss_gb" in real.columns else real.iloc[0:0]
    if not rss.empty:
        ax.plot(rss["segment_po2"], rss["peak_rss_gb"], marker="o", color="tab:blue",
                lw=2, label="real STARK prover peak RSS (CPU-only)")
        # Annotate each point with its peak RSS and crossing status.
        for _, row in rss.iterrows():
            crossed = row["peak_rss_gb"] > 15.0
            ax.annotate(
                f"{row['peak_rss_gb']:.1f} GB" + ("\nexceeds 15 GB laptop RAM" if crossed else ""),
                xy=(row["segment_po2"], row["peak_rss_gb"]),
                xytext=(0, 8), textcoords="offset points", fontsize=8,
                ha="center", color="crimson" if crossed else "#444",
            )

    # Shade host RSS above a small laptop's system-RAM ceiling.
    ax.axhspan(15.0, max(ram_gb * 1.05, 18), color="#fdecec", zorder=0)
    ax.axhline(15.0, color="crimson", ls="--", alpha=0.85)
    x0 = rss["segment_po2"].min() if not rss.empty else 20
    ax.text(x0, 15.4, "15 GB laptop system-RAM ceiling", fontsize=8, color="crimson")
    ax.axhline(ram_gb, color="tab:green", ls=":", alpha=0.7)
    ax.text(x0, ram_gb - 6, f"{ram_gb:.0f} GB unified LPDDR5X (carries it)",
            fontsize=8, color="tab:green")

    ax.set_xlabel("ZKRAG_SEGMENT_PO2 (r0vm segment size, log2 cycles/segment)")
    ax.set_ylabel("prover peak RSS (GB)")
    ax.set_ylim(bottom=0, top=max(ram_gb * 1.05, 18))
    ax.grid(True, which="both", ls=":", alpha=0.5)
    if not rss.empty:
        ax.set_xticks(sorted(rss["segment_po2"].unique()))
        ax.legend(fontsize=9, loc="upper left")
    ax.set_title(
        "zkRAG Group B — raising the r0vm segment size crosses a laptop RAM ceiling\n"
        "(RISC0 STARK is CPU-only on AMD; this is host RSS, not GPU VRAM)",
        fontsize=10)
    fig.tight_layout()
    if save is not None:
        fig.savefig(str(save), dpi=110, bbox_inches="tight")
    return fig


def plot_zkrag_msm_speedup(df=None, save: Optional[Path] = None):
    """Plot the Phase 3 zkRAG-retrieval iGPU-vs-CPU Groth16 story; return the ``Figure``.

    Two panels vs padded circuit size ``m`` (log2):

    1. **Groth16 prove time vs m** — iGPU (bellperson/opencl, BLS12-381) vs CPU
       (``BELLMAN_NO_GPU``), log y; the SAME committed ``zkrag.index.json`` the
       RISC0 STARK attests, relation-parity-gated before timing.
    2. **Speedup = CPU / iGPU** vs ``m`` with a dashed **1.0× parity** line; the
       iGPU climbs to parity only at the largest ~2^22 instance (the size-gated
       crossover is marked) — a property of the ec-gpu OpenCL kernel, not of MSM
       and not of the shared LPDDR5X.

    HONESTY: this is the MSM-SNARK re-cast where the iGPU genuinely accelerates the
    PROOF primitives (size-gated, parity ~1.0× at 2^22); the RISC0 zkRAG STARK
    itself stays CPU-only on AMD. Defaults to :func:`load_zkrag_msm`.
    """
    if df is None:
        df = load_zkrag_msm()
    plt = _get_plt()
    import numpy as np

    ok = df[df["gpu_prove_s"].notna() & df["cpu_prove_s"].notna()].sort_values("m").copy()
    ok["log2m"] = np.log2(ok["m"].to_numpy())
    xs = ok["log2m"]

    fig, (ax_t, ax_sp) = plt.subplots(1, 2, figsize=(14, 5))

    ax_t.plot(xs, ok["gpu_prove_s"], marker="o", label="iGPU (bellperson OpenCL)")
    ax_t.plot(xs, ok["cpu_prove_s"], marker="s", ls="--",
              label="CPU (BELLMAN_NO_GPU)")
    ax_t.set_yscale("log")
    ax_t.set_xlabel("log2(padded constraints m)")
    ax_t.set_ylabel("Groth16 prove time (s)")
    ax_t.set_title("zkRAG-retrieval Groth16 on BLS12-381 — iGPU vs CPU")
    ax_t.grid(True, which="both", ls=":", alpha=0.5)
    ax_t.legend(fontsize=8)

    ax_sp.plot(xs, ok["speedup"], marker="d", color="tab:green",
               label="end-to-end Groth16 (BLS12-381)")
    ax_sp.axhline(1.0, color="gray", ls="--", alpha=0.7)
    if len(xs):
        ax_sp.text(xs.iloc[0], 1.02, "parity (iGPU == CPU)", fontsize=8,
                   color="gray")
    cross = ok[ok["speedup"] >= 1.0]
    if not cross.empty:
        cx = float(cross["log2m"].iloc[0])
        cm = float(cross["m"].iloc[0])
        cs = float(cross["speedup"].iloc[0])
        ax_sp.axvline(cx, color="crimson", ls=":", alpha=0.8)
        ax_sp.annotate(f"iGPU reaches parity\nm ≈ {cm / 1e6:.2f}M (2^{int(round(cx))})  "
                       f"{cs:.2f}×",
                       xy=(cx, cs), xytext=(cx - 1.2, max(cs * 0.6, 0.45)),
                       fontsize=8, color="crimson", ha="right",
                       arrowprops=dict(arrowstyle="->", color="crimson"))
    ax_sp.set_xlabel("log2(padded constraints m)")
    ax_sp.set_ylabel("speedup = CPU / iGPU")
    ax_sp.set_ylim(bottom=0)
    ax_sp.set_title("Speedup (>1 ⇒ iGPU wins) — size-gated, parity ≈ 2^22")
    ax_sp.grid(True, which="both", ls=":", alpha=0.5)
    ax_sp.legend(fontsize=8)

    fig.suptitle(
        "zkRAG Phase 3 — retrieval proof re-cast as Groth16/BLS12-381 on AMD iGPU "
        "(bellperson OpenCL) vs Zen 5 — the iGPU touches the PROOF (size-gated); "
        "the RISC0 STARK stays CPU-only", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save is not None:
        fig.savefig(str(save), dpi=110, bbox_inches="tight")
    return fig


def plot_bigmodel_memory(df=None, save: Optional[Path] = None,
                         discrete_gb: float = 16.0, cap_key: str = "cap_16gb",
                         order: Optional[list] = None):
    """Plot the Demo H unified-memory flagship; return the ``Figure``.

    Two panels over the three conditions (``full_igpu`` | ``cap_16gb`` | ``cpu``):

    1. **Peak iGPU VRAM used (GB)** per condition as bars, with a red dashed
       **device-pool budget** (16 GB by default). The full-offload bar crosses it;
       this marks where spilling begins, not where execution becomes impossible.
       Discrete host-mapped / managed memory remains addressable across PCIe.
    2. **Throughput** (prefill + generation tok/s) per condition — the cliff
       from full iGPU offload down to the ~16GB-budget cap (and the CPU baseline).

    HONESTY: this accelerates the AI MODEL (the RAG generator), NOT the proof —
    the RISC0 STARK stays CPU-only on AMD. The iGPU VRAM is a 32GB carveout of
    the 94GB unified pool. The ``cap_16gb`` contrast is GENEROUS to the discrete
    card (the CPU spill here is the same LPDDR5X, not a PCIe spill / hard OOM).
    Defaults to :func:`load_bigmodel`.

    The discrete-card reference line (``discrete_gb``), the cap condition
    (``cap_key``), and the condition ``order`` are parametrized; the defaults
    reproduce the original 16GB figure byte-for-byte. Pass
    ``discrete_gb=32.0, cap_key="cap_32gb"`` to render the >32GB flagship
    variants. A missing ``cpu`` row (e.g. the BF16 Halo sweep) is tolerated —
    only the conditions actually present in ``df`` are drawn.
    """
    if df is None:
        df = load_bigmodel()
    plt = _get_plt()

    if order is None:
        order = ["full_igpu", cap_key, "cpu"]
    labels = {
        "full_igpu": "full iGPU\n(-ngl 99)",
        cap_key: f"{discrete_gb:.0f}GB cap\n(partial -ngl)",
        "cpu": "CPU\n(-ngl 0)",
    }
    by = {r["condition"]: r for _, r in df.iterrows()}
    conds = [c for c in order if c in by]
    xs = list(range(len(conds)))
    xlabels = [labels.get(c, c) for c in conds]

    def _num(row, key):
        try:
            v = float(row[key])
            return 0.0 if v != v else v  # NaN -> 0
        except (KeyError, TypeError, ValueError):
            return 0.0

    vram = [_num(by[c], "peak_vram_gb") for c in conds]
    gtt = [_num(by[c], "peak_gtt_gb") for c in conds]
    total = [_num(by[c], "peak_gpu_gb") or (vram[i] + gtt[i]) for i, c in enumerate(conds)]
    pp = [_num(by[c], "prefill_tps") for c in conds]
    gen = [_num(by[c], "gen_tps") for c in conds]
    weights = next((_num(by[c], "weights_gb") for c in conds if _num(by[c], "weights_gb")), 0.0)
    DISCRETE_GB = float(discrete_gb)

    fig, (ax_v, ax_t) = plt.subplots(1, 2, figsize=(13, 5))

    top_v = max(max(total) * 1.18 if total else 0, DISCRETE_GB * 1.3)
    ax_v.axhspan(DISCRETE_GB, top_v, color="#fdecec", zorder=0)
    ax_v.bar(xs, vram, width=0.6, color="#2a9d8f", label="VRAM carveout used", zorder=2)
    ax_v.bar(xs, gtt, width=0.6, bottom=vram, color="#8ecae6",
             label="GTT used (unified LPDDR5X)", zorder=2)
    ax_v.axhline(DISCRETE_GB, color="#e63946", ls="--", lw=2, zorder=3)
    ax_v.text(len(conds) - 0.5, DISCRETE_GB + 0.3,
              f"{DISCRETE_GB:.0f}GB device-pool budget (spill above)",
              color="#e63946", ha="right", fontsize=9, fontweight="bold")
    for x, t in zip(xs, total):
        if t:
            crossed = t > DISCRETE_GB
            ax_v.text(x, t + 0.3,
                      f"{t:.1f} GB" + (f"\n> {DISCRETE_GB:.0f}GB" if crossed else ""),
                      ha="center", fontsize=9,
                      color="crimson" if crossed else "#333")
    ax_v.set_xticks(xs)
    ax_v.set_xticklabels(xlabels, fontsize=9)
    ax_v.set_ylabel("peak iGPU GPU-resident memory (GB)")
    ax_v.set_ylim(0, top_v)
    ax_v.set_title(f"Memory: {weights:.0f}GB model needs > {DISCRETE_GB:.0f}GB GPU memory\n"
                   f"(VRAM carveout + GTT; full offload crosses the {DISCRETE_GB:.0f}GB line)")
    ax_v.legend(fontsize=8, loc="upper right")
    ax_v.grid(True, axis="y", ls=":", alpha=0.5)

    w = 0.38
    ax_t.bar([x - w / 2 for x in xs], pp, width=w, label="prefill tok/s",
             color="#264653")
    ax_t.bar([x + w / 2 for x in xs], gen, width=w, label="generation tok/s",
             color="#f4a261")
    for x, val in zip([x - w / 2 for x in xs], pp):
        if val:
            ax_t.text(x, val, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    for x, val in zip([x + w / 2 for x in xs], gen):
        if val:
            ax_t.text(x, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax_t.set_xticks(xs)
    ax_t.set_xticklabels(xlabels, fontsize=9)
    ax_t.set_ylabel("throughput (tok/s)")
    ax_t.set_title(f"Throughput cliff:\nfull iGPU offload vs {DISCRETE_GB:.0f}GB cap vs CPU")
    ax_t.legend(fontsize=9)
    ax_t.grid(True, axis="y", ls=":", alpha=0.5)

    fig.suptitle(
        f"Demo H — >{DISCRETE_GB:.0f}GB LLM on the AMD Radeon 8060S iGPU (llama.cpp/HIP, gfx1151)\n"
        "accelerates the AI MODEL (RAG generator), not the proof; iGPU VRAM is a "
        "32GB carveout of the 94GB pool",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _fig_source_note(fig, BIGMODEL_CSV)
    if save is not None:
        fig.savefig(str(save), dpi=110, bbox_inches="tight")
    return fig


def plot_uma_bandwidth(df=None, save: Optional[Path] = None):
    """Track D — APU unified-memory bandwidth bar chart; return the ``Figure``.

    Horizontal bars of measured SAXPY GB/s per (alloc_kind, op): the explicit
    ``hipMalloc`` H2D/D2H staging copies vs the device-resident kernel, the
    ``hipHostMalloc`` zero-copy kernel (no staging), and the
    ``hipMallocManaged`` cold-first-touch vs warm-resident kernel. The teaching
    point is visible at a glance: on this APU the zero-copy / page-migrated
    kernels reach (warm) the same device-resident bandwidth because CPU and iGPU
    share one LPDDR5X pool — there is no PCIe staging tax. Defaults to
    :func:`load_uma_bandwidth`.
    """
    if df is None:
        df = load_uma_bandwidth()
    plt = _get_plt()

    labels, vals, colors = [], [], []
    palette = {
        "hipMalloc": "#8d99ae",
        "hipHostMalloc": "#2a9d8f",
        "hipMallocManaged": "#e76f51",
    }
    for _, r in df.iterrows():
        kind = str(r.get("alloc_kind", "?"))
        op = str(r.get("op", "?"))
        labels.append(f"{kind}\n{op}")
        try:
            vals.append(float(r.get("gbytes_s")))
        except Exception:
            vals.append(0.0)
        colors.append(palette.get(kind, "#666666"))

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55 * len(labels) + 1.5)))
    y = list(range(len(labels)))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("effective bandwidth (GB/s) — SAXPY, memory-bound")
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.0f}", va="center", fontsize=8)
    solo = str(df["solo"].iloc[0]) if "solo" in df.columns and len(df) else "?"
    ax.set_title(
        "APU unified memory — hipMalloc vs hipHostMalloc vs hipMallocManaged "
        f"(Radeon 8060S / gfx1151; solo={solo})\n"
        "zero-copy + page-migrated reach device-resident BW: one LPDDR5X pool, "
        "no PCIe staging copy",
        fontsize=10)
    fig.tight_layout()
    if save is not None:
        fig.savefig(str(save), dpi=110, bbox_inches="tight")
    return fig


def plot_rocm_libs(df=None, save: Optional[Path] = None):
    """Track A — ROCm library-ecosystem comparison; return the ``Figure``.

    Two panels:

    1. **SGEMM GFLOP/s** — ``rocblas`` / ``hipblaslt`` (Tensile-autotuned) vs the
       course's fixed 16x16 ``hand_tiled`` kernel. (``SKIP`` rows are omitted.)

       🔴 **"The library should win" was an EXPECTATION, and the committed
       artefact only half-confirms it — so it is not written as a fact here.**
       On gfx1151 ``rocblas`` does win (3135.16 vs 1627.84 GFLOP/s = 1.93×), but
       ``hipblaslt`` — the *Tensile-autotuned* arm, i.e. precisely the one the
       "autotuning is the motivation" story was about — **LOSES to the fixed hand
       kernel** at 1196.99 GFLOP/s (**0.74×**). The panel label therefore names
       the winning library rather than saying "library", and states the losing
       arm on the figure as well: the retired label took ``max()`` over the two
       libraries, so it reported a true 1.93× while hiding a 0.74× regression —
       the same retired-best-of-N failure mode being cleaned up elsewhere in this
       module.
    2. **rocFFT complex-C2C GFLOP/s** vs FFT size — a real, fast COMPLEX FFT, with
       a caption reminding that a ZK prover needs a finite-field **NTT** (BN254
       Fr), which rocFFT cannot compute — the wrong tool for SNARK polynomial math.

    Defaults to :func:`load_rocm_libs`.
    """
    if df is None:
        df = load_rocm_libs()
    plt = _get_plt()

    sg = df[df["workload"] == "sgemm"].copy()
    sg = sg[sg["verify"].astype(str) != "SKIP"]
    ft = df[df["workload"] == "fft_complex"].copy()
    ft = ft[ft["verify"].astype(str) != "SKIP"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- panel 1: SGEMM library vs hand kernel ---
    ax = axes[0]
    impl_color = {"rocblas": "#2a9d8f", "hipblaslt": "#3a86b8",
                  "hand_tiled": "#e76f51"}
    impls = list(sg["impl"]) if len(sg) else []
    gfs = [float(x) for x in sg["gflops"]] if len(sg) else []
    cols = [impl_color.get(i, "#666666") for i in impls]
    x = list(range(len(impls)))
    ax.bar(x, gfs, color=cols, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(impls, fontsize=9)
    ax.set_ylabel("GFLOP/s (FP32, compute-bound)")
    n = int(sg["n"].iloc[0]) if len(sg) else 0
    ax.set_title(f"SGEMM {n}x{n}: library (autotuned) vs hand 16x16 tile")
    for xi, v in zip(x, gfs):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    if len(sg) and "hand_tiled" in impls:
        try:
            base = gfs[impls.index("hand_tiled")]
            # BEST-OF-N GUARD. The retired label was
            #     f"library ≈ {max(gfs)/base:.1f}x the hand kernel"
            # whose arithmetic was exact (3135.16/1627.84 = 1.925) but which took
            # max() over the two libraries and then called the result "library".
            # On this artefact that hides a regression: hipBLASLt (1196.99) is
            # 0.74x the hand kernel, i.e. it LOSES. Name the winner, and keep the
            # loser visible — never let one arm speak for the family.
            libs = [(i, g) for i, g in zip(impls, gfs) if i != "hand_tiled"]
            if libs:
                win = max(libs, key=lambda t: t[1])
                lose = min(libs, key=lambda t: t[1])
                # Headroom so the verdict sits ABOVE the bars: at the old y=0.92
                # the text was drawn on top of the (tallest) rocBLAS bar and was
                # unreadable regardless of what it said.
                ax.set_ylim(0, max(gfs) * 1.32)
                ax.text(0.5, 0.985,
                        f"{win[0]} = {win[1] / base:.2f}× the hand kernel",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=9, color=impl_color.get(win[0], "#2a9d8f"))
                if lose[0] != win[0] and lose[1] < base:
                    ax.text(0.5, 0.915,
                            f"but {lose[0]} = {lose[1] / base:.2f}× — it LOSES "
                            "to the hand kernel",
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=8.5, color="#b3261e")
                elif lose[0] != win[0]:
                    ax.text(0.5, 0.915,
                            f"{lose[0]} = {lose[1] / base:.2f}×",
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=8.5, color=impl_color.get(lose[0], "#666666"))
        except Exception:
            pass

    # --- panel 2: rocFFT complex sweep ---
    ax = axes[1]
    if len(ft):
        sizes = [int(x) for x in ft["n"]]
        gfs2 = [float(x) for x in ft["gflops"]]
        xs = list(range(len(sizes)))
        ax.plot(xs, gfs2, "o-", color="#6a4c93", linewidth=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"2^{int(round(math.log2(s)))}"
                            for s in sizes], fontsize=9)
        ax.set_ylabel("GFLOP/s (5·N·log2 N model)")
        for xi, v in zip(xs, gfs2):
            ax.text(xi, v, f" {v:.0f}", va="bottom", fontsize=8)
    ax.set_xlabel("FFT size (complex points)")
    ax.set_title("rocFFT complex C2C — fast, but the WRONG tool for ZK\n"
                 "(SNARKs need a finite-field NTT, not a complex FFT)",
                 fontsize=9)
    fig.suptitle("ROCm library ecosystem on gfx1151 — rocBLAS/hipBLASLt + rocFFT "
                 "(real iGPU run)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save is not None:
        fig.savefig(str(save), dpi=110, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Path I (frontier, scoped) — RISC0 rv32im segment-STARK on the gfx1151 iGPU.
# Plotters for the seal / correctness / speed / phase-Amdahl panels (nb23). Every
# number is pulled from the committed artefacts via the load_risc0_rocm_* loaders;
# all are headless-safe and never require ROCm to render the replay figure.
# ---------------------------------------------------------------------------
_RISC0_IGPU_COLOR = "#2a9d8f"   # iGPU / HipHal (clean win / GPU path)
_RISC0_CPU_COLOR = "#e76f51"    # CPU (same-code baseline)
_RISC0_DELEG_COLOR = "#e9c46a"  # CPU-delegated witgen+accum (the Part B target)
_RISC0_GLUE_COLOR = "#adb5bd"   # CPU glue / installed rzup build


def plot_risc0_rocm_gpu_evidence(gate=None, save: Optional[Path] = None):
    """Panel (a) — the GPU-produced seal is REAL, not a CPU fallback; return the ``Figure``.

    The **audit differential test** as a two-bar chart, drawn from the ``audit_*`` fields
    of :func:`load_risc0_rocm_gate`: on one audited session the ``rocm`` binary holds the
    iGPU at **95% busy** (375/385 rocm-smi samples, 4 HipHal markers) while the
    **non-rocm** binary leaves it at **0%** (156 samples / 178 s, 0 markers) — definitive
    proof the STARK compute ran on the gfx1151. The verdict strip states the hard
    guarantee: the GPU seal is accepted by the **stock** ``cargo risczero verify``
    (``Receipt is valid!``), is the **same size** as the golden CPU seal with the
    **journal region bit-identical**, and passes risc0's own DualHal equality tests.

    **Scope of the receipt claim (do not widen).** ``stage4-gate.md`` states identical
    **size** only, and the seal is **not** byte-reproducible: golden ``step.proof.bin``
    vs ``step.rocm.proof.bin`` differ in 971,857 of 1,112,064 bytes (87.39%); only a
    156-byte serialization header and the 385-byte journal tail
    (``pre_root || mcycle=100 || post_root``, matching ``step.public.json``) match.
    ``z2-bringup-report.md`` §3.3 measured five runs producing five *different*
    1,112,064-byte files, all stock-verified — same size, different bytes, expected
    rather than a defect. **bit-for-bit belongs to the HAL ops (DualHal 15/15) and
    ``eval_check``, never to the receipt file.**

    Both bars come from the SAME audited run. The gate prove's separate single-condition
    capture (``prove_gpu_*``, 80/83 samples) has no non-rocm control and is deliberately
    NOT drawn here — pairing it against the audit's non-rocm arm would compare two
    different experiments. If the audit sentence cannot be parsed as a pair
    (``audit_paired`` is False) the comparison is refused rather than approximated.
    Defaults to :func:`load_risc0_rocm_gate`.
    """
    if gate is None:
        gate = load_risc0_rocm_gate()
    plt = _get_plt()

    gpu_busy = gate.get("audit_gpu_busy_pct")
    cpu_busy = gate.get("audit_nonrocm_busy_pct")
    paired = bool(gate.get("audit_paired")) and None not in (gpu_busy, cpu_busy)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    if not paired:
        # Never approximate a controlled A/B out of unpaired numbers.
        ax.text(0.5, 0.5, "audit differential unavailable\n"
                          "(the paired rocm / non-rocm arms could not be read from "
                          "stage4-gate.md)",
                transform=ax.transAxes, ha="center", va="center", fontsize=10,
                color="#7a2b1a")
        ax.set_axis_off()
        fig.tight_layout()
        if save is not None:
            fig.savefig(str(save), dpi=110, bbox_inches="tight")
        return fig

    gs, gt = gate.get("audit_gpu_busy_samples"), gate.get("audit_gpu_total_samples")
    mk = gate.get("audit_gpu_markers")
    fs, fsec = gate.get("audit_nonrocm_samples"), gate.get("audit_nonrocm_seconds")
    fm = gate.get("audit_nonrocm_markers")

    labels = ["fork GPU\n(r0vm --features rocm)",
              "fork CPU\n(non-rocm, SAME session)"]
    vals = [float(gpu_busy), float(cpu_busy)]
    cols = [_RISC0_IGPU_COLOR, _RISC0_CPU_COLOR]
    notes = [
        (f"{gs}/{gt} samples nonzero\n{mk} HipHal markers"
         if gs and gt else (f"{mk} HipHal markers" if mk else "")),
        ((f"{fs} samples / {fsec} s" if fsec else f"{fs} samples")
         + f"\n{fm} HipHal markers" if fs is not None else ""),
    ]

    x = list(range(len(vals)))
    ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.7, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("iGPU busy during the prove (%)")
    ax.set_ylim(0, 108)
    for xi, v, note in zip(x, vals, notes):
        ax.text(xi, v + 2, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
        if note:
            if v > 40:  # tall bar: caption sits inside, below the top
                ax.text(xi, v - 8, note, ha="center", va="top",
                        fontsize=8, color="#20303a")
            else:       # short bar: caption sits above the % label (no overlap)
                ax.text(xi, v + 12, note, ha="center", va="bottom",
                        fontsize=8, color="#7a2b1a")

    verify = gate.get("verify_ok")
    rb = gate.get("receipt_bytes")
    dp, dt = gate.get("dualhal_pass"), gate.get("dualhal_total")
    strip = []
    if verify:
        strip.append("stock `cargo risczero verify` -> Receipt is valid!")
    if rb:
        # Size + journal only — never "== golden" unqualified (see the scope note
        # in the docstring): the seal body differs on every run.
        strip.append(f"seal {rb:,} B — same size as the golden CPU seal")
        strip.append("journal bit-identical; seal bytes differ per run")
    if dp and dt:
        strip.append(f"DualHal {dp}/{dt} bit-for-bit")
    if strip:
        # Two rows: scoping the receipt claim to size + journal made the old
        # single-line strip wider than the 8.4in figure.
        half = (len(strip) + 1) // 2
        rows = [r for r in ("  ·  ".join(strip[:half]), "  ·  ".join(strip[half:])) if r]
        ax.text(0.5, -0.30, "\n".join(rows), transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color="#1a7f37", linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.4", fc="#eaf6ee", ec="#1a7f37", lw=0.8))
    ax.set_title("The GPU path is REAL — audit differential test on gfx1151\n"
                 "(both arms from ONE audited session; rv32im SEGMENT-STARK seal, "
                 "stock-verifier-accepted)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    _fig_source_note(fig, RISC0_ROCM_GATE_MD, y=0.135)
    if save is not None:
        fig.savefig(str(save), dpi=110, bbox_inches="tight")
    return fig


def plot_risc0_rocm_correctness(rows=None, gate=None, save: Optional[Path] = None):
    """Panel (b) — bit-for-bit GPU==CpuHal per layer; return the ``Figure``.

    A horizontal log-scale bar of the GPU==CPU equality-check counts per HAL layer
    (:func:`load_risc0_rocm_correctness`): field / hash / poly / **circuit (the 26k-LOC
    generated ``eval_check``)** / Merkle. The header states the umbrella guarantee from
    :func:`load_risc0_rocm_gate` — risc0's own ``DualHal`` harness passes **15/15** and the
    generated ``eval_check`` is bit-for-bit GPU==CPU. Defaults to
    :func:`load_risc0_rocm_correctness`.
    """
    if rows is None:
        rows = load_risc0_rocm_correctness()
    if gate is None:
        gate = load_risc0_rocm_gate()
    plt = _get_plt()

    rows = [r for r in rows if r.get("checks")]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    if rows:
        labels = [r["layer"] for r in rows]
        vals = [int(r["checks"]) for r in rows]
        y = list(range(len(rows)))[::-1]  # first layer on top
        cols = [(_RISC0_IGPU_COLOR if r["layer"].lower().startswith("circuit")
                 else "#3a86b8") for r in rows]
        ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.6, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("GPU == CPU equality checks that PASS (log scale)")
        for yi, r in zip(y, rows):
            ax.text(int(r["checks"]) * 1.05, yi, f"  {r['gate']}",
                    va="center", fontsize=8, color="#333")
        ax.set_xlim(right=max(vals) * 3.2)
    else:
        ax.text(0.5, 0.5, "correctness table unavailable — replay a committed artefact",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.axis("off")

    dp, dt = gate.get("dualhal_pass"), gate.get("dualhal_total")
    banner = "Every HAL op bit-for-bit == risc0's own CpuHal / CPU-C++ golden"
    if dp and dt:
        banner = (f"risc0's own DualHal harness: {dp}/{dt} CpuHal==HipHal PASS  ·  "
                  "the 26k-LOC generated eval_check is bit-for-bit GPU==CPU")
    ax.set_title("Correctness is the HARD guarantee — " + banner, fontsize=9.8)
    fig.tight_layout()
    _fig_source_note(fig, RISC0_ROCM_PATH_I_MD, RISC0_ROCM_GATE_MD)
    if save is not None:
        fig.savefig(str(save), dpi=110, bbox_inches="tight")
    return fig


def plot_risc0_rocm_bench(df=None, save: Optional[Path] = None):
    """Panel (c) — the honest same-code speed (5.46x) + codegen caveat; return the ``Figure``.

    Wall-time bars for the three configs in :func:`load_risc0_rocm_bench`: the installed
    rzup ``r0vm``, the same fork built no-rocm (CPU), and the fork built ``--features rocm``
    (iGPU hybrid). Annotates the **apples-to-apples 5.46x** (iGPU vs same-code CPU) and the
    **1.25x local-vs-shipped codegen gap** (installed vs same-code CPU) that inflated the
    old ~6.6-6.8x. The caption keeps the scope honest: workload-specific (poseidon2 +
    hybrid), correctness is the hard guarantee. Defaults to :func:`load_risc0_rocm_bench`.
    """
    if df is None:
        df = load_risc0_rocm_bench()
    plt = _get_plt()

    order = [c for c in ("installed", "fork-cpu", "fork-gpu") if c in set(df["config"])]
    label_map = {
        "installed": "installed r0vm 2.3.2\n(rzup binary, CPU 32t)",
        "fork-cpu": "fork CPU\n(same code, no-rocm, 32t)",
        "fork-gpu": "fork GPU\n(rocm gfx1151 + CPU witgen)",
    }
    color_map = {"installed": _RISC0_GLUE_COLOR, "fork-cpu": _RISC0_CPU_COLOR,
                 "fork-gpu": _RISC0_IGPU_COLOR}
    walls = {c: float(df.loc[df["config"] == c, "wall_s"].iloc[0]) for c in order}

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = list(range(len(order)))
    ax.bar(x, [walls[c] for c in order], width=0.6,
           color=[color_map[c] for c in order], edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[c] for c in order], fontsize=9)
    ax.set_ylabel("prove wall (s) — lower is better")
    top = max(walls.values())
    ax.set_ylim(0, top * 1.28)
    for xi, c in zip(x, order):
        ax.text(xi, walls[c] + top * 0.015, f"{walls[c]:.1f} s",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    if "fork-gpu" in walls and "fork-cpu" in walls:
        sp = walls["fork-cpu"] / walls["fork-gpu"]
        ax.text(x[order.index("fork-gpu")], walls["fork-gpu"] + top * 0.14,
                f"{sp:.2f}x\nvs same-code CPU", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#1a7f37")
    if "installed" in walls and "fork-cpu" in walls:
        cg = walls["installed"] / walls["fork-cpu"]
        ax.text(x[order.index("installed")], walls["installed"] + top * 0.09,
                f"{cg:.2f}x codegen gap\n(local build vs rzup)", ha="center",
                va="bottom", fontsize=8.5, color="#7a5b00")
    ax.set_title("iGPU hybrid vs same-code CPU — 5.46x on ONE ~4-seg poseidon2 "
                 "Cartesi-step prove\n(flat ~5.3-5.5x; workload-specific — correctness "
                 "is the hard guarantee, speed the scoped secondary)", fontsize=9.6)
    fig.tight_layout()
    _fig_source_note(fig, RISC0_ROCM_BENCH_CSV)
    if save is not None:
        fig.savefig(str(save), dpi=110, bbox_inches="tight")
    return fig


def plot_risc0_rocm_phases(df=None, save: Optional[Path] = None):
    """Panel (d) — per-phase engine split + the Part-B Amdahl ceiling; return the ``Figure``.

    Left: horizontal bars of each phase's share of the prove
    (:func:`load_risc0_rocm_phases`), coloured by engine — iGPU (HipHal) vs the
    CPU-delegated **witgen + accum** (the Part B target) vs CPU glue. Right: the Amdahl
    envelope from :func:`risc0_rocm_amdahl` — overall speedup vs the GPU speedup ``s`` on
    the witgen+accum slice, with the Amdahl ceiling (if that 28.9% went to 0 on the
    GPU) drawn as the unreachable asymptote — **<=1.41x on the currently committed
    phase split**. Both the asymptote label and the right-hand panel title FORMAT
    that ceiling out of :func:`risc0_rocm_amdahl`, so a re-measured phase split
    moves the two together; neither is a hardcoded literal. Since accum is a sequential grand-product
    (GPU-hostile) and ``s<=1`` is plausible, the hybrid CPU/GPU split is the measured sweet
    spot. Defaults to :func:`load_risc0_rocm_phases`.
    """
    if df is None:
        df = load_risc0_rocm_phases()
    plt = _get_plt()
    am = risc0_rocm_amdahl(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # --- panel 1: per-phase engine split ---
    ax = axes[0]
    ph = df[df["phase"] != "prove_total"].copy()
    ph = ph.sort_values("pct_of_prove", ascending=True)

    def _eng_style(engine: str):
        e = str(engine)
        if e.startswith("GPU"):
            return _RISC0_IGPU_COLOR, "iGPU (HipHal)"
        if e == "CPU_delegated":
            return _RISC0_DELEG_COLOR, "CPU-delegated (witgen+accum = Part B target)"
        return _RISC0_GLUE_COLOR, "CPU (glue / combos)"

    y = list(range(len(ph)))
    cols = [_eng_style(e)[0] for e in ph["engine"]]
    ax.barh(y, ph["pct_of_prove"], color=cols, edgecolor="black", linewidth=0.5,
            height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(ph["phase"], fontsize=8.5)
    ax.set_xlabel("share of the segment-prove wall (%)")
    for yi, v in zip(y, ph["pct_of_prove"]):
        ax.text(float(v) + 0.3, yi, f"{float(v):.1f}%", va="center", fontsize=8)
    gpu_pct = am.get("gpu_busy_pct")
    cpu_pct = am.get("cpu_pct")
    title = "Where the hybrid prove's wall goes"
    if gpu_pct is not None:
        title += f" — iGPU {gpu_pct:.1f}% / CPU {cpu_pct:.1f}%"
    ax.set_title(title, fontsize=9.8)
    ax.set_xlim(0, float(ph["pct_of_prove"].max()) * 1.25)
    # de-duplicated engine legend
    seen, handles = {}, []
    from matplotlib.patches import Patch  # local import — headless-safe
    for e in ph["engine"]:
        col, lab = _eng_style(e)
        if lab not in seen:
            seen[lab] = True
            handles.append(Patch(facecolor=col, edgecolor="black", label=lab))
    ax.legend(handles=handles, fontsize=7.4, loc="lower right", framealpha=0.95)

    # --- panel 2: Amdahl envelope for Part B ---
    ax = axes[1]
    f = (am.get("witgen_accum_pct") or 28.87) / 100.0
    ceiling = am.get("ceiling") or (1.0 / (1.0 - f))
    ss = [0.3 + 0.02 * i for i in range(int((60 - 0.3) / 0.02) + 1)]
    ax.plot(ss, [1.0 / ((1.0 - f) + f / s) for s in ss], color=_RISC0_IGPU_COLOR,
            linewidth=2.2, label="overall speedup if witgen+accum -> GPU at s")
    ax.axhline(ceiling, ls="--", color="#c1121f", linewidth=1.4)
    ax.text(0.4, ceiling + 0.012, f"Amdahl ceiling <={ceiling:.2f}x "
            f"(witgen+accum = {f*100:.1f}% -> 0)", fontsize=8.4, color="#c1121f")
    ax.axhline(1.0, ls=":", color="#555", linewidth=1.0)
    ax.text(0.34, 1.005, "1.0x = hybrid today (CPU witgen+accum)", fontsize=7.8,
            color="#555")
    env = am.get("envelope") or {}
    for s, mk in ((0.5, "s=0.5x (GPU slower — plausible for the\nsequential accum)"),
                  (1.0, None), (2.0, "s=2x"), (5.0, "s=5x")):
        val = env.get(s) or env.get(float(s))
        if val is None:
            val = 1.0 / ((1.0 - f) + f / s)
        ax.plot([s], [val], "o", color="#264653", ms=5)
        if mk:
            ax.annotate(f"{mk} -> {val:.2f}x", (s, val),
                        textcoords="offset points", xytext=(6, -2 if s < 1 else 6),
                        fontsize=7.6, color="#264653")
    ax.set_xscale("log")
    ax.set_xlabel("GPU speedup s on the witgen+accum slice (log)")
    ax.set_ylabel("overall prove speedup")
    ax.set_ylim(0.6, ceiling + 0.12)
    # DERIVED, never hardcoded. The retired title carried "<=1.41x" as a literal
    # while the red asymptote label ~20 lines up already FORMATTED the same
    # quantity out of risc0_rocm_amdahl(). They agree today (ceiling =
    # 1.40587... -> "1.41"), so this was a latent drift rather than a live error —
    # but one number with two sources of truth on one figure is how the drift
    # starts, and only one of the two would have moved on a re-measure.
    ax.set_title(f"Part B ceiling <={ceiling:.2f}x -> the hybrid split is the "
                 "measured sweet spot", fontsize=9.6)

    fig.suptitle("RISC0 rv32im segment-STARK on gfx1151 — hybrid phase breakdown "
                 "(iGPU STARK math + eval_check; CPU witgen+accum)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _fig_source_note(fig, RISC0_ROCM_PHASE_CSV)
    if save is not None:
        fig.savefig(str(save), dpi=110, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# plot_e2e_pipeline() — the capstone: one query across all five engines.
# ---------------------------------------------------------------------------
def _e2e_engine_color(engine: str) -> Tuple[str, str]:
    """Map a stage's engine string to (fill, accent) colors by keyword."""
    e = engine.lower()
    if "evm" in e or "anvil" in e or "on-chain" in e:
        return ("#e9e2f5", "#6a4c93")          # on-chain — purple
    if "stark" in e or ("cpu" in e and "llama" not in e and "qwen" not in e):
        return ("#dff3ef", "#2a9d8f")          # CPU STARK — teal
    if "unified" in e or "lpddr" in e or "memory" in e:
        return ("#e3f1f8", "#3a86b8")          # unified memory — blue
    if "igpu" in e or "cpu (qwen" in e or "llama" in e or "qwen" in e:
        return ("#fde8df", "#e76f51")          # iGPU AI model — orange
    return ("#eeeeee", "#666666")


def plot_e2e_pipeline(timeline: Optional[dict] = None, save: Optional[Path] = None):
    """Plot the verifiable-RAG e2e pipeline as a 5-stage engine-attribution strip.

    One box per stage, left-to-right, colored by the engine that did the work
    (orange = iGPU AI model · blue = unified memory · teal = CPU STARK · purple =
    on-chain), each carrying the stage name, the engine, a ``LIVE``/``REPLAY``
    badge, and the key metric. Arrows thread the SAME query through all five. The
    footer states the honesty rule: the iGPU does stages 1 & 5 (embed + generate);
    the CPU does stage 3 (the STARK); stage 4 verifies on-chain — the proof is
    CPU-only, the iGPU never proves. Defaults to :func:`load_e2e_timeline`.
    """
    import textwrap

    if timeline is None:
        timeline = load_e2e_timeline()
    plt = _get_plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    stages = sorted(timeline.get("stages", []), key=lambda s: s.get("stage", 0))
    n = len(stages) or 1
    fig, ax = plt.subplots(figsize=(3.0 * n, 4.6))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_w, box_h, y0 = 0.86, 0.52, 0.30
    centers = []
    for i, s in enumerate(stages):
        fill, accent = _e2e_engine_color(s.get("engine", ""))
        x = i + (1 - box_w) / 2
        cx = x + box_w / 2
        centers.append(cx)
        ax.add_patch(FancyBboxPatch(
            (x, y0), box_w, box_h, boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=2, edgecolor=accent, facecolor=fill, zorder=2))
        # stage number + name
        ax.text(cx, y0 + box_h - 0.06, f"{s.get('stage')}. {s.get('name','')}",
                ha="center", va="top", fontsize=10.5, fontweight="bold", color="#222")
        # engine
        eng = "\n".join(textwrap.wrap(s.get("engine", ""), 24))
        ax.text(cx, y0 + box_h - 0.165, eng, ha="center", va="top",
                fontsize=8.2, color=accent, fontweight="bold")
        # status badge
        status = s.get("status", "").upper()
        badge_c = "#1a7f37" if status == "LIVE" else "#8a6d00"
        ax.text(cx, y0 + 0.085, status, ha="center", va="center", fontsize=7.6,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc=badge_c, ec="none"))
        # metric (wrapped, small)
        metric = "\n".join(textwrap.wrap(s.get("metric", ""), 30)[:4])
        ax.text(cx, y0 - 0.02, metric, ha="center", va="top", fontsize=6.6,
                color="#444")
        # arrow to next
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + box_w, y0 + box_h / 2), (i + 1 + (1 - box_w) / 2, y0 + box_h / 2),
                arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#555", zorder=1))

    # query banner (top) + grounded-answer terminus (implied)
    q = timeline.get("query", "")
    ax.text(n / 2, 0.95, "ONE query:  " + textwrap.shorten(q, 92),
            ha="center", va="center", fontsize=9.5, style="italic", color="#0969da")

    fig.suptitle(timeline.get("title",
                 "Verifiable RAG on Strix Halo — one query across all five engines"),
                 fontsize=12.5, fontweight="bold", y=1.02)
    ax.text(n / 2, 0.045,
            "Honesty: the iGPU does stages 1 & 5 (embed + generate — the AI model); "
            "the CPU does stage 3 (the RISC0 STARK); stage 4 verifies on-chain.\n"
            "The proof is CPU-only — the iGPU never proves. \"Verifiable RAG\" = the "
            "retrieval is proven honest and the answer is grounded in the proven docs "
            "(the LLM output itself is NOT proven).",
            ha="center", va="center", fontsize=7.8, color="#57606a")

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _fig_source_note(fig, E2E_TIMELINE)
    if save is not None:
        fig.savefig(str(save), dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# summary_from_full_run_info() — parse the Demo B run summary.
# ---------------------------------------------------------------------------
_TARGET_RE = re.compile(r"^(\S+)\.status=(\S+)\s+wall_seconds=(\S+)\s*$")


def summary_from_full_run_info(path: Optional[Path] = None) -> dict:
    """Parse ``full-run.info`` into a dict (with a ``targets`` ``DataFrame``).

    The file is a mix of ``key=value`` scalars and ``<target>.status=...
    wall_seconds=...`` lines. Returns a dict of every scalar (values kept as
    strings; e.g. ``host``, ``cpu_model``, ``unified_ram_gb``, ``image_id``,
    ``npu.verdict``) plus a ``"targets"`` key holding a pandas ``DataFrame`` with
    columns ``target, status, wall_seconds`` (``wall_seconds`` numeric, ``NaN``
    for ``n/a``). Defaults to :data:`FULL_RUN_INFO`.
    """
    pd = _require_pandas()
    path = Path(path or FULL_RUN_INFO)
    if not path.is_file():
        raise FileNotFoundError(f"labkit: expected committed summary at {path}")

    scalars: dict = {}
    targets = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tmatch = _TARGET_RE.match(line)
        if tmatch:
            name, status, wall = tmatch.groups()
            targets.append({"target": name, "status": status, "wall_seconds": wall})
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            scalars[key.strip()] = value.strip()

    tdf = pd.DataFrame(targets, columns=["target", "status", "wall_seconds"])
    if not tdf.empty:
        tdf["wall_seconds"] = pd.to_numeric(tdf["wall_seconds"], errors="coerce")
    scalars["targets"] = tdf
    return scalars


# ---------------------------------------------------------------------------
# Hardware telemetry — sample rocm-smi while a live workload runs (Phase 1).
# ---------------------------------------------------------------------------
# This is the lab's first window onto the *silicon*: a background thread polls
# ``rocm-smi --csv`` (iGPU use% / package power / VRAM) at a fixed interval while
# the wrapped demo runs, writing a tidy time-series CSV. On the real Strix Halo
# this proves a heavy proof/inference genuinely lights up the iGPU; on a laptop
# (no rocm-smi) the sampler degrades to a no-op and the notebook replays a
# committed snapshot instead. HONESTY: on an APU the rocm-smi ``Power(W)`` is the
# whole-SoC package draw (the iGPU shares the budget with the Zen5 cores), and
# ``GPU use %`` is a coarse activity metric — telemetry shows the engine is busy,
# not a clean per-block power attribution.
def _resolve_telemetry_tool(tool: str) -> Optional[str]:
    """Pick an available SMI tool: prefer ``tool``, then rocm-smi, then amd-smi."""
    for candidate in (tool, "rocm-smi", "amd-smi"):
        if candidate in ("rocm-smi", "amd-smi") and shutil.which(candidate):
            return candidate
    return None


def _parse_rocm_smi_csv(text: str):
    """Parse ``rocm-smi …--csv`` into (power_w, gpu_use_pct, vram_used_mb, vram_total_mb).

    Maps columns by header substring (robust to ordering/label drift) and reads
    the first ``cardN`` device row. Returns ``None`` if nothing parseable.
    """
    rows = list(csv.reader(line for line in text.splitlines() if line.strip()))
    if len(rows) < 2:
        return None
    header = [h.strip().lower() for h in rows[0]]

    def col(*substrs):
        for i, h in enumerate(header):
            if all(s in h for s in substrs):
                return i
        return None

    i_pow = col("power", "(w)")
    i_use = col("gpu use")
    i_total = col("vram total memory")  # excludes the "used" column (distinct text)
    i_used = col("vram total used memory")

    for row in rows[1:]:
        if not row or not row[0].strip().lower().startswith("card"):
            continue

        def fnum(i):
            if i is None or i >= len(row):
                return None
            try:
                return float(row[i].strip())
            except ValueError:
                return None

        power = fnum(i_pow)
        use = fnum(i_use)
        total_b = fnum(i_total)
        used_b = fnum(i_used)
        return (
            power,
            use,
            used_b / 1024.0 / 1024.0 if used_b is not None else None,
            total_b / 1024.0 / 1024.0 if total_b is not None else None,
        )
    return None


def _sample_rocm_smi(timeout: float = 8.0):
    """One rocm-smi sample -> (power_w, gpu_use_pct, vram_used_mb, vram_total_mb) | None."""
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showuse", "--showpower", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_rocm_smi_csv(proc.stdout or "")


def _sample_amd_smi(timeout: float = 8.0):
    """Best-effort amd-smi fallback sample (JSON). Returns ``None`` on any trouble."""
    import json as _json

    try:
        proc = subprocess.run(
            ["amd-smi", "metric", "-p", "-u", "-m", "--json"],
            capture_output=True, text=True, timeout=timeout,
        )
        data = _json.loads(proc.stdout or "")
    except Exception:  # noqa: BLE001 — fallback must never raise
        return None
    gpus = data if isinstance(data, list) else data.get("gpus", [data])
    if not gpus:
        return None
    g = gpus[0]

    def deep(d, *keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return None
        if isinstance(d, dict):
            d = d.get("value", d)
        return d

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    power = num(deep(g, "power", "socket_power")) or num(deep(g, "power", "average_socket_power"))
    use = num(deep(g, "usage", "gfx_activity")) or num(deep(g, "usage", "gfx"))
    used = num(deep(g, "mem_usage", "used_vram")) or num(deep(g, "memory", "used"))
    total = num(deep(g, "mem_usage", "total_vram")) or num(deep(g, "memory", "total"))
    if power is None and use is None:
        return None
    return (power, use, used, total)


class _GpuTelemetry:
    """Background iGPU telemetry sampler used as a context manager.

    See :func:`gpu_telemetry`. Samples on a daemon thread between ``__enter__``
    and ``__exit__``; on exit it always writes ``out_csv`` (header + any rows) and
    exposes the captured time-series as a DataFrame on ``.df``.
    """

    def __init__(self, out_csv, interval_s: float = 0.3, tool: str = "rocm-smi"):
        self.out_csv = Path(out_csv)
        self.interval_s = max(0.05, float(interval_s))
        self.tool = _resolve_telemetry_tool(tool)
        self._rows = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0: Optional[float] = None
        self.df = None

    def _sample(self):
        if self.tool == "rocm-smi":
            return _sample_rocm_smi()
        if self.tool == "amd-smi":
            return _sample_amd_smi()
        return None

    def _loop(self):
        while not self._stop.is_set():
            elapsed = time.monotonic() - self._t0
            sample = self._sample()
            if sample is not None:
                power, use, used_mb, total_mb = sample
                self._rows.append((round(elapsed, 3), power, use, used_mb, total_mb))
            self._stop.wait(self.interval_s)

    def __enter__(self):
        if self.tool is None:
            print("[telemetry] no rocm-smi / amd-smi on PATH — skipping live sampling; "
                  "use replay: lk.load_gpu_telemetry(<committed snapshot>)")
            return self
        print(f"[telemetry] sampling {self.tool} every {self.interval_s:.2f}s "
              f"-> {self.out_csv.name}")
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name="gpu-telemetry", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 5 + 2.0)
        self._finalise()
        return False  # never suppress the wrapped workload's exceptions

    def _finalise(self):
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(TELEMETRY_COLUMNS)
            for row in self._rows:
                writer.writerow(["" if v is None else v for v in row])
        try:
            self.df = load_gpu_telemetry(self.out_csv)
        except Exception:  # noqa: BLE001 — keep the df accessible even if pandas chokes
            self.df = None
        if self._rows:
            print(f"[telemetry] captured {len(self._rows)} samples over "
                  f"~{self._rows[-1][0]:.1f}s -> {self.out_csv}")
        else:
            print(f"[telemetry] no samples captured (tool unavailable or workload too "
                  f"short); wrote header-only {self.out_csv}")

    def dataframe(self):
        """Return the captured telemetry DataFrame (also on ``.df``)."""
        return self.df


def gpu_telemetry(out_csv, interval_s: float = 0.3, tool: str = "rocm-smi") -> _GpuTelemetry:
    """Context manager: sample iGPU telemetry while a live workload runs.

    Spawn a background thread that polls ``rocm-smi --showuse --showpower
    --showmeminfo vram --csv`` (``amd-smi`` JSON as a fallback) every
    ``interval_s`` seconds and records ``elapsed_s, power_w, gpu_use_pct,
    vram_used_mb, vram_total_mb`` (:data:`TELEMETRY_COLUMNS`) into ``out_csv``.
    Wrap a heavy proof/inference run::

        with lk.gpu_telemetry(out_csv) as tele:
            subprocess.run(["make", "demo-e-msm"], cwd=lk.REPO_ROOT, check=True)
        df = tele.dataframe()          # -> DataFrame, also tele.df

    On exit the CSV is always written (header + any samples) and ``.df`` holds the
    time-series. **Safe everywhere**: if neither ``rocm-smi`` nor ``amd-smi`` is on
    PATH (any laptop) the sampler degrades to a no-op — it does not raise, prints a
    steer toward replay, and writes a header-only CSV — so the notebook can fall
    back to :func:`load_gpu_telemetry` against a committed snapshot. HONESTY: on an
    APU ``power_w`` is the whole-SoC package draw (iGPU shares the budget with the
    CPU) and ``gpu_use_pct`` is a coarse activity metric.
    """
    return _GpuTelemetry(out_csv, interval_s=interval_s, tool=tool)


def load_gpu_telemetry(path):
    """Load a committed iGPU telemetry snapshot CSV as a ``DataFrame``.

    Columns: ``elapsed_s, power_w, gpu_use_pct, vram_used_mb, vram_total_mb``
    (:data:`TELEMETRY_COLUMNS`). ``path`` is required — pass one of the
    ``TELEMETRY_*`` constants (or iterate :data:`TELEMETRY_DIR`).
    """
    return _read_csv(Path(path), numeric=TELEMETRY_COLUMNS)


def plot_gpu_telemetry(df=None, label: Optional[str] = None):
    """Plot an iGPU telemetry time-series; return the ``Figure``.

    Two panels over ``elapsed_s``: (1) GPU use % (left axis) + whole-SoC package
    power W (right twin axis), with peak/mean annotated; (2) VRAM used (MB) with
    the total-VRAM ceiling line. Degrades gracefully on an empty/headless-only
    snapshot (draws a "no telemetry — replay a committed snapshot" placeholder).
    ``df`` is a frame from :func:`gpu_telemetry`/:func:`load_gpu_telemetry`;
    ``label`` names the captured workload in the title. HONESTY: ``power_w`` is the
    whole-SoC package draw on this APU and ``gpu_use_pct`` is a coarse activity
    metric — this shows the silicon is busy, not a clean per-kernel power split.
    """
    plt = _get_plt()
    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(14, 4.8))
    title = "AMD iGPU telemetry (rocm-smi)"
    if label:
        title += f" — {label}"

    if df is None or len(df) == 0 or "elapsed_s" not in getattr(df, "columns", []):
        for ax in (ax_u, ax_v):
            ax.text(0.5, 0.5, "no telemetry samples\n(replay a committed snapshot)",
                    ha="center", va="center", fontsize=10, color="#888")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig

    t = df["elapsed_s"].to_numpy()
    use = df["gpu_use_pct"].to_numpy() if "gpu_use_pct" in df.columns else None
    power = df["power_w"].to_numpy() if "power_w" in df.columns else None

    # Panel 1: GPU use % (left) + package power W (right twin).
    if use is not None:
        ax_u.plot(t, use, color="tab:green", lw=1.8, label="GPU use %")
        ax_u.fill_between(t, use, color="tab:green", alpha=0.12)
        peak_u = float(df["gpu_use_pct"].max())
        mean_u = float(df["gpu_use_pct"].mean())
        ax_u.axhline(peak_u, color="tab:green", ls=":", alpha=0.6)
        ax_u.annotate(f"peak {peak_u:.0f}% · mean {mean_u:.0f}%",
                      xy=(0.02, 0.93), xycoords="axes fraction", fontsize=9,
                      color="tab:green", va="top")
    ax_u.set_ylim(0, 105)
    ax_u.set_xlabel("elapsed (s)")
    ax_u.set_ylabel("GPU use (%)", color="tab:green")
    ax_u.tick_params(axis="y", labelcolor="tab:green")
    ax_u.grid(True, ls=":", alpha=0.4)

    if power is not None and df["power_w"].notna().any():
        axp = ax_u.twinx()
        axp.plot(t, power, color="tab:red", lw=1.5, alpha=0.85, label="package power (W)")
        peak_p = float(df["power_w"].max())
        mean_p = float(df["power_w"].mean())
        axp.set_ylabel("whole-SoC package power (W)", color="tab:red")
        axp.tick_params(axis="y", labelcolor="tab:red")
        axp.set_ylim(bottom=0)
        axp.annotate(f"peak {peak_p:.0f} W · mean {mean_p:.0f} W",
                     xy=(0.02, 0.82), xycoords="axes fraction", fontsize=9,
                     color="tab:red", va="top")
    ax_u.set_title("iGPU activity + package power over the run")

    # Panel 2: VRAM used (MB) + total ceiling.
    if "vram_used_mb" in df.columns and df["vram_used_mb"].notna().any():
        ax_v.plot(t, df["vram_used_mb"].to_numpy(), color="tab:blue", lw=1.8,
                  label="VRAM used (MB)")
        ax_v.fill_between(t, df["vram_used_mb"].to_numpy(), color="tab:blue", alpha=0.12)
        if "vram_total_mb" in df.columns and df["vram_total_mb"].notna().any():
            total_mb = float(df["vram_total_mb"].max())
            ax_v.axhline(total_mb, color="#888", ls="--", alpha=0.7)
            ax_v.text(t[0] if len(t) else 0, total_mb,
                      f" VRAM pool {total_mb / 1024.0:.0f} GB (unified LPDDR5X)",
                      fontsize=8, color="#555", va="bottom")
            ax_v.set_ylim(0, total_mb * 1.12)
        peak_v = float(df["vram_used_mb"].max())
        ax_v.annotate(f"peak {peak_v:.0f} MB", xy=(0.02, 0.93),
                      xycoords="axes fraction", fontsize=9, color="tab:blue", va="top")
        ax_v.legend(fontsize=8, loc="upper right")
    else:
        ax_v.text(0.5, 0.5, "no VRAM samples", ha="center", va="center",
                  fontsize=10, color="#888")
    ax_v.set_xlabel("elapsed (s)")
    ax_v.set_ylabel("VRAM used (MB)")
    ax_v.grid(True, ls=":", alpha=0.4)
    ax_v.set_title("VRAM footprint over the run")

    fig.suptitle(
        f"{title}\nthe iGPU is genuinely executing this workload "
        "(APU: power_w is the whole-SoC package draw; gpu_use% is coarse activity)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ---------------------------------------------------------------------------
# scorecard_table() — one dynamic, drift-free cross-engine AMD summary (Phase 2).
# ---------------------------------------------------------------------------
# Every number is pulled live from the committed loaders above (never hard-coded),
# so the scorecard cannot drift from the artefacts the rest of the lab plots. Each
# row is computed defensively: a missing/blank loader marks that row "n/a (…)"
# rather than raising, so the table always renders.
def _rel_to_repo(path) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except (ValueError, TypeError):
        return str(path)


def _fmt_x(value: float) -> str:
    return f"{value:.2f}x"


def plot_hip_primitives(save: Optional[Path] = None):
    """Plot the native-HIP (Part 1) SNARK-primitive artefacts in a 2×2 glance.

    Draws, each panel degrading gracefully (a missing/absent source renders an
    explanatory note instead of raising, so ``make lab-replay`` never errors):

    * **MSM min-wall vs size** — native-HIP direct G1 (from :func:`load_hip_msm`
      if the live artefact exists, else the 128·CU baseline in
      :func:`load_hip_msm_rocprim`), the OpenCL G1 path, and native-HIP G2 —
      log-y ms.
    * **NTT copy vs APU zero-copy** — :func:`load_hip_ntt` copy/zero-copy ms with
      the (honest-negative) Δ% annotated.
    * **work_units 2c sweep** — native-HIP direct-MSM tuned-512·CU ÷ OpenCL and ÷
      128·CU-baseline ratios per size (:func:`load_hip_msm_tune`), with a 1.0×
      break-even line (the 2.0–2.2× vs-OpenCL win is native-port-vs-OpenCL, not
      vs-CPU).
    * **APU zero-copy headroom** — the H2D-copy fraction of the OpenCL MSM wall
      (:func:`load_zerocopy_headroom`), i.e. the small ceiling zero-copy could
      reclaim on an MSM whose wall is kernel execution, not transfer.

    Returns the ``Figure``. All source numbers are bit-for-bit gated + solo-guarded
    (nothing fabricated); ratios are vs the OpenCL path on the same iGPU.
    """
    plt = _get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (axA, axB), (axC, axD) = axes

    def _note(ax, msg):
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9,
                color="#888", transform=ax.transAxes, wrap=True)

    # --- Panel A: MSM min-wall vs size (log-y ms) ---------------------------
    try:
        plotted = False
        # direct G1: prefer the live tuned standalone artefact, else the committed
        # 128·CU baseline from the rocPRIM 3-way file.
        try:
            m = load_hip_msm()
            b = m[m["min_ms"].notna()].sort_values("log_size")
            if not b.empty:
                axA.plot(b["log_size"], b["min_ms"], "o-", color="tab:red",
                         label="direct G1 (HIP, tuned 512·CU)")
                plotted = True
        except Exception:  # noqa: BLE001
            pass
        try:
            rp = load_hip_msm_rocprim().sort_values("log_size")
            if not plotted:
                axA.plot(rp["log_size"], rp["direct_hip_ms"], "o-", color="tab:red",
                         label="direct G1 (HIP, 128·CU baseline)")
                plotted = True
            axA.plot(rp["log_size"], rp["opencl_ms"], "s--", color="tab:gray",
                     label="G1 (OpenCL ec-gpu)")
        except Exception:  # noqa: BLE001
            pass
        try:
            g2 = load_hip_msm_g2()
            g2b = g2[g2["min_ms"].notna()].sort_values("log_size")
            if not g2b.empty:
                axA.plot(g2b["log_size"], g2b["min_ms"], "^-", color="tab:blue",
                         label="G2/Fq2 (HIP)")
                plotted = True
        except Exception:  # noqa: BLE001
            pass
        if plotted:
            axA.set_yscale("log")
            axA.set_xlabel("log₂ n")
            axA.set_ylabel("min wall (ms, log)")
            axA.set_title("Native-HIP MSM min-wall (bit-for-bit == arkworks)")
            axA.grid(True, ls=":", alpha=0.4)
            axA.legend(fontsize=8)
        else:
            _note(axA, "MSM timing artefacts not present\n(run `make demo-e-hip-msm[-g2]` on the Halo)")
    except Exception as exc:  # noqa: BLE001
        _note(axA, f"MSM panel unavailable\n({exc})")

    # --- Panel B: NTT copy vs APU zero-copy ---------------------------------
    try:
        ntt = load_hip_ntt().sort_values("log_size")
        x = ntt["log_size"].tolist()
        axB.plot(x, ntt["copy_ms"], "o-", color="tab:green", label="copy (hipMemcpy)")
        axB.plot(x, ntt["zerocopy_ms"], "s--", color="tab:purple",
                 label="zero-copy (mapped LPDDR5X)")
        axB.set_yscale("log")
        for _, r in ntt.iterrows():
            axB.annotate(f"Δ{r['zerocopy_delta_pct']:+.0f}%",
                         (r["log_size"], max(r["copy_ms"], r["zerocopy_ms"])),
                         textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=7, color="#a00")
        axB.set_xlabel("log₂ n")
        axB.set_ylabel("forward NTT (ms, log)")
        axB.set_title("Native-HIP NTT — copy vs APU zero-copy\n(Δ<0 ⇒ honest negative: mapped path not amortized ≤2²⁰)")
        axB.grid(True, ls=":", alpha=0.4)
        axB.legend(fontsize=8)
    except Exception as exc:  # noqa: BLE001
        _note(axB, f"NTT copy/zero-copy artefact unavailable\n({exc})")

    # --- Panel C: work_units 2c tuning ratios (native-HIP G1 vs OpenCL / baseline) ---
    try:
        t = load_hip_msm_tune()
        g1 = t[t["primitive"] == "msm_g1_hip"].sort_values("log2n")
        if g1.empty:
            raise ValueError("no msm_g1_hip summary rows")
        import numpy as _np
        xs = _np.arange(len(g1))
        w = 0.38
        axC.bar(xs - w / 2, g1["tuned_vs_opencl"], w, color="tab:red",
                label="512·CU ÷ OpenCL")
        axC.bar(xs + w / 2, g1["tuned_vs_baseline"], w, color="tab:orange",
                label="512·CU ÷ 128·CU baseline")
        axC.axhline(1.0, color="gray", ls="--", alpha=0.8)
        for i, r in enumerate(g1.itertuples()):
            axC.text(i - w / 2, r.tuned_vs_opencl + 0.03, f"{r.tuned_vs_opencl:.2f}×",
                     ha="center", fontsize=7)
            axC.text(i + w / 2, r.tuned_vs_baseline + 0.03, f"{r.tuned_vs_baseline:.2f}×",
                     ha="center", fontsize=7)
        axC.set_xticks(xs)
        axC.set_xticklabels([f"2^{int(v)}" for v in g1["log2n"]])
        axC.set_ylabel("speedup ratio (×)")
        axC.set_title("Direct G1 MSM — 512·CU occupancy tuning [2c]\n(vs OpenCL ec-gpu on the SAME iGPU — not vs CPU)")
        axC.grid(True, axis="y", ls=":", alpha=0.4)
        axC.legend(fontsize=8)
    except Exception as exc:  # noqa: BLE001
        _note(axC, f"work_units tune summary unavailable\n({exc})")

    # --- Panel D: APU zero-copy headroom (H2D fraction of the MSM wall) ------
    try:
        z = load_zerocopy_headroom().sort_values("log_size")
        pct = z["h2d_fraction"] * 100.0
        axD.bar([f"2^{int(v)}" for v in z["log_size"]], pct, color="tab:cyan")
        for i, v in enumerate(pct):
            axD.text(i, v + 0.05, f"{v:.1f}%", ha="center", fontsize=8)
        axD.set_ylabel("H2D copy — % of MSM wall")
        axD.set_ylim(0, max(4.0, float(pct.max()) * 1.3))
        axD.set_title("APU zero-copy headroom on MSM\n(H2D is a tiny fraction ⇒ small zero-copy ceiling)")
        axD.grid(True, axis="y", ls=":", alpha=0.4)
    except Exception as exc:  # noqa: BLE001
        _note(axD, f"zero-copy headroom artefact unavailable\n({exc})")

    fig.suptitle("Native-HIP BN254 SNARK primitives on the Radeon 8060S iGPU (gfx1151) — "
                 "Part 1 (bit-for-bit == arkworks, solo-guarded)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save is not None:
        fig.savefig(str(save), dpi=120, bbox_inches="tight")
    return fig


def scorecard_table():
    """Assemble the cross-engine AMD scorecard as a ``DataFrame`` — dynamically.

    Returns one row per (engine, workload) covering every track in the lab, with
    columns ``engine, workload, metric, measured, evidence_file``. Each ``measured``
    cell is computed **from the committed loaders** (:func:`load_throughput`,
    :func:`load_gpu_primitive`, :func:`load_gpu_groth16`, :func:`load_ai_inference`,
    :func:`load_demo_c_gpu`, :func:`load_zkrag_msm`, :func:`load_zkllm_msm`,
    :func:`summary_from_full_run_info`, …) at call time, so the numbers can never
    drift from the artefacts the other notebooks plot. Loaders with missing data
    yield a graceful ``"n/a (…)"`` row instead of raising. The four engines are
    represented (Zen5 CPU, Radeon iGPU OpenCL / MIGraphX, XDNA2 NPU) plus the
    unified-memory enabler. HONESTY is preserved per row (CPU-only proving vs
    size-gated iGPU wins vs AI-model-only acceleration vs NPU dispatch-evidence).
    """
    pd = _require_pandas()
    rows = []

    # perf-per-watt: most rows have no isolated energy capture, so the column
    # defaults to a clear "n/a". Rows whose workload has committed energy data
    # (ai-inference.csv embeddings_per_joule + the whole-SoC rocm-smi telemetry
    # snapshots) fill it live below — never a hard-coded number. HONESTY: on this
    # APU rocm-smi power_w is the WHOLE-SoC package draw (iGPU shares the LPDDR5X
    # budget with the Zen 5 cores), so package energy is a GPU-side efficiency view,
    # not a clean GPU-vs-CPU energy split (the CPU side is not separately telemetered).
    _PW_NA = "n/a (no energy capture)"

    def _pkg_energy(telemetry_path):
        """'<kJ> kJ @ <W> W avg (whole-SoC pkg, <s>s)' from a committed telemetry
        snapshot, or None if the snapshot is unavailable."""
        try:
            e = telemetry_energy(telemetry_path)
            if e.get("samples", 0) >= 2 and e.get("joules", 0) > 0:
                return (f"{e['joules'] / 1000.0:.1f} kJ @ {e['avg_power_w']:.0f} W avg "
                        f"(whole-SoC pkg, {e['duration_s']:.0f}s)")
        except Exception:  # noqa: BLE001 - a missing snapshot just yields n/a
            pass
        return None

    def add(engine, workload, metric, measured, evidence, perf_watt=_PW_NA,
            bar_speedup=None):
        # bar_speedup: the ONE figure plot_scorecard should draw for this row.
        # Default None keeps the historical behaviour (scrape the largest Nx out
        # of `measured`). Set it explicitly whenever the cell mentions more than
        # one arm — notably the two Demo C folding rows, whose cells now name a
        # current paired figure AND the superseded figure it replaces AND an
        # un-re-measured wide floor. Picking by max() there is correct only by
        # arithmetic accident; naming the arm makes the bar auditable.
        rows.append({
            "engine": engine, "workload": workload, "metric": metric,
            "measured": measured, "perf_per_watt": perf_watt,
            "evidence_file": evidence, "bar_speedup": bar_speedup,
        })

    # 1. Zen5 CPU — RISC0 STARK Rayon thread-scaling (speedup 64t vs 1t).
    try:
        tp = load_throughput()
        ok = tp[(tp["wall_seconds"] > 0) & (tp["proof_bytes"] > 0)]
        speedups = []
        for mc in sorted(ok["max_mcycle"].dropna().unique()):
            grp = ok[ok["max_mcycle"] == mc].sort_values("rayon_threads")
            if len(grp) >= 2:
                base = float(grp["wall_seconds"].iloc[0])
                best = float(grp["wall_seconds"].min())
                if best > 0:
                    speedups.append(base / best)
        if speedups:
            add("Zen5 CPU (16c/32t)", "RISC0 r0vm STARK — Rayon thread-scaling",
                "speedup (best threads vs 1)",
                f"{min(speedups):.1f}x–{max(speedups):.1f}x (CPU-only proving)",
                _rel_to_repo(THROUGHPUT_CSV))
        else:
            raise ValueError("no scalable groups")
    except Exception:  # noqa: BLE001
        add("Zen5 CPU (16c/32t)", "RISC0 r0vm STARK — Rayon thread-scaling",
            "speedup (best threads vs 1)", "n/a (throughput.csv unavailable)",
            _rel_to_repo(THROUGHPUT_CSV))

    # 2. Zen5 CPU + 94 GB unified — Demo B real full STARK prove wall time.
    try:
        summ = summary_from_full_run_info()
        tdf = summ.get("targets")
        row = tdf[tdf["target"] == "demo-b-full"]
        wall = float(row["wall_seconds"].iloc[0])
        status = str(row["status"].iloc[0])
        add("Zen5 CPU + 94 GB unified", "Demo B real --full STARK prove (Cartesi step)",
            "prove wall (s)", f"{wall:.0f} s, {status} — CPU-only stock "
            "(scoped v2.3.2 fork: iGPU rv32im segment-STARK+eval_check, hybrid/verified — path-i)",
            _rel_to_repo(FULL_RUN_INFO))
    except Exception:  # noqa: BLE001
        add("Zen5 CPU + 94 GB unified", "Demo B real --full STARK prove (Cartesi step)",
            "prove wall (s)", "n/a (full-run.info unavailable)",
            _rel_to_repo(FULL_RUN_INFO))

    # 3. Radeon iGPU OpenCL — Path E NTT, i.e. the BLS12-381 FFT vs blstrs curve.
    #    NO BOUND IS CLAIMED HERE. The repo has no NTT roofline / arithmetic-intensity
    #    measurement and no memory-side PMC for it: the one capture,
    #    rocprof-ntt.csv, collected only VALU/wave/busy counters, so the VALU:LDS
    #    ratio that licenses the MSM row's arithmetic-bound reading cannot be formed,
    #    and rocprof-compute/omniperf refuses gfx1151 ("Cannot find a supported
    #    arch"). So this row reports the measured PHENOMENON — above parity across
    #    the sweep, with its peak — and names the CURVE, because "NTT wins" is a
    #    property of this curve and not of NTT: the same iGPU's BN254 Fr NTT vs
    #    arkworks LOSES at 2^18 (read live from MSM_NTT_BACKEND_CSV below).
    #    Wording tracks workshop/futuremode-2026/slides.md S4/S16.
    _ntt_label = "Path E NTT — BLS12-381 FFT vs blstrs parallel_fft"
    try:
        prim = load_gpu_primitive()
        fft = prim[(prim["primitive"] == "fft") & prim["speedup"].notna()]
        peak = float(fft["speedup"].max())
        lr = int(fft.loc[fft["speedup"].idxmax(), "log_size"])
        # The honest counterpoint, read from the same-binary backend A/B rather
        # than asserted, so it can never drift from the artefact.
        counter = ""
        try:
            _bk = pd.read_csv(MSM_NTT_BACKEND_CSV, comment="#")
            _oc = _bk[(_bk["primitive"] == "ntt") & (_bk["backend"] == "opencl")]
            _lose = _oc[pd.to_numeric(_oc["speedup_vs_cpu"], errors="coerce") < 1.0]
            if len(_lose):
                _w = _lose.sort_values("speedup_vs_cpu").iloc[0]
                # 3 decimals: 0.963x is the figure the deck and the ledger quote.
                counter = (f"; same iGPU's BN254 Fr NTT vs arkworks loses at "
                           f"2^{int(_w['log_size'])} ({float(_w['speedup_vs_cpu']):.3f}x)")
        except Exception:  # noqa: BLE001 - counterpoint is additive, never fatal
            pass
        add("Radeon iGPU (OpenCL / ec-gpu)", _ntt_label,
            "speedup vs 32t CPU",
            f"above parity across the sweep, peak {_fmt_x(peak)} @ 2^{lr} "
            f"(no bound claimed — no NTT roofline/memory-side PMC in repo{counter})",
            _rel_to_repo(GPU_PRIMITIVE_CSV))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (OpenCL / ec-gpu)", _ntt_label,
            "speedup vs 32t CPU", "n/a (gpu-primitive.csv unavailable)",
            _rel_to_repo(GPU_PRIMITIVE_CSV))

    # 4. Radeon iGPU OpenCL — Path E MSM (G1 multiexp), size-gated (ec-gpu kernel).
    try:
        prim = load_gpu_primitive()
        msm = prim[(prim["primitive"] == "msm") & prim["speedup"].notna()].sort_values("log_size")
        top = msm.iloc[-1]
        _e = _pkg_energy(TELEMETRY_DEMO_E_MSM)
        add("Radeon iGPU (OpenCL / ec-gpu)", "Path E MSM (G1 multi-scalar mult.)",
            "speedup vs 32t CPU",
            f"{_fmt_x(float(top['speedup']))} @ 2^{int(top['log_size'])} "
            f"(size-gated crossover — ec-gpu OpenCL path)",
            _rel_to_repo(GPU_PRIMITIVE_CSV),
            perf_watt=(f"{_e} — size-gated, so efficiency win only past the crossover"
                       if _e else _PW_NA))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (OpenCL / ec-gpu)", "Path E MSM (G1 multi-scalar mult.)",
            "speedup vs 32t CPU", "n/a (gpu-primitive.csv unavailable)",
            _rel_to_repo(GPU_PRIMITIVE_CSV))

    # 4b–4e. Radeon iGPU NATIVE HIP primitives (Part 1) — the ec-gpu OpenCL kernels
    #     ported to native HIP (hip/*.hip), bit-for-bit == arkworks + solo-guarded.
    #     These are native-port-vs-generated-OpenCL comparisons on the SAME iGPU, so
    #     they are rendered with a unicode "×" (NOT ASCII "x") on purpose: plot_scorecard
    #     draws only vs-CPU "Nx" bars, and these are NOT vs-CPU claims (the size-gated
    #     vs-CPU crossover is carried by the OpenCL rows above; the vs-OpenCL detail is
    #     drawn as bars by plot_hip_primitives). The durable claim is correctness.
    #
    # 4b. Native-HIP direct G1 MSM after the 512·CU occupancy tuning (~2.0–2.2× vs OpenCL).
    try:
        t = load_hip_msm_tune()
        g1 = t[t["primitive"] == "msm_g1_hip"]
        if g1.empty:
            raise ValueError("no msm_g1_hip summary rows")
        tvo_lo, tvo_hi = float(g1["tuned_vs_opencl"].min()), float(g1["tuned_vs_opencl"].max())
        tvb_lo, tvb_hi = float(g1["tuned_vs_baseline"].min()), float(g1["tuned_vs_baseline"].max())
        lo_p, hi_p = int(g1["log2n"].min()), int(g1["log2n"].max())
        add("Radeon iGPU (native HIP / ec-gpu port)",
            "native-HIP BN254 G1 direct MSM (tuned 512·CU)",
            "speedup vs OpenCL ec-gpu (same iGPU)",
            f"{tvo_lo:.1f}×–{tvo_hi:.1f}× vs OpenCL ec-gpu ({tvb_lo:.1f}×–{tvb_hi:.1f}× vs 128·CU "
            f"baseline) @ 2^{lo_p}–2^{hi_p} · bit-for-bit == arkworks",
            _rel_to_repo(HIP_MSM_TUNE_CSV))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (native HIP / ec-gpu port)",
            "native-HIP BN254 G1 direct MSM (tuned 512·CU)",
            "speedup vs OpenCL ec-gpu (same iGPU)", "n/a (hip-msm-tune unavailable)",
            _rel_to_repo(HIP_MSM_TUNE_CSV))

    # 4c. Native-HIP G2 (Fq2) MSM vs OpenCL G2 (which itself loses to CPU — honest negative).
    try:
        t = load_hip_msm_tune()
        g2 = t[t["primitive"] == "msm_g2_hip"]
        if g2.empty:
            raise ValueError("no msm_g2_hip summary rows")
        lo, hi = float(g2["tuned_vs_opencl"].min()), float(g2["tuned_vs_opencl"].max())
        gg = load_hip_msm_g2()
        verified = gg["verify"].astype(str).str.strip().str.upper().eq("OK").all()
        add("Radeon iGPU (native HIP / ec-gpu port)",
            "native-HIP BN254 G2 (Fq2) MSM (tuned 512·CU)",
            "speedup vs OpenCL G2 (same iGPU)",
            f"{lo:.1f}×–{hi:.1f}× vs OpenCL G2 (OpenCL-G2 is itself 0.79–0.96× vs 32t CPU — "
            f"honest negative){' · bit-for-bit == arkworks' if verified else ''}",
            _rel_to_repo(HIP_MSM_G2_CSV))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (native HIP / ec-gpu port)",
            "native-HIP BN254 G2 (Fq2) MSM (tuned 512·CU)",
            "speedup vs OpenCL G2 (same iGPU)", "n/a (hip-msm-g2 / tune unavailable)",
            _rel_to_repo(HIP_MSM_G2_CSV))

    # 4d. Native-HIP NTT — APU zero-copy Δ (an honest negative at these sizes).
    try:
        ntt = load_hip_ntt()
        dmin = float(ntt["zerocopy_delta_pct"].min())
        dmax = float(ntt["zerocopy_delta_pct"].max())
        verified = ntt["verify"].astype(str).str.strip().str.lower().eq("ok").all()
        add("Radeon iGPU (native HIP / ec-gpu port)",
            "native-HIP BN254 Fr NTT — APU zero-copy Δ",
            "zero-copy Δ (copy vs mapped LPDDR5X)",
            f"Δ {dmin:+.0f}%…{dmax:+.0f}% (copy path wins ≤2^20 — honest negative: the mapped-buffer "
            f"setup is not amortized at these sizes){' · bit-for-bit == arkworks' if verified else ''}",
            _rel_to_repo(HIP_NTT_CSV))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (native HIP / ec-gpu port)",
            "native-HIP BN254 Fr NTT — APU zero-copy Δ",
            "zero-copy Δ (copy vs mapped LPDDR5X)", "n/a (hip-ntt.csv unavailable)",
            _rel_to_repo(HIP_NTT_CSV))

    # 4e. APU zero-copy headroom — H2D-copy fraction of the OpenCL MSM wall (a small ceiling).
    try:
        z = load_zerocopy_headroom()
        fmin = float(z["h2d_fraction"].min()) * 100.0
        fmax = float(z["h2d_fraction"].max()) * 100.0
        add("Radeon iGPU (OpenCL / ec-gpu)",
            "APU zero-copy headroom (MSM H2D fraction)",
            "H2D copy share of the MSM wall",
            f"{fmin:.1f}%–{fmax:.1f}% of the MSM wall is H2D input copy — the zero-copy ceiling "
            f"(the MSM wall is kernel execution, not transfer)",
            _rel_to_repo(ZEROCOPY_HEADROOM_CSV))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (OpenCL / ec-gpu)",
            "APU zero-copy headroom (MSM H2D fraction)",
            "H2D copy share of the MSM wall", "n/a (zerocopy-headroom.csv unavailable)",
            _rel_to_repo(ZEROCOPY_HEADROOM_CSV))

    # 5. Radeon iGPU OpenCL — Path E full Groth16 prove (MSM-dominated).
    try:
        g16 = load_gpu_groth16()
        ok = g16[g16["speedup"].notna()].sort_values("constraints_pow")
        top = ok.iloc[-1]
        verify = str(top.get("verify_ok", "")).strip().lower() in ("true", "1", "yes")
        add("Radeon iGPU (OpenCL / bellperson)", "Path E full Groth16 prove",
            "speedup vs 32t CPU",
            f"{_fmt_x(float(top['speedup']))} @ 2^{int(top['constraints_pow'])} "
            f"(parity){' · verify ✓' if verify else ''}",
            _rel_to_repo(GPU_GROTH16_CSV))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (OpenCL / bellperson)", "Path E full Groth16 prove",
            "speedup vs 32t CPU", "n/a (gpu-groth16.csv unavailable)",
            _rel_to_repo(GPU_GROTH16_CSV))

    # 6. Radeon iGPU MIGraphX — Path F MiniLM forward (the AI MODEL, not the proof).
    try:
        ai = load_ai_inference()
        cpu = ai[ai["backend"] == "cpu"].set_index(["batch", "seq_len"])["fwd_ms"]
        gpu = ai[ai["backend"] == "rocm"].set_index(["batch", "seq_len"])["fwd_ms"]
        keys = sorted(set(cpu.index) & set(gpu.index))
        sp = [float(cpu[k]) / float(gpu[k]) for k in keys if float(gpu[k]) > 0]
        # perf/watt: embeddings_per_joule (fp32 rocm) peak vs the CPU peak — the iGPU
        # is BOTH faster AND more energy-efficient on the AI model. The 14.41 emb/J
        # peak is the value 05-roofline.md cites (drift-checked).
        epj = pd.to_numeric(ai.get("embeddings_per_joule"), errors="coerce")
        g_epj = float(epj[ai["backend"] == "rocm"].max())
        c_epj = float(epj[ai["backend"] == "cpu"].max())
        pw_cell = (f"{g_epj:.2f} emb/J iGPU vs {c_epj:.2f} CPU "
                   f"({g_epj / c_epj:.1f}× more energy-efficient on the AI model)")
        add("Radeon iGPU (MIGraphX / ROCm)", "Path F full MiniLM forward (embedding)",
            "speedup vs 32t CPU",
            f"{min(sp):.1f}x–{max(sp):.1f}x (compute-bound GEMM — accelerates the AI model)",
            _rel_to_repo(AI_INFER_CSV),
            perf_watt=pw_cell)
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (MIGraphX / ROCm)", "Path F full MiniLM forward (embedding)",
            "speedup vs 32t CPU", "n/a (ai-inference.csv unavailable)",
            _rel_to_repo(AI_INFER_CSV))

    # 7. Radeon iGPU OpenCL — folding DeciderEth Groth16 (Demo C), native + on-chain.
    #
    #     The G1-only arm's published 0.70× is SUPERSEDED and must not be quoted
    #     as the current figure: that GPU row was n=1, solo=false (preempted
    #     mid-flight by a third-party train job) and un-interleaved, and it was
    #     published at the time as a *lower bound*. The paired re-bench
    #     (DEMO_C_PAIRED_CHECKOFF_OPENCL_CSV: one session, one binary, both arms
    #     FOLD_GPU_MSM_CHECK=0, arms interleaved, 3 reps each, median, solo
    #     re-verified before every run) fixes all three: CPU 60.847 s vs
    #     OpenCL-GPU 61.206 s = 0.994×. The published figure was pessimistic by
    #     1.42×. Cross-check: demo-c-phase-split measured the four offloadable G1
    #     MSMs at 7.730 s of a 60.625 s D::prove = 12.75% ⇒ ceiling 1.146×, and
    #     the measured 0.994× sits below it, so the two agree.
    #
    #     🔴 WORDING GATE — the only defensible word is PARITY. The arm-to-arm
    #     gap is 0.59% while the WITHIN-arm spreads are 0.93% (cpu) and 1.21%
    #     (gpu), so the gap is smaller than the noise: at n=3 the two arms are
    #     indistinguishable. NEVER "the iGPU wins" and NEVER "0.6% slower".
    #     Mean-based is 0.990×, same verdict. And parity is NOT acceleration —
    #     the blanket "we do not claim GPU-accelerated proofs" line is untouched;
    #     what survives is the correctness result (GPU proof bit-for-bit == CPU,
    #     native + on-chain VERIFIED), which is independent of the speed sign.
    #
    #     SCOPE: ONLY the OpenCL G1-only arm was re-measured. gpu-wide
    #     (G1+G2+FFT) was NOT re-measured, keeps the same n=1 + contention
    #     defects, and its 0.74× stays a FLOOR — the G1 correction must not be
    #     extrapolated to it. Four arms, four states, never averaged and never
    #     blended: OpenCL G1-only 0.994× parity (paired), native-HIP G1-only
    #     1.048× small win (paired, row 7b), gpu-wide 0.74× floor, hip-wide
    #     0.77× floor.
    #     Wording tracks workshop/futuremode-2026/slides.md S2 and
    #     docs/INTEGRITY-REPORT.md §TL;DR (2026-08-29).
    try:
        dc = load_demo_c_gpu()
        sp_col = "speedup_vs_cpu" if "speedup_vs_cpu" in dc.columns else "speedup_vs_cpu_median"
        by_mode = {}
        for _, r in dc.iterrows():
            m = str(r["mode"])
            v = pd.to_numeric(pd.Series([r.get(sp_col)]), errors="coerce").iloc[0]
            if v == v and v > 0:  # finite, positive
                by_mode[m] = float(v)
        wide = by_mode.get("gpu-wide")

        # The G1-only arm comes from the PAIRED OpenCL file, computed from its
        # two medians rather than read from a note, so it cannot drift. 3
        # decimals because 0.994x is the figure the deck and the ledger quote.
        g1 = demo_c_opencl_g1_paired_speedup()

        if g1 is not None or wide is not None:
            parts = []
            if g1 is not None:
                parts.append(f"{g1:.3f}x (G1 MSM — PARITY, paired same-session, "
                             f"both arms check-off, interleaved, n=3 each; "
                             f"supersedes 0.70x)")
            if wide is not None:
                parts.append(f"{_fmt_x(wide)} (G1+G2+FFT — NOT re-measured, a floor)")
            _e = _pkg_energy(TELEMETRY_DEMO_C_FOLD)
            # The bar plot_scorecard draws from this row is the G1-only paired
            # figure, so the workload must name the arm: an unqualified label
            # would read as a claim about the (unchanged) wide arm too.
            add("Radeon iGPU (OpenCL / ark-groth16)",
                "folding DeciderEth Groth16 (Demo C, FOLD_N=2) "
                "— OpenCL (bar = G1-only paired)",
                "speedup vs CPU prove",
                " → ".join(parts)
                + " · native + on-chain ✓ · SCOPE: G1-only arm re-measured; "
                  "gpu-wide and the native-HIP rows are separate arms",
                (f"{_rel_to_repo(DEMO_C_PAIRED_CHECKOFF_OPENCL_CSV)} (G1-only) + "
                 f"{_rel_to_repo(DEMO_C_GPU_CSV)} (gpu-wide)"),
                perf_watt=(f"{_e} — parity at 2^24 ⇒ NOT an efficiency win"
                           if _e else _PW_NA),
                bar_speedup=g1)
        else:
            raise ValueError("no gpu speedup rows")
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (OpenCL / ark-groth16)",
            "folding DeciderEth Groth16 (Demo C, FOLD_N=2)",
            "speedup vs CPU prove", "n/a (demo-c-gpu.csv unavailable)",
            _rel_to_repo(DEMO_C_GPU_CSV))

    # 7b. Radeon iGPU NATIVE HIP — the SAME Demo C DeciderEth prove, but the
    #     offloaded G1/G2 MSM + QAP FFT run through the native HIP kernels
    #     (FOLD_GPU_BACKEND=hip).
    #
    #     The G1-only arm's published 0.86× is SUPERSEDED and must not be quoted:
    #     run-gpu-fold-hip.sh:74 defaults FOLD_GPU_MSM_CHECK=1, which recomputes
    #     every offloaded MSM on the CPU *inside the timed region* (measured
    #     10.878 s = 18.5%) while run_cpu() never pays it, and CPU_RERUN defaults
    #     to 0 so the denominator was a two-month-old median from a different
    #     build. Both defects pushed the ratio the same way. The paired re-bench
    #     (DEMO_C_PAIRED_CHECKOFF_CSV: one session, one binary, both arms
    #     check-off, arms interleaved, 3 reps, median) flips the sign: CPU
    #     61.602 s vs native-HIP G1-only 58.762 s = 1.048×, a small WIN. Its
    #     same-session check-ON control reads 69.640 s ⇒ 61.602/69.640 = 0.885×,
    #     reproducing the published figure, so the discrepancy is the harness and
    #     not drift. Cross-check: msm_g1 median 6336.684 ms of a 58.762 s
    #     D::prove = 10.8% offloadable ⇒ ceiling 1.12×, and the measured 1.048×
    #     sits below it, so the two independent measurements agree.
    #
    #     SCOPE (the CSV header states it too, and it is carried into the cell):
    #     ONLY the native-HIP G1-only arm was re-measured here. hip-wide is NOT
    #     re-measured and stays at its published 0.77×, a floor.
    #
    #     🔴 This row is NOT the deck's S2 headline. The OpenCL arms (row 7a, a
    #     different script that already defaults the check off) got their own
    #     paired re-bench on the same day and landed ELSEWHERE: OpenCL G1-only
    #     is 0.994× — PARITY, not a win. Four arms, four states, never averaged
    #     and never blended: OpenCL G1-only 0.994× parity (paired), native-HIP
    #     G1-only 1.048× small win (paired, this row), gpu-wide 0.74× floor (not
    #     re-measured), hip-wide 0.77× floor (not re-measured).
    #
    #     NOT "a CPU-Nova-dominated prove": Nova's prove_steps are timed
    #     separately and total ~0.6 s, and are not inside this wall at all; the
    #     ~87% the seam cannot reach is the Groth16 prover's own CPU work (QAP
    #     FFT, G2 MSM, synthesis) — see DEMO_C_PHASE_SPLIT_CSV. Unchanged either
    #     way: the correctness/plumbing deliverable (GPU==CPU bit-for-bit + native
    #     + on-chain) on a fully native (no OpenCL) path.
    #     Wording tracks workshop/futuremode-2026/slides.md S2/S6b/S16 notes and
    #     presenter-kit.md "Paired re-bench of our own harness".
    try:
        dch = pd.read_csv(DEMO_C_GPU_HIP_CSV)
        sp = pd.to_numeric(dch["speedup_vs_cpu_median"], errors="coerce")
        by = {str(m): float(v) for m, v in zip(dch["mode"], sp) if v == v and v > 0}
        wideh = by.get("hip-wide")

        # The G1-only arm comes from the PAIRED file, computed from its two
        # medians rather than read from a note, so it cannot drift. 3 decimals
        # because 1.048x is the figure the deck and the ledger quote.
        g1h = None
        try:
            pcs = pd.read_csv(DEMO_C_PAIRED_CHECKOFF_CSV, comment="#")
            _med = {str(a): r for a, r in zip(pcs["arm"], pcs.to_dict("records"))}
            _c = float(_med["cpu-median"]["decider_prove_s"])
            _g = float(_med["hip-gpu-median"]["decider_prove_s"])
            if _c > 0 and _g > 0:
                g1h = _c / _g
        except Exception:  # noqa: BLE001 - fall through to the n/a row below
            pass

        if g1h is not None or wideh is not None:
            parts = []
            if g1h is not None:
                parts.append(f"{g1h:.3f}x (G1 MSM — paired same-session, "
                             f"both arms check-off; supersedes 0.86x)")
            if wideh is not None:
                parts.append(f"{_fmt_x(wideh)} (G1+G2+FFT — NOT re-measured)")
            solo_clean = (
                dch["solo"].astype(str).str.strip().str.lower()
                .isin(["true", "1", "yes"]).all()
            )
            # The workload names the arm: plot_scorecard labels its bar from this
            # column alone, and the max-Nx it draws is the G1-only paired figure,
            # so an unqualified label would read as "the folding slowdown was
            # retracted" next to the (unchanged) OpenCL row.
            add("Radeon iGPU (native HIP / ark-groth16)",
                "folding DeciderEth Groth16 (Demo C, FOLD_N=2) "
                "— native HIP (bar = G1-only paired)",
                "speedup vs CPU prove",
                " → ".join(parts)
                + (" · solo" if solo_clean else "")
                + " · native + on-chain ✓ · SCOPE: native-HIP G1-only arm "
                  "re-measured; hip-wide NOT re-measured (a floor); the OpenCL "
                  "arms are separate and landed elsewhere (G1-only 0.994x "
                  "parity, gpu-wide 0.74x floor) — never blend or average them",
                (f"{_rel_to_repo(DEMO_C_PAIRED_CHECKOFF_CSV)} (G1-only) + "
                 f"{_rel_to_repo(DEMO_C_GPU_HIP_CSV)} (hip-wide)"),
                perf_watt="n/a (no separate HIP telemetry) ⇒ no perf-per-watt claim either way",
                bar_speedup=g1h)
        else:
            raise ValueError("no hip speedup rows")
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (native HIP / ark-groth16)",
            "folding DeciderEth Groth16 (Demo C, FOLD_N=2) — native HIP",
            "speedup vs CPU prove", "n/a (demo-c-gpu-hip.csv unavailable)",
            _rel_to_repo(DEMO_C_GPU_HIP_CSV))

    # 7c. Radeon iGPU (OpenCL + native HIP) — W3 Milestone A: a FULL halo2 KZG
    #     prove whose BN254 MSM + NTT hotspots run on the iGPU via the vendored
    #     PSE/zkonduit halo2_proofs fork + the gpu-msm seam onto Bn254Gpu. This is
    #     the halo2-layer unblock of "EZKL/halo2 on AMD" — bit-for-bit == the CPU
    #     prove (both verify_proof OK). HONESTY: enablement at a size-gated ceiling
    #     (MSM crossover ~2^22, NTT parity-to-~2x), so the full-prove wall is BELOW
    #     parity (~2x the 32t CPU) at 2^16–2^20 — portability, not a speedup.
    try:
        h = load_halo2_gpu_prove()
        bench = h[h["gpu_ms"].notna() & (h["gpu_ms"] > 0) & h["speedup"].notna()].sort_values("k")
        solo_clean = (
            h["solo"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).all()
        )
        if len(bench):
            top = bench.iloc[-1]
            sp = float(top["speedup"]); kk = int(top["k"])
            add("Radeon iGPU (halo2 KZG / vendored fork+seam)",
                "W3 halo2 KZG prove — full prove, iGPU MSM+NTT",
                "GPU==CPU + speedup vs 32t CPU",
                f"bit-for-bit == CPU ✓ · {_fmt_x(sp)} @ 2^{kk}"
                + (" · solo" if solo_clean else "")
                + " (enablement; MSM crossover ~2^22, not a speedup)",
                _rel_to_repo(HALO2_GPU_PROVE_CSV))
        else:
            vok = h["verify"].astype(str).str.strip().str.upper().eq("OK").any()
            add("Radeon iGPU (halo2 KZG / vendored fork+seam)",
                "W3 halo2 KZG prove — full prove, iGPU MSM+NTT",
                "GPU==CPU + speedup vs 32t CPU",
                ("bit-for-bit == CPU ✓ (gate only; timing hardware-gated)" if vok
                 else "n/a (no verified rows)"),
                _rel_to_repo(HALO2_GPU_PROVE_CSV))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (halo2 KZG / vendored fork+seam)",
            "W3 halo2 KZG prove — full prove, iGPU MSM+NTT",
            "GPU==CPU + speedup vs 32t CPU",
            "n/a (halo2-gpu-prove.csv unavailable — produced live on the Halo)",
            _rel_to_repo(HALO2_GPU_PROVE_CSV))

    # 8. Radeon iGPU OpenCL — zkRAG retrieval re-cast as Groth16 (the iGPU on a PROOF).
    try:
        zr = load_zkrag_msm()
        ok = zr[zr["speedup"].notna()].sort_values("m")
        top = ok.iloc[-1]
        all_verified = (
            zr["verified"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).all()
        )
        import math
        add("Radeon iGPU (OpenCL / bellperson)",
            "zkRAG retrieval MSM-SNARK (Groth16/BLS12-381)",
            "speedup vs 32t CPU",
            f"{_fmt_x(float(top['speedup']))} @ 2^{int(round(math.log2(float(top['m']))))} "
            f"(size-gated parity){' · every row verified ✓' if all_verified else ''}",
            _rel_to_repo(ZKRAG_MSM_CSV))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (OpenCL / bellperson)",
            "zkRAG retrieval MSM-SNARK (Groth16/BLS12-381)",
            "speedup vs 32t CPU", "n/a (zkrag-msm.csv unavailable)",
            _rel_to_repo(ZKRAG_MSM_CSV))

    # 9. Radeon iGPU OpenCL — zkLLM attention-matmul Groth16 (engine-split track C).
    #     HONESTY: every row in zkllm-msm.csv is stamped solo=false — the
    #     end-to-end prove_speedup is the SAME contention-prone metric the Demo C
    #     track refuted (advantage.md / INTEGRITY-REPORT decline it as a win). So
    #     this is reported as a CORRECTNESS/PARITY result (GPU==CPU verified), not
    #     a speed win: the speedup is rendered with "×" so plot_scorecard does not
    #     draw it as a green win-bar (its regex matches ASCII "x" only).
    try:
        zl = load_zkllm_msm()
        ok = zl[zl["prove_speedup"].notna()].sort_values("constraints_pow")
        top = ok.iloc[-1]
        all_verified = (
            zl["verify_ok"].astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes"]).all()
        )
        all_solo_false = (
            zl["solo"].astype(str).str.strip().str.lower()
            .isin(["false", "0", "no"]).all()
        )
        gate = (
            "solo=false (contended upper bound, not a verified speed win — see INTEGRITY-REPORT)"
            if all_solo_false else "size-gated end-to-end crossover"
        )
        add("Radeon iGPU (OpenCL / bellperson)",
            "zkLLM attention-matmul Groth16 (split track C)",
            "correctness (speed not a verified win)",
            f"{'GPU==CPU verified ✓; ' if all_verified else ''}"
            f"end-to-end prove_speedup {float(top['prove_speedup']):.2f}× @ "
            f"2^{int(top['constraints_pow'])} is {gate}",
            _rel_to_repo(ZKLLM_MSM_CSV))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (OpenCL / bellperson)",
            "zkLLM attention-matmul Groth16 (split track C)",
            "speedup vs 32t CPU", "n/a (zkllm-msm.csv unavailable)",
            _rel_to_repo(ZKLLM_MSM_CSV))

    # 9b. Radeon iGPU NATIVE HIP — end-to-end BN254 Groth16 prove (zkRAG padded,
    #     MSM/NTT-dominated): the curve where the iGPU wins the WHOLE prove. Peak
    #     win + the largest measured size (the plan-C 2^22 extension).
    try:
        zb = pd.read_csv(ZKRAG_BN254_PROVE_CSV)
        hip = zb[zb["backend"].astype(str).str.lower() == "hip"].copy()
        hip["speedup_vs_cpu"] = pd.to_numeric(hip["speedup_vs_cpu"], errors="coerce")
        hip["pow"] = pd.to_numeric(hip["pow"], errors="coerce")
        hip = hip[hip["speedup_vs_cpu"].notna()]
        peak = hip.loc[hip["speedup_vs_cpu"].idxmax()]
        top = hip.loc[hip["pow"].idxmax()]
        all_eq = (
            zb[zb["backend"].astype(str).str.lower() != "cpu"]["gpu_eq_cpu"]
            .astype(str).str.strip().str.lower().isin(["yes", "true", "1"]).all()
        )
        add("Radeon iGPU (native HIP / BN254 ark-groth16)",
            "zkRAG end-to-end Groth16 prove (padded, MSM/NTT-dominated)",
            "speedup vs 32t CPU",
            f"{_fmt_x(float(peak['speedup_vs_cpu']))} @ 2^{int(peak['pow'])} peak, "
            f"holds {_fmt_x(float(top['speedup_vs_cpu']))} @ 2^{int(top['pow'])} "
            f"(512·CU OOM-safe) · solo{' · GPU==CPU ✓' if all_eq else ''}",
            _rel_to_repo(ZKRAG_BN254_PROVE_CSV))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (native HIP / BN254 ark-groth16)",
            "zkRAG end-to-end Groth16 prove (padded, MSM/NTT-dominated)",
            "speedup vs 32t CPU", "n/a (zkrag-bn254-prove.csv unavailable)",
            _rel_to_repo(ZKRAG_BN254_PROVE_CSV))

    # 9c. Radeon iGPU NATIVE HIP — zkLLM attention re-cast to BN254 (the prior
    #     PROJECTION, now MEASURED end-to-end across real multi-head MiniLM sizes).
    try:
        zh = pd.read_csv(ZKLLM_MSM_HIP_CSV)
        hip = zh[zh["backend"].astype(str).str.lower() == "hip"].copy()
        hip["speedup_vs_cpu"] = pd.to_numeric(hip["speedup_vs_cpu"], errors="coerce")
        hip["speedup_vs_opencl"] = pd.to_numeric(hip["speedup_vs_opencl"], errors="coerce")
        hip = hip[hip["speedup_vs_cpu"].notna()]
        lo, hi = float(hip["speedup_vs_cpu"].min()), float(hip["speedup_vs_cpu"].max())
        ol, oh = float(hip["speedup_vs_opencl"].min()), float(hip["speedup_vs_opencl"].max())
        all_eq = (
            zh[zh["backend"].astype(str).str.lower() != "cpu"]["gpu_eq_cpu"]
            .astype(str).str.strip().str.lower().isin(["yes", "true", "1"]).all()
        )
        add("Radeon iGPU (native HIP / BN254 ark-groth16)",
            "zkLLM attention re-cast Groth16 (MEASURED, was a projection)",
            "speedup vs 32t CPU",
            f"{lo:.2f}x–{hi:.2f}x vs CPU (vs OpenCL {ol:.2f}x–{oh:.2f}x; OpenCL alone "
            f"loses to CPU) @ 2^20–2^22 · solo{' · GPU==CPU ✓' if all_eq else ''}",
            _rel_to_repo(ZKLLM_MSM_HIP_CSV))
    except Exception:  # noqa: BLE001
        add("Radeon iGPU (native HIP / BN254 ark-groth16)",
            "zkLLM attention re-cast Groth16 (MEASURED, was a projection)",
            "speedup vs 32t CPU", "n/a (zkllm-msm-hip.csv unavailable)",
            _rel_to_repo(ZKLLM_MSM_HIP_CSV))

    # 10. XDNA2 NPU — AI-model dispatch (no live telemetry — xrt-smi absent).
    try:
        summ = summary_from_full_run_info()
        detected = str(summ.get("npu.detected", "")).strip().lower() in ("yes", "true", "1")
        verdict = summ.get("npu.verdict", "").split("—")[0].strip() or "driver state probed"
        add("XDNA2 NPU (XDNA2 / RyzenAI)", "AI-model dispatch (research line)",
            "dispatch evidence",
            (f"DISPATCH-OK ({verdict}); " if detected else "probed; ")
            + "no live telemetry (xrt-smi absent) — proofs stay CPU-only",
            _rel_to_repo(FULL_RUN_INFO))
    except Exception:  # noqa: BLE001
        add("XDNA2 NPU (XDNA2 / RyzenAI)", "AI-model dispatch (research line)",
            "dispatch evidence", "n/a (full-run.info unavailable)",
            _rel_to_repo(FULL_RUN_INFO))

    # 11. Unified memory — CPU-only proving uses system RAM. This is not a GPU
    #     VRAM claim: the 36.5 GB figure is host RSS and a 15 GB laptop cannot run it.
    #     The RAM figure is the `unified_ram_gb` CAPTURED in the cited artefact, never
    #     the live host: the row describes the machine the proves actually ran on, so
    #     a laptop or a 512 GB server regenerating this table must still print 94 GB.
    try:
        ram = int(float(summary_from_full_run_info()["unified_ram_gb"]))
        add("Zen5 CPU + unified system RAM", "all CPU-only proves at scale (STARK / Halo2)",
            "host-RSS headroom",
            f"{ram} GB system RAM — 36.5 GB CPU-prover RSS fits; 15 GB laptop does not",
            _rel_to_repo(FULL_RUN_INFO))
    except Exception:  # noqa: BLE001
        add("Zen5 CPU + unified system RAM", "all CPU-only proves at scale (STARK / Halo2)",
            "host-RSS headroom", "n/a (full-run.info unavailable)",
            _rel_to_repo(FULL_RUN_INFO))

    # 12. Demo H XL — what happens when the AI-model working set exceeds a 32 GB
    #     device pool. The comparison is same-machine and same-weights; it measures
    #     spill cost, not an inability of a discrete card to address host memory.
    for path in (BIGMODEL_XL_JSON, BIGMODEL_XL_GEMMA_JSON):
        try:
            bm = load_bigmodel_info(path)
            full = bm["conditions"]["full_igpu"]
            cap = bm["conditions"]["cap_32gb"]
            loss = (1.0 - float(cap["gen_tps"]) / float(full["gen_tps"])) * 100.0
            add("Radeon iGPU (llama.cpp / HIP)",
                f"Demo H XL — {bm['model']} (AI model)",
                "GPU-resident working set / 32 GB-cap spill cost",
                f"{full['peak_gpu_gb']:.2f} GB resident; 32 GB cap: "
                f"{full['gen_tps']:.2f} → {cap['gen_tps']:.2f} gen tok/s "
                f"(−{loss:.0f}%; APU LPDDR5X spill — discrete PCIe spill unmeasured)",
                _rel_to_repo(path))
        except Exception:  # noqa: BLE001
            add("Radeon iGPU (llama.cpp / HIP)",
                f"Demo H XL — {path.stem}",
                "GPU-resident working set / 32 GB-cap spill cost",
                f"n/a ({path.name} unavailable)", _rel_to_repo(path))

    return pd.DataFrame(
        rows,
        columns=["engine", "workload", "metric", "measured", "perf_per_watt",
                 "evidence_file", "bar_speedup"],
    )


#: Matches the ``Nx`` speedup figures inside a :func:`scorecard_table` ``measured``
#: cell. The boundary guards are load-bearing, NOT cosmetic: this pattern used to be
#: ``r"(\d+(?:\.\d+)?)\s*x"`` with ``re.IGNORECASE``, which read the ``5X`` of
#: ``LPDDR5X`` as ``5x``. That invented a **5.00x green "iGPU win" bar** for the two
#: Demo H XL rows, whose ``measured`` cells carry only memory-capacity figures
#: ("44.77 GB resident; ... APU LPDDR5X spill") and which
#: :func:`plot_scorecard` therefore has to route to the enabler annotation instead.
#: So: no ``IGNORECASE`` (a bare ``[xX]`` is explicit), the digit run may not start
#: mid-identifier, and the ``x`` may not be followed by a letter.
_SPEEDUP_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[xX](?![A-Za-z])")


def plot_scorecard(df=None, save: Optional[Path] = None):
    """Plot a horizontal-bar glance of per-track iGPU speedups; return the ``Figure``.

    Extracts the representative speedup from each :func:`scorecard_table` row whose
    ``measured`` cell carries an ``Nx`` figure (the iGPU OpenCL/MIGraphX tracks),
    draws them as horizontal bars with a dashed **1.0× break-even** line, colours
    clean wins (>1) green and size-gated/parity rows (≤1) amber, and lists the
    CPU-only / NPU / unified-memory rows as **enabler** annotations (no speedup
    bar — they are not GPU-speedup claims). Defaults to :func:`scorecard_table`.

    A row earns a bar from its explicit ``bar_speedup`` column when it sets one,
    otherwise from :data:`_SPEEDUP_RE` finding a real ``Nx`` figure. Read that
    constant's note before loosening it: the memory-capacity rows (Demo H XL)
    say ``LPDDR5X``, and a case-insensitive ``\\d+x`` turns that ``5X`` into a
    fabricated 5.00x green win bar — a speedup no artefact ever measured.

    **Why ``bar_speedup`` exists.** The regex fallback takes the largest ``Nx``
    in the cell, which is only ever right by accident once a cell names more
    than one arm. Both Demo C folding cells now name three numbers — the current
    paired figure, the superseded figure it replaces, and an un-re-measured wide
    floor — and a bar chosen by ``max()`` over that prose is one edit away from
    plotting a retracted number. The folding rows therefore name their arm
    explicitly, and their workload label says which arm the bar is
    (``bar = G1-only paired``).

    🔴 The OpenCL folding bar is **0.994×, parity** — amber, at break-even, not
    a green win and not a red slowdown. Its four sibling arms are never averaged
    into one bar: OpenCL G1-only 0.994× parity, native-HIP G1-only 1.048× small
    win, ``gpu-wide`` 0.74× floor, ``hip-wide`` 0.77× floor.
    """
    if df is None:
        df = scorecard_table()
    plt = _get_plt()

    bars, enablers = [], []
    for _, r in df.iterrows():
        measured = str(r["measured"])
        engine = str(r["engine"])
        workload = str(r["workload"])
        is_gpu = "igpu" in engine.lower() or "radeon" in engine.lower()
        # An explicit bar_speedup always wins: rows whose cell names several
        # arms (the Demo C folding rows name a current paired figure, the
        # superseded figure it replaces, and an un-re-measured wide floor) must
        # not have their bar chosen by max() over whatever digits appear in the
        # prose. Only fall back to the regex when the row did not name one.
        explicit = r.get("bar_speedup") if hasattr(r, "get") else None
        try:
            explicit = float(explicit) if explicit is not None and explicit == explicit else None
        except (TypeError, ValueError):
            explicit = None
        match = _SPEEDUP_RE.search(measured)
        if is_gpu and explicit is not None and explicit > 0:
            bars.append((workload, explicit))
        elif is_gpu and match and "n/a" not in measured.lower():
            # Use the LARGEST Nx in the cell (e.g. the wider folding offload).
            vals = [float(m) for m in _SPEEDUP_RE.findall(measured)]
            bars.append((workload, max(vals)))
        else:
            enablers.append(workload)

    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.55 * len(bars) + 1.5)))
    if bars:
        labels = [b[0] for b in bars][::-1]
        vals = [b[1] for b in bars][::-1]
        colors = ["tab:green" if v > 1.0 else "tab:orange" for v in vals]
        y = range(len(vals))
        ax.barh(list(y), vals, color=colors, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + 0.05, yi, f"{v:.2f}x", va="center", fontsize=9)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(1.0, color="gray", ls="--", alpha=0.8)
        ax.text(1.0, len(vals) - 0.4, "break-even (iGPU == CPU)", fontsize=8,
                color="gray", ha="center")
        ax.set_xlim(0, max(vals) * 1.18)
    else:
        ax.text(0.5, 0.5, "no iGPU speedup rows available", ha="center", va="center")
    ax.set_xlabel("speedup = CPU / iGPU  (>1 ⇒ iGPU wins; green = clean, amber = size-gated/parity)")
    # No bound is claimed for NTT here (no roofline / memory-side PMC in the repo)
    # and the curve is named, so this title cannot contradict the NTT row it draws.
    # "~2²² parity" was a FAMILY-level literal and it was wrong for the very
    # curve it names first: the BLS12-381 G1 MSM of gpu-primitive.csv crosses at
    # 2²⁰ (1.059×) and peaks 1.213× at 2²². ~2²² is the BN254 MSM / full-Groth16
    # / retrieval crossover, so the band has to be stated as a band and the two
    # ends attributed, or this title contradicts its own 1.21× bar.
    ax.set_title("AMD Strix Halo cross-engine scorecard — iGPU speedups by track\n"
                 "(NTT BLS12-381 vs blstrs wins across the sweep; MSM/Groth16 "
                 "size-gated to ~2²⁰–2²² parity —\n"
                 "BLS12-381 G1 MSM crosses at 2²⁰, full Groth16 / retrieval only "
                 "at 2²²; MiniLM forward is the AI model, not the proof)",
                 fontsize=11)
    ax.grid(True, axis="x", ls=":", alpha=0.4)

    if enablers:
        note = ("CPU-only / enabler rows (no GPU-speedup claim): "
                + "; ".join(enablers[:6]) + (" …" if len(enablers) > 6 else ""))
        fig.text(0.01, 0.005, note, fontsize=7.5, color="#666", ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if save is not None:
        fig.savefig(str(save), dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Efficiency, roofline & NPU GEMM (nb15) — all derived from committed artefacts
# (no new heavy runs / no live capture): #2 energy trapezoid-integrates the
# committed GPU telemetry power, #3 roofline aggregates the existing iGPU-vs-CPU
# speedup loaders, #5 NPU reads the committed npu-dispatch.json. Every helper is
# headless-safe via _get_plt() and degrades (never raises) on a missing source.
# ---------------------------------------------------------------------------

#: Committed XDNA2 NPU dispatch verdict + measured attention-GEMM shapes (JSON).
#: 11 int8 GEMM shapes at MiniLM/BERT attention sizes driven via the proven
#: ``whole_array`` harness; ``attention_gemm_peak_gflops`` = 1491.31 (min-latency
#: representative). HONESTY: the NPU accelerates the AI model forward, NEVER the
#: proof. See :func:`load_npu_dispatch` / :func:`plot_npu_gemm`.
NPU_DISPATCH_JSON: Path = _RISC0_DEMO / "artefacts" / "npu-dispatch.json"


def telemetry_energy(df_or_path) -> dict:
    """Derive GPU-side energy + power stats from a committed telemetry track.

    Trapezoid-integrates the ``power_w`` samples over ``elapsed_s`` (the committed
    ``*.telemetry.csv`` schema :data:`TELEMETRY_COLUMNS`) to get the energy of the
    run in joules — pure replay, no new capture. ``df_or_path`` is either a
    telemetry ``DataFrame`` (from :func:`load_gpu_telemetry`) or a path to one
    (e.g. a :data:`TELEMETRY_DIR` value). Returns::

        {"joules", "wh", "duration_s", "avg_power_w", "peak_power_w", "samples"}

    HONESTY: on this APU ``rocm-smi`` ``power_w`` is the **whole-SoC package**
    draw (the iGPU shares the LPDDR5X budget with the Zen 5 cores), so this is a
    GPU-side / whole-package energy figure, not a clean per-kernel split; the CPU
    side is not separately telemetered here (``amd-smi`` returns N/A for CPU power
    on Strix Halo). Use it for energy-per-unit-work + perf-per-watt on the iGPU
    tracks, never as a GPU-vs-CPU energy comparison.
    """
    if hasattr(df_or_path, "columns"):
        df = df_or_path
    else:
        df = load_gpu_telemetry(df_or_path)
    sub = df[["elapsed_s", "power_w"]].dropna().sort_values("elapsed_s")
    t = sub["elapsed_s"].tolist()
    p = sub["power_w"].tolist()
    joules = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt > 0:
            joules += 0.5 * (p[i] + p[i - 1]) * dt  # trapezoid: W·s = J
    duration = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    if duration > 0:
        avg_power = joules / duration
    else:
        avg_power = float(sum(p) / len(p)) if p else 0.0
    return {
        "joules": float(joules),
        "wh": float(joules / 3600.0),
        "duration_s": duration,
        "avg_power_w": float(avg_power),
        "peak_power_w": float(max(p)) if p else 0.0,
        "samples": int(len(t)),
    }


def plot_energy(tracks=None):
    """Plot GPU-side energy-per-run + package power across the telemetry tracks.

    Two panels over the committed iGPU telemetry tracks (default
    :data:`TELEMETRY_DIR`): (1) **energy per run** (kJ, trapezoid-integrated
    ``power_w``×``elapsed_s`` via :func:`telemetry_energy`); (2) **average + peak
    whole-SoC package power** (W). ``tracks`` is a ``{label: path}`` mapping.
    Returns the ``Figure``.

    HONESTY: ``power_w`` is the whole-SoC package draw on this APU (shared LPDDR5X
    budget), not an isolated iGPU rail, and the CPU side is not separately
    telemetered (``amd-smi`` CPU power = N/A). This is a GPU-side efficiency view,
    NOT a GPU-vs-CPU energy bar.
    """
    if tracks is None:
        tracks = TELEMETRY_DIR
    plt = _get_plt()

    labels, kj, avg_w, peak_w = [], [], [], []
    for label, path in tracks.items():
        try:
            e = telemetry_energy(path)
        except Exception:  # noqa: BLE001 - a missing track degrades, never breaks
            continue
        labels.append(label)
        kj.append(e["joules"] / 1000.0)
        avg_w.append(e["avg_power_w"])
        peak_w.append(e["peak_power_w"])

    fig, (ax_e, ax_p) = plt.subplots(1, 2, figsize=(14, 5))
    if labels:
        x = list(range(len(labels)))
        ax_e.bar(x, kj, color="tab:blue", alpha=0.85)
        for xi, v in zip(x, kj):
            ax_e.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax_e.set_xticks(x)
        ax_e.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax_e.set_ylabel("energy per run (kJ)")
        ax_e.set_title("GPU-side energy per run\n(∫ package power dt — committed telemetry)")
        ax_e.grid(True, axis="y", ls=":", alpha=0.5)

        width = 0.4
        ax_p.bar([i - width / 2 for i in x], avg_w, width=width,
                 color="tab:green", alpha=0.85, label="avg")
        ax_p.bar([i + width / 2 for i in x], peak_w, width=width,
                 color="tab:orange", alpha=0.85, label="peak")
        ax_p.set_xticks(x)
        ax_p.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax_p.set_ylabel("whole-SoC package power (W)")
        ax_p.set_title("Average + peak package power\n(rocm-smi power_w = shared SoC budget)")
        ax_p.grid(True, axis="y", ls=":", alpha=0.5)
        ax_p.legend(fontsize=8)
    else:
        for ax in (ax_e, ax_p):
            ax.text(0.5, 0.5, "no telemetry tracks\n(replay a committed snapshot)",
                    ha="center", va="center")

    fig.suptitle("AMD Strix Halo iGPU energy & power — committed rocm-smi telemetry "
                 "(whole-SoC package draw; CPU side not telemetered)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def crossover_table():
    """Aggregate every committed iGPU-vs-CPU track into one roofline/crossover view.

    Pulls speedup-vs-size from the existing loaders — :func:`load_gpu_primitive`
    (MSM G1 + NTT/FFT), :func:`load_gpu_groth16`, :func:`load_zkrag_msm`,
    :func:`load_zkllm_msm` (standalone BN254 MSM column) and :func:`load_demo_c_gpu`
    (folding decider) — and tags each with the **regime** its measured curve is in:
    **above parity across sweep** (the NTT/FFT track) or **size-gated (OpenCL)**
    (MSM / Groth16 / retrieval / folding: parity only at large sizes — BLS12-381
    MSM crosses at 2²⁰, BN254 MSM / Groth16 / retrieval only at ~2²²). That
    gating is a property of the ec-gpu OpenCL kernel, not of MSM and not of the
    shared LPDDR5X — a native HIP kernel on the same chip is 2.0–2.2× faster and
    wins from 2^16. Returns a ``DataFrame``:
    ``track, primitive, regime, sizes, best_speedup, crossover_log2, note``.
    ``crossover_log2`` is the smallest ``log2(size)`` whose speedup ≥ 1 (``NaN`` ⇒
    never crosses in the swept range). A loader with missing data degrades to a
    skipped row, never an exception.

    The column is ``regime``, not ``bound``, and that is deliberate. **No bound is
    claimed for any row here.** The repo has no NTT roofline / arithmetic-intensity
    or memory-side PMC (``rocprof-ntt.csv`` collected only VALU/wave/busy; omniperf
    refuses gfx1151), so a ``bound`` field would have made an unmeasured claim a
    schema guarantee — and "size-gated (OpenCL)" was never a bound either, it is a
    kernel-path observation. The NTT track is also named by its **curve**
    (BLS12-381 FFT vs blstrs ``parallel_fft``) because staying above parity is a
    property of that curve, not of NTT: the same iGPU's BN254 Fr NTT vs arkworks
    loses at 2^18 (0.963×, :data:`MSM_NTT_BACKEND_CSV`).
    """
    import math

    pd = _require_pandas()
    rows = []

    def _crossover(xs, sp):
        hits = [x for x, s in zip(xs, sp) if s >= 1.0]
        return float(min(hits)) if hits else float("nan")

    try:
        prim = load_gpu_primitive()
        for p, track, regime, note in (
            ("fft", "Path E NTT — BLS12-381 FFT vs blstrs",
             "above parity across sweep",
             "BLS12-381 FFT vs blstrs parallel_fft — above parity at every swept "
             "size; NO bound claimed (no NTT roofline / memory-side PMC in repo). "
             "The same iGPU's BN254 Fr NTT vs arkworks loses at 2^18 (0.963×)"),
            ("msm", "Path E MSM (G1 multiexp)", "size-gated (OpenCL)",
             "BLS12-381 G1 multiexp — crossover 2²⁰ (1.059×), best 1.213× at "
             "2²², on the ec-gpu OpenCL path. The BN254 G1 multiexp is a "
             "SEPARATE curve on the same kernel and crosses later: 0.65–1.09× "
             "over 2¹⁶–2²² (crossover ~2²², gpu-bn254.csv) — the earlier "
             "1.1–1.35× small-size win was contention-inflated and is retired"),
        ):
            g = prim[prim["primitive"] == p].dropna(subset=["log_size", "speedup"])
            if g.empty:
                continue
            xs, sp = g["log_size"].tolist(), g["speedup"].tolist()
            rows.append((track, p, regime, f"2^{int(min(xs))}–2^{int(max(xs))}",
                         float(max(sp)), _crossover(xs, sp), note))
    except Exception:  # noqa: BLE001
        pass

    try:
        g = load_gpu_groth16().dropna(subset=["constraints_pow", "speedup"])
        if not g.empty:
            xs, sp = g["constraints_pow"].tolist(), g["speedup"].tolist()
            rows.append(("Path E Groth16 (full prove)", "groth16",
                         "size-gated (OpenCL)",
                         f"2^{int(min(xs))}–2^{int(max(xs))}", float(max(sp)),
                         _crossover(xs, sp),
                         "full Groth16 prove — only just flips past parity at "
                         "2²² (1.015×) (ec-gpu OpenCL path)"))
    except Exception:  # noqa: BLE001
        pass

    try:
        z = load_zkrag_msm().dropna(subset=["m", "speedup"])
        if not z.empty:
            xs = [math.log2(m) for m in z["m"].tolist()]
            sp = z["speedup"].tolist()
            rows.append(("zkRAG retrieval MSM-SNARK", "groth16",
                         "size-gated (OpenCL)",
                         f"2^{int(min(xs))}–2^{int(max(xs))}", float(max(sp)),
                         _crossover(xs, sp),
                         "iGPU touches the PROOF — parity ~1.02× at 2²² "
                         "(ec-gpu OpenCL path)"))
    except Exception:  # noqa: BLE001
        pass

    try:
        z = load_zkllm_msm().dropna(subset=["constraints_pow", "bn254_msm_speedup"])
        if not z.empty:
            xs = z["constraints_pow"].tolist()
            sp = z["bn254_msm_speedup"].tolist()
            rows.append(("zkLLM attn-matmul BN254 MSM", "msm", "size-gated (OpenCL)",
                         f"2^{int(min(xs))}–2^{int(max(xs))}", float(max(sp)),
                         _crossover(xs, sp),
                         "standalone BN254 G1 MSM at matched m — crossover ~2²²"))
    except Exception:  # noqa: BLE001
        pass

    try:
        d = load_demo_c_gpu()
        gpu = d[d["mode"].isin(["gpu", "gpu-wide"])].dropna(subset=["speedup_vs_cpu_median"])
        # best_speedup is the PAIRED G1-only figure (0.994x, parity) — the
        # published 0.70x it supersedes was an n=1 contended lower bound. The
        # gpu-wide 0.74x in this CSV was NOT re-measured and is a floor, so it
        # must not be averaged with, or upgraded by, the G1 correction.
        g1p = demo_c_opencl_g1_paired_speedup()
        wide_pub = None
        try:
            _w = gpu[gpu["mode"] == "gpu-wide"]["speedup_vs_cpu_median"]
            wide_pub = float(_w.iloc[0]) if not _w.empty else None
        except Exception:  # noqa: BLE001
            pass
        if g1p is not None or not gpu.empty:
            best = g1p if g1p is not None else float(gpu["speedup_vs_cpu_median"].max())
            note = ("BN254 G1 offload @2²⁴ — "
                    + (f"{g1p:.3f}× PARITY" if g1p is not None else "see artefact")
                    + " (paired, interleaved, n=3 each; supersedes the n=1 "
                      "contended 0.70× lower bound; gap 0.59% < within-arm "
                      "spreads 0.93%/1.21% ⇒ indistinguishable, NOT a win and "
                      "NOT 0.6% slower; parity is not acceleration)")
            if wide_pub is not None:
                note += (f" · wide (G1+G2+FFT) {wide_pub:.2f}× NOT re-measured, "
                         "a floor — the G1 correction does not transfer")
            note += " · see docs/INTEGRITY-REPORT.md"
            rows.append(("folding DeciderEth (Demo C)", "groth16", "size-gated (OpenCL)",
                         "2^24", best, float("nan"), note))
    except Exception:  # noqa: BLE001
        pass

    return pd.DataFrame(rows, columns=["track", "primitive", "regime", "sizes",
                                       "best_speedup", "crossover_log2", "note"])


def plot_roofline(save: Optional[Path] = None):
    """Plot the unified size-gating roofline of every iGPU-vs-CPU track.

    One overlay of speedup (CPU/iGPU) vs ``log2(size)`` for the committed tracks
    (:func:`load_gpu_primitive` MSM + NTT, :func:`load_gpu_groth16`,
    :func:`load_zkrag_msm`, :func:`load_zkllm_msm` BN254 MSM, plus the folding
    decider point), with a **break-even = 1.0×** line and the **~2²⁰–2²²
    crossover band** marked. The teaching point: the **NTT/FFT track stays above
    parity at every swept size**, while **MSM / Groth16 / retrieval / folding are
    size-gated on the ec-gpu OpenCL path**.

    **The band is a band, and each end is attributed.** This function draws the
    BLS12-381 G1 MSM curve itself, and that curve crosses at **2²⁰** (1.059×,
    best 1.213× at 2²²) — so the retired family-level "~2²² crossover" was
    contradicted by the blue series on the same axes. ~2²² is where **full
    Groth16** (1.015×), **zkRAG retrieval** (1.019×) and the **BN254** G1 MSM
    (1.091×; zkLLM matched-m 1.174×) cross; the **folding decider is still at
    parity at 2²⁴** (0.994×, paired G1-only) and never crosses in this sweep.
    That gating is a property of that kernel, not of MSM and not of the shared
    LPDDR5X — a native HIP kernel on the same chip is 2.0–2.2× faster and wins
    from 2^16. Returns the ``Figure``.

    **NO bound is claimed for the NTT/FFT track, and its legend names the curve.**
    The repo has no NTT roofline / arithmetic-intensity or memory-side PMC, and the
    curve drawn here is **BLS12-381 FFT vs blstrs** ``parallel_fft`` — not BN254
    ``Fr``. The legend used to read "Path E NTT (Fr FFT) — compute-bound", which
    both claimed an unmeasured bound and put the wrong field on the curve: the same
    iGPU's BN254 Fr NTT vs arkworks *loses* at 2^18 (0.963×,
    :data:`MSM_NTT_BACKEND_CSV`), i.e. the opposite direction.
    """
    import math

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(12, 6))

    series = []  # (label, xs, ys, color, marker, linestyle)
    try:
        prim = load_gpu_primitive()
        for p, lab, color, mk in (
            ("fft", "Path E NTT — BLS12-381 FFT vs blstrs (above parity)",
             "tab:green", "o"),
            ("msm", "Path E MSM (G1) — size-gated (OpenCL)", "tab:blue", "s"),
        ):
            g = prim[prim["primitive"] == p].dropna(
                subset=["log_size", "speedup"]).sort_values("log_size")
            if not g.empty:
                series.append((lab, g["log_size"].tolist(), g["speedup"].tolist(),
                               color, mk, "-"))
    except Exception:  # noqa: BLE001
        pass
    try:
        g = load_gpu_groth16().dropna(
            subset=["constraints_pow", "speedup"]).sort_values("constraints_pow")
        if not g.empty:
            series.append(("Path E Groth16 — size-gated (OpenCL)",
                           g["constraints_pow"].tolist(),
                           g["speedup"].tolist(), "tab:red", "^", "--"))
    except Exception:  # noqa: BLE001
        pass
    try:
        z = load_zkrag_msm().dropna(subset=["m", "speedup"]).sort_values("m")
        if not z.empty:
            series.append(("zkRAG retrieval MSM-SNARK — size-gated (OpenCL)",
                           [math.log2(m) for m in z["m"].tolist()],
                           z["speedup"].tolist(), "tab:purple", "D", "--"))
    except Exception:  # noqa: BLE001
        pass
    try:
        z = load_zkllm_msm().dropna(
            subset=["constraints_pow", "bn254_msm_speedup"]).sort_values("constraints_pow")
        if not z.empty:
            series.append(("zkLLM BN254 MSM (matched m) — size-gated (OpenCL)",
                           z["constraints_pow"].tolist(),
                           z["bn254_msm_speedup"].tolist(), "tab:brown", "v", "--"))
    except Exception:  # noqa: BLE001
        pass

    for lab, xs, ys, color, mk, ls in series:
        ax.plot(xs, ys, marker=mk, ls=ls, color=color, label=lab)

    # The two folding arms are drawn as two separate points with two separate
    # labels, never as one range: the G1-only arm has a paired re-bench and sits
    # at parity, while the wide arm was never re-measured and is only a floor.
    # Collapsing them into "0.70-0.74x" would state a superseded number as
    # current AND imply the correction transfers to the wide arm. It does not.
    try:
        g1p = demo_c_opencl_g1_paired_speedup()
        if g1p is not None:
            ax.scatter([24], [g1p], marker="X", s=110, color="black", zorder=6,
                       label=(f"folding decider @2²⁴ G1-only — {g1p:.3f}× PARITY "
                              "(paired, n=3/arm; supersedes 0.70×)"))
    except Exception:  # noqa: BLE001
        pass
    try:
        d = load_demo_c_gpu()
        wide = d[d["mode"] == "gpu-wide"].dropna(subset=["speedup_vs_cpu_median"])
        if not wide.empty:
            ax.scatter([24], [float(wide["speedup_vs_cpu_median"].iloc[0])],
                       marker="v", s=90, facecolors="none", edgecolors="black",
                       linewidths=1.6, zorder=6,
                       label=(f"folding decider @2²⁴ wide (G1+G2+FFT) — "
                              f"{float(wide['speedup_vs_cpu_median'].iloc[0]):.2f}× "
                              "NOT re-measured, a floor"))
    except Exception:  # noqa: BLE001
        pass

    ax.axhline(1.0, color="gray", ls="--", alpha=0.8)
    ax.text(ax.get_xlim()[0], 1.03, "break-even (iGPU == CPU)", fontsize=8, color="gray")
    # The band spans 2²⁰–2²² and the arrow lands on 2²⁰, because 2²⁰ is where the
    # FIRST curve on these axes crosses. The retired form shaded 21.5–22.5 and
    # called it "crossover ~2²²", which left the blue BLS12-381 G1 MSM series
    # crossing 1.0× at 2²⁰ — visibly OUTSIDE the band the label called the
    # crossover. Both ends are now named so the label cannot outlive the curve.
    ax.axvspan(19.5, 22.5, color="gold", alpha=0.15)
    ax.annotate("size-gated crossover band ~2²⁰–2²² (ec-gpu OpenCL path)\n"
                "BLS12-381 G1 MSM crosses first at 2²⁰ (1.059×);\n"
                "Groth16 / retrieval / BN254 MSM only at 2²²",
                xy=(20, 1.0), xytext=(16.15, 2.45), fontsize=9, color="#806000",
                arrowprops=dict(arrowstyle="->", color="#806000"))
    ax.set_xlabel("log2(problem size)  —  MSM/NTT points · Groth16 constraints m")
    ax.set_ylabel("speedup = CPU / iGPU   (>1 ⇒ iGPU wins)")
    ax.set_title("AMD Strix Halo crossover map — why iGPU ZK acceleration is size-gated\n"
                 "NTT (BLS12-381 FFT vs blstrs) stays above parity — no bound claimed; "
                 "MSM / Groth16 / folding\n(ec-gpu OpenCL path) reach parity only at "
                 "~2²⁰–2²²: BLS12-381 G1 MSM at 2²⁰,\nGroth16 / retrieval / BN254 MSM "
                 "at 2²², folding still at parity at 2²⁴")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    if save is not None:
        fig.savefig(str(save), dpi=120, bbox_inches="tight")
    return fig


def load_npu_dispatch(path: Optional[Path] = None):
    """Load the committed XDNA2 NPU measured attention-GEMM shapes as a ``DataFrame``.

    Reads :data:`NPU_DISPATCH_JSON` (default) and flattens its
    ``attention_gemm_shapes`` list into one row per GEMM: ``label, M, K, N, kind,
    role, avg_us, min_us, avg_gflops, peak_gflops, result``. ``kind`` is ``real``
    (dim natively tileable: M%128, K%32, N%128) or ``proxy`` (A·V with ``N=d_head``
    padded 32/64 → 128, the 4-col array minimum). ``peak_gflops`` (min-latency) is
    the representative NPU figure; ``avg`` is depressed under concurrent CPU load.
    HONESTY: these int8 GEMMs accelerate the **AI model forward** (MiniLM/BERT
    attention + FFN matmuls) on the proven ``whole_array`` harness — never the ZK
    proof; the r0vm STARK + Groth16 wrap stay CPU-only on AMD.
    """
    import pandas as pd

    _require_pandas()
    d = _read_json(path or NPU_DISPATCH_JSON)
    df = pd.DataFrame(d.get("attention_gemm_shapes", []))
    for col in ("M", "K", "N", "avg_us", "min_us", "avg_gflops", "peak_gflops"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cols = [c for c in ("label", "M", "K", "N", "kind", "role", "avg_us",
                        "min_us", "avg_gflops", "peak_gflops", "result")
            if c in df.columns]
    return df[cols] if cols else df


def plot_npu_gemm(df=None, save: Optional[Path] = None):
    """Plot the measured XDNA2 NPU per-shape peak int8 GEMM throughput; return ``Figure``.

    Horizontal bars of ``peak_gflops`` per attention-GEMM shape from
    :func:`load_npu_dispatch`, coloured by ``kind`` (real vs proxy), sorted
    ascending, with the **peak ≈ 1491 GFLOPs** ceiling marked. HONESTY: this is
    the AI-model forward (MiniLM/BERT attention + FFN matmuls) running int8 on the
    NPU's MAC array via the proven ``whole_array`` harness — the NPU **never
    touches the proof**; the r0vm STARK + Groth16 wrap stay CPU-only on AMD.
    """
    from matplotlib.patches import Patch

    if df is None:
        df = load_npu_dispatch()
    plt = _get_plt()
    g = df.dropna(subset=["peak_gflops"]).sort_values("peak_gflops")

    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.5 * len(g) + 1.5)))
    if not g.empty:
        labels = g["label"].tolist()
        vals = g["peak_gflops"].tolist()
        kinds = g["kind"].tolist() if "kind" in g.columns else ["real"] * len(g)
        colors = ["tab:blue" if k == "real" else "tab:orange" for k in kinds]
        y = list(range(len(vals)))
        ax.barh(y, vals, color=colors, height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.0f}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        peak = max(vals)
        ax.axvline(peak, color="crimson", ls="--", alpha=0.7)
        ax.text(peak, len(vals) - 0.4, f"peak ≈ {peak:.0f} GFLOPs", color="crimson",
                fontsize=9, ha="right", va="top")
        ax.set_xlim(0, peak * 1.14)
        ax.legend(handles=[Patch(color="tab:blue", label="real (natively tileable)"),
                           Patch(color="tab:orange", label="proxy (A·V N padded → 128)")],
                  fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "no NPU GEMM shapes", ha="center", va="center")
    ax.set_xlabel("measured peak int8 GFLOPs (min-latency)")
    ax.set_title("XDNA2 NPU — measured attention-GEMM throughput (DISPATCH-OK)\n"
                 "accelerates the AI model forward (MiniLM/BERT matmuls), never the proof")
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    if save is not None:
        fig.savefig(str(save), dpi=120, bbox_inches="tight")
    return fig
