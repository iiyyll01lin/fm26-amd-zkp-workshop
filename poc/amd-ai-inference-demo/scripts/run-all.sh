#!/usr/bin/env bash
# run-all.sh — Path F (Demo F) end-to-end:
#   amd-accel-detect (pick AI_GPU_BACKEND) -> venv (torch/onnx/ort) -> build the
#   all-MiniLM-L6-v2 ONNX -> MiniLM forward iGPU-vs-CPU sweep -> plot (PNG via
#   matplotlib, else a markdown table).
#
# Knobs (env): BATCHES, SEQ_LENS (comma lists), REPS, WARMUP, AI_BENCH_BACKENDS.
#   BATCHES=1,8,32 SEQ_LENS=32,128,256 REPS=10 bash scripts/run-all.sh
#
# When AI_GPU_BACKEND=rocm and AI_BENCH_BACKENDS is not overridden, the sweep
# now also runs the reduced-precision MIGraphX variants
# (rocm-fp16/rocm-int8/rocm-fp8) so ai-inference.csv carries the perf/watt of
# each quantization rung (see FP8-INT8-SCOPE.md). HONESTY: random-init weights ->
# throughput/footprint/perf-watt capability, NOT accuracy; FP8 is a research rung.
#
# SOLO GUARD: this is a timed benchmark, so — like every timed bench in this repo
# — the sweep passes the solo guard FIRST (exit 42 under iGPU/CPU contention) and
# stamps every emitted CSV row with `solo` + `loadavg`. Set ZKP_SOLO_OVERRIDE=1
# to record anyway (stamped solo=false).
#
# HONESTY: this benchmarks the iGPU (Radeon 8060S / gfx1151) running the AI
# MODEL forward pass (a MiniLM sentence embedding) via AMD's native MIGraphX
# (ROCm 7.2.3). It does NOT move any ZK proof onto the GPU — EZKL Halo2 (Demo A)
# and the RISC0 r0vm STARK (Demo B) proving stay CPU-only on AMD.
# See reading-notes/path-f-amd-ai-inference.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "${HERE}/.." && pwd)"
REPO="$(cd "${DEMO}/../.." && pwd)"
cd "${DEMO}"

: "${BATCHES:=1,8,32}"
: "${SEQ_LENS:=32,128,256}"
: "${REPS:=20}"
: "${WARMUP:=5}"
export BATCHES SEQ_LENS REPS WARMUP

# ROCm MIGraphX python bindings live under the ROCm tree (NOT pip-installable).
ROCM_DIR="$(ls -d /opt/rocm-* /opt/rocm 2>/dev/null | head -1 || true)"
[ -n "${ROCM_DIR}" ] && export PYTHONPATH="${ROCM_DIR}/lib:${PYTHONPATH:-}"

echo "######################################################################"
echo "# Path F (Demo F) — iGPU AI-inference (MiniLM forward) vs CPU"
echo "#   BATCHES=${BATCHES}  SEQ_LENS=${SEQ_LENS}  REPS=${REPS}  WARMUP=${WARMUP}"
echo "######################################################################"

# 1. AMD capability detection -> exports AI_GPU_BACKEND / RAYON_NUM_THREADS /
#    HSA_* (read-only; safe to source; never fatal).
DETECT="${REPO}/poc/risc0-cartesi-step-demo/scripts/amd-accel-detect.sh"
if [ -f "${DETECT}" ]; then
    # shellcheck source=/dev/null
    source "${DETECT}" || true
fi
echo "[run-all] AI_GPU_BACKEND=${AI_GPU_BACKEND:-cpu}  RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-?}"

# Default backend set: on a rocm box, sweep CPU + FP32 iGPU + the three quantized
# iGPU rungs (FP16/INT8/FP8). Override with AI_BENCH_BACKENDS=cpu,rocm to get the
# old two-backend run. Honesty: quantized rows are a perf/footprint capability on
# random-init weights, not an accuracy claim (FP8 is a research rung — scope §5).
if [ -z "${AI_BENCH_BACKENDS:-}" ] && [ "${AI_GPU_BACKEND:-cpu}" = "rocm" ]; then
    AI_BENCH_BACKENDS="cpu,rocm,rocm-fp16,rocm-int8,rocm-fp8"
fi
export AI_BENCH_BACKENDS
echo "[run-all] AI_BENCH_BACKENDS=${AI_BENCH_BACKENDS:-<auto>}"

# 2. self-contained venv (kept inside the demo dir; NOT the repo .venv).
PY="${DEMO}/.venv/bin/python"
if [ ! -x "${PY}" ]; then
    echo "[run-all] creating self-contained venv (torch-cpu + onnx + onnxruntime)..."
    "${PYTHON:-python3.13}" -m venv "${DEMO}/.venv"
    "${PY}" -m pip install --quiet --upgrade pip
    "${PY}" -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
    "${PY}" -m pip install --quiet -r "${DEMO}/requirements.txt"
fi

# 3. build the MiniLM-L6 ONNX (idempotent authoring step).
mkdir -p artefacts
if [ ! -f artefacts/minilm-l6.onnx ]; then
    "${PY}" src/build_model.py
fi

# 4. SOLO GUARD — refuse to record a contended timed bench (after venv/onnx
#    build, before any timing). Sets SOLO_STATUS / SOLO_LOADAVG for stamping.
GUARD="${REPO}/poc/risc0-cartesi-step-demo/scripts/solo-guard.sh"
if [ -f "${GUARD}" ]; then
    # shellcheck source=/dev/null
    source "${GUARD}"
    ZKP_GPU_LOCK="${ZKP_GPU_LOCK:-/tmp/zkp-gpu.lock}"
    cleanup() { rm -f "${ZKP_GPU_LOCK}"; }
    solo_guard_require                   # exits 42 if contended; sets SOLO_*
    echo "$$ ai-inference" >"${ZKP_GPU_LOCK}"
    trap cleanup EXIT
    SOLO="${SOLO_STATUS}"; LOADAVG="${SOLO_LOADAVG}"
else
    echo "[run-all] WARN: solo-guard.sh not found; running unguarded." >&2
    SOLO="unknown"; LOADAVG="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo na)"
fi

# 5. sweep -> raw CSV (stdout) + log (stderr); then stamp solo + loadavg onto
#    every data row (schema stays backward-compatible — columns appended).
echo "[run-all] running MiniLM forward sweep (iGPU MIGraphX vs CPU onnxruntime)..."
"${PY}" src/embed_bench.py \
  >artefacts/.ai-inference.raw \
  2>artefacts/ai-inference.log || true
awk -v solo="${SOLO}" -v la="${LOADAVG}" 'BEGIN{FS=OFS=","}
     NR==1 {print $0,"solo","loadavg"; next}
     NF    {print $0,solo,la}' \
  artefacts/.ai-inference.raw >artefacts/ai-inference.csv
rm -f artefacts/.ai-inference.raw
echo "[run-all] sweep complete -> artefacts/ai-inference.csv (solo=${SOLO} loadavg=${LOADAVG})"
cat artefacts/ai-inference.log

# 6. plot (PNG via matplotlib; degrades to a markdown table).
"${PY}" scripts/plot-ai-inference.py || true

echo "[run-all] artefacts:"
ls -l artefacts/
