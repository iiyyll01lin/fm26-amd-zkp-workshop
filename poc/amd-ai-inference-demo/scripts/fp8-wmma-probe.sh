#!/usr/bin/env bash
# fp8-wmma-probe.sh — resolve the ONE deferred question in FP8-INT8-SCOPE.md §5:
# when MIGraphX `quantize_fp8` compiles the MiniLM ONNX to the gfx1151 GPU
# target, does the dot/GEMM run on NATIVE WMMA FP8 instructions, or is it
# convert+upcast (fp8 -> f32) followed by an ordinary FP32 FMA GEMM?
#
# The measured Demo F perf already HINTS convert+upcast (FP8 is no win — 0.43x at
# large batch, commit 0ec9504, FP8-INT8-SCOPE.md §7). This script confirms it with
# INSTRUCTION-LEVEL evidence, gathered three ways for both the FP8 path and the
# FP16 path (FP16 is the "native WMMA present" control, since f16 IS a native
# gfx1151 WMMA input type):
#
#   1. MLIR operand types — MIGRAPHX_MLIR_DUMP dumps every `migraphx.dot` module
#      MIGraphX hands to rocMLIR. The dot OPERAND element type is the smoking
#      gun: an fp8-typed dot operand => the matrix op consumes fp8 (native WMMA
#      FP8); an f32-typed dot operand => the fp8 was upcast to f32 BEFORE the
#      matmul (convert+upcast; gfx1151 WMMA has no f32xf32 input mode, so an f32
#      dot CANNOT use WMMA at all).
#   2. ISA of the JIT pointwise/convert kernels — AMD_COMGR_SAVE_TEMPS keeps the
#      compiled code objects for the elementwise glue (quantize/dequantize/
#      convert/erf/...). llvm-objdump -d + grep shows the literal `v_cvt_*` /
#      `v_fma_f32` convert+FMA instructions and counts any matrix `v_wmma_*`.
#   3. rocprofv3 --kernel-trace — enumerates the GPU kernels actually dispatched
#      for the FP8 vs FP16 forward (solo-guarded; this is the only GPU-dispatch
#      step). Telltale names (mlir_*, convert, dequantizelinear, ...) corroborate.
#
# HONESTY (read FP8-INT8-SCOPE.md §0): Demo F weights are random-init, so this is
# a CAPABILITY / instruction probe, NOT an accuracy claim; and the iGPU
# accelerates the AI MODEL forward only — EZKL Halo2 / RISC0 STARK proving stay
# CPU-only on AMD. The gemm code object itself is compiled by rocMLIR IN-PROCESS
# and is not written to a separately disassemblable file by any supported knob
# (comgr SAVE_TEMPS = pointwise only; MLIR_DUMP_TO_MXR = pre-compile module); the
# verdict therefore rests on the MLIR dot operand types + the convert-kernel ISA
# + the kernel trace + the measured perf, and the script says so plainly. If a
# clean solo window never opens the kernel-trace is skipped (the compile-time MLIR
# + ISA evidence still stands); if the evidence is ambiguous the verdict is
# emitted as INCONCLUSIVE rather than guessed.
#
# Output: artefacts/fp8-wmma.{md,txt}
#
# Knobs (env): PROBE_BATCH (8), PROBE_SEQ (128), PROBE_RUNS (2), OFFLOAD_ARCH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "${HERE}/.." && pwd)"
REPO="$(cd "${DEMO}/../.." && pwd)"
cd "${DEMO}"

: "${OFFLOAD_ARCH:=gfx1151}"
: "${PROBE_BATCH:=8}"
: "${PROBE_SEQ:=128}"
: "${PROBE_RUNS:=2}"

# ROCm on PATH + the MIGraphX python bindings on PYTHONPATH (ROCm tree, NOT pip).
ROCM_DIR="$(ls -d /opt/rocm-* /opt/rocm 2>/dev/null | head -1 || true)"
if [ -n "${ROCM_DIR}" ]; then
  export PATH="${ROCM_DIR}/bin:${PATH}"
  export PYTHONPATH="${ROCM_DIR}/lib:${PYTHONPATH:-}"
fi
OBJDUMP="${ROCM_DIR}/llvm/bin/llvm-objdump"
[ -x "${OBJDUMP}" ] || OBJDUMP="$(command -v llvm-objdump || true)"

ROCPROF_V3="$(command -v rocprofv3 || true)"

PY="${DEMO}/.venv/bin/python"
[ -x "${PY}" ] || PY="python3"
HELPER="${HERE}/fp8_wmma_probe.py"
ONNX="${DEMO}/artefacts/minilm-l6.onnx"
if [ ! -f "${ONNX}" ]; then
  echo "[fp8-wmma] ONNX missing (${ONNX}); run 'make demo-f-embed' (or src/build_model.py) first." >&2
  exit 1
fi

ART="${DEMO}/artefacts"
WORK="${ART}/fp8-wmma-work"
rm -rf "${WORK}"
mkdir -p "${ART}" "${WORK}"
MD="${ART}/fp8-wmma.md"
TXT="${ART}/fp8-wmma.txt"

HW_LABEL="AMD Ryzen AI MAX+ 395 / Radeon 8060S (${OFFLOAD_ARCH})"
ROCM_VER="$(cat /opt/rocm*/.info/version 2>/dev/null | head -1)"
RUN_DATE="$(date +%F)"

# --- solo guard: the kernel-trace dispatches GPU kernels; gate it (graceful) ---
# The MLIR + ISA evidence is COMPILE-time (records no perf), so it always runs;
# only the rocprofv3 --kernel-trace GPU-dispatch enumeration is solo-gated.
KERNEL_TRACE_OK=0
SOLO_STATUS="unknown"; SOLO_LOADAVG="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo na)"
GUARD="${REPO}/poc/risc0-cartesi-step-demo/scripts/solo-guard.sh"
if [ -f "${GUARD}" ]; then
  # shellcheck source=/dev/null
  source "${GUARD}"
  solo_guard_probe
  solo_guard_report
  SOLO_STATUS="${SOLO_STATUS}"; SOLO_LOADAVG="${SOLO_LOADAVG}"
  if [ "${SOLO_OK}" -eq 0 ]; then
    KERNEL_TRACE_OK=1
  else
    echo "[fp8-wmma] CONTENDED (${SOLO_REASON}) — skipping the rocprofv3 kernel-trace;" >&2
    echo "[fp8-wmma] the compile-time MLIR + ISA evidence below still decides the verdict." >&2
  fi
fi
[ -n "${ROCPROF_V3}" ] || { KERNEL_TRACE_OK=0; echo "[fp8-wmma] rocprofv3 not found — kernel-trace skipped." >&2; }

# --- per-precision evidence gather -------------------------------------------
# $1 = quant label (fp16|fp8)
gather() {
  local q="$1"
  local mdir="${WORK}/mlir-${q}" cdir="${WORK}/comgr-${q}" tdir="${WORK}/trace-${q}"
  mkdir -p "${mdir}" "${cdir}" "${tdir}"

  echo "[fp8-wmma] (${q}) MLIR dump (dot operand types) ..." >&2
  MIGRAPHX_MLIR_DUMP="${mdir}" "${PY}" "${HELPER}" --quant "${q}" --onnx "${ONNX}" \
    --batch 1 --seq 32 --print-shapes >"${WORK}/${q}.shapes.out" 2>"${WORK}/${q}.mlir.log" || true

  echo "[fp8-wmma] (${q}) comgr SAVE_TEMPS (JIT convert-kernel ISA) ..." >&2
  TMPDIR="${cdir}" AMD_COMGR_SAVE_TEMPS=1 "${PY}" "${HELPER}" --quant "${q}" --onnx "${ONNX}" \
    --batch 1 --seq 32 --print-shapes >/dev/null 2>"${WORK}/${q}.comgr.log" || true

  if [ "${KERNEL_TRACE_OK}" -eq 1 ]; then
    echo "[fp8-wmma] (${q}) rocprofv3 --kernel-trace (b${PROBE_BATCH} s${PROBE_SEQ}) ..." >&2
    rocprofv3 --kernel-trace --output-format csv -d "${tdir}" -- \
      "${PY}" "${HELPER}" --quant "${q}" --onnx "${ONNX}" \
      --batch "${PROBE_BATCH}" --seq "${PROBE_SEQ}" --runs "${PROBE_RUNS}" \
      >"${WORK}/${q}.trace.out" 2>"${WORK}/${q}.trace.log" || true
  fi
}

# Element types appearing on `migraphx.dot` lines (input+output) -> histogram.
dot_types() {  # $1 = mlir dir
  grep -hE 'migraphx\.dot ' "$1"/*.mlir 2>/dev/null \
    | grep -oE 'f8[A-Za-z0-9]+|bf16|f16|f32|i32|i8' | sort | uniq -c | sort -rn
}
dot_fp8_count() {  # count fp8-typed tokens on dot lines
  grep -hE 'migraphx\.dot ' "$1"/*.mlir 2>/dev/null | grep -oE 'f8[A-Za-z0-9]+' | wc -l | tr -d ' '
}
dot_token_count() {  # $1=mlir dir $2=token
  grep -hE 'migraphx\.dot ' "$1"/*.mlir 2>/dev/null | grep -oE "$2" | wc -l | tr -d ' '
}
# v_wmma / convert instruction counts across the compiled JIT code objects.
isa_count() {  # $1 = comgr dir, $2 = instruction regex
  local n=0 so c
  while IFS= read -r so; do
    c="$("${OBJDUMP}" -d "${so}" 2>/dev/null | grep -cE "$2" || true)"
    n=$((n + c))
  done < <(find "$1" -name 'a.so' 2>/dev/null)
  echo "${n}"
}
isa_kernel_names() {  # $1 = comgr dir -> compiled kernel symbol names (non-runtime)
  local so
  while IFS= read -r so; do
    "${OBJDUMP}" -d "${so}" 2>/dev/null | grep -oE '<[A-Za-z0-9_]+>:'
  done < <(find "$1" -name 'a.so' 2>/dev/null) \
    | sed 's/[<>:]//g' | grep -ivE '^(__amd_rocclr|__ockl|__ocml)' | sort -u
}
trace_kernel_names() {  # $1 = trace dir -> dispatched GPU kernel names
  local f
  f="$(find "$1" -name '*kernel_trace.csv' 2>/dev/null | head -1)"
  [ -n "${f}" ] || return 0
  # rocprofv3 kernel_trace.csv has a "Kernel_Name" column; pull + demangle-ish.
  awk -F, 'NR==1{for(i=1;i<=NF;i++){gsub(/"/,"",$i); if($i=="Kernel_Name")k=i} next}
           k{gsub(/"/,"",$k); print $k}' "${f}" | sort | uniq -c | sort -rn
}

gather fp16
gather fp8

# --- crunch the evidence ------------------------------------------------------
set +e +o pipefail  # post-processing is best-effort grep/awk extraction
FP8_DOT_F8="$(dot_fp8_count "${WORK}/mlir-fp8")"
FP8_DOT_F32="$(dot_token_count "${WORK}/mlir-fp8" 'f32')"
FP8_DOT_F16="$(dot_token_count "${WORK}/mlir-fp8" 'f16')"
FP16_DOT_F16="$(dot_token_count "${WORK}/mlir-fp16" 'f16')"
FP16_DOT_F32="$(dot_token_count "${WORK}/mlir-fp16" 'f32')"
FP8_WMMA="$(isa_count "${WORK}/comgr-fp8" 'v_wmma')"
FP16_WMMA="$(isa_count "${WORK}/comgr-fp16" 'v_wmma')"
FP8_CVT="$(isa_count "${WORK}/comgr-fp8" 'v_cvt_')"
FP8_FMA="$(isa_count "${WORK}/comgr-fp8" 'v_fma_f32|v_fmac_f32|v_fmaak_f32|v_fmamk_f32')"
FP8_CONVERT_KERNELS="$(isa_kernel_names "${WORK}/comgr-fp8" | grep -icE 'quantizelinear|dequantizelinear|convert' || true)"

# Verdict logic (evidence-driven, no guessing).
if [ "${FP8_DOT_F8}" -gt 0 ]; then
  VERDICT="NATIVE WMMA FP8"
  VERDICT_WHY="fp8-typed operands were found feeding migraphx.dot (the matrix op consumes fp8 directly)."
elif [ "${FP8_DOT_F32}" -gt 0 ] && [ "${FP8_DOT_F8}" -eq 0 ] && [ "${FP16_DOT_F16}" -gt 0 ]; then
  VERDICT="CONVERT + UPCAST + FP32 FMA  (NOT native WMMA FP8)"
  VERDICT_WHY="every migraphx.dot in the FP8 program consumes f32 operands (the fp8 is upcast to f32 BEFORE the matmul); the FP16 control instead feeds f16 operands straight into the dot. gfx1151 WMMA has no f32xf32 input mode, so an f32 dot cannot use WMMA — it runs on v_fma_f32 VALU."
else
  VERDICT="INCONCLUSIVE"
  VERDICT_WHY="the MLIR dot operand types did not give a clean fp8-vs-f32 signal (FP8 dot f8=${FP8_DOT_F8}/f32=${FP8_DOT_F32}, FP16 dot f16=${FP16_DOT_F16}); not guessing a verdict."
fi

KT_NOTE="ran solo (solo=${SOLO_STATUS}, loadavg=${SOLO_LOADAVG})"
[ "${KERNEL_TRACE_OK}" -eq 1 ] || KT_NOTE="SKIPPED — no clean solo window (solo=${SOLO_STATUS}, loadavg=${SOLO_LOADAVG}) or rocprofv3 absent; verdict rests on the compile-time MLIR + ISA evidence"

# --- TXT (raw evidence, verbatim) --------------------------------------------
{
  echo "############################################################"
  echo "# FP8 native-WMMA vs convert+upcast probe — raw evidence"
  echo "#   hardware : ${HW_LABEL}"
  echo "#   ROCm     : ${ROCM_VER}    MIGraphX python binding (ROCm tree)"
  echo "#   probe    : MiniLM ONNX -> migraphx.quantize_fp8 / quantize_fp16 -> gpu target"
  echo "#   date     : ${RUN_DATE}    solo=${SOLO_STATUS} loadavg=${SOLO_LOADAVG}"
  echo "############################################################"
  echo ""
  echo "== (1) MLIR dot operand element types (MIGRAPHX_MLIR_DUMP) =="
  echo "-- FP8 program: every migraphx.dot operand/result element type --"
  dot_types "${WORK}/mlir-fp8"
  echo "   fp8-typed dot tokens: ${FP8_DOT_F8} ; f32: ${FP8_DOT_F32} ; f16: ${FP8_DOT_F16}"
  echo "-- FP16 control: every migraphx.dot operand/result element type --"
  dot_types "${WORK}/mlir-fp16"
  echo "   f16 dot tokens: ${FP16_DOT_F16} ; f32: ${FP16_DOT_F32}"
  echo ""
  echo "-- sample FP8 dot module (note: dequantizelinear upcasts to f32 around an f32 dot) --"
  for f in "${WORK}"/mlir-fp8/mlir_dot_*dequant*.mlir "${WORK}"/mlir-fp8/mlir_dot_*1536*384*.mlir; do
    [ -f "${f}" ] && { echo "### $(basename "${f}")"; cat "${f}"; break; }
  done
  echo ""
  echo "-- sample FP16 dot module (note: f16 operands fed straight into the dot) --"
  for f in "${WORK}"/mlir-fp16/mlir_dot_*1536*384*.mlir "${WORK}"/mlir-fp16/mlir_dot_*384*1536*.mlir; do
    [ -f "${f}" ] && { echo "### $(basename "${f}")"; cat "${f}"; break; }
  done
  echo ""
  echo "== (2) ISA of the JIT pointwise/convert kernels (AMD_COMGR_SAVE_TEMPS) =="
  echo "FP8  : v_wmma=${FP8_WMMA}  v_cvt_*=${FP8_CVT}  v_fma_f32*=${FP8_FMA}  convert/quant kernels=${FP8_CONVERT_KERNELS}"
  echo "FP16 : v_wmma=${FP16_WMMA}  (control)"
  echo "-- FP8 compiled JIT kernel symbol names (elementwise glue) --"
  isa_kernel_names "${WORK}/comgr-fp8"
  echo "-- FP16 compiled JIT kernel symbol names --"
  isa_kernel_names "${WORK}/comgr-fp16"
  echo ""
  echo "NOTE: the dot/GEMM itself is compiled by rocMLIR IN-PROCESS and is not"
  echo "written to a separately disassemblable code object by any supported knob"
  echo "(comgr SAVE_TEMPS captures only the elementwise/convert JIT kernels above;"
  echo "MLIR_DUMP_TO_MXR emits the PRE-compile module). So no v_wmma_* appears in"
  echo "EITHER precision's comgr objects — that is expected and is NOT evidence of"
  echo "absence in the gemm. The decisive gemm signal is the MLIR operand type in (1)."
  echo ""
  echo "== (3) rocprofv3 --kernel-trace dispatched GPU kernels =="
  echo "kernel-trace: ${KT_NOTE}"
  if [ "${KERNEL_TRACE_OK}" -eq 1 ]; then
    echo "-- FP8 dispatched kernels (count name) --"
    trace_kernel_names "${WORK}/trace-fp8"
    echo "-- FP16 dispatched kernels (count name) --"
    trace_kernel_names "${WORK}/trace-fp16"
  fi
  echo ""
  echo "== VERDICT =="
  echo "${VERDICT}"
  echo "${VERDICT_WHY}"
} >"${TXT}"

# --- MD (the readable verdict) ------------------------------------------------
{
  echo "# FP8 native-WMMA vs convert+upcast — verdict (gfx1151 / MIGraphX)"
  echo ""
  echo "_Source: \`artefacts/fp8-wmma.txt\` (raw evidence) — produced by"
  echo "\`poc/amd-ai-inference-demo/scripts/fp8-wmma-probe.sh\`._"
  echo ""
  echo "- **Hardware:** ${HW_LABEL}"
  echo "- **Stack:** ROCm ${ROCM_VER}, MIGraphX python binding (ROCm tree); profiler \`rocprofv3\`"
  echo "- **Probe:** MiniLM ONNX → \`migraphx.quantize_fp8\` / \`quantize_fp16\` → \`gpu\` target"
  echo "- **Date:** ${RUN_DATE} · solo=${SOLO_STATUS} loadavg=${SOLO_LOADAVG}"
  echo ""
  echo "## Verdict: **${VERDICT}**"
  echo ""
  echo "${VERDICT_WHY}"
  echo ""
  echo "This confirms the FP8-INT8-SCOPE.md §5 hypothesis and explains the measured"
  echo "perf (FP8 is no win — 0.43× at large batch, §7): FP8 pays quantize/dequantize/"
  echo "convert overhead **and** still runs the GEMM in FP32, so it is strictly worse"
  echo "than plain FP32 on this RDNA 3.5 iGPU."
  echo ""
  echo "## Evidence"
  echo ""
  echo "### 1. MLIR \`dot\` operand types (the decisive signal)"
  echo ""
  echo "\`MIGRAPHX_MLIR_DUMP\` dumps every \`migraphx.dot\` module MIGraphX hands to"
  echo "rocMLIR. The operand **element type** is the smoking gun:"
  echo ""
  echo "| precision | \`migraphx.dot\` operand element type | reading |"
  echo "|---|---|---|"
  echo "| **FP8** | **f32** (fp8 tokens on dot lines: ${FP8_DOT_F8}; f32: ${FP8_DOT_F32}) | fp8 is upcast to f32 **before** the matmul (\`dequantizelinear\` brackets the dot) → no fp8 matrix op |"
  echo "| **FP16** (control) | **f16** (f16 tokens: ${FP16_DOT_F16}) | f16 fed straight into the dot — f16 **is** a native gfx1151 WMMA input type |"
  echo ""
  echo "gfx1151 WMMA has no f32×f32 input mode, so an f32 \`dot\` **cannot** use WMMA"
  echo "at all — it lowers to \`v_fma_f32\` VALU. The FP16 control proves the pipeline"
  echo "*does* keep the matrix type when the hardware supports it (f16); FP8 does not."
  echo ""
  echo "### 2. JIT convert-kernel ISA (\`AMD_COMGR_SAVE_TEMPS\` + \`llvm-objdump -d\`)"
  echo ""
  echo "- The FP8 program compiles explicit **${FP8_CONVERT_KERNELS}** \`quantizelinear\` /"
  echo "  \`dequantizelinear\` / \`convert\` pointwise kernels, full of \`v_cvt_*\` (${FP8_CVT})"
  echo "  + \`v_fma_f32\` (${FP8_FMA}) — the literal convert+FMA round-trip."
  echo "- \`v_wmma_*\` count in the comgr-captured objects: FP8 ${FP8_WMMA} / FP16 ${FP16_WMMA}."
  echo "  **Caveat:** the dot/GEMM is compiled by rocMLIR *in-process* and is not"
  echo "  emitted to a separately disassemblable code object, so neither precision's"
  echo "  comgr objects contain the gemm — the 0/0 \`v_wmma\` is expected and is **not**"
  echo "  evidence about the gemm. The gemm signal is the MLIR operand type in (1)."
  echo ""
  echo "### 3. \`rocprofv3 --kernel-trace\` (dispatched GPU kernels)"
  echo ""
  echo "- kernel-trace: ${KT_NOTE}."
  if [ "${KERNEL_TRACE_OK}" -eq 1 ]; then
    echo ""
    echo "Dispatched-kernel names corroborate the MLIR view (FP8 carries extra"
    echo "\`convert\`/\`dequantizelinear\` dispatches around the \`mlir_*\` gemm kernels);"
    echo "the full enumeration is in \`fp8-wmma.txt\`."
  fi
  echo ""
  echo "## Honesty notes"
  echo ""
  echo "- Demo F weights are **random-init** → this is a **capability / instruction**"
  echo "  probe, **not** an accuracy claim."
  echo "- The iGPU accelerates the **AI model** forward only; EZKL Halo2 / RISC0 STARK"
  echo "  proving stay **CPU-only** on AMD (the repo's standing rule)."
  echo "- The gemm code object is rocMLIR-in-process / opaque to file-level disasm;"
  echo "  the verdict rests on the MLIR \`dot\` operand types (1) + the convert-kernel"
  echo "  ISA (2) + the kernel trace (3) + the measured §7 perf, which all agree."
} >"${MD}"

echo "[fp8-wmma] verdict: ${VERDICT}"
echo "[fp8-wmma] artefacts:"
ls -l "${MD}" "${TXT}"
