#!/usr/bin/env bash
# npu-run.sh — OPT-IN, HEAVY XDNA2 NPU *dispatch* stage for the AMD Strix Halo
# (Ryzen AI MAX+ 395). This is the explicit graduation step beyond the read-only
# npu-probe.sh: it tries to take the NPU from DRIVER-READY -> ENUMERATED ->
# DISPATCH-OK by standing up the community user-space stack (Peano + MLIR-AIE +
# IRON via pip), enumerating the device through XRT, and running the IRON `axpy`
# example (and a small int8 GEMM toward BitNet prefill) on /dev/accel0.
#
#   bash poc/risc0-cartesi-step-demo/scripts/npu-run.sh
#
# UNLIKE npu-probe.sh (pure read-only, never-fail), this stage is allowed to do
# real work and MAY FAIL. It is engineered to be SAFE regardless:
#   * NEVER uses sudo, NEVER installs system packages, NEVER touches /opt, /lib,
#     /etc, the kernel, firmware, or any system path. ALL state lives under
#     $HOME/.cache/zkp-npu (a venv + a shallow mlir-aie checkout for examples).
#   * NEVER runs `xrt-smi validate`-style stress; only enumerate + the tiny axpy.
#   * ALWAYS records a result to artefacts/npu-dispatch.{log,json} (verdict +
#     precise blocked-reason) whether it succeeds or not, and ALWAYS exits 0 so
#     it can never break a run-on-halo.sh run when wired as an opt-in stage.
#
# HONESTY RULE (same as npu-probe.sh / amd-accel-detect.sh): even fully enabled,
# the NPU accelerates the AI MODEL (BitNet prefill / DEAAP embedding GEMM), NOT
# the ZK proof. r0vm STARK + Groth16 wrap stay CPU-only on AMD. This is a
# research track with NO end-to-end guarantee — a precise documented blocker is
# an acceptable, honest outcome. See reading-notes/path-d-npu-xdna2.md and
# docs/amd-strix-halo-acceleration.md.
#
# Verdict graduation written to npu-dispatch.json:
#   DRIVER-READY  -> kernel side ready (driver + accel node + firmware) but no
#                    user-space XRT amdxdna shim yet (== npu-probe baseline)
#   COMPILE-READY -> authoring toolchain (Peano + MLIR-AIE + IRON) stood up
#                    no-root; an xclbin CAN be compiled, but XRT cannot enumerate
#   ENUMERATED    -> `xrt-smi examine` / pyxrt see the NPU + firmware version
#   DISPATCH-OK   -> IRON axpy ran on the NPU (e.g. 160/160 PASS) + timing
#   BLOCKED-*     -> a precise, documented stop (see blocked_reason)
#
# Tunables (env): NPU_CACHE (default ~/.cache/zkp-npu), NPU_RUN_NO_INSTALL=1 to
# skip the pip step, NPU_AXPY_DIR to point at a prebuilt example dir, NPU_OFFLINE
# is auto-detected.
#
# OPT-IN MHA SPIKE (stage 5c, strictly env-gated, default ladder UNCHANGED):
#   NPU_MHA=1        enable the time-boxed fused-MHA spike (clone amd/IRON, try
#                    to build/run its MHA operator against the PINNED mlir_aie
#                    wheel, scrape PASS + GFLOPs/Latency). OFF by default.
#   NPU_MHA_DIR      point at a prebuilt amd/IRON checkout (skip the clone).
#   See reading-notes/path-d-npu-xdna2.md §9. A documented BLOCKED-MHA-KERNEL
#   negative result is an acceptable, expected outcome (no kernel authored).

set -u
umask 022

# ---------------------------------------------------------------------------
# Paths (resolve relative to this script so it works from any CWD).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DEMO_DIR="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
ART_DIR="${DEMO_DIR}/artefacts"
mkdir -p "${ART_DIR}"
LOG="${ART_DIR}/npu-dispatch.log"
JSON="${ART_DIR}/npu-dispatch.json"

NPU_CACHE="${NPU_CACHE:-${HOME}/.cache/zkp-npu}"
VENV="${NPU_CACHE}/venv"
SRC_DIR="${NPU_CACHE}/mlir-aie"          # shallow checkout for programming_examples
mkdir -p "${NPU_CACHE}"

PY="$(command -v python3 || true)"

# All stdout/stderr is tee'd into the log from here on.
exec > >(tee "${LOG}") 2>&1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "# [$(ts)] $*"; }

# ---------------------------------------------------------------------------
# Result accumulators (flat key=value, later serialised to JSON).
# ---------------------------------------------------------------------------
declare -A R
R[generated_utc]="$(ts)"
R[host]="$(hostname 2>/dev/null || echo unknown)"
R[kernel]="$(uname -r 2>/dev/null || echo unknown)"
R[cpu_model]="$(LC_ALL=C lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
R[stage_baseline]="DRIVER-READY"
R[verdict]="DRIVER-READY"
R[graduation]="DRIVER-READY"
R[compile_ready]="no"
R[enumerated]="no"
R[dispatch_ok]="no"
R[blocked_reason]=""
R[honesty]="NPU accelerates the AI model (BitNet prefill / DEAAP embedding), NOT the ZK proof; r0vm STARK + Groth16 wrap stay CPU-only on AMD. Research track, no end-to-end guarantee."

emit_json() {
    # Robust JSON emission via python3 (escapes everything). Falls back to a
    # hand-rolled object if no python3 is available.
    if [[ -n "${PY}" ]]; then
        {
            for k in "${!R[@]}"; do printf '%s\t%s\n' "$k" "${R[$k]}"; done
        } | "${PY}" -c '
import sys, json
d = {}
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line: continue
    k, _, v = line.partition("\t")
    d[k] = v
# Promote the attention-GEMM sweep (stored as a compact single-line JSON
# string in a private key) into a proper nested "attention_gemm_shapes" array.
aj = d.pop("_attention_gemm_json", "")
if aj:
    try:
        d["attention_gemm_shapes"] = json.loads(aj)
    except Exception:
        d["attention_gemm_shapes"] = aj
print(json.dumps(d, indent=2, sort_keys=True))
' > "${JSON}" 2>/dev/null && return 0
    fi
    # Fallback (no python): minimal, values assumed free of embedded quotes.
    {
        echo "{"
        local first=1
        for k in "${!R[@]}"; do
            [[ $first -eq 1 ]] && first=0 || echo ","
            printf '  "%s": "%s"' "$k" "${R[$k]//\"/\\\"}"
        done
        echo ""
        echo "}"
    } > "${JSON}"
}

FINISHED=0
finish() {
    [[ "${FINISHED}" == "1" ]] && exit 0
    FINISHED=1
    # Build the human-readable graduation ladder from the flags reached.
    local g="DRIVER-READY"
    [[ "${R[compile_ready]:-no}" == "yes" ]] && g="${g} -> COMPILE-READY"
    [[ "${R[enumerated]:-no}"   == "yes" ]] && g="${g} -> ENUMERATED"
    [[ "${R[dispatch_ok]:-no}"  == "yes" ]] && g="${g} -> DISPATCH-OK"
    [[ "${R[verdict]}" == BLOCKED-* ]] && g="${g} -> [${R[verdict]}]"
    R[graduation]="${g}"
    emit_json
    log "verdict=${R[verdict]} enumerated=${R[enumerated]} dispatch_ok=${R[dispatch_ok]}"
    [[ -n "${R[blocked_reason]}" ]] && log "blocked_reason=${R[blocked_reason]}"
    log "artefacts: ${LOG} ; ${JSON}"
    echo "######################################################################"
    echo "# XDNA2 NPU dispatch — VERDICT: ${R[verdict]}"
    [[ -n "${R[blocked_reason]}" ]] && echo "#   blocked_reason: ${R[blocked_reason]}"
    echo "#   (research track; NPU accelerates the AI model, NOT the ZK proof)"
    echo "######################################################################"
    # Opt-in research stage: never break the parent run.
    exit 0
}
trap finish EXIT

echo "######################################################################"
echo "# AMD XDNA2 NPU dispatch run ($(ts)) — OPT-IN / heavy / research"
echo "#   host   : ${R[host]}"
echo "#   kernel : ${R[kernel]}"
echo "#   cpu    : ${R[cpu_model]}"
echo "#   cache  : ${NPU_CACHE}   (no sudo, no system writes)"
echo "######################################################################"

# ===========================================================================
# Stage 1 — kernel-side readiness (read-only; mirrors npu-probe baseline).
# ===========================================================================
log "stage 1/5: kernel-side readiness (read-only)"
R[amdxdna_loaded]="$(lsmod 2>/dev/null | grep -q '^amdxdna' && echo yes || echo no)"
ACCEL_NODE=""
for n in /dev/accel/accel* /dev/accel*; do
    [[ -e "$n" && "$n" != "/dev/accel" ]] && ACCEL_NODE="$n" && break
done
R[accel_node]="${ACCEL_NODE:-<none>}"
R[accel_readable]="$([[ -r "${ACCEL_NODE:-/nonexistent}" || -w "${ACCEL_NODE:-/nonexistent}" ]] && echo yes || echo no)"
R[pci]="$(lspci -nn 2>/dev/null | grep -iE '1022:17f0' | head -1 | sed 's/^[[:space:]]*//' || true)"
R[fw_version]="$(cat /sys/class/accel/accel0/device/fw_version 2>/dev/null || echo n/a)"
R[render_group]="$(id -nG 2>/dev/null | tr ' ' '\n' | grep -qx render && echo yes || echo no)"
echo "#   amdxdna loaded : ${R[amdxdna_loaded]}"
echo "#   accel node     : ${R[accel_node]} (readable=${R[accel_readable]}, render-group=${R[render_group]})"
echo "#   NPU firmware   : fw_version=${R[fw_version]}"
echo "#   PCI            : ${R[pci]:-not seen}"

if [[ "${R[amdxdna_loaded]}" != "yes" || -z "${ACCEL_NODE}" ]]; then
    R[verdict]="BLOCKED-NO-DRIVER"
    R[blocked_reason]="amdxdna driver / /dev/accel node not present — this box is not a driver-ready XDNA2 host."
    finish
fi

# ===========================================================================
# Stage 2 — is a usable user-space XRT amdxdna shim already present? (no-root)
# ===========================================================================
log "stage 2/5: locate user-space XRT runtime + amdxdna shim"
XRT_SMI=""
for c in "$(command -v xrt-smi 2>/dev/null)" /opt/xilinx/xrt/bin/xrt-smi "${NPU_CACHE}"/xrt/opt/xilinx/xrt/bin/xrt-smi; do
    [[ -n "$c" && -x "$c" ]] && XRT_SMI="$c" && break
done
XDNA_SHIM="$(ls /opt/xilinx/xrt/lib/libxrt_driver_xdna.so* "${NPU_CACHE}"/xrt/opt/xilinx/xrt/lib/libxrt_driver_xdna.so* 2>/dev/null | head -1 || true)"
R[xrt_smi]="${XRT_SMI:-<absent>}"
R[xdna_shim]="${XDNA_SHIM:-<absent>}"
echo "#   xrt-smi        : ${R[xrt_smi]}"
echo "#   amdxdna shim   : ${R[xdna_shim]}"

# ===========================================================================
# Stage 3 — stand up the AUTHORING toolchain no-root (Peano+MLIR-AIE+IRON).
#           This is always safe (a user-space venv) and proves COMPILE-READY.
# ===========================================================================
log "stage 3/5: authoring toolchain (Peano + MLIR-AIE + IRON) in venv"
NPU_OFFLINE=0
if ! timeout 8 bash -c ': >/dev/tcp/github.com/443' 2>/dev/null; then NPU_OFFLINE=1; fi
R[network]="$([[ $NPU_OFFLINE -eq 0 ]] && echo online || echo offline)"

have_aie() { [[ -x "${VENV}/bin/python" ]] && "${VENV}/bin/python" -c 'import aie' >/dev/null 2>&1; }

if [[ "${NPU_RUN_NO_INSTALL:-0}" != "1" ]] && ! have_aie; then
    if [[ $NPU_OFFLINE -eq 1 ]]; then
        log "  offline: skipping wheel install"
    elif [[ -z "${PY}" ]]; then
        log "  no python3 found: skipping wheel install"
    else
        log "  creating venv + pip install mlir_aie + llvm-aie (heavy, ~360MB)"
        "${PY}" -m venv "${VENV}" >/dev/null 2>&1 || true
        "${VENV}/bin/python" -m pip install --upgrade pip -q >/dev/null 2>&1 || true
        "${VENV}/bin/pip" install -q \
            mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-2 \
            llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
            >/dev/null 2>&1 || log "  (pip install reported a non-zero status; continuing)"
    fi
fi

AIE_BIN=""; PEANO_DIR=""
if have_aie; then
    AIE_BIN="$("${VENV}/bin/python" -c 'import mlir_aie,os;print(os.path.join(os.path.dirname(mlir_aie.__file__),"bin"))' 2>/dev/null || true)"
    [[ -d "$AIE_BIN" ]] || AIE_BIN="$(dirname "$(find "${VENV}" -name aie-opt -type f 2>/dev/null | head -1)" 2>/dev/null || true)"
    PEANO_DIR="$(dirname "$(find "${VENV}" -path '*llvm-aie*/bin/clang' 2>/dev/null | head -1)" 2>/dev/null || true)"
fi
R[authoring_present]="$(have_aie && echo yes || echo no)"
R[mlir_aie_version]="$([[ -n "$AIE_BIN" && -x "$AIE_BIN/aie-opt" ]] && "$AIE_BIN/aie-opt" --version 2>/dev/null | sed -n 's/^aie-opt //p' | head -1 || echo n/a)"
R[peano_version]="$([[ -x "${VENV}/bin/pip" ]] && "${VENV}/bin/pip" show llvm-aie 2>/dev/null | sed -n 's/^Version: //p' | head -1 || echo n/a)"
echo "#   authoring      : ${R[authoring_present]} (mlir-aie=${R[mlir_aie_version]}, peano=${R[peano_version]})"

# Does the no-root stack include a runtime able to enumerate? The mlir_aie wheel
# ships authoring + libxrt_coreutil + the compile-side _xrt binding, but NOT
# pyxrt and NOT the amdxdna device shim — so by itself it cannot enumerate.
R[pyxrt_runtime]="$([[ "${R[authoring_present]}" == "yes" ]] && ("${VENV}/bin/python" -c 'import pyxrt' >/dev/null 2>&1 && echo yes || echo no) || echo n/a)"
echo "#   pyxrt runtime  : ${R[pyxrt_runtime]}"

if [[ "${R[authoring_present]}" == "yes" ]]; then
    R[verdict]="COMPILE-READY"
    R[compile_ready]="yes"
fi

# ===========================================================================
# Stage 4 — ENUMERATE the NPU through XRT.
# ===========================================================================
log "stage 4/5: enumerate the NPU through XRT"
ENUM_OK=0
if [[ -n "${XRT_SMI}" ]]; then
    # shellcheck disable=SC1091
    [[ -f /opt/xilinx/xrt/setup.sh ]] && source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || true
    EX="$("${XRT_SMI}" examine 2>&1 || true)"
    echo "${EX}" | sed 's/^/#   xrt-smi> /' | head -40
    if echo "${EX}" | grep -qiE 'NPU|RyzenAI|Ryzen AI|17f0|accel'; then
        ENUM_OK=1
        R[xrt_examine_fw]="$(echo "${EX}" | sed -n 's/.*[Ff]irmware [Vv]ersion[[:space:]]*:[[:space:]]*//p' | head -1)"
    fi
elif [[ "${R[pyxrt_runtime]}" == "yes" ]]; then
    if "${VENV}/bin/python" - <<'PY' >/tmp/npu_enum.out 2>&1; then ENUM_OK=1; fi
import sys
try:
    import pyxrt
    n = pyxrt.system().enumerate_devices() if hasattr(pyxrt, "system") else 1
    d = pyxrt.device(0)
    print("pyxrt device(0) OK:", d.get_info(pyxrt.xrt_info_device.name) if hasattr(pyxrt,'xrt_info_device') else "device")
    sys.exit(0)
except Exception as e:
    print("pyxrt enumerate ERR:", repr(e))
    sys.exit(1)
PY
    sed 's/^/#   pyxrt> /' /tmp/npu_enum.out 2>/dev/null | head -20
fi

if [[ ${ENUM_OK} -eq 1 ]]; then
    R[enumerated]="yes"
    R[verdict]="ENUMERATED"
    echo "#   ENUMERATED: XRT sees the NPU (firmware=${R[xrt_examine_fw]:-n/a})"
else
    R[enumerated]="no"
    # Precise, evidence-based blocked-reason for the no-shim case.
    if [[ "${R[xrt_smi]}" == "<absent>" && "${R[xdna_shim]}" == "<absent>" ]]; then
        R[blocked_reason]="No user-space XRT amdxdna shim (libxrt_driver_xdna.so) and no xrt-smi present. The kernel side is fully ready (amdxdna loaded, ${R[accel_node]} present & ${R[accel_readable]}-readable via render group, firmware ${R[fw_version]} loaded), but XRT cannot enumerate /dev/accel0 without the device shim. Installing it requires root: (a) non-interactive sudo needs a password here; (b) amd/xdna-driver ships NO prebuilt release .deb to extract no-root; (c) Ubuntu-archive XRT is 202210.2.13.466 (2022, pre-NPU: no xrt-smi, no xdna shim); (d) the no-root mlir_aie wheel provides authoring (aie-opt/aie-translate/aiecc/bootgen + Peano) + libxrt_coreutil + the compile-side _xrt binding, but NO pyxrt runtime and NO xdna shim. Graduation halts at COMPILE-READY; ENUMERATE/DISPATCH need a root-built XRT+xrt_plugin(amdxdna) matched to the kernel driver."
    else
        R[blocked_reason]="XRT present but did not enumerate the NPU (driver/firmware/XRT-shim version mismatch — check dmesg for 'Incompatible firmware protocol'). xrt-smi=${R[xrt_smi]} shim=${R[xdna_shim]} fw=${R[fw_version]}."
    fi
    [[ -n "${R[blocked_reason]}" ]] && R[verdict]="BLOCKED-ENUMERATION"
    finish
fi

# ===========================================================================
# Stage 5 — DISPATCH: IRON axpy (known-passing on npu5) + a small int8 GEMM.
#           Only reachable once ENUMERATED. Builds an xclbin via Peano and runs
#           it on /dev/accel0 through XRT. Captures pass/fail + timing.
# ===========================================================================
log "stage 5/5: dispatch IRON axpy (+ int8 GEMM) on the NPU"

# Bring the authoring + runtime env onto PATH for aiecc / make. The venv MUST be
# first so `python3` resolves to the one with `aie` installed (the example's
# Makefile shells out to a bare `python3`) and so a venv-provided `cmake` wins.
export PATH="${VENV}/bin:${AIE_BIN}:${PEANO_DIR}:${PATH}"
export PEANO_INSTALL_DIR="$(dirname "${PEANO_DIR}" 2>/dev/null)"
[[ -f /opt/xilinx/xrt/setup.sh ]] && source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || true

# This box is XDNA2 (aie2p / RyzenAI-npu5): build the examples for the `npu2`
# device family (the example Makefiles default to npu1 unless NPU2=1 is set).
export NPU2=1

# The IRON example host testbenches need CMake >= 3.30; if the system cmake is
# older, drop a modern one into the venv (no root). The venv is first on PATH.
if ! cmake --version 2>/dev/null | awk 'NR==1{n=split($3,v,"."); exit !(v[1]>3 || (v[1]==3 && v[2]>=30))}'; then
    log "  system cmake < 3.30; installing a modern cmake into the venv (no root)"
    "${VENV}/bin/pip" install -q cmake >/dev/null 2>&1 || log "  (cmake pip install reported non-zero; continuing)"
fi

# Acquire the programming_examples (shallow) if not pointed at a prebuilt dir.
AXPY_DIR="${NPU_AXPY_DIR:-${SRC_DIR}/programming_examples/basic/vector_scalar_add}"
if [[ ! -d "${SRC_DIR}/.git" && $NPU_OFFLINE -eq 0 ]]; then
    log "  cloning Xilinx/mlir-aie (shallow) for programming_examples"
    git clone --depth 1 https://github.com/Xilinx/mlir-aie "${SRC_DIR}" >/dev/null 2>&1 || \
        log "  (clone failed; set NPU_AXPY_DIR to a prebuilt example)"
fi

# Pin the example tree to the SAME commit as the installed mlir_aie wheel. The
# IRON Python API (aie.iron) drifts fast, so the repo HEAD examples will not
# import against an older wheel (e.g. `CompileTime` was added after the pinned
# wheel). The wheel records its commit in the local-version segment
# (`Version: 0.0.1.<date>+<sha7>`). Best-effort + logged: leaves the current
# checkout if the commit cannot be fetched (e.g. offline).
if [[ -d "${SRC_DIR}/.git" && "${NPU_AXPY_DIR:-}" == "" ]]; then
    WHEEL_SHA="$("${VENV}/bin/pip" show mlir_aie 2>/dev/null | sed -n 's/^Version:.*+//p' | tr -d '[:space:]')"
    if [[ -n "${WHEEL_SHA}" ]]; then
        git -C "${SRC_DIR}" cat-file -e "${WHEEL_SHA}^{commit}" 2>/dev/null || \
            git -C "${SRC_DIR}" fetch --depth 1 origin "${WHEEL_SHA}" >/dev/null 2>&1 || true
        if git -C "${SRC_DIR}" checkout -q "${WHEEL_SHA}" 2>/dev/null; then
            log "  pinned mlir-aie examples to wheel commit ${WHEEL_SHA}"
        else
            log "  (could not pin examples to wheel commit ${WHEEL_SHA}; using current checkout)"
        fi
    fi
fi

run_example() {
    # $1 = example dir, $2 = label. Captures output + a wall-clock for `make run`.
    local dir="$1" label="$2" out rc t0 t1
    [[ -d "$dir" ]] || { echo "#   ${label}: example dir not found ($dir)"; return 2; }
    ( cd "$dir" && make clean >/dev/null 2>&1 || true; make >/dev/null 2>&1 ) || \
        { echo "#   ${label}: build (xclbin) failed"; return 3; }
    t0=$(date +%s.%N)
    out="$(cd "$dir" && timeout 180 make run 2>&1 || true)"
    t1=$(date +%s.%N)
    echo "${out}" | sed "s/^/#   ${label}> /" | tail -25
    R["${label}_wall_s"]="$(awk "BEGIN{printf \"%.3f\", ${t1}-${t0}}")"
    # Capture matmul-style on-NPU timing if the testbench reported it (the GEMM
    # example prints "Avg NPU matmul time: <x>us." / "Avg NPU gflops: <y>").
    R["${label}_npu_us"]="$(echo "${out}"   | sed -n 's/.*Avg NPU matmul time:[[:space:]]*\([0-9.]*\)us.*/\1/p' | tail -1)"
    R["${label}_gflops"]="$(echo "${out}"   | sed -n 's/.*Avg NPU gflops:[[:space:]]*\([0-9.]*\).*/\1/p'        | tail -1)"
    R["${label}_min_us"]="$(echo "${out}"   | sed -n 's/.*Min NPU matmul time:[[:space:]]*\([0-9.]*\)us.*/\1/p' | tail -1)"
    R["${label}_max_gflops"]="$(echo "${out}" | sed -n 's/.*Max NPU gflops:[[:space:]]*\([0-9.]*\).*/\1/p'      | tail -1)"
    if echo "${out}" | grep -qiE 'PASS!?|test passed|results correct'; then
        R["${label}_result"]="PASS"
        R["${label}_counts"]="$(echo "${out}" | grep -oE '[0-9]+/[0-9]+' | tail -1)"
        return 0
    fi
    R["${label}_result"]="FAIL"
    return 1
}

# ---------------------------------------------------------------------------
# OPT-IN fused-MHA spike (stage 5c). Strictly gated by NPU_MHA=1 so the default
# ladder is byte-for-byte unchanged. Time-boxed with hard `timeout`s; a precise
# BLOCKED-MHA-KERNEL negative result is an acceptable, expected outcome. We do
# NOT author a kernel and do NOT mutate the DISPATCH-OK venv: amd/IRON is exposed
# purely via PYTHONPATH and we never `pip install` its requirements (which would
# UPGRADE the pinned mlir_aie wheel). All state lives under ${NPU_CACHE}.
# ---------------------------------------------------------------------------
run_mha_spike() {
    log "stage 5c (OPT-IN, NPU_MHA=1): fused-MHA spike via amd/IRON, time-boxed"
    R[mha_attempted]="yes"
    R[mha_repo]="amd/IRON (SEPARATE repo; NOT the default Xilinx/mlir-aie checkout)"
    R[mha_operator]="iron/operators/mha (fused matmul+softmax+matmul; aie_kernels/aie2p/mha.cc)"
    R[mha_target_shape]="MiniLM: d_model=384, 12 heads, d_head=32, seq=256"
    R[mha_honesty]="NPU accelerates the model forward only, never the proof."
    R[mha_pinned_wheel]="$("${VENV}/bin/pip" show mlir_aie 2>/dev/null | sed -n 's/^Version: //p' | head -1)"

    local IRON_SRC="${NPU_MHA_DIR:-${NPU_CACHE}/IRON}"
    # Clone amd/IRON (shallow, time-boxed) unless a prebuilt dir was supplied.
    if [[ ! -d "${IRON_SRC}/.git" && -z "${NPU_MHA_DIR:-}" ]]; then
        if [[ $NPU_OFFLINE -eq 1 ]]; then
            R[mha_verdict]="BLOCKED-MHA-KERNEL"
            R[mha_blocked_reason]="offline: cannot clone amd/IRON (separate, not-cloned repo)."
            echo "#   mha: offline, cannot clone amd/IRON"
            return 1
        fi
        log "  cloning amd/IRON (shallow, separate repo) -> ${IRON_SRC}"
        timeout 240 git clone --depth 1 https://github.com/amd/IRON.git "${IRON_SRC}" >/dev/null 2>&1 || \
            log "  (amd/IRON clone failed/timed out)"
    fi
    if [[ ! -d "${IRON_SRC}/iron" ]]; then
        R[mha_verdict]="BLOCKED-MHA-KERNEL"
        R[mha_blocked_reason]="amd/IRON checkout not available at ${IRON_SRC} (clone failed/timed out within the time box)."
        echo "#   mha: amd/IRON checkout unavailable"
        return 1
    fi
    R[mha_repo_commit]="$(git -C "${IRON_SRC}" log -1 --format='%h %ci' 2>/dev/null | head -1)"
    # The wheel version amd/IRON HEAD itself pins (drift signal vs our pinned wheel).
    R[mha_iron_requires_wheel]="$(sed -n 's/^[[:space:]]*mlir_aie==/mlir_aie==/p' "${IRON_SRC}/requirements.txt" 2>/dev/null | head -1)"
    echo "#   mha: IRON ${R[mha_repo_commit]:-?}; IRON requires ${R[mha_iron_requires_wheel]:-?}; pinned wheel mlir_aie ${R[mha_pinned_wheel]:-?}"

    # Bring XRT (for pyxrt, if present) onto the env, then probe import->build
    # against the PINNED wheel under a hard timeout. PYTHONPATH only (no pip).
    [[ -f /opt/xilinx/xrt/setup.sh ]] && source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || true
    local probe="${NPU_CACHE}/mha_probe.out"
    PYTHONPATH="${IRON_SRC}:${PYTHONPATH:-}" timeout 150 "${VENV}/bin/python" - "${IRON_SRC}" <<'PY' > "${probe}" 2>&1 || true
import sys, traceback
iron_src = sys.argv[1]
def kv(k, v): print(f"{k}={v}")
# 1) IRON MHA operator import against the pinned wheel.
try:
    from iron.operators.mha.op import MHA
    kv("import", "OK")
except Exception as e:
    kv("import", "FAIL")
    kv("import_err", repr(e))
    sys.exit(0)
# 2) Try to construct the requested MiniLM-shaped op (d_head=32).
try:
    MHA(num_heads=12, seq_len=256, d=32, num_KV_heads=0)
    kv("construct_minilm", "OK")
except Exception as e:
    kv("construct_minilm", "FAIL")
    kv("construct_err", repr(e))
PY
    sed 's/^/#   mha-probe> /' "${probe}" 2>/dev/null | head -20
    local imp imp_err con con_err
    imp="$(sed -n 's/^import=//p' "${probe}" | head -1)"
    imp_err="$(sed -n 's/^import_err=//p' "${probe}" | head -1)"
    con="$(sed -n 's/^construct_minilm=//p' "${probe}" | head -1)"
    con_err="$(sed -n 's/^construct_err=//p' "${probe}" | head -1)"

    # Static, code-grounded constraints (true regardless of which gate trips first).
    local pyver xrt_pyxrt
    pyver="$("${VENV}/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
    xrt_pyxrt="$(ls /opt/xilinx/xrt/python/pyxrt*.so 2>/dev/null | head -1 | sed 's#.*/##')"
    local consts="MHA op constraints (iron/operators/mha/op.py): d(head_dim)==64 ONLY (raise ValueError if d!=64, so MiniLM d_head=32 is rejected); bf16-only (mha.cc built -Dbf16_bf16_ONLY, bf16 emulated via bfp16 + bf16 softmax.cc), NOT the int8 path validated at DISPATCH-OK; test config is seq_len=16384 across 8 AIE pipeline columns, not seq=256."
    local drift="IRON HEAD (${R[mha_repo_commit]:-?}) pins ${R[mha_iron_requires_wheel]:-?} (latest-wheels-3), newer than the DISPATCH-OK-validated pinned wheel mlir_aie ${R[mha_pinned_wheel]:-?} (latest-wheels-2)."
    local pyxrt_note="IRON HEAD restructured MHA into an iron.common MLIROperator framework whose import hard-requires the pyxrt runtime; the pinned mlir_aie wheel ships NO pyxrt and /opt/xilinx/xrt ships pyxrt only as ${xrt_pyxrt:-a cp312 .so} (ABI-incompatible with the wheel's Python ${pyver:-3.13} venv)."

    if [[ "${imp}" == "OK" && "${con}" == "OK" ]]; then
        # Imports + MiniLM op constructed against the pinned wheel: attempt the
        # actual build/run (pytest harness) under a hard timeout, scrape numbers.
        log "  mha: IRON MHA imported + MiniLM op constructed; attempting build/run (time-boxed)"
        local out
        out="$(cd "${IRON_SRC}" && PYTHONPATH="${IRON_SRC}" timeout 240 "${VENV}/bin/python" -m pytest -q iron/operators/mha/test.py 2>&1 || true)"
        echo "${out}" | sed 's/^/#   mha-run> /' | tail -25
        R[mha_latency_us]="$(echo "${out}" | sed -n 's/.*Latency (us):[[:space:]]*\([0-9.]*\).*/\1/p' | tail -1)"
        R[mha_bandwidth_gbps]="$(echo "${out}" | sed -n 's/.*Effective Bandwidth:[[:space:]]*\([0-9.eE+-]*\).*/\1/p' | tail -1)"
        if echo "${out}" | grep -qiE '1 passed|test passed|PASS'; then
            R[mha_result]="PASS"
            R[mha_verdict]="MHA-DISPATCH-OK"
            return 0
        fi
        R[mha_result]="FAIL"
        R[mha_verdict]="BLOCKED-MHA-KERNEL"
        R[mha_blocked_reason]="IRON MHA imported + constructed against the pinned wheel but the build/run did not PASS within the time box. ${consts} ${drift} Negative result; ${R[mha_honesty]}"
        return 1
    fi

    # Otherwise: classify the precise gate that blocked, with the full chain.
    R[mha_result]="BLOCKED"
    R[mha_verdict]="BLOCKED-MHA-KERNEL"
    if [[ "${imp}" == "FAIL" ]]; then
        R[mha_blocked_reason]="IRON's fused-MHA operator does not import against the pinned mlir_aie wheel (${R[mha_pinned_wheel]:-?}, latest-wheels-2). RUNTIME/API drift: ${pyxrt_note} Observed: ${imp_err:-ImportError}. WHEEL drift: ${drift} Even past imports, ${consts} Authoring a fused kernel from scratch is out of scope. Negative result; ${R[mha_honesty]}"
    elif [[ "${con}" == "FAIL" ]]; then
        R[mha_blocked_reason]="IRON's fused-MHA operator imports but REJECTS the MiniLM shape: ${con_err:-ValueError}. ${consts} WHEEL drift: ${drift} Authoring a fused kernel from scratch is out of scope. Negative result; ${R[mha_honesty]}"
    else
        R[mha_blocked_reason]="IRON MHA spike did not complete within the time box (no clear import/construct signal). ${pyxrt_note} ${drift} ${consts} Negative result; ${R[mha_honesty]}"
    fi
    return 1
}

if run_example "${AXPY_DIR}" "axpy"; then
    R[dispatch_ok]="yes"
    R[verdict]="DISPATCH-OK"
    echo "#   DISPATCH-OK: axpy ${R[axpy_counts]:-} ${R[axpy_result]} in ${R[axpy_wall_s]}s"
    # Secondary attempt toward BitNet prefill: a small int8 GEMM. Non-fatal.
    # int8 in/out on the aie2p array, with a few warmup+timed iters so the
    # testbench reports avg/min NPU time + GFLOPs (BitNet-style ternary/int8).
    GEMM_DIR="${NPU_GEMM_DIR:-${SRC_DIR}/programming_examples/basic/matrix_multiplication/whole_array}"
    export dtype_in="${NPU_GEMM_DTYPE_IN:-i8}" dtype_out="${NPU_GEMM_DTYPE_OUT:-i8}"
    export runargs="${NPU_GEMM_RUNARGS:--v 1 --warmup 5 --iters 50}"
    if run_example "${GEMM_DIR}" "int8_gemm"; then
        echo "#   int8 GEMM (512x512x512 ${dtype_in}): PASS  avg ${R[int8_gemm_npu_us]:-?}us / ${R[int8_gemm_gflops]:-?} GFLOPs (min ${R[int8_gemm_min_us]:-?}us, peak ${R[int8_gemm_max_gflops]:-?} GFLOPs)"
    else
        log "  int8 GEMM attempt did not pass (axpy DISPATCH-OK still stands)"
    fi
    unset dtype_in dtype_out runargs

    # =======================================================================
    # Stage 5b — ATTENTION / MiniLM-shaped int8 GEMM sweep.
    #   Reuses the SAME proven `whole_array` int8 GEMM harness (NO new MLIR-AIE
    #   kernel): only M/K/N are varied to match the matmul shapes of a
    #   transformer encoder layer — X.W projections, QK^T, A.V and the FFN —
    #   for MiniLM (d_model=384, 12 heads, d_head=32) and BERT-base
    #   (d_model=768, d_head=64). All shapes are mapped onto the harness'
    #   tiling constraints for the default tile (m=k=n=32, n_aie_cols=4, i8):
    #     M % 128 == 0,  K % 32 == 0,  N % 128 == 0.
    #   Every projection / attention-score / FFN dim already satisfies these
    #   ("real"). The ONLY non-native dim is A.V's N = d_head (32 / 64), which
    #   is below the 128-column minimum of the 4-column array; we pad it up to
    #   128 and label that row a PROXY (running a new attention-specific kernel
    #   that supports N<128 is explicitly out of scope here).
    #   Honesty rule unchanged: this accelerates the MODEL FORWARD only, never
    #   the ZK proof.
    # Tunables: NPU_NO_ATTENTION_GEMM=1 to skip; NPU_ATTN_RUNARGS to override.
    if [[ "${R[dispatch_ok]}" == "yes" && "${NPU_NO_ATTENTION_GEMM:-0}" != "1" ]]; then
        log "stage 5b: attention/MiniLM-shaped int8 GEMM sweep (whole_array harness)"
        export dtype_in="${NPU_GEMM_DTYPE_IN:-i8}" dtype_out="${NPU_GEMM_DTYPE_OUT:-i8}"
        # Fewer iters than the 512^3 baseline: the testbench runs a full CPU
        # reference matmul per iteration (verify), which dominates and can time
        # out on the larger FFN shapes when the box is under concurrent CPU
        # load. The per-iter min-latency (peak GFLOPs) is unaffected.
        export runargs="${NPU_ATTN_RUNARGS:--v 1 --warmup 3 --iters 10}"

        # label | M | K | N | kind(real/proxy) | role
        ATTN_SHAPES=(
            "mha_qkv_proj_d384_t256|256|384|384|real|X.Wq/k/v & attn-out projection (MiniLM d_model=384), T=256 tokens"
            "mha_qkt_d384_t256|256|32|256|real|QK^T per head (d_head=32), T=256"
            "mha_av_dhead_t256_proxy|256|256|128|proxy|A.V per head (true N=d_head 32/64; padded to 128 = min N for the 4-col array), T=256"
            "mha_ffn1_d384_t256|256|384|1536|real|FFN1 384->1536, T=256"
            "mha_ffn2_d384_t256|256|1536|384|real|FFN2 1536->384, T=256"
            "mha_qkv_proj_d384_t1024|1024|384|384|real|X.W projection (MiniLM d_model=384), T=1024"
            "mha_qkt_d384_t1024|1024|32|1024|real|QK^T per head (d_head=32), T=1024"
            "mha_ffn1_d384_t1024|1024|384|1536|real|FFN1 384->1536, T=1024"
            "mha_qkv_proj_d768_t256|256|768|768|real|X.W projection (BERT-base d_model=768), T=256"
            "mha_qkt_d768_t256|256|64|256|real|QK^T per head (d_head=64), T=256"
            "mha_ffn1_d768_t256|256|768|3072|real|FFN1 768->3072, T=256"
        )
        attn_rows="$(mktemp 2>/dev/null || echo /tmp/npu_attn_rows.$$)"
        : > "${attn_rows}"
        attn_peak="0"
        for spec in "${ATTN_SHAPES[@]}"; do
            IFS='|' read -r a_label M K N a_kind a_role <<< "${spec}"
            export M K N
            run_example "${GEMM_DIR}" "${a_label}" || true
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${a_label}" "${M}" "${K}" "${N}" "${a_kind}" \
                "${R[${a_label}_result]:-FAIL}" \
                "${R[${a_label}_npu_us]:-}" "${R[${a_label}_gflops]:-}" \
                "${R[${a_label}_min_us]:-}" "${R[${a_label}_max_gflops]:-}" \
                "${a_role}" >> "${attn_rows}"
            echo "#   ${a_label} (${M}x${K}x${N} ${a_kind}): ${R[${a_label}_result]:-FAIL}  avg ${R[${a_label}_gflops]:-?} GFLOPs, peak ${R[${a_label}_max_gflops]:-?} GFLOPs (min ${R[${a_label}_min_us]:-?}us)"
            # Track the best peak; keep top-level JSON tidy by dropping the
            # per-shape flat keys (the structured array below preserves them).
            pk="${R[${a_label}_max_gflops]:-}"
            [[ -n "${pk}" ]] && awk "BEGIN{exit !(${pk}>${attn_peak})}" && attn_peak="${pk}"
            for suf in result counts wall_s npu_us gflops min_us max_gflops; do
                unset "R[${a_label}_${suf}]"
            done
            unset M K N
        done
        unset dtype_in dtype_out runargs

        # Serialise the sweep into a compact single-line JSON array (promoted to
        # the nested "attention_gemm_shapes" field by emit_json).
        if [[ -n "${PY}" && -s "${attn_rows}" ]]; then
            R[_attention_gemm_json]="$("${PY}" -c '
import sys, json
rows = []
for line in open(sys.argv[1]):
    line = line.rstrip("\n")
    if not line:
        continue
    f = line.split("\t")
    f += [""] * (11 - len(f))
    rows.append({
        "label": f[0], "M": int(f[1]), "K": int(f[2]), "N": int(f[3]),
        "kind": f[4], "result": f[5],
        "avg_us": f[6], "avg_gflops": f[7],
        "min_us": f[8], "peak_gflops": f[9], "role": f[10],
    })
print(json.dumps(rows, separators=(",", ":")))
' "${attn_rows}" 2>/dev/null)"
            R[attention_gemm_count]="$(wc -l < "${attn_rows}" | tr -d ' ')"
            R[attention_gemm_peak_gflops]="${attn_peak}"
            R[attention_gemm_note]="int8 GEMM driven at attention/MiniLM matmul shapes via the proven whole_array harness (no new MLIR kernel). 'real' = dim natively tileable (M%128,K%32,N%128); 'proxy' = A.V N=d_head padded 32/64 -> 128. peak_gflops (min-latency) is the representative NPU figure; avg is depressed under concurrent CPU load. Model forward only, never the proof."
        fi
        rm -f "${attn_rows}" 2>/dev/null || true
    fi

    # =======================================================================
    # Stage 5c — OPT-IN fused-MHA spike (amd/IRON). Strictly env-gated by
    #   NPU_MHA=1 so the default ladder behaves identically; records its own
    #   mha_* fields (and an mha_verdict) WITHOUT touching the top-level
    #   DISPATCH-OK verdict. A documented BLOCKED-MHA-KERNEL negative result is
    #   the expected, acceptable outcome (no kernel authored). Model fwd only.
    # =======================================================================
    if [[ "${NPU_MHA:-0}" == "1" ]]; then
        run_mha_spike || true
        echo "#   MHA spike: ${R[mha_verdict]:-?} (result=${R[mha_result]:-?}, latency=${R[mha_latency_us]:-n/a}us, bw=${R[mha_bandwidth_gbps]:-n/a}GB/s)"
        [[ -n "${R[mha_blocked_reason]:-}" ]] && echo "#     blocker: ${R[mha_blocked_reason]}"
    fi
else
    R[verdict]="BLOCKED-DISPATCH"
    R[blocked_reason]="NPU enumerated but the IRON axpy example did not build/run/PASS (see log). Likely an xclbin compile or XRT host mismatch."
fi

finish
