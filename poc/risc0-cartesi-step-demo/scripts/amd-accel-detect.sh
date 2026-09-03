#!/usr/bin/env bash
# amd-accel-detect.sh — detect the AMD Strix Halo (Ryzen AI MAX+ 395) compute
# fabric and export the right env for each workload. SAFE TO `source` (it never
# calls `exit` and tolerates `set -euo pipefail` in the caller).
#
#   source poc/risc0-cartesi-step-demo/scripts/amd-accel-detect.sh
#   bash   poc/risc0-cartesi-step-demo/scripts/amd-accel-detect.sh   # just the banner
#
# HONESTY RULE (printed in the banner too): RISC0 r0vm STARK proving and the
# Groth16 wrap are CPU-ONLY on AMD — there is NO ROCm/Vulkan RISC0 prover. The
# iGPU (Radeon 8060S / gfx1151) and the XDNA2 NPU accelerate the AI MODEL
# (DEAAP embedding / LoRA agent, EZKL exec), NOT the proof. So we tune two
# different things:
#   * proving  -> RAYON_NUM_THREADS (all Zen 5 threads) + the 128 GB unified RAM
#   * inference-> ROCm 7.2+ / Vulkan backend selection for the AI containers
#
# Exports (only if not already set by the caller):
#   RAYON_NUM_THREADS         logical CPU count (r0vm/EZKL CPU proving)
#   HSA_OVERRIDE_GFX_VERSION  per-arch override (gfx1151: 11.5.1 on kernel>=6.19)
#   HSA_ENABLE_SDMA           per-arch (gfx1151: 0 for Strix Halo stability)
#   AI_GPU_BACKEND            rocm | vulkan | cpu  (for the AI-inference stacks)
#   AMD_UNIFIED_RAM_GB        detected total RAM (unified LPDDR5X)
#   ROCM_VERSION              parsed ROCm release (e.g. 7.2.3) or "unknown"
#
# PORTABILITY: the HSA_OVERRIDE_GFX_VERSION / HSA_ENABLE_SDMA / recommended
# kernel / recommended ROCm values are looked up from a gfx-arch table (see
# `gfx_arch_profile` below) instead of being hardcoded for gfx1151, so the same
# script behaves sanely on other AMD parts. The gfx1151 row is authoritative for
# this box; the gfx1100/gfx1101/gfx1102/gfx942 rows are DEFAULTS distilled from
# public ROCm docs (clearly marked) for when this script is reused elsewhere.
#
# ---------------------------------------------------------------------------
# CI MATRIX SKELETON (plan §C2 — needs hardware, NOT wired here on purpose).
# ---------------------------------------------------------------------------
# A true ROCm-version / gfx-arch portability matrix needs MULTIPLE physical
# machines, so it cannot be exercised on a single box. When self-hosted runners
# with the labels below exist, add a sibling workflow (NOT in this file; do not
# break the existing .github/workflows/gpu-cpu-equality.yml single-box gate):
#
#   # .github/workflows/rocm-portability-matrix.yml  (skeleton — add when HW lands)
#   strategy:
#     matrix:
#       runner: [amd-gfx1151, amd-gfx1100, rocm-7.2]   # self-hosted labels
#   runs-on: ${{ matrix.runner }}
#   steps:
#     - uses: actions/checkout@v4
#     - run: source poc/risc0-cartesi-step-demo/scripts/amd-accel-detect.sh   # arch+ROCm detect
#     - run: make demo-hip                                                     # gfx canary build/run
#
# Until those runners exist the matrix is intentionally empty; this script's
# table-driven detect + `make demo-hip` are the per-runner canary it would call.

# Detection lives in a function so that invoking it as `amd_accel_detect || true`
# suppresses the caller's errexit for the whole body (a sourced script must not
# abort its parent on a missing optional tool like rocminfo).

# gfx_arch_profile <gfx> — look up the portability row for a gfx ISA. Sets the
# globals GFX_OVERRIDE / GFX_SDMA / GFX_OVERRIDE_KMIN / GFX_REC_KERNEL /
# GFX_REC_ROCM / GFX_NOTE. GFX_OVERRIDE empty => no HSA override for that arch;
# GFX_OVERRIDE_KMIN is the minimum kernel major.minor at which the override is
# applied ("0" => always).
gfx_arch_profile() {
    case "${1}" in
        gfx1151)
            # AUTHORITATIVE for this box (Strix Halo APU / Radeon 8060S). gfx1151
            # on kernel 6.19.x mis-detects its ISA -> force 11.5.1; SDMA off for
            # Strix Halo stability. Must stay byte-identical to historical runs.
            GFX_OVERRIDE="11.5.1"; GFX_SDMA="0"; GFX_OVERRIDE_KMIN="6.19"
            GFX_REC_KERNEL=">=6.18.4 (HSA override on 6.19.x)"
            GFX_REC_ROCM=">=6.4 (7.2.x tested here)"
            GFX_NOTE="Strix Halo APU — authoritative for this box"
            ;;
        gfx1100|gfx1101|gfx1102)
            # DEFAULT (public ROCm docs): RDNA3 discrete (RX 7900/7800/7600 class)
            # is natively enumerated, needs NO HSA override, runs with SDMA at the
            # ROCm default. Provided so the script is sane if reused on these.
            GFX_OVERRIDE=""; GFX_SDMA="1"; GFX_OVERRIDE_KMIN="0"
            GFX_REC_KERNEL=">=6.5"
            GFX_REC_ROCM=">=6.0"
            GFX_NOTE="RDNA3 discrete (DEFAULT; no HSA override needed)"
            ;;
        gfx942)
            # DEFAULT (public ROCm docs): CDNA3 (MI300X/MI300A) is natively
            # enumerated and is a first-class ROCm target; no override, SDMA on.
            GFX_OVERRIDE=""; GFX_SDMA="1"; GFX_OVERRIDE_KMIN="0"
            GFX_REC_KERNEL=">=6.5"
            GFX_REC_ROCM=">=6.0"
            GFX_NOTE="CDNA3 MI300 (DEFAULT; natively enumerated)"
            ;;
        *)
            # Unknown / rocminfo absent: fall back to the historical gfx1151
            # assumption for THIS repo's box so a degraded detection stays
            # byte-identical to the pre-table behavior (override 11.5.1 on
            # kernel>=6.19, SDMA off).
            GFX_OVERRIDE="11.5.1"; GFX_SDMA="0"; GFX_OVERRIDE_KMIN="6.19"
            GFX_REC_KERNEL=">=6.18.4"
            GFX_REC_ROCM=">=6.4"
            GFX_NOTE="unknown arch — legacy gfx1151 fallback"
            ;;
    esac
}

amd_accel_detect() {
    local cpu_model cpu_threads ram_kb ram_gb kver kmajor kminor
    local rocm="no" vulkan="no" npu="no" gfx="unknown" backend="cpu"
    local rocm_version="" vf

    # ---- CPU + unified RAM ----
    cpu_threads="$(nproc 2>/dev/null || echo 1)"
    cpu_model="$(LC_ALL=C lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
    [[ -z "${cpu_model}" ]] && cpu_model="$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -1)"
    [[ -z "${cpu_model}" ]] && cpu_model="unknown CPU"
    ram_kb="$(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p' /proc/meminfo 2>/dev/null | head -1)"
    [[ -z "${ram_kb}" ]] && ram_kb=0
    ram_gb=$(( ram_kb / 1024 / 1024 ))

    # ---- kernel (gfx1151 needs HSA_OVERRIDE on 6.19.x) ----
    kver="$(uname -r 2>/dev/null || echo 0.0)"
    kmajor="${kver%%.*}"
    kminor="${kver#*.}"; kminor="${kminor%%.*}"
    [[ "${kmajor}" =~ ^[0-9]+$ ]] || kmajor=0
    [[ "${kminor}" =~ ^[0-9]+$ ]] || kminor=0

    # ---- ROCm ----
    if command -v rocminfo >/dev/null 2>&1; then
        local rocm_out
        rocm_out="$(rocminfo 2>/dev/null || true)"
        if echo "${rocm_out}" | grep -qiE 'gfx[0-9]'; then
            rocm="yes"
            gfx="$(echo "${rocm_out}" | grep -oiE 'gfx[0-9a-f]+' | head -1)"
        fi
    fi

    # ---- ROCm version (parse /opt/rocm-*/.info/version, fall back to hipconfig) ----
    for vf in /opt/rocm-*/.info/version /opt/rocm/.info/version; do
        if [[ -r "${vf}" ]]; then
            rocm_version="$(head -1 "${vf}" 2>/dev/null | tr -d '[:space:]')"
            [[ -n "${rocm_version}" ]] && break
        fi
    done
    if [[ -z "${rocm_version}" ]] && command -v hipconfig >/dev/null 2>&1; then
        rocm_version="$(hipconfig --version 2>/dev/null | head -1 | tr -d '[:space:]')"
    fi
    [[ -z "${rocm_version}" ]] && rocm_version="unknown"

    # ---- Vulkan ----
    if command -v vulkaninfo >/dev/null 2>&1; then
        if vulkaninfo --summary >/dev/null 2>&1 || vulkaninfo >/dev/null 2>&1; then
            vulkan="yes"
        fi
    fi

    # ---- NPU (XDNA2) ----
    if command -v xrt-smi >/dev/null 2>&1; then
        if xrt-smi examine >/dev/null 2>&1; then
            npu="yes"
        fi
    fi

    # ---- backend choice for AI inference (NOT for the prover) ----
    if [[ "${rocm}" == "yes" ]]; then
        backend="rocm"
    elif [[ "${vulkan}" == "yes" ]]; then
        backend="vulkan"
    else
        backend="cpu"
    fi

    # ---- gfx-arch portability lookup (table-driven, replaces gfx1151 hardcode) ----
    gfx_arch_profile "${gfx}"

    # ---- exports (respect anything the caller already set) ----
    export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${cpu_threads}}"
    export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-${GFX_SDMA}}"
    export AI_GPU_BACKEND="${AI_GPU_BACKEND:-${backend}}"
    export AMD_UNIFIED_RAM_GB="${AMD_UNIFIED_RAM_GB:-${ram_gb}}"
    export ROCM_VERSION="${ROCM_VERSION:-${rocm_version}}"
    # HSA override is per-arch: apply the table's version only when the kernel is
    # at/above the arch's threshold (gfx1151: 6.19; "0" => always). Harmless to set.
    if [[ -z "${HSA_OVERRIDE_GFX_VERSION:-}" && -n "${GFX_OVERRIDE}" ]]; then
        local omajor ominor
        if [[ "${GFX_OVERRIDE_KMIN}" == "0" ]]; then
            omajor=0; ominor=0
        else
            omajor="${GFX_OVERRIDE_KMIN%%.*}"
            ominor="${GFX_OVERRIDE_KMIN#*.}"; ominor="${ominor%%.*}"
        fi
        [[ "${omajor}" =~ ^[0-9]+$ ]] || omajor=0
        [[ "${ominor}" =~ ^[0-9]+$ ]] || ominor=0
        if (( kmajor > omajor || (kmajor == omajor && kminor >= ominor) )); then
            export HSA_OVERRIDE_GFX_VERSION="${GFX_OVERRIDE}"
        fi
    fi

    # ---- banner ----
    echo "######################################################################"
    echo "# AMD Strix Halo capability map ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "#   CPU            : ${cpu_model}"
    echo "#   Logical threads: ${cpu_threads}  -> RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"
    echo "#   Unified RAM    : ${ram_gb} GB (LPDDR5X)"
    echo "#   Kernel         : ${kver}"
    echo "#   ROCm (rocminfo): ${rocm}  (gpu=${gfx})"
    echo "#   ROCm version   : ${ROCM_VERSION}"
    echo "#   gfx profile    : ${gfx} -> ${GFX_NOTE}"
    echo "#   recommend      : kernel ${GFX_REC_KERNEL}  ·  ROCm ${GFX_REC_ROCM}"
    echo "#   Vulkan         : ${vulkan}"
    echo "#   NPU (xrt-smi)  : ${npu}"
    echo "#   AI_GPU_BACKEND : ${AI_GPU_BACKEND}   (AI inference only)"
    echo "#   HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-<unset>}  HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA}"
    echo "#"
    echo "#   NOTE: r0vm STARK + Groth16 wrap are CPU-ONLY on AMD (no ROCm/Vulkan"
    echo "#   prover). The iGPU/NPU above accelerate the AI MODEL, not the proof."
    echo "######################################################################"
}

amd_accel_detect || true
