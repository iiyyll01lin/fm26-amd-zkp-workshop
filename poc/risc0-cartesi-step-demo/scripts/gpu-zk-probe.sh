#!/usr/bin/env bash
# gpu-zk-probe.sh — PURE READ-ONLY probe for whether this AMD host can run the
# Path E "GPU-accelerated ZK primitive" track: ROCm + OpenCL on the Radeon iGPU
# (gfx1151 on Strix Halo). It reports readiness and CHANGES NOTHING: no install,
# no modprobe, no kernel build, no compute dispatch — it only reads what tools
# (rocminfo / clinfo / ldconfig) and /dev/sys already expose.
#
#   bash poc/risc0-cartesi-step-demo/scripts/gpu-zk-probe.sh
#
# NEVER HARD-FAILS. Path E is a research/auxiliary track; this script ALWAYS
# exits 0 (and `return 0` if sourced) so it can never break a run that calls it,
# even on a box with no ROCm and no GPU.
#
# HONESTY RULE (mirrors amd-accel-detect.sh / npu-probe.sh). A GPU-ZK-READY
# verdict means the iGPU can accelerate ZK *primitives* (MSM / NTT) and a
# Groth16 prover via the cross-vendor **OpenCL** path (ec-gpu / bellperson) —
# Tier 1/2 of Path E. It does **NOT** make the Demo B `r0vm` STARK main line
# GPU-accelerated: RISC0 has no AMD ROCm/Vulkan prover, so the STARK + Groth16
# wrap stay CPU-only on AMD. See docs/amd-strix-halo-acceleration.md and
# reading-notes/path-e-amd-gpu-zk-primitives.md.
#
# What it checks (each optional, each degrades gracefully to "no"):
#   * /dev/kfd + /dev/dri/renderD*   (amdgpu compute nodes)
#   * ROCm install + version         (/opt/rocm*, .info/version)
#   * rocminfo                       (HSA agent + gfx ISA target, e.g. gfx1151)
#   * OpenCL ICD + libOpenCL         (/etc/OpenCL/vendors, ldconfig)
#   * clinfo                         (platform + GPU device + driver version)
#   * kernel version vs gfx1151 stable line (>= 6.18.4 recommended)

gpu_zk_probe() {
    local cpu_model is_halo kver kmajor kminor kpatch kernel_ok
    local kfd="no" dri_render=""
    local rocm="no" rocm_path="" rocm_ver="n/a"
    local rocminfo_bin="no" gfx="" hsa_gpu="no"
    local ocl_icd="" libocl="no"
    local clinfo_bin="no" ocl_platform="" ocl_device="" ocl_driver="" ocl_cu="n/a" ocl_mem="n/a" ocl_gpu="no"
    local verdict hsa_hint n

    # ---- host context (informational) ----
    cpu_model="$(LC_ALL=C lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
    [[ -z "${cpu_model}" ]] && cpu_model="$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -1)"
    [[ -z "${cpu_model}" ]] && cpu_model="unknown CPU"
    case "${cpu_model}" in
        *"AI MAX+ PRO 395"*|*"AI MAX+ 395"*|*"AI MAX 395"*|*"Strix Halo"*|*"Radeon 8060S"*) is_halo="yes (Strix Halo / gfx1151)" ;;
        *"Ryzen AI"*)                                  is_halo="maybe (a Ryzen AI part)" ;;
        *)                                             is_halo="no" ;;
    esac
    kver="$(uname -r 2>/dev/null || echo unknown)"
    # parse MAJOR.MINOR.PATCH for the >= 6.18.4 gfx1151 stable-line check
    kmajor="${kver%%.*}"; kminor="${kver#*.}"; kminor="${kminor%%.*}"
    kpatch="${kver#*.*.}"; kpatch="${kpatch%%[!0-9]*}"
    [[ "${kmajor}" =~ ^[0-9]+$ ]] || kmajor=0
    [[ "${kminor}" =~ ^[0-9]+$ ]] || kminor=0
    [[ "${kpatch}" =~ ^[0-9]+$ ]] || kpatch=0
    if (( kmajor > 6 )) || (( kmajor == 6 && kminor > 18 )) || \
       (( kmajor == 6 && kminor == 18 && kpatch >= 4 )); then
        kernel_ok="yes (>= 6.18.4 gfx1151 stable line)"
    else
        kernel_ok="below 6.18.4 (gfx1151 may be flaky; ROCm can still detect it)"
    fi

    # ---- amdgpu compute device nodes ----
    [[ -e /dev/kfd ]] && kfd="yes (/dev/kfd)"
    for n in /dev/dri/renderD*; do
        [[ -e "${n}" ]] || continue
        dri_render+="${n} "
    done
    dri_render="${dri_render% }"

    # ---- ROCm install ----
    local d
    for d in /opt/rocm /opt/rocm-*; do
        [[ -d "${d}" ]] || continue
        rocm="yes"
        rocm_path="${d}"
        break
    done
    for d in /opt/rocm/.info/version /opt/rocm-*/.info/version; do
        [[ -r "${d}" ]] || continue
        rocm_ver="$(cat "${d}" 2>/dev/null | head -1)"
        break
    done

    # ---- rocminfo: HSA agents + gfx ISA target ----
    if command -v rocminfo >/dev/null 2>&1; then
        rocminfo_bin="yes"
        local ri
        ri="$(rocminfo 2>/dev/null || true)"
        gfx="$(echo "${ri}" | grep -oE 'gfx[0-9a-z]+' | sort -u | tr '\n' ' ' | sed 's/ $//')"
        echo "${ri}" | grep -qiE 'Device Type:[[:space:]]*GPU' && hsa_gpu="yes"
    fi

    # ---- OpenCL ICD + loader ----
    for d in /etc/OpenCL/vendors/*.icd; do
        [[ -e "${d}" ]] || continue
        ocl_icd+="$(basename "${d}") "
    done
    ocl_icd="${ocl_icd% }"
    ldconfig -p 2>/dev/null | grep -qi 'libOpenCL\.so' && libocl="yes"

    # ---- clinfo: platform + GPU device + driver ----
    if command -v clinfo >/dev/null 2>&1; then
        clinfo_bin="yes"
        local ci
        ci="$(clinfo 2>/dev/null || true)"
        # clinfo prints "Label:<tabs>value"; consume the colon + surrounding ws.
        ocl_platform="$(echo "${ci}" | sed -n 's/^[[:space:]]*Platform Name[[:space:]]*:[[:space:]]*//p' | head -1)"
        ocl_device="$(echo "${ci}" | sed -n 's/^[[:space:]]*Device Name[[:space:]]*:[[:space:]]*//p' | head -1)"
        ocl_driver="$(echo "${ci}" | sed -n 's/^[[:space:]]*Driver [Vv]ersion[[:space:]]*:[[:space:]]*//p' | head -1)"
        ocl_cu="$(echo "${ci}" | sed -n 's/^[[:space:]]*Max compute units[[:space:]]*:[[:space:]]*//p' | head -1)"
        ocl_mem="$(echo "${ci}" | sed -n 's/^[[:space:]]*Global memory size[[:space:]]*:[[:space:]]*//p' | head -1)"
        echo "${ci}" | grep -qiE 'Device Type[[:space:]]*:[[:space:]]*CL_DEVICE_TYPE_GPU' && ocl_gpu="yes"
        # Fall back to the ROCm marketing/gfx name if clinfo omits Device Name.
        [[ -n "${ocl_device}" ]] || ocl_device="$(rocminfo 2>/dev/null | sed -n 's/^[[:space:]]*Marketing Name:[[:space:]]*//p' | grep -i radeon | head -1)"
        [[ -n "${ocl_device}" ]] || ocl_device="${gfx:-<unnamed GPU>}"
        [[ -n "${ocl_cu}" ]]  || ocl_cu="n/a"
        [[ -n "${ocl_mem}" ]] || ocl_mem="n/a"
    fi

    # ---- HSA env hint (only kernel >= 6.19 needs the gfx1151 ISA override) ----
    if (( kmajor == 6 && kminor >= 19 )) || (( kmajor > 6 )); then
        hsa_hint="kernel >= 6.19: export HSA_OVERRIDE_GFX_VERSION=11.5.1 HSA_ENABLE_SDMA=0"
    else
        hsa_hint="kernel < 6.19: no HSA_OVERRIDE needed (rocminfo detects gfx1151 natively)"
    fi

    # ---- verdict ----
    if [[ "${hsa_gpu}" == "yes" && "${ocl_gpu}" == "yes" && "${libocl}" == "yes" ]]; then
        verdict="GPU-ZK-READY — ROCm sees ${gfx:-a GPU} and OpenCL enumerates a GPU ('${ocl_device}'). Tier 1/2 can run on the iGPU."
    elif [[ "${rocm}" == "yes" && ( "${hsa_gpu}" == "yes" || -n "${gfx}" ) ]]; then
        verdict="PARTIAL — ROCm/rocminfo present but OpenCL device not enumerated (install rocm-opencl / check ICD). Tier 1/2 blocked until clinfo lists the GPU."
    elif [[ "${kfd}" == yes* || -n "${dri_render}" ]]; then
        verdict="DRIVER-ONLY — amdgpu nodes present but no ROCm/OpenCL stack. Install ROCm 7.2+/nightly + rocm-opencl."
    else
        verdict="NOT-READY — no amdgpu compute stack here (expected on a non-AMD or CPU-only box). Tier 1/2 documented-but-skipped."
    fi

    # ---- banner (matches amd-accel-detect.sh / npu-probe.sh house style) ----
    echo "######################################################################"
    echo "# Path E GPU-ZK readiness probe ($(date -u +%Y-%m-%dT%H:%M:%SZ)) — READ-ONLY"
    echo "#   CPU               : ${cpu_model}"
    echo "#   Strix Halo?       : ${is_halo}"
    echo "#   Kernel            : ${kver}  [${kernel_ok}]"
    echo "#   /dev/kfd          : ${kfd}"
    echo "#   /dev/dri/render   : ${dri_render:-<none>}"
    echo "#   ROCm              : ${rocm}${rocm_path:+ (${rocm_path})}  version=${rocm_ver}"
    echo "#   rocminfo          : ${rocminfo_bin}  gfx=${gfx:-<none>}  gpu_agent=${hsa_gpu}"
    echo "#   OpenCL ICD        : ${ocl_icd:-<none>}  libOpenCL=${libocl}"
    echo "#   clinfo platform   : ${ocl_platform:-<none>}"
    echo "#   clinfo device     : ${ocl_device:-<none>}"
    echo "#   clinfo driver     : ${ocl_driver:-n/a}  CUs=${ocl_cu}  globalmem=${ocl_mem}"
    echo "#   HSA env hint      : ${hsa_hint}"
    echo "#"
    echo "#   VERDICT : ${verdict}"
    echo "#"
    echo "#   NOTE: GPU-ZK-READY enables Path E Tier 1/2 (MSM/NTT + a Groth16"
    echo "#   prover) on the iGPU via the OpenCL path (ec-gpu / bellperson). It"
    echo "#   does NOT GPU-accelerate the Demo B r0vm STARK main line: RISC0 has"
    echo "#   no AMD ROCm/Vulkan prover, so the STARK + Groth16 wrap stay CPU-only."
    echo "######################################################################"
}

gpu_zk_probe || true

# Never hard-fail: exit 0 when run directly, return 0 when sourced.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exit 0
else
    return 0 2>/dev/null || true
fi
