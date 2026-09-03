#!/usr/bin/env bash
# npu-probe.sh — PURE READ-ONLY probe for the AMD Strix Halo (Ryzen AI MAX+ 395)
# XDNA2 NPU. It reports whether the NPU is visible on this Linux host and prints
# the documented *community* path to enable it. It CHANGES NOTHING: no modprobe,
# no install, no firmware touch, no `xrt-smi validate` (which would run a
# workload) — it only reads /proc, /sys, /dev, /lib/firmware and queries tools
# that are already installed.
#
#   bash poc/risc0-cartesi-step-demo/scripts/npu-probe.sh
#
# RESEARCH-ONLY + NEVER HARD-FAILS. The XDNA2 NPU is an experimental track in
# this report, NOT a dependency of any proof. This script ALWAYS exits 0 (and
# `return 0` if it is sourced) so it can never break a run that calls or sources
# it, even when there is no NPU, no driver, no firmware and no XRT present.
#
# HONESTY RULE (printed in the banner too, mirroring amd-accel-detect.sh): even
# when the NPU is fully enabled it accelerates the **AI MODEL** (e.g. a BitNet
# prefill / DEAAP embedding), **NOT the ZK proof**. RISC0 r0vm STARK proving and
# the Groth16 wrap stay CPU-only on AMD — see docs/amd-strix-halo-acceleration.md
# and reading-notes/path-d-npu-xdna2.md.
#
# What it checks (each optional, each degrades gracefully to "no"):
#   * amdxdna kernel driver  (lsmod, /sys/module/amdxdna, dmesg if readable)
#   * /dev/accel/accel*      (DRM-accel node the amdxdna driver creates)
#   * PCI NPU function       (1022:17f0-class, best-effort via lspci)
#   * NPU firmware           (/lib/firmware/amdnpu/<dev>/npu*.sbin)
#   * XRT runtime + xrt-smi  (`xrt-smi examine` — read-only enumerate + fw ver)
#   * IRON / MLIR-AIE / Peano(the authoring toolchain that compiles+dispatches)

# Detection lives in a function so `npu_probe || true` swallows any errexit a
# caller may have set, and so a single missing optional tool can never abort the
# parent shell. The function itself never calls `exit`.
npu_probe() {
    local cpu_model is_halo kver
    local amdxdna="no" amdxdna_ver="n/a" dmesg_hit=""
    local accel_nodes="" pci="unknown(no lspci)"
    local fw="no" fw_path=""
    local xrt="no" xrt_smi="no" xrt_dev="no" xrt_fw="n/a"
    local iron="no" peano="no" mlir_aie="no"
    local verdict toolchain n

    # ---- host context (purely informational) ----
    cpu_model="$(LC_ALL=C lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
    [[ -z "${cpu_model}" ]] && cpu_model="$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -1)"
    [[ -z "${cpu_model}" ]] && cpu_model="unknown CPU"
    case "${cpu_model}" in
        *"AI MAX+ 395"*|*"AI MAX 395"*|*"Strix Halo"*) is_halo="yes (Strix Halo / STX-H)" ;;
        *"Ryzen AI"*)                                  is_halo="no (a Ryzen AI part, not the 395/Halo)" ;;
        *)                                             is_halo="no" ;;
    esac
    kver="$(uname -r 2>/dev/null || echo unknown)"

    # ---- amdxdna kernel driver ----
    if lsmod 2>/dev/null | grep -q '^amdxdna'; then
        amdxdna="yes (lsmod)"
    elif [[ -d /sys/module/amdxdna ]]; then
        amdxdna="yes (/sys/module/amdxdna)"
    fi
    [[ -r /sys/module/amdxdna/version ]] && amdxdna_ver="$(cat /sys/module/amdxdna/version 2>/dev/null || echo n/a)"
    # dmesg only if the kernel allows unprivileged reads (kernel.dmesg_restrict=0);
    # otherwise this is silently empty — never sudo, never fail.
    dmesg_hit="$(dmesg 2>/dev/null | grep -iE 'amdxdna|XDNA|aie2' | tail -1 || true)"

    # ---- DRM accel device node (amdxdna creates /dev/accel/accel0) ----
    for n in /dev/accel/accel* /dev/accel*; do
        [[ -e "${n}" ]] || continue
        [[ "${n}" == "/dev/accel" ]] && continue   # skip the directory entry itself
        case " ${accel_nodes} " in *" ${n} "*) continue ;; esac   # de-dup
        accel_nodes+="${n} "
    done
    accel_nodes="${accel_nodes% }"

    # ---- PCI NPU function (1022:17f0 = Strix NPU), best-effort ----
    if command -v lspci >/dev/null 2>&1; then
        if lspci -nn 2>/dev/null | grep -qiE '1022:17f0|Signal processing controller'; then
            pci="present (1022:17f0-class / Signal processing controller)"
        else
            pci="not seen by lspci"
        fi
    fi

    # ---- NPU firmware blobs ----
    local fwbase
    for fwbase in /lib/firmware/amdnpu /usr/lib/firmware/amdnpu; do
        [[ -d "${fwbase}" ]] || continue
        fw="yes"
        fw_path="${fwbase}"
        break
    done

    # ---- XRT runtime ----
    [[ -d /opt/xilinx/xrt ]] && xrt="yes (/opt/xilinx/xrt)"
    local xrt_bin=""
    if command -v xrt-smi >/dev/null 2>&1; then
        xrt_bin="$(command -v xrt-smi)"
        xrt_smi="yes (${xrt_bin})"
    elif [[ -x /opt/xilinx/xrt/bin/xrt-smi ]]; then
        xrt_bin="/opt/xilinx/xrt/bin/xrt-smi"
        xrt_smi="yes (${xrt_bin})"
    fi
    # `xrt-smi examine` only enumerates (read-only); `validate` would run a
    # workload, so we deliberately never call it here.
    if [[ -n "${xrt_bin}" ]]; then
        local examine
        examine="$("${xrt_bin}" examine 2>/dev/null || true)"
        if echo "${examine}" | grep -qiE 'NPU|RyzenAI|Ryzen AI|accel|17f0'; then
            xrt_dev="yes"
        fi
        xrt_fw="$(echo "${examine}" | grep -iE 'Firmware Version' | head -1 | sed 's/.*:[[:space:]]*//' || true)"
        [[ -n "${xrt_fw}" ]] || xrt_fw="n/a"
    fi

    # ---- IRON / MLIR-AIE / Peano authoring toolchain ----
    command -v aie-opt   >/dev/null 2>&1 && peano="yes (aie-opt on PATH)"
    command -v aiecc.py  >/dev/null 2>&1 && mlir_aie="yes (aiecc.py on PATH)"
    if command -v python3 >/dev/null 2>&1; then
        # find_spec() locates the package without importing it (fast + side-effect free).
        if python3 -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('aie') else 1)" >/dev/null 2>&1; then
            iron="yes (python 'aie' module)"
        fi
    fi

    # ---- verdicts ----
    if [[ "${xrt_dev}" == "yes" ]]; then
        verdict="ENABLED — XRT enumerates the NPU (firmware ${xrt_fw})"
    elif [[ -n "${accel_nodes}" && "${amdxdna}" != "no" ]]; then
        verdict="DRIVER-READY — amdxdna + accel node present, but XRT did not enumerate it (check fw/XRT version match)"
    elif [[ "${amdxdna}" != "no" || -n "${accel_nodes}" ]]; then
        verdict="PARTIAL — kernel side visible, user-space XRT runtime missing"
    else
        verdict="NOT ENABLED here — no amdxdna driver / accel node visible (expected on a non-Halo dev box)"
    fi
    if [[ "${iron}" != "no" || "${peano}" != "no" || "${mlir_aie}" != "no" ]]; then
        toolchain="present — can author/compile an xclbin locally"
    else
        toolchain="absent — install IRON + MLIR-AIE + Peano to author kernels"
    fi

    # ---- banner (matches amd-accel-detect.sh house style) ----
    echo "######################################################################"
    echo "# AMD XDNA2 NPU probe ($(date -u +%Y-%m-%dT%H:%M:%SZ)) — READ-ONLY / research"
    echo "#   CPU               : ${cpu_model}"
    echo "#   Strix Halo?       : ${is_halo}"
    echo "#   Kernel            : ${kver}"
    echo "#   PCI NPU function  : ${pci}"
    echo "#   amdxdna driver    : ${amdxdna}  (version=${amdxdna_ver})"
    echo "#   /dev/accel node   : ${accel_nodes:-<none>}"
    echo "#   NPU firmware      : ${fw}${fw_path:+ (${fw_path})}"
    echo "#   XRT runtime       : ${xrt}"
    echo "#   xrt-smi           : ${xrt_smi}"
    echo "#   xrt-smi examine   : device=${xrt_dev}  firmware=${xrt_fw}"
    echo "#   IRON (python aie) : ${iron}"
    echo "#   Peano (llvm-aie)  : ${peano}"
    echo "#   MLIR-AIE (aiecc)  : ${mlir_aie}"
    [[ -n "${dmesg_hit}" ]] && echo "#   dmesg (last hit)  : ${dmesg_hit}"
    echo "#"
    echo "#   VERDICT   : ${verdict}"
    echo "#   TOOLCHAIN : ${toolchain}"
    echo "#"
    echo "#   NOTE: the NPU accelerates the AI MODEL (BitNet prefill / DEAAP"
    echo "#   embedding), NOT the ZK proof. r0vm STARK + Groth16 wrap stay"
    echo "#   CPU-only on AMD. This probe is research-only and never a dependency."
    echo "######################################################################"

    # ---- documented community enablement path (Strix Halo / Linux) ----
    cat <<'PATH_EOF'
# Documented path to enable the XDNA2 NPU on Strix Halo (Linux, COMMUNITY route):
#   The official AMD Ryzen AI SW Platform on Linux does NOT support Strix Halo
#   (STX-H); the only working Linux route today is the open-source stack below.
#
#   1. Kernel   : Linux >= 6.10 built with CONFIG_DRM_ACCEL + CONFIG_AMD_IOMMU.
#                 The in-tree `amdxdna` driver landed in 6.14 (Ubuntu 25.04),
#                 but it lags upstream amd/xdna-driver by ~1 year and the
#                 driver / firmware / XRT-shim versions are tightly coupled.
#   2. Stack    : build amd/xdna-driver from source for a MATCHED set —
#                 DKMS amdxdna.ko + dev firmware (npu.dev.sbin) under
#                 /lib/firmware/amdnpu/<dev>/ + the matching xrt_plugin .deb.
#                 Then `source /opt/xilinx/xrt/setup.sh` and `xrt-smi examine`.
#                 Set `memlock unlimited` (/etc/security/limits.d/).
#   3. Authoring: pip-install Peano (llvm-aie) + MLIR-AIE + IRON (amd/IRON).
#                 design.py (IRON Python API) -> aie-opt/aie-translate (Peano)
#                 -> xclbin + insts.bin -> dispatch via XRT (xrt::device/kernel/bo).
#                 Strix Halo NPU = AIE2P / NPU2 (4 rows x 8 columns).
#   4. Workload : a BitNet-1.58 *prefill* GEMM (or the IRON Llama-3.2-1B
#                 reference under applications/llama_3.2_1b/) is the candidate.
#                 It accelerates the AI MODEL, NOT the ZK proof.
#
#   Full write-up + DEAAP mapping : reading-notes/path-d-npu-xdna2.md
#   Engine matrix / honesty layer : docs/amd-strix-halo-acceleration.md
PATH_EOF
}

npu_probe || true

# Never hard-fail. Detect sourced-vs-executed via BASH_SOURCE/$0: exit 0 when run
# directly (`bash npu-probe.sh`), return 0 when sourced (so a caller that
# `source`s the probe is never knocked out of its own shell). Research-only.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exit 0
else
    return 0 2>/dev/null || true
fi
