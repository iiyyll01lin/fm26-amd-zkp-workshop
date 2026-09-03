#!/usr/bin/env bash
# diagnose.sh — gfx1151 ROCm bring-up runbook, as a runnable capability probe.
#
# WHY THIS EXISTS: the course (M1 / M2 / the ROCm FAQ) *describes* every gfx1151
# bring-up trap — kernel >= 6.18.4, `HSA_OVERRIDE_GFX_VERSION=11.5.1` on 6.19.x,
# `HSA_ENABLE_SDMA=0`, the `linux-firmware-20251125` trap, `/dev/kfd`+`/dev/dri`
# perms, Docker device mounts, and raising the TTM/GTT page limit for full
# unified-memory use — but there was no single hands-on artefact you could RUN
# that checks each one on the box in front of you and tells you what to fix.
# This script is that artefact. It is the runnable companion to the read-only
# `amd-accel-detect.sh` capability banner: where that one EXPORTS env for the
# workloads, this one DIAGNOSES each bring-up precondition and emits a structured
# `artefacts/bringup-report.json` plus human repair hints.
#
# DESIGN RULES (match repo precedent):
#   * idempotent + no-sudo-by-default: every check is read-only; the only writes
#     are under artefacts/. Commands that WOULD need root (usermod, modprobe,
#     writing sysfs) are PRINTED as repair hints, never executed.
#   * never hard-fails: safe to run on a laptop with no ROCm at all — it just
#     records `rocm:false` and the honest stop-point, like the other probes.
#   * honest: it reports readiness; it does NOT claim the GPU proves anything.
#     iGPU/NPU accelerate AI models; iGPU OpenCL accelerates SNARK primitives
#     (size-gated); RISC0 STARK is CPU-only on AMD.
#
# Usage:
#   bash scripts/diagnose.sh                 # probe + write artefacts/bringup-report.json
#   BRINGUP_TTM=1 bash scripts/diagnose.sh   # also read the live amdgpu TTM/GTT sysfs knobs
#
# Exit code is ALWAYS 0 (a probe must not abort a caller's pipeline); the verdict
# lives in the JSON `ready` field, not the exit status.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "${HERE}/.." && pwd)"
ART="${DEMO}/artefacts"
mkdir -p "${ART}"
# REPORT path is overridable so before/after snapshots (e.g. a TTM-raise
# verification) can be captured without clobbering the committed baseline.
REPORT="${REPORT:-${ART}/bringup-report.json}"

# ---- tiny helpers (no external deps; pure bash) -----------------------------
have() { command -v "$1" >/dev/null 2>&1; }
jstr() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }   # JSON-escape a string

CHECK_NAMES=()        # parallel arrays: name / status(ok|warn|fail) / detail / hint
CHECK_STATUS=()
CHECK_DETAIL=()
CHECK_HINT=()
add_check() {         # add_check NAME STATUS DETAIL HINT
    CHECK_NAMES+=("$1"); CHECK_STATUS+=("$2"); CHECK_DETAIL+=("$3"); CHECK_HINT+=("$4")
}

echo "######################################################################"
echo "# gfx1151 ROCm bring-up diagnose ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
echo "######################################################################"

# ---- 1. kernel: >= 6.18.4 stable; 6.19.x mis-detects gfx1151 ISA ------------
KVER="$(uname -r 2>/dev/null || echo 0.0.0)"
KMAJ="${KVER%%.*}"; KREST="${KVER#*.}"; KMIN="${KREST%%.*}"; KPAT="${KREST#*.}"; KPAT="${KPAT%%[-.]*}"
[[ "${KMAJ}" =~ ^[0-9]+$ ]] || KMAJ=0
[[ "${KMIN}" =~ ^[0-9]+$ ]] || KMIN=0
[[ "${KPAT}" =~ ^[0-9]+$ ]] || KPAT=0
NEED_OVERRIDE="no"
if (( KMAJ > 6 || (KMAJ == 6 && KMIN >= 19) )); then
    NEED_OVERRIDE="yes"
    add_check "kernel" "warn" "kernel ${KVER} >= 6.19 — gfx1151 ISA mis-detect window" \
        "export HSA_OVERRIDE_GFX_VERSION=11.5.1 (force ISA 11.5.1) and HSA_ENABLE_SDMA=0"
elif (( KMAJ == 6 && KMIN == 18 && KPAT >= 4 )) || (( KMAJ == 6 && KMIN > 18 && KMIN < 19 )); then
    add_check "kernel" "ok" "kernel ${KVER} on the 6.18.4+ stable line — no override needed" ""
else
    add_check "kernel" "warn" "kernel ${KVER} below the 6.18.4 recommended line" \
        "upgrade to kernel >= 6.18.4 (the repo's Path E data was taken on 6.17 — works, but not recommended)"
fi

# ---- 2. HSA env: override only on >= 6.19, SDMA always off for stability ----
HSA_OV="${HSA_OVERRIDE_GFX_VERSION:-<unset>}"
HSA_SD="${HSA_ENABLE_SDMA:-<unset>}"
if [[ "${NEED_OVERRIDE}" == "yes" && "${HSA_OV}" == "<unset>" ]]; then
    add_check "hsa_override" "warn" "kernel needs the ISA override but HSA_OVERRIDE_GFX_VERSION is unset" \
        "export HSA_OVERRIDE_GFX_VERSION=11.5.1"
elif [[ "${NEED_OVERRIDE}" == "no" && "${HSA_OV}" != "<unset>" ]]; then
    add_check "hsa_override" "warn" "HSA_OVERRIDE_GFX_VERSION=${HSA_OV} set on a kernel that does not need it" \
        "unset HSA_OVERRIDE_GFX_VERSION on the 6.18.x line — forcing it can mis-codegen on a correct driver"
else
    add_check "hsa_override" "ok" "HSA_OVERRIDE_GFX_VERSION=${HSA_OV} consistent with kernel ${KVER}" ""
fi
if [[ "${HSA_SD}" == "0" ]]; then
    add_check "hsa_sdma" "ok" "HSA_ENABLE_SDMA=0 (conservative copy path; Strix Halo stability)" ""
else
    add_check "hsa_sdma" "warn" "HSA_ENABLE_SDMA=${HSA_SD} — SDMA can intermittently hang on early Halo stacks" \
        "export HSA_ENABLE_SDMA=0 for stability (harmless on the stable line too)"
fi

# ---- 3. firmware: avoid linux-firmware-20251125 ----------------------------
FW_VER="unknown"
for f in /lib/firmware/amdgpu /usr/lib/firmware/amdgpu; do
    [[ -d "$f" ]] || continue
done
if have dpkg-query; then
    FW_VER="$(dpkg-query -W -f='${Version}' linux-firmware 2>/dev/null || echo unknown)"
elif have rpm; then
    FW_VER="$(rpm -q --qf '%{VERSION}' linux-firmware 2>/dev/null || echo unknown)"
fi
if [[ "${FW_VER}" == *20251125* ]]; then
    add_check "firmware" "fail" "linux-firmware ${FW_VER} — the known-bad 20251125 build for gfx1151" \
        "install a linux-firmware build before/after 20251125, then reboot to reload microcode"
else
    add_check "firmware" "ok" "linux-firmware ${FW_VER} (not the 20251125 trap build)" ""
fi

# ---- 4. device nodes: /dev/kfd + /dev/dri ----------------------------------
KFD_OK="no"; DRI_OK="no"
[[ -r /dev/kfd ]] && KFD_OK="yes"
ls /dev/dri/renderD* >/dev/null 2>&1 && [[ -r "$(ls /dev/dri/renderD* 2>/dev/null | head -1)" ]] && DRI_OK="yes"
if [[ "${KFD_OK}" == "yes" && "${DRI_OK}" == "yes" ]]; then
    add_check "device_nodes" "ok" "/dev/kfd + /dev/dri/renderD* readable by $(id -un)" ""
else
    add_check "device_nodes" "fail" "/dev/kfd readable=${KFD_OK} /dev/dri readable=${DRI_OK}" \
        "sudo usermod -aG render,video ${USER:-\$USER}  # then log out/in so the group takes effect"
fi

# ---- 5. ROCm enumerates gfx1151 --------------------------------------------
ROCM_OK="no"; GFX="unknown"
if have rocminfo; then
    ROCM_OUT="$(rocminfo 2>/dev/null || true)"
    if echo "${ROCM_OUT}" | grep -qiE 'gfx[0-9]'; then
        ROCM_OK="yes"
        GFX="$(echo "${ROCM_OUT}" | grep -oiE 'gfx[0-9a-f]+' | head -1)"
    fi
fi
if [[ "${ROCM_OK}" == "yes" ]]; then
    add_check "rocm" "ok" "rocminfo enumerates ${GFX}" ""
else
    add_check "rocm" "fail" "rocminfo did not enumerate a gfxNNNN device" \
        "install ROCm 7.2+ and ensure rocminfo is on PATH; re-check device nodes + groups above"
fi

# ---- 6. hipcc toolchain present --------------------------------------------
HIPCC_VER="none"
if have hipcc; then
    HIPCC_VER="$(hipcc --version 2>/dev/null | sed -n 's/^HIP version: //p' | head -1)"
    [[ -z "${HIPCC_VER}" ]] && HIPCC_VER="present"
    add_check "hipcc" "ok" "hipcc ${HIPCC_VER}" ""
else
    add_check "hipcc" "warn" "hipcc not on PATH (HIP demos cannot compile)" \
        "add /opt/rocm/bin to PATH, or install the ROCm hip dev packages"
fi

# ---- 7. ROCm math libraries (Track A: rocBLAS / hipBLASLt / rocFFT) --------
LIB_FOUND=()
for lib in rocblas hipblaslt rocfft; do
    if ls /opt/rocm*/lib/lib${lib}.so* >/dev/null 2>&1; then LIB_FOUND+=("${lib}"); fi
done
if (( ${#LIB_FOUND[@]} == 3 )); then
    add_check "rocm_libs" "ok" "rocBLAS + hipBLASLt + rocFFT present under /opt/rocm*/lib" ""
elif (( ${#LIB_FOUND[@]} > 0 )); then
    add_check "rocm_libs" "warn" "only found: ${LIB_FOUND[*]}" \
        "install the missing ROCm math libs (rocblas/hipblaslt/rocfft) for the library-ecosystem demo"
else
    add_check "rocm_libs" "warn" "no rocBLAS/hipBLASLt/rocFFT under /opt/rocm*/lib" \
        "install rocblas hipblaslt rocfft dev packages (needed by poc/amd-rocm-libs-demo)"
fi

# ---- 8. TTM / GTT page limits — the unified-memory unlock -------------------
# Strix Halo's headline is 94 GB usable unified LPDDR5X, but the amdgpu TTM
# (Translation Table Manager) caps how many pages the GPU may pin/pool. TWO knobs
# matter and BOTH gate the big-model cameo:
#   * ttm.pages_limit     — max pages TTM may allocate overall.
#   * ttm.page_pool_size  — the pool the ROCr/HIP allocator draws from; on gfx1151
#                           this is the BINDING runtime ceiling for a full-offload
#                           model. A raised pages_limit ALONE does not help if
#                           page_pool_size is left at the default — the effective
#                           pool is the SMALLER of the two. (Measured: with
#                           pages_limit=60 GiB but page_pool_size=47 GiB, a 54 GB
#                           BF16 model "failed to load" — see TTM-RUNBOOK.md.)
# amdgpu.gttsize (MiB) sizes the GTT aperture. If any is left at the small default
# you hit a GPU-side OOM while tens of GB of RAM sit free. These are READ-ONLY
# reads of the live knobs; the repair hints show the concrete kernel-cmdline /
# modprobe commands (root + reboot) without running them — see
# poc/amd-rocm-bringup/config/ + TTM-RUNBOOK.md.
TTM_PAGES="unknown"; POOL_PAGES="unknown"; TTM_GB="unknown"; POOL_GB="unknown"; EFF_GB="unknown"
PGSIZE="$(getconf PAGE_SIZE 2>/dev/null || echo 4096)"
_pages_to_gib(){ awk -v p="$1" -v s="${PGSIZE}" 'BEGIN{printf "%.1f", p*s/1024/1024/1024}'; }
if [[ -r /sys/module/ttm/parameters/pages_limit ]]; then
    TTM_PAGES="$(cat /sys/module/ttm/parameters/pages_limit 2>/dev/null || echo unknown)"
    [[ "${TTM_PAGES}" =~ ^[0-9]+$ ]] && TTM_GB="$(_pages_to_gib "${TTM_PAGES}")"
fi
if [[ -r /sys/module/ttm/parameters/page_pool_size ]]; then
    POOL_PAGES="$(cat /sys/module/ttm/parameters/page_pool_size 2>/dev/null || echo unknown)"
    [[ "${POOL_PAGES}" =~ ^[0-9]+$ ]] && POOL_GB="$(_pages_to_gib "${POOL_PAGES}")"
fi
# amdgpu gttsize is in MiB (-1 == "let TTM decide" == half of RAM by default).
GTTSIZE_MIB="unknown"
[[ -r /sys/module/amdgpu/parameters/gttsize ]] && GTTSIZE_MIB="$(cat /sys/module/amdgpu/parameters/gttsize 2>/dev/null || echo unknown)"
RAM_KB="$(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p' /proc/meminfo 2>/dev/null | head -1)"
[[ -z "${RAM_KB}" ]] && RAM_KB=0
RAM_GB=$(( RAM_KB / 1024 / 1024 ))
# Effective GPU-addressable pool = the SMALLER of pages_limit and page_pool_size
# (the binding runtime cap). Fall back to pages_limit if page_pool_size is unknown.
EFF_PAGES="unknown"
if [[ "${TTM_PAGES}" =~ ^[0-9]+$ && "${POOL_PAGES}" =~ ^[0-9]+$ ]]; then
    EFF_PAGES=$(( TTM_PAGES < POOL_PAGES ? TTM_PAGES : POOL_PAGES ))
elif [[ "${TTM_PAGES}" =~ ^[0-9]+$ ]]; then
    EFF_PAGES="${TTM_PAGES}"
fi
[[ "${EFF_PAGES}" =~ ^[0-9]+$ ]] && EFF_GB="$(_pages_to_gib "${EFF_PAGES}")"
# Verdict: "ok" iff the EFFECTIVE pool covers most (>=50%) of unified RAM. The
# common trap on a half-raised box: pages_limit was bumped but page_pool_size was
# not, so every full-offload allocation is still capped at the ~47 GiB default —
# exactly what makes a >47 GiB BF16 model fail to load.
TTM_STATUS="warn"
TTM_DETAIL="ttm.pages_limit=${TTM_PAGES} (~${TTM_GB} GiB), ttm.page_pool_size=${POOL_PAGES} (~${POOL_GB} GiB), effective pool ~${EFF_GB} GiB, amdgpu.gttsize=${GTTSIZE_MIB} MiB, RAM=${RAM_GB} GB"
HALF_PAGES=0
[[ "${RAM_KB}" -gt 0 ]] && HALF_PAGES=$(( (RAM_KB * 1024 / PGSIZE) / 2 ))
if [[ "${EFF_PAGES}" =~ ^[0-9]+$ && "${HALF_PAGES}" -gt 0 ]] && (( EFF_PAGES >= HALF_PAGES )); then
    TTM_STATUS="ok"
fi
# Explicitly flag the half-raised trap even when pages_limit alone looks healthy.
PARTIAL_RAISE="no"
if [[ "${TTM_PAGES}" =~ ^[0-9]+$ && "${POOL_PAGES}" =~ ^[0-9]+$ ]] && (( POOL_PAGES < TTM_PAGES )); then
    PARTIAL_RAISE="yes"
fi
BIG_PAGES=$(( RAM_GB * 1024 * 1024 * 1024 / PGSIZE ))   # pages to cover the full pool
if [[ "${PARTIAL_RAISE}" == "yes" ]]; then
    TTM_STATUS="warn"
    TTM_HINT="PARTIAL RAISE: ttm.pages_limit is ~${TTM_GB} GiB but ttm.page_pool_size is only ~${POOL_GB} GiB, so the ROCr/HIP pool (and every full-offload model) is still capped at ~${POOL_GB} GiB — a model larger than that 'fails to load'. Raise page_pool_size TOO: persist via poc/amd-rocm-bringup/config/ (kernel cmdline 'amdgpu.gttsize=-1 ttm.pages_limit=${BIG_PAGES} ttm.page_pool_size=${BIG_PAGES}' or /etc/modprobe.d), update-grub/initramfs, reboot; then re-check. See poc/amd-rocm-bringup/TTM-RUNBOOK.md."
elif [[ "${TTM_STATUS}" != "ok" ]]; then
    TTM_HINT="raise the GTT/TTM page limits to use the full ${RAM_GB} GB pool. Boot-time (persistent): add 'amdgpu.gttsize=-1 ttm.pages_limit=${BIG_PAGES} ttm.page_pool_size=${BIG_PAGES}' to the kernel cmdline (N = bytes/PAGE_SIZE, ~${BIG_PAGES} pages for ${RAM_GB} GB on ${PGSIZE}B pages), update-grub, reboot. Config files + runbook: poc/amd-rocm-bringup/config/ + TTM-RUNBOOK.md. Read-only check: cat /sys/module/ttm/parameters/{pages_limit,page_pool_size}"
else
    TTM_HINT=""
fi
add_check "ttm_gtt" "${TTM_STATUS}" "${TTM_DETAIL}" "${TTM_HINT}"

# ---- 9. Docker device-mount readiness (informational) ----------------------
if have docker; then
    add_check "docker" "ok" "docker present — mount ROCm with --device=/dev/kfd --device=/dev/dri --group-add video --group-add render" ""
else
    add_check "docker" "warn" "docker not on PATH (only needed for containerised ROCm)" \
        "install docker if you want to run ROCm inside a container; pass --device=/dev/kfd --device=/dev/dri --group-add video,render"
fi

# ---- verdict ---------------------------------------------------------------
# READY == rocm enumerates gfx1151 AND device nodes readable AND no hard fail.
N_FAIL=0; N_WARN=0
for s in "${CHECK_STATUS[@]}"; do
    [[ "$s" == "fail" ]] && N_FAIL=$((N_FAIL+1))
    [[ "$s" == "warn" ]] && N_WARN=$((N_WARN+1))
done
READY="false"
if [[ "${ROCM_OK}" == "yes" && "${KFD_OK}" == "yes" && "${DRI_OK}" == "yes" && "${N_FAIL}" -eq 0 ]]; then
    READY="true"
fi

# ---- emit JSON -------------------------------------------------------------
{
    printf '{\n'
    printf '  "schema": "amd-rocm-bringup/1",\n'
    printf '  "generated_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "host": "%s",\n' "$(jstr "$(hostname 2>/dev/null || echo unknown)")"
    printf '  "kernel": "%s",\n' "$(jstr "${KVER}")"
    printf '  "gfx": "%s",\n' "$(jstr "${GFX}")"
    printf '  "rocm": %s,\n' "$([[ "${ROCM_OK}" == yes ]] && echo true || echo false)"
    printf '  "hipcc": "%s",\n' "$(jstr "${HIPCC_VER}")"
    printf '  "unified_ram_gb": %s,\n' "${RAM_GB}"
    printf '  "ttm_pages_limit": "%s",\n' "$(jstr "${TTM_PAGES}")"
    printf '  "ttm_pages_limit_gb": "%s",\n' "$(jstr "${TTM_GB}")"
    printf '  "ttm_page_pool_size": "%s",\n' "$(jstr "${POOL_PAGES}")"
    printf '  "ttm_page_pool_size_gb": "%s",\n' "$(jstr "${POOL_GB}")"
    printf '  "ttm_effective_pool_gb": "%s",\n' "$(jstr "${EFF_GB}")"
    printf '  "amdgpu_gttsize_mib": "%s",\n' "$(jstr "${GTTSIZE_MIB}")"
    printf '  "needs_hsa_override": %s,\n' "$([[ "${NEED_OVERRIDE}" == yes ]] && echo true || echo false)"
    printf '  "ready": %s,\n' "${READY}"
    printf '  "n_fail": %s,\n' "${N_FAIL}"
    printf '  "n_warn": %s,\n' "${N_WARN}"
    printf '  "checks": [\n'
    last=$(( ${#CHECK_NAMES[@]} - 1 ))
    for i in "${!CHECK_NAMES[@]}"; do
        comma=","; [[ "$i" -eq "$last" ]] && comma=""
        printf '    {"name": "%s", "status": "%s", "detail": "%s", "hint": "%s"}%s\n' \
            "$(jstr "${CHECK_NAMES[$i]}")" "$(jstr "${CHECK_STATUS[$i]}")" \
            "$(jstr "${CHECK_DETAIL[$i]}")" "$(jstr "${CHECK_HINT[$i]}")" "${comma}"
    done
    printf '  ],\n'
    printf '  "honesty": "iGPU/NPU accelerate AI models; iGPU OpenCL accelerates SNARK primitives (size-gated); RISC0 STARK is CPU-only on AMD. This probe reports bring-up readiness only and makes no proving claim."\n'
    printf '}\n'
} >"${REPORT}"

# ---- human summary to stdout -----------------------------------------------
for i in "${!CHECK_NAMES[@]}"; do
    case "${CHECK_STATUS[$i]}" in
        ok)   mark="[ OK ]";;
        warn) mark="[WARN]";;
        *)    mark="[FAIL]";;
    esac
    printf '%s %-14s %s\n' "${mark}" "${CHECK_NAMES[$i]}" "${CHECK_DETAIL[$i]}"
    [[ -n "${CHECK_HINT[$i]}" && "${CHECK_STATUS[$i]}" != "ok" ]] && printf '        fix: %s\n' "${CHECK_HINT[$i]}"
done
echo "----------------------------------------------------------------------"
echo "verdict: ready=${READY}  (fails=${N_FAIL} warns=${N_WARN})  gfx=${GFX} kernel=${KVER}"
echo "report : ${REPORT}"
echo "honesty: this probe reports bring-up readiness only; the iGPU never proves."
exit 0
