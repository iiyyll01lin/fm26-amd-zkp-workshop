#!/usr/bin/env bash
# solo-guard.sh — contention guard for the GPU/CPU benchmarks.
#
# WHY THIS EXISTS: the published Demo C headline ("iGPU drives the DeciderEth
# proof at 1.34x") was taken while a sibling track contended the machine. A
# later re-bench under contention showed the CPU baseline collapsing
# 173.8s -> 63.7s, which would REVERSE the claim. A benchmark must NEVER again
# silently record numbers taken under contention. This guard:
#   * measures iGPU busy % (rocm-smi), the 1-min CPU load average, and an
#     advisory lockfile BEFORE any timing starts;
#   * REFUSES to record (exits with a clear message) if the machine is contended;
#   * exposes the measured solo flag + load average so callers can stamp every
#     emitted CSV row with `solo` (true/false) + `loadavg` columns.
#
# This file defines FUNCTIONS and override-friendly DEFAULTS only — sourcing it
# has no side effects.
#
# Public API:
#   solo_guard_probe         populate SOLO_* globals (GPU %, loadavg, ok flag, reason)
#   solo_guard_report        echo a human-readable contention report to stderr
#   solo_guard_require       probe + report; exit $SOLO_GUARD_EXIT (42) if contended
#                            (set ZKP_SOLO_OVERRIDE=1 to bypass the gate; the run is
#                            then stamped solo=false so contamination stays visible)
#
# Globals set by solo_guard_probe (for CSV columns):
#   SOLO_GPU_PCT   current iGPU busy %         (or "na" if rocm-smi unavailable)
#   SOLO_LOADAVG   current 1-min load average
#   SOLO_STATUS    "true" | "false"            (true == solo / safe to record)
#   SOLO_OK        0 (solo) | 1 (contended)
#   SOLO_REASON    short human reason when contended
#
# Tunables (env):
#   ZKP_GPU_BUSY_MAX   max iGPU busy % to still count as solo   (default 25)
#   ZKP_LOADAVG_MAX    max 1-min load avg                       (default nproc*0.25)
#   ZKP_GPU_LOCK       advisory lockfile path                   (default /tmp/zkp-gpu.lock)
#   ZKP_SOLO_OVERRIDE  =1 bypass the gate (records solo=false; explicit debugging only)

SOLO_GUARD_EXIT="${SOLO_GUARD_EXIT:-42}"

# --- iGPU busy % via rocm-smi (returns "na" if unavailable) -------------------
solo_guard_gpu_pct() {
    local pct=""
    if command -v rocm-smi >/dev/null 2>&1; then
        # rocm-smi --showuse --json -> {"card0": {"GPU use (%)": "17"}}
        pct="$(rocm-smi --showuse --json 2>/dev/null \
                 | grep -oE '"GPU use \(%\)": "[0-9]+"' \
                 | grep -oE '[0-9]+' | sort -rn | head -1)"
    fi
    [[ -z "${pct}" ]] && pct="na"
    printf '%s' "${pct}"
}

# --- 1-minute load average ----------------------------------------------------
solo_guard_loadavg() {
    awk '{print $1}' /proc/loadavg 2>/dev/null || echo "0"
}

# --- populate SOLO_* globals --------------------------------------------------
solo_guard_probe() {
    local gpu_busy_max load_max ncpu lockfile
    gpu_busy_max="${ZKP_GPU_BUSY_MAX:-25}"
    ncpu="$(nproc 2>/dev/null || echo 8)"
    # default loadavg ceiling: a quarter of the cores busy before the bench starts
    load_max="${ZKP_LOADAVG_MAX:-$(awk -v n="${ncpu}" 'BEGIN{printf "%.2f", n*0.25}')}"
    lockfile="${ZKP_GPU_LOCK:-/tmp/zkp-gpu.lock}"

    SOLO_GPU_PCT="$(solo_guard_gpu_pct)"
    SOLO_LOADAVG="$(solo_guard_loadavg)"
    SOLO_OK=0
    SOLO_REASON="solo"
    SOLO_GPU_BUSY_MAX="${gpu_busy_max}"
    SOLO_LOAD_MAX="${load_max}"

    # 1) advisory lockfile held by another benchmark process
    if [[ -f "${lockfile}" ]]; then
        local holder owner_pid
        holder="$(cat "${lockfile}" 2>/dev/null)"
        owner_pid="${holder%% *}"
        if [[ -n "${owner_pid}" && "${owner_pid}" =~ ^[0-9]+$ ]] && kill -0 "${owner_pid}" 2>/dev/null && [[ "${owner_pid}" != "$$" ]]; then
            SOLO_OK=1; SOLO_REASON="gpu lockfile held by pid ${holder}"
        fi
    fi

    # 2) iGPU busy above threshold
    if [[ "${SOLO_OK}" -eq 0 && "${SOLO_GPU_PCT}" != "na" ]]; then
        if (( SOLO_GPU_PCT > gpu_busy_max )); then
            SOLO_OK=1; SOLO_REASON="iGPU busy ${SOLO_GPU_PCT}% > ${gpu_busy_max}%"
        fi
    fi

    # 3) CPU load average above threshold (the CPU baseline's contamination axis)
    if [[ "${SOLO_OK}" -eq 0 ]]; then
        if awk -v l="${SOLO_LOADAVG}" -v m="${load_max}" 'BEGIN{exit !(l>m)}'; then
            SOLO_OK=1; SOLO_REASON="loadavg ${SOLO_LOADAVG} > ${load_max}"
        fi
    fi

    if [[ "${SOLO_OK}" -eq 0 ]]; then SOLO_STATUS="true"; else SOLO_STATUS="false"; fi
}

# --- human-readable report to stderr ------------------------------------------
solo_guard_report() {
    {
        echo "------------------------------------------------------------------"
        echo "[solo-guard] iGPU busy : ${SOLO_GPU_PCT}%  (max ${SOLO_GPU_BUSY_MAX}%)"
        echo "[solo-guard] loadavg(1m): ${SOLO_LOADAVG}   (max ${SOLO_LOAD_MAX}, nproc=$(nproc 2>/dev/null||echo '?'))"
        echo "[solo-guard] lockfile  : ${ZKP_GPU_LOCK:-/tmp/zkp-gpu.lock}"
        if [[ "${SOLO_OK}" -eq 0 ]]; then
            echo "[solo-guard] STATUS    : SOLO (safe to record)"
        else
            echo "[solo-guard] STATUS    : CONTENDED — ${SOLO_REASON}"
        fi
        echo "------------------------------------------------------------------"
    } >&2
}

# --- gate: refuse to record under contention ----------------------------------
solo_guard_require() {
    solo_guard_probe
    solo_guard_report
    if [[ "${SOLO_OK}" -ne 0 ]]; then
        if [[ "${ZKP_SOLO_OVERRIDE:-0}" == "1" ]]; then
            echo "[solo-guard] ZKP_SOLO_OVERRIDE=1 set — recording anyway, solo=false" >&2
            SOLO_STATUS="false"
            return 0
        fi
        echo "[solo-guard] REFUSING to record a contended benchmark (${SOLO_REASON})." >&2
        echo "[solo-guard] Wait for a solo window or set ZKP_SOLO_OVERRIDE=1 to force." >&2
        exit "${SOLO_GUARD_EXIT}"
    fi
}
