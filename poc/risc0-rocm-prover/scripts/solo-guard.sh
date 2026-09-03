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
#   SOLO_GPU_PCT   busy % of the card under test (or "na" if rocm-smi unavailable)
#   SOLO_LOADAVG   current 1-min load average
#   SOLO_STATUS    "true" | "false"            (true == solo / safe to record)
#   SOLO_OK        0 (solo) | 1 (contended)
#   SOLO_REASON    short human reason when contended
#   SOLO_SCOPE     "card" | "machine"          (see SCOPE below)
#   SOLO_GPU_BDF   PCI bus id of the card under test ("" in machine scope)
#   SOLO_OTHER_GPU_PCT  busiest OTHER card's % ("na" in machine scope)
#   SOLO_OTHER_GPU_BDF  which other card that was
#
# SCOPE: with ZKP_TARGET_GPU_BDF set the busy check applies to THAT card only.
# A multi-card host running an unrelated job on a neighbour is not contention for
# a run that is isolated to a different card, and blocking on it would leave the
# only honest options as "wait forever" or ZKP_SOLO_OVERRIDE, which voids the
# evidence. The neighbours are still measured and stamped onto every row via
# SOLO_OTHER_GPU_PCT, so the reader can see the machine was shared. With
# ZKP_TARGET_GPU_BDF unset the behaviour is exactly the old machine-wide check,
# and on a single-GPU host the two scopes are identical -- so evidence recorded
# before this existed keeps its meaning.
#
# Tunables (env):
#   ZKP_TARGET_GPU_BDF card under test, e.g. 0000:03:00.0  (unset => machine scope)
#   ZKP_GPU_BUSY_MAX   max GPU busy % to still count as solo   (default 25)
#   ZKP_LOADAVG_MAX    max 1-min load avg                       (default nproc*0.25)
#   ZKP_GPU_LOCK       advisory lockfile path                   (default /tmp/zkp-gpu.lock)
#   ZKP_SOLO_OVERRIDE  =1 bypass the gate (records solo=false; explicit debugging only)

SOLO_GUARD_EXIT="${SOLO_GUARD_EXIT:-42}"

# --- per-card busy %, paired with the card's PCI bus id -----------------------
# rocm-smi --showuse --showbus --json emits one object per card:
#   {"card0": {"GPU use (%)": "0", "PCI Bus": "0000:03:00.0"}, "card1": {...}}
# Cards are separated by '}, "', which cannot appear inside a value here, so the
# use%/bus pairing survives without depending on a JSON parser being installed.
solo_guard_gpu_table() {
    command -v rocm-smi >/dev/null 2>&1 || return 0
    rocm-smi --showuse --showbus --json 2>/dev/null \
      | sed 's/}, *"/}\n"/g' \
      | while IFS= read -r line; do
            local pct bdf
            pct="$(printf '%s' "${line}" | grep -oE '"GPU use \(%\)": *"[0-9]+"' | grep -oE '[0-9]+' | head -1)"
            [[ -n "${pct}" ]] || continue
            bdf="$(printf '%s' "${line}" | grep -oiE '"PCI Bus": *"[^"]+"' \
                     | grep -oiE '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | head -1)"
            printf '%s %s\n' "${pct}" "$(printf '%s' "${bdf:-na}" | tr 'A-Z' 'a-z')"
        done
}

# --- populate the GPU half of the SOLO_* globals ------------------------------
solo_guard_gpu_scan() {
    local target other_max="" target_pct="" pct bdf found=1
    target="$(printf '%s' "${ZKP_TARGET_GPU_BDF:-}" | tr 'A-Z' 'a-z')"
    SOLO_OTHER_GPU_PCT="na"
    SOLO_OTHER_GPU_BDF=""
    SOLO_GPU_MISSING=0

    while read -r pct bdf; do
        if [[ -n "${target}" && "${bdf}" == "${target}" ]]; then
            target_pct="${pct}"; found=0; continue
        fi
        if [[ -z "${other_max}" ]] || (( pct > other_max )); then
            other_max="${pct}"; SOLO_OTHER_GPU_BDF="${bdf}"
        fi
    done < <(solo_guard_gpu_table)

    # no card requested -> the historical machine-wide maximum
    if [[ -z "${target}" ]]; then
        SOLO_SCOPE="machine"
        SOLO_GPU_BDF=""
        SOLO_OTHER_GPU_BDF=""
        SOLO_GPU_PCT="${other_max:-na}"
        return 0
    fi

    SOLO_SCOPE="card"
    SOLO_GPU_BDF="${target}"
    SOLO_OTHER_GPU_PCT="${other_max:-na}"
    # asked for a card that rocm-smi does not report: fail closed, never guess
    if (( found != 0 )); then
        SOLO_GPU_MISSING=1
        SOLO_GPU_PCT="na"
        return 0
    fi
    SOLO_GPU_PCT="${target_pct}"
}

# --- busy % of the scoped card, on stdout -------------------------------------
# The card-scoped rewrite replaced the old machine-wide solo_guard_gpu_pct with
# solo_guard_gpu_scan, but solo_guard_watch_start and three sibling run scripts
# still call it by this name; without it the watcher logged an empty first field
# every sample and the post-run "was the GPU actually used" check read that as
# ~0% and aborted the run. Callers use it in $(...), so the scan writes to the
# SOLO_* globals only inside the subshell.
solo_guard_gpu_pct() {
    solo_guard_gpu_scan
    printf '%s' "${SOLO_GPU_PCT:-na}"
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

    solo_guard_gpu_scan
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

    # 2) the card under test was requested but is not visible to rocm-smi
    if [[ "${SOLO_OK}" -eq 0 && "${SOLO_GPU_MISSING:-0}" -eq 1 ]]; then
        SOLO_OK=1; SOLO_REASON="target card ${SOLO_GPU_BDF} not reported by rocm-smi"
    fi

    # 3) the card under test is busy above threshold
    if [[ "${SOLO_OK}" -eq 0 && "${SOLO_GPU_PCT}" != "na" ]]; then
        if (( SOLO_GPU_PCT > gpu_busy_max )); then
            SOLO_OK=1; SOLO_REASON="iGPU busy ${SOLO_GPU_PCT}% > ${gpu_busy_max}%"
        fi
    fi

    # 4) CPU load average above threshold (the CPU baseline's contamination axis)
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
        if [[ "${SOLO_SCOPE:-machine}" == "card" ]]; then
            echo "[solo-guard] scope     : card ${SOLO_GPU_BDF}"
        else
            echo "[solo-guard] scope     : machine (busiest of all GPUs)"
        fi
        echo "[solo-guard] iGPU busy : ${SOLO_GPU_PCT}%  (max ${SOLO_GPU_BUSY_MAX}%)"
        if [[ "${SOLO_SCOPE:-machine}" == "card" ]]; then
            echo "[solo-guard] other GPUs: ${SOLO_OTHER_GPU_PCT}% busiest${SOLO_OTHER_GPU_BDF:+ (${SOLO_OTHER_GPU_BDF})} - stamped on every row, not a veto"
        fi
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
            # shellcheck disable=SC2034 # exported to sourcing benchmark scripts
            SOLO_STATUS="false"
            return 0
        fi
        echo "[solo-guard] REFUSING to record a contended benchmark (${SOLO_REASON})." >&2
        echo "[solo-guard] Wait for a solo window or set ZKP_SOLO_OVERRIDE=1 to force." >&2
        exit "${SOLO_GUARD_EXIT}"
    fi
}

# --- runtime observation during a timed run ------------------------------------
# solo_guard_probe only gates the START of a run. These helpers sample while the
# run is in flight and record what was observed, as ADDITIVE fields; existing
# SOLO_* semantics are untouched so older evidence stays comparable.
#
# What actually varies here is thermal/power state, not contention. On the paired
# A/B evidence the 1-minute load average correlates with run duration at r=+0.105
# while execution order correlates at r=+0.716, and the 7-11 s container stage
# shows no order effect at all (r=-0.112) because it ends before the package
# heats. So sclk/temperature/power are the fields worth recording; loadavg and
# GPU busy % are kept only as context and must not be read as an explanation.
#
# The one unambiguous external signal remains another live process taking the
# advisory lockfile, reported by SOLO_RUNTIME_LOCK_STOLEN.
SOLO_RUNTIME_MAX_GPU_PCT="na"
SOLO_RUNTIME_MAX_LOADAVG="na"
SOLO_RUNTIME_SAMPLES=0
SOLO_RUNTIME_LOCK_STOLEN="false"
SOLO_RUNTIME_MIN_SCLK_MHZ="na"
SOLO_RUNTIME_MAX_TEMP_C="na"
SOLO_RUNTIME_MAX_POWER_W="na"
SOLO_WATCH_PID=""
SOLO_WATCH_FILE=""

# sclk / edge temperature / package power in one rocm-smi call.
solo_guard_thermal() {
    local out sclk temp power
    out="$(rocm-smi --showclocks --showtemp --showpower 2>/dev/null)" || out=""
    sclk="$(printf '%s' "${out}" | grep -oE 'sclk clock level: [0-9]+: \(([0-9]+)Mhz\)' | grep -oE '[0-9]+Mhz' | grep -oE '[0-9]+' | head -1)"
    temp="$(printf '%s' "${out}" | grep -oE 'Temperature \(Sensor edge\) \(C\): [0-9.]+' | grep -oE '[0-9.]+$' | head -1)"
    power="$(printf '%s' "${out}" | grep -oE 'Power \(W\): [0-9.]+' | grep -oE '[0-9.]+$' | head -1)"
    printf '%s %s %s' "${sclk:-na}" "${temp:-na}" "${power:-na}"
}

solo_guard_watch_start() {
    local interval="${1:-${ZKP_SOLO_WATCH_INTERVAL:-2}}"
    local lockfile="${ZKP_GPU_LOCK:-/tmp/zkp-gpu.lock}"
    SOLO_WATCH_FILE="$(mktemp "${TMPDIR:-/tmp}/zkp-solo-watch.XXXXXX")"
    (
        while :; do
            printf '%s %s %s %s\n' \
                "$(solo_guard_gpu_pct)" \
                "$(solo_guard_loadavg)" \
                "$(solo_guard_thermal)" \
                "$(if [[ -f "${lockfile}" ]]; then
                       holder="$(cat "${lockfile}" 2>/dev/null)"
                       owner="${holder%% *}"
                       if [[ "${owner}" =~ ^[0-9]+$ ]] && [[ "${owner}" != "$$" ]] \
                          && kill -0 "${owner}" 2>/dev/null; then echo stolen; else echo free; fi
                   else echo free; fi)" >>"${SOLO_WATCH_FILE}"
            sleep "${interval}"
        done
    ) &
    SOLO_WATCH_PID=$!
}

solo_guard_watch_stop() {
    if [[ -n "${SOLO_WATCH_PID}" ]] && kill -0 "${SOLO_WATCH_PID}" 2>/dev/null; then
        kill "${SOLO_WATCH_PID}" 2>/dev/null || true
        wait "${SOLO_WATCH_PID}" 2>/dev/null || true
    fi
    SOLO_WATCH_PID=""
    SOLO_RUNTIME_MAX_GPU_PCT="na"
    SOLO_RUNTIME_MAX_LOADAVG="na"
    SOLO_RUNTIME_SAMPLES=0
    SOLO_RUNTIME_LOCK_STOLEN="false"
    SOLO_RUNTIME_MIN_SCLK_MHZ="na"
    SOLO_RUNTIME_MAX_TEMP_C="na"
    SOLO_RUNTIME_MAX_POWER_W="na"
    if [[ -n "${SOLO_WATCH_FILE}" && -s "${SOLO_WATCH_FILE}" ]]; then
        read -r SOLO_RUNTIME_MAX_GPU_PCT SOLO_RUNTIME_MAX_LOADAVG \
                SOLO_RUNTIME_MIN_SCLK_MHZ SOLO_RUNTIME_MAX_TEMP_C \
                SOLO_RUNTIME_MAX_POWER_W SOLO_RUNTIME_SAMPLES \
                SOLO_RUNTIME_LOCK_STOLEN < <(
            awk '{
                if ($1 != "na" && ($1 + 0) > g) g = $1 + 0
                if (($2 + 0) > l) l = $2 + 0
                if ($3 != "na" && (s == 0 || ($3 + 0) < s)) s = $3 + 0
                if ($4 != "na" && ($4 + 0) > t) t = $4 + 0
                if ($5 != "na" && ($5 + 0) > p) p = $5 + 0
                if ($6 == "stolen") stolen = 1
                n++
            } END {
                printf "%s %.2f %s %s %s %d %s\n",
                    (g == 0 ? "0" : g), l,
                    (s == 0 ? "na" : s), (t == 0 ? "na" : t), (p == 0 ? "na" : p),
                    n, (stolen ? "true" : "false")
            }' "${SOLO_WATCH_FILE}")
    fi
    [[ -z "${SOLO_WATCH_FILE}" ]] || rm -f "${SOLO_WATCH_FILE}"
    SOLO_WATCH_FILE=""
}

# Wait for a real solo window without allowing an override. Useful for serialized
# release gates that may be queued behind an unrelated, already-running job.
solo_guard_wait() {
    local max_wait="${1:-${ZKP_SOLO_WAIT_MAX:-7200}}"
    local interval="${2:-${ZKP_SOLO_WAIT_INTERVAL:-15}}"
    local waited=0
    [[ "${ZKP_SOLO_OVERRIDE:-0}" != "1" ]] || {
        echo "[solo-guard] override is forbidden while waiting for a clean window" >&2
        exit 2
    }
    while true; do
        solo_guard_probe
        solo_guard_report
        if [[ "${SOLO_OK}" -eq 0 ]]; then
            return 0
        fi
        if (( waited >= max_wait )); then
            echo "[solo-guard] no solo window after ${waited}s (${SOLO_REASON})" >&2
            exit "${SOLO_GUARD_EXIT}"
        fi
        sleep "${interval}"
        waited=$((waited + interval))
    done
}
