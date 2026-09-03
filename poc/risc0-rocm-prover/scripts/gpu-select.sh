#!/usr/bin/env bash
# gpu-select.sh -- pin a run to one physical AMD GPU, selected by PCI bus id.
#
# WHY THIS EXISTS: on a multi-card host the ROCr/HIP device order is NOT the
# rocm-smi order. rocm-smi enumerates by PCI bus; ROCr/HIP enumerate by KFD
# node id. On the R9700 host the two orders disagree completely:
#
#   rocm-smi GPU[0] 0000:03:00.0 == ROCr index 2
#   rocm-smi GPU[1] 0000:23:00.0 == ROCr index 3
#   rocm-smi GPU[2] 0000:d3:00.0 == ROCr index 1
#   rocm-smi GPU[3] 0000:f3:00.0 == ROCr index 0
#
# Asking for "device 0" therefore lands on a DIFFERENT card than the one
# rocm-smi reported idle, which can silently contend a neighbour's job and
# invalidate every measurement taken next to it. Callers must select by bus id
# and assert the resolved mapping; never pass a bare ordinal.
#
# Sourcing this file has no side effects: it defines functions and defaults.
#
# Public API:
#   zkp_gpu_rocr_index <bdf>        ROCr/HIP index for a PCI bus id
#   zkp_gpu_smi_index <bdf>         rocm-smi index for a PCI bus id
#   zkp_gpu_busy_pct <smi_index>    current busy percentage
#   zkp_gpu_vram_used <smi_index>   current VRAM bytes in use
#   zkp_gpu_report <bdf>            human-readable state of one card
#   zkp_gpu_require_idle <bdf>      fail closed unless that card is free
#   zkp_gpu_isolate <bdf>           export the visibility filter for that card
#
# Tunables (env):
#   ZKP_TARGET_GPU_BDF   default card                 (default 0000:03:00.0)
#   ZKP_GPU_CARD_BUSY_MAX   max busy % to call free  (default 5)
#   ZKP_GPU_CARD_VRAM_MAX   max VRAM bytes in use     (default 1073741824)

ZKP_TARGET_GPU_BDF="${ZKP_TARGET_GPU_BDF:-0000:03:00.0}"
ZKP_GPU_CARD_BUSY_MAX="${ZKP_GPU_CARD_BUSY_MAX:-5}"
ZKP_GPU_CARD_VRAM_MAX="${ZKP_GPU_CARD_VRAM_MAX:-1073741824}"
ZKP_GPU_SELECT_EXIT="${ZKP_GPU_SELECT_EXIT:-43}"

zkp_gpu_normalize_bdf() {
    local bdf
    bdf="$(printf '%s' "${1:-}" | tr 'A-Z' 'a-z')"
    case "${bdf}" in
        *:*:*) ;;
        *:*) bdf="0000:${bdf}" ;;
        *) return 2 ;;
    esac
    printf '%s' "${bdf}"
}

# Emits "<rocr_index> <bdf> <properties_path>" for every GPU-capable KFD node in
# ROCr enumeration order, which is ascending node id. CPU nodes report zero SIMDs
# and are skipped because ROCr does not give them device indices.
zkp_gpu_enumerate() {
    local node props simd domain location bus device bdf index=0
    for node in $(ls -1v /sys/class/kfd/kfd/topology/nodes 2>/dev/null); do
        props="/sys/class/kfd/kfd/topology/nodes/${node}/properties"
        [[ -r "${props}" ]] || continue
        simd="$(awk '/^simd_count /{print $2}' "${props}")"
        [[ -n "${simd}" && "${simd}" -gt 0 ]] || continue
        domain="$(awk '/^domain /{print $2}' "${props}")"
        location="$(awk '/^location_id /{print $2}' "${props}")"
        [[ -n "${location}" ]] || continue
        bus=$(( (location >> 8) & 0xff ))
        device=$(( (location >> 3) & 0x1f ))
        bdf="$(printf '%04x:%02x:%02x.0' "${domain:-0}" "${bus}" "${device}")"
        printf '%s %s %s\n' "${index}" "${bdf}" "${props}"
        index=$(( index + 1 ))
    done
}

zkp_gpu_rocr_index() {
    local target index bdf props
    target="$(zkp_gpu_normalize_bdf "${1:-${ZKP_TARGET_GPU_BDF}}")" || return 2
    while read -r index bdf props; do
        if [[ "${bdf}" == "${target}" ]]; then
            printf '%s' "${index}"
            return 0
        fi
    done < <(zkp_gpu_enumerate)
    echo "gpu-select: no KFD GPU node for ${target}" >&2
    return 1
}

# Reports the gfx target of one specific card, so a build targets the card the
# run will actually use instead of whatever a host-wide query happens to list
# first. KFD encodes the target as major*10000 + minor*100 + step, with minor and
# step rendered as hex digits (gfx90a is 90010, gfx1201 is 120001).
zkp_gpu_arch() {
    local target index bdf props version major minor step
    target="$(zkp_gpu_normalize_bdf "${1:-${ZKP_TARGET_GPU_BDF}}")" || return 2
    while read -r index bdf props; do
        [[ "${bdf}" == "${target}" ]] || continue
        version="$(awk '/^gfx_target_version /{print $2}' "${props}")"
        [[ -n "${version}" && "${version}" -gt 0 ]] || return 1
        major=$(( version / 10000 ))
        minor=$(( (version / 100) % 100 ))
        step=$(( version % 100 ))
        printf 'gfx%d%x%x' "${major}" "${minor}" "${step}"
        return 0
    done < <(zkp_gpu_enumerate)
    echo "gpu-select: no gfx target for ${target}" >&2
    return 1
}

zkp_gpu_smi_index() {
    local target
    target="$(zkp_gpu_normalize_bdf "${1:-${ZKP_TARGET_GPU_BDF}}")" || return 2
    rocm-smi --showbus 2>/dev/null | awk -v target="${target}" '
        /PCI Bus:/ {
            index_field = $0
            sub(/^GPU\[/, "", index_field)
            sub(/\].*/, "", index_field)
            if (tolower($NF) == target) { print index_field; found = 1; exit }
        }
        END { if (!found) exit 1 }
    '
}

zkp_gpu_busy_pct() {
    rocm-smi --showuse 2>/dev/null | awk -v idx="${1}" '
        /GPU use \(%\)/ {
            index_field = $0
            sub(/^GPU\[/, "", index_field)
            sub(/\].*/, "", index_field)
            if (index_field == idx) { print $NF; found = 1; exit }
        }
        END { if (!found) exit 1 }
    '
}

zkp_gpu_vram_used() {
    rocm-smi --showmeminfo vram 2>/dev/null | awk -v idx="${1}" '
        /VRAM Total Used Memory/ {
            index_field = $0
            sub(/^GPU\[/, "", index_field)
            sub(/\].*/, "", index_field)
            if (index_field == idx) { print $NF; found = 1; exit }
        }
        END { if (!found) exit 1 }
    '
}

zkp_gpu_report() {
    local bdf rocr smi busy vram
    bdf="$(zkp_gpu_normalize_bdf "${1:-${ZKP_TARGET_GPU_BDF}}")" || return 2
    rocr="$(zkp_gpu_rocr_index "${bdf}")" || return 1
    smi="$(zkp_gpu_smi_index "${bdf}")" || return 1
    busy="$(zkp_gpu_busy_pct "${smi}")" || busy="na"
    vram="$(zkp_gpu_vram_used "${smi}")" || vram="na"
    printf 'gpu bdf=%s rocr_index=%s smi_index=%s busy_pct=%s vram_used_bytes=%s\n' \
        "${bdf}" "${rocr}" "${smi}" "${busy}" "${vram}"
}

# Refuse to touch a card that another job is using. There is no override: a
# neighbour's run must never be perturbed, and a contended card must never be
# measured.
zkp_gpu_require_idle() {
    local bdf smi busy vram
    bdf="$(zkp_gpu_normalize_bdf "${1:-${ZKP_TARGET_GPU_BDF}}")" || return 2
    smi="$(zkp_gpu_smi_index "${bdf}")" || {
        echo "gpu-select: cannot read rocm-smi state for ${bdf}" >&2
        return "${ZKP_GPU_SELECT_EXIT}"
    }
    busy="$(zkp_gpu_busy_pct "${smi}")" || {
        echo "gpu-select: cannot read busy% for ${bdf}" >&2
        return "${ZKP_GPU_SELECT_EXIT}"
    }
    vram="$(zkp_gpu_vram_used "${smi}")" || {
        echo "gpu-select: cannot read VRAM use for ${bdf}" >&2
        return "${ZKP_GPU_SELECT_EXIT}"
    }
    if (( busy > ZKP_GPU_CARD_BUSY_MAX )); then
        echo "gpu-select: ${bdf} is busy ${busy}% > ${ZKP_GPU_CARD_BUSY_MAX}%; refusing" >&2
        return "${ZKP_GPU_SELECT_EXIT}"
    fi
    if (( vram > ZKP_GPU_CARD_VRAM_MAX )); then
        echo "gpu-select: ${bdf} holds ${vram} VRAM bytes > ${ZKP_GPU_CARD_VRAM_MAX}; refusing" >&2
        return "${ZKP_GPU_SELECT_EXIT}"
    fi
    return 0
}

# Filter at the ROCr level so the process can only ever see the chosen card,
# which then appears as HIP device 0. HIP_VISIBLE_DEVICES is cleared because a
# second filter would be applied to the already-filtered list.
zkp_gpu_isolate() {
    local bdf rocr
    bdf="$(zkp_gpu_normalize_bdf "${1:-${ZKP_TARGET_GPU_BDF}}")" || return 2
    rocr="$(zkp_gpu_rocr_index "${bdf}")" || return 1
    export ROCR_VISIBLE_DEVICES="${rocr}"
    unset HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES
    export ZKP_SELECTED_GPU_BDF="${bdf}"
    export ZKP_SELECTED_ROCR_INDEX="${rocr}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    zkp_gpu_report "${1:-${ZKP_TARGET_GPU_BDF}}"
fi
