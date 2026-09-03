#!/usr/bin/env bash
# Shared ROCm target selection for the RISC0-HIP scripts.
#
# Selection order:
#   1. RISC0_HIP_OFFLOAD_ARCH supplied by the caller;
#   2. the single gfx target reported by rocminfo or rocm-smi;
#   3. gfx1151, retained as the documented compatibility default.
#
# If more than one target is detected, selection is ambiguous and the caller
# must set RISC0_HIP_OFFLOAD_ARCH explicitly.

risc0_rocm_detect_arches() {
    local output=""

    if command -v rocminfo >/dev/null 2>&1; then
        output="$(
            rocminfo 2>/dev/null \
                | awk '$1 == "Name:" && $2 ~ /^gfx[0-9a-f]+$/ { print $2 }' \
                | sort -u
        )"
    fi

    if [[ -z "${output}" ]] && command -v rocm-smi >/dev/null 2>&1; then
        output="$(
            rocm-smi --showproductname 2>/dev/null \
                | awk '
                    {
                        for (i = 1; i <= NF; i++) {
                            if ($i ~ /^gfx[0-9a-f]+$/) {
                                gsub(/[^0-9a-fx]/, "", $i)
                                print $i
                            }
                        }
                    }
                ' \
                | sort -u
        )"
    fi

    printf '%s\n' "${output}"
}

risc0_rocm_resolve_arch() {
    local requested="${RISC0_HIP_OFFLOAD_ARCH:-}"
    local detected=""
    local -a arches=()

    if [[ -n "${requested}" ]]; then
        if [[ ! "${requested}" =~ ^gfx[0-9a-f]+$ ]]; then
            echo "invalid RISC0_HIP_OFFLOAD_ARCH=${requested}; expected gfx<hex>" >&2
            return 2
        fi
        RISC0_HIP_OFFLOAD_ARCH_SOURCE="explicit"
    else
        detected="$(risc0_rocm_detect_arches)"
        if [[ -n "${detected}" ]]; then
            mapfile -t arches <<<"${detected}"
        fi
        case "${#arches[@]}" in
            0)
                requested="gfx1151"
                RISC0_HIP_OFFLOAD_ARCH_SOURCE="default"
                echo "warning: no ROCm gfx target detected; defaulting to gfx1151" >&2
                ;;
            1)
                requested="${arches[0]}"
                RISC0_HIP_OFFLOAD_ARCH_SOURCE="detected"
                ;;
            *)
                echo "multiple ROCm targets detected: ${arches[*]}" >&2
                echo "set RISC0_HIP_OFFLOAD_ARCH explicitly" >&2
                return 2
                ;;
        esac
    fi

    RISC0_HIP_OFFLOAD_ARCH="${requested}"
    export RISC0_HIP_OFFLOAD_ARCH RISC0_HIP_OFFLOAD_ARCH_SOURCE
}
