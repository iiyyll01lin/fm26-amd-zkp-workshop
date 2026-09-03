#!/usr/bin/env bash
# run-stage4-bench.sh — honest iGPU-vs-32t-CPU bench, same fork code (rocm vs
# no-rocm), solo-guarded. Emits artefacts/stage4-bench.csv.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC="$(cd "${SCRIPT_DIR}/.." && pwd)"
FORK="${POC}/vendor/risc0"
DEMO="$(cd "${POC}/../risc0-cartesi-step-demo" && pwd)"
ART="${POC}/artefacts"
REPS="${STAGE4_REPS:-2}"
# shellcheck source=scripts/rocm-arch.sh
source "${SCRIPT_DIR}/rocm-arch.sh"
risc0_rocm_resolve_arch
ARCH="${RISC0_HIP_OFFLOAD_ARCH}"
mkdir -p "${ART}"

# Solo-guard: refuse to record under contention (records solo flag either way).
source "${DEMO}/scripts/solo-guard.sh"
solo_guard_probe
solo_guard_report

ELF="${DEMO}/dist/cartesi-risc0-guest-step-prover.bin"
IN="${DEMO}/artefacts/step.bin"
IMG="$(tr -d '[:space:]' < "${DEMO}/dist/cartesi-risc0-guest-step-prover-image-id.txt")"

echo "==> building fork r0vm (rocm) and (cpu)"
( cd "${FORK}" && RISC0_HIP_OFFLOAD_ARCH="${ARCH}" cargo build -p risc0-r0vm --features rocm --release )
cp "${FORK}/target/release/r0vm" /tmp/r0vm-rocm
( cd "${FORK}" && cargo build -p risc0-r0vm --release )
cp "${FORK}/target/release/r0vm" /tmp/r0vm-cpu

bench() { # tag bin
    local tag="$1" bin="$2" best=99999 d
    for r in $(seq 1 "${REPS}"); do
        local t0 t1
        t0=$(date +%s.%N)
        env -u RISC0_ROCM_WITGEN -u RISC0_ROCM_ACCUM \
            RISC0_HIP_OFFLOAD_ARCH="${ARCH}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-32}" \
            "${bin}" --elf "${ELF}" \
            --initial-input "${IN}" --receipt "/tmp/b.${tag}.bin" >/dev/null 2>&1
        t1=$(date +%s.%N)
        d=$(awk "BEGIN{printf \"%.1f\", ${t1}-${t0}}")
        echo "  ${tag} run${r}: ${d}s"
        awk "BEGIN{exit !(${d}<${best})}" && best="${d}"
    done
    echo "${best}"
}

CPU_BEST=$(bench cpu /tmp/r0vm-cpu | tail -1)
GPU_BEST=$(bench gpu /tmp/r0vm-rocm | tail -1)
CPU_V=$(cargo risczero verify --path /tmp/b.cpu.bin "${IMG}" 2>&1 | tail -1)
GPU_V=$(cargo risczero verify --path /tmp/b.gpu.bin "${IMG}" 2>&1 | tail -1)

{
    echo "config,backend,threads,wall_s,receipt_bytes,verify,solo,loadavg"
    echo "fork-cpu,cpu(no-rocm),${RAYON_NUM_THREADS:-32},${CPU_BEST},$(stat -c%s /tmp/b.cpu.bin),${CPU_V},${SOLO_STATUS},${SOLO_LOADAVG}"
    echo "fork-gpu,rocm ${ARCH},${RAYON_NUM_THREADS:-32},${GPU_BEST},$(stat -c%s /tmp/b.gpu.bin),${GPU_V},${SOLO_STATUS},${SOLO_LOADAVG}"
} | tee "${ART}/stage4-bench.csv"
awk -v c="${CPU_BEST}" -v g="${GPU_BEST}" 'BEGIN{printf "iGPU speedup vs fork-CPU: %.2fx (solo=%s)\n", c/g, "'"${SOLO_STATUS}"'"}'
