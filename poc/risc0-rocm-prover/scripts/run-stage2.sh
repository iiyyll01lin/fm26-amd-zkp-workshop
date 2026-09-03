#!/usr/bin/env bash
# run-stage2.sh — Stage 2 bit-for-bit gate: polynomial HAL ops (NTT family,
# zk_shift, eltwise, fri_fold, batch_evaluate_any, mix_poly_coeffs) ported to
# native HIP, validated GPU(target) == risc0 CpuHal (the golden is the actual
# CpuHal run in the dumper). Contention-independent (no solo window needed).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC="$(cd "${SCRIPT_DIR}/.." && pwd)"
FORK="${RISC0_FORK_PATH:-${POC}/vendor/risc0}"
ART="${RISC0_ROCM_ARTIFACT_DIR:-${POC}/artefacts}"
# shellcheck source=scripts/rocm-arch.sh
source "${SCRIPT_DIR}/rocm-arch.sh"
risc0_rocm_resolve_arch
ARCH="${RISC0_HIP_OFFLOAD_ARCH}"
mkdir -p "${POC}/build" "${POC}/hip/vectors" "${ART}"
LOG="${ART}/stage2-run.log"
: > "${LOG}"

echo "==> [1/3] dump golden vectors by running each op on the real CpuHal" | tee -a "${LOG}"
( cd "${FORK}" && cargo run -q -p risc0-zkp --features prove --example dump_stage2_vectors -- \
    "${POC}/hip/vectors" "${POC}/kernels/hip/rou_constants_generated.hpp" ) 2>&1 | tee -a "${LOG}"

echo "==> [2/3] hipcc compile harness (--offload-arch=${ARCH})" | tee -a "${LOG}"
hipcc --offload-arch="${ARCH}" -O3 -std=c++17 "${POC}/hip/stage2_test.hip" \
    -I"${POC}/kernels/hip" -o "${POC}/build/stage2_test" 2>/dev/null
echo "    built stage2_test" | tee -a "${LOG}"

echo "==> [3/3] run bit-for-bit gate on ${ARCH}" | tee -a "${LOG}"
rc=0
"${POC}/build/stage2_test" "${POC}/hip/vectors/stage2.txt" | tee -a "${LOG}" || rc=1
if [[ "${rc}" -eq 0 ]]; then
    echo "STAGE 2: PASS" | tee -a "${LOG}"
else
    echo "STAGE 2: FAIL" | tee -a "${LOG}"
fi
exit "${rc}"
