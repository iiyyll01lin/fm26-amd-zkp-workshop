#!/usr/bin/env bash
# run-stage3-merkle.sh — Stage 3 Merkle build gate: hash_rows + hash_fold for
# SHA-256 and Poseidon2, ported to native HIP, validated GPU(target) == CpuHal.
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
LOG="${ART}/stage3-merkle-run.log"
: > "${LOG}"

echo "==> [1/3] dump CpuHal hash_rows/hash_fold golden (SHA + Poseidon2)" | tee -a "${LOG}"
( cd "${FORK}" && cargo run -q -p risc0-zkp --features prove --example dump_stage3_merkle -- \
    "${POC}/hip/vectors" ) 2>&1 | tee -a "${LOG}"
echo "==> [2/3] hipcc compile harness" | tee -a "${LOG}"
hipcc --offload-arch="${ARCH}" -O3 -std=c++17 "${POC}/hip/merkle_test.hip" \
    -I"${POC}/kernels/hip" -o "${POC}/build/merkle_test" 2>/dev/null
echo "==> [3/3] run bit-for-bit gate on ${ARCH}" | tee -a "${LOG}"
rc=0
"${POC}/build/merkle_test" "${POC}/hip/vectors/merkle.txt" | tee -a "${LOG}" || rc=1
if [[ "${rc}" -eq 0 ]]; then
    echo "STAGE 3 (merkle): PASS" | tee -a "${LOG}"
else
    echo "STAGE 3 (merkle): FAIL" | tee -a "${LOG}"
fi
exit "${rc}"
