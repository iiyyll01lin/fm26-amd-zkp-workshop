#!/usr/bin/env bash
# run-stage1.sh — Stage 1 bit-for-bit gate: BabyBear Fp/Fp4 + Poseidon2 + SHA-256,
# ported to native HIP, validated GPU(target) == risc0 CpuHal.
#
#   1. dump golden vectors from the pinned risc0 CpuHal (pure CPU)
#   2. hipcc-compile the 3 harnesses (single-TU, __host__ __device__ headers)
#   3. run each on the selected gfx target: assert every op == CpuHal golden
#
# Correctness is contention-independent (deterministic bitwise equality), so this
# gate does NOT need a solo window; timing/bench (Stage 4) does.
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
LOG="${ART}/stage1-run.log"
: > "${LOG}"

echo "==> [1/3] dump golden vectors from pinned risc0 CpuHal" | tee -a "${LOG}"
( cd "${FORK}" && cargo run -q -p risc0-zkp --example dump_stage1_vectors -- \
    "${POC}/hip/vectors" "${POC}/kernels/hip/poseidon2_constants_generated.hpp" ) 2>&1 | tee -a "${LOG}"

echo "==> [2/3] hipcc compile harnesses (--offload-arch=${ARCH})" | tee -a "${LOG}"
for t in field_test poseidon2_test sha256_test; do
    hipcc --offload-arch="${ARCH}" -O3 -std=c++17 "${POC}/hip/${t}.hip" \
        -I"${POC}/kernels/hip" -o "${POC}/build/${t}" 2>/dev/null
    echo "    built ${t}" | tee -a "${LOG}"
done

echo "==> [3/3] run bit-for-bit gates on ${ARCH}" | tee -a "${LOG}"
rc=0
"${POC}/build/field_test"     "${POC}/hip/vectors/field.txt"     | tee -a "${LOG}" || rc=1
"${POC}/build/poseidon2_test" "${POC}/hip/vectors/poseidon2.txt" | tee -a "${LOG}" || rc=1
"${POC}/build/sha256_test"    "${POC}/hip/vectors/sha256.txt"    | tee -a "${LOG}" || rc=1

if [[ "${rc}" -eq 0 ]]; then
    echo "STAGE 1: PASS (all field/Poseidon2/SHA ops GPU==CPU bit-for-bit)" | tee -a "${LOG}"
else
    echo "STAGE 1: FAIL" | tee -a "${LOG}"
fi
exit "${rc}"
