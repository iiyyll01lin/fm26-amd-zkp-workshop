#!/usr/bin/env bash
# run-stage4-prove.sh — end-to-end: build r0vm --features rocm, prove the Demo B
# step on the selected ROCm target via HipHal, and verify the GPU-produced seal.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC="$(cd "${SCRIPT_DIR}/.." && pwd)"
FORK="${RISC0_FORK_PATH:-${POC}/vendor/risc0}"
DEMO="$(cd "${POC}/../risc0-cartesi-step-demo" && pwd)"
# shellcheck source=scripts/rocm-arch.sh
source "${SCRIPT_DIR}/rocm-arch.sh"
risc0_rocm_resolve_arch
ARCH="${RISC0_HIP_OFFLOAD_ARCH}"
ART="${RISC0_ROCM_ARTIFACT_DIR:-${POC}/artefacts}"
mkdir -p "${ART}"

ELF="${DEMO}/dist/cartesi-risc0-guest-step-prover.bin"
IN="${DEMO}/artefacts/step.bin"
IMG="$(tr -d '[:space:]' < "${DEMO}/dist/cartesi-risc0-guest-step-prover-image-id.txt")"
PROOF="${RISC0_ROCM_PROOF_PATH:-${ART}/step.rocm.proof.bin}"
mkdir -p "$(dirname "${PROOF}")"

echo "==> [1/3] build r0vm --features rocm (release)"
( cd "${FORK}" && RISC0_HIP_OFFLOAD_ARCH="${ARCH}" cargo build -p risc0-r0vm --features rocm --release )
TARGET_DIR="${CARGO_TARGET_DIR:-${FORK}/target}"
R0VM="${TARGET_DIR}/release/r0vm"

echo "==> [2/3] prove Demo B step on ${ARCH} (real STARK)"
rm -f "${PROOF}"
env -u RISC0_ROCM_WITGEN -u RISC0_ROCM_ACCUM \
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-32}" "${R0VM}" \
    --elf "${ELF}" --initial-input "${IN}" --receipt "${PROOF}"

echo "==> [3/3] cargo risczero verify (real STARK seal from the GPU path)"
cargo risczero verify --path "${PROOF}" "${IMG}"
echo "seal: ${PROOF} ($(stat -c%s "${PROOF}") bytes)"
