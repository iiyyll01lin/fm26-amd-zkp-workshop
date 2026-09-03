#!/usr/bin/env bash
# run-zkrag-bn254-onchain.sh — STEP 3 driver.
#
# Re-casts the zkRAG HNSW retrieval relation (the four checks + algebraic Horner
# commitment) to an arkworks `ConstraintSynthesizer<ark_bn254::Fr>`, proves it
# with the Demo C vendored+patched `ark-groth16` (BN254) — i.e. the SAME G1+G2+FFT
# iGPU offload seam — emits a snarkJS-style Groth16 Solidity verifier (sonobe
# `solidity_verifiers::g16`), and replays the proof on a local anvil (mirrors
# 08-fold-steps.sh's NovaDecider replay). A clean run == the BN254 zkRAG proof
# verifies on-chain.
#
# CPU by default (the vendored prover is byte-identical CPU with the gpu-msm seam
# dormant; Demo C proved GPU==CPU bit-for-bit, so the on-chain verification is
# identical whether GPU or CPU produced the proof). The iGPU-accelerated *timing*
# variant is opt-in and solo-guarded:
#
#   # CPU correctness + on-chain replay (no GPU; always safe):
#   bash scripts/run-zkrag-bn254-onchain.sh
#   # iGPU-offloaded prove (G1+G2+FFT), solo-gated (exit 42 under contention):
#   FOLD_GPU_MSM=1 FOLD_GPU_G2=1 FOLD_GPU_FFT=1 ZKRAG_BN254_POW=22 \
#       bash scripts/run-zkrag-bn254-onchain.sh
#
# Env:
#   ZKRAG_BN254_POW=p   pad the circuit up toward 2^p constraints (default: real
#                       ~2^13.4 size; raise to 22+ for a GPU-offload measurement).
#   FOLD_GPU_MSM=1      offload the BN254 G1 MSMs to the iGPU (+ FOLD_GPU_G2,
#                       FOLD_GPU_FFT for the wide path); triggers the solo guard.
#   ANVIL_PORT=8545     local anvil port (forge-lib.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FOLD_DIR}/../.." && pwd)"
ART="${FOLD_DIR}/artefacts/zkrag-bn254"
FORGE_DIR="${FOLD_DIR}/forge"
RISC0_SCRIPTS="${REPO_ROOT}/poc/risc0-cartesi-step-demo/scripts"

# shellcheck source=/dev/null
source "${RISC0_SCRIPTS}/forge-lib.sh"
# shellcheck source=/dev/null
source "${RISC0_SCRIPTS}/solo-guard.sh"

FORGE_STD_REPO="${FORGE_STD_REPO:-https://github.com/foundry-rs/forge-std}"
FORGE_STD_TAG="${FORGE_STD_TAG:-v1.9.5}"
ZKP_GPU_LOCK="${ZKP_GPU_LOCK:-/tmp/zkp-gpu.lock}"
GPU="${FOLD_GPU_MSM:-0}"

cleanup_all() { stop_anvil; [[ "${GPU}" == "1" ]] && rm -f "${ZKP_GPU_LOCK}"; return 0; }
trap cleanup_all EXIT

# --- build (from the crate dir so its rust-toolchain.toml = stable >=1.85 is used;
#     the repo-root default may be RISC0's 1.83 which cannot build the gpu-msm tree)
FEAT=()
[[ "${GPU}" == "1" ]] && FEAT=(--features gpu-msm)
echo "==> [zkrag-bn254] cargo build --release --bin zkrag-bn254-onchain ${FEAT[*]:-} (locked/offline)"
( cd "${FOLD_DIR}" && cargo build --release --bin zkrag-bn254-onchain "${FEAT[@]}" --locked --offline )

# --- SOLO GUARD: only gate the GPU-offloaded *timing* run; the CPU correctness +
#     on-chain replay produces no GPU timing, so it never needs the gate.
if [[ "${GPU}" == "1" ]]; then
    solo_guard_require                       # exits 42 under contention
    echo "$$ zkrag-bn254" >"${ZKP_GPU_LOCK}"
fi

# --- prove + native verify + emit Groth16Verifier.sol + proof.json -----------
echo "==> [zkrag-bn254] prove (FOLD_GPU_MSM=${GPU}) + native verify + emit Solidity"
( cd "${FOLD_DIR}" && ./target/release/zkrag-bn254-onchain )

for f in "${ART}/Groth16Verifier.sol" "${ART}/proof.json"; do
    [[ -s "${f}" ]] || { echo "ERROR: missing artefact ${f}" >&2; exit 8; }
done

# --- on-chain replay (forge build + anvil + verifyProof) ----------------------
echo "==> [zkrag-bn254] on-chain replay (anvil + Groth16Verifier.verifyProof)"
cp "${ART}/Groth16Verifier.sol" "${FORGE_DIR}/src/Groth16Verifier.sol"
mkdir -p "${FORGE_DIR}/lib"
clone_dep "${FORGE_STD_REPO}" "${FORGE_STD_TAG}" "${FORGE_DIR}/lib/forge-std"
ensure_foundry || exit 9
( cd "${FORGE_DIR}" && forge build )
start_anvil "${ART}/anvil.log" || exit 10
( cd "${FORGE_DIR}" && forge script script/VerifyZkRag.s.sol:VerifyZkRagScript \
    --rpc-url "${ANVIL_RPC}" --private-key "${ANVIL_KEY}" --broadcast -vv ) \
    | grep -E "VERIFIED ON-CHAIN" \
    || { echo "ERROR: zkRAG BN254 on-chain replay FAILED" >&2; exit 11; }

echo "==> [zkrag-bn254] DONE — zkRAG BN254 proof verified natively AND on-chain."
