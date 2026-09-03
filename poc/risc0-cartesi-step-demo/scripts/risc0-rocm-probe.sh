#!/usr/bin/env bash
# risc0-rocm-probe.sh — PURE READ-ONLY probe documenting whether the Demo B
# main line (RISC0 r0vm STARK + Groth16 wrap) can be GPU-accelerated on AMD.
# It CHANGES NOTHING: no clone, no cargo build, no install. It only reads which
# GPU toolchains exist locally (hipcc / nvcc / r0vm) and states the upstream
# reality.
#
#   bash poc/risc0-cartesi-step-demo/scripts/risc0-rocm-probe.sh
#
# RESEARCH-ONLY + NEVER HARD-FAILS (always exits 0 / returns 0 when sourced).
# This started as the Path E Tier 3 "negative result" (no AMD GPU prover for the
# RISC0 zkVM STARK, official OR community). As of 2026-07 it is a *measured
# partial positive* — see the UPDATE below.
#
# THE ORIGINAL FINDING (verified 2026-06):
#   * Official RISC Zero accelerates the STARK prover on CUDA (NVIDIA) and
#     Metal (Apple) ONLY — the `cuda` / `metal` cargo features. There is no
#     `rocm` / `hip` / `vulkan` / `opencl` prover feature UPSTREAM.
#   * No credible community AMD fork exists. The most-cited fork, hemilabs/risc0
#     (a Hemi BTC-L2 fork), carries the SAME CUDA/Metal codebase — it adds no
#     HIP/ROCm prover.
#   * WHY: the STARK prover's hot kernels (NTT, hashing, accumulation) are
#     hand-written CUDA + Metal; an AMD path means reimplementing them in
#     HIP/ROCm (or HIPIFY-porting + re-tuning) and re-validating.
#
# UPDATE (2026-07) — poc/risc0-rocm-prover/ (path-i) — NOW END-TO-END:
#   * We forked risc0 @ v2.3.2 and ported the STARK prover's entire hot-kernel set
#     to native HIP (BabyBear Fp/Fp4, Poseidon2, SHA-256, NTT family, fri_fold/
#     mix/eltwise/…, the 26k-LOC generated rv32im eval_check, Merkle build), each
#     validated BIT-FOR-BIT vs risc0's own CpuHal/CPU-C++ on gfx1151.
#   * Then WIRED IT END-TO-END: a full HipHal (passes risc0's own DualHal 15/15) +
#     a rv32im HipCircuitHal (eval_check on iGPU; witgen/accum CPU-delegated over
#     unified memory) + a `rocm` cargo feature through zkvm->circuit->zkp->sys.
#   * RESULT: `r0vm --features rocm` proves the Demo B rv32im segment STARK (a
#     ~4-segment Cartesi-step composite; po2=20 = per-segment limit) on the gfx1151
#     iGPU (GPU 95% busy, marker fires 4x) and the STOCK `cargo risczero verify`
#     -> "Receipt is valid!" (independently audited, incl. a differential fallback
#     test). Honest bench (solo, same-code, 32t) on THIS workload: iGPU 26.1s vs
#     same-code CPU 142.4s = ~5.46x (flat ~5.3-5.5x across po2 16->21); the older
#     ~6.6-6.8x = 5.46x x a 1.25x local-vs-shipped codegen gap (installed rzup
#     r0vm 177.6s) -- a workload/hashfn-specific figure, not a general claim;
#     correctness is the hard guarantee. Still CPU: witgen/accum, recursion,
#     Groth16 wrap.
#
# WHERE THE WIN ACTUALLY IS (Path E Tier 1/2, already runnable on this box):
#   The cross-vendor OpenCL path (ec-gpu / bellperson) DOES accelerate SNARK
#   *primitives* (MSM/NTT) and a full Groth16 prover on this iGPU — see
#   poc/amd-gpu-zk-primitive-demo/. That is the AMD-GPU ZK story; it does not
#   move the Demo B zkVM STARK off the CPU.

risc0_rocm_probe() {
    local kver cpu_model
    local hipcc="no" hipcc_path="" hipcc_ver="n/a"
    local nvcc="no" nvcc_path=""
    local r0vm="no" r0vm_path="" r0vm_ver="n/a"
    local rocm="no" rocm_ver="n/a"
    local verdict d

    kver="$(uname -r 2>/dev/null || echo unknown)"
    cpu_model="$(LC_ALL=C lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
    [[ -z "${cpu_model}" ]] && cpu_model="unknown CPU"

    # ROCm HIP compiler (present on this box, but RISC0 has no HIP backend to feed it)
    if command -v hipcc >/dev/null 2>&1; then
        hipcc="yes"; hipcc_path="$(command -v hipcc)"
        hipcc_ver="$(hipcc --version 2>/dev/null | sed -n 's/.*HIP version:[[:space:]]*//p' | head -1)"
        [[ -n "${hipcc_ver}" ]] || hipcc_ver="installed"
    fi
    # CUDA toolkit (the ONLY toolchain RISC0's GPU prover can actually build)
    if command -v nvcc >/dev/null 2>&1; then
        nvcc="yes"; nvcc_path="$(command -v nvcc)"
    fi
    # RISC0 prover
    if command -v r0vm >/dev/null 2>&1; then
        r0vm="yes"; r0vm_path="$(command -v r0vm)"
        r0vm_ver="$(r0vm --version 2>/dev/null | head -1)"
        [[ -n "${r0vm_ver}" ]] || r0vm_ver="installed"
    fi
    for d in /opt/rocm/.info/version /opt/rocm-*/.info/version; do
        [[ -r "${d}" ]] || continue
        rocm_ver="$(cat "${d}" 2>/dev/null | head -1)"; rocm="yes"; break
    done

    # The verdict (2026-07): rv32im segment STARK now RUNS + VERIFIES on the AMD
    # iGPU via the risc0-rocm-prover fork (r0vm --features rocm).
    verdict="rv32im segment STARK now RUNS ON THE AMD iGPU end-to-end: r0vm --features rocm (poc/risc0-rocm-prover/) proves the Demo B ~4-segment Cartesi-step composite on gfx1151 and the stock 'cargo risczero verify' -> Receipt is valid! (GPU HipHal path, audited). On this workload ~5.46x vs same-code 32t CPU (flat ~5.3-5.5x; the old ~6.6-6.8x = 5.46x x a 1.25x local-vs-shipped codegen gap) -- workload/hashfn-specific, not general; correctness is the hard guarantee. Still CPU: witgen/accum, recursion, Groth16. The shipped r0vm 2.3.2 remains CPU-only until you build the fork."

    echo "######################################################################"
    echo "# RISC0 AMD-GPU prover probe ($(date -u +%Y-%m-%dT%H:%M:%SZ)) — READ-ONLY / research"
    echo "#   CPU             : ${cpu_model}"
    echo "#   Kernel          : ${kver}"
    echo "#   ROCm            : ${rocm} (version=${rocm_ver})"
    echo "#   hipcc (HIP)     : ${hipcc}${hipcc_path:+ (${hipcc_path})}  version=${hipcc_ver}"
    echo "#   nvcc (CUDA)     : ${nvcc}${nvcc_path:+ (${nvcc_path})}"
    echo "#   r0vm            : ${r0vm}${r0vm_path:+ (${r0vm_path})}  ${r0vm_ver}"
    echo "#"
    echo "#   RISC0 STARK GPU support (upstream) : CUDA + Metal ONLY (cargo: cuda, metal)."
    echo "#   AMD ROCm/HIP prover (upstream)     : DOES NOT EXIST (official or community)."
    echo "#   THIS REPO (2026-07, path-i)        : rv32im segment STARK RUNS + VERIFIES on the"
    echo "#                                        gfx1151 iGPU. r0vm --features rocm proves the"
    echo "#                                        Demo B step on GPU, 'cargo risczero verify' ->"
    echo "#                                        Receipt is valid! (poc/risc0-rocm-prover/)."
    echo "#"
    echo "#   VERDICT : ${verdict}"
    echo "#"
    echo "#   On this box: hipcc IS installed, and (as of path-i) risc0's rv32im"
    echo "#   segment STARK compiles + RUNS + VERIFIES on gfx1151 via r0vm --features"
    echo "#   rocm (built from the fork). nvcc is absent (official CUDA prover cannot"
    echo "#   build here); the SHIPPED r0vm 2.3.2 remains CPU-only until you build the"
    echo "#   fork. STARK math + eval_check on iGPU; witgen/accum still on CPU."
    echo "#"
    echo "#   The real AMD-GPU ZK win is Path E Tier 1/2 (ec-gpu/bellperson OpenCL"
    echo "#   for SNARK primitives + Groth16): poc/amd-gpu-zk-primitive-demo/."
    echo "#   That accelerates the SNARK layer, NOT the zkVM STARK main line."
    echo "######################################################################"

    cat <<'PATH_EOF'
# Status of the AMD r0vm STARK prover (poc/risc0-rocm-prover/, path-i) — DONE:
#   1. Option A (DONE): HIPIFY-ported risc0/zkp + risc0/circuit/rv32im kernels to
#      native HIP + a `rocm` cargo feature (build.rs -> hipcc). The generated
#      eval_check hipifies by swapping only the field backend; every kernel is
#      bit-for-bit vs risc0's CpuHal/CPU-C++ on gfx1151.
#   2. Wired end-to-end: HipHal (passes risc0's DualHal 15/15) + rv32im
#      HipCircuitHal (eval_check on iGPU, witgen/accum CPU-delegated over unified
#      memory) + rocm feature through zkvm->circuit->zkp->sys. `r0vm --features
#      rocm` proves the Demo B step on gfx1151 and the seal verifies for real.
#   3. Scoped out (still CPU): witgen/accum kernels (steps.cu), recursion, Groth16.
#      Option B (Vulkan) still N/A. Reproduce: poc/risc0-rocm-prover/scripts/run-stage4-*.sh
#
#   Full write-up: reading-notes/path-e-amd-gpu-zk-primitives.md (Tier 3)
#   Engine matrix : docs/amd-strix-halo-acceleration.md
PATH_EOF
}

risc0_rocm_probe || true

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exit 0
else
    return 0 2>/dev/null || true
fi
