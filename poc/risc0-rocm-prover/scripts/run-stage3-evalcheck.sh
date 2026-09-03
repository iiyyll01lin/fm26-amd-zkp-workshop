#!/usr/bin/env bash
# run-stage3-evalcheck.sh — Stage 3 hardest gate: the GENERATED rv32im
# eval_check `poly_fp` (26k LOC across 4 files) ported to HIP by swapping the
# field backend to the validated native-HIP babybear.hpp, checked bit-for-bit
# against risc0's own CPU C++ reference (cxx/rust_poly_fp_*.cpp) on identical
# random inputs. Correctness is contention-independent.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC="$(cd "${SCRIPT_DIR}/.." && pwd)"
FORK="${RISC0_FORK_PATH:-${POC}/vendor/risc0}"
ART="${RISC0_ROCM_ARTIFACT_DIR:-${POC}/artefacts}"
CUDA="${FORK}/risc0/circuit/rv32im-sys/kernels/cuda"
CXX="${FORK}/risc0/circuit/rv32im-sys/kernels/cxx"
SYSCXX="${FORK}/risc0/sys/cxx"
# shellcheck source=scripts/rocm-arch.sh
source "${SCRIPT_DIR}/rocm-arch.sh"
risc0_rocm_resolve_arch
ARCH="${RISC0_HIP_OFFLOAD_ARCH}"
WORK="${POC}/build/eval_check_hip"
LOG="${ART}/stage3-evalcheck-run.log"
mkdir -p "${WORK}/supra" "${ART}"
: > "${LOG}"

echo "==> [1/5] stage generated eval_check + field shim (no edits to generated code)" | tee -a "${LOG}"
cp "${CUDA}"/eval_check_0.cu "${CUDA}"/eval_check_1.cu "${CUDA}"/eval_check_2.cu \
   "${CUDA}"/eval_check_3.cu "${WORK}/"
# eval_check.cuh has no include guard; unity-including it 4x redefines its
# namespace-scope constants. Add #pragma once to the *copy* (generated logic
# untouched) so a single-TU unity build is legal.
{ echo "#pragma once"; cat "${CUDA}/eval_check.cuh"; } > "${WORK}/eval_check.cuh"
cp "${POC}/kernels/hip/babybear.hpp" "${WORK}/"
cp "${POC}/hip/eval_check_kernel.hip" "${WORK}/"
cp "${POC}/rocm-port/files/risc0/circuit/rv32im-sys/kernels/hip/eval_check_launch.hpp" \
   "${WORK}/"
cat > "${WORK}/supra/fp.h" <<'EOF'
#pragma once
// HIP field shim: risc0 supra Fp/FpExt -> validated native-HIP babybear.hpp.
#include "../babybear.hpp"
typedef FpExt Fp4;
EOF

echo "==> [2/5] compile risc0 CPU C++ poly_fp golden (g++, ~52k LOC)" | tee -a "${LOG}"
if [[ -f "${WORK}/libcpupolyfp.a" && "${STAGE3_REBUILD_CPU:-0}" != "1" ]]; then
    echo "    reuse cached libcpupolyfp.a (STAGE3_REBUILD_CPU=1 to force)" | tee -a "${LOG}"
else
    rm -f "${WORK}"/cpu_*.o "${WORK}/libcpupolyfp.a"
    for f in rust_poly_fp_0 rust_poly_fp_1 rust_poly_fp_2 rust_poly_fp_3 eval_check; do
        g++ -O2 -std=c++17 -fPIC -I"${SYSCXX}" -I"${CXX}" -c "${CXX}/${f}.cpp" -o "${WORK}/cpu_${f}.o" 2>>"${LOG}"
    done
    ar rcs "${WORK}/libcpupolyfp.a" "${WORK}"/cpu_*.o
    echo "    built libcpupolyfp.a" | tee -a "${LOG}"
fi

echo "==> [3/5] hipcc compile the HIP unity kernel (--offload-arch=${ARCH})" | tee -a "${LOG}"
if [[ -f "${WORK}/eval_check_kernel.o" && "${STAGE3_REBUILD_KERNEL:-0}" != "1" ]]; then
    echo "    reuse cached eval_check_kernel.o (STAGE3_REBUILD_KERNEL=1 to force)" | tee -a "${LOG}"
else
    ( cd "${WORK}" && hipcc --offload-arch="${ARCH}" -O3 -std=c++17 -fPIC -x hip \
        -c eval_check_kernel.hip -I. -o eval_check_kernel.o 2>>"${LOG}" )
    echo "    built eval_check_kernel.o" | tee -a "${LOG}"
fi

echo "==> [4/5] compile + link the gate" | tee -a "${LOG}"
hipcc -O3 -std=c++17 -c "${POC}/hip/eval_check_gate.cpp" -o "${WORK}/eval_check_gate.o" 2>>"${LOG}"
hipcc --offload-arch="${ARCH}" "${WORK}/eval_check_gate.o" "${WORK}/eval_check_kernel.o" \
    "${WORK}/libcpupolyfp.a" -lamdhip64 -o "${WORK}/eval_check_gate" 2>>"${LOG}"

echo "==> [5/5] run bit-for-bit gate on ${ARCH}" | tee -a "${LOG}"
rc=0
"${WORK}/eval_check_gate" | tee -a "${LOG}" || rc=1
if [[ "${rc}" -eq 0 ]]; then
    echo "STAGE 3 (eval_check): PASS" | tee -a "${LOG}"
else
    echo "STAGE 3 (eval_check): FAIL" | tee -a "${LOG}"
fi
exit "${rc}"
