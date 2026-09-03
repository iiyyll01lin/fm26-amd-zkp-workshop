# RISC0 v2.3.2 STARK GPU kernel inventory (Stage 0 recon)

Pin: `risc0/risc0 @ v2.3.2` (`218e3bc4a8ffcd203a9cd4e46f921bf60aa7e2bd`). All line
counts are `wc -l` of the actual pinned tree under `vendor/risc0/`.

## GPU HAL surface = `risc0_zkp::hal::Hal` + `CircuitHal` traits

`risc0/zkp/src/hal/mod.rs` defines the trait every backend implements. Golden
reference = `hal/cpu.rs` (`CpuHal`, 755 LOC). GPU backends: `hal/cuda.rs`
(`CudaHal`, 1069 LOC, uses `cust` + sppark) and `hal/metal.rs` (1033 LOC). The
**NEW** `hal/hip.rs` (`HipHal`) is the port target.

`hal/dual.rs` (`DualHal`) already runs each op on **two** HALs and asserts
equality — this is risc0's own bit-for-bit harness, reused verbatim for the
GPU==CPU gate (cf. `.github/workflows/gpu-cpu-equality.yml`).

### `Hal` trait method → CUDA kernel → stage

| trait method | CUDA kernel / FFI | source file | LOC | stage |
| --- | --- | --- | --- | --- |
| `alloc_*` / `copy_from_*` | `cust` DeviceBuffer | `hal/cuda.rs` | — | 2 |
| `batch_expand_into_evaluate_ntt` | `batch_expand` + `batch_evaluate_ntt` (SupraSeal) | `supra/ntt.cu`, `supra/api.cu` | 152+153 | 2 |
| `batch_interpolate_ntt` | `batch_interpolate_ntt` (SupraSeal) | `supra/ntt.cu` | ↑ | 2 |
| `batch_bit_reverse` | `batch_bit_reverse` | `supra/ntt.cu` / `kernels.cu` | 132 | 2 |
| `batch_evaluate_any` | `batch_evaluate_any` | `kernels.cu` | ↑ | 2 |
| `zk_shift` | `zk_shift` | `kernels.cu` | ↑ | 2 |
| `mix_poly_coeffs` | `mix_poly_coeffs` | `kernels.cu` | ↑ | 2 |
| `eltwise_add_elem` | `eltwise_add_fp` | `eltwise.cu` | 82 | 2 |
| `eltwise_sum_extelem` | `eltwise_sum_fpext` | `eltwise.cu` | ↑ | 2 |
| `eltwise_copy_elem[_slice]` | `eltwise_copy_fp[_region]` | `eltwise.cu` | ↑ | 2 |
| `eltwise_zeroize_elem` | `eltwise_zeroize_fp` | `eltwise.cu` | ↑ | 2 |
| `fri_fold` | `fri_fold` | `kernels.cu` | ↑ | 2 |
| `gather_sample` | `gather_sample` | `kernels.cu` | ↑ | 2 |
| `scatter` | `scatter` | `kernels.cu` | ↑ | 2 |
| `prefix_products` | `calc_prefix_operation` | `supra/calc_prefix_operation.cuh` | 193 | 2 |
| `combos_prepare` / `combos_divide` | (default impl, **CPU** poly_divide) | `hal/mod.rs` | — | — |
| `hash_rows` (SHA) | `risc0_zkp_cuda_sha_rows` | `sha.cu`, `sha256.h` | 29+237 | 1/3 |
| `hash_fold` (SHA) | `risc0_zkp_cuda_sha_fold` | `sha.cu` | ↑ | 1/3 |
| `hash_rows`/`hash_fold` (Poseidon2) | `sppark_poseidon2_rows`/`_fold` | `supra/poseidon2.cuh` | 163 | 1/3 |
| `CircuitWitnessGenerator::generate_witness` | `risc0_circuit_rv32im_cuda_witgen` → `par_stepExec<<<>>>` | `hal/cuda.rs:100`, `rv32im-sys/kernels/cuda/{ffi.cu,steps.cu}` | 30310 | 3 |
| `CircuitAccumulator::step_accum` | `risc0_circuit_rv32im_cuda_accum` → `stepAccum<<<>>>` + `thrust::inclusive_scan` + `finalizeAccum<<<>>>` | `hal/cuda.rs:155`, `rv32im-sys/kernels/cuda/ffi.cu` | ↑ | 3 |
| `CircuitHal::accumulate` | **empty on CUDA** (`hal/cuda.rs:160-170`); the work is in `step_accum` above | `hal/cuda.rs` | — | 3 |
| `CircuitHal::eval_check` | rv32im `eval_check_*` | `rv32im-sys/kernels/cuda/eval_check_{0..3}.cu` | **26263** | 3 |

**Note for the port.** The two rows above are the reason the shipped hybrid is a
*deviation* and not parity: **upstream CUDA runs witgen and accumulation on the
GPU unconditionally.** Our HIP `HipCircuitHal` defaults both to the CPU C++
goldens behind `RISC0_ROCM_WITGEN` / `RISC0_ROCM_ACCUM`
(`hal/hip.rs:75`, `:110`, `:125`, `:172`), for the measured reasons in
[`partb-status.md`](partb-status.md). Never describe the CPU delegation as
"matching the CUDA path".

### Field / hash primitive headers (Stage 1 port targets)

| header | what | LOC | representation |
| --- | --- | --- | --- |
| `sys/kernels/zkp/cuda/fp.h` | BabyBear `Fp`, P=15·2²⁷+1, M=0x88000001, R2=1172168163 | 196 | Montgomery |
| `sys/kernels/zkp/cuda/fpext.h` | `FpExt` = Fp⁴, x⁴−11 | 230 | Montgomery (4×Fp) |
| `sys/kernels/zkp/cuda/sha256.h` | SHA-256 compress + Merkle | 237 | big-endian words |
| `sys/kernels/zkp/cuda/supra/poseidon2.cuh` | Poseidon2-BabyBear (24 cells, 8 full + 21 partial) | 163 | non-Montgomery consts |
| `sys/kernels/zkp/cuda/supra/poseidon2_constants.cuh` | ROUND_CONSTANTS[213], M_INT_DIAG_HZN[24] | 46 | non-Montgomery |

### Two field ABIs (critical)

- **Native** kernels (`eltwise.cu`, `kernels.cu`, `sha.cu`) use risc0's own
  `Fp`/`FpExt` from `fp.h`/`fpext.h` (self-contained, `__device__ constexpr`).
- **SupraSeal** kernels (`supra/ntt.cu`, `supra/poseidon2.cuh`) use sppark's
  `bb31_t`/`bb31_4_t` via `#include <ff/baby_bear.hpp>` (`supra/fp.h` typedefs
  `Fp = bb31_t`). This pulls the **sppark** dependency (`DEP_SPPARK_ROOT`), which
  is CUDA-oriented. Porting the supra NTT to HIP therefore requires either an
  sppark-HIP path or re-porting the NTT over risc0's native `Fp` — the main
  Stage 2 fork in the road.

## rv32im circuit kernels (Stage 3 — the hard part)

`risc0/circuit/rv32im-sys/kernels/`:

| file | LOC | role |
| --- | --- | --- |
| `cuda/eval_check_{0,1,2,3}.cu` | 7695+6323+6203+6042 = **26263** | generated check-poly eval (GPU) |
| `cuda/steps.cu` | 30310 | generated witness-gen steps (GPU) |
| `cuda/layout.cu.inc`, `types.cuh.inc` | 8218, 3381 | generated layout/types |
| `cxx/rust_poly_fp_{0..3}.cpp` | ~52000 | **CPU golden** for eval_check |
| `cxx/steps.cpp` | 30309 | CPU golden for steps |

The generated `eval_check_*.cu` + `steps.cu` (≈56k LOC of machine-generated
BabyBear arithmetic) are the single largest work item and the most likely
intractability point, exactly as flagged in the plan's risk section.

## Build path

- `risc0-sys/build.rs` compiles the zkp kernels via
  `risc0_build_kernel::KernelBuild::new(KernelType::Cuda)` when
  `CARGO_FEATURE_CUDA`; `cuda = ["dep:cust", "dep:sppark"]`.
- `rv32im-sys/build.rs` compiles the circuit kernels similarly.
- The `rocm` port adds `KernelType::Hip` (hipcc `--offload-arch=gfx1151`) and a
  `rocm` feature mirroring `cuda` but **without** cust/sppark (native HIP FFI).

## Golden CPU baseline (frozen this run)

- Receipt: `poc/risc0-cartesi-step-demo/artefacts/step.proof.bin` (1112064 B).
- `image_id = 3aec8c717d9b47f9b617c1695955d75e2bb525085283c1076591f97ae643c990`.
- `pre_root = 0x9137e1ec36576f5ac2eab045cd143cb9944f12d7f108247031ac83c7ec7a2125`.
- `post_root = 0xd140940c18940a63197463549c1b0c1932c5ff5b13f204af87cc705157906def`.
- `cargo risczero verify` (2.3.2) → `✅ Receipt is valid!` (0.072 s), re-confirmed
  this run. This is the Stage 4 target: a ROCm-produced seal must also verify.
