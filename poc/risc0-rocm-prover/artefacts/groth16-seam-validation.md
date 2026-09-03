# Groth16 ROCm seam validation ledger

Status: **all seven gates PASS (six correctness + benchmark)** on `gfx1151` at
commit `2ef5eaa7` on 2026-07-24 (`feature/risc0-rocm-groth16`). The benchmark
records same-box wall-clock that the project owner authorized for workshop/labs
on 2026-07-24. It remains workload-specific and not cross-vendor. Evidence:
`artefacts/groth16-seam-evidence-20260724T060845Z/` (`verdict=PASS`). The earlier
six-gate run remains at `artefacts/groth16-seam-evidence-20260723T195825Z/`.

## Implemented boundary

- A CGO seam under `backend/groth16/bn254/rocm` dynamically loads the native-HIP
  BN254 primitives `bn254_hip_msm_g1`, `bn254_hip_msm_g2`, and `bn254_hip_ntt`
  from `libzkp_hip.so`, wiring them into the pinned gnark `v0.12.0-risczero.0`
  prover through a tracked patch + overlay.
- **Groth16 is a hybrid, like the segment STARK**: the MSM/NTT run on the iGPU;
  witness generation (the circom-generated `stark_verify` C++) stays on the CPU.
- The stock `r0vm` is unchanged; the ROCm path is opt-in through the patched
  Docker launcher (`RISC0_GROTH16_PROVER_IMAGE` + `RISC0_GROTH16_ACCELERATOR=rocm`).

## Gates (strict order, stop-on-first-blocker)

| # | gate | proves |
|---|---|---|
| 1 | primitive | `TestPrimitivesGPUEqualCPU`: GPU MSM/NTT == CPU (bit-for-bit) |
| 2 | native | native-HIP C ABI layout + dlopen bridge |
| 3 | image | builds `risc0-groth16-prover:rocm` (34.3 GB) end-to-end |
| 4 | risczero | r0vm proves a BN254 Groth16 receipt on the iGPU; stock verify accepts |
| 5 | journal | GPU and same-input CPU receipts share a byte-identical journal |
| 6 | onchain | the seal verifies on-chain (`verifyStep -> true` on anvil) |
| 7 | benchmark | CPU (shipped upstream image) vs GPU (ROCm seam) end-to-end wall-clock, 3 solo reps, medians (public in project workshop/labs) |

## Results (2026-07-24)

- **GPU Groth16 prove**: `[risc0-groth16-rocm] ACTIVE acceleration=rocm
  backend=groth16 curve=bn254 device="Radeon 8060S Graphics" compute_units=20
  nbConstraints=5,676,573`, `prover done took=8881 ms`, `Receipt is valid!`.
- **Journal equality**: the GPU and same-input CPU receipts both emit journal
  digest `0x6ffd15b4566971e82332ddfb05c81c7a95f484bd0621bebcc1aaa000769ab50d`.
- **On-chain**: `verifyStep(seal, pre, mcycle, post) -> true [ON-CHAIN VERIFY OK]`
  on anvil (chain 31337).
- Speed is a same-box, workload-specific comparison authorized by the project
  owner for workshop/labs on 2026-07-24; see the Benchmark section below.

## Benchmark (2026-07-24, public in project workshop/labs)

> Same-box, same-workload wall-clock numbers, project-owner authorized for
> workshop/labs on 2026-07-24. Workload-specific and NOT a cross-vendor claim.
> Evidence:
> `artefacts/groth16-seam-evidence-20260724T060845Z/benchmark/`.

- **Method**: `RISC0_GROTH16_RUN_BENCHMARK=1`, `BENCH_REPS=3`, 30 s cooldown, and
  a solo-guard wait before every timed prove (each row stamps solo/loadavg/gpu%).
  Both modes prove the identical ELF + input, pass the stock verifier, and carry
  a single image id `3aec8c71...`; the GPU rows carry the `[risc0-groth16-rocm]
  ACTIVE` marker and the CPU rows do not.
- **CPU baseline = shipped upstream image** `risczero/risc0-groth16-prover:v2025-01-31.1`.
  The fork only ADDS an opt-in ROCm accelerator and leaves the gnark CPU path
  unchanged, so the shipped image is the same-code CPU baseline. (Unlike the
  STARK there is no separate local-vs-shipped codegen gap to isolate; the fork
  ROCm image cannot itself run a CPU prove.)
- **End-to-end wall-clock** (5,676,573 constraints, `groth16-benchmark-summary.md`):

  | mode | reps | min s | median s | max s |
  |---|---|---|---|---|
  | cpu-shipped | 3 | 181.90 | 182.37 | 182.42 |
  | gpu-rocm | 3 | 186.66 | 187.44 | 187.53 |

  Median gpu-rocm vs cpu-shipped = **0.973x** (the iGPU is ~2.7% slower
  end-to-end on this workload).
- **Why parity, not a win**: the end-to-end `r0vm` Groth16 wall-clock is dominated
  by **CPU witness generation** (the circom `stark_verify` witness program). The
  iGPU accelerates only the gnark BN254 MSM/NTT prove, logged at **~8.9 s**
  (`prover done ... took=8924 ms`) - a small slice of the ~183 s total, which is
  the same on both paths. So the Groth16 iGPU offload is a **capability /
  correctness** result (verified, stock-accepted, on-chain), **not an end-to-end
  speed win**; it mirrors the STARK's Amdahl ceiling. Accelerating the MSM/NTT
  further, or the CPU-bound witness generation, is future work, not a current
  claim.

## Build-environment fixes (this branch)

The stock end-to-end image build needed four reproducibility fixes, each a
tracked commit:

- `63dbe18` circom `cargo install --locked` (upstream indexmap 2.14.0 requires
  edition2024, above the pinned Cargo 1.84);
- `90f3a77` `reclone-fork.sh` materializes Git LFS (`stark_verify.circom` is a
  58 MB LFS object, not a pointer stub);
- `556b263` install `patch` in the gnark build stage;
- `ffa5294` make the r0vm launcher check pipefail-safe (`strings | grep -q` on a
  79 MB binary hit SIGPIPE under `set -o pipefail`, a false negative).

## Reproduce

Wait for a solo GPU window, then:

```bash
RISC0_GROTH16_VALIDATION_GATES=I_UNDERSTAND_GPU_MUST_BE_IDLE \
RISC0_HIP_OFFLOAD_ARCH=gfx1151 \
bash poc/risc0-rocm-prover/groth16-seam/scripts/run-gates.sh
```

`git-lfs` must be installed so `reclone-fork.sh` can materialize
`stark_verify.circom`. The benchmark gate is opt-in via
`RISC0_GROTH16_RUN_BENCHMARK=1` and stays gated behind AMD sign-off.