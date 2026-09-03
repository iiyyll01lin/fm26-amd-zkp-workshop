# risc0-rocm-prover — RISC0 zkVM STARK kernels ported to native ROCm/HIP

This PoC takes the [M13](https://github.com/iiyyll01lin/zkp-final/blob/main/course/modules/13-port-zk-to-rocm.md) moonshot the
repo marked **`A3 RED`** — "port the RISC0 `r0vm` STARK prover to AMD" — off the
"impossible / CPU-only" shelf and turns the
[`risc0-rocm-probe.sh`](../risc0-cartesi-step-demo/scripts/risc0-rocm-probe.sh)
**negative result** into a **measured, bit-for-bit-validated partial positive**:
the STARK prover's entire hot-kernel set is now ported to **native HIP** and runs
**bit-for-bit identical to risc0's own `CpuHal`**. Clean-room correctness is
validated on Strix Halo `gfx1151` and RDNA4 `gfx1201`.

> **Honest headline.** The STARK prover's full kernel set (BabyBear field,
> Poseidon2, SHA-256, the NTT family, `fri_fold`/`mix_poly_coeffs`/…, the
> **26k-LOC generated rv32im `eval_check`**, Merkle build) is ported to native
> HIP and **bit-for-bit gated on gfx1151 and gfx1201**, AND wired end-to-end: **`r0vm
> --features rocm` proves the Demo B step (a ~4-segment Cartesi-step composite;
> `po2=20` = per-segment limit) on the iGPU and the stock `cargo risczero verify`
> → `✅ Receipt is valid!`** — provably via the HipHal path (95% GPU busy, 4
> markers; **independently audited**). **Hard guarantee (flat):** stock-verifier
> seal + bit-for-bit CpuHal==HipHal (DualHal 15/15). A solo **segment-po2 sweep**
> ([`artefacts/stage4-roofline.md`](artefacts/stage4-roofline.md)) turns the single
> point into a curve: a **flat ~5.3–5.5×** over the same-code 32t CPU across po2
> 16→21 (no crossover; iGPU wall stable ~26 s; the cleanest baseline is 5.52×, and
> the earlier 6.6× / audited 6.79× = 5.46× × a 1.25× local-vs-shipped codegen gap, not random noise). An
> honest **workload-specific** figure (poseidon2 is the only hashfn the rv32im
> prover accepts; hybrid design), **not** a general "iGPU is faster" claim. Honest
> split: **STARK math + eval_check on the iGPU; witgen + accum on the CPU**
> (delegated over unified memory). Recursion eval/accum and Groth16 MSM/NTT now run on iGPU; their witness/sequential portions stay CPU. Complete Groth16 receipt is 0.973× (witness-bound), documented in Lab 24.

This fully overturns M13 §9 / the probe's "RISC0 STARK is 100% CPU-only on AMD, a
multi-month effort with no path": the segment STARK now runs, verifiably, on the
AMD iGPU.

## Pin

`risc0/risc0 @ v2.3.2` (`218e3bc…`), matching the installed `r0vm 2.3.2`. The fork
is a pinned, git-ignored checkout under `vendor/risc0/` (see
[`VENDOR.md`](VENDOR.md)); the rocm-port deltas live in `rocm-port/`.

## Methodology (mirrors `poc/amd-gpu-zk-primitive-demo/hip/`)

Mechanical CUDA→HIP translation (kernel body unchanged, only the field backend /
launch dialect swapped) + a **bit-for-bit GPU==CPU gate** for every kernel, with
the golden dumped from the **actual risc0 `CpuHal`** (not a re-derivation). BabyBear
Montgomery constants are byte-identical to `risc0_core`, so results compare on the
**raw field word**.

## Stage gates (all evidence from real `hipcc` + gfx1151 runs)

| stage | what | gate | result |
| --- | --- | --- | --- |
| 0 | recon + pin + scaffold + `rocm` feature | `cargo build --features rocm` + hipcc probe | **PASS** ([stage0](artefacts/stage0-gate.md)) |
| 1 | BabyBear Fp/Fp4 + Poseidon2 + SHA-256 | GPU==CpuHal bit-for-bit | **PASS** — 5632/1768/400 checks ([stage1](artefacts/stage1-gate.md)) |
| 2 | NTT family, zk_shift, eltwise, fri_fold, mix, eval_any | per-op GPU==CpuHal | **PASS** — 38096 checks ([stage2](artefacts/stage2-gate.md)) |
| 3a | **rv32im `eval_check` (26k LOC generated)** | GPU==CPU-C++ bit-for-bit | **PASS** — 1024 checks ([stage3](artefacts/stage3-gate.md)) |
| 3b | Merkle `hash_rows`/`hash_fold` (SHA+Poseidon2) | GPU==CpuHal | **PASS** — 1984 checks |
| 3c | accum/witgen (`steps.cu`) | — | CPU-delegated over unified memory (Stage 4) |
| 4a | full `HipHal` (zkp `Hal` trait) | risc0 **DualHal** CpuHal==HipHal | **PASS** — 15/15 tests |
| 4b | rv32im `HipCircuitHal` eval_check | GPU==CPU crate test | **PASS** |
| 4c | `r0vm --features rocm` prove Demo B step (~4-seg composite) + verify | GPU seal, **stock** `✅ Receipt is valid!` | **PASS** ([stage4](artefacts/stage4-gate.md)) |
| 4d | bench iGPU vs 32t CPU (solo, po2 sweep) | honest where-wins/loses curve | **flat ~5.3–5.5×** across po2 16→21 (cleanest 26.1 s vs 144.2 s = 5.52×; 6.6/6.79× = 5.46× × 1.25× codegen gap, not noise) ([roofline](artefacts/stage4-roofline.md)) |

## Cross-architecture clean-room validation

| target | clean-room result | durable evidence |
| --- | --- | --- |
| `gfx1151` | **PASS** | DualHal 15/15, eval_check equality, 1,112,064-byte stock-verified receipt, GPU telemetry |
| `gfx1201` | initial **FAIL** preserved; fixed run **PASS** | 16-thread eval_check, receipt SHA-256 `91f40fe…6487a`, 24/26 non-zero telemetry samples, 100% max |

`poly_fp` consumes **24,288 private bytes per thread**. On `gfx1201`,
workgroups of 32 or more caused scratch corruption and nondeterminism; a
16-thread workgroup passes repeated equality gates on both targets without
changing expected values or weakening the gate. Production and standalone gates
now share one target-neutral launch policy: it reads the kernel's
`localSizeBytes` plus kernel/device thread limits, applies a conservative 512 KiB
private-storage budget per workgroup, and falls back to 16 when required runtime
metadata is incomplete. Every launch logs its choice and limits.
`RISC0_ROCM_EVAL_CHECK_BLOCK_SIZE` may request a smaller debugging block; invalid
or unsafe values are rejected rather than silently launched.

The post-rebase adaptive validation used an 88-file source manifest with
SHA-256
`50f9bf12ac2422c4bd72d882af3a4988346949256f610a8df00f997834bb0b15`.
The original fixed-launch clean-room snapshot remains
`4e2d696180ad4a6f02a744d5233ef3a09bd74483204dbbf31fc78ca924005f56`.

The RDNA4 host's incomplete ROCm installation also required a local `hipcc`
adapter, `HIP_SYMBOL(...)`, and explicit `-lamdhip64` linking. This is a
toolchain-packaging caveat, not an architecture-specific source fork.

Performance remains per-host and workload-specific: the solo RDNA4 run measured
95.940 s CPU versus 26.590 s GPU (**3.61×**) on that host, while the existing
`gfx1151` same-code sweep is ~5.3–5.5× on its host. These numbers are not
cross-host comparisons; correctness is the portable result.

## Run the gates (needs ROCm 7.2 and a supported AMD GPU)

```bash
bash scripts/reclone-fork.sh            # pin v2.3.2 -> vendor/risc0 (git-ignored)
bash rocm-port/apply-overlay.sh         # stamp the rocm-port overlay
bash scripts/run-stage1.sh              # field + Poseidon2 + SHA-256 bit-for-bit
bash scripts/run-stage2.sh              # NTT/eltwise/fri_fold/mix/eval bit-for-bit
bash scripts/run-stage3-evalcheck.sh    # rv32im eval_check poly_fp bit-for-bit (~4 min compile)
bash scripts/run-stage3-merkle.sh       # Merkle hash_rows/hash_fold bit-for-bit
# Stage 4 — end-to-end GPU prover:
( cd vendor/risc0 && cargo test -p risc0-zkp --features rocm hal::hip -- --test-threads=1 )  # DualHal 15/15
bash scripts/run-stage4-prove.sh        # build r0vm --features rocm, prove Demo B step on iGPU, verify
bash scripts/run-stage4-bench.sh        # solo-guarded iGPU vs 32t CPU bench
```

For the same correctness boundary used by manual self-hosted CI (with no
performance pass/fail), run:

```bash
RISC0_HIP_OFFLOAD_ARCH=gfx1151 \
  bash scripts/run-amd-correctness-ci.sh
```

The `RISC0 ROCm correctness (manual AMD)` workflow dispatches this entrypoint to
separate `[self-hosted, linux, amd-rocm, gfx1151]` and
`[self-hosted, linux, amd-rocm, gfx1201]` runners. Missing GPU/tooling fails by
default; a preflight skip is possible only through the explicit dispatch input.

The scripts use an explicitly supplied `RISC0_HIP_OFFLOAD_ARCH` or detect the
single attached `gfx*` target with `rocminfo`/`rocm-smi`. `gfx1151` remains only
the compatibility default when detection is unavailable.

For the write-once release gate, use the clean-room orchestrator. It clones the
exact v2.3.2 pin into a fresh scratch tree, uses fresh Cargo/RISC0 caches, applies
the overlay, and requires DualHal 15/15, rv32im `eval_check` equality, a real
receipt, stock-verifier success, HipHal plus default CPU-witgen/CPU-accum
markers, and overlapping non-zero `rocm-smi` telemetry:

```bash
bash scripts/run-clean-room.sh --timing
```

Existing evidence paths are rejected rather than overwritten. Timing is recorded
only when `solo-guard.sh` passes; correctness still runs when timing is skipped.

## Part-B runtime switches

`RISC0_ROCM_WITGEN` and `RISC0_ROCM_ACCUM` independently select the HIP witness
and accumulation kernels. Both remain default-off. The accepted clean-room and
benchmark paths explicitly unset both switches.

**Default-off is a divergence from upstream, taken on measurement.** The stock
v2.3.2 CUDA prover runs both on the GPU with no switch at all
(`hal/cuda.rs:100` → `risc0_circuit_rv32im_cuda_witgen`, `:155` →
`risc0_circuit_rv32im_cuda_accum`); our `hal/hip.rs` defaults to the CPU C++
goldens at `:110` and `:172`. Two reasons: witgen+accum are 28.87% of the
measured prove, so a perfect port tops out at **~1.41×**; and the 2026-07-31
`gfx1201` A/B found all four switch combinations **within 2% on wall while the
GPU arms burn 13–15% more energy**, with all 24 receipts stock-verified. The
earlier "cross-cycle write-before-read blocker" was root-caused to a destructive
buffer fill and fixed on 2026-07-22; it is no longer the reason. See
[`artefacts/partb-status.md`](artefacts/partb-status.md).

## Layout

- `kernels/hip/` — ported HIP kernels: `babybear.hpp`, `sha256.hpp`,
  `poseidon2.hpp`, `hal_ntt.hpp`, `hal_kernels.hpp` (+ generated constant headers).
- `hip/` — standalone bit-for-bit harnesses (single-TU, `__host__ __device__`).
- `dump/`-style dumpers live as `vendor/risc0/risc0/zkp/examples/dump_*` (run the
  real `CpuHal` to emit golden vectors).
- `rocm-port/` — the committed overlay onto the pinned fork (`build_kernel` HIP
  support, `rocm` cargo features, `HipHal` stub); `apply-overlay.sh`.
- `artefacts/` — per-stage gate evidence + kernel inventory + honest Stage 4 status.
