# Stage 0 gate — recon + baseline + scaffolding

All evidence below is from real commands on the Strix Halo (gfx1151, ROCm 7.2.3).

## Preconditions (hard gates — all PASS)

| check | result |
| --- | --- |
| `hipcc --version` | HIP 7.2.53211, AMD clang 22.0.0, ROCm 7.2.3 |
| `rocminfo` gfx | `Name: gfx1151` (AMD Radeon 8060S, Ryzen AI MAX+ 395) |
| `r0vm --version` | `risc0-r0vm 2.3.2` |
| `cargo-risczero --version` | `cargo-risczero 2.3.2` |
| network → github.com/risc0/risc0 | `git ls-remote` returned refs |
| free disk | 58 G on `/` |

## Clone / pin

- `git clone --depth 1 --branch v2.3.2` → HEAD `218e3bc4a8ffcd203a9cd4e46f921bf60aa7e2bd` (matches tag `v2.3.2`). Vendored to `vendor/risc0/` (git-ignored, re-clone via `scripts/reclone-fork.sh`).

## Golden CPU baseline (frozen)

```
$ cargo risczero verify --path poc/risc0-cartesi-step-demo/artefacts/step.proof.bin \
    3aec8c717d9b47f9b617c1695955d75e2bb525085283c1076591f97ae643c990
✅ Receipt is valid!   (real 0m0.072s)
```

- receipt 1112064 B, image_id `3aec…c990`, pre `0x9137e1ec…`, post `0xd140940c…`.
- This is the Stage 4 target: a ROCm-produced seal must also `verify` OK.

## Kernel inventory

See `artefacts/kernel-inventory.md` — the full `Hal`/`CircuitHal` method →
CUDA kernel → stage mapping, drawn from the pinned tree (`wc -l` real).

## Scaffolding (`rocm` cargo feature)

Overlay onto the pinned fork (see `rocm-port/`):

- `risc0/build_kernel`: new `KernelType::Hip` → `compile_hip()` drives
  `hipcc --offload-arch=gfx1151` per-TU (non-RDC, self-contained objects that link
  into rustc's PIE output) + `ar` archive + `amdhip64` link.
- `risc0/sys`: `rocm` feature (no cust/sppark); `build.rs` `build_hip_kernels()`
  compiles `kernels/zkp/hip/*.hip`; `src/rocm.rs` FFI surface.
- `risc0/zkp`: `rocm` feature = `["prove", "risc0-sys/rocm"]`; `hal/hip.rs`
  `HipHal` stub (empty; full `Hal` impl is Stage 2); `hal/mod.rs` wires the module.

## Gate: `cargo build --features rocm` builds the empty HipHal stub + hipcc probe

```
$ cargo build -p risc0-sys  --features rocm      # hipcc compiles risc0 BabyBear Fp kernel -> librisc0_zkp_rocm.a
    Finished `dev` profile ... in 0.89s          # SYS_BUILD_RC=0
$ cargo build -p risc0-zkp  --features rocm      # HipHal stub compiles + links HIP archive
    Finished `dev` profile ... in 8.90s          # ZKP_BUILD_RC=0
$ cargo run  -p risc0-zkp  --features rocm --example rocm_probe
[STAGE0-ROCM] HipHal probe PASS: risc0 BabyBear Fp launched + verified on gfx1151   # PROBE_RUN_RC=0
```

The probe compiles risc0's **real** `kernels/zkp/cuda/fp.h` (BabyBear Montgomery
`Fp`) with hipcc for gfx1151, launches it from Rust through the `HipHal`, and
checks a device-computed identity (`i + (2i+1) == 3i+1 mod P`, N=256). The full
hipcc → static-lib → Rust-FFI → gfx1151 path is live.

**STAGE 0: PASS.**
