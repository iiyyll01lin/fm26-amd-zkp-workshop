# Vendored risc0 fork — pin & reproduction

This PoC ports the **RISC0 zkVM STARK prover** to a native **ROCm/HIP HAL
backend** (`rocm` cargo feature), validated first on the Strix Halo `gfx1151`
iGPU. It follows
the repo's established **vendored-fork** discipline (cf.
`poc/folding-step-demo/vendor/ark-groth16`) but, because the risc0 tree is large
and mostly machine-generated, the pin is kept as a *reproducible checkout* plus a
committed **port overlay**, rather than committing the whole tree.

## Pin

| field | value |
| --- | --- |
| upstream | `https://github.com/risc0/risc0` |
| tag | `v2.3.2` |
| commit | `218e3bc4a8ffcd203a9cd4e46f921bf60aa7e2bd` |
| matches | installed `r0vm 2.3.2` / `cargo-risczero 2.3.2` (see `poc/risc0-cartesi-step-demo/scripts/04-prove.sh`) |
| crate versions | `risc0-zkp 2.0.3`, `risc0-sys 1.4.0`, `risc0-circuit-rv32im`, `risc0-core`, `risc0-build-kernel` |

## Why not commit the whole tree

The fork is ~184 MB and includes ~100k LOC of **generated** kernels
(`rv32im-sys/kernels/cuda/eval_check_*.cu` = 26k LOC, `steps.cu` = 30k LOC,
`cxx/rust_poly_fp_*.cpp` = 52k LOC). Committing that into the course repo would
bloat it with upstream-unchanged, machine-generated code. Instead:

- `vendor/risc0/`  — the pinned checkout (git-ignored; re-created by the script
  below). This is the single, coherent build base; **all** rocm edits go here.
- `rocm-port/`     — the committed overlay: every file this PoC authors or
  modifies in the fork (build_kernel HIP support, `rocm` cargo features, the
  `HipHal` stub, Cargo patches), plus `apply-overlay.sh` to stamp them onto a
  fresh checkout.
- `kernels/hip/`   — the ported HIP kernels (BabyBear field, Poseidon2, SHA-256,
  NTT, …), authored here.
- `hip/`, `dump/`  — standalone bit-for-bit harnesses + the Rust reference-vector
  dumper (mirrors `poc/amd-gpu-zk-primitive-demo/hip/` + its `*-dump` bins).

Isolation guarantee is identical to a full vendored fork: a pinned commit, no
floating deps, all edits contained under this PoC.

## Reproduce the checkout

```bash
bash poc/risc0-rocm-prover/scripts/reclone-fork.sh   # clones v2.3.2 -> vendor/risc0
bash poc/risc0-rocm-prover/rocm-port/apply-overlay.sh # stamps the manifest-listed overlay
```

## Reproduce the build + verification (selected target + ROCm 7.2)

```bash
# Scripts auto-detect the single attached gfx target. Optional override for
# cross-compilation or multi-target hosts:
# export RISC0_HIP_OFFLOAD_ARCH=gfx1151

# 1) Correctness gate — risc0's OWN DualHal (CpuHal vs HipHal), 15/15:
make risc0-rocm-dualhal
#   == cargo test -p risc0-zkp --features rocm hal::hip -- --test-threads=1

# 2) End-to-end GPU seal + STOCK verifier:
make risc0-rocm-prove      # r0vm --features rocm proves the Demo B step on the iGPU
#   -> "[risc0-rocm-prover] HipHal segment prover ACTIVE on ${RISC0_HIP_OFFLOAD_ARCH} …"
#   -> cargo risczero verify … -> "✅ Receipt is valid!"

# 3) Honest roofline (Phase B) — solo segment-po2 sweep, iGPU vs 32t CPU:
R0VM_CPU=… R0VM_GPU=… SWEEP_PO2S="16 18 20 21" \
  bash poc/risc0-rocm-prover/scripts/run-stage4-sweep.sh   # -> artefacts/stage4-sweep.csv
```

`gfx1151` above is the documented compatibility default. The stage and
clean-room scripts prefer an explicit value, then live detection.

For a fresh clone/cache with isolated, write-once evidence, run:

```bash
bash poc/risc0-rocm-prover/scripts/run-clean-room.sh --timing
```

Strict PASS requires DualHal 15/15, rv32im `eval_check` equality, a real
non-empty receipt, stock-verifier success, HipHal plus CPU-delegated
witgen/accum markers, and overlapping non-zero `rocm-smi` telemetry. The
acceptance run explicitly unsets both research switches; the HIP witgen/accum
paths are not release gates.

Canonical clean-room evidence now covers both `gfx1151` and `gfx1201`. The
cross-architecture eval fix selected a 16-thread workgroup because `poly_fp`
consumes 24,288 private bytes per thread and `gfx1201` workgroups of 32 or more
corrupted scratch. Production and standalone gates share a target-neutral
policy that queries `hipFuncGetAttributes` and device properties, derives a
power-of-two block under a 512 KiB private-storage budget, logs the decision,
and conservatively falls back to 16 if metadata is incomplete. The
`RISC0_ROCM_EVAL_CHECK_BLOCK_SIZE` debugging override is accepted only at or
below the derived safe limit. The post-rebase 88-file source manifest used for
the adaptive `gfx1201` validation has SHA-256
`50f9bf12ac2422c4bd72d882af3a4988346949256f610a8df00f997834bb0b15`;
the original fixed-launch clean-room snapshot remains
`4e2d696180ad4a6f02a744d5233ef3a09bd74483204dbbf31fc78ca924005f56`.
See `RELEASE-EVIDENCE-CHECKLIST.md` for exact evidence paths and limitations.

### Verification evidence (re-run 2026-07-20, this fork)

- **DualHal 15/15 PASS** — `cargo test -p risc0-zkp --features rocm hal::hip`:
  all 15 CpuHal==HipHal equality tests pass (NTT family, `fri_fold`, `mix_poly_coeffs`,
  eltwise, `hash_rows`/`hash_fold` for **both** poseidon2 and sha-256, `zk_shift`,
  `gather_sample`, `batch_*`). Correctness is contention-independent.
- **Stock verifier accepts the GPU seal** — the `r0vm --features rocm` seal for the
  ~4-segment Cartesi-step prove is accepted by the unmodified `cargo risczero verify
  2.3.2` (`✅ Receipt is valid!`); the HipHal path is proven by runtime markers +
  95% iGPU busy + an audit differential-fallback test.
- **Roofline (Phase B, solo):** flat **~5.3–5.5×** vs the same-code 32t CPU across
  segment po2 16→21 (cleanest baseline 5.52×; iGPU wall stable ~26 s; the 6.6/6.79×
  = 5.46× × a 1.25× local-vs-shipped codegen gap, not noise). hashfn is **poseidon2-only** (risc0
  2.3.2 rejects sha-256 for the rv32im composite prover); iGPU peak RSS reaches
  18.9 GB at po2=21 (past the 16 GB wall, held by the 94 GB unified pool). Full
  analysis: [`artefacts/stage4-roofline.md`](artefacts/stage4-roofline.md).

### Initial clean-room reproduction (2026-07-20 baseline)

A fresh `git clone --depth 1 --branch v2.3.2` into a scratch dir + `apply-overlay`
(34 files, per `rocm-port/OVERLAY-MANIFEST.txt`) + `cargo test -p risc0-zkp
--features rocm hal::hip` **reproduced DualHal 15/15 from scratch in ~78 s**. The
build log shows `hipcc --offload-arch=gfx1151 … ffi.hip` compiling the zkp HIP
kernel **fresh** (fresh `librisc0_sys`/`risc0_zkp` artefacts) — the DualHal path
needs only `risc0-zkp` + `risc0-sys`, **not** the 26k-LOC rv32im `eval_check`, so it
is light. The **full `r0vm --features rocm`** build (incl. that rv32im kernel) was
exercised from an **empty cache** in Phase B (~31 min → GPU prove + stock-verify +
the po2 roofline sweep). This proves the overlay is self-contained (no reliance on
ad-hoc edits in `vendor/risc0`); it is kept byte-identical to the built+tested fork
via `snapshot-overlay.sh` (a re-snapshot shows zero drift).

The later two-architecture evidence in `RELEASE-EVIDENCE-CHECKLIST.md`
supersedes this initial 34-file baseline for release decisions.
