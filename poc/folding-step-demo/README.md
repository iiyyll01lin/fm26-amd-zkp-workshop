# folding-step-demo — Path C faithful folding PoC

Folds **N chained Cartesi state-transition steps** into a **single
on-chain-verifiable proof** using
[Sonobe](https://github.com/privacy-scaling-explorations/sonobe)
(Nova + CycleFold → DeciderEth Groth16/BN254), emits a `NovaDecider.sol`
verifier, and replays the folded proof's calldata on a local anvil.

> **✅ Status (2026-06-10, run on the Strix Halo) — RESOLVED end-to-end.** The
> crate **builds reproducibly** on the **`stable`** toolchain (clean `cargo
> build --release` in ~37 s; committed `Cargo.lock` resolves Sonobe
> `fcircuit-extinp@7682fe0` + plain crates.io **arkworks 0.5**, with **no
> `[patch.crates-io]`** and **no vendored `ark-groth16`**), the fold runs
> end-to-end to a **DeciderEth Groth16 proof**, and — the part that used to
> fail — that proof now **verifies natively AND on-chain**:
>
> ```text
> [fold] decider proof generated in 79.70s
> [fold] DeciderEth native verification: OK        # was SNARKVerificationFail on gr1cs 63f2930
> NovaDecider(folded calldata) -> true  [FOLDED PROOF VERIFIED ON-CHAIN]   # anvil replay, 1028 B calldata
> ```
>
> The earlier `SNARKVerificationFail` was an **upstream Sonobe regression in the
> `gr1cs` migration** (`main@63f2930`), reproduced with Sonobe's *own*
> `examples/full_flow.rs`. The fix here is **not** a rev+`[patch]` bump: we
> ported this crate back to the newest **pre-gr1cs** ref (`fcircuit-extinp`,
> arkworks-0.5, `ark_relations::r1cs`) whose `full_flow` natively verifies. See
> [`../../docs/IMPLEMENTATION-STATUS.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/IMPLEMENTATION-STATUS.md) §4 and
> the bisection record in
> [`../../docs/folding-sonobe-upstream-issue.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/folding-sonobe-upstream-issue.md).

> **Honest scope.** The circuit reproduces Demo B's *journal* relation
> **byte-faithfully** —
> `d_i = sha256(abi.encode(pre_i, mcycle_i, post_i)) = sha256(pre_i || mcycle_i_be32 || post_i)`
> (three 32-byte ABI words; the `ark-crypto-primitives` SHA-256 R1CS gadget)
> **plus step chaining** (`pre_i == post_{i-1}`) — identical to the deployed
> `StepVerifier.verifyStep` and to `06-snark.sh`. It does **not** re-prove the
> Cartesi RV64 execution / RISC0 STARK in-circuit — that stays the per-step
> STARK's (`seal`) job. Sonobe is **experimental and unaudited**; this is a
> research PoC, not production.

## What gets folded

```
steps.json (N chained (pre_root, post_root) roots, 1 mcycle each)
  → StepFCircuit            d_i = sha256(pre_i||mcycle_i_be32||post_i);  pre_i == post_{i-1};
                            acc_{i+1} = Poseidon(acc_i, Poseidon(pack(d_i)))
  → Nova + CycleFold        (BN254 / Grumpkin) fold N steps
  → DeciderEth Groth16      (BN254) compress the IVC proof → one Groth16 proof
  → NovaDecider.sol         on-chain verifier (deployed + replayed on anvil)
```

IVC state `z = [acc, post_lo, post_hi]` (`state_len = 3`). A 256-bit Cartesi
root exceeds the BN254 scalar field `Fr`, so roots are carried as **`UInt8`
byte vars** (32 each) and only *packed* into two field elements (31 + 1 bytes)
for the chaining-equality check and the accumulator — using the same
`ToConstraintField(Gadget)` packing natively and in-circuit. `z_0 = [0,
pack(genesis_pre_root)]`, so the step-0 chaining check holds by construction.
See `src/step_circuit.rs` for the full rationale.

## Run it (on the Strix Halo, not the 15 GB authoring box)

> **Toolchain:** this `fcircuit-extinp`/arkworks-0.5 ref builds on the **`stable`**
> toolchain (the repo default is 1.83 for the RISC0 demo, which is too old for the
> crates.io arkworks-0.5 set). Run via `RUSTUP_TOOLCHAIN=stable bash
> …/08-fold-steps.sh …` or `FOLD_DOCKER=1`. The full path (fold → DeciderEth →
> emit `NovaDecider.sol` → anvil replay) verifies end-to-end on the Strix Halo.

The single entrypoint is the driver script (wired into `make demo-c-fold`):

```bash
RUSTUP_TOOLCHAIN=stable bash poc/risc0-cartesi-step-demo/scripts/08-fold-steps.sh           # --full (default)
RUSTUP_TOOLCHAIN=stable bash poc/risc0-cartesi-step-demo/scripts/08-fold-steps.sh --mock    # random chained roots
```

It collects N chained steps into `artefacts/steps.json`, runs this crate
(Nova+CycleFold → DeciderEth) to emit `NovaDecider.sol` + the calldata, then
deploys + verifies on a local anvil via the `forge/` harness. Modes mirror
02/03: `--mock` (random roots, no emulator), `--dev`/`--full` (minimal machine),
`--full-rootfs` (real Linux-rootfs machine). Knobs:

| env | default | meaning |
| --- | --- | --- |
| `FOLD_N` | `2` | number of steps to fold (SHA-256-in-circuit is heavy — keep small; `FOLD_STEPS` is accepted as an alias) |
| `FOLD_DOCKER` | `0` | build + run the crate via the `Dockerfile` instead of host `cargo` |
| `FOLD_DEGRADED` | `0` | build with `--features poseidon-digest` (**non-faithful** fallback) |
| `ANVIL_PORT` | `8545` | local anvil port for the on-chain verify |

## Outputs (`artefacts/`)

- `folded.public.json` — public IO (`z_0`, `z_i`, step count, genesis/final roots).
- `folded.proof.json` — proof metadata + the ABI calldata + verify entrypoint.

## Standalone (just the prover, no driver)

```bash
cargo run --release -- --steps steps.example.json --out artefacts
# optional in-process revm check of the emitted contract (needs `solc`):
FOLD_SELF_CHECK=1 cargo run --release -- --steps steps.example.json --out artefacts
```

The on-chain leg lives in `forge/` (`forge script script/VerifyFolded.s.sol`),
driven by `08-fold-steps.sh`.

## Layout

- `src/step_circuit.rs` — the faithful `StepFCircuit` (SHA-256 journal + chaining) with a `poseidon-digest` fallback.
- `src/main.rs` — reads `steps.json`, folds, runs DeciderEth, writes artefacts + `NovaDecider.sol`.
- `Dockerfile` — reproducible heavy build (host build uses the `stable` toolchain).
- `forge/` — self-contained Foundry harness (`foundry.toml`, `script/VerifyFolded.s.sol`); the generated `NovaDecider.sol` is copied into `forge/src/` at run time, and `forge/lib/forge-std` is fetched by the driver.

## Dependency note

Sonobe's API drifts. This crate is pinned to the **pre-gr1cs** commit
`7682fe0fa190` (`fcircuit-extinp`) — the newest ref whose Nova+CycleFold →
DeciderEth `full_flow` natively verifies. Unlike the old `main@63f2930` (gr1cs)
pin, this ref builds against **plain crates.io arkworks 0.5** (`ark_relations::r1cs`),
so there is **no `[patch.crates-io]` block and no vendored `ark-groth16`** — every
`ark-*` resolves from crates.io `^0.5.0` and the exact versions are frozen in the
committed `Cargo.lock`. `main@63f2930` regressed `full_flow` to
`SNARKVerificationFail`; that bisection is recorded in
[`../../docs/folding-sonobe-upstream-issue.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/folding-sonobe-upstream-issue.md).
If you bump the rev, re-check `examples/full_flow.rs` upstream first.
