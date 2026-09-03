# Demo C — GPU-driven DeciderEth Groth16 prove (Track 2, todo `demo-c-fork`)

The Sonobe **Nova + CycleFold → DeciderEth (Groth16/BN254)** fold, with the final
Groth16 prover's **BN254 G1 multi-scalar-multiplications offloaded to the AMD
Strix Halo iGPU** (Radeon 8060S `gfx1151`, ROCm 7.2.3 + OpenCL) via the B0
library `amd_gpu_zk_primitive_demo`.

**Result: the GPU-MSM-produced DeciderEth proof verifies natively AND on-chain,
bit-for-bit equivalent to the pure-CPU proof.** The plumbing — vendor + patch
`ark-groth16` 0.5.0 so the working iGPU MSM actually drives Sonobe's decider —
is the deliverable, and it is **independently true regardless of the speed
verdict**.

> **INTEGRITY UPDATE (2026-06-16, clean solo re-bench).** The previously-published
> headline "iGPU **drives/wins** the DeciderEth proof at **1.34× (G1) / 1.64×
> (wide)**" was taken under iGPU+CPU contention and **REVERSES** on a fair, solo,
> contention-guarded re-measurement: the **clean solo CPU median is 66.52 s**
> (3 runs: 70.89 / 66.52 / 64.98 — *not* the contention-inflated 173.83 s used
> before). Against that fair baseline the iGPU offload is a **slowdown, not a win**:
> G1-only `D::prove` 95.33 s ⇒ **0.70×**, wide (G1+G2+FFT) 89.68 s ⇒ **0.74×**.
> (The two GPU rows were themselves taken with a third-party ROCm training job
> sharing the iGPU mid-run, so 0.70× / 0.74× are *lower bounds* on the true solo
> speedup; see `docs/INTEGRITY-REPORT.md`.) The honest framing: the iGPU
> G1/G2/FFT offload is a **correctness / plumbing deliverable, not a speed win**
> at this size. See "Honest caveats" below.

> # 🔴 AMENDED 2026-08-29 — the G1-only `0.70×` above is SUPERSEDED
>
> **The `0.70×` in the update above was a *floor*, and a paired re-bench has since
> proved it was too pessimistic by 1.42×. Do not quote `0.70×` as the current
> OpenCL G1-only figure.**
>
> The 2026-06-16 OpenCL rows carried three unfixed defects: **(1)** the GPU arm was
> `n=1` against a CPU arm of `n=3`; **(2)** that single sample was preempted
> mid-flight; **(3)** the arms were never interleaved — a forge+anvil on-chain
> replay sat between them. A paired re-bench on 2026-08-29 fixed all three (one
> session, one binary, both arms `FOLD_GPU_MSM_CHECK=0`, arms **interleaved**,
> `n=3` each, median, solo re-verified before every run):
>
> | arm | median `D::prove` | within-arm spread | vs CPU median |
> |---|---|---|---|
> | CPU (paired) | **60.847 s** | 0.93% | **1.000×** |
> | OpenCL GPU G1-only (paired) | **61.206 s** | 1.21% | **`0.994×`** |
>
> Source: `demo-c-paired-checkoff-opencl-gfx1151.{csv,log}`. **This file's `.csv`
> is unmodified** — the paired run is a separate artefact.
>
> 🔴 **The only defensible word is PARITY.** The arm-to-arm gap is **0.59%**,
> *smaller* than both within-arm spreads, so at `n=3` the two arms are
> **statistically indistinguishable** — **never** "the iGPU wins", **never** "the
> iGPU is 0.6% slower". Mean-based is **0.990×**, same verdict. **Honest
> disclosure:** the slowest GPU rep (61.914 s) had its guard measure foreign CPU
> **9%** (under the 10% abort threshold); the median excludes it *by construction*.
>
> 🔴 **`gpu-wide` (0.74×) was NOT re-measured** and keeps all three defects, so it
> remains a **floor, not an estimate** — the G1 correction **must not** be
> extrapolated to it, and the two arms **must not** be averaged or compared.
>
> 🔴 **Parity is not acceleration.** The "no GPU-accelerated-proof claim" line is
> untouched; the **correctness / plumbing** framing below is unchanged and remains
> the deliverable.
>
> **Four arms, four states, never mixed:** OpenCL G1-only **`0.994×` parity
> (paired)** · OpenCL wide **`0.74×` floor (not re-measured)** · native-HIP G1-only
> **`1.048×` small win (paired)** · native-HIP wide **`0.77×` floor (not
> re-measured)**.

## How it works

`vendor/ark-groth16` is the **exact crates.io `ark-groth16` 0.5.0 source**
(checksum `88f1d0f3a534bb54188b8dcc104307db6c56cdae574ddc3212aec0625740fc7e`,
frozen in `Cargo.lock`) plus a single feature/env-gated seam in
`src/prover.rs`. When cargo feature `gpu-msm` is built **and** env
`FOLD_GPU_MSM=1` at runtime, the four BN254 **G1** MSM call sites in
`create_proof_with_assignment`

| query    | MSM size `n` | live terms (after filtering) |
|----------|--------------|------------------------------|
| `h_query`   | 16,777,215 | 16,777,215 (dense)         |
| `l_query`   | 10,651,819 |  6,016,186                 |
| `a_query`   | 10,651,861 |  3,022,314                 |
| `b_g1_query`| 10,651,861 |  5,703,258                 |

dispatch to a single reused `Bn254Gpu` OpenCL context (one kernel JIT for the
whole prove). The `[patch.crates-io]` in `Cargo.toml` routes BOTH our direct
`ark-groth16` dep AND sonobe `folding-schemes`' transitive one through the
patched prover — that is how the GPU MSM reaches the DeciderEth prove.

**Originally out of scope, now optionally offloaded (Step 4a):** the BN254 **G2**
(Fq2) MSM (the Groth16 `B` query) and the `ark-poly` QAP **FFT** in `witness_map`.
The B0 lib now provides a templated `Bn254G2_multiexp` kernel and bit-for-bit
radix-2 (i)fft / coset wrappers; `FOLD_GPU_G2=1` / `FOLD_GPU_FFT=1` route them to
the iGPU (the `gpu-wide` row, 0.74× slowdown). Dispatch is keyed on the concrete point type
+ scalar field via `TypeId`, so G2/FFT for any non-BN254 / non-radix2 domain
transparently takes the CPU path; with the env flags unset the G1-only behavior
(and the pure-CPU default) is unchanged.

**Default build (no `gpu-msm` feature, or env unset) is byte-identical to stock
`ark-groth16`** — the seam compiles to exactly `G::Group::msm_bigint`.

### Correctness gate

A `FOLD_GPU_MSM_CHECK=1` run recomputed every offloaded MSM on the CPU and
compared: **all four reported `GPU == CPU` OK** (see `gpu/fold-check.log`). The
one subtlety vs the B0 random-input bench: a real Groth16 CRS query vector is
*sparse* — it contains identity (point-at-infinity) bases and zero scalars. The
B0 GPU kernel has no infinity representation (it reads raw `(x, y)` limbs), so
the patch **filters degenerate pairs** (zero scalar or identity base — each
contributes nothing) before dispatch, exactly matching arkworks' own
`msm_bigint` (which already drops zero scalars). With that, GPU output is
bit-for-bit equal to the CPU MSM.

## Measurements (this AMD Strix Halo, gfx1151 / ROCm 7.2.3, 32-thread Zen 5)

`FOLD_N=2`, mock chained steps, `--steps artefacts/steps.json`. The timed
quantity is the **DeciderEth Groth16 `D::prove` wall time** (the only stage that
runs the patched G1 MSMs); the Nova `prove_step`s are unaffected (they use
folding-schemes' own commitments, not ark-groth16).

**Clean solo re-bench (2026-06-16, `CPU_RUNS=3`, solo-guarded, `RAYON_NUM_THREADS=32`):**

| mode | `D::prove` wall | native verify | on-chain replay | speedup vs clean CPU median | solo |
|------|-----------------|---------------|-----------------|-----------------------------|------|
| CPU median of 3 (`FOLD_GPU_MSM=0`) | **66.52 s** (70.89 / 66.52 / 64.98) | OK | VERIFIED | 1.00× | **true (clean)** |
| GPU G1-only (`FOLD_GPU_MSM=1`) | **95.33 s** | OK | VERIFIED | **0.70×** — ⚠️ **SUPERSEDED, do not quote**: the paired re-bench reads **`0.994×`, parity** (see the amendment at the top) | false¹ |
| GPU wide (`+FOLD_GPU_G2=1 FOLD_GPU_FFT=1`) | **89.68 s** | OK | VERIFIED | **0.74×** (slowdown) — **not re-measured; a lower bound** | false¹ |

¹ The two GPU rows' `D::prove` stages overlapped a third-party ROCm
`train_language.py` (pid 194983) that woke at 08:43:52 and saturated the iGPU, so
they are **contended upper bounds** (the displayed 0.70× / 0.74× are *lower
bounds* on the true solo speedup — a label the 2026-08-29 paired re-bench
confirmed for the G1 arm, which lands at **`0.994×`, parity**; `gpu-wide` was
**not** re-measured and its 0.74× stays a floor). The CPU baseline is GPU-independent and was
verified clean (run 1 finished at 08:41:37, fully before the train job, and is the
*slowest* of the three — so the train job added no CPU inflation). A clean solo GPU
re-bench is **blocked** by the resumed third-party session.

**Superseded (contention-inflated) numbers, for the record:** CPU 173.83 s → GPU
G1 129.55 s (claimed 1.34×) → wide 106.19 s (claimed 1.64×). The CPU 173.83 s was a
single contended sample; the true solo CPU is ~66.5 s, which **reverses** the
headline.

### Step 4a — wider offload (G1 + G2 MSM + QAP FFT all on the iGPU)

Pushing past the G1-only seam, the `gpu-wide` row additionally offloads the BN254
**G2 (Fq2)** `B`-query MSM (templated `Bn254G2_multiexp`) and the QAP **radix-2 FFT**
(`ark-poly` ifft + coset-fft + coset-ifft, all at the `2^24` decider domain) via
`FOLD_GPU_G2=1 FOLD_GPU_FFT=1`. On the 2026-06-16 re-bench the wider offload was
faster in wall-clock than the G1-only row (89.68 s vs 95.33 s — the FFT-at-2^24
savings are real), **but it does not turn into a win**: against the clean solo CPU
median of 66.52 s it is still **0.74×, a slowdown**.

> 🔴 **Do not read that as "wide narrows the gap vs G1-only" any more.** Since
> 2026-08-29 the two arms are **no longer measured the same way**: the G1-only arm
> got a paired, interleaved, `n=3`-per-arm re-bench and reads **`0.994×` (parity)**,
> while `gpu-wide` was **never re-measured** and still carries `n=1` + contention +
> no interleaving. Comparing 89.68 s against 95.33 s compares two contended single
> samples; comparing 0.74× against 0.994× compares a floor against a paired median.
> **Neither comparison is valid.** The honest statement is that the wide arm simply
> has **no clean measurement yet**, and the G1 correction **does not transfer to it**.

The `FOLD_GPU_MSM_CHECK=1` run re-confirmed
every offloaded primitive bit-for-bit: all four G1 MSMs `GPU == CPU OK`, the G2 MSM
`GPU == CPU OK`, and all of the radix-2 ifft / coset-fft / coset-ifft passes
`GPU == ark-poly OK`. The proof verified **natively AND on-chain** (`anvil` +
`NovaDecider.sol` → `FOLDED PROOF VERIFIED ON-CHAIN`).

The earlier "1.64× genuine win" reading was an artefact of comparing the wide GPU
time against a **contention-inflated** 173.83 s CPU sample. On a fair solo CPU
baseline the wide offload is a slowdown (0.74×, and that GPU row was itself iGPU-
contended mid-run, so its true-solo number would be *faster* but still cannot reach
parity from 89.68 s vs a 66.5 s CPU at this 2^24 size). Reproduce a clean run with
`FOLD_WIDE=1 FOLD_GPU_MSM_CHECK=1 bash scripts/run-gpu-fold.sh` **in a solo window**
(the guard enforces it; it exits 42 under contention).

## Native-HIP backend (plan Part 1, 2026-06-18)

The offload above runs on ec-gpu's **OpenCL** kernels. Plan Part 1 adds a second,
**fully native HIP** backend for the same `Bn254Gpu` API: the B0 lib gained a `hip`
cargo feature whose `build.rs` compiles `hip/zkp_hip_ffi.cpp` (the native
`bn254_g1_multiexp` / `bn254_g2_multiexp` / `bn254_fr_radix_fft` kernels) into a
static `libzkp_hip.a` and links it. `Bn254Gpu` became a backend enum selected at
runtime by **`FOLD_GPU_BACKEND=hip`** (default `opencl`); the device region (upload
→ launch → read back) goes through the HIP FFI while the host-side window
recombination / twiddle precompute / coset scaling stay in Rust shared with OpenCL,
so results are bit-for-bit identical. **The Demo C seam, `prover.rs`, and
`r1cs_to_qap.rs` are untouched** — only the B0 lib's backend widened. Build it with
the new pass-through feature `--features gpu-msm-hip` (the default `gpu-msm` OpenCL
build is unchanged).

**Result (clean solo window, `iGPU 13% / loadavg 6.17`, `FOLD_N=2`, 2026-06-24):**

| mode | backend | `D::prove` wall | native verify | on-chain replay | speedup vs clean CPU median (66.52 s) | solo |
|------|---------|-----------------|---------------|-----------------|----------------------------------------|------|
| HIP G1-only (`FOLD_GPU_BACKEND=hip`)            | native HIP | **77.45 s** | OK | VERIFIED | **0.86×** — ⚠️ **SUPERSEDED, do not quote** (see below) | **true (clean)** |
| HIP wide (`FOLD_GPU_BACKEND=hip` + G1+G2+FFT)   | native HIP | **85.87 s** | OK | VERIFIED | **0.77×** (slowdown) — **not re-measured; a lower bound** | **true (clean)** |

This re-bench ran fully under the solo-guard (`make demo-c-fold-gpu-hip`), both folds
solo from start to finish (no third-party iGPU job woke mid-run, unlike the earlier
`95.44 s` upper-bound sample on 2026-06-18).

### ⚠️ The G1-only `0.86×` above is SUPERSEDED — the paired re-bench reads **1.048×**

**Headline (2026-08-29):** on a paired, same-session, same-binary re-bench with
**both arms check-off**, native-HIP G1-only is **1.048× — a small win, not a
slowdown**: CPU **61.602 s** vs native-HIP G1-only **58.762 s** (medians of 3
interleaved reps each; spreads 0.83% / 0.61%). Artefact:
[`demo-c-paired-checkoff-gfx1151.csv`](demo-c-paired-checkoff-gfx1151.csv).

**Why the published `0.86×` was wrong** — the two arms were never comparable:

1. `scripts/run-gpu-fold-hip.sh:74` defaults **`FOLD_GPU_MSM_CHECK=1`** on the GPU
   arms, recomputing every offloaded MSM on the CPU **inside the timed region**,
   while `run_cpu()` never pays that cost. Measured at **10.878 s**
   (69.640 − 58.762), i.e. it inflates the check-off GPU wall by **18.5%**.
2. `CPU_RERUN` defaults to `0`, so the denominator was a durable `CPU_MED` of
   **66.52 s** captured on 2026-06-16 against a **different build**.

Both defects push the ratio the same way. The same-session **check-ON control**
reads **69.640 s ⇒ 61.602 / 69.640 = 0.885×**, reproducing the published figure
almost exactly — so the discrepancy is **harness asymmetry, not hardware drift**.

**Independent cross-check.** The `msm_g1` stage median is **6336.684 ms** of a
**58.762 s** `D::prove` = **10.8% offloadable ⇒ an Amdahl ceiling of 1.12×**. The
measured **1.048×** sits below that ceiling, so the two independent measurements agree.

🔴 **SCOPE — this supersedes exactly one number.**

- ✅ **Re-measured:** the **native-HIP G1-only** arm only → **1.048×**.
- ❌ **NOT re-measured:** **hip-wide `0.77×`** (G1+G2+FFT). It was produced by the
  same check-on harness, so it is a **lower bound**; its true paired value is
  unknown until it is re-run.
- ❌ **NOT re-measured:** the **OpenCL rows** at the top of this file
  (`demo-c-gpu.csv`) — a different script (`run-gpu-fold.sh`) that already defaults
  the check off. They are **outside this correction and remain exactly as published**.

**What did not change.** Even at **1.048×** this is still a
**correctness / plumbing** deliverable, not a speed story: only **10.8%** of the
prove is reachable by the seam at all. The remaining ~89% is **not** "a
CPU-Nova-dominated prove" — Nova's `prove_step`s are timed **separately**, total
**~0.6 s**, and sit **outside** the `D::prove` wall entirely
([`demo-c-phase-split-gfx1151.csv`](demo-c-phase-split-gfx1151.csv),
`inside_decider_wall=false`). What the seam cannot reach is the **Groth16 prover's
own CPU work** — R1CS synthesis, the QAP radix-2 FFT, the G2 (Fq2) B-query MSM,
witness map and serialisation (**52.895 s** in that capture).

The native HIP path produces a **bit-for-bit-equal, natively + on-chain verified**
DeciderEth proof — the `FOLD_GPU_MSM_CHECK=1` run reported all four BN254 **G1**
MSMs and the **G2 (Fq2)** MSM `GPU == CPU OK`, and every QAP radix-2 ifft /
coset-fft / coset-ifft pass (n=2^24) `GPU == ark-poly OK` (see
`artefacts/gpu-hip-wide/fold-check.log`). The value is the same
**correctness / plumbing** deliverable as OpenCL, now on a fully native (no
ec-gpu/OpenCL) toolchain. The micro-bench native-HIP MSM/NTT speedups (where the MSM
*is* the whole cost, e.g. the zkRAG `1.51×@2^20` end-to-end prove) are recorded under
`reading-notes/path-e-amd-gpu-zk-primitives.md` and the cross-engine scorecard.
Reproduce in a solo window with `make demo-c-fold-gpu-hip` (==
`bash scripts/run-gpu-fold-hip.sh`). Data: [`artefacts/demo-c-gpu-hip.csv`](demo-c-gpu-hip.csv).

## Honest caveats

- **The iGPU offload is not a win at this 2^24 decider size — the G1-only arm is at
  parity, and the wide arm has no clean measurement.** The earlier 1.34× / 1.64×
  "win" was produced by dividing the GPU times by a contention-inflated 173.83 s CPU
  sample; it does not survive a fair baseline. Against the clean solo CPU baseline
  (66.52 s median) this file's rows read **0.70×** (G1-only) and **0.74×** (wide) —
  ⚠️ but **the G1-only 0.70× is SUPERSEDED**: the 2026-08-29 paired re-bench reads
  **CPU 60.847 s vs iGPU 61.206 s = `0.994×`, parity**
  (`demo-c-paired-checkoff-opencl-gfx1151.csv`). 🔴 **Say parity and nothing else** —
  the 0.59% arm gap is smaller than the 0.93% / 1.21% within-arm spreads, so at
  `n=3` the arms are indistinguishable; never "the iGPU wins", never "0.6% slower".
  🔴 **`gpu-wide`'s 0.74× was NOT re-measured** and stays a floor. 🔴 **Parity is
  not acceleration.** The value of Demo C is unchanged by all of this: it is the
  **correctness/plumbing** result (GPU-MSM-driven DeciderEth proof == CPU proof,
  verifies natively + on-chain), which is independent of the speed sign.
- **iGPU contention (still partially in force).** The clean CPU baseline was captured
  solo, but the two GPU rows' `D::prove` then overlapped a third-party ROCm
  `train_language.py` (pid 194983) that woke at 08:43:52 and pinned the iGPU at
  ~100%. So the GPU numbers are **contended upper bounds** and a fully-clean solo GPU
  re-bench remains **blocked** by that resumed session. The reversal verdict does not
  depend on it: even a faster clean GPU time cannot beat 66.5 s from 89–95 s here.
- **`NovaDecider.sol` differs run-to-run** (CPU vs GPU contracts are NOT
  byte-identical) because each full run performs its own randomized Groth16
  trusted setup (`D::preprocess` with `OsRng`). This is expected; each contract
  verifies its own proof's calldata. The verification *pipeline* is unchanged.

## Reproduce

```bash
cd poc/folding-step-demo
cargo build --release --features gpu-msm           # stable toolchain (>=1.85)

# CPU baseline (same binary, GPU seam dormant):
FOLD_N=2 FOLD_GPU_MSM=0 ./target/release/folding-step-demo \
    --steps artefacts/steps.json --out artefacts/cpu

# GPU MSM offload:
FOLD_N=2 FOLD_GPU_MSM=1 ./target/release/folding-step-demo \
    --steps artefacts/steps.json --out artefacts/gpu
# add FOLD_GPU_MSM_CHECK=1 to gate every offloaded MSM against the CPU result.

# On-chain replay (per artefacts dir): copy NovaDecider.sol -> forge/src/,
# folded.calldata.hex -> artefacts/, then forge build + anvil + forge script
# script/VerifyFolded.s.sol  (see INTEGRATION-SPEC.md / Makefile demo-c-fold-gpu).
```
