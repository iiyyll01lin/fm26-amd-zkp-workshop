# Demo C — where the DeciderEth prove actually spends its time (phase split)

**What this answers.** The deck (S6) and several notes attribute the Demo C
slowdown to Amdahl and then name the wrong denominator: *"the wall is dominated
by the CPU-side Nova `prove_step`"*. This artefact measures the split directly.
**Nova `prove_step` is ~1% of the wall and is not inside the timed quantity at
all.** The real denominator is `D::prove` itself, of which the offloadable G1
MSMs are a **small minority**, and the rest is the Groth16 prover's own CPU work.

## The measurement

One clean solo run, `FOLD_N=2`, native-HIP G1-only arm, on the same
`gfx1151` / 32-thread Zen 5 host as every other Demo C row.

```
FOLD_N=2 FOLD_GPU_BACKEND=hip FOLD_GPU_MSM=1 FOLD_GPU_G2=0 FOLD_GPU_FFT=0 \
FOLD_GPU_MSM_CHECK=0 RAYON_NUM_THREADS=32 \
  ./target/release/folding-step-demo --steps artefacts/steps.json --out <scratch>
```

Provenance: **2026-08-29**, `solo=true` (iGPU busy **0%** by `rocm-smi`, 1-min
loadavg **0.28**, no `/tmp/zkp-gpu.lock` holder), binary built 2026-08-12 with
`--features gpu-msm-hip`, shipped defaults `work_units=2560 window=10`.
`DeciderEth native verification: OK`. Raw log:
[`demo-c-phase-split-gfx1151.log`](demo-c-phase-split-gfx1151.log).

| phase | engine | wall | share of `D::prove` | offloadable? |
|---|---|---|---|---|
| Nova `prove_step` ×2 | CPU | **0.6018 s** | *not in this wall* (0.98% of the two combined) | no |
| 4 × BN254 **G1 MSM** | **iGPU** | **7.7303 s** | **12.75%** | **yes — this is the whole numerator** |
| rest of `D::prove` | CPU | **52.8951 s** | **87.25%** | no |
| `D::prove` total | — | **60.6255 s** | 100% | — |

G1 stage breakdown (`g1_phase_ms`): alloc **482.713**, H2D **178.529**, kernel
**7019.535**, D2H **0.277**, free **49.285** ms.

**The Amdahl consequence.** Drive the offloaded MSMs to *zero* and `D::prove`
still cannot beat `1 / (1 − 0.1275)` = **1.15×**. A 2× faster MSM — which the
native-HIP port really does deliver at the primitive level — buys
`1 / (1 − 0.1275/2)` = **1.07×**. That is the honest reason the end-to-end
verdict barely moves, and it has nothing to do with Nova's folding steps.

## Cross-check against the committed check-on capture

`hip-gpu/fold.log` from the 2026-08-12 window-cap A/B session (same host, same
defaults, but `FOLD_GPU_MSM_CHECK=1`) recorded `D::prove` **69.973 s** with
`stage_ms msm_g1` **6342.461 ms** and `prove_step` **0.6010 s** — an offloadable
share of **9.06%**, ceiling **1.10×**. So across the two captures available the
numerator is **9–13%** and the ceiling **1.10–1.15×**. The conclusion is
insensitive to which one you take; only the second decimal moves.

## Two caveats, and one thing this run does NOT establish

1. **Single sample, and the GPU stage is the noisy part.** `msm_g1` measured
   6342 ms on 2026-08-12 and 7730 ms here at identical `work_units`/`window`
   (kernel 6097 → 7020 ms, alloc 157 → 483 ms). Both are one run. Quote the
   share as a **range**, not a point.
2. **`work_units=2560 window=10` is the shipped default, not the tuned pair.**
   `demo-c-window-cap-ab-gfx1151.csv` shows `10240`/`8` taking the same stage to
   **4350.224 ms**, which *shrinks* the numerator further — i.e. tuning the MSM
   makes the Amdahl ceiling **worse**, not better. That is the point.
3. **This run is NOT a re-bench of the `0.86×` headline — but one now exists.**
   `D::prove` here is 60.63 s against a `cpu-median` of 66.52 s recorded on
   2026-06-16 with a different build — an **unpaired** comparison across two
   months, so **no speedup is claimed from *this* file**. What it did do is
   surface a real methodology defect (recorded below), and that defect has since
   been corrected by a dedicated paired re-bench:
   [`demo-c-paired-checkoff-gfx1151.csv`](demo-c-paired-checkoff-gfx1151.csv)
   reads **1.048×** for the native-HIP **G1-only** arm, superseding `0.86×`.

## Flagged → since RESOLVED for G1-only: the published HIP rows were not an apples-to-apples pair

`scripts/run-gpu-fold-hip.sh` defaults **`FOLD_GPU_MSM_CHECK=1`** (line 74) for
its GPU arms, while `run_cpu()` runs with `FOLD_GPU_MSM=0` and no check. The
check **recomputes every offloaded MSM on the CPU inside the timed region**, so
the `hip-gpu` **77.45 s** and `hip-wide` **85.87 s** rows in
`demo-c-gpu-hip.csv` each carry CPU work their own baseline never pays, and the
resulting **0.86× / 0.77×** are therefore *floors*, not estimates. (The OpenCL
sibling `run-gpu-fold.sh` defaults the check to **0**, so the two harnesses do
not even agree with each other.)

**The paired re-bench this section called for has since been run — for the
G1-only arm.** [`demo-c-paired-checkoff-gfx1151.csv`](demo-c-paired-checkoff-gfx1151.csv)
(2026-08-29, one session, one binary, **both arms `FOLD_GPU_MSM_CHECK=0`**, arms
interleaved, 3 reps each, median) reads CPU **61.602 s** vs native-HIP G1-only
**58.762 s** = **1.048×**. The sign flips from slowdown to a small win, so
**`0.86×` is SUPERSEDED and must not be quoted.** The same-session **check-ON**
control reads **69.640 s ⇒ 61.602 / 69.640 = 0.885×**, reproducing the published
figure almost exactly — confirming the cause was **harness asymmetry, not drift**.
The Amdahl arithmetic above agrees rather than fights it: in that session `msm_g1`
was **6336.684 ms** of a **58.762 s** `D::prove` = **10.8% offloadable ⇒ a ceiling
of 1.12×**, and the measured **1.048×** sits below it.

🔴 **Scope — exactly one number moved.**

- ✅ **native-HIP G1-only**: re-measured, **0.86× → 1.048×**.
- ❌ **hip-wide `0.77×`**: **not** re-measured. It was produced by the same
  check-on harness, so it remains a **lower bound**, not an estimate.
- ❌ **The OpenCL rows** (`demo-c-gpu.csv`) came from `run-gpu-fold.sh`, which
  already defaults the check off; they are **outside the scope of this correction
  and remain exactly as published**.
