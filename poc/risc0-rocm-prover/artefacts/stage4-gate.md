# Stage 4 gate — end-to-end `rocm` prove + verify + bench: PASS

**A GPU-produced rv32im segment STARK seal verifies, and it was produced by the
HipHal path (not a CPU fallback).** This is the plan's Stage 4 gate. All evidence
is from real commands on the Strix Halo (gfx1151, ROCm 7.2.3).

> **2026-07-20 re-measurement + Part A/B (supersedes the point numbers below).**
> Reconstructed from the pinned overlay (DualHal **15/15**, GPU prove **26.3 s**,
> stock-verify ✅), then re-measured **solo**: fresh headline **5.46×** (iGPU 26.1 s vs
> same-code 32t CPU 142.4 s, po2=20); the older "~6.6–6.8×" = 5.46× **×** a **1.25×**
> local-vs-installed codegen gap (installed rzup r0vm = 177.6 s). New data:
> **A1 phase breakdown** — iGPU 59.4 % / CPU 40.6 %, witgen+accum **28.87 %** ⇒ ≤**1.41×**
> Amdahl ceiling ([`phase-breakdown.md`](phase-breakdown.md)); **A2 handoff** — **0
> hipMemcpy** ([`handoff-cost.md`](handoff-cost.md)); **A3** ([`a3-speed.md`](a3-speed.md));
> **B** — `steps.cu` compiles to native HIP and the GPU witgen/accum paths pass their
> gates; they stay default-off because the gfx1201 A/B measures them wall-neutral at
> 13–15% more energy ([`partb-status.md`](partb-status.md)). Engine map:
> [`engine-map-ledger.md`](engine-map-ledger.md).

## What was built (the two deferred integration sub-parts + end-to-end)

- **HipBuffer + HIP runtime FFI** — HIP *managed* (unified) memory
  (`hipMallocManaged`), so one pointer serves host + gfx1151. `has_unified_memory
  = true`. (`risc0/zkp/src/hal/hip.rs`.)
- **Full `HipHal` (zkp `Hal` trait)** over the Stage-1/2-validated kernels
  (`risc0/sys/kernels/zkp/hip/ffi.hip`, butterfly-parallel NTT). **Validated by
  risc0's OWN `DualHal` harness (CpuHal vs HipHal): 15/15 equality tests PASS**
  (`cargo test -p risc0-zkp --features rocm hal::hip`).
- **`HipCircuitHal` (rv32im)** — `eval_check` on the iGPU (the 26k-LOC generated
  poly, field-swapped); **witgen + accum DELEGATED to risc0's CPU C++** writing
  directly into the managed buffers the GPU reads. eval_check GPU==CPU crate test
  PASS (`cargo test -p risc0-circuit-rv32im --features rocm prove::hal::hip`).
- **`rocm` feature threaded** `zkvm → circuit/rv32im → zkp → sys` (scoped to the
  rv32im segment STARK + zkp HAL; recursion/keccak stay CPU per the plan), and
  `r0vm --features rocm` builds (82 MB binary).

Note on task #1 (hipify `steps.cu`): the witgen/accum are CPU-delegated over
unified memory rather than hipified — this yields a **working, verified** seal
without porting the 30k-LOC witgen. The honest split is stated everywhere:
**STARK math + eval_check on iGPU; witgen + accum on CPU.**

## Gate: GPU seal verifies (real STARK, GPU path)

```
$ r0vm(--features rocm) --elf cartesi-risc0-guest-step-prover.bin \
      --initial-input step.bin --receipt step.rocm.proof.bin
[risc0-rocm-prover] HipHal segment prover ACTIVE on gfx1151 \
    (STARK math + eval_check on iGPU; witgen/accum CPU-delegated)   # runtime marker
$ cargo risczero verify --path step.rocm.proof.bin 3aec…c990
✅ Receipt is valid!
```

- Receipt: **1112064 bytes** — identical size to the golden CPU seal.
- **GPU busy peaked at 95%** (80/83 rocm-smi samples nonzero) during the prove —
  the iGPU genuinely did the STARK compute (not a CPU fallback).
- The marker line is emitted only by `prove::hal::hip::segment_prover()`.

> **Annotation 2026-08-29 — the receipt bullet above is correct but conservative.**
> The line is **kept verbatim as the record of what this gate measured**; this note only
> adds what has been measured *since*. "Identical size" is the safe floor. The defensible
> stronger statement is *same size **plus** a bit-identical journal*, and the equally
> load-bearing limit is that the seal **body** is not byte-reproducible.
>
> - **Journal tail — 385 bytes, bit-identical.** The golden
>   [`step.proof.bin`](../../risc0-cartesi-step-demo/artefacts/) and
>   `step.rocm.proof.bin` share a 156-byte serialization header
>   (first differing byte at offset `0x9c`) and a 385-byte tail. That tail carries
>   `pre_root || mcycle || post_root`, the middle 32-byte word being `00…0064`
>   (`0x64` = mcycle 100), field-for-field equal to
>   [`step.public.json`](../../risc0-cartesi-step-demo/artefacts/step.public.json); the
>   96-byte concatenation matches `JOURNAL_HEX` in
>   [`groth16-seam-evidence-20260724T060845Z/journal.log`](groth16-seam-evidence-20260724T060845Z/journal.log)
>   exactly, and occurs twice inside that 385-byte tail.
> - **Seal body — 87.39% of the bytes differ.** The same pair differs in **971,857 of
>   1,112,064 bytes**. All **12** committed 1,112,064-byte seals in this repo have
>   pairwise-distinct md5s — e.g. `receipt.gpu.bin` `73eed04a…` vs `receipt.cpu.bin`
>   `6b65e97f…` in
>   `clean-room/rdna4-gfx1201-evalfix-benchmark-20260721T201329Z/`.
> - **Measured head-on, five times.**
>   [`z2-bringup-report.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/workshop/futuremode-2026/z2-bringup-report.md) §3.3
>   and [`cameo-shot-list.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/workshop/futuremode-2026/cameo-shot-list.md) §3.4
>   both record **five runs producing five *different* 1,112,064-byte files, all passing
>   stock verify** — same size, different bytes: **expected behaviour, not a defect.**
> - **The correct scope of "bit-for-bit" is the core ops, never the receipt file.**
>   Stage 1 field / Poseidon2 / SHA-256 (`checked=5632 / 1768 / 400`,
>   [`stage1-run.log`](stage1-run.log)); Stage 2 polynomial HAL kernels
>   (`checked=38096`); Stage 3 Merkle (`checked=1984`) + rv32im `eval_check`
>   (`checked=1024`); and **DualHal 15/15** — `dualhal.log:283-300` lists all 15 named
>   tests as `running 15 tests` / `15 passed`, and
>   [`kernel-inventory.md`](kernel-inventory.md) (§ HAL surface) records that `DualHal`
>   "runs each op on **two** HALs and asserts equality".
> - **Mechanism deliberately not attributed.** Per-run seal variation is *consistent
>   with* zk blinding (`zk_shift` is one of the 15 DualHal ops), but nothing in this repo
>   measures a mechanism-level attribution. Cite that as **inference, not evidence.**

## Bench (honest; solo=true, loadavg 0.59)

**Workload:** the **~4-segment composite Cartesi-step prove** (`po2=20` is the
per-segment **limit** — the HipHal marker fires **4×** and preflight shows 4
segments: 3×po2=20 + 1×po2=19). Both CPU and GPU prove the **identical** session,
so the ratio is unaffected by segment labeling. (An earlier draft mislabeled this
"one po2=20 segment, ~920k cycles"; `920485` was a preflight *suspend* value, not
a verified total cycle count, and has been removed.)

Controlled, **same fork code**, 32 threads, `RAYON_NUM_THREADS=32`:

| config | backend | wall | receipt | verify |
| --- | --- | --- | --- | --- |
| fork CPU (no rocm) | 32t Zen5 | **172.9 s** | 1112064 | ✅ valid |
| fork GPU (rocm) | gfx1151 (+CPU witgen) | **26.2 s** (min of 26.6/26.2) | 1112064 | ✅ valid |
| installed r0vm 2.3.2 | 32t Zen5 (rzup build) | 254.3 s | 1112064 | ✅ valid |

**iGPU end-to-end ≈ 6.6× vs this fork-CPU run's 172.9 s** (≈9.7× vs the rzup binary)
on this workload — **superseded** by the fresh same-code headline **~5.46×** (26.1 s
vs 142.4 s; see the banner). The **independent audit's 6.79×** (178.0 s CPU vs 26.2 s
GPU) divided by the installed rzup r0vm's slower codegen, i.e. `~6.6–6.8× = 5.46× ×
a 1.25× local-vs-shipped codegen gap`, not random CPU noise.

### GPU path is real (not a CPU fallback) — audit differential test

The audit ran the **non-rocm** binary on the identical session: the iGPU stayed at
**0% busy across 156 samples / 178 s with 0 HipHal markers**, vs the rocm binary's
**95% busy (375/385 samples) + 4 markers**. This is definitive proof the STARK
compute ran on the gfx1151, not on the CPU.

### Honest framing — what is safe to state (mirroring docs/INTEGRITY-REPORT.md)

- **HARD correctness guarantee (state flatly):** the GPU-produced seal is accepted
  by the **stock** `cargo risczero verify`, and every HAL op is **bit-for-bit
  CpuHal==HipHal** (DualHal 15/15 + eval_check GPU==CPU).
- **Speed is a scoped secondary result:** a later solo **segment-po2 sweep**
  ([`stage4-roofline.md`](stage4-roofline.md), [`stage4-sweep.csv`](stage4-sweep.csv))
  turns this single point into a curve: a **flat ~5.3–5.5×** across po2 16→21 (no
  crossover; iGPU wall stable ~26 s). The cleanest solo CPU baseline (po2=20,
  loadavg 0.07) is 144.2 s ⇒ **5.52×** (matching the fresh same-code 5.46×); the
  ~6.6× / audited 6.79× divided by the installed rzup r0vm's slower codegen
  (177.6 / 178 s), i.e. `5.46× × a 1.25× local-vs-shipped codegen gap`, not
  CPU-baseline noise. It
  is **workload-specific** — poseidon2 (the **only** hashfn the rv32im prover
  accepts) + the **HYBRID** design (CPU witgen+accum + GPU STARK over unified
  memory). It is **not** a general "iGPU is faster" claim; peak RSS reaches 18.9 GB
  at po2=21 (past the 16 GB wall, held by the 94 GB unified pool).
- **recursion + Groth16 wrap remain CPU** (scoped out).

## Verdict

**STAGE 4: PASS.** GPU-produced ~4-segment Cartesi-step STARK verifies
(`✅ Receipt is valid!`), provably via the HipHal iGPU path (audit-confirmed by a
differential fallback test), and is **~5.46× faster** than the same-code CPU build
on this workload (flat ~5.3–5.5×; the old ~6.6–6.8× = 5.46× × a 1.25× local-vs-shipped
codegen gap) — a workload-specific figure; correctness is
the hard guarantee (see framing above). Reproduce:
`bash scripts/run-stage4-prove.sh` + `scripts/run-stage4-bench.sh`.
