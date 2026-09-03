# Demo E — ZK on the AMD Radeon iGPU (ec-gpu / bellperson OpenCL)

> Path E Tier 1 + Tier 2. The honest, **measured** counter-point to the repo's
> "ZK proving is CPU-only on AMD" caveat: the *full r0vm STARK* has no AMD GPU
> backend, but the SNARK building blocks DO run on this Radeon 8060S
> (`gfx1151`) iGPU via the cross-vendor **OpenCL** path (Filecoin's `ec-gpu` /
> `bellperson` stack):
>
> - **Tier 1** (`gpu-primitive-bench`): the two hot primitives — **MSM** and
>   **NTT/FFT** — benchmarked GPU-vs-CPU.
> - **Tier 2** (`groth16-bench`): a **full Groth16 prove** on the iGPU vs CPU,
>   toggling `BELLMAN_NO_GPU` in-process for an apples-to-apples baseline.

## What it proves (and does not)

- **Does**: the AMD iGPU genuinely executes BLS12-381 MSM and scalar-field NTT
  through ROCm's OpenCL runtime, and we measure the speedup over the 32-thread
  Zen 5 CPU baseline (the same CPU baseline the Demo B STARK sweep uses).
- **Does NOT**: move the Demo B `r0vm` STARK or its Groth16 wrap onto the GPU.
  RISC0 only accelerates on CUDA/Metal; there is no AMD ROCm RISC0 prover, so
  that main line stays CPU-only. This is a **primitive-level capability**
  benchmark on BLS12-381 (blstrs), not the BN254 circuit of Demo B/C.

## Result (measured on AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S, ROCm 7.2.3)

`speedup = cpu_ms / gpu_ms` (>1 means the iGPU beats 32-thread Zen 5):

- MSM (G1 multiexp): `2^16` 0.80x, `2^18` 0.91x, `2^20` 1.06x, `2^22` **1.21x**
  — the iGPU only pulls ahead at large sizes; MSM is **size-gated (ec-gpu OpenCL
  path)**, so break-even is ~`2^20` (2026-06-18 clean-solo re-bench,
  `solo=true`). **That is a property of that kernel, not of MSM and not of the
  shared memory** — see "Why the MSM is size-gated" below.
- NTT/FFT (scalar field): `2^16` 4.75x, `2^20` 3.83x, `2^22` **5.55x**, `2^24` 5.13x
  — the iGPU is a consistent ~5x faster; NTT maps cleanly onto the GPU.

### Why the MSM is size-gated — the kernel, not the shared memory

Same gfx1151, same 32-thread Zen 5, same binary, switching only
`FOLD_GPU_BACKEND`: BN254 G1 MSM at `2^16`/`2^18`/`2^20` goes from
39.602/122.475/443.479 ms (ec-gpu OpenCL) to 18.005/58.788/219.943 ms (native
HIP) — 2.0–2.2× faster — flipping the CPU comparison from losing
(0.646/0.819/0.821×) to winning (1.510/1.744/1.632×).

**Say what that one variable actually changes.** `FOLD_GPU_BACKEND` selects the
`Bn254Gpu` construction path, and the two paths differ in *two* ways: OpenCL runs
`work_units = 128*CU` with `window_cap = 10` (`src/lib.rs:640-641`), native HIP runs
`512*CU` with an integrated `window_cap = 8` (`:86-88`, `:119-125`, `:674-683`). So
2.0–2.2× is **backend/toolchain × occupancy tuning**, and the paired 128*CU rows in
[`artefacts/hip-msm-tune.solo-2026-06-18.csv`](artefacts/hip-msm-tune.solo-2026-06-18.csv)
split it: at equal `work_units` the HIP kernel is **1.56× / 1.57× / 1.62×** (24.82 vs
38.66, 77.30 vs 121.08, 268.23 vs 434.56 ms), and the occupancy step adds **1.40× /
1.30× / 1.32×**. The comparison is still fair — `512*CU @ w=8` and `128*CU @ w=10`
allocate **identical** bucket bytes, 240 MB at 20 CU (`:93-97`) — so state it as
"backend swap **plus** occupancy tuning, at equal bucket memory", not "only the backend
changed".

**Same memory, same
contention** — if the LPDDR5X roof were really the seal, neither factor could exist:
the pure kernel swap alone is 1.56–1.62× at identical launch geometry, and occupancy
has nothing to do with bandwidth. The HIP-side G1 point-addition primitive measures at
**95.8–97.8%** of its own raw multiply ceiling (gfx1151, `n=2^20`), i.e.
arithmetic-bound. Evidence:
[`artefacts/msm-ntt-backend-gfx1151.csv`](artefacts/msm-ntt-backend-gfx1151.csv),
`artefacts/baseline-efficiency-gfx1151-20260812T141527Z/`.
**Note**: this does not establish that MSM is compute-bound; it establishes that
*that* OpenCL kernel was nowhere near the roof.

Full data: [`artefacts/gpu-primitive.csv`](artefacts/gpu-primitive.csv) /
[`artefacts/gpu-primitive.md`](artefacts/gpu-primitive.md);
chart: `artefacts/gpu-primitive.png`.

### Tier 2 — full Groth16 prove (BLS12-381, bellperson)

`speedup = cpu_prove_ms / gpu_prove_ms` (>1 means the iGPU beats 32-thread Zen 5):

- `2^16` 0.64x, `2^18` 0.74x, `2^20` 0.87x, `2^22` **1.015x** — the prove-time
  gap closes monotonically and the iGPU reaches **parity (a slight 1.015x win) at
  ~4M constraints** (`2^22`, 2026-06-18 clean-solo; was 0.997x), extrapolating to
  a clearer GPU win beyond that. Verification passes on both
  GPU and CPU proofs at every size. The crossover matches Tier 1: a Groth16
  prove is MSM-dominated, and MSM only favours the iGPU at large sizes because
  of the ec-gpu OpenCL kernel it runs on — not because it shares LPDDR5X with
  the 32-thread CPU (see "Why the MSM is size-gated" above).

Data: [`artefacts/gpu-groth16.csv`](artefacts/gpu-groth16.csv) /
[`artefacts/gpu-groth16.md`](artefacts/gpu-groth16.md);
chart: `artefacts/gpu-groth16.png`. This is a
BLS12-381 *capability* demo — NOT the Demo B RISC0 STARK->SNARK wrap (BN254,
x86 prover, CPU-only) nor the Demo C Sonobe DeciderEth.

## Prerequisites

- GPU-ZK-READY box per
  [`../risc0-cartesi-step-demo/scripts/gpu-zk-probe.sh`](../risc0-cartesi-step-demo/scripts/gpu-zk-probe.sh):
  ROCm 7.2+ with the OpenCL ICD (`clinfo` enumerates the GPU), `/dev/kfd` +
  `/dev/dri/renderD*`. Kernel >= 6.18.4 is the recommended gfx1151 stable line
  (this run was captured on 6.17 and worked).
- Rust `stable` (>= 1.85; pinned via `rust-toolchain.toml` — ec-gpu's deps need
  edition2024, so the RISC0-pinned 1.83 default is too old here).

## Run

```bash
# Tier 1 — MSM + NTT primitives
bash scripts/run-all.sh
MSM_POWERS=16,18,20,22 FFT_POWERS=16,18,20,22,24 REPS=3 bash scripts/run-all.sh

# Tier 2 — full Groth16 prove (committed set incl. the 2^22 crossover):
bash scripts/run-groth16.sh
G16_POWERS=16,18,20,22 SAMPLES=3 bash scripts/run-groth16.sh   # ~7 min (2^22 setup heavy)
```

Each `run-*.sh` runs the readiness probe, `cargo build --release`, the
GPU-vs-CPU sweep (CSV + log), then its plot script (PNG via matplotlib, else a
markdown table). The first GPU prove JIT-compiles the OpenCL kernels into
`~/.rust-gpu-tools` (one-time; excluded from the reported min times).

## Layout

```
amd-gpu-zk-primitive-demo/
├── Cargo.toml            # blstrs 0.7 + ec-gpu-gen + bellperson (opencl), no CUDA
├── rust-toolchain.toml   # channel = stable
├── build.rs              # SourceBuilder: add_fft::<Scalar>() + add_multiexp::<G1Affine, Fp>()
├── src/
│   ├── main.rs           # Tier 1: GPU (MultiexpKernel/FftKernel) vs CPU (multiexp_cpu/parallel_fft)
│   └── bin/groth16-bench.rs  # Tier 2: bellperson Groth16 prove, BELLMAN_NO_GPU toggle
├── scripts/
│   ├── run-all.sh        # Tier 1: probe -> build -> sweep -> plot
│   ├── run-groth16.sh    # Tier 2: probe -> build -> sweep -> plot
│   ├── plot-primitive.py # 3-panel MSM/NTT GPU-vs-CPU chart, markdown fallback
│   └── plot-groth16.py   # 2-panel Groth16 prove-time + speedup chart, markdown fallback
└── artefacts/            # gpu-primitive.* + gpu-groth16.* (committed as evidence)
```

See [`../../docs/amd-strix-halo-acceleration.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/amd-strix-halo-acceleration.md)
and [`../../reading-notes/path-e-amd-gpu-zk-primitives.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/reading-notes/path-e-amd-gpu-zk-primitives.md)
for the full engine map and where this sits in the report.
