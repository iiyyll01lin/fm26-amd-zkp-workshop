# RISC0 ROCm release/evidence checklist

> **DRAFT ONLY — do not push, open a PR, or publish AMD performance claims.**

## Fixed source snapshot

- Upstream pin: `risc0/risc0` `v2.3.2`,
  `218e3bc4a8ffcd203a9cd4e46f921bf60aa7e2bd`.
- Post-rebase adaptive-launch source-manifest SHA-256:
  `50f9bf12ac2422c4bd72d882af3a4988346949256f610a8df00f997834bb0b15`
  (88 files, checked before and after the `gfx1201` run). The original
  fixed-launch clean-room manifest remains
  `4e2d696180ad4a6f02a744d5233ef3a09bd74483204dbbf31fc78ca924005f56`.
- Portability fix: a shared target-neutral `eval_check` launch policy reads
  function/device limits and per-thread private storage, then selects a
  power-of-two block under a conservative 512 KiB workgroup budget. It selects
  16 for the measured 24,288-byte `poly_fp`, falls back to 16 if metadata is
  incomplete, logs its reasoning, and rejects unsafe debugging overrides.
  Exact GPU==CPU gates are unchanged.
- Toolchain portability: `HIP_SYMBOL(...)`, configurable `HIPCC`/`ROCM_PATH`,
  and explicit `amdhip64` linking cover the tested incomplete-ROCm adapter.

## Canonical evidence directories

- `artefacts/clean-room/gfx1151-20260721T172533Z/`
  - Clean clone/cache PASS on `gfx1151`; default hybrid; stock-verified
    1,112,064-byte receipt and non-zero GPU telemetry.
- `artefacts/clean-room/rdna4-gfx1201-20260721T182652Z/`
  - Preserved initial `gfx1201` FAIL: standalone `eval_check` mismatched
    1024/1024. Later gates and benchmark were correctly stopped.
- `artefacts/clean-room/rdna4-gfx1201-eval-blockprobe-20260721T190301Z/`
  - Root-cause evidence: 24,288 private bytes/thread; blocks 1–16 stable;
    blocks >=32 corrupt/nondeterministic.
- `artefacts/clean-room/gfx1151-evalfix-20260721T191253Z/`
  - Repeated 16-thread equality PASS on `gfx1151`, plus Stage 1/2/Merkle
    regressions.
- `artefacts/clean-room/rdna4-gfx1201-evalfix-full-20260721T194113Z/`
  - Full default-hybrid PASS on `gfx1201`: DualHal 15/15, eval equality,
    build/prove/stock verify, 1,112,064-byte receipt SHA-256
    `91f40feac7912041f9a170774a68b1c7c65d63dbadd97329c960bf514ed6487a`,
    and 24/26 non-zero telemetry samples with 100% max use.
- `artefacts/clean-room/rdna4-gfx1201-evalfix-benchmark-20260721T201329Z/`
  - Solo same-host benchmark PASS: CPU 95.940 s, GPU 26.590 s, 3.61×;
    both receipts stock-verified.
- `artefacts/correctness/gfx1151-adaptive-launch-20260722T014502Z/`
  - Shared-policy standalone and production eval 3/3 each, Stage 1/2/Merkle,
    DualHal 15/15, validated override/rejection behavior, and stock-verified
    default-hybrid receipt.
- `artefacts/correctness/gfx1201-adaptive-launch-final-20260722T033500Z/`
  - Exact post-rebase source, one visible `gfx1201` device, standalone and
    production eval 3/3 each, Stage 0/1/2/Merkle, DualHal 15/15, and
    stock-verified default-hybrid receipt. Tooling/harness failures from the two
    stopped attempts are preserved in the final evidence.

Supporting preserved records:

- `artefacts/clean-room/rdna4-gfx1201-prep-20260721T165955Z/`
- `artefacts/clean-room/rdna4-gfx1201-eval-investigation-20260721T185717Z/`
- `artefacts/clean-room/rdna4-gfx1201-evalfix-20260721T191253Z/`
- `artefacts/clean-room/rdna4-gfx1201-evalfix-full-20260721T191253Z/`

These include host preparation, compiler-adapter classification, a focused
fixed-kernel PASS, and an honestly preserved failed full-run attempt.

## Durable claims

- [x] Correctness passes on `gfx1151` and `gfx1201`.
- [x] End-to-end accepted path is hybrid: STARK math + `eval_check` on HIP;
  segment witgen + accum CPU-delegated; recursion and Groth16 MSM/NTT on iGPU, witness/sequential parts CPU.
- [x] Stock `cargo risczero verify` accepts both canonical clean-room receipts.
- [x] Failed `gfx1201` evidence remains present and checksummed.
- [x] `gfx1151` speed remains its same-host, same-code ~5.3–5.5× range.
- [x] `gfx1201` speed is only the separate same-host 3.61× result.
- [x] No speed comparison is made across hosts.

## Known limitations

- GPU witgen/accum is default-off, diverging from upstream CUDA, which runs both
  on the GPU unconditionally (`hal/cuda.rs:100`, `:155`). The cross-cycle-race
  diagnosis was withdrawn on 2026-07-22 once the destructive buffer fill was
  found; the switches stay off because the measured 28.87% share caps the ideal
  gain at ~1.41× and the gfx1201 A/B measures all four combinations within 2% on
  wall at 13–15% more energy.
- Only `gfx1151` and `gfx1201` are validated; other RDNA targets and CDNA are
  untested.
- One `gfx1201` installation required a local HIP compiler adapter because the
  ROCm layout was incomplete.
- The port remains pinned to RISC0 `v2.3.2`. The manual AMD-GPU workflow needs
  separately labelled `gfx1151` and `gfx1201` self-hosted runners; missing
  runners remain queued and missing tooling fails unless dispatch explicitly
  permits a recorded skip.
- Workload-specific performance is not an architecture-wide, cross-host,
  cross-vendor, or stock-upstream claim.

## Local release gates

- [x] `make lab-replay` reports **25/25 PASS** (2026-07-24), including Lab 24 bottleneck forensics.
- [x] `make course-build` and `make course-build-en` pass.
- [x] Clean-room helper safety checks, `bash -n`, `shellcheck`, and diff lint pass.
- [x] With GPU witgen/accum unset, DualHal passes 15/15 and the focused rv32im
  `eval_check` test passes 1/1 on local `gfx1151`.
- [x] All committed evidence checksum manifests and secret/credential scans pass.
- [x] Local commit series is reviewed and prepared for a clean-tree handoff.

## Manual-only publication gates

- [x] Rebase the local series onto current `origin/main` without rewriting
  remote history.
- [x] Re-run required correctness gates after that rebase on both architectures.
- [x] Review DCO/sign-offs and the final source diff.
- [ ] Obtain maintainer approval for scope and the incomplete-ROCm adapter.
- [ ] Manually push the reviewed branch.
- [ ] Manually open a PR only after removing the `DRAFT ONLY — do not open`
  hold.
- [ ] Publish any AMD performance statement only through its separate approval
  process.
