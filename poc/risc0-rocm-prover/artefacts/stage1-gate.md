# Stage 1 gate — BabyBear field + Poseidon2 + SHA-256 (bit-for-bit)

Ported risc0's STARK hash/field **primitives** to native HIP and validated them
**bit-for-bit against the risc0 CpuHal golden** on the Strix Halo gfx1151. This is
the plan's Stage 1 gate: "field ops + Poseidon2 + SHA GPU==CPU bit-for-bit + field
constant host cross-check."

## What was ported (`kernels/hip/`)

| header | source | port change |
| --- | --- | --- |
| `babybear.hpp` | `sys/kernels/zkp/cuda/{fp.h,fpext.h}` | `__device__` → `__host__ __device__`; else byte-identical |
| `sha256.hpp` | `sys/kernels/zkp/cuda/sha256.h` | same |
| `poseidon2.hpp` | RUST `zkp/src/core/hash/poseidon2/mod.rs` over native `Fp` | re-expressed over `Fp` (not sppark `bb31_t`) |
| `poseidon2_constants_generated.hpp` | GENERATED from `risc0_zkp::…::{ROUND_CONSTANTS,M_INT_DIAG_HZN}` | raw Montgomery, zero transcription |

Montgomery constants (M=0x88000001, R2=1172168163) are byte-identical to
`risc0_core::field::baby_bear`, so results compare **raw-word bit-for-bit**.

## Golden = risc0 CpuHal (pure CPU dumper)

`vendor/risc0/risc0/zkp/examples/dump_stage1_vectors.rs` emits from the pinned
risc0 crates: field add/sub/mul/neg/inv/sqr (Fp + Fp4), `poseidon2_mix`,
`unpadded_hash`, `hash_elem_slice`, `hash_pair`. The Poseidon2 golden also matches
the **independent hard-coded test vector** in risc0's own `#[test]`
(`poseidon2_test_vectors`): input `0..23` → std-form `0x2ed3e23d, 0x12921fb0, …`.

## Gate run (`bash scripts/run-stage1.sh`, gfx1151, ROCm 7.2.3)

```
[STAGE1-FIELD]     checked=5632 fail=0 -> PASS (GPU==CPU bit-for-bit, host+device)
[STAGE1-POSEIDON2] checked=1768 fail=0 -> PASS (GPU==CPU bit-for-bit incl. golden test vectors)
[STAGE1-SHA256]    checked=400  fail=0 -> PASS (GPU==CPU bit-for-bit, host+device)
STAGE 1: PASS
```

- Each harness checks **both** the on-device kernel result and an on-host
  recomputation (the `__host__ __device__` headers) against the CpuHal golden —
  the contention-independent field-constant cross-check the plan asks for.
- Fp/Fp4: 128 random pairs × 6 ops × (host+device) = 5632 word-checks.
- Poseidon2: golden mix (raw+std) + 32 random mixes + 10 variable-length
  `unpadded_hash` (rate-aligned & unaligned), host+device = 1768 checks.
- SHA-256: 9 `hash_elem_slice` (pad=false) + 16 `hash_pair`, host+device = 400.

Correctness is deterministic bitwise equality, so this gate is
contention-independent (no solo window needed; timing/bench is Stage 4).

**STAGE 1: PASS.**
