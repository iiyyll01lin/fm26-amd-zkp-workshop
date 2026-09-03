# Stage 3 gate — Merkle build + rv32im eval_check (bit-for-bit)

Stage 3 is the plan's flagged hardest stage ("`eval_check.cu` 自動產生且巨大 →
最可能卡住"). Result: the two hard kernel families both port to gfx1151 and match
risc0 **bit-for-bit**. The accum/witgen kernel is characterised honestly below.

## 3a. rv32im `eval_check` / `poly_fp` — PASS (the flagged-hardest kernel)

The generated `eval_check_{0,1,2,3}.cu` (7695+6323+6203+6042 = **26,263 LOC** of
machine-generated BabyBear arithmetic; 20 `rv32im_v2_*` fns + `poly_fp`) was
ported to gfx1151 by **swapping only the field backend** — a one-line
`supra/fp.h` shim mapping risc0's `Fp`/`FpExt` onto the Stage-1-validated
`babybear.hpp`. **No generated code was edited.** Built as a single unity TU
(hipcc `--offload-arch=gfx1151`, ~4 min) so `poly_fp` sees the cross-file
`__device__` fns without device linking.

Golden = risc0's **own CPU C++ reference** `cxx/rust_poly_fp_*.cpp` (~52k LOC),
compiled with g++ and called via `risc0_circuit_rv32im_cpu_poly_fp`.

```
$ bash scripts/run-stage3-evalcheck.sh
[STAGE3-EVALCHECK] checked=1024 fail=0 -> PASS (rv32im poly_fp GPU==CPU C++ bit-for-bit)
STAGE 3 (eval_check): PASS
```

256 cycles × 4 FpExt components = 1024 bit-for-bit comparisons vs the CPU C++
golden, all equal. **This refutes the compile-level intractability**: the "huge
generated kernel" hipifies mechanically via the field swap and is bitwise correct
on the iGPU. (`poly_fp` is a deterministic function of its input buffers, so a
random — not necessarily valid-witness — input is a sound equality test.)

## 3b. Merkle build `hash_rows` / `hash_fold` — PASS

Reuses the Stage-1-validated `sha256.hpp` + `poseidon2.hpp`. Golden = the actual
`CpuHal::{hash_rows,hash_fold}` for **both** hash suites.

```
$ bash scripts/run-stage3-merkle.sh
[STAGE3-MERKLE] checked=1984 fail=0 -> PASS (hash_rows/hash_fold SHA+Poseidon2 GPU==CPU bit-for-bit)
STAGE 3 (merkle): PASS
```

## 3c. accum / witgen (`steps.cu`, `ffi.cu`) — NOT completed (honest)

The permutation-accumulation + witness-generation kernels (`stepAccum`,
`par/rev/fwd_stepExec` in `ffi.cu`; `steps.cu` = 30k LOC) use risc0's **native**
`fp.h` (shim-compatible, like eval_check) **plus** `<cuda/std/array>` and the
`cuda.h` launch glue. Porting is the *same class* of mechanical hipify as
eval_check but with more touch points (`cuda/std/array` → `std::array`, `cuda.h`
→ HIP launch, `cudaMemcpyToSymbol` → `hipMemcpyToSymbol`). **No fundamental
blocker was found**, but it was not completed this run — it is integration work,
not new math, and the field/hash/eval_check math it depends on is already proven.

## Verdict

- **eval_check (hardest): PASS bit-for-bit on gfx1151.**
- **Merkle build: PASS bit-for-bit on gfx1151.**
- accum/witgen: tractable mechanical hipify, deferred (integration).

The plan's predicted intractability point (`eval_check`) is **cleared** at the
kernel/correctness level. The remaining Stage 3 item and all of Stage 4 are
integration (wire kernels into the Rust HAL + witgen + build `r0vm --features
rocm`), which is the honest boundary reached this run.
