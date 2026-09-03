# Stage 4 status — SUPERSEDED

The earlier honest "BLOCKED (integration)" status was resolved in a follow-up
run. Stage 4 is now **PASS**: a GPU-produced rv32im segment STARK seal (the
**~4-segment composite Cartesi-step prove**; po2=20 is the per-segment limit) is
accepted by the stock `cargo risczero verify` (`✅ Receipt is valid!`), provably
via the HipHal iGPU path, and benches ~5.46× (flat ~5.3–5.5×; the old ~6.6–6.8× =
5.46× × a 1.25× local-vs-shipped codegen gap) faster than the same-code 32-thread
CPU build on that workload. The speedup is
workload/hashfn/hybrid-specific; the hard guarantee is correctness.

See **`stage4-gate.md`** (authoritative) + `stage4-bench.csv`.
