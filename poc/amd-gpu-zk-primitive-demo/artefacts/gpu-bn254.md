# Path E Phase 3 — BN254 G1 MSM, iGPU OpenCL vs arkworks CPU

_Source: `poc/amd-gpu-zk-primitive-demo/artefacts/gpu-bn254.csv` — rendered alongside PNG._

_iGPU = AMD Radeon 8060S (gfx1151), ec-gpu OpenCL `Bn254G1_multiexp` kernel; CPU = 32-thread Zen 5 via arkworks `VariableBaseMSM` (the MSM Sonobe/Demo C call). `speedup = cpu_ms / gpu_ms` (>1 = iGPU wins). Every cell is verified (GPU result == arkworks CPU result). This is the **BN254** curve the real proof path uses — a curve-faithful extension of the Tier 1 BLS12-381 capability proof; it does NOT move the r0vm STARK onto the GPU (no upstream AMD RISC0 prover; scoped fork = path-i)._

| primitive | log_size | size | gpu_ms | cpu_ms | speedup | gpu_device | cpu_threads |
| --- | --- | --- | --- | --- | --- | --- | --- |
| msm | 16 | 65536 | 39.255 | 25.681 | 0.654 | gfx1151 | 32 |
| msm | 18 | 262144 | 123.316 | 100.361 | 0.814 | gfx1151 | 32 |
| msm | 20 | 1048576 | 452.99 | 383.001 | 0.845 | gfx1151 | 32 |
| msm | 22 | 4194304 | 1518.493 | 1656.943 | 1.091 | gfx1151 | 32 |
| ntt | 8 | 256 | 0.15 | 1.101 | 7.367 | gfx1151 | 32 |
| ntt | 10 | 1024 | 0.156 | 17.909 | 114.698 | gfx1151 | 32 |
| ntt | 12 | 4096 | 0.199 | 292.208 | 1469.749 | gfx1151 | 32 |
