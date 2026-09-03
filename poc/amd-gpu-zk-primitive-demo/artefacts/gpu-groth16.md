# Path E Tier 2 — full Groth16 prove, iGPU OpenCL vs CPU

_Source: `poc/amd-gpu-zk-primitive-demo/artefacts/gpu-groth16.csv` — rendered alongside PNG._

_BLS12-381 bellman-style Groth16 (bellperson, ec-gpu OpenCL). `speedup = cpu_prove_ms / gpu_prove_ms` (>1 = iGPU wins). A *capability* demo that AMD's iGPU runs a real Groth16 prover; NOT the Demo B RISC0 STARK->SNARK wrap (CPU-only)._

| constraints_pow | constraints | setup_ms | gpu_prove_ms | cpu_prove_ms | speedup | verify_ok | gpu_device | cpu_threads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 65536 | 2010.6 | 820.5 | 525.6 | 0.641 | true | gfx1151 | 32 |
| 18 | 262144 | 7367.6 | 2529.3 | 1867.7 | 0.738 | true | gfx1151 | 32 |
| 20 | 1048576 | 28415.0 | 8415.3 | 7294.6 | 0.867 | true | gfx1151 | 32 |
| 22 | 4194304 | 111068.7 | 28931.6 | 29373.8 | 1.015 | true | gfx1151 | 32 |
