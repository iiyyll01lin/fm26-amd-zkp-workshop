# Path E — GPU-vs-CPU ZK primitive sweep (iGPU OpenCL)

_Source: `poc/amd-gpu-zk-primitive-demo/artefacts/gpu-primitive.csv` — rendered alongside PNG._

_iGPU = AMD Radeon 8060S (gfx1151) via ec-gpu OpenCL; CPU = 32-thread Zen 5 (same baseline as the Demo B STARK sweep). `speedup = cpu_ms / gpu_ms` (>1 = GPU wins). This accelerates SNARK primitives, NOT the r0vm STARK main line (which has no upstream AMD GPU prover; scoped fork = path-i)._

| primitive | log_size | size | gpu_ms | cpu_ms | speedup | gpu_device | cpu_threads |
| --- | --- | --- | --- | --- | --- | --- | --- |
| msm | 16 | 65536 | 74.669 | 60.03 | 0.804 | gfx1151 | 32 |
| msm | 18 | 262144 | 230.824 | 209.381 | 0.907 | gfx1151 | 32 |
| msm | 20 | 1048576 | 804.318 | 851.754 | 1.059 | gfx1151 | 32 |
| msm | 22 | 4194304 | 2641.489 | 3204.013 | 1.213 | gfx1151 | 32 |
| fft | 16 | 65536 | 1.453 | 6.899 | 4.749 | gfx1151 | 32 |
| fft | 18 | 262144 | 7.76 | 24.098 | 3.105 | gfx1151 | 32 |
| fft | 20 | 1048576 | 27.157 | 104.034 | 3.831 | gfx1151 | 32 |
| fft | 22 | 4194304 | 90.217 | 500.942 | 5.553 | gfx1151 | 32 |
| fft | 24 | 16777216 | 360.132 | 1847.881 | 5.131 | gfx1151 | 32 |
