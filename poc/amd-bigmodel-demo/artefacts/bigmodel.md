# Unified-memory flagship — >16GB LLM on the Strix Halo iGPU

_Source: `poc/amd-bigmodel-demo/artefacts/bigmodel.csv` — rendered alongside PNG._

_iGPU = AMD Radeon 8060S (gfx1151), llama.cpp built with HIP. The `full_igpu` peak VRAM **> 16GB** is the crux: a 16GB discrete card cannot fully hold this model. The `cap_16gb` row caps GPU offload to a ~16GB budget on the SAME box — a contrast GENEROUS to the discrete card (CPU spill here is the same LPDDR5X, not a PCIe spill / hard OOM). This accelerates the AI model (RAG generator), NOT the proof; RISC0 STARK stays CPU-only. iGPU VRAM is a 32GB carveout of the 94GB pool._

| condition | ngl | weights_gb | peak_vram_gb | peak_gtt_gb | peak_gpu_gb | prefill_tps | gen_tps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_igpu | 99 | 19.85 | 2.13 | 25.51 | 27.62 | 287.44 | 10.12 |
| cap_16gb | 32 | 19.85 | 2.33 | 15.32 | 17.59 | 259.15 | 6.3 |
| cpu | 0 | 19.85 | 2.36 | 5.71 | 8.04 | 242.91 | 4.7 |
