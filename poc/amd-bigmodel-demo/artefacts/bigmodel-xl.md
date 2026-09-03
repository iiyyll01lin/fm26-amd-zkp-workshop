# Unified-memory flagship — >32GB LLM on the Strix Halo iGPU

_Source: `poc/amd-bigmodel-demo/artefacts/bigmodel.csv` — rendered alongside PNG._

_iGPU = AMD Radeon 8060S (gfx1151), llama.cpp built with HIP. The `full_igpu` peak VRAM **> 32GB** is the crux: a 32GB discrete card cannot fully hold this model. The `cap_32gb` row caps GPU offload to a ~32GB budget on the SAME box — a contrast GENEROUS to the discrete card (CPU spill here is the same LPDDR5X, not a PCIe spill / hard OOM). This accelerates the AI model (RAG generator), NOT the proof; RISC0 STARK stays CPU-only. iGPU VRAM is a 32GB carveout of the 94GB pool._

| condition | ngl | weights_gb | peak_vram_gb | peak_gtt_gb | peak_gpu_gb | prefill_tps | gen_tps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_igpu | 99 | 42.52 | 0.39 | 44.39 | 44.77 | 124.64 | 4.86 |
| cap_32gb | 55 | 42.52 | 0.32 | 30.54 | 30.87 | 118.15 | 3.76 |
| cpu | 0 | 42.52 | 0.31 | 2.09 | 2.4 | 111.61 | 2.55 |
