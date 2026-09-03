# zkLLM on AMD Strix Halo — one attention block, three engines

_Source: `split.json` — rendered alongside PNG._

| stage | engine | measure | verdict |
| --- | --- | --- | --- |
| 1 · forward | iGPU gfx1151 (MIGraphX/ROCm) vs Zen5 CPU (onnxruntime) | cpu 0.012 ms · igpu 0.129 ms (speedup 0.093) | iGPU forward only (size-gated) |
| 2 · prove | Zen5 CPU (16c/32t) + 94 GB unified memory — EZKL Halo2 (KZG/BN254) | prove 7.852 s · proof 545679 B · PROOF VERIFIED | CPU-only proof |
| 3 · MSM frontier | gfx1151 OpenCL | BN254 MSM 0.654-1.091x | offloadable; EZKL wires CUDA/Metal only (EZKL-GPU-BLOCKED-ON-AMD) |

iGPU accelerates the model FORWARD only (size-gated; a single attention sub-block at seq=8 is dispatch-bound so the CPU wins). EZKL Halo2 proving is CPU-only on AMD (win = 32 threads + 94 GB unified memory). The KZG MSM could run on the iGPU, but not faster at these sizes (Path E BN254 G1 MSM clean-solo 0.654-1.091x, below parity until ~2^22; the old 1.1-1.35x was contention-inflated and is retired), and EZKL wires only CUDA/Metal -> documented frontier, NOT a GPU-proving claim. Weights are seeded; swapping trained HF tensors is a one-liner that does not change the circuit.
