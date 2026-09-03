# Groth16 ROCm benchmark summary

- image id: `3aec8c717d9b47f9b617c1695955d75e2bb525085283c1076591f97ae643c990`
- **project-owner authorized for workshop/labs (2026-07-24)**
- same-box same-workload wall-clock; workload-specific, not a cross-vendor claim

| mode | reps | min (s) | median (s) | max (s) |
|---|---|---|---|---|
| cpu-shipped | 3 | 181.90 | 182.37 | 182.42 |
| gpu-rocm | 3 | 186.66 | 187.44 | 187.53 |

| comparison | median speedup |
|---|---|
| gpu-rocm vs cpu-shipped (shipped upstream gnark CPU path) | 0.973x |

