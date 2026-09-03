# Groth16 container comparison (standalone, one archived seal)

Image digest: `sha256:0c39c4c8fff2503f778b3d5bb9669a4928e479b596a3d2862585303471f19b51`
Seal SHA-256: `b5da3c426f08a628d5d19c56eb405bc4858a22df052bf9d74424c37c14bd4a92`

| config | accelerator | witness | witness median (s) | prove median (s) | total median (s) |
|---|---|---|---:|---:|---:|
| cpu-witness | cpu | generate | 1.698304 | 5.75267 | 5.763942 |
| rocm-replay | rocm | replay | 0.904631 | 10.327679 | 10.33984 |
| rocm-witness | rocm | generate | 1.693262 | 11.091787 | 11.102247 |

## Accelerator

ROCm container total 11.102247 s vs CPU 5.763942 s (ROCm - CPU = 5.338305 s, CPU/ROCm = 0.519169x).

## Witness critical path (measured, not inferred)

Replaying a pre-generated witness through the same FIFO removes 0.762407 s (11.102247 s -> 10.33984 s). This is the ceiling a perfect witness backend could return.
