# Groth16 container comparison (standalone, one archived seal)

Image digest: `sha256:91f515e2995be1cb61b4ae458fa6f998c724db3324af1813e1acaff54466e4ef`
Seal SHA-256: `b5da3c426f08a628d5d19c56eb405bc4858a22df052bf9d74424c37c14bd4a92`

| config | accelerator | witness | witness median (s) | prove median (s) | total median (s) |
|---|---|---|---:|---:|---:|
| cpu-witness | cpu | generate | 1.702105 | 5.705002 | 5.716154 |
| rocm-replay | rocm | replay | 0.860566 | 10.082724 | 10.09272 |
| rocm-witness | rocm | generate | 1.709065 | 10.991388 | 11.003757 |

## Accelerator

ROCm container total 11.003757 s vs CPU 5.716154 s (ROCm - CPU = 5.287603 s, CPU/ROCm = 0.519473x).

## Witness critical path (measured, not inferred)

Replaying a pre-generated witness through the same FIFO removes 0.911037 s (11.003757 s -> 10.09272 s). This is the ceiling a perfect witness backend could return.
