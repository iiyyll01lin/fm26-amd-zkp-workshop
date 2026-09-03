# Groth16 container comparison (standalone, one archived seal)

Image digest: `sha256:91f515e2995be1cb61b4ae458fa6f998c724db3324af1813e1acaff54466e4ef`
Seal SHA-256: `b5da3c426f08a628d5d19c56eb405bc4858a22df052bf9d74424c37c14bd4a92`

| config | accelerator | witness | witness median (s) | prove median (s) | total median (s) |
|---|---|---|---:|---:|---:|
| cpu-witness | cpu | generate | 1.70774 | 5.78141 | 5.790889 |
| rocm-replay | rocm | replay | 0.869607 | 11.204706 | 11.216145 |
| rocm-witness | rocm | generate | 1.716074 | 12.006302 | 12.016017 |

## Accelerator

ROCm container total 12.016017 s vs CPU 5.790889 s (ROCm - CPU = 6.225128 s, CPU/ROCm = 0.481931x).

## Witness critical path (measured, not inferred)

Replaying a pre-generated witness through the same FIFO removes 0.799872 s (12.016017 s -> 11.216145 s). This is the ceiling a perfect witness backend could return.
