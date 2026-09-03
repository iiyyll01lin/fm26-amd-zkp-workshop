# Groth16 container comparison (standalone, one archived seal)

Image digest: `sha256:91f515e2995be1cb61b4ae458fa6f998c724db3324af1813e1acaff54466e4ef`
Seal SHA-256: `b5da3c426f08a628d5d19c56eb405bc4858a22df052bf9d74424c37c14bd4a92`

| config | accelerator | witness | witness median (s) | prove median (s) | total median (s) |
|---|---|---|---:|---:|---:|
| cpu-witness | cpu | generate | 1.6944 | 5.714021 | 5.724339 |
| rocm-replay | rocm | replay | 0.869708 | 6.874092 | 6.885833 |
| rocm-witness | rocm | generate | 1.719739 | 7.712533 | 7.723975 |

## Accelerator

ROCm container total 7.723975 s vs CPU 5.724339 s (ROCm - CPU = 1.999636 s, CPU/ROCm = 0.741113x).

## Witness critical path (measured, not inferred)

Replaying a pre-generated witness through the same FIFO removes 0.838142 s (7.723975 s -> 6.885833 s). This is the ceiling a perfect witness backend could return.
