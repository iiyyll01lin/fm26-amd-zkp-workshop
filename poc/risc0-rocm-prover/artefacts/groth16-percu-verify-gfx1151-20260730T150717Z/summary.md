# ROCm MSM scheduling sweep (gfx1151, one archived seal)

Image digest: `sha256:0c39c4c8fff2503f778b3d5bb9669a4928e479b596a3d2862585303471f19b51`
Seal SHA-256: `b5da3c426f08a628d5d19c56eb405bc4858a22df052bf9d74424c37c14bd4a92`

| window_max | work_units | reps | total median (s) | gnark prove median (ms) |
|---:|---:|---:|---:|---:|
| 8 | 10240 | 3 | 7.477132 | 5228.268 |
| 9 | 10240 | 3 | 7.529284 | 5277.561 |
| 8 | 15360 | 3 | 7.671331 | 5421.476 |

## Verdict

Best tuned ROCm: **7.477132 s** (window_max=8, work_units=10240).
Untuned ROCm default: 11.102247 s (-3.625 s).
CPU gnark baseline: 5.763942 s (+1.713 s).

**Tuning overturns the CPU win: NO.**
