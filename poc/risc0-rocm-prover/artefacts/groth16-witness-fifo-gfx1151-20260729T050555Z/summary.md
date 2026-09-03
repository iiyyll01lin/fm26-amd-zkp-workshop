# Groth16 witness compiler baseline

Witness transport: fifo
Measured opt levels: O0
Journal SHA-256 (identical across reps): `c07fc6646401c025e2a276df15efabacf7d5cf48cbcb414e4107f5bf8a5cb127`

Production FIFO transport: witness and prover overlap through the pipe, so their
times include mutual backpressure and MUST NOT be summed; only the total is
additive. A FIFO retains no payload to hash.

| opt | witness median (s) | prove median (s) | total median (s) |
|---|---:|---:|---:|
| O0 | 1.718337 | 11.122012 | 11.130289 |
