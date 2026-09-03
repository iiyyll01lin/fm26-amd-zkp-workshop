# CPU STARK throughput sweep

_Source: `poc/risc0-cartesi-step-demo/artefacts/bench/throughput.csv`._

_r0vm proves on the CPU only (no ROCm/Vulkan/NPU prover); this is pure CPU thread-scaling + RAM-headroom data. `proof_bytes=0` marks a cell that produced no seal (e.g. OOM)._

## Report takeaways

- 15 complete full-STARK cells were captured on AMD Ryzen AI MAX+ PRO 395 / 94 GB unified RAM.
- For `max_mcycle=1`, 1 -> 64 Rayon threads improves wall time from 2413.448 s to 122.136 s (~19.8x), with diminishing returns after 32 threads.
- For `max_mcycle=10`, 1 -> 32 threads improves wall time from 2273.893 s to 96.590 s (~23.5x); 64 threads is effectively tied at 96.792 s.
- Peak RSS stays near 9.59 GB across the sweep, so this benchmark is thread-scaling data, not GPU/NPU acceleration data.

| max_mcycle | rayon_threads | wall_seconds | peak_rss_kb | proof_bytes | mode | timestamp |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 2413.448 | 9589300 | 562812 | full | 2026-06-09T06:19:09Z |
| 1 | 2 | 1422.772 | 9588104 | 562812 | full | 2026-06-09T06:59:23Z |
| 1 | 4 | 643.508 | 9588108 | 562812 | full | 2026-06-09T07:23:06Z |
| 1 | 8 | 335.696 | 9587180 | 562812 | full | 2026-06-09T07:33:49Z |
| 1 | 16 | 191.062 | 9587260 | 562812 | full | 2026-06-09T07:39:25Z |
| 1 | 32 | 127.664 | 9589612 | 562812 | full | 2026-06-09T07:42:36Z |
| 1 | 64 | 122.136 | 9594744 | 562812 | full | 2026-06-09T07:44:44Z |
| 10 | 1 | 2273.893 | 9588736 | 562812 | full | 2026-06-09T07:46:49Z |
| 10 | 2 | 1060.697 | 9588776 | 562812 | full | 2026-06-09T08:24:43Z |
| 10 | 4 | 546.722 | 9587696 | 562812 | full | 2026-06-09T08:42:23Z |
| 10 | 8 | 292.874 | 9588796 | 562812 | full | 2026-06-09T08:51:30Z |
| 10 | 16 | 153.336 | 9588808 | 562812 | full | 2026-06-09T08:56:23Z |
| 10 | 32 | 96.590 | 9588424 | 562812 | full | 2026-06-09T08:58:56Z |
| 10 | 64 | 96.792 | 9594388 | 562812 | full | 2026-06-09T09:00:33Z |
| 100 | 1 | 2368.956 | 9593128 | 806878 | full | 2026-06-09T09:02:10Z |
