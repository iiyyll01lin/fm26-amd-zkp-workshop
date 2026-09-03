# `ai-inference.csv` — provenance · 出處說明

> **Read this before you try to verify `ai-inference.csv` against
> `ai-inference.log`. They will not agree, and that is not a defect.**
> · **拿 `.log` 驗 `.csv` 之前先讀這頁**：兩者對不上是刻意的，不是壞掉。

`ai-inference.csv` is a **deliberate splice of two runs**. Neither file may be
"repaired" into agreement — doing so would either re-import a contended CPU
baseline or destroy a real measurement log.

`ai-inference.csv` 是**兩次 run 的刻意拼接**。兩個檔都不可以為了「對得上」而修改。

---

## Which rows came from where · 各列的來源

The `loadavg` column is the tell — it is in the CSV itself:

| rows · 列 | `loadavg` | source · 來源 |
|---|---|---|
| `cpu,*` and `rocm,*` (18 rows) | `1.59` | `ai-inference.solo-rebench-2026-06-18.{csv,log}` |
| `rocm-fp16,*` `rocm-int8,*` `rocm-fp8,*` (27 rows) | `2.85` | `ai-inference.log` (the file sitting next to this one) |

The 18 clean-solo rows are **byte-identical** to
`ai-inference.solo-rebench-2026-06-18.csv`; diff the two files and see.

那 18 列與 solo 重測 CSV **逐位元相同**，可直接 diff 驗證。

---

## The specific disagreement · 具體對不上的地方

At the `b1·s256` point, the two runs measured different things:

| | current `ai-inference.log` | `solo-rebench` log | `ai-inference.csv` |
|---|---|---|---|
| `cpu` best | `9.97 ms` (median 14.06, stdev 3.45) | `6.93 ms` (median 7.22, stdev 1.08) | **`6.932`** |
| `rocm` best | `1.54 ms` | `1.75 ms` | **`1.753`** |

**The CSV takes both from the solo re-bench, so the pair is measured under one
set of conditions** — that is the whole point. The contended run's
**stdev is 3.2× larger and its median nearly 2× its own best**, which is what a
CPU baseline looks like when something else is on the machine.

CSV 的 cpu 與 rocm **都取自同一次 solo 重測**，所以那個比值是同條件下的配對。
被競爭污染那次的 stdev 大 3.2 倍、median 幾乎是自己 best 的兩倍。

---

## Why it matters · 為什麼這件事要緊

Quoting the contended CPU number would inflate every iGPU speedup derived from
it. This repo has caught that same failure four times (`1.34×`, `1.1–1.35×`,
`0.86×`, `15.64×`) and the fingerprint is always the same: **a CPU baseline
measured under contention makes the GPU look better than it is.** The
recalibration is recorded in `4f8b139` (2026-06-18,
`9-25x -> ~1.9-10.6x`) and `c84e772` (2026-06-22).

用被污染的 CPU 數字會讓所有由它推導的 iGPU 加速比同步膨脹——這個 repo 已經抓到
四次同型錯誤，指紋都一樣：**CPU 基線被污染，GPU 就看起來比實際好。**

---

## Rules · 規則

1. **Do not edit either file to make them match.** The CSV is correct; the log
   is a truthful record of a different, contended run.
   · **不要為了對齊而改任何一個檔。**
2. **Do not quote `ai-inference.log` as the source for a `cpu` or `rocm` row.**
   Cite `ai-inference.solo-rebench-2026-06-18.log` instead.
   · 引用 `cpu`/`rocm` 列時，出處要寫 solo 重測那份 log。
3. **Anything derived from these rows must be regenerated, not hand-edited.**
   `three-engine.json` restates `6.932` / `1.753` and divides them; it froze at
   `15.64×` for two months because nothing checked that quotient. It is now
   pinned by the `M7-three-engine-derived` assertion in
   `scripts/course-drift-check.py` — re-run `make showcase` after any change
   here, then `python scripts/course-drift-check.py`.
   · 由這些列推導的數字一律**重跑產生**、不要手改。

## Verify · 驗證

```bash
diff <(head -19 ai-inference.csv) ai-inference.solo-rebench-2026-06-18.csv
awk -F, 'NR>1 {print $1, $NF}' ai-inference.csv | sort -u   # the loadavg split
```
