# FUTUREMODE 2026 · Verifiable AI × Web3 on an AMD APU — attendee repo

> **60-minute hybrid workshop** · **60 分鐘 hybrid 工作坊**
> Attendee landing page — start here. · 與會者入口頁，從這裡開始。
> Runs live on one **AMD Ryzen AI MAX+ 395** (Strix Halo) APU — a **~300 W HP Z2 Mini G1a** small-form-factor desktop — and **replays on any laptop**, no GPU required.
> 現場在一顆 **AMD Ryzen AI MAX+ 395**（Strix Halo）APU 上跑——一台 **~300 W 的 HP Z2 Mini G1a** 小桌機——並可在**任何筆電上 replay**，不需要 GPU。

**This repo is a slice of the full 24 course repo, trimmed to the six curated notebooks a CPU-only replay needs.** ·

> ℹ️ **Links to `zkp-final` need access — they are not broken.** · (not public available yet)
> Links in this repo that point at `github.com/iiyyll01lin/zkp-final` go to the **full 25-notebook upstream repository**, which is not currently public: without repo access GitHub answers `404`. **That is expected behaviour, not a dead link.** ·

---

## Start here

The QR on your handout card points at this repo; the URL is below if you would rather type it. · 手卡上的 QR 就是指到這個 repo；不想掃碼就直接打下面的網址。

```bash
git clone https://github.com/iiyyll01lin/fm26-amd-zkp-workshop.git
cd fm26-amd-zkp-workshop
make verify          # labkit imports; every artefact the six notebooks need is present
make replay          # the six curated notebooks, replay-only, non-destructive
```

**HTTPS on purpose** — it needs no key setup, so it works on a bare seat. If your GitHub account already has an SSH key, `git@github.com:iiyyll01lin/fm26-amd-zkp-workshop.git` clones the same repo. · **刻意用 HTTPS**——不需要設金鑰，空機座位也能跑；GitHub 帳號已設 SSH 金鑰的話，`git@github.com:iiyyll01lin/fm26-amd-zkp-workshop.git` clone 到的是同一個 repo。

On the **AMD AUP Learning Cloud** seat, open a terminal in Jupyter (*File → New → Terminal*), run the `git clone` above, then open [`lab/01_zkml_embedding_ezkl.ipynb`](lab/01_zkml_embedding_ezkl.ipynb) from the file browser. · 在 AUPLC 座位上開 terminal 跑上面的 clone，再從檔案瀏覽器打開 nb01。

Dependencies (if your seat is bare): `make install` installs [`lab/requirements.txt`](lab/requirements.txt) — pandas, matplotlib, jupyter. Nothing AMD, nothing CUDA. · 依賴只有 pandas / matplotlib / jupyter。

Make the venv first. `make install` installs into whatever `PYTHON` points at, which defaults to the system `python3`, while [`make replay`](run-replay.sh) and `make lab` both prefer `.venv/bin/jupyter` whenever one exists — so create the venv, then install into it: · 建議先建 venv 再裝。`make install` 會裝進 `PYTHON` 指到的直譯器（預設是系統 `python3`），而 `make replay` 與 `make lab` 只要 `.venv/bin/jupyter` 存在就會優先用它：

```bash
python3 -m venv .venv
make install PYTHON=.venv/bin/python
```

---

## What this is · 這是什麼

A hands-on cut of the *Verifiable AI on an AMD APU* course: how one APU runs a complete verifiable pipeline, then how honest bottleneck forensics turns a capability success into the next engineering target. You will watch six curated notebooks and run yourself.

- **Format · 形式**: presenter demo + attendee hands-on.
- **Where you run · 你在哪裡跑**: your **AMD AUP Learning Cloud** seat (a CPU-only replay image) or your own laptop — any Python 3 + browser.
- **Capacity · 容納**: ~30–50 concurrent CPU replay seats.

---

This workshop is built on a repository whose numbers are *deliberately conservative*. 


Full case study (目前尚未public) · 完整案例：[`docs/INTEGRITY-REPORT.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/INTEGRITY-REPORT.md). Evidence index · 證據索引：[`docs/validation-ledger.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/validation-ledger.md).

**The honesty boundary we repeat all session · 全程重複的誠實邊界:**

- iGPU / NPU accelerate the **AI model** (embedding, LLM forward). · iGPU / NPU 加速 **AI 模型**（embedding、LLM 前向）。
- iGPU OpenCL accelerates **SNARK primitives / Groth16**, but **size-gated**. · iGPU OpenCL 加速 **SNARK 原語 / Groth16**，但 **size-gated**。
- The **stock RISC0 zkVM STARK is CPU-only on AMD** — in the main-line pipeline the iGPU never proves. **This session's climax turns that around (honestly scoped)**: a pinned **v2.3.2 fork** produces a **GPU-run rv32im *segment*-STARK seal the stock `cargo risczero verify` accepts**, **bit-for-bit == CpuHal** (DualHal 15/15) — a **hybrid** (witgen/accum stay CPU; recursion lift/join/succinct and Groth16 now on the iGPU), **5.46×** workload-specific; see [`nb23`](lab/23_risc0_rocm_stark.ipynb) and the retained [`poc/risc0-rocm-prover` evidence index](poc/risc0-rocm-prover/README.md). The **stock `r0vm` stays CPU-only**; never "the iGPU proves the whole zkVM." · stock RISC0 zkVM STARK 在 AMD 上是 CPU-only；這次我的implemetation翻轉了它，但 stock `r0vm` 仍 CPU-only。
- In the capstone: **retrieval is proven, the LLM output is not.** · 在 capstone 裡：**檢索被證明，LLM 輸出沒有。**

---

## The curated 6-notebook path · 策展的 6-notebook 路線

Six notebooks, one AI × Web3 spine. Lab 24 adds the replayable lesson behind the result: Groth16 **0.973×** parity, measured to be an accelerator regression at this circuit size rather than witness-bound, and a **192-VGPR** `eval_check` negative result.

六本 notebook，一條 AI × Web3 主線；Lab 24 專教 0.973× parity 的真正成因（此電路規模下的 accelerator regression，非 witness-bound）、192-VGPR 負結果與量測方法。

| # | Notebook | Engine focus · 引擎焦點 | Why it's in the cut · 為何選它 | Mode |
|---|---|---|---|---|
| **1** | [`lab/00_amd_engine_map.ipynb`](lab/00_amd_engine_map.ipynb) | All engines (mental model) · 全引擎心智模型 | One APU, three engines, 94 GB unified memory; which AI/Web3 work maps where + the honesty rule. · 一機三引擎、94 GB unified memory。 | live detect (laptop-safe) / replay |
| **2** | [`lab/01_zkml_embedding_ezkl.ipynb`](lab/01_zkml_embedding_ezkl.ipynb) | CPU Halo2 + iGPU/NPU (AI fwd) | The **smallest verifiable AI**: EZKL Halo2 prove+verify → `PROOF VERIFIED`. **This is the hands-on.** · 最短的**可驗證 AI**。 | live / **replay** |
| **3** | [`lab/16_verifiable_rag_e2e.ipynb`](lab/16_verifiable_rag_e2e.ipynb) | All five engines (capstone) · 五引擎 capstone | The star: one query → iGPU embed · CPU STARK retrieval proof · BN254 Groth16 on-chain · iGPU LLM answer. · 旗艦 capstone。 | replay-only |
| **4** | [`lab/23_risc0_rocm_stark.ipynb`](lab/23_risc0_rocm_stark.ipynb) | iGPU hybrid STARK (RISC0→ROCm) · 前沿高潮 | **The frontier climax**: a GPU-produced rv32im *segment*-STARK seal the **stock** verifier accepts, **bit-for-bit == CpuHal** (DualHal 15/15); **5.46×** workload-specific; hybrid/scoped. · 前沿高潮。 | live (Strix Halo) / **replay** |
| **5** | [`lab/24_risc0_rocm_bottleneck_lab.ipynb`](lab/24_risc0_rocm_bottleneck_lab.ipynb) | bottleneck forensics · 瓶頸鑑識 | Complete Groth16 receipt **0.973×** + `eval_check` **192-VGPR** negative result → concrete GPU-witness roadmap. | replay-only |
| **6** | [`lab/14_unified_memory_bigmodel.ipynb`](lab/14_unified_memory_bigmodel.ipynb) | iGPU + 94 GB unified · unified memory | The David-vs-Goliath cameo: Qwen2.5-32B Q4_K_M (~20 GB) held **27.6 GB** GPU-resident. | replay-only |

> These six are the only notebooks in this export. Per-notebook engine focus: the table above. The other 19 notebooks and the full teaching routes live upstream at [`lab/`](https://github.com/iiyyll01lin/zkp-final/blob/main/lab). · (not public)
---

## hands-on

Everyone reproduces `PROOF VERIFIED` themselves — a real zkML proof verified on a CPU-only replay, no GPU needed.

每個人親手重現 `PROOF VERIFIED`——一個真 zkML proof 在 CPU-only replay 上被驗證，不需 GPU。

1. **Open the cloud · 開啟雲端**: browse to the **AMD AUP Learning Cloud** URL printed on your handout card and log in with the **workshop account** on the card. · 用手卡上的網址開 **AMD AUP Learning Cloud**，以卡上的 **workshop 帳號**登入。
2. **Clone this repo · clone 本 repo**: open a terminal in Jupyter (*File → New → Terminal*) and run the `git clone` from [Start here](#start-here--60-秒開始) above — the handout card's QR is that same URL. Then `cd fm26-amd-zkp-workshop`. If your seat already has the repo, `git pull` instead. · 在 Jupyter 開 terminal，跑上方〈Start here〉的 `git clone`（手卡 QR 就是同一個網址），再 `cd fm26-amd-zkp-workshop`；座位若已有此 repo 就改 `git pull`。
3. **Open the notebook · 開 notebook**: [`lab/01_zkml_embedding_ezkl.ipynb`](lab/01_zkml_embedding_ezkl.ipynb).
4. **Run it · 跑起來**: run the first cell (`lk.capability_badge()`, a read-only AMD probe), then **Run All**. · 先跑第一個 cell，再 **Run All**。
5. **Watch for the verdict · 等 verdict**:

```text
[REPLAY]  Demo A · EZKL Halo2  ->  PROOF VERIFIED   (committed proof.json + vk.key)
```

6. **What you just did · 你剛剛做了什麼**: on a laptop with **no GPU and no network**, `labkit` read the committed `proof.json` + `vk.key`, re-checked their structure, and reported the same **`PROOF VERIFIED`** verdict the Strix Halo produces live. · 在一台**沒有 GPU、沒有網路**的筆電上，`labkit` 讀了 committed 的 `proof.json` + `vk.key`、重驗其結構，回報與 Strix Halo live 跑一致的 **`PROOF VERIFIED`**。



---

## Live vs replay
Every heavy cell calls `labkit.detect()`. Detect real Strix Halo / ROCm → **live**; detect nothing (any laptop) → **replay** the committed artefacts. Priority: `LAB_FORCE_REPLAY` ＞ `LAB_RUN_HEAVY` + hardware precondition ＞ default replay.

每個重運算 cell 都呼叫 `labkit.detect()`。偵測到真 Strix Halo / ROCm → **live**；偵測不到（任何筆電）→ **replay** committed artefacts。

- **Attendees** always run **replay**. This export ships the committed artefacts, not the multi-hundred-MB live inputs, so a live path would simply fall back.  

Mechanism implementation · 機制實作：[`lab/labkit.py`](lab/labkit.py).

---

## License · 授權

Dual-licensed, split by what the file *is*: code is Apache-2.0, prose is CC-BY-4.0. · 雙授權，依「檔案是什麼」切分：程式碼 Apache-2.0、文字 CC-BY-4.0。

| What · 什麼 | Files · 檔案 | Licence |
|---|---|---|
| **Code** · 程式碼 | [`lab/labkit.py`](lab/labkit.py), the code cells in `lab/*.ipynb`, the Rust / Python / HIP / shell sources under `poc/`, [`verify-export.py`](verify-export.py), [`run-replay.sh`](run-replay.sh), [`Makefile`](Makefile) | **Apache-2.0** — [`LICENSE`](LICENSE) |
| **Prose** · 文字 | this `README.md`, the markdown cells in `lab/*.ipynb`, and the measurement write-ups under `poc/**/artefacts/*.md` | **CC-BY-4.0** — [`LICENSE-docs`](LICENSE-docs) |
| **Measurement data** · 量測資料 | the committed `*.csv` / `*.json` / `*.info` / `*.log` / proof artefacts under `poc/**/artefacts/` | **CC-BY-4.0** — [`LICENSE-docs`](LICENSE-docs) |

Copyright 2026 Jason YY, Lin. Reuse either way needs attribution to the [upstream repository](https://github.com/iiyyll01lin/zkp-final), and every measurement must stay with its stated scope. · 兩種授權都要標明上游出處，量測數字必須與其範圍註記一起保留。



### Third-party components · 第三方元件

These keep their **own** licence — Apache-2.0 above does **not** relicense them, and their headers must stay intact: · 下列檔案維持原授權，Apache-2.0 不覆蓋它們，其授權標頭請勿移除：

| Path · 路徑 | Origin · 來源 | Licence |
|---|---|---|
| `poc/folding-step-demo/artefacts/zkrag-bn254/Groth16Verifier.sol` | generated from the [snarkjs](https://github.com/iden3/snarkjs) `verifier_groth16.sol.ejs` template — Copyright 2021 0KIMS association | **GPL-3.0** (`SPDX-License-Identifier` in the file header) |

> That Solidity verifier is **committed evidence** (the on-chain verification artefact nb16 narrates), not a build input — nothing in this repo compiles or links it, so its copyleft does not reach the rest of the tree. · 這個 Solidity verifier 是 committed 證據、不是 build 輸入，repo 內沒有任何東西編譯或連結它。
>
> Where a `Cargo.toml` under `poc/` carries a `license` field it declares `MIT OR Apache-2.0` or `Apache-2.0`, consistent with the Apache-2.0 above; `poc/zkml-faithful-demo/zkrag/Cargo.toml` is a workspace manifest with no `license` field and its member crates are not part of this export. Their upstream dependencies — including `ec-gpu` / `bellperson` — are **not vendored here** and carry their own licences wherever you fetch them. · `poc/` 下有宣告 `license` 的 crate 皆為 `MIT OR Apache-2.0` 或 `Apache-2.0`；其上游依賴（含 `ec-gpu` / `bellperson`）**未 vendored 進本 repo**。
