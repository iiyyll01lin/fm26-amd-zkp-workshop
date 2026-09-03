# FUTUREMODE 2026 · Verifiable AI × Web3 on an AMD APU — attendee repo

> **35-minute hybrid workshop** · **35 分鐘 hybrid 工作坊**
> Attendee landing page — start here. · 與會者入口頁，從這裡開始。
> Runs live on one **AMD Ryzen AI MAX+ 395** (Strix Halo) APU — a **~300 W HP Z2 Mini G1a** small-form-factor desktop — and **replays on any laptop**, no GPU required.
> 現場在一顆 **AMD Ryzen AI MAX+ 395**（Strix Halo）APU 上跑——一台 **~300 W 的 HP Z2 Mini G1a** 小桌機——並可在**任何筆電上 replay**，不需要 GPU。

**This repo is a 3.5 MB slice of the full 19 GB course repo, trimmed to the six curated notebooks a CPU-only replay needs.** · 本 repo 是完整 19 GB 課程 repo 的 3.5 MB 精簡版，只留 CPU-only replay 需要的六本策展 notebook。 That is tracked content; `du` reports more, because a tree of small text files rounds up to whole filesystem blocks. · 那是 tracked 內容的口徑；`du` 會報更多，因為小文字檔會進位到整個 block。 See [`PROVENANCE.md`](PROVENANCE.md) — it records the export step, the trim, the `poc/` re-prune, the evidence restored afterwards, the exact byte count, and the `make verify` / `make replay` run that checked all of it.

> ℹ️ **Links to `zkp-final` need access — they are not broken.** · **指向 `zkp-final` 的連結需要授權，不是壞連結。**
> Links in this repo that point at `github.com/iiyyll01lin/zkp-final` go to the **full 25-notebook upstream repository**, which is not currently public: without repo access GitHub answers `404`. **That is expected behaviour, not a dead link.** · 本 repo 內指向 `github.com/iiyyll01lin/zkp-final` 的連結都是**完整 25 本的上游 repo**，該 repo 目前未公開；沒有存取權的話 GitHub 會回 `404`。**這是預期行為，不是連結壞了。**
> **Nothing the workshop needs sits behind those links.** All six curated notebooks are in this repo with their outputs already committed, the artefacts every replay cell reads are under `poc/`, and the links *between* the six notebooks are local relative paths — so `make verify` and `make replay` work with no network and no upstream access. The upstream links are there for the other 19 notebooks, the course site and the raw measurement logs. · **工作坊需要的東西一個都不在那些連結後面**：六本策展 notebook 全在這裡、output 已 commit，replay 讀的 artefact 都在 `poc/`，六本之間的相互連結是本地相對路徑——所以 `make verify` 與 `make replay` 不需要網路、也不需要上游存取權。上游連結是給其餘 19 本、course 網站與原始量測 log 用的。

---

## Start here · 60 秒開始

The clone URL is the QR on your handout card. · clone 網址就是手卡上的 QR。

```bash
git clone <URL from the handout QR>
cd zkp-workshop-futuremode-2026
make verify          # labkit imports; every artefact the six notebooks need is present
make replay          # the six curated notebooks, replay-only, non-destructive
```

On the **AUP Learning Cloud** seat, open a terminal in Jupyter (*File → New → Terminal*), run the `git clone` above, then open [`lab/01_zkml_embedding_ezkl.ipynb`](lab/01_zkml_embedding_ezkl.ipynb) from the file browser. · 在 AUPLC 座位上開 terminal 跑上面的 clone，再從檔案瀏覽器打開 nb01。

Dependencies (if your seat is bare): `make install` installs [`lab/requirements.txt`](lab/requirements.txt) — pandas, matplotlib, jupyter. Nothing AMD, nothing CUDA. · 依賴只有 pandas / matplotlib / jupyter。

Make the venv first. `make install` installs into whatever `PYTHON` points at, which defaults to the system `python3`, while [`make replay`](run-replay.sh) and `make lab` both prefer `.venv/bin/jupyter` whenever one exists — so create the venv, then install into it: · 建議先建 venv 再裝。`make install` 會裝進 `PYTHON` 指到的直譯器（預設是系統 `python3`），而 `make replay` 與 `make lab` 只要 `.venv/bin/jupyter` 存在就會優先用它：

```bash
python3 -m venv .venv
make install PYTHON=.venv/bin/python
```

---

## What this is · 這是什麼

**EN** — A hands-on cut of the *Verifiable AI on an AMD APU* course: how one APU runs a complete verifiable pipeline, then how honest bottleneck forensics turns a capability success into the next engineering target. You will watch six curated notebooks and run one yourself.

**ZH** — 這是 *Verifiable AI on an AMD APU* 課程的 hands-on 精簡版：一顆 APU 跑完整條可驗證 pipeline，再用誠實瓶頸鑑識把 capability success 轉成下一個工程目標。你會看六本策展 notebook，並親手跑其中一本。

- **Format · 形式**: presenter demo (replay, with an optional Strix Halo live cameo) + a 3-minute attendee hands-on.
- **Where you run · 你在哪裡跑**: your **AUP Learning Cloud** seat (a CPU-only replay image) or your own laptop — any Python 3 + browser.
- **Capacity · 容納**: ~30–50 concurrent CPU replay seats.

---

## The one trust hook — read this first · 先讀這條：誠實原則

This workshop is built on a repository whose numbers are committed, reproducible, and *deliberately conservative*. Its opening exhibit is a **correction of our own headline**:

本工作坊建立在一個「數字全部 committed、可重現、且刻意保守」的 repo 上。它的開場展品是一次**對我們自己 headline 的更正**：

> We published an iGPU proof-offload as **`1.34×`**. A contention-guarded **solo** re-measurement showed the clean CPU median was **66.5 s**, making it **`0.70×`** — and **that correction labelled its own replacement a lower bound**, because its GPU sample was `n=1`, contended, and never interleaved with the CPU arm. A properly **paired** re-test (one session, one binary, both arms check-off, arms interleaved, `n=3` each, median) then corrected it **back up**: CPU **60.847 s** vs iGPU **61.206 s** = **`0.994×` — statistical parity**, so the published `0.70×` was pessimistic by **1.42×**. 🔴 Parity is the *only* word for it: the 0.59% arm-to-arm gap is smaller than the 0.93% / 1.21% within-arm spreads, so it is **not** an iGPU win and **not** "0.6% slower" (mean-based `0.990×`, same verdict) — and **parity is not acceleration**. 🔴 Scope: only the **OpenCL G1-only** arm was re-benched; **`gpu-wide`'s `0.74×` was never re-measured and remains a floor**, and native-HIP is a separate pair (**G1-only `1.048×`**, **hip-wide `0.77×`, a floor**). Four arms, four sentences — never mixed, never averaged. We publish the correction — in **both** directions — not the hype.
>
> 我們曾把一個 iGPU proof-offload 公布為 **`1.34×`**。有 contention 防護的**單機 (solo)** 重測顯示乾淨 CPU 中位數是 **66.5 s**，於是變成 **`0.70×`**——而**那次更正當時就自標「這只是下界」**，因為 GPU 那臂 `n=1`、受競爭、且未與 CPU 臂交錯。之後一次真正**配對**的重測（同 session、同 binary、兩臂皆關 MSM check、arms 交錯、每臂 3 reps 取中位數）把它**往上**修回：CPU **60.847 s** vs iGPU **61.206 s** ＝ **`0.994×` ＝打平 (parity)**，原本的 `0.70×` 悲觀了 **1.42 倍**。🔴 唯一站得住的講法就是「打平」：臂間差距 0.59% 小於臂內離散 0.93% / 1.21%，所以**不是**「iGPU 勝」，也**不是**「慢 0.6%」（用平均 `0.990×`，結論相同）——而**「打平」不是「加速」**。🔴 範圍：只有 **OpenCL G1-only** 這一臂被重測；**`gpu-wide` 的 `0.74×` 未重測、仍只是地板**，native-HIP 是另外兩臂（**G1-only `1.048×`**、**hip-wide `0.77×` 地板**）。四臂四種寫法，不可混講也不可平均。我們公布更正——而且是**兩個方向**的更正——不是行銷話術。

Full case study · 完整案例：[`docs/INTEGRITY-REPORT.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/INTEGRITY-REPORT.md). Evidence index · 證據索引：[`docs/validation-ledger.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/validation-ledger.md).

**The honesty boundary we repeat all session · 全程重複的誠實邊界:**

- iGPU / NPU accelerate the **AI model** (embedding, LLM forward). · iGPU / NPU 加速 **AI 模型**（embedding、LLM 前向）。
- iGPU OpenCL accelerates **SNARK primitives / Groth16**, but **size-gated**. · iGPU OpenCL 加速 **SNARK 原語 / Groth16**，但 **size-gated**。
- The **stock RISC0 zkVM STARK is CPU-only on AMD** — in the main-line pipeline the iGPU never proves. **This session's climax turns that around (honestly scoped)**: a pinned **v2.3.2 fork** produces a **GPU-run rv32im *segment*-STARK seal the stock `cargo risczero verify` accepts**, **bit-for-bit == CpuHal** (DualHal 15/15) — a **hybrid** (witgen/accum stay CPU; recursion lift/join/succinct and Groth16 now on the iGPU), **5.46×** workload-specific; see [`nb23`](lab/23_risc0_rocm_stark.ipynb) + [path-i](reading-notes/path-i-risc0-rocm-stark.md). The **stock `r0vm` stays CPU-only**; never "the iGPU proves the whole zkVM." · stock RISC0 zkVM STARK 在 AMD 上是 CPU-only；本場高潮誠實 scoped 地翻轉它，但 stock `r0vm` 仍 CPU-only。
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

> These six are the only notebooks in this export. Per-notebook engine focus and the teaching route · 逐本引擎焦點與教學路線：[`lab/README.md`](lab/README.md) · [`lab/TEACHING-GUIDE.md`](lab/TEACHING-GUIDE.md). The other 19 notebooks (00–24 in full) live upstream at [`lab/`](https://github.com/iiyyll01lin/zkp-final/blob/main/lab). · 其餘 19 本在上游。

---

## The 3-minute hands-on · 3 分鐘 hands-on（13:00–16:00 of the runsheet）

Everyone reproduces `PROOF VERIFIED` themselves — a real zkML proof verified on a CPU-only replay, no GPU needed.

每個人親手重現 `PROOF VERIFIED`——一個真 zkML proof 在 CPU-only replay 上被驗證，不需 GPU。

1. **Open the cloud · 開啟雲端**: browse to the **AUP Learning Cloud** URL printed on your handout card and log in with the **workshop account** on the card. · 用手卡上的網址開 **AUP Learning Cloud**，以卡上的 **workshop 帳號**登入。
2. **Clone this repo · clone 本 repo**: open a terminal in Jupyter (*File → New → Terminal*) and run the `git clone` from the handout card's QR — the same command as [Start here](#start-here--60-秒開始) above. Then `cd` into it. If your seat already has the repo, `git pull` instead. · 在 Jupyter 開 terminal，跑手卡 QR 上的 `git clone`（同上方指令），再 `cd` 進去；座位若已有此 repo 就改 `git pull`。
3. **Open the notebook · 開 notebook**: [`lab/01_zkml_embedding_ezkl.ipynb`](lab/01_zkml_embedding_ezkl.ipynb).
4. **Run it · 跑起來**: run the first cell (`lk.capability_badge()`, a read-only AMD probe), then **Run All**. · 先跑第一個 cell，再 **Run All**。
5. **Watch for the verdict · 等 verdict**:

```text
[REPLAY]  Demo A · EZKL Halo2  ->  PROOF VERIFIED   (committed proof.json + vk.key)
```

6. **What you just did · 你剛剛做了什麼**: on a laptop with **no GPU and no network**, `labkit` read the committed `proof.json` + `vk.key`, re-checked their structure, and reported the same **`PROOF VERIFIED`** verdict the Strix Halo produces live. · 在一台**沒有 GPU、沒有網路**的筆電上，`labkit` 讀了 committed 的 `proof.json` + `vk.key`、重驗其結構，回報與 Strix Halo live 跑一致的 **`PROOF VERIFIED`**。

> **No wifi / clone failed? · 沒有 wifi / clone 失敗？** Ask for the USB offline kit — it carries this same repo. Then `make replay` works with no network at all. · 跟工作人員拿 USB 離線包，內容相同，`make replay` 完全離線可跑。

---

## Live vs replay — how it runs anywhere · live vs replay：為何到哪都能跑

Every heavy cell calls `labkit.detect()`. Detect real Strix Halo / ROCm → **live**; detect nothing (any laptop) → **replay** the committed artefacts. Priority: `LAB_FORCE_REPLAY` ＞ `LAB_RUN_HEAVY` + hardware precondition ＞ default replay.

每個重運算 cell 都呼叫 `labkit.detect()`。偵測到真 Strix Halo / ROCm → **live**；偵測不到（任何筆電）→ **replay** committed artefacts。

- **Attendees** always run **replay**. This export ships the committed artefacts, not the multi-hundred-MB live inputs, so a live path would simply fall back. · **與會者**永遠走 replay。
- **Presenter** demos in replay, with an **optional live cameo** on a Strix Halo box. · **簡報者**用 replay demo，並可選 live cameo。

Mechanism details · 機制細節：[`lab/README.md`](lab/README.md).

---

## What's in this repo · 本 repo 內容

| Path · 路徑 | Contents · 內容 |
|---|---|
| [`lab/`](lab/) | The six curated notebooks + [`labkit.py`](lab/labkit.py) (the hybrid live/replay toolkit, upstream's version unabridged — nothing pruned — with provenance printing added on top) + [`requirements.txt`](lab/requirements.txt). · 六本策展 notebook；`labkit.py` 為上游完整版、未精簡，另加出處列印。 |
| `poc/<demo>/artefacts/` | The committed CSV / JSON / proof evidence every replay cell reads. Path shape is identical to upstream, because `labkit` builds absolute paths from it. · 路徑形狀與上游完全相同。 |
| [`reading-notes/path-i-risc0-rocm-stark.md`](reading-notes/path-i-risc0-rocm-stark.md) | The Path I write-up nb23 renders inline. |
| [`LICENSE`](LICENSE) · [`LICENSE-docs`](LICENSE-docs) | Apache-2.0 for the code, CC-BY-4.0 for the prose. See [License · 授權](#license--授權) below. |
| [`Makefile`](Makefile) · [`run-replay.sh`](run-replay.sh) · [`verify-export.py`](verify-export.py) | Replay + self-check. The `Makefile` is also the repo-root marker `labkit` looks for — don't delete it. |
| [`PROVENANCE.md`](PROVENANCE.md) | Where this came from and what was left behind. |

> ℹ️ **`labkit.py` is unabridged, `poc/` is not — so some `labkit` constants point at files this repo does not ship.** · **`labkit.py` 未精簡、`poc/` 已精簡——所以有些 `labkit` 常數指向本包沒有的檔。**
> `poc/` carries the artefacts the six curated notebooks actually read; [`labkit.py`](lab/labkit.py) is copied from upstream **unpruned**, so it still exposes the loaders written for the other 19 notebooks. Of its 84 `Path` constants, **60 still resolve and 24 point at artefacts that were left upstream**. Calling a loader behind one of those 24 raises `FileNotFoundError` — **that is expected, not a broken install.** · `labkit.py` 仍帶著為其餘 19 本寫的 loader：84 個 `Path` 常數中 **60 個仍解析得到、24 個指向留在上游的 artefact**；呼叫那 24 個對應的 loader 會拿到 `FileNotFoundError`，**這是預期行為，不是安裝壞了**。
> `make verify` accounts for this: it asserts the constants the six notebooks can reach, the paths those notebooks hard-code, and the measurement files the prose cites by name but no notebook reads, and reports the pruned upstream constants as expected information rather than failing on them. A green `make verify` and "24 constants don't resolve" are both true at once. · `make verify` 已把這件事納入：它斷言六本可達的常數、notebook 寫死的路徑，以及正文以檔名引用但 notebook 不讀的量測檔，並把剪掉的上游常數列為預期資訊而非錯誤。所以 `make verify` 綠燈與「24 個常數指不到檔」同時成立。
> The full set of artefacts lives upstream at [`poc/`](https://github.com/iiyyll01lin/zkp-final/blob/main/poc); which constants dangle and why is spelled out in the labkit API reference in [`lab/README.md`](lab/README.md). · 完整 artefact 在上游 `poc/`；逐項說明見 `lab/README.md` 的 labkit API 一節。

> 🔴 **"Unabridged" is not the same as "byte-identical to upstream" — `labkit.py` has provenance printing added, and nothing else.** · 🔴 **「未精簡」不等於「與上游逐位元相同」——`labkit.py` 加了出處列印，而且僅此而已。**
> Nothing was pruned: every constant, loader and plotter is still there, all **84 `Path` constants** included. What was added is display: each CSV/JSON/markdown loader now prints `[REPLAY] source: <repo-relative path>` when it reads, and ten figures carry the same line as a caption, so a number and the artefact behind it stay together even when a PNG is dragged out of the notebook. Paths are rendered through the existing `_rel_to_repo()` helper, so a provenance string never carries an absolute path, a host name or an account name. · 沒有刪掉任何東西——常數、loader、plotter 全在，**84 個 `Path` 常數**一個不少。加的是顯示：每個 CSV/JSON/markdown loader 讀檔時會印 `[REPLAY] source: <repo 相對路徑>`，十張圖把同一行做成 caption，讓數字與其背後的 artefact 不會分家。路徑一律經過既有的 `_rel_to_repo()`，所以出處字串不會帶絕對路徑、主機名或帳號名。
> 🔴 **No calculation and no number changed, and that was proved mechanically rather than asserted.** Two checks: an **AST comparison** of the pre- and post-edit file found the multiset of arithmetic and comparison expressions (`BinOp` / `UnaryOp` / `Compare` / `AugAssign`) **identical across all 133 shared functions** and at module level; and **all 17 loaders and derived helpers the six notebooks call return data whose sha256 is identical** under both versions. Re-running the notebooks produced **zero numeric differences**. See [`PROVENANCE.md`](PROVENANCE.md) → *Provenance markers added to the notebook outputs*. · 🔴 **沒有改動任何計算、任何數值，而且這是機器證明的，不是聲稱的。** 兩道核對：改動前後檔案的 **AST 比對**顯示算術與比較運算式（`BinOp`／`UnaryOp`／`Compare`／`AugAssign`）的多重集合在**全部 133 個共用函式**與 module 層級**完全相同**；六本 notebook 會呼叫的 **17 個 loader 與衍生 helper，其回傳資料的 sha256 在兩個版本下完全一致**。重跑 notebook 的**數值差異為零**。詳見 `PROVENANCE.md`〈為 notebook output 加上出處標記〉。

---

## Take it home · 帶回家

This export is the replay kit, not the whole course — the take-home material lives upstream. · 本 repo 只是 replay 包；帶回家的材料在上游。

- **7 days**: the take-home 7-day path, upstream at [`study-plan/7-day-checklist.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/study-plan/7-day-checklist.md). · 7 天自學清單在上游。
- **Build something**: the hackathon starter kit — [`hackathon/futuremode-2026/CHALLENGE.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/hackathon/futuremode-2026/CHALLENGE.md), [`SUBMISSION.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/hackathon/futuremode-2026/SUBMISSION.md) and the fork-me template — upstream at [`hackathon/futuremode-2026/`](https://github.com/iiyyll01lin/zkp-final/tree/main/hackathon/futuremode-2026). · hackathon 起始包在上游。
- **The other 19 notebooks**: upstream [`lab/`](https://github.com/iiyyll01lin/zkp-final/blob/main/lab), with the five full teaching routes in [`lab/TEACHING-GUIDE.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/lab/TEACHING-GUIDE.md). · 其餘 19 本與五條完整教學路線在上游。
- **Go deeper**: the full course, the live AMD demos, and every raw measurement live upstream at [`https://github.com/iiyyll01lin/zkp-final`](https://github.com/iiyyll01lin/zkp-final).

---

## License · 授權

Dual-licensed, split by what the file *is*: code is Apache-2.0, prose is CC-BY-4.0. · 雙授權，依「檔案是什麼」切分：程式碼 Apache-2.0、文字 CC-BY-4.0。

| What · 什麼 | Files · 檔案 | Licence |
|---|---|---|
| **Code** · 程式碼 | [`lab/labkit.py`](lab/labkit.py), the code cells in `lab/*.ipynb`, the Rust / Python / HIP / shell sources under `poc/`, [`verify-export.py`](verify-export.py), [`run-replay.sh`](run-replay.sh), [`Makefile`](Makefile) | **Apache-2.0** — [`LICENSE`](LICENSE) |
| **Prose** · 文字 | this `README.md`, [`PROVENANCE.md`](PROVENANCE.md), [`lab/README.md`](lab/README.md), [`lab/TEACHING-GUIDE.md`](lab/TEACHING-GUIDE.md), the markdown cells in `lab/*.ipynb`, [`reading-notes/`](reading-notes/), and the measurement write-ups under `poc/**/artefacts/*.md` | **CC-BY-4.0** — [`LICENSE-docs`](LICENSE-docs) |
| **Measurement data** · 量測資料 | the committed `*.csv` / `*.json` / `*.info` / `*.log` / proof artefacts under `poc/**/artefacts/` | **CC-BY-4.0** — [`LICENSE-docs`](LICENSE-docs) |

Copyright 2026 Jason YY, Lin. Reuse either way needs attribution to the upstream repo — see [`PROVENANCE.md`](PROVENANCE.md) §"Rules for anything derived from this repo". · 兩種授權都要標明出處，規則見 `PROVENANCE.md`。

🔴 **Quoting a number? Bring its scope note.** Every headline figure in this repo is scoped to a specific arm, backend and repetition count — the scope is part of the claim, and it is stated inline wherever the figure appears (see "The one trust hook" above). CC-BY-4.0 lets you reuse the prose; it does **not** license you to restate a figure without the conditions it was measured under. Re-run the artefact instead of vendoring the number. · 引用數字請整組帶走它的範圍註記；CC-BY 授權你重用文字，不授權你把數字從量測條件裡拆出來。

### Third-party components · 第三方元件

These keep their **own** licence — Apache-2.0 above does **not** relicense them, and their headers must stay intact: · 下列檔案維持原授權，Apache-2.0 不覆蓋它們，其授權標頭請勿移除：

| Path · 路徑 | Origin · 來源 | Licence |
|---|---|---|
| `poc/folding-step-demo/artefacts/zkrag-bn254/Groth16Verifier.sol` | generated from the [snarkjs](https://github.com/iden3/snarkjs) `verifier_groth16.sol.ejs` template — Copyright 2021 0KIMS association | **GPL-3.0** (`SPDX-License-Identifier` in the file header) |

> That Solidity verifier is **committed evidence** (the on-chain verification artefact nb16 narrates), not a build input — nothing in this repo compiles or links it, so its copyleft does not reach the rest of the tree. · 這個 Solidity verifier 是 committed 證據、不是 build 輸入，repo 內沒有任何東西編譯或連結它。
>
> Where a `Cargo.toml` under `poc/` carries a `license` field it declares `MIT OR Apache-2.0` or `Apache-2.0`, consistent with the Apache-2.0 above; `poc/zkml-faithful-demo/zkrag/Cargo.toml` is a workspace manifest with no `license` field and its member crates are not part of this export. Their upstream dependencies — including `ec-gpu` / `bellperson` — are **not vendored here** and carry their own licences wherever you fetch them. · `poc/` 下有宣告 `license` 的 crate 皆為 `MIT OR Apache-2.0` 或 `Apache-2.0`；其上游依賴（含 `ec-gpu` / `bellperson`）**未 vendored 進本 repo**。
