# Demo B — Cartesi Machine Step Proof via RISC0 zkVM

> 對應簡報 §「PoC Demo B」與 Cartesi v0.20.0 章節。
> 一句話：用 v0.20.0 釋出的 `cartesi-risc0-guest-step-prover.bin` 對齊 release shape，並在同一個 container 內以 `cargo risczero new` 跑一個**真實** RISC0 hello-world prove + verify，作為「toolchain 與真實密碼學可運作」的 demo 端到端證據。

---

## Validation status

| 項目 | 結果 |
|------|------|
| `docker build` 從零跑通 | ✅ ~700 s 估算（rustup ~50 s + `cargo install cargo-risczero` ~300 s + `rzup install rust` ~70 s + apt RV64 toolchain (gcc-riscv64-linux-gnu, +~360 MB) ~30 s + pre-warm hello-world build ~135 s + 其餘 ~150 s）|
| `docker run … run-all.sh --mock-mode` 全 0 退出 | ✅ exit 0 in **~55 s** (cached cargo pre-warm; host: 15 GB RAM / 11 GB free, x86_64) |
| `docker run … run-all.sh --dev-mode` 全 0 退出 | ✅ exit 0 in **~69 s** (peak RSS ~1.14 GiB; host: same) — **Tier 2 PASS** |
| Stage 00 real RISC0 receipt verify | ✅ `risc0-hello-world.log` 末尾印 `[00] receipt verified successfully against MULTIPLY_ID (real STARK seal)` |
| Stage 01 official prover binary fetched | ✅ `cartesi-risc0-guest-step-prover.bin` 888 764 bytes（≈ 868 KB，符合 release notes） |
| Stage 02 real cartesi-machine snapshot (dev-mode) | ✅ `machine-snapshot/` 1.5 MB (sha256 hash tree, phtc_size=64)；mock-mode 走 stub |
| Stage 03 real `--log-step` step log (dev-mode) | ✅ `step.bin` 18 944 bytes，header 解析得 mcycle=1、pre/post roots |
| Stage 04 real r0vm dev prove (dev-mode) | ✅ `step.proof.bin` 393 bytes（dev receipt，非真 STARK seal）+ `WARNING: proving in dev mode` 提示 |
| Stage 05 real `cargo risczero verify` (dev-mode) | ✅ 印 `✅ Receipt is valid!` + `[DEV-REAL] step.proof.bin verified: pre_root↔post_root match, mcycle=1, dev-receipt=true` |
| Stages 03/04/05 mock artefacts (mock-mode) | ✅ 5 dense uarch hashes + 264-byte `MOCKPRF`-magic proof + `pre_root↔post_root` verify line（regression 仍綠）|
| Artefacts committed | ✅ `poc/risc0-cartesi-step-demo/artefacts/`；Tier 2 當時是 dev-mode 的 11 個檔（~240 KB），**目前 commit 的是 `--full-rootfs` 真值——光 `step.proof.bin` 就 1 112 064 B**。⚠️ `machine-snapshot/` **沒有**進 git（`poc/.gitignore:13` 排除），要自己跑 `--dev-mode` / `--full-rootfs` 才會在本機生出來 |

完整 artefact list 與 size：見 [§ Committed artefacts](#6-committed-artefacts)。

### Full-mode 實測（AMD Strix Halo, 2026-06-09）

真 `--full` 路徑已在 128 GB Strix Halo（Ryzen AI MAX+ 395, 94 GB, kernel 6.17, `RAYON_NUM_THREADS=32`）端到端跑通並 commit。逐項見 [`docs/IMPLEMENTATION-STATUS.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/IMPLEMENTATION-STATUS.md) 與 `artefacts/full-run.info`：

| 項目 | 結果 |
|------|------|
| `make demo-b-full`（真 STARK seal） | ✅ PASS（**163 s**，`artefacts/full-run.info:15`）；mcycle=1 STARK seal 562,812 B（該 run 當下的 `step.proof.bin`）；bench 尾端已推進到 mcycle=1000 seal 1,393,314 B（806,878 B 現在只對應 mcycle=100 的 1-thread 與 2-thread 兩格，見 `artefacts/bench/throughput.csv`） |
| `make demo-b-groth16`（06 wrap + 07 鏈上） | ✅ PASS（**159 s**，`artefacts/full-run.info:17`）；`step.groth16.bin` 260 B；`journal_digest_matches_step_verifier=true`；anvil 上 `verifyStep(seal, pre, mcycle, post) -> true` |
| `make bench`（CPU STARK 吞吐掃描） | ✅ 28 cells captured（`artefacts/full-run.info:34` `bench.throughput_csv_rows=28`）；mcycle=1/10/100/1000 × 1/2/4/8/16/32/64 threads 全滿；mcycle=1/64 threads 約 122 s，mcycle=10/32 threads 約 96.6 s；peak RSS ~9.59 GB → `artefacts/bench/throughput.{csv,md,png}` |
| `make npu-probe` | ✅ XDNA2 DRIVER-READY（研究用；ZK 證明在 AMD 上純 CPU） |

> ⚠️ **`artefacts/step.proof.bin` 現在裝的是什麼**：`c077201`（2026-06-10）的 `--full-rootfs` run 把它覆寫成 **1,112,064 B 的真 r0vm STARK seal**（`step.public.json`：`mode=full-rootfs`、`mcycle=100`、`real_stark=true`、`dev_receipt=false`；stage 05 `cargo risczero verify` → `Receipt is valid!`）。上表的 **562,812 B 是 mcycle=1 的 canonical seal**——被 Groth16 包裝且鏈上驗過的就是它——但它現在只存在於 git 歷史（`b08ee43`）與 `bench/throughput.csv` 的 mcycle=1 七格裡，**不在工作目錄的 `step.proof.bin`**。要講「鏈上 verifyStep=true」以 `step.groth16.json` 為準。

> ⚠️ 鏈上 `verifyStep` 能通過，是因為 2026-06-08 修正了 `StepVerifier.sol` 的 journal 排版（補上 `mcycle`，見 §7）。Path C folding 的鏈上 replay 仍受上游 Sonobe bug 阻擋（見 [`../folding-step-demo/README.md`](../folding-step-demo/README.md)）。

---

## 0. 為什麼是這個 demo

- **時機性**：Cartesi v0.20.0（2026-04-09）首次原生支援 RISC0 zkVM。這份 demo 直接對齊該 release。
- **延續性**：扣回 DEAAP 多 BU 共識 + Cartesi rollup 的既有架構，把「下一步要往 zkRollup 走」具象化。
- **可錄影性**：hybrid mode 全程 ≤ 90 秒且不需要重型硬體（11 GB free RAM 就夠），真實 prove+verify 出現在 stage 00。
- **誠實性**：mock vs real 邊界在腳本 banner、JSON 內 `"mock": true`、`MOCKPRF` 檔頭、README 都標清楚，slide 講稿也會明確說。

---

## 1. Prerequisites

| Tool | Version | Mode 需求 |
|------|---------|----------|
| Docker | ≥ 24 | 兩種模式皆建議 |
| Disk | ~10 GB | image 約 8 GB、artefacts < 1 MB |
| RAM | 4 GB (mock) / ~1.14 GiB (dev) / **≥32 GB (full 真 STARK)** | `--full` 在 ≥32 GB 主機（如 128 GB Strix Halo）跑；過去 11 GB free 的錄影機會撞上 ≥16 GB OOM 牆 |
| Host arch | x86_64 或 aarch64 | release deb 同時提供 |
| 網路 | 第一次需要 GitHub releases / RISC0 artifact CDN | image 跑過一次後可離線重 run |

---

## 2. 快速開始（Hybrid mode；錄影預設）

```bash
cd poc/risc0-cartesi-step-demo

docker build -t risc0-cartesi-demo:local .

mkdir -p artefacts dist
docker run --rm \
    -v "$(pwd)/artefacts:/work/artefacts" \
    -v "$(pwd)/dist:/work/dist" \
    risc0-cartesi-demo:local \
    bash scripts/run-all.sh --mock-mode
```

預期最後一行：

```
[MOCK] step.proof.bin verified: pre_root↔post_root match, mcycle_count=100
```

而 stage 00 結束時則會印：

```
[00] receipt verified successfully against MULTIPLY_ID (real STARK seal)
```

第一行是 mock-pipeline 的 cosmetic 驗證，第二行則是來自**真實**的 RISC0 STARK 驗證器。兩者一起出現就算 demo 通過。

`docker-compose.yml` 可用 `docker compose run --rm step-demo` 等價呼叫。

---

## 3. 三模式對照表

這份 PoC 的**核心承諾**：清楚標示哪一段是真密碼學、哪一段只是 pipeline shape。Tier 2 後新增 `--dev-mode`，把 02–05 從 pure mock 升級到「真 Cartesi 機器 + RISC0 dev receipt」；Round 5 再把 `--full` 補成**真 STARK seal**（無 dev mode），詳見下方 `--full` 小節。

| Stage | `--mock-mode` (預設, ~74 s, ~1 GB) | `--dev-mode` (新增, ~3–5 min, ~3–5 GB) | `--full`（≥32 GB 主機；真 STARK seal） |
|---|---|---|---|
| **00** risc0 hello-world | 🟢 **REAL**（同三模式） | 🟢 **REAL**（同三模式） | 🟢 **REAL** |
| **01** fetch prover bin | 🟢 **REAL** | 🟢 **REAL** | 🟢 **REAL** |
| **02** build machine | 🔴 stub snapshot dir | 🟢 **REAL** `cartesi-machine --no-ram-image --no-root-flash-drive --hash-tree=hash_function:sha256 --store=…` | 🟢 **REAL**（重用 dev 的真 cartesi-machine 路徑；`--full-rootfs` 再升級成真 Linux-rootfs） |
| **03** collect step | 🔴 5×`openssl rand -hex 32` + skeleton JSON | 🟢 **REAL** `cartesi-machine --load=… --max-mcycle=1 --log-step=1,step.bin`；parse 32 B pre_root + 8 B mcycle + 32 B post_root header | 🟢 **REAL**（同 dev 的真 `--log-step`；`--full-rootfs` 走 multi-mcycle） |
| **04** prove | 🔴 264 B `MOCKPRF\0`-magic file | 🟡 **REAL r0vm + `RISC0_DEV_MODE=1`**：`r0vm --elf cartesi-risc0-guest-step-prover.bin --initial-input step.bin --receipt step.proof.bin`（~400 B dev receipt） | 🟢 **REAL STARK**：同指令但**無** `RISC0_DEV_MODE` → 真 STARK seal（+ Groth16 wrap 可選） |
| **05** verify | 🔴 magic+root cross-check, prints `[MOCK]` line | 🟡 **REAL `cargo risczero verify`** (with `RISC0_DEV_MODE=1`), prints `[DEV-REAL] … dev-receipt=true` | 🟢 **REAL** `cargo risczero verify`（無 dev mode）→ `Receipt is valid!` + `[FULL-REAL]`；可再 `forge` 鏈上 `verifyStep` |

**真實度評分**：mock ≈ 30%，dev-real ≈ 80%，**full = 100%（真 STARK seal，已實作；在 ≥32 GB 主機如 Strix Halo 跑 `make demo-b-full`）**。

### 為什麼 02 在 dev-mode 不用 Linux rootfs？

Cartesi v0.20.0 release 的 `machine-emulator_amd64.deb` **沒有** ship `linux.bin` / `rootfs.ext2`（`/usr/share/cartesi-machine/images/` 目錄是空的）。所以原 plan 的 `cartesi-machine -- /bin/echo hello` 路線無法直接走。

我們的解法（更乾淨）：用 `--no-ram-image --no-root-flash-drive` 啟一台**最小 cartesi 機器**（沒有 Linux kernel，沒有 rootfs，只剩 microarchitecture + shadow state + 各個 PMA range 的空 backing store）。每個 mcycle 仍然是一次**真實**的 microarchitecture state transition（透過 `uarch-ram.bin` 微碼），所以 RISC0 guest 證的還是真的 Cartesi state transition——只是 transition 內容是「沒有指令可以執行」的 noop。

副效果：snapshot 從 ~190 MB 縮到 ~1.5 MB（`phtc_size:64` 把 page hash tree cache 從 50 MB 縮到 < 1 MB）。

### Dev-mode 的 dev-receipt 是什麼？

`RISC0_DEV_MODE=1` 開啟 RISC Zero 的 dev mode：
- prover 仍跑完整 guest（讀 step.bin、parse header、驗 pre/post root、commit journal）
- 但 STARK 部分被換成 dev placeholder（~400 bytes 而非真實的 ~150 KB STARK seal）
- verifier 在同樣 `RISC0_DEV_MODE=1` 環境變數下會 accept 這個 dev receipt
- **不是密碼學上 sound** — production 不可用
- 重點：把 guest binary、image_id、step log binary format、host wiring 全部走過一輪，比 mock-mode 的 `MOCKPRF` magic 強得多

### `--full` flag（真 STARK seal，已實作）

`run-all.sh --full`（或頂層 `make demo-b-full`）現在走**真 STARK**：02/03 重用 dev 的真 cartesi-machine 路徑，04 跑 `r0vm --elf cartesi-risc0-guest-step-prover.bin --initial-input step.bin --receipt step.proof.bin`（**不開** `RISC0_DEV_MODE`），05 用 `cargo risczero verify`（無 dev mode）印 `✅ Receipt is valid!` + `[FULL-REAL] step.proof.bin verified: real STARK seal`。`step.public.json` 會標 `mode="full", real_stark=true, dev_receipt=false, receipt_bytes=<size>`。

更上層還串了：Groth16 STARK→SNARK（`06-snark.sh` → ~200 B seal）+ `forge` 在本地 `anvil` 上對 vendored verifier 跑 `verifyStep(...)` 鏈上驗證（`07-onchain-verify.sh` / `make demo-b-groth16`），以及 `--full-rootfs`（真 Linux-rootfs Cartesi machine，prove 一個 multi-mcycle 有意義的 state transition，取代 empty-noop）。

唯一前提是**主機規格**：真 STARK prove 過去在 11 GB free 的錄影機會撞上 ≥16 GB OOM 牆，所以 live capture 留在具備條件的 ≥32 GB 主機（如 **128 GB unified memory 的 AMD Strix Halo**）——一鍵 `bash scripts/run-on-halo.sh`，build→detect→run→capture→commit，實測 wall time / peak RSS / proof size 寫入 `artefacts/full-run.info`（本檔**不預填數字**，避免假數據）。

### AMD Strix Halo 硬體加速（誠實分層）

`--full` 真 STARK + Groth16 wrap 在 AMD 上**只靠 CPU**：RISC0 `r0vm` 沒有 ROCm/Vulkan prover（只在 CUDA / Metal 加速），Groth16 prover 為 x86-only、跑在 Docker、CPU-bound；Strix Halo 的紅利是 **16-core / 32-thread Zen 5**（`RAYON_NUM_THREADS`）+ **128 GB unified LPDDR5X**（移除 ≥16 GB OOM 牆）。iGPU（Radeon 8060S, gfx1151）/ NPU（XDNA2）只加速 **AI 模型**（DEAAP embedding / LoRA / EZKL 模型執行），**不加速 ZK proof**。完整引擎矩陣與 ROCm/kernel 細節見 [`../../docs/amd-strix-halo-acceleration.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/amd-strix-halo-acceleration.md)。

---

## 3a. Dev-real mode

```bash
docker run --rm \
    -v "$(pwd)/artefacts:/work/artefacts" \
    -v "$(pwd)/dist:/work/dist" \
    risc0-cartesi-demo:local \
    bash scripts/run-all.sh --dev-mode
```

預期 stage banners：
- `[02/DEV] OK — snapshot=… (1.5M)`
- `[03/DEV] OK` + 印出 `pre`, `post`, `mcycle_count`
- `[04/DEV] OK` + `WARNING: proving in dev mode. This will not generate valid, secure proofs.`
- `[DEV-REAL] step.proof.bin verified: pre_root↔post_root match, mcycle=1, dev-receipt=true`

`--dev-mode` 多生的 artefacts（相對於 `--mock-mode`）：
- `artefacts/machine-snapshot/` 整個目錄（~1.5 MB；真實 cartesi-machine store）
- `artefacts/machine-snapshot.info` （文字摘要：size、hash function、phtc_size、file list）
- `artefacts/step.bin` （v0.20.0 step log binary，~19 KB）
- `artefacts/step.uarch-hashes.txt` （`<mcycle>,<uarch>: <hex>` dense hash 列；錄影時可秀）

---

## 4. Demo runtime breakdown

實測（host: 15 GB RAM / 11 GB free, x86_64, Docker 29.4, image pre-warmed）。Tier 2 後並列 mock-mode 與 dev-mode 兩條 timing：

### Mock-mode (default, `~55 s` after Tier 2 cargo cache warmed)

| Stage | 耗時 | 主要花在哪 |
|---|---|---|
| 00 RISC0 hello-world | **~52 s** | host crate `cargo run --release`（compile cached ~52 s + prove + verify ~1 s + serialise）|
| 01 fetch prover bin | **<1 s** | dist/ 已 cache，跳過 |
| 02 build machine (mock) | **<1 s** | 寫 stub `README` + `config.json` |
| 03 collect step (mock) | **<1 s** | 5 × `openssl rand -hex 32`，寫 3 個檔 |
| 04 prove (mock) | **<1 s** | 264 byte 檔 + JSON |
| 05 verify (mock) | **<1 s** | header check + jq cross-check |
| **總計** | **~55 s** | — |

### Dev-mode (`~69 s` Tier 2 measurement, peak RSS ~1.14 GiB)

| Stage | 耗時 | 主要花在哪 |
|---|---|---|
| 00 RISC0 hello-world | **~56 s** | 同 mock-mode（cached cargo + STARK prove+verify） |
| 01 fetch prover bin | **<1 s** | dist/ 已 cache |
| 02 build machine (dev) | **~7 s** | `apt install machine-emulator_amd64.deb` ~5 s + minimal cartesi-machine snapshot store ~2 s |
| 03 collect step (dev) | **~2 s** | `cartesi-machine --load=... --max-mcycle=1 --log-step=1,step.bin` + header parse |
| 04 prove (dev) | **~2 s** | `RISC0_DEV_MODE=1 r0vm --elf ...` — dev seal 跳過了 STARK 部分 |
| 05 verify (dev) | **~2 s** | `cargo risczero verify`（dev mode 下幾百 ms 完成）|
| **總計** | **~69 s** | — |

> Dev-mode 比預期（~3-5 min, ~3-5 GB RAM）快很多 / 省記憶體很多，因為我們的 cartesi-machine 是「最小機器」（`--no-ram-image --no-root-flash-drive`），跑 1 mcycle 不會碰到 Linux kernel 啟動成本；同時 `RISC0_DEV_MODE=1` 把 STARK 部分換成 dev placeholder，prover 在幾秒內結束。

首次冷啟 docker run（image 內 hello-world 還沒 pre-warm 過）約多 ~90 s 額外編譯時間；現行 Dockerfile 在 build 階段就 pre-warm 了，所以錄影時直接 cached。

---

## 5. 五個 stage 腳本一覽

每個 script 接受 `--mock-mode | --dev-mode | --full`（也讀 `MOCK_PROVER=1` / `DEV_MODE=1` env）並分支：

| # | Script | 作用 | mock-mode | dev-mode (Tier 2 新增) |
|---|--------|------|-----------|------------------------|
| 00 | `scripts/00-risc0-hello-world.sh` | scaffold `cargo risczero new --guest-name multiply hello-world`，patch host main.rs 加 `bincode::serialize(&receipt)` 與 verifier-success print，`cargo run --release` | 真實 prove + verify | 同 mock |
| 01 | `scripts/01-fetch-prover-bin.sh` | 從 Cartesi v0.20.0 release 抓 `*.bin`、`*image-id.txt`、`machine-emulator_*.deb` | 真實 GitHub release 下載 | 同 mock |
| 02 | `scripts/02-build-machine.sh` | 裝 `.deb` + 建 Cartesi machine snapshot | stub `machine-snapshot/` 目錄（README + config.json） | **real** `cartesi-machine --no-ram-image --no-root-flash-drive --hash-tree=hash_function:sha256,phtc_size:64 --max-mcycle=0 --store=...` |
| 03 | `scripts/03-collect-step.sh` | 跑 1 mcycle、收 pre/post root hashes + step log | 5 × `openssl rand -hex 32` + JSON skeleton | **real** `cartesi-machine --load=... --max-mcycle=1 --log-step=1,step.bin`；parse step.bin header（32 B pre_root + 8 B mcycle LE + 32 B post_root）+ dump dense uarch hashes |
| 04 | `scripts/04-prove.sh` | RISC0 prove against step log | 264 bytes `MOCKPRF`-magic file + `step.public.json` | **real** `RISC0_DEV_MODE=1 r0vm --elf cartesi-risc0-guest-step-prover.bin --initial-input step.bin --receipt step.proof.bin`（~393 B dev receipt） |
| 05 | `scripts/05-verify.sh` | RISC0 verify | magic header + jq root cross-check → `[MOCK]` line | **real** `RISC0_DEV_MODE=1 cargo risczero verify --path step.proof.bin <image_id>` → `✅ Receipt is valid!` + `[DEV-REAL]` line |

每個 script idempotent；可以單獨跑（例：`DEV_MODE=1 bash scripts/03-collect-step.sh` 或 `bash scripts/04-prove.sh --dev-mode`）。

### 5.1 Next-step / 延伸腳本（HOST-run；預設不在錄影路徑）

下列是「下一步」延伸交付，已 wire 進頂層 `Makefile` 與 `scripts/run-on-halo.sh` 的 opt-in flag（`--bench` / `--folding` / `--npu-probe`，預設 OFF）。重型 / 實驗性，設計在 ≥32 GB 主機（如 Strix Halo）上跑，不在本機錄影路徑：

| Script | 作用 | 入口 / 備註 |
|--------|------|-------------|
| `scripts/bench-stark-throughput.sh` | CPU STARK throughput sweep：對 `MCYCLES` × `RAYON_NUM_THREADS` 跑真 `r0vm` `--full` prove，`/usr/bin/time -v` 收 wall + peak RSS | `make bench`；預設 `--full`（也吃 `--full-rootfs`）；env：`MCYCLES` / `THREADS` / `MIN_RAM_GB` / `SKIP_BUILD`；輸出 `artefacts/bench/throughput.csv` |
| `scripts/plot-throughput.py` | 把上面的 CSV 畫成 `artefacts/bench/throughput.png`（threads-vs-time、mcycle-vs-time…）；缺 matplotlib 時退化成 `throughput.md` | 由 `make bench` 接著呼叫（plot 行前綴 `-`，缺 matplotlib 不會讓 target 失敗） |
| `scripts/07-onchain-verify.sh`（**hardened Sepolia 路徑**） | local anvil（預設）或 **Sepolia testnet** 上部署 vendored Groth16 verifier + StepVerifier 並 `verifyStep(seal,pre,post)` | `STEP_SEPOLIA=1 make deploy-sepolia`；需 `SEPOLIA_RPC_URL` + `SEPOLIA_PRIVATE_KEY`（env only，key 永不 log）+ 真 Groth16 seal（先跑 `06-snark.sh`），**會花真 testnet ETH**；寫 `artefacts/sepolia-deploy.json`；見 [`../../docs/sepolia-deploy.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/sepolia-deploy.md) |
| `scripts/08-fold-steps.sh` | **experimental**：折疊 N 個 chained Cartesi step 的 `sha256(pre‖post)` journal relation（Sonobe Nova+CycleFold → DeciderEth Groth16）→ emit `NovaDecider.sol` → anvil 上 verify。crate 在 [`../folding-step-demo/`](../folding-step-demo/) | `make demo-c-fold`；env：`FOLD_N`（預設 2）/ `FOLD_DOCKER=1` / `FOLD_DEGRADED=1`；本機不 `cargo build`（重型 git deps，留給 Halo） |
| `scripts/npu-probe.sh` | **experimental / research-only**：唯讀探測 XDNA2 NPU（amdxdna / `/dev/accel*` / firmware / XRT / IRON），**永不 hard-fail**（always exit 0）；NPU 加速的是 **AI 模型，不是 ZK proof** | `make npu-probe`；見 [`../../reading-notes/path-d-npu-xdna2.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/reading-notes/path-d-npu-xdna2.md) |

---

## 6. Committed artefacts

`poc/risc0-cartesi-step-demo/artefacts/` 內已 commit 的快照（學生 clone 完 repo 就可以對著 verify）。**Tier 2 當時**目錄反映的是最近一次 `--dev-mode` run 的真實輸出（含 real cartesi-machine snapshot + real step log + r0vm dev receipt）；**`aa54a38`（2026-06-08）起改為真 `--full` run 覆寫、`c077201`（2026-06-10）起是 `--full-rootfs` 的真值**，所以下表的「dev-mode 實測」欄位讀作「該模式會產出的大小」，不是目前 commit 在 repo 裡的大小（實況見該列括號）：

| 檔案 | 大小（dev-mode 實測） | 內容 |
|------|----------------------|------|
| `risc0-hello-world.log` | ~390 B | stage 00 cargo run stdout，含 `[00] receipt verified successfully against MULTIPLY_ID (real STARK seal)` |
| `risc0-hello-world.receipt.bin` | 209 506 B (~205 KB) | bincode-serialised RISC0 STARK receipt（含 seal + journal） |
| `machine-snapshot/` | ~1.5 MB | **dev-mode 新增** — 真實 cartesi-machine v0.20.0 store dir（sha256 hash tree, phtc_size=64） |
| `machine-snapshot.info` | ~2.2 KB | **dev-mode 新增** — `du -sh`、hash function、file list 文字摘要 |
| `step.bin` | 18 944 B (~19 KB)（**目前 commit：35 744 B**，`--full-rootfs` mcycle=100） | **dev-mode 新增** — cartesi-machine `--log-step=1,...` 的原始 binary step log（32 B pre_root + 8 B mcycle LE + 32 B post_root + per-cycle access log entries） |
| `step.uarch-hashes.txt` | ~140 B | **dev-mode 新增** — `<mcycle>[,<uarch_cycle>]: <hex>` 行，從 cartesi-machine stdout 抓回 |
| `step.pre.hash` | 67 B | pre-state root (`0x` + 64 hex + LF)。dev: 從 step.bin 抓；mock: openssl random |
| `step.post.hash` | 67 B | post-state root。同上 |
| `step.log.json` | ~560 B（**目前 commit：594 B**，`"mock": false` / `mode=full-rootfs`） | dev-mode: `{mode=dev-real, version=v0.20.0, mcycle_count, hash_function, ...}` 真實 metadata；mock: 5×uarch hash skeleton |
| `step.proof.bin` | dev: **393 B** / mock: 264 B（**目前 commit：1 112 064 B — `--full-rootfs` 的真 r0vm STARK seal**） | dev: RISC0 dev receipt (binary, 不是 STARK)；mock: 8 B `MOCKPRF\0` + 256 random；**full / full-rootfs: 真 STARK seal（無 `RISC0_DEV_MODE`）** |
| `step.public.json` | ~460 B（**目前 commit：526 B**，含 `real_stark=true` / `dev_receipt=false` / `receipt_bytes=1112064`） | `{mode, image_id, pre_root, post_root, mcycle, dev_receipt?, receipt_bytes?, note}` |

**錄影路徑**：當錄影需要 mock-mode 的乾淨輸出時，跑一次 `bash scripts/run-all.sh --mock-mode` 會把上表 5 個檔（pre/post/log/proof/public）替換為 mock 版本（machine-snapshot/ 變成 stub README + config.json）。dev-mode 新增的 4 個檔（step.bin、step.uarch-hashes.txt、machine-snapshot.info 不會被 mock 觸到，但 machine-snapshot/ 內容會被 stub 取代）。

`poc/risc0-cartesi-step-demo/dist/` 內的 fetched binaries（`cartesi-risc0-guest-step-prover.bin` 868 KB、`machine-emulator_amd64.deb` ~55 MB、`*-image-id.txt` 64 B）**沒有** commit — students 跑 `bash scripts/01-fetch-prover-bin.sh` 重新抓即可。

---

## 7. Solidity 驗證（選用）

`solidity/StepVerifier.sol` 是 reference snippet（**not deployable as-is**）。它包住 RISC0 官方 `IRiscZeroVerifier`：

```solidity
// 真實 cartesi-risc0-guest-step-prover.bin 的 journal 是 abi.encode(pre, mcycle, post)
// （三個 32-byte ABI words），不是 (pre, post)。2026-06-08 在 Strix Halo 用真 seal
// 實跑 07 才發現並修正——少了 mcycle 會讓鏈上 verifyStep revert(VerificationFailed)。
journalDigest = sha256(abi.encode(preStateHash, mcycle, postStateHash));
RISC_ZERO_VERIFIER.verify(seal, STEP_PROVER_IMAGE_ID, journalDigest);
```

`verifyStep(seal, pre, mcycle, post)` 已在本地 anvil 實跑回傳 `true`（見 [`docs/IMPLEMENTATION-STATUS.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/IMPLEMENTATION-STATUS.md)）。簡報投影片可以截這段，搭配 v0.20.0 release notes 的 "Solidity implementations" 那一條。要在 testnet 上跑見 [`docs/sepolia-deploy.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/docs/sepolia-deploy.md)。

---

## 8. Troubleshooting (top 5)

| 症狀 | 可能原因 | 解法 |
|------|---------|------|
| `cargo run --release` for hello-world 卡在 compile > 10 min | toolchain 還在下載 / 首次 build risc0-zkvm + dependencies | 等；或 set `RISC0_DEV_MODE=1 cargo run --release` 切 dev-mode（receipt 仍真實生成，但 STARK 由 dev placeholder 替代，prove 時間從 30 s 變幾百 ms） |
| `Risc Zero Rust toolchain not found. Try running rzup install rust` | image 沒在 build 階段 install RISC0 fork rust | Dockerfile 已固定執行 `rzup install rust`；若你魔改了，確認 builder stage 有這行 |
| `/usr/bin/ld: ... Relocations in generic ELF (EM: 243)` | 你在 mock-mode 嘗試 cross-compile counter（已棄用路徑） | Tier 2 後 stage 02 mock 不再 cross-compile；換 `--dev-mode` 走最小機器，或裝 `gcc-riscv64-linux-gnu`（Dockerfile 已加） |
| `zk_merkle_tree_hash: hash_tree_target must be 1` (r0vm panic) | snapshot 用了 keccak256 hash tree | Tier 2 後 `02-build-machine.sh --dev-mode` 已硬編碼 `--hash-tree=hash_function:sha256`；若手動跑請保持 sha256 |
| `WARNING: proving in dev mode. This will not generate valid, secure proofs.` | dev-mode 預期輸出 | **不是 bug**——這代表 04 的 r0vm 用了 RISC0_DEV_MODE=1，receipt 不是 STARK seal |
| `01-fetch` curl 抓回 HTML 不是 deb | release tag 拼錯 | 確認 `CARTESI_RELEASE_TAG=v0.20.0`；瀏覽 <https://github.com/cartesi/machine-emulator/releases/tag/v0.20.0> |
| `apt-get install ./machine-emulator_amd64.deb` 報依賴錯 | base image 不是 Trixie | Dockerfile 用 `debian:trixie-slim`；bare-metal 請升級到 Debian 13 / Ubuntu 25.04 |

---

## 9. Backup mode（整個 Cartesi 整合垮掉的退路）

退到官方 RISC Zero hello-world：

```bash
docker run --rm risc0-cartesi-demo:local \
    bash -c 'cd /work/risc0-hello-world && cargo run --release'
```

這就是我們 stage 00 跑的同一條路徑。跑出 receipt 後一樣可以 `r0vm verify` 演示，雖然缺少「Cartesi state transition」的故事線，但 zkVM prove/verify 的精神不變。

---

## 10. Recording tips（Day 6 錄影流程）

> Tier 2 後可挑兩條路線錄影：mock（~55 s, laptop-friendly）或 dev-real（~69 s, 真 Cartesi+RISC0）。下面流程以 mock 為主；要切 dev-real 把 `--mock-mode` 換成 `--dev-mode` 並把 zoom-in 2 改成「stage 03 真實 step.bin header」、zoom-in 3 改成 `cargo risczero verify --path step.proof.bin <image_id>` 印 `✅ Receipt is valid!`。

建議錄製順序（mock-mode，總長 ~3 分 30 秒）：

1. **0:00–0:20** `cat README.md | sed -n '1,40p'` 介紹 demo 目的，特別念出「stage 00 是真 RISC0、stage 03–05 是 mock，shape 對齊 release」。
2. **0:20–0:35** `ls scripts/ dist/ artefacts/` 介紹檔案結構；點出 `cartesi-risc0-guest-step-prover.bin` 868 KB 真檔 + `risc0-hello-world.receipt.bin` 205 KB 真 STARK receipt。
3. **0:35–2:00** 跑 `docker run --rm -v $(pwd)/artefacts:/work/artefacts -v $(pwd)/dist:/work/dist risc0-cartesi-demo:local bash scripts/run-all.sh --mock-mode`。整段 ~74 s，逐 stage banner 都會出現。
   - **zoom-in 1**: stage 00 結束的 `[00] receipt verified successfully against MULTIPLY_ID (real STARK seal)` — 強調這是真的密碼學。
   - **zoom-in 2**: stage 03 的 `step.0.hash=0x...` × 5 行 — 強調 shape 跟 release notes 的 `--dense-uarch-hashes` 一樣。
   - **zoom-in 3**: stage 05 的 `[MOCK] step.proof.bin verified: pre_root↔post_root match, mcycle_count=100` — 誠實標 `[MOCK]`。
4. **2:00–2:30** `cat artefacts/step.public.json | jq` 展示輸出 schema；`xxd artefacts/step.proof.bin | head -3` 展示 `MOCKPRF` magic。
5. **2:30–2:55** `xxd -l 32 artefacts/risc0-hello-world.receipt.bin` 展示真 receipt 的前 32 bytes（這是 STARK seal 開頭，不是 mock magic）；對比 mock proof 強調差異。
6. **2:55–3:30** `cat solidity/StepVerifier.sol | head -50` 結尾介紹「真的要上鏈長這樣」，提一句「`--full` 已實作：把 02–05 切回**真 STARK seal**（無 dev mode）+ Groth16 wrap + 鏈上 `verifyStep`，在 ≥32 GB 主機（如 128 GB Strix Halo）以 `make demo-b-full` 實跑；錄影機 RAM 不足，故 live capture 在 Halo 上、實測寫入 `artefacts/full-run.info`」。

> 投影片建議搭配截圖：(a) `--dense-uarch-hashes` 從 release notes 的 changelog；(b) `r0vm verify` 終端輸出；(c) `MOCKPRF` xxd dump（誠實展示）。

---

## 11. References

- Cartesi Machine Emulator v0.20.0 release: <https://github.com/cartesi/machine-emulator/releases/tag/v0.20.0>
- PR #343 feature/risc0: <https://github.com/cartesi/machine-emulator/pull/343>
- RISC Zero releases: <https://github.com/risc0/risc0/releases>
- rzup installer: <https://risczero.com/install>
- cartesi/dave verifier reference: <https://github.com/cartesi/dave>
- RISC0 zkVM docs: <https://dev.risczero.com/api/zkvm/>
- Hello-world fallback example: <https://github.com/risc0/risc0/tree/main/examples/hello-world>

<!-- demo-B-status: PASS build_seconds=700 run_seconds_mock=55 run_seconds_dev=69 dev_mode_ram_peak_gb=1.14 real_stages_mock="00,01" real_stages_dev="00,01,02,03,04,05" mock_stages="02,03,04,05" hello_world_verified=true dev_receipt_verified=true full_mode=implemented full_real_stages="00,01,02,03,04,05" full_real_stark=implemented groth16_onchain=implemented full_rootfs=implemented full_live_capture=pending_on_strix_halo full_metrics=artefacts/full-run.info amd_doc=docs/amd-strix-halo-acceleration.md -->
