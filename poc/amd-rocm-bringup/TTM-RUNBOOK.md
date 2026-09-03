# TTM/GTT Runbook · the >47 GiB unified-memory unlock for gfx1151

> **Goal.** Let the Strix Halo iGPU grow into the full unified LPDDR5X pool so a
> **>47 GiB-resident** model (e.g. Gemma-3-27B **BF16**, ~54 GB) fits at full
> offload on the presenter **HP Z2 Mini G1a** — not just on the 96 GB-carveout
> Halo B. **Applying the change is a HUMAN step (root + reboot); verification is
> read-only and wired into [`scripts/diagnose.sh`](scripts/diagnose.sh) +
> [upstream `lab/20_rocm_bringup.ipynb`](https://github.com/iiyyll01lin/zkp-final/blob/main/lab/20_rocm_bringup.ipynb).**
> That notebook is **not** one of the six carried in this trimmed repo, so the
> link goes upstream.
> · **目標**：把 TTM/GTT 調高，讓 iGPU 吃進整個 unified pool，>47 GiB 的模型也能全量 offload。**套用需 root + 重開機（人工）；驗證全唯讀。**

---

## 0. Honesty · 誠實邊界

This is a **capacity** unlock (does a bigger model *fit*?), never a cross-vendor
speed claim. The iGPU accelerates the **AI model**, **not** the proof; the RISC0
STARK stays CPU-only. Peak GPU-resident memory is contention-robust, but any
`tok/s` remark still measures under the repo solo-guard. · 這是**容量**解鎖，非跨廠速度對打；iGPU 加速**模型**不是 proof；STARK 仍 CPU-only。

---

## 1. The two-knob truth (measured) · 兩個旋鈕（實測）

Strix Halo advertises ~94 GB usable unified LPDDR5X on Linux, but `amdgpu`'s TTM
caps how many pages the GPU may pin/pool. **TWO knobs gate a full-offload model,
and the effective pool is the SMALLER of them:**

| knob · 旋鈕 | what it caps · 管什麼 | default on the 94 GB Z2 |
|---|---|---|
| `ttm.pages_limit` | max pages TTM may allocate overall | ~47 GiB (half of RAM) |
| `ttm.page_pool_size` | the pool the **ROCr/HIP allocator** draws from — the **binding** ceiling for a full-offload model | ~47 GiB (half of RAM) |
| `amdgpu.gttsize` (MiB) | GTT aperture size (`-1` = let TTM decide) | auto |

> **The trap this runbook fixes.** On the Z2 as shipped for this workshop,
> `pages_limit` had been raised to **60 GiB** but `page_pool_size` was **left at
> the ~47 GiB default** — so the effective pool was still 47 GiB and the 54 GB
> BF16 model **failed to load**:
>
> ```text
> ggml_backend_cuda_buffer_type_alloc_buffer: allocating 51518.17 MiB ... out of memory
> alloc_tensor_range: failed to allocate ROCm0 buffer of size 54020713472
> ```
>
> Evidence: [`artefacts/ttm-bigmodel-ceiling.log`](artefacts/ttm-bigmodel-ceiling.log)
> (the same model loads fine at `-ngl 34`, peak 31.79 GB). `llama.cpp` even
> reports the device as `48161 MiB` = `page_pool_size`, **not** the 60 GiB
> `pages_limit`. **Raising `pages_limit` alone is a no-op — raise
> `page_pool_size` (and `gttsize`) too.** · **陷阱**：只調 `pages_limit` 沒用，`page_pool_size` 才是 ROCr 池的實際上限，要一起調。

---

## 2. Apply — HUMAN step (root + reboot) · 套用（人工：root + 重開機）

Two equivalent persistent forms; pick one. Both set `pages_limit` **and**
`page_pool_size` to the same target. Recommended target for the 94 GB Z2:
**72 GiB = `18874368` pages** (holds the 54 GB model → ~56 GB resident with
headroom, leaves ~22 GB for the OS). Page maths + a size table are in the config
files below.

**Option A — kernel cmdline (preferred; survives an initramfs-baked driver):**
see `config/ttm-kernel-cmdline.txt`.

```bash
# append the three tokens to GRUB_CMDLINE_LINUX_DEFAULT, then:
sudo update-grub && sudo reboot
#   ...adds: amdgpu.gttsize=-1 ttm.pages_limit=18874368 ttm.page_pool_size=18874368
```

**Option B — modprobe.d drop-in:** install
`config/amdgpu-ttm.conf` to `/etc/modprobe.d/`, then:

```bash
sudo cp poc/amd-rocm-bringup/config/amdgpu-ttm.conf /etc/modprobe.d/amdgpu-ttm.conf
sudo update-initramfs -u && sudo reboot
```

> **Do NOT run these here.** This repo agent/CI never reboots or writes kernel
> params; the config files are templates and the checks below are read-only. · **本 repo 不代跑**：不重開機、不寫核心參數。

---

## 3. Verify — READ-ONLY (no sudo) · 驗證（唯讀）

```bash
cat /proc/cmdline                               # cmdline tokens present (Option A)
cat /sys/module/ttm/parameters/pages_limit      # = 18874368
cat /sys/module/ttm/parameters/page_pool_size   # = 18874368  (the one that was missing)
make rocm-bringup                               # or: bash poc/amd-rocm-bringup/scripts/diagnose.sh
```

`diagnose.sh` now reads **both** knobs, computes the **effective pool**, and
flags the half-raised trap. Capture a named snapshot without clobbering the
committed baseline via the `REPORT` override:

```bash
REPORT="$PWD/poc/amd-rocm-bringup/artefacts/bringup-report-ttm.json" \
  bash poc/amd-rocm-bringup/scripts/diagnose.sh
```

- **Before the raise / partial raise** → `ttm_gtt` is `[WARN]` with
  `effective pool ~47.0 GiB` and a `PARTIAL RAISE` hint (committed evidence:
  [`artefacts/bringup-report-ttm.json`](artefacts/bringup-report-ttm.json)).
- **After the full raise** → `ttm_gtt` is `[ OK ]` with
  `effective pool ~72 GiB` (> 47 GiB), and the 54 GB BF16 model loads at
  `-ngl 99`. Re-run the capacity measurement with
  [`../../workshop/futuremode-2026/phase-f/run-phase-f.sh`](https://github.com/iiyyll01lin/zkp-final/blob/main/workshop/futuremode-2026/phase-f/run-phase-f.sh)
  (`SKIP_CPU=1 XL_BASENAME=bigmodel-xl-z2-gemma-bf16`).

On any laptop, [upstream `lab/20_rocm_bringup.ipynb`](https://github.com/iiyyll01lin/zkp-final/blob/main/lab/20_rocm_bringup.ipynb)
renders the committed reports (baseline + the TTM snapshot) read-only.

---

## 4. Where this fits · 定位

- **Halo B (96 GB carveout)** already holds the 54 GB BF16 model at **56.13 GB**
  resident — committed in
  [`../amd-bigmodel-demo/artefacts/bigmodel-xl-gemma-bf16.csv`](../amd-bigmodel-demo/artefacts/bigmodel-xl-gemma-bf16.csv).
  It needs no raise.
- **The Z2** needs this runbook to reproduce >47 GiB on the presenter's own box.
  Until `page_pool_size` is raised, the Z2 tops out at ~47 GiB (measured); the
  committed Z2 result is **44.77 GB** (Llama-70B Q4,
  [`bigmodel-xl.csv`](../amd-bigmodel-demo/artefacts/bigmodel-xl.csv)).
- The public ">32 GB card" claim upgrade stays **gated** on `peak_gpu_gb > 32`
  **and** AMD §1.2 sign-off — see
  [`../../workshop/futuremode-2026/phase-f/artefact-schema.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/workshop/futuremode-2026/phase-f/artefact-schema.md)
  §4 and the pending-wiring checklist. This runbook only removes the *hardware*
  blocker; it flips no headline.

## Cross-references · 交叉連結

- Config templates · 設定樣板: `config/amdgpu-ttm.conf` · `config/ttm-kernel-cmdline.txt`
- Measured ceiling evidence · 實測上限: [`artefacts/ttm-bigmodel-ceiling.log`](artefacts/ttm-bigmodel-ceiling.log)
- Bring-up probe · 上機探測: [`scripts/diagnose.sh`](scripts/diagnose.sh) · [`README.md`](README.md)
- Course module · 課程: [`../../course/modules/12-rocm-bringup-runbook.md`](https://github.com/iiyyll01lin/zkp-final/blob/main/course/modules/12-rocm-bringup-runbook.md)
- Big-model harness · 大模型量測: [`../amd-bigmodel-demo/README.md`](../amd-bigmodel-demo/README.md)
