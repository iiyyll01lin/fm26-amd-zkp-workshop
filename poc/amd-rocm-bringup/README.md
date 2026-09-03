# `amd-rocm-bringup` — gfx1151 ROCm bring-up runbook (runnable)

The course (M1 / M2 / the ROCm FAQ) *describes* every gfx1151 bring-up trap. This
demo makes that knowledge **runnable**: a single idempotent, no-sudo-by-default
probe that checks each precondition on the box in front of you and tells you
exactly what to fix, emitting a structured `artefacts/bringup-report.json`.

It is the hands-on companion to the read-only
[`amd-accel-detect.sh`](../risc0-cartesi-step-demo/scripts/amd-accel-detect.sh)
banner: where that one **exports env** for the workloads, this one **diagnoses
each bring-up precondition** and records a machine-readable verdict.

> **Honesty rule (held everywhere in this repo).** iGPU/NPU accelerate **AI
> models**; the iGPU's OpenCL path accelerates **SNARK primitives** (MSM/NTT,
> size-gated); the **RISC0 `r0vm` STARK is CPU-only on AMD**. This probe reports
> *bring-up readiness only* — it makes no proving claim. Backs course module
> **M12 — ROCm bring-up runbook**.

## Run it

```bash
make rocm-bringup            # from repo root: probe + write artefacts/bringup-report.json
# or directly:
bash poc/amd-rocm-bringup/scripts/diagnose.sh
BRINGUP_TTM=1 bash poc/amd-rocm-bringup/scripts/diagnose.sh   # also read live TTM/GTT knobs
```

The script **always exits 0** (a probe must never abort a caller's pipeline);
the verdict lives in the JSON `ready` field, not the exit status. On a laptop
with no ROCm it records `rocm:false` and the honest stop-point instead of
crashing.

## What it checks (the bring-up ladder)

| # | check | what "ok" means | repair hint when not |
|---|---|---|---|
| 1 | `kernel` | on the 6.18.4+ stable line | upgrade to ≥ 6.18.4; on 6.19.x set the ISA override |
| 2 | `hsa_override` | `HSA_OVERRIDE_GFX_VERSION` consistent with the kernel | set `11.5.1` only on ≥ 6.19, unset it on 6.18.x |
| 3 | `hsa_sdma` | `HSA_ENABLE_SDMA=0` (conservative copy path) | export `HSA_ENABLE_SDMA=0` for stability |
| 4 | `firmware` | **not** the `linux-firmware-20251125` trap build | install a build before/after 20251125, reboot |
| 5 | `device_nodes` | `/dev/kfd` + `/dev/dri/renderD*` readable by you | `usermod -aG render,video $USER`, re-login |
| 6 | `rocm` | `rocminfo` enumerates `gfxNNNN` | install ROCm 7.2+, re-check nodes + groups |
| 7 | `hipcc` | HIP compiler on PATH | add `/opt/rocm/bin` to PATH |
| 8 | `rocm_libs` | rocBLAS + hipBLASLt + rocFFT present | install the ROCm math-lib dev packages |
| 9 | `ttm_gtt` | TTM page limit covers most of unified RAM | raise the GTT/TTM page limit (below) |
| 10 | `docker` | docker present for containerised ROCm | optional; mount `/dev/kfd` + `/dev/dri` |

`ready=true` requires: ROCm enumerates `gfx1151` **and** the device nodes are
readable **and** no hard `fail`. Warnings (e.g. a 6.17 kernel that still works)
do not block readiness — they are honest "works, but not recommended" notes.

## TTM / GTT page-limit tuning — the unified-memory unlock (concrete commands)

Strix Halo's headline is **94 GB usable unified LPDDR5X**, but the `amdgpu`
driver's **TTM (Translation Table Manager)** caps how many pages the GPU may pin
through **GTT (Graphics Translation Table)**. Leave it at the default and you hit
a **GPU-side OOM while tens of GB of RAM sit free** — the exact wall M1/M2/the
FAQ warn about but never show how to move.

> **Two-knob truth (measured) + full procedure → [`TTM-RUNBOOK.md`](TTM-RUNBOOK.md).**
> A full-offload model is bound by the **smaller** of `ttm.pages_limit` and
> `ttm.page_pool_size`. Raising `pages_limit` **alone is a no-op**: on the Z2,
> `pages_limit=60 GiB` with `page_pool_size` left at the ~47 GiB default still
> OOM'd a 54 GB BF16 model (`allocating 51518.17 MiB ... out of memory` —
> evidence: [`artefacts/ttm-bigmodel-ceiling.log`](artefacts/ttm-bigmodel-ceiling.log)).
> Raise **both** (+ `gttsize`). Ready-to-apply templates:
> `config/amdgpu-ttm.conf` (modprobe.d) ·
> `config/ttm-kernel-cmdline.txt` (grub). The
> partial-raise state is captured read-only in
> [`artefacts/bringup-report-ttm.json`](artefacts/bringup-report-ttm.json).

### 1. Read the live knobs (no root)

```bash
cat /sys/module/ttm/parameters/pages_limit       # max pages TTM will pin (GPU-addressable)
cat /sys/module/ttm/parameters/page_pool_size    # TTM page pool size
cat /sys/module/amdgpu/parameters/gttsize        # amdgpu GTT size in MiB (-1 == TTM decides)
getconf PAGE_SIZE                                 # usually 4096 (4 KiB)
```

Convert pages → GiB: `GiB = pages * PAGE_SIZE / 1024^3`. On this box the default
`pages_limit` was **12,329,039 pages ≈ 47 GB** — i.e. exactly **half** of the
94 GB pool (TTM's default is "half of system RAM"). That half-RAM default is the
silent ceiling: a model or proof that needs > 47 GB GPU-addressable memory OOMs
even though 94 GB is installed.

### 2. Raise it for the current boot (root, non-persistent)

```bash
# Pick a page count for the GiB you want. For 90 GB on 4 KiB pages:
#   pages = 90 * 1024^3 / 4096 = 23592960
echo 23592960 | sudo tee /sys/module/ttm/parameters/pages_limit
echo 23592960 | sudo tee /sys/module/ttm/parameters/page_pool_size
```

(Some ROCm/kernel builds expose these as read-only after boot; if `tee` reports
*permission denied even as root*, use the boot-time method below.)

### 3. Make it persistent (root; survives reboot)

Set the `amdgpu` / `ttm` module parameters via the kernel command line so they
apply before the GPU is brought up:

```bash
# /etc/default/grub  ->  append to GRUB_CMDLINE_LINUX_DEFAULT:
#   amdgpu.gttsize=-1 ttm.pages_limit=23592960 ttm.page_pool_size=23592960
sudo nano /etc/default/grub
sudo update-grub        # (Debian/Ubuntu)  — or: grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot
```

Or via a modprobe conf (loaded at driver init):

```bash
echo 'options ttm pages_limit=23592960 page_pool_size=23592960' | \
  sudo tee /etc/modprobe.d/amdgpu-ttm.conf
echo 'options amdgpu gttsize=-1' | sudo tee /etc/modprobe.d/amdgpu-gtt.conf
sudo update-initramfs -u && sudo reboot
```

`amdgpu.gttsize=-1` means "let TTM decide" (use the raised `pages_limit`); a
positive value pins GTT to that many **MiB**. After reboot re-run
`make rocm-bringup` — the `ttm_gtt` check should now report a limit covering most
of the 94 GB pool, and a >16 GB model/proof can stay GPU-resident
(see [M2 — Unified memory](https://github.com/iiyyll01lin/zkp-final/blob/main/course/modules/02-unified-memory.md) and
[`lab/14`](../../lab/14_unified_memory_bigmodel.ipynb)).

> **Honest stop-point.** Steps 2–3 need root and a reboot; this probe **prints**
> them as hints and **never runs** them. The committed
> `artefacts/bringup-report.json` was produced on the real Strix Halo
> (`gfx1151`, ROCm 7.2.3, kernel 6.17) by the read-only path.

## Layout

```
amd-rocm-bringup/
├── scripts/
│   └── diagnose.sh                  # idempotent, no-sudo probe -> bringup-report.json
│                                    #   (reads pages_limit + page_pool_size + effective pool)
├── config/                          # ready-to-apply TTM/GTT raise TEMPLATES (human: root + reboot)
│   ├── amdgpu-ttm.conf              #   /etc/modprobe.d drop-in
│   └── ttm-kernel-cmdline.txt       #   kernel-cmdline (grub) snippet + pages table
├── artefacts/
│   ├── bringup-report.json          # committed baseline probe (source of truth for M12 / lab 20)
│   ├── bringup-report-ttm.json      # read-only snapshot showing the partial-raise trap
│   └── ttm-bigmodel-ceiling.log     # measured OOM: 54 GB BF16 vs the ~47 GiB page_pool_size cap
├── TTM-RUNBOOK.md                   # the >47 GiB unlock: apply (human) + verify (read-only)
└── README.md
```

## Replay path

The "result" of a bring-up is a **verdict + repair list**, not a chart, so its
replay is simple: on the real Strix Halo `make rocm-bringup` re-probes live; on
any laptop the committed `artefacts/bringup-report.json` is the source of truth
that [upstream `lab/20_rocm_bringup.ipynb`](https://github.com/iiyyll01lin/zkp-final/blob/main/lab/20_rocm_bringup.ipynb) renders
(via `labkit.load_bringup_report()`), so the notebook opens and tells the same
story on a machine with no GPU. That notebook is **not** one of the six carried
in this trimmed repo — the artefact it reads is, so the link goes upstream.
