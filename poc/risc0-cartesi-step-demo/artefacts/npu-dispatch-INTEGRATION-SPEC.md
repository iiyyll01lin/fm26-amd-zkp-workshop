# Phase 2 (G2) — XDNA2 NPU dispatch: INTEGRATION SPEC for the closeout agent

> **Why this file exists.** Phase 2 produced a new opt-in runner
> (`poc/risc0-cartesi-step-demo/scripts/npu-run.sh`) and committed artefacts
> (`artefacts/npu-dispatch.{log,json}`), and updated the read-only probe's
> reading-note. To keep parallel phases conflict-free, Phase 2 did **NOT** edit
> any shared file (`Makefile`, `scripts/run-on-halo.sh`, `lab/*.ipynb`,
> `docs/IMPLEMENTATION-STATUS.md`, `docs/amd-strix-halo-acceleration.md`,
> `README.md`, `lab/labkit.py`). This spec gives the closeout agent the **exact**
> edits to wire G2 in. All line numbers are as of the Phase-2 snapshot; match on
> the quoted anchor text rather than the number if it has drifted.

## Outcome to thread through everything

- **Runner:** `poc/risc0-cartesi-step-demo/scripts/npu-run.sh` (opt-in, heavy, no
  sudo, no system writes — all state under `$HOME/.cache/zkp-npu`; always
  `exit 0`; always writes `artefacts/npu-dispatch.{log,json}`).
- **Verdict on this real Strix Halo (kernel 6.17.0-29, `1022:17f0` rev 11):**
  `BLOCKED-ENUMERATION`.
- **Graduation:** `DRIVER-READY -> COMPILE-READY -> [BLOCKED-ENUMERATION]`.
- **One-liner:** kernel driver + firmware (`fw_version=1.0.0.166`) + `render`-group
  access to `/dev/accel0` + the **authoring** toolchain (`mlir_aie` + Peano,
  installed no-root) are all green; the **only** missing piece is the user-space
  **XRT `amdxdna` device shim** (`libxrt_driver_xdna.so` / `xrt_plugin-amdxdna`),
  which can only be installed with root here — and every root route is blocked
  (non-interactive sudo needs a password; `amd/xdna-driver` ships **0** prebuilt
  releases; Ubuntu-archive XRT is `202210.2.13.466`, pre-NPU; the `mlir_aie` wheel
  has no `pyxrt` and no shim). So `xrt-smi` cannot enumerate `/dev/accel0`.
- **Honesty (unchanged):** the NPU accelerates the **AI model** (BitNet prefill /
  DEAAP embedding), **not** the ZK proof. Zero impact on prover time / proof size
  / on-chain gas.

The committed `npu-dispatch.json` is the **replay source of truth**. Relevant keys:
`verdict`, `graduation`, `enumerated`, `compile_ready`, `dispatch_ok`,
`fw_version`, `mlir_aie_version`, `peano_version`, `blocked_reason`.

---

## (a) `Makefile` — target name `npu-run`

Add `npu-run` to the `.PHONY` list (the line that currently ends
`... demo-c-fold npu-probe \`) and add the target right **after** the existing
`npu-probe:` target (anchor: `bash $(RISC0_SCRIPTS)/npu-probe.sh`). Also add one
help line near the existing `make npu-probe` help line.

Help line (put next to the `npu-probe` help echo, ~line 52):

```make
	@echo "  make npu-run             # OPT-IN heavy XDNA2 NPU dispatch (no sudo; MAY fail; writes npu-dispatch.{log,json})"
```

Target (put right after the `npu-probe:` recipe, ~line 144):

```make
# Opt-in HEAVY XDNA2 NPU *dispatch* stage — the graduation step beyond npu-probe.
# Stands up the community user-space stack (Peano+MLIR-AIE+IRON) no-root in a venv,
# tries to enumerate the NPU via XRT and run the IRON axpy (+ small int8 GEMM) on
# /dev/accel0. NEVER uses sudo / touches system paths; MAY fail; always writes
# artefacts/npu-dispatch.{log,json} and exits 0. Research-only: the NPU
# accelerates the AI MODEL, NOT the ZK proof.
npu-run:
	bash $(RISC0_SCRIPTS)/npu-run.sh
```

Exact command the target runs: `bash poc/risc0-cartesi-step-demo/scripts/npu-run.sh`
(`$(RISC0_SCRIPTS)` already resolves to `poc/risc0-cartesi-step-demo/scripts`).

---

## (b) `scripts/run-on-halo.sh` — opt-in flag `--npu-run` + `full-run.info` fields

Five edits (mirror exactly how `--npu-probe`/`RUN_NPU` is already wired):

1. **Usage comment** (~line 18, after the `--npu-probe` usage line):

```bash
#       --npu-run     make npu-run     (OPT-IN heavy XDNA2 NPU dispatch; no sudo; MAY fail; never breaks the run)
```

and append `--npu-run` to the opt-in list comment (~line 33):

```bash
#   opt-in : --bench --folding --npu-probe --npu-run   (default OFF; heavy/experimental)
```

2. **Init flag** (~line 49, the `RUN_BENCH=0 RUN_FOLDING=0 RUN_NPU=0` line) — add `RUN_NPU_RUN=0`:

```bash
RUN_BENCH=0 RUN_FOLDING=0 RUN_NPU=0 RUN_NPU_RUN=0
```

3. **Arg parse** (~line 60, right after the `--npu-probe) RUN_NPU=1 ;;` case):

```bash
        --npu-run)      RUN_NPU_RUN=1 ;;
```

4. **Stage exec** (~line 168, right after the `--npu-probe` `timed_target` line):

```bash
[[ "${RUN_NPU_RUN}" == "1" ]] && timed_target "npu-run"    make -C "${REPO_ROOT}" npu-run
```

5. **Add `npu-run` to BOTH `for t in ... npu-probe` loops** (the status table loop
   ~line 237 and the exit-gate/summary loop ~line 305): change `... demo-c-fold npu-probe`
   to `... demo-c-fold npu-probe npu-run` in both. (Like `npu-probe`, `npu-run` must
   stay OUT of the exit gate — only `demo-b-full` gates the exit code.)

6. **Metrics → `full-run.info`.** After the existing `NPU_OUT`/`NPU_VERDICT` block
   (~lines 211-220), add a parse of the committed JSON (the runner writes it under
   `ART`), then emit the new fields in the "Next-step stages" block (after the
   `npu.verdict=...` echo, ~line 255). Parse block:

```bash
# Opt-in NPU dispatch stage (npu-run): read the committed verdict JSON (guarded;
# a not-run stage leaves the file from a prior run or prints n/a).
NPU_DISPATCH_JSON="${ART}/npu-dispatch.json"
NPU_DISP_VERDICT="n/a"; NPU_DISP_GRAD="n/a"; NPU_DISP_ENUM="n/a"; NPU_DISP_COMPILE="n/a"
if [[ -f "${NPU_DISPATCH_JSON}" ]]; then
    NPU_DISP_VERDICT="$(jqf "${NPU_DISPATCH_JSON}" '.verdict')";       [[ -n "${NPU_DISP_VERDICT}" ]] || NPU_DISP_VERDICT="n/a"
    NPU_DISP_GRAD="$(jqf "${NPU_DISPATCH_JSON}" '.graduation')";       [[ -n "${NPU_DISP_GRAD}" ]] || NPU_DISP_GRAD="n/a"
    NPU_DISP_ENUM="$(jqf "${NPU_DISPATCH_JSON}" '.enumerated')";       [[ -n "${NPU_DISP_ENUM}" ]] || NPU_DISP_ENUM="n/a"
    NPU_DISP_COMPILE="$(jqf "${NPU_DISPATCH_JSON}" '.compile_ready')"; [[ -n "${NPU_DISP_COMPILE}" ]] || NPU_DISP_COMPILE="n/a"
fi
```

Emit lines (add right after `echo "npu.verdict=${NPU_VERDICT}"`):

```bash
    echo "npu.dispatch.verdict=${NPU_DISP_VERDICT}"
    echo "npu.dispatch.graduation=${NPU_DISP_GRAD}"
    echo "npu.dispatch.enumerated=${NPU_DISP_ENUM}"
    echo "npu.dispatch.compile_ready=${NPU_DISP_COMPILE}"
```

**Exact `full-run.info` field names to add:** `npu.dispatch.verdict`,
`npu.dispatch.graduation`, `npu.dispatch.enumerated`, `npu.dispatch.compile_ready`.
Expected committed values on this box: `BLOCKED-ENUMERATION`,
`DRIVER-READY -> COMPILE-READY -> [BLOCKED-ENUMERATION]`, `no`, `yes`.

(Optional: also bump the comment on the "Next-step stages" header to mention `--npu-run`.)

---

## (c) Notebook `lab/05_npu_xdna2_probe.ipynb` — live-or-replay dispatch cell

Keep the existing read-only probe cell (cell 3) as-is. **Add a new code cell after
it** (before the "Reading the verdict" markdown), using the same `lk.live_or_replay`
pattern. **Live = run `npu-run.sh`** then parse `artefacts/npu-dispatch.json`;
**replay = read the committed `artefacts/npu-dispatch.json`** (no `full-run.info`
dependency needed, but the `npu.dispatch.*` fields from (b) are an equivalent
fallback). It must print the verdict **`BLOCKED-ENUMERATION`** and the graduation
ladder.

```python
import json, subprocess

NPU_RUN = lk.repo_path("poc", "risc0-cartesi-step-demo", "scripts", "npu-run.sh")
NPU_DISPATCH_JSON = lk.repo_path("poc", "risc0-cartesi-step-demo", "artefacts", "npu-dispatch.json")


def _read_dispatch_json():
    with open(NPU_DISPATCH_JSON) as f:
        return json.load(f)


def live_dispatch():
    """LIVE: run the opt-in heavy npu-run.sh (no sudo; never breaks), then read its JSON."""
    subprocess.run(["bash", str(NPU_RUN)], text=True, capture_output=True, timeout=900)
    d = _read_dispatch_json()
    d["source"] = "live npu-run.sh"
    return d


def replay_dispatch():
    """REPLAY: read the committed npu-dispatch.json (source of truth)."""
    d = _read_dispatch_json()
    d["source"] = "committed npu-dispatch.json"
    return d


res, mode = lk.live_or_replay(
    live_fn=live_dispatch, replay_fn=replay_dispatch,
    label="XDNA2 NPU dispatch (opt-in, heavy)",
)

print(f"XDNA2 NPU dispatch  —  source: {res['source']}  (mode={mode})")
print(f"  graduation : {res['graduation']}")
print(f"  VERDICT    : {res['verdict']}")
print(f"  enumerated : {res['enumerated']}   compile_ready : {res['compile_ready']}   dispatch_ok : {res['dispatch_ok']}")
if res.get("blocked_reason"):
    print(f"\n  blocked_reason:\n  {res['blocked_reason']}")
```

Expected printed verdict (replay, committed today):

```
  graduation : DRIVER-READY -> COMPILE-READY -> [BLOCKED-ENUMERATION]
  VERDICT    : BLOCKED-ENUMERATION
  enumerated : no   compile_ready : yes   dispatch_ok : no
```

> **`make lab-replay` safety:** the live branch must be gated by `lk.live_or_replay`
> so replay never runs `npu-run.sh`. The committed `npu-dispatch.json` keeps replay
> green on any laptop. If the live gate also wants a "heavy opt-in" guard, reuse the
> same env switch the lab uses for other heavy live cells. Also update the markdown
> cell after it to narrate the graduation `DRIVER-READY -> COMPILE-READY -> [blocked
> at enumerate: root-only XRT amdxdna shim]` instead of stopping at `DRIVER-READY`.

---

## (d) `docs/IMPLEMENTATION-STATUS.md` — exact section text to paste

**(d.1) Update the TL;DR table row** (anchor line ~15, currently
`| NPU probe | ✅ | XDNA2 DRIVER-READY（研究用,不在證明路徑） |`) to:

```markdown
| NPU probe + dispatch（G2） | ✅ probe / ⛔ dispatch（documented blocker） | XDNA2 `DRIVER-READY -> COMPILE-READY -> [BLOCKED-ENUMERATION]`：kernel+韌體+authoring 全綠,唯缺 root-only XRT amdxdna shim;研究用,不在證明路徑 |
```

**(d.2) Add a new numbered section** (append after the current last numbered
section; renumber if needed). Paste verbatim:

```markdown
## 11. Path D — XDNA2 NPU dispatch 嘗試（G2,2026-06-11 新增）

把 NPU 從 read-only 的 `DRIVER-READY`（§0、`make npu-probe`）往「真的 dispatch 一顆 kernel」推。新增**明確 opt-in、可以失敗**的 runner [`poc/risc0-cartesi-step-demo/scripts/npu-run.sh`](../scripts/npu-run.sh)（**完全不用 sudo、不寫任何系統路徑**,狀態全放 `$HOME/.cache/zkp-npu` 的 venv,永遠 `exit 0`,結果寫 [`artefacts/npu-dispatch.{log,json}`](./npu-dispatch.json)）。

**這台真機（kernel 6.17.0-29、Ryzen AI MAX+ 395、`1022:17f0` rev 11）的 verdict:`BLOCKED-ENUMERATION`**,graduation = `DRIVER-READY -> COMPILE-READY -> [BLOCKED-ENUMERATION]`。

| 層 | 狀態 | 證據 |
|---|---|---|
| kernel driver | ✅ | `amdxdna` 已載入;`/dev/accel/accel0` 存在,使用者在 `render` group → 不需 root 可開 |
| 韌體 | ✅ | `fw_version=1.0.0.166`（對應 `/lib/firmware/amdnpu/17f0_11/`） |
| authoring（Peano+MLIR-AIE+IRON) | ✅ **COMPILE-READY（no-root）** | `pip install mlir_aie + llvm-aie` 成功:`aie-opt`(LLVM 22.0.0,`3940144`)、Peano `21.0.0` 全就位 → 可在本機編 xclbin |
| XRT runtime + amdxdna shim | ⛔ | 無 `xrt-smi`、`/opt/xilinx` 不存在、無 `libxrt_driver_xdna.so`、`pyxrt` import 失敗 → 無法 enumerate |

**精確 blocker:** 唯一缺口是 user-space 的 XRT `amdxdna` device shim,本環境只有 root 路徑且全數受阻——(a) non-interactive sudo 需密碼;(b) `amd/xdna-driver` 0 個 prebuilt release（無 `.deb` 可 no-root 解出);(c) Ubuntu archive XRT 是 `202210.2.13.466`（2022,pre-NPU,連 `xrt-smi` 都沒有);(d) no-root 的 `mlir_aie` wheel 只有 authoring + `libxrt_coreutil` + compile-side `_xrt`,**無 `pyxrt`、無 shim**。在有 root 的 Strix Halo 上自建 matched DKMS + dev 韌體 + `xrt_plugin .deb` 後,`npu-run.sh` 會自動往下跑 stage 5（編 xclbin → XRT 跑 axpy 抓 PASS/timing → 小 int8 GEMM）。

**對證明路徑影響:零。** 結論與 §0 誠實前提一致:NPU 只動 AI 模型前向,動不到 prover time / proof size / on-chain gas。這是一個有完整證據鏈的 negative result,符合本軌「研究 bullet、可以是 documented blocker」的定位。完整推導:[`reading-notes/path-d-npu-xdna2.md`](../../../reading-notes/path-d-npu-xdna2.md) §7。

| `make npu-run`（opt-in） | ⛔ BLOCKED-ENUMERATION | <1 s | no sudo;`npu-dispatch.json` verdict=`BLOCKED-ENUMERATION` |
```

**(d.3) Optional Demo-B target-table row** (the `## 1.` table near line 32, after
the `make npu-probe` row) — add:

```markdown
| `make npu-run`（opt-in,heavy） | ⛔ BLOCKED-ENUMERATION | <1 s | no-root;authoring COMPILE-READY,缺 root-only XRT amdxdna shim |
```

---

## (e) `docs/amd-strix-halo-acceleration.md` — engine-matrix NPU row + §4

**(e.1) Engine matrix NPU row** (anchor line ~31, currently
`| NPU | **XDNA2** (RyzenAI-npu5) | experimental on Linux only |`) → replace with:

```markdown
| NPU | **XDNA2** (RyzenAI-npu5) | experimental on Linux; `DRIVER-READY -> COMPILE-READY -> [BLOCKED-ENUMERATION]` — kernel+fw+authoring ready, only the root-only XRT `amdxdna` shim is missing (accelerates the AI model, not the proof) |
```

**(e.2) §4 "NPU (XDNA2 / RyzenAI-npu5) → experimental on Linux"** (anchor line ~75)
— append a bullet after the existing two:

```markdown
- **G2 dispatch attempt (2026-06-11, real Strix Halo).** The opt-in
  `make npu-run` (`poc/risc0-cartesi-step-demo/scripts/npu-run.sh`, no sudo, never
  destabilizes) reached **`COMPILE-READY`** no-root (Peano + MLIR-AIE + IRON
  installed; an xclbin *can* be compiled) but **could not enumerate** the NPU:
  the only gap is the user-space XRT **`amdxdna` device shim**, installable only
  with root here (non-interactive sudo blocked; `amd/xdna-driver` has no prebuilt
  release; Ubuntu XRT is the 2022 pre-NPU 2.13). Verdict **`BLOCKED-ENUMERATION`**,
  recorded in `artefacts/npu-dispatch.{log,json}`. Still **never a dependency** —
  this confirms the hardware ceiling without over-claiming. Details:
  `reading-notes/path-d-npu-xdna2.md` §7.
```

---

## Notes / guardrails for the closeout agent

- **Do not** make `npu-run` part of any exit gate or default `make lab-replay`
  path — it is heavy + opt-in. Replay must read committed `npu-dispatch.json`.
- The runner is **idempotent** and **re-runnable**: it reuses the cached venv
  (`have_aie` short-circuits the pip step) and re-derives the verdict from live
  hardware state each time.
- `README.md` workload→engine table (NPU row): mirror (e.1) — note the NPU stays
  `experimental`, now with the precise `BLOCKED-ENUMERATION` (root-only shim) sub-status.
- If a future box gets a root-built XRT + `xrt_plugin(amdxdna)`, re-running
  `make npu-run` should auto-graduate to `ENUMERATED`/`DISPATCH-OK` and the same
  fields/cells will reflect it with no code change.

---

# Track-1 result (todo `npu-xrt`, 2026-06-11) — root-install prep done, blocker SHARPENED

> Appended by the Track-1 parallel agent on the real Strix Halo. This section
> supersedes the **wording** (not the structure) of (c)/(d)/(e) above wherever it
> gives sharper text. The committed `npu-dispatch.json` verdict is **unchanged**
> (`BLOCKED-ENUMERATION`); what changed is that the blocker is now a **one-command
> fix** with a fully prepared root script, plus a precise firmware-protocol finding.

## Verdict reached

- **`BLOCKED-ENUMERATION` — needs the user to run the prepared root installer
  `poc/risc0-cartesi-step-demo/scripts/npu-xrt-install.sh` (or enable passwordless
  sudo and re-run it).** Not `DISPATCH-OK`: this agent runs non-interactively and
  `sudo -n true` fails (password required), so the root install (apt build-deps +
  `dpkg`/`apt` of the XRT base + xdna plugin) could not be performed here.
- **All no-root prep is DONE:** `amd/xdna-driver` is cloned `--recursive` at
  `~/.cache/zkp-npu/xdna-driver` (HEAD = `688712e` "accel/amdxdna: add new
  device/rev IDs" — the commit that adds **rev-11/npu5** support; tag `2.21.75`
  predates it; bundled XRT submodule ~`202610.2.23.0`, `info.json` xrt
  `202620.2.25.23`, os_rel 22.04/24.04 → this box is Ubuntu **24.04 noble**). The
  exact build-dep set was enumerated (boost/ssl/udev/opencl-headers/rapidjson/
  protobuf/uuid/curl/dw/systemd/ffi/ncurses are the missing ones, all apt→root)
  and the full ordered install recipe is captured in the script.
- **A no-root build is NOT possible** (XRT needs `libboost-*`, `libssl-dev`, …
  installed via apt = root); hence the work correctly STOPS at the install step
  and hands off the script.

## Sharper firmware-protocol finding (the only real risk, now pinned to exact versions)

- Loaded **production** firmware on this box: **`1.0.0.166`**
  (`/lib/firmware/amdnpu/17f0_11/npu.sbin.1.0.0.166.zst`, from distro
  linux-firmware; `17f0_11` = device `1022:17f0` rev 11 = **npu5 / Strix Halo**).
- The xdna-driver plugin DEB for `17f0_11` bundles the **dev** firmware
  **`1.1.2.65`** as **`npu.dev.sbin`** (per `tools/info.json` →
  `.../amdnpu/17f0_11/1.7_npu.sbin.1.1.2.65`). Because it is named `npu.dev.sbin`
  (not `npu.sbin`), a normal plugin install is **firmware-safe**: it does NOT
  overwrite the production `npu.sbin` the kernel loads by default — **the working
  `1.0.0.166` stays loaded**, so the install does not break the kernel.
- **Most-likely outcome after install: `DISPATCH-OK`** (XRT shim ↔ in-kernel
  amdxdna ABI on a 6.17 kernel; firmware protocol is negotiated in-kernel and the
  kernel already loaded `1.0.0.166` cleanly). **Only if** `xrt-smi examine` prints
  `Incompatible firmware protocol major X` is the matched dev fw `1.1.2.65`
  (`/usr/lib/firmware/amdnpu/17f0_11/npu.dev.sbin`) the documented fallback — and
  that switch is **gated, not automatic** (`FORCE_DEV_FW=1`), because it changes
  the loaded kernel firmware state. If the user declines that, the sharper blocker
  to record is: *"XRT shim protocol major > firmware 1.0.0.166 protocol; dev fw
  1.1.2.65 available but withheld to protect the working kernel."*

## Files created/edited by Track 1

- **NEW** `poc/risc0-cartesi-step-demo/scripts/npu-xrt-install.sh` — the single
  ready-to-run **root** installer (exact ordered `sudo` steps, commented; clones
  if missing → `amdxdna_deps.sh` → build XRT base → install base → build plugin →
  install plugin → memlock-unlimited drop-in → `source setup.sh` → `xrt-smi
  examine` → `npu-run.sh`; with the gated `FORCE_DEV_FW=1` fallback). Runs as the
  normal user and calls `sudo` only for privileged steps.
- `npu-dispatch.{log,json}` — **unchanged** (still the source-of-truth replay;
  `verdict=BLOCKED-ENUMERATION`). They will auto-graduate to `DISPATCH-OK` the
  moment the installer is run and `npu-run.sh` re-fires (no code change needed).
- This **Track-1 result** section.

## Closeout wire-in deltas (override the wording above where sharper)

**path-d §7 update text** (closeout edits `reading-notes/path-d-npu-xdna2.md` §7):
add a closing paragraph —
> *Track 1 (`npu-xrt`, 2026-06-11) took this from "no root path exists" to "one
> command away". `amd/xdna-driver` (main, rev-11/npu5 support) is cloned and the
> exact firmware-matched build/install recipe is captured in
> `poc/risc0-cartesi-step-demo/scripts/npu-xrt-install.sh`. The shim build needs
> apt-installed deps (root), so on this non-interactive run it stops at the
> install step. Firmware is safe-by-default: the plugin ships dev fw `1.1.2.65` as
> `npu.dev.sbin`, which does not displace the loaded production `1.0.0.166`. After
> a human runs the installer (or enables passwordless sudo), `make npu-run`
> auto-graduates `DRIVER-READY → COMPILE-READY → ENUMERATED → DISPATCH-OK`; the
> only residual risk is a shim↔fw protocol-major mismatch, whose remedy (dev fw
> `1.1.2.65`) is pre-staged but gated to avoid touching the working kernel.*

**Engine-matrix NPU row** (closeout edits `docs/amd-strix-halo-acceleration.md`,
overrides (e.1)) → use:
```markdown
| NPU | **XDNA2** (RyzenAI-npu5) | experimental on Linux; `DRIVER-READY → COMPILE-READY → [BLOCKED-ENUMERATION]` — kernel+fw(1.0.0.166)+authoring ready; the only gap is the root-only XRT `amdxdna` shim, now one command away via `scripts/npu-xrt-install.sh` (firmware-safe: dev fw 1.1.2.65 ships as `npu.dev.sbin`, prod stays loaded). Accelerates the AI model, not the proof |
```

**Notebook-05 live-or-replay cell behavior** (closeout edits
`lab/05_npu_xdna2_probe.ipynb`): keep the cell from (c) **as-is** — it still prints
`BLOCKED-ENUMERATION` from the committed JSON in replay, and in live mode re-runs
`npu-run.sh` (which auto-graduates to `DISPATCH-OK` *iff* the installer has been
run). **Add one markdown line** under it:
> *To enumerate/dispatch on real hardware, first run (with sudo)
> `poc/risc0-cartesi-step-demo/scripts/npu-xrt-install.sh`; then re-run this cell
> live and the verdict graduates to `DISPATCH-OK` automatically.*
No code change to the cell is required.

**IMPLEMENTATION-STATUS text** (closeout edits `docs/IMPLEMENTATION-STATUS.md`,
refines (d.1)/(d.2)): keep verdict `BLOCKED-ENUMERATION` but change the
`blocker`/next-step prose to —
> *NPU (G2): kernel + firmware (`1.0.0.166`) + authoring 全綠;唯缺 root-only XRT
> `amdxdna` shim。Track 1 已把它變成「一行指令」:`amd/xdna-driver`(main,含
> rev-11/npu5)已 clone,firmware-matched 安裝腳本
> `poc/risc0-cartesi-step-demo/scripts/npu-xrt-install.sh` 就緒。需使用者以 sudo
> 執行該腳本(或開 passwordless sudo)後 `make npu-run` 即自動
> graduate 到 `DISPATCH-OK`。Firmware 安全:plugin 帶的 dev fw 1.1.2.65 以
> `npu.dev.sbin` 並存,不覆蓋已載入的 production 1.0.0.166。*

## Guardrails specific to Track 1

- The installer runs as the **normal user** and calls `sudo` per-step (NOT
  `sudo bash …`), so build artefacts stay user-owned and only the privileged
  steps escalate.
- It is **firmware-safe by default**; the dev-firmware switch is opt-in
  (`FORCE_DEV_FW=1`) and reversible (symlink backup + `modprobe -r/​modprobe`).
- Do **not** commit the cloned `~/.cache/zkp-npu/xdna-driver` tree or any built
  `.deb` into the repo — they live outside the repo by design.

---

# Step-1 final result (todo `npu-xrt`, 2026-06-15) — verdict GRADUATED to `DISPATCH-OK`

> **This section is the authoritative closeout.** It SUPERSEDES every
> `BLOCKED-ENUMERATION` claim above. The user ran the prepared root installer
> `poc/risc0-cartesi-step-demo/scripts/npu-xrt-install.sh` on this real Strix
> Halo; the user-space XRT `amdxdna` shim is now installed, the NPU enumerates,
> and `npu-run.sh` dispatched two real kernels onto `/dev/accel0`. Wherever (c),
> (d), (e), and the "Track-1 result" section say the verdict is
> `BLOCKED-ENUMERATION` / "one command away", use **`DISPATCH-OK`** and the text
> below instead. The committed `npu-dispatch.{json,log}` are the replay
> source-of-truth and already carry the graduated verdict.

## Final verdict + numbers (from committed `npu-dispatch.json`, generated `2026-06-15T05:02:53Z`)

- **Verdict:** `DISPATCH-OK`.
- **Graduation:** `DRIVER-READY -> COMPILE-READY -> ENUMERATED -> DISPATCH-OK` (full ladder).
- **Enumeration:** `xrt-smi examine` saw `[0000:c4:00.1] RyzenAI-npu5 / aie2p / 6x8`,
  XRT `2.25.0`, NPU firmware `1.0.0.166` — **no** "Incompatible firmware protocol"
  warning, so the gated `FORCE_DEV_FW=1` dev-fw (`1.1.2.65` / `npu.dev.sbin`) was
  **not** needed and the loaded production firmware was left untouched.
- **Shim installed:** `/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2`; `xrt-smi` at
  `/opt/xilinx/xrt/bin/xrt-smi`.
- **axpy (IRON `vector_scalar_add`):** **PASS** in `5.752 s` wall (23/23 elements
  `Correct output … == …`, `PASS!`).
- **int8 GEMM (512×512×512, `i8`→`i8`, toward BitNet prefill):** **PASS** —
  avg `521.22 µs` / `515.014 GFLOPs`; min `246 µs` / peak `1091.2 GFLOPs`
  (max `3235 µs` / min `82.9785 GFLOPs`); wall `18.570 s`.
- **Authoring (unchanged):** mlir-aie `3940144f8772e14f794022d9a50c1410090d6fc3`,
  Peano `21.0.0.2026052701+9e603b76`.
- **Honesty (unchanged):** the NPU accelerates the **AI model** (BitNet prefill /
  DEAAP embedding), **not** the ZK proof; r0vm STARK + Groth16 wrap stay CPU-only
  on AMD. Now a **demonstrated capability**, still **never a dependency** — zero
  impact on prover time / proof size / on-chain gas.
- **sudo-password hygiene:** the installer prompts for the sudo password on the
  TTY only; it is **never** echoed, written to any artefact/log, or committed.
  `npu-dispatch.{log,json}` contain no credential material (the runner itself uses
  no sudo). Confirmed by grep over the repo.

JSON keys to read in replay: `verdict=DISPATCH-OK`, `graduation`, `enumerated=yes`,
`compile_ready=yes`, `dispatch_ok=yes`, `axpy_result=PASS`, `axpy_wall_s=5.752`,
`int8_gemm_result=PASS`, `int8_gemm_npu_us=521.22`, `int8_gemm_gflops=515.014`,
`int8_gemm_min_us=246`, `int8_gemm_max_gflops=1091.2`, `int8_gemm_wall_s=18.570`,
`xrt_examine_fw=1.0.0.166`, `xdna_shim`, `blocked_reason=""` (empty).

## (path-d §7) — exact closing text for `reading-notes/path-d-npu-xdna2.md` §7

Replace the "Track 1 收尾" closing paragraph (the "從『沒有 root 路徑』推到『一行指令』"
one) — or append after it — with this **Step-1 收尾** paragraph, and flip the trailing
`path-d-dispatch (G2)` HTML comment's `verdict=`/`graduation=`/`blocker=` fields to match:

> **Step-1 收尾（`npu-xrt`，2026-06-15）：verdict 已 graduate 到 `DISPATCH-OK`。**
> 使用者在這台真機上跑了 ready-to-run 的 root 安裝腳本
> [`poc/risc0-cartesi-step-demo/scripts/npu-xrt-install.sh`](../scripts/npu-xrt-install.sh)，從 `amd/xdna-driver`（main，含 rev-11/npu5）自建並安裝了
> firmware-matched 的 **XRT 2.25.0** base/headers + `amdxdna` device shim
> （`/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2`）。`xrt-smi examine` 隨即 enumerate 出
> `RyzenAI-npu5`（`aie2p`、6×8），對上**已載入的 production 韌體 `1.0.0.166`**、
> **沒有** protocol-major 不符 → gated 的 `FORCE_DEV_FW=1`（dev fw `1.1.2.65` /
> `npu.dev.sbin`）**完全用不到**，working kernel 毫髮無傷。`npu-run.sh` 接著在
> `/dev/accel0` 上 dispatch 了兩顆真 kernel：IRON `vector_scalar_add`（**axpy — PASS**，
> 5.752 s）與一顆 512³ **int8 GEMM**（**PASS**，avg 521.22 µs / 515.014 GFLOPs，peak
> ~1.09 TFLOPs int8，朝 BitNet-style prefill）。至此完整 graduation =
> `DRIVER-READY → COMPILE-READY → ENUMERATED → DISPATCH-OK`。**對證明路徑影響仍是零**：
> 這只是把「模型前向」那層 demonstrated 在 NPU 上跑得起來，動不到 prover time /
> proof size / on-chain gas（那三項仍由 32-thread Zen 5 + unified RAM 扛）；NPU 永遠不是
> 依賴。committed `npu-dispatch.{json,log}` 即 replay 真相來源，已帶 `DISPATCH-OK`。

Updated trailing HTML comment (replace the existing `path-d-dispatch (G2)` line):

```html
<!-- path-d-dispatch (G2): runner=poc/risc0-cartesi-step-demo/scripts/npu-run.sh artefacts=poc/risc0-cartesi-step-demo/artefacts/npu-dispatch.{log,json} verdict=DISPATCH-OK graduation="DRIVER-READY -> COMPILE-READY -> ENUMERATED -> DISPATCH-OK" kernel=6.17.0-29 fw=1.0.0.166 xrt=2.25.0 shim=/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2 enumerated="RyzenAI-npu5/aie2p/6x8" axpy=PASS(5.752s) int8_gemm=PASS(avg=521.22us,515.014GFLOPs,peak=1091.2GFLOPs) accelerates=ai-model-not-zk-proof e2e_guarantee=none opt_in=true never_destabilizes=true -->
```

## (engine-matrix) — exact NPU row for `docs/amd-strix-halo-acceleration.md`

Replace the current NPU row (anchor: `| NPU | **XDNA2** (RyzenAI-npu5) |`) with:

```markdown
| NPU | **XDNA2** (RyzenAI-npu5) | experimental on Linux; `DRIVER-READY → COMPILE-READY → ENUMERATED → DISPATCH-OK` — root install of XRT 2.25.0 + the `amdxdna` shim (via `scripts/npu-xrt-install.sh`) enumerated the NPU (`aie2p`, 6×8, fw 1.0.0.166, no protocol mismatch) and ran IRON `axpy` (**PASS**) + a 512³ int8 GEMM (**PASS**, ~521 µs / 515 GFLOPs, peak ~1.09 TFLOPs) on `/dev/accel0`. Accelerates the AI model, not the proof |
```

Also update §4 ("NPU (XDNA2 / RyzenAI-npu5) → experimental on Linux"): change the
"G2 dispatch attempt … could not enumerate … Verdict `BLOCKED-ENUMERATION`" bullet to
report the **`DISPATCH-OK`** outcome — the root installer supplied the shim, the NPU
enumerated cleanly against fw `1.0.0.166`, and both `axpy` and the int8 GEMM PASSed on
`/dev/accel0`; still never a dependency. Keep the `reading-notes/path-d-npu-xdna2.md §7`
pointer.

## (IMPLEMENTATION-STATUS) — exact closeout text for `docs/IMPLEMENTATION-STATUS.md`

**TL;DR table row** (anchor: the NPU probe row) → use:

```markdown
| NPU probe + dispatch（G2） | ✅ probe / ✅ dispatch（`DISPATCH-OK`） | XDNA2 `DRIVER-READY → COMPILE-READY → ENUMERATED → DISPATCH-OK`：kernel+韌體(1.0.0.166)+authoring 全綠,root 裝好 XRT 2.25.0 + `amdxdna` shim 後 enumerate 成功,IRON axpy + 512³ int8 GEMM 皆 PASS;研究用,不在證明路徑 |
```

**Path-D numbered section** — replace the verdict line and the `make npu-run` table
row with the graduated result, and swap the blocker/next-step prose for this closeout:

> *NPU (G2，Step-1 收尾 2026-06-15)：verdict 已從 `BLOCKED-ENUMERATION` graduate 到
> **`DISPATCH-OK`**。使用者以 sudo 跑了 firmware-matched 的安裝腳本
> `poc/risc0-cartesi-step-demo/scripts/npu-xrt-install.sh`,自建並裝上 XRT 2.25.0 base/headers
> + `amdxdna` device shim（`/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2`）。`xrt-smi examine`
> enumerate 出 `RyzenAI-npu5`（`aie2p`、6×8）對上已載入韌體 `1.0.0.166`、無 protocol 不符
> → gated 的 dev fw `1.1.2.65`（`npu.dev.sbin`）未動用,working kernel 不受影響。`npu-run.sh`
> 隨即在 `/dev/accel0` dispatch：IRON axpy **PASS**（5.752 s）+ 512³ int8 GEMM **PASS**
> （avg 521.22 µs / 515.014 GFLOPs,peak ~1.09 TFLOPs）。對 prover time / proof size /
> on-chain gas 影響仍為零——NPU 只動 AI 模型前向,永遠不是依賴。*

`make npu-run` table row (replace the `⛔ BLOCKED-ENUMERATION` rows in both the
Path-D section table and the Demo-B target table):

```markdown
| `make npu-run`（opt-in,heavy） | ✅ DISPATCH-OK | ~25 s | root 已裝 XRT 2.25.0 + `amdxdna` shim;axpy + int8 GEMM 皆 PASS on `/dev/accel0` |
```

## (notebook) — `lab/05_npu_xdna2_probe.ipynb`

Done in Step 1: the narrative cells already read `DISPATCH-OK`, and the live-or-replay
dispatch cell's **cached replay output was refreshed** to the graduated verdict
(`DRIVER-READY -> COMPILE-READY -> ENUMERATED -> DISPATCH-OK`, `dispatch_ok=yes`, empty
`blocked_reason`) so the committed notebook is self-consistent with `npu-dispatch.json`.
The cell code is unchanged (still gated by `lk.live_or_replay` + `requires=[_heavy_npu]`),
so `make lab-replay` never fires `npu-run.sh`.
