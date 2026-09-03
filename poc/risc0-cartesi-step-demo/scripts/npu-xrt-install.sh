#!/usr/bin/env bash
# npu-xrt-install.sh — ROOT install of the user-space XRT amdxdna shim for the
# AMD Strix Halo (Ryzen AI MAX+ 395, XDNA2 NPU, 1022:17f0 rev 11 / npu5).
#
#   WHY THIS SCRIPT EXISTS
#   ----------------------
#   Track 1 (todo `npu-xrt`) graduates the NPU from BLOCKED-ENUMERATION to
#   DISPATCH-OK. The kernel side is ALREADY green on this box:
#     * amdxdna kernel driver loaded, /dev/accel/accel0 present
#     * production firmware fw_version=1.0.0.166 loaded
#       (/lib/firmware/amdnpu/17f0_11/npu.sbin.1.0.0.166.zst)
#     * the user is in the `render` group (can open /dev/accel0 without root)
#     * the AUTHORING toolchain (Peano + MLIR-AIE + IRON) is COMPILE-READY,
#       installed no-root under ~/.cache/zkp-npu/venv
#   The ONLY missing piece is the user-space XRT `amdxdna` device shim
#   (libxrt_driver_xdna.so / xrt_plugin-amdxdna, xrt-smi, /opt/xilinx/xrt,
#   pyxrt). Building + installing it needs root, which is why this step is
#   carved out into its own root script: the parallel agent that prepared this
#   runs NON-INTERACTIVELY (no TTY to type a sudo password) and therefore could
#   NOT run the install itself. Everything that did NOT need root is already
#   done: amd/xdna-driver is cloned --recursive at $XDNA_SRC (default below),
#   the firmware-protocol match was analysed, and the exact build deps were
#   enumerated. You (a human with interactive sudo) run THIS to finish.
#
#   HOW TO RUN
#   ----------
#       bash poc/risc0-cartesi-step-demo/scripts/npu-xrt-install.sh
#   Run it as your normal user (NOT `sudo bash ...`): it calls `sudo` only for
#   the specific privileged steps and keeps the heavy compile as your user so
#   build artefacts are not root-owned. You will be prompted for your password.
#
#   FIRMWARE SAFETY (do NOT break the working kernel/firmware)
#   ----------------------------------------------------------
#   This box currently runs PRODUCTION firmware 1.0.0.166. The xrt_plugin DEB
#   for 17f0_11 (npu5) ships a DEV firmware sidecar version 1.1.2.65 named
#   `npu.dev.sbin` (per xdna-driver tools/info.json). `npu.dev.sbin` does NOT
#   overwrite the production `npu.sbin` the kernel loads by default, so a normal
#   install is firmware-safe: the loaded 1.0.0.166 stays loaded. ONLY if
#   `xrt-smi examine` reports `Incompatible firmware protocol major X` would you
#   need to switch to the dev firmware — that step is NOT done automatically
#   here (it changes the working kernel firmware state). See the clearly-gated
#   FORCE_DEV_FW block at the bottom and the matching note in
#   artefacts/npu-dispatch-INTEGRATION-SPEC.md (Track-1 result).
#
#   SOURCE / VERSION MATCH (as analysed 2026-06-11)
#   -----------------------------------------------
#     * amd/xdna-driver @ main (HEAD commit: "accel/amdxdna: add new device/rev
#       IDs" — this is the commit that adds rev-11 support; tag 2.21.75 predates
#       it). Bundled XRT submodule ~ 202610.2.23.0 (tools/info.json xrt
#       202620.2.25.23), os_rel 22.04/24.04 — this box is Ubuntu 24.04 noble.
#     * Do NOT use Ubuntu-archive XRT (202210.2.13.466, 2022, pre-NPU: no
#       xrt-smi, no xdna shim).
#
# This script is idempotent-ish: re-running rebuilds and `apt reinstall`s.
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
XDNA_SRC="${XDNA_SRC:-$HOME/.cache/zkp-npu/xdna-driver}"
XDNA_REMOTE="${XDNA_REMOTE:-https://github.com/amd/xdna-driver.git}"
NJOBS="${NJOBS:-$(nproc)}"

say() { printf '\n\033[1;36m# %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m# ERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Pre-flight (read-only): confirm this really is the driver-ready XDNA2 host.
# ---------------------------------------------------------------------------
say "0/8  pre-flight checks (read-only)"
[[ "$(id -u)" -ne 0 ]] || die "run as your NORMAL user (not root/sudo); the script calls sudo itself."
lsmod 2>/dev/null | grep -q '^amdxdna' || die "amdxdna kernel module not loaded — this is not a driver-ready XDNA2 host."
[[ -e /dev/accel/accel0 ]]            || die "/dev/accel/accel0 missing — kernel side not ready."
FW_LOADED="$(cat /sys/class/accel/accel0/device/fw_version 2>/dev/null || echo unknown)"
echo "#   amdxdna loaded : yes"
echo "#   accel node     : /dev/accel/accel0"
echo "#   loaded fw      : ${FW_LOADED}   (expected 1.0.0.166)"
echo "#   render group   : $(id -nG | tr ' ' '\n' | grep -qx render && echo yes || echo NO)"
id -nG | tr ' ' '\n' | grep -qx render || echo "#   WARNING: not in 'render' group — you may need sudo to open /dev/accel0."

# ---------------------------------------------------------------------------
# Ensure the source tree is present (clone --recursive if missing).
# (The non-interactive prep already cloned it here; this is a safety net.)
# ---------------------------------------------------------------------------
say "1/8  ensure amd/xdna-driver source (recursive) at ${XDNA_SRC}"
if [[ ! -d "${XDNA_SRC}/.git" ]]; then
    mkdir -p "$(dirname "${XDNA_SRC}")"
    git clone --recursive --jobs 8 "${XDNA_REMOTE}" "${XDNA_SRC}"
else
    echo "#   present; refreshing submodules"
    git -C "${XDNA_SRC}" submodule update --init --recursive
fi
git -C "${XDNA_SRC}" log -1 --oneline

# ===========================================================================
# Step 2 (ROOT): install build dependencies.
#   amdxdna_deps.sh -> xrtdeps.sh -> apt-get install of the full XRT build set
#   (libboost-*, libssl-dev, libudev-dev, opencl-headers, rapidjson-dev,
#   protobuf-compiler, uuid-dev, libcurl4-openssl-dev, libdw-dev,
#   libsystemd-dev, libffi-dev, libncurses5-dev, dkms, jq, ...).
# ===========================================================================
say "2/8  [sudo] install build dependencies (apt)"
sudo "${XDNA_SRC}/tools/amdxdna_deps.sh"

# ===========================================================================
# Step 3 (non-root): build the bundled XRT BASE with NPU support.
#   Produces  xrt/build/Release/xrt_*-amd64-base.deb  (Ubuntu 24.04).
# ===========================================================================
say "3/8  build XRT base (xrt/build/build.sh -npu -opt)  [no sudo]"
( cd "${XDNA_SRC}/xrt/build" && ./build.sh -npu -opt -j "${NJOBS}" )

XRT_BASE_DEB="$(ls -t "${XDNA_SRC}"/xrt/build/Release/xrt_*-amd64-base.deb 2>/dev/null | head -1 || true)"
[[ -n "${XRT_BASE_DEB}" ]] || die "XRT base .deb not found under xrt/build/Release/ — check the build log above."
echo "#   XRT base deb: ${XRT_BASE_DEB}"

# ===========================================================================
# Step 4 (ROOT): install the freshly-built XRT base package.
#   `apt reinstall` (vs dpkg -i) so apt resolves runtime deps cleanly.
# ===========================================================================
say "4/8  [sudo] install XRT base package"
sudo apt-get install -y --reinstall "${XRT_BASE_DEB}" || sudo apt-get -y -f install

# Also install the matching -base-dev headers (xrt/xrt_device.h, …). Without
# them the IRON example host testbench (test.cpp) cannot compile, which would
# strand the dispatch at the host-build step even though the NPU enumerates.
XRT_BASE_DEV_DEB="$(ls -t "${XDNA_SRC}"/xrt/build/Release/xrt_*-amd64-base-dev.deb 2>/dev/null | head -1 || true)"
if [[ -n "${XRT_BASE_DEV_DEB}" ]]; then
    say "4b/8 [sudo] install XRT base-dev headers package"
    echo "#   XRT base-dev deb: ${XRT_BASE_DEV_DEB}"
    sudo apt-get install -y --reinstall "${XRT_BASE_DEV_DEB}" || sudo apt-get -y -f install
else
    echo "#   WARNING: xrt_*-base-dev.deb not found — IRON host testbench may fail to compile."
fi

# ===========================================================================
# Step 5 (non-root): build the XDNA plugin (the amdxdna shim) + DKMS driver.
#   `-release` downloads the matched NPU firmware (for 17f0_11 -> dev fw
#   1.1.2.65, npu.dev.sbin) and packages the shim libs + DKMS source.
#   Produces  build/Release/xrt_plugin*amdxdna.deb .
#
#   NOTE: the plugin DEB also ships a DKMS amdxdna.ko. On this box the in-kernel
#   6.17 amdxdna already works; DKMS builds a compatible module for 6.17.0-29.
#   If you want to be conservative and NOT replace the kernel module, you can
#   build the shim-only with `-nokmod` instead (then the running in-kernel
#   driver is used as-is):
#       ( cd "${XDNA_SRC}/build" && ./build.sh -release -nokmod -j "${NJOBS}" )
# ===========================================================================
say "5/8  build XDNA plugin (build/build.sh -release)  [no sudo]"
( cd "${XDNA_SRC}/build" && ./build.sh -release -j "${NJOBS}" )

XRT_PLUGIN_DEB="$(ls -t "${XDNA_SRC}"/build/Release/xrt_plugin*amdxdna.deb 2>/dev/null | head -1 || true)"
[[ -n "${XRT_PLUGIN_DEB}" ]] || die "xrt_plugin*amdxdna.deb not found under build/Release/ — check the build log above."
echo "#   XDNA plugin deb: ${XRT_PLUGIN_DEB}"

# ===========================================================================
# Step 6 (ROOT): install the XDNA plugin package (the shim + xrt-smi NPU bits).
# ===========================================================================
say "6/8  [sudo] install XDNA plugin package"
sudo apt-get install -y --reinstall "${XRT_PLUGIN_DEB}" || sudo apt-get -y -f install

# ===========================================================================
# Step 7 (ROOT): set memlock unlimited (NPU BO allocation needs it) via a
#   limits.d drop-in (survives package upgrades). Takes effect on next login.
# ===========================================================================
say "7/8  [sudo] set memlock unlimited (/etc/security/limits.d/99-amdxdna.conf)"
sudo mkdir -p /etc/security/limits.d
sudo tee /etc/security/limits.d/99-amdxdna.conf >/dev/null <<'EOF'
* soft memlock unlimited
* hard memlock unlimited
EOF
echo "#   (log out/in or reboot for the memlock change to fully apply; current shell uses ulimit -l)"

# ===========================================================================
# Step 8: enumerate, then run the existing stage-5 dispatch.
# ===========================================================================
say "8/8  source XRT + enumerate the NPU"
# shellcheck disable=SC1091
source /opt/xilinx/xrt/setup.sh
EXAMINE="$(xrt-smi examine 2>&1 || true)"
echo "${EXAMINE}" | sed 's/^/#   xrt-smi> /'

if echo "${EXAMINE}" | grep -qiE 'Incompatible firmware protocol major'; then
    cat <<'WARN'

######################################################################
# FIRMWARE-PROTOCOL MISMATCH DETECTED.
#   xrt-smi reports "Incompatible firmware protocol major X".
#   The loaded PRODUCTION firmware (1.0.0.166) speaks an older protocol
#   than this XRT shim expects. The matched DEV firmware (1.1.2.65) was
#   installed by the plugin as:
#       /usr/lib/firmware/amdnpu/17f0_11/npu.dev.sbin
#   Switching to it CHANGES the working kernel firmware state, so it is
#   NOT done automatically (don't-break-the-kernel rule). If you accept
#   the risk and want to try the dev firmware, re-run with FORCE_DEV_FW=1
#   (see the gated block below). Otherwise STOP here: record this as the
#   sharper, exact-version blocker (shim protocol major vs fw 1.0.0.166).
######################################################################
WARN
    if [[ "${FORCE_DEV_FW:-0}" == "1" ]]; then
        say "FORCE_DEV_FW=1 — switching kernel to the dev firmware 1.1.2.65 (RISKY, opt-in)"
        # The kernel driver loads npu.sbin.zst by default. To use the dev fw,
        # point the symlink at it and reload the module. Reverting: restore the
        # original symlink (-> npu.sbin.1.0.0.166.zst) and reload again.
        echo "#   [sudo] backing up + repointing /lib/firmware/amdnpu/17f0_11/npu.sbin.zst"
        sudo cp -a /lib/firmware/amdnpu/17f0_11/npu.sbin.zst /lib/firmware/amdnpu/17f0_11/npu.sbin.zst.bak 2>/dev/null || true
        # (If the dev fw was installed uncompressed as npu.dev.sbin, the kernel
        #  can be pointed at it via the amdxdna `dev_mode`/fw module param on
        #  some builds; otherwise compress + symlink. Consult `modinfo amdxdna`.)
        echo "#   NOTE: exact repoint/reload depends on your amdxdna module params;"
        echo '#         run: modinfo amdxdna | grep -i fw   and follow its fw override.'
        echo "#   To REVERT: sudo mv .../npu.sbin.zst.bak .../npu.sbin.zst && sudo modprobe -r amdxdna && sudo modprobe amdxdna"
    fi
fi

say "run the existing stage-5 dispatch (no sudo) — IRON axpy + int8 GEMM"
bash "${SCRIPT_DIR}/npu-run.sh"

cat <<EOF

######################################################################
# DONE. Read the verdict in:
#   poc/risc0-cartesi-step-demo/artefacts/npu-dispatch.json   (.verdict)
#   poc/risc0-cartesi-step-demo/artefacts/npu-dispatch.log
# Expected on success: verdict=DISPATCH-OK (axpy PASS + int8 GEMM timing).
# To re-run the dispatch later without reinstalling:
#   make npu-run     # (or: bash poc/risc0-cartesi-step-demo/scripts/npu-run.sh)
######################################################################
EOF
