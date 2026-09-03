#!/usr/bin/env bash
# reclone-fork.sh — recreate the pinned risc0 fork checkout.
# Pin: tag v2.3.2 == commit 218e3bc4a8ffcd203a9cd4e46f921bf60aa7e2bd
# (matches the installed r0vm 2.3.2 / cargo-risczero 2.3.2).
# Destination: argv[1], RISC0_FORK_PATH, or vendor/risc0 (in that order).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${1:-${RISC0_FORK_PATH:-${POC_ROOT}/vendor/risc0}}"
PIN_TAG="v2.3.2"
PIN_COMMIT="218e3bc4a8ffcd203a9cd4e46f921bf60aa7e2bd"

if [[ -e "${DEST}" || -L "${DEST}" ]]; then
    echo "clone destination already exists at ${DEST}; refusing to overwrite it." >&2
    exit 73
fi

mkdir -p "${POC_ROOT}/vendor"
echo "==> cloning risc0 @ ${PIN_TAG} (${PIN_COMMIT}) -> ${DEST}"
git clone --depth 1 --branch "${PIN_TAG}" https://github.com/risc0/risc0 "${DEST}"

pushd "${DEST}" >/dev/null
GOT="$(git rev-parse HEAD)"
popd >/dev/null
if [[ "${GOT}" != "${PIN_COMMIT}" ]]; then
    echo "ERROR: HEAD ${GOT} != pinned ${PIN_COMMIT}" >&2
    exit 1
fi
printf '%s\n' "${PIN_COMMIT}" >"${DEST}/.risc0-rocm-pin"
# Materialize Git LFS content (e.g. groth16_proof/groth16/stark_verify.circom,
# risc0/circuit/recursion/src/recursion_zkr.zip) before stripping .git so the
# vendored copy carries real files rather than LFS pointer stubs.
if command -v git-lfs >/dev/null 2>&1; then
    pull_rc=0
    ( cd "${DEST}" && git lfs install --local >/dev/null 2>&1 && git lfs pull ) || pull_rc=$?
    # `git lfs pull` has been seen to exit 0 on GitHub-hosted runners while leaving
    # stark_verify.circom a pointer, so the artefact decides, not the exit code.
    fetch_rc="not attempted"
    if grep -rlq "git-lfs.github.com/spec" "${DEST}/groth16_proof" 2>/dev/null; then
        echo "==> lfs pull exited ${pull_rc} but left pointers; retrying with fetch + checkout" >&2
        fetch_rc=0
        ( cd "${DEST}" && git lfs fetch --all && git lfs checkout ) || fetch_rc=$?
    fi
    # Last resort, and the only step that does not depend on .gitattributes: the
    # pointer carries its own oid and size, so smudge it directly and accept the
    # result only when the bytes hash to what the pointer says they should.
    # Upstream's .gitattributes still names compact_proof/groth16/stark_verify.circom
    # while the file lives at groth16_proof/..., which is why an attribute-driven
    # pull can decide there is nothing to do and exit 0.
    stragglers="$(grep -rl "git-lfs.github.com/spec" "${DEST}/groth16_proof" 2>/dev/null || true)"
    if [[ -n "${stragglers}" ]]; then
        while IFS= read -r f; do
            want_oid="$(sed -n 's/^oid sha256://p' "$f")"
            want_size="$(sed -n 's/^size //p' "$f")"
            ( cd "${DEST}" && git lfs smudge < "$f" > "$f.lfstmp" ) || true
            got_size="$(stat -c%s "$f.lfstmp" 2>/dev/null || echo 0)"
            got_oid="$(sha256sum "$f.lfstmp" 2>/dev/null | cut -d' ' -f1)"
            if [[ "${got_oid}" == "${want_oid}" && "${got_size}" == "${want_size}" ]]; then
                mv "$f.lfstmp" "$f"
                echo "==> smudged ${f#"${DEST}"/} directly: ${got_size} bytes, sha256 matches the pointer" >&2
            else
                rm -f "$f.lfstmp"
                echo "==> smudge of ${f#"${DEST}"/} produced ${got_size} bytes / ${got_oid:0:16}, wanted ${want_size} / ${want_oid:0:16}" >&2
            fi
        done <<<"${stragglers}"
    fi
    remaining="$(grep -rl "git-lfs.github.com/spec" "${DEST}/groth16_proof" 2>/dev/null || true)"
    if [[ -n "${remaining}" ]]; then
        echo "ERROR: Git LFS pointers remain after pull and after fetch+checkout." >&2
        echo "       git lfs pull exit=${pull_rc}, fetch+checkout exit=${fetch_rc}" >&2
        echo "       still pointers:" >&2
        while IFS= read -r f; do
            echo "         ${f#"${DEST}"/} ($(stat -c%s "$f") bytes)" >&2
        done <<<"${remaining}"
        echo "       $(git lfs version)" >&2
        ( cd "${DEST}" && git lfs env 2>/dev/null | grep -iE '^endpoint' | head -2 | sed 's/^/       /' ) >&2
        exit 1
    fi
else
    echo "ERROR: git-lfs not installed; stark_verify.circom would remain an LFS pointer." >&2
    echo "       install git-lfs (https://git-lfs.com) and re-run reclone-fork.sh." >&2
    exit 1
fi
# Strip .git by default so the default destination remains a plain vendored
# copy. Clean-room callers may set RISC0_KEEP_GIT=1 for provenance capture.
if [[ "${RISC0_KEEP_GIT:-0}" != "1" ]]; then
    rm -rf "${DEST}/.git"
fi
echo "==> done. Next: bash rocm-port/apply-overlay.sh"
