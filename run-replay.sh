#!/usr/bin/env bash
#
# run-replay.sh — replay the workshop notebooks in this export, safely.
#
# Adapted from workshop/futuremode-2026/run-festival-replay.sh upstream and
# keeps its three guarantees:
#
#   * REPLAY only     — exports LAB_FORCE_REPLAY=1, so labkit.live_or_replay()
#                       takes the replay branch even on a real Strix Halo.
#   * NEVER --inplace — nbconvert writes executed copies into a mktemp dir.
#   * No source file  — each notebook's sha256 is compared before/after and the
#     is modified       script aborts if a single byte differs.
#
# Usage:
#   bash run-replay.sh            # the 6 curated notebooks (00/01/16/23/24/14)
#   ALL=1 bash run-replay.sh      # every notebook in lab/
#   KEEP_OUTPUT=1 bash run-replay.sh
#
# Env:
#   ALL=1           run every lab/*.ipynb instead of the curated six.
#   KEEP_OUTPUT=1   keep the throwaway output dir instead of deleting it.
#   CELL_TIMEOUT=N  per-cell timeout in seconds (default 600).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$REPO_ROOT/lab"

CURATED=(
  "00_amd_engine_map.ipynb"
  "01_zkml_embedding_ezkl.ipynb"
  "16_verifiable_rag_e2e.ipynb"
  "23_risc0_rocm_stark.ipynb"
  "24_risc0_rocm_bottleneck_lab.ipynb"
  "14_unified_memory_bigmodel.ipynb"
)

if [[ "${ALL:-0}" == "1" ]]; then
  NOTEBOOKS=()
  while IFS= read -r nb; do NOTEBOOKS+=("$(basename "$nb")"); done \
    < <(find "$LAB_DIR" -maxdepth 1 -name '*.ipynb' | sort)
else
  NOTEBOOKS=("${CURATED[@]}")
fi

if [[ -x "$REPO_ROOT/.venv/bin/jupyter" ]]; then
  JUPYTER="$REPO_ROOT/.venv/bin/jupyter"
elif command -v jupyter >/dev/null 2>&1; then
  JUPYTER="$(command -v jupyter)"
else
  echo "[replay] ERROR: 'jupyter' not found. Run: make install" >&2
  exit 1
fi

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

export LAB_FORCE_REPLAY=1
unset LAB_RUN_HEAVY LAB_FORCE_LIVE 2>/dev/null || true

CELL_TIMEOUT="${CELL_TIMEOUT:-600}"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/workshop-replay.XXXXXX")"
KEEP_OUTPUT="${KEEP_OUTPUT:-0}"
cleanup() {
  if [[ "$KEEP_OUTPUT" == "1" ]]; then
    echo "[replay] KEEP_OUTPUT=1 — outputs left in: $OUT_DIR"
  else
    rm -rf "$OUT_DIR"
  fi
}
trap cleanup EXIT

echo "[replay] repo root  : $REPO_ROOT"
echo "[replay] jupyter    : $JUPYTER"
echo "[replay] mode       : REPLAY (LAB_FORCE_REPLAY=1), never --inplace"
echo "[replay] output dir : $OUT_DIR (throwaway)"
echo "[replay] notebooks  : ${#NOTEBOOKS[@]}"
echo

fail=0
for nb in "${NOTEBOOKS[@]}"; do
  src="$LAB_DIR/$nb"
  if [[ ! -f "$src" ]]; then
    echo "[replay] ERROR: missing notebook: $src" >&2
    fail=1
    continue
  fi

  before="$(sha256_of "$src")"
  echo "[replay] >>> $nb"

  if ( cd "$LAB_DIR" && "$JUPYTER" nbconvert \
        --to notebook --execute \
        --output-dir "$OUT_DIR" \
        --ExecutePreprocessor.timeout="$CELL_TIMEOUT" \
        "$nb" ); then
    status="OK"
  else
    status="FAILED"
    fail=1
  fi

  after="$(sha256_of "$src")"
  if [[ "$before" != "$after" ]]; then
    echo "[replay] FATAL: source notebook was modified: $src" >&2
    exit 3
  fi

  echo "[replay] <<< $nb  [$status]  (source sha256 unchanged)"
  echo
done

if [[ "$fail" -ne 0 ]]; then
  echo "[replay] RESULT: one or more notebooks FAILED." >&2
  exit 1
fi
echo "[replay] RESULT: all ${#NOTEBOOKS[@]} notebooks replayed cleanly."
