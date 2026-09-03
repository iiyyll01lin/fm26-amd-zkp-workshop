#!/usr/bin/env python3
"""plot-ai-inference.py — visualise the Path F MiniLM iGPU-vs-CPU forward sweep.

Reads artefacts/ai-inference.csv (from embed_bench.py) and renders
artefacts/ai-inference.png with two panels:

  1. forward latency (ms) vs (batch x seq_len), CPU(onnxruntime, all Zen5
     threads) vs iGPU(MIGraphX gfx1151), one bar pair per workload.
  2. speedup (cpu_ms / gpu_ms) per workload with a break-even line at 1.0
     (>1 => the iGPU forward beats the CPU forward).

If the CSV carries the appended perf/watt columns (``power_w``, ``joules``,
``embeddings_per_joule`` — written by embed_bench.py when rocm-smi telemetry is
available), the markdown table includes them and adds a perf-per-watt summary.
These columns are OPTIONAL: an older CSV without them still renders the original
8-column table unchanged (fully backward-compatible).

If matplotlib is unavailable this DEGRADES GRACEFULLY to a markdown table at
artefacts/ai-inference.md and never crashes.

Usage:
    python3 scripts/plot-ai-inference.py [path/to/ai-inference.csv]
    AI_INFERENCE_CSV=/some/ai-inference.csv python3 scripts/plot-ai-inference.py

HONESTY: this is the iGPU (Radeon 8060S / gfx1151) accelerating the AI MODEL
forward pass (MiniLM embedding), NOT a ZK proof. EZKL Halo2 and the RISC0 r0vm
STARK proving stay CPU-only on AMD.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_ROOT = SCRIPT_DIR.parent
ART_DIR = DEMO_ROOT / "artefacts"
DEFAULT_CSV = ART_DIR / "ai-inference.csv"
PNG_PATH = ART_DIR / "ai-inference.png"
MD_PATH = ART_DIR / "ai-inference.md"

BASE_COLS = ["backend", "batch", "seq_len", "fwd_ms", "tokens_per_s",
             "embeddings_per_s", "device", "cpu_threads"]
# Appended, OPTIONAL per-shape perf/watt columns (empty/absent on older CSVs).
ENERGY_COLS = ["power_w", "joules", "embeddings_per_joule"]
COLS = BASE_COLS + ENERGY_COLS


def resolve_csv() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    env = os.environ.get("AI_INFERENCE_CSV")
    return Path(env).expanduser() if env else DEFAULT_CSV


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append({
                "backend": (raw.get("backend") or "").strip(),
                "batch": to_int(raw.get("batch")),
                "seq_len": to_int(raw.get("seq_len")),
                "fwd_ms": to_float(raw.get("fwd_ms")),
                "tokens_per_s": to_float(raw.get("tokens_per_s")),
                "embeddings_per_s": to_float(raw.get("embeddings_per_s")),
                "device": (raw.get("device") or "").strip(),
                "cpu_threads": (raw.get("cpu_threads") or "").strip(),
                # OPTIONAL perf/watt columns — None when absent/empty.
                "power_w": to_float(raw.get("power_w")),
                "joules": to_float(raw.get("joules")),
                "embeddings_per_joule": to_float(raw.get("embeddings_per_joule")),
            })
    return rows


def has_energy(rows) -> bool:
    """True if any row carries a measured perf/watt value (else old-schema CSV)."""
    return any(r.get(c) is not None for r in rows for c in ENERGY_COLS)


def perfwatt_table(rows):
    """Rows with measured embeddings_per_joule -> (backend, b, s, power_w, epj)."""
    out = []
    for r in rows:
        if r.get("embeddings_per_joule") is not None:
            out.append((r["backend"], r["batch"], r["seq_len"],
                        r.get("power_w"), r["embeddings_per_joule"]))
    return out


def speedup_table(rows):
    """Pair cpu/rocm rows by (batch, seq_len) -> list of (b, s, cpu, gpu, spd)."""
    cpu = {(r["batch"], r["seq_len"]): r["fwd_ms"]
           for r in rows if r["backend"] == "cpu" and r["fwd_ms"]}
    gpu = {(r["batch"], r["seq_len"]): r["fwd_ms"]
           for r in rows if r["backend"] == "rocm" and r["fwd_ms"]}
    out = []
    for key in sorted(set(cpu) & set(gpu)):
        b, s = key
        out.append((b, s, cpu[key], gpu[key], cpu[key] / gpu[key]))
    return out


def write_markdown(rows, csv_path: Path, reason: str) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    table_cols = COLS if has_energy(rows) else BASE_COLS
    lines = [
        "# Path F — MiniLM all-MiniLM-L6-v2 forward: iGPU vs CPU",
        "",
        f"_Source: `{csv_path}` — {reason}._",
        "",
        "_iGPU = AMD Radeon 8060S (gfx1151) via MIGraphX (ROCm 7.2.3); CPU = "
        "Zen 5 via onnxruntime (all threads), SAME ONNX graph. This accelerates "
        "the AI MODEL forward pass, NOT any ZK proof (EZKL/RISC0 proving stays "
        "CPU-only on AMD)._",
        "",
        "| " + " | ".join(table_cols) + " |",
        "| " + " | ".join(["---"] * len(table_cols)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(
            "" if r.get(c) is None else str(r.get(c)) for c in table_cols) + " |")

    spd = speedup_table(rows)
    if spd:
        lines += [
            "",
            "## Speedup (cpu_ms / gpu_ms; >1 => iGPU forward wins)",
            "",
            "| batch | seq_len | cpu_ms | gpu_ms | speedup |",
            "| --- | --- | --- | --- | --- |",
        ]
        for b, s, c, g, x in spd:
            lines.append(f"| {b} | {s} | {c:.2f} | {g:.2f} | {x:.2f}x |")

    pw = perfwatt_table(rows)
    if pw:
        lines += [
            "",
            "## Perf-per-watt (embeddings per joule; higher = better)",
            "",
            "_`power_w` is the **whole-SoC package** draw (rocm-smi) measured over "
            "a short sustained window per shape — the iGPU shares the LPDDR5X "
            "budget with the Zen 5 cores, so this is a GPU-side / whole-package "
            "efficiency view, NOT a clean per-engine power split._",
            "",
            "| backend | batch | seq_len | power_w | embeddings_per_joule |",
            "| --- | --- | --- | --- | --- |",
        ]
        for backend, b, s, power, epj in pw:
            pw_s = "" if power is None else f"{power:.1f}"
            lines.append(f"| {backend} | {b} | {s} | {pw_s} | {epj:.2f} |")
    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"[plot-ai] wrote markdown table -> {MD_PATH}")


def plot(rows, csv_path: Path) -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot-ai] matplotlib unavailable ({exc!r}); markdown only.")
        write_markdown(rows, csv_path, "matplotlib unavailable")
        return 0

    write_markdown(rows, csv_path, "rendered alongside PNG")
    spd = speedup_table(rows)
    if not spd:
        print("[plot-ai] no paired cpu/rocm cells; markdown table only.")
        return 0

    dev = next((r["device"] for r in rows if r["backend"] == "rocm" and r["device"]),
               "gfx1151")
    threads = next((r["cpu_threads"] for r in rows
                    if r["backend"] == "cpu" and r["cpu_threads"]), "?")
    labels = [f"b{b}\xb7s{s}" for b, s, *_ in spd]
    cpu_ms = [c for _, _, c, _, _ in spd]
    gpu_ms = [g for _, _, _, g, _ in spd]
    speed = [x for *_, x in spd]

    import numpy as np
    x = np.arange(len(labels))
    w = 0.38

    fig, (ax_t, ax_sp) = plt.subplots(1, 2, figsize=(14, 5))

    ax_t.bar(x - w / 2, cpu_ms, w, label=f"CPU {threads}t (onnxruntime)",
             color="tab:orange")
    ax_t.bar(x + w / 2, gpu_ms, w, label=f"iGPU {dev} (MIGraphX)",
             color="tab:blue")
    ax_t.set_title("MiniLM forward latency (lower = better)")
    ax_t.set_ylabel("forward wall time (ms)")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_t.grid(True, axis="y", ls=":", alpha=0.5)
    ax_t.legend(fontsize=9)

    ax_sp.plot(x, speed, marker="o", color="tab:green")
    ax_sp.axhline(1.0, color="gray", ls="--", alpha=0.7)
    ax_sp.text(0, 1.02, "break-even (iGPU == CPU)", fontsize=8, color="gray")
    ax_sp.set_title("Speedup = CPU / iGPU  (>1 => iGPU forward wins)")
    ax_sp.set_ylabel("speedup x")
    ax_sp.set_xticks(x)
    ax_sp.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_sp.set_ylim(bottom=0)
    ax_sp.grid(True, axis="y", ls=":", alpha=0.5)

    fig.suptitle(
        f"Path F — all-MiniLM-L6-v2 forward on AMD {dev} (MIGraphX/ROCm) vs "
        f"{threads}-thread Zen 5 — iGPU accelerates the AI MODEL, not the proof",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=120)
    print(f"[plot-ai] wrote plot -> {PNG_PATH}")
    return 0


def main() -> int:
    csv_path = resolve_csv()
    if not csv_path.is_file():
        print(f"[plot-ai] CSV not found: {csv_path}\n"
              "  run the sweep first:  bash scripts/run-all.sh", file=sys.stderr)
        return 1
    rows = load_rows(csv_path)
    if not rows:
        print(f"[plot-ai] CSV has no data rows: {csv_path}", file=sys.stderr)
        return 1
    return plot(rows, csv_path)


if __name__ == "__main__":
    raise SystemExit(main())
