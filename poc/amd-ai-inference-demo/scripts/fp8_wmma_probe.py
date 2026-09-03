#!/usr/bin/env python3
"""fp8_wmma_probe.py — compile + run the MiniLM ONNX through MIGraphX at a chosen
precision so the wrapping shell script can capture WHAT GPU instructions the
quantized dot/GEMM actually executes.

This is the python half of ``fp8-wmma-probe.sh``. It deliberately reuses the
SAME lowering path as ``src/embed_bench.py``'s ``run_migraphx`` (parse_onnx ->
quantize_{fp8,fp16} -> compile gpu target -> run) so the kernels it dispatches
are exactly the ones Demo F's perf/watt sweep measured (FP8 = no win, 0.43x at
large batch — FP8-INT8-SCOPE.md §7). The open question (scope §5): does
``quantize_fp8`` lower to NATIVE WMMA FP8 on gfx1151, or to convert+upcast+FP32
FMA? This script does not decide that itself; it just (a) runs the forward so an
outer ``rocprofv3 --kernel-trace`` can name the dispatched kernels, and (b) when
asked, prints the MIGraphX MLIR / generated-source dumps and leaves the JIT temp
dir on disk for ISA disassembly.

HONESTY: Demo F weights are random-init, so this is a CAPABILITY / instruction
probe, NOT an accuracy claim; and it accelerates the AI MODEL forward only (the
repo's standing rule — proofs stay CPU-only on AMD).

Usage:
  fp8_wmma_probe.py --quant fp8 --onnx <minilm.onnx> --batch 32 --seq 256 \
      [--runs 3] [--print-shapes]

Env knobs honoured by MIGraphX itself (set by the shell wrapper, not here):
  MIGRAPHX_MLIR_DUMP=<dir>      dump per-gemm MLIR modules
  MIGRAPHX_GPU_DUMP_SRC=1       dump generated HIP source for JIT kernels
  MIGRAPHX_DEBUG_SAVE_TEMP_DIR=1 keep the hiprtc/clang temp dir (code objects)
  MIGRAPHX_TRACE_MLIR / _HIPRTC / _COMPILE  trace the lowering to stderr
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

INPUT_NAMES = ("input_ids", "token_type_ids", "position_ids")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def make_inputs(batch: int, seq_len: int, vocab: int = 30522, seed: int = 42):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, size=(batch, seq_len), dtype=np.int64)
    typ = np.zeros((batch, seq_len), dtype=np.int64)
    pos = np.tile(np.arange(seq_len, dtype=np.int64), (batch, 1))
    return {"input_ids": ids, "token_type_ids": typ, "position_ids": pos}


def make_calibration(migraphx, prog, batch, seq_len, n):
    if n <= 0:
        return []
    names = list(prog.get_parameter_shapes().keys())
    calib = []
    for i in range(n):
        feeds_np = make_inputs(batch, seq_len, seed=1000 + i)
        calib.append({k: migraphx.argument(feeds_np[k]) for k in names
                      if k in feeds_np})
    return calib


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quant", choices=["fp32", "fp16", "fp8"], required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--calib", type=int,
                    default=int(os.environ.get("AI_BENCH_CALIB_BATCHES", "4")))
    ap.add_argument("--print-shapes", action="store_true",
                    help="print parameter shapes then exit (no GPU run)")
    args = ap.parse_args()

    if not args.onnx.is_file():
        log(f"[fp8-probe] ONNX not found: {args.onnx}")
        return 2

    import migraphx  # provided via PYTHONPATH=/opt/rocm-*/lib

    target = migraphx.get_target("gpu")
    b, s = args.batch, args.seq
    dims = {n: [b, s] for n in INPUT_NAMES}
    log(f"[fp8-probe] quant={args.quant} shape={b}x{s} parsing {args.onnx.name}")
    prog = migraphx.parse_onnx(str(args.onnx), map_input_dims=dims)

    if args.quant == "fp16":
        migraphx.quantize_fp16(prog)
    elif args.quant == "fp8":
        calib = make_calibration(migraphx, prog, b, s, args.calib)
        migraphx.quantize_fp8(prog, target, calibration=calib)

    log(f"[fp8-probe] compiling gpu target (gfx1151) ...")
    prog.compile(target)

    pshapes = prog.get_parameter_shapes()
    if args.print_shapes:
        for k, v in pshapes.items():
            log(f"[fp8-probe]   param {k}: {v}")
        return 0

    feeds_np = make_inputs(b, s)
    feeds = {n: migraphx.argument(feeds_np[n]) for n in pshapes.keys()
             if n in feeds_np}
    log(f"[fp8-probe] running {args.runs} forward(s) ...")
    for _ in range(max(1, args.runs)):
        res = prog.run(feeds)
        np.array(res[-1])  # force host sync so the dispatch really happened
    log("[fp8-probe] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
