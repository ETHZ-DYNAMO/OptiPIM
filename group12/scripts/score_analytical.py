#!/usr/bin/env python3
"""Score mapping traces with the authors' analytical model (validation/analytical_modeling).
Usage (from the repo root):
  python3 group12/scripts/score_analytical.py <hbmpim|simdram> trace1 [trace2 ...]
pim-opt writes the DilationStride line as dilation,dilation,stride,stride, while the authors' own
baseline traces (and get_trace_perf.py) use stride,stride,dilation,dilation. Traces whose file name
starts with "optipim" are therefore read dilation-first. This only matters for strided convs (C1).
Prints every numeric field of the model's result (finalPerformance and its cost parts)
and writes group12/results/analytical_check_<target>.csv.
"""
import csv
import sys

sys.path.insert(0, "validation")
import os
from analytical_modeling.parser_trace import TraceParser
from analytical_modeling.analytical import cal_performance_conv2D, cal_performance_FC, ArchInfo

ARCH = {"hbmpim": ("data/hbm_pim.json", 1), "simdram": ("data/dram.json", 0)}


def score(trace, arch, dev):
    p = TraceParser(trace)
    if p.operation_type == "conv2d":
        ds = p.dilation_stride
        if os.path.basename(trace).startswith("optipim"):
            dilation, stride = ds[0], ds[2]
        else:
            stride, dilation = ds[0], ds[2]
        print(f"   DilationStride={list(ds)} -> stride={stride} dilation={dilation}")
        return cal_performance_conv2D(arch, p.get_loop_bounds(), p.get_trans_coeff(), stride, dilation, dev)
    return cal_performance_FC(arch, p.get_loop_bounds(), dev)


def numeric_fields(r):
    try:
        items = vars(r).items()
    except TypeError:
        items = ((k, getattr(r, k)) for k in dir(r) if not k.startswith("_"))
    return {k: v for k, v in items if isinstance(v, (int, float)) and not isinstance(v, bool)}


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ARCH:
        sys.exit(__doc__)
    target = sys.argv[1]
    arch_file, dev = ARCH[target]
    arch = ArchInfo(arch_file)
    rows = []
    for t in sys.argv[2:]:
        fields = numeric_fields(score(t, arch, dev))
        print(t)
        print("  ", fields)
        rows.append({"trace": t, "target": target, **fields})
    cols = ["trace", "target"] + sorted({k for r in rows for k in r} - {"trace", "target"})
    out = f"group12/results/analytical_check_{target}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


main()
