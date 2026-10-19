#!/usr/bin/env python3
"""Score mapping traces with the authors' analytical model (validation/analytical_modeling).
Usage (from the repo root):
  python3 group12/scripts/score_analytical.py <hbmpim|simdram> trace1 [trace2 ...]
Prints every numeric field of the model's result (finalPerformance and its cost parts)
and writes group12/results/analytical_check_<target>.csv.
"""
import csv
import sys

sys.path.insert(0, "validation")
from analytical_modeling.get_trace_perf import get_trace_perf
from analytical_modeling.analytical import ArchInfo

ARCH = {"hbmpim": ("data/hbm_pim.json", 1), "simdram": ("data/dram.json", 0)}


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
        fields = numeric_fields(get_trace_perf(t, arch, dev))
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
