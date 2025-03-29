from __future__ import annotations
from .parser_trace import TraceParser
from .analytical import cal_performance_conv2D, cal_performance_FC, ArchInfo
from pathlib import Path
from typing import List, Tuple
import csv
import os

RESULT_PATH = "exp_results/fig9"

def get_trace_perf(trace_file: str, arch_info: ArchInfo, device_type: int):
    parser = TraceParser(trace_file)
    layer_type = parser.operation_type
    loop_bounds = parser.get_loop_bounds()

    if layer_type == "conv2d":
        trans_coeffs = parser.get_trans_coeff()
        stride: int = parser.dilation_stride[0]
        dilation: int = parser.dilation_stride[2]
        return cal_performance_conv2D(arch_info, loop_bounds, trans_coeffs, stride, dilation, device_type)
    elif layer_type == "gemm":
        return cal_performance_FC(arch_info, loop_bounds, device_type)
    else:
        raise ValueError("Unsupported layer type")

def iterate_traces(trace_dir: str, arch_file: str, device_type: int):
    arch_info = ArchInfo(arch_file)

    results: List[Tuple[str, float]] = []

    # Specify the root folder of your directory tree
    root = Path(trace_dir)

    # Recursively find all .txt files under the root folder
    for txt_file in root.rglob('*.txt'):
        try:
            # Read the content of the file
            perf: float = get_trace_perf(str(txt_file), arch_info, device_type).finalPerformance
            
            relative_path = txt_file.relative_to(root)
            label = relative_path.with_suffix("")
            results.append((str(label), perf))
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
            exit()

    return results

def get_all_traces_perf(trace_file, arch_file, device_type):
    perf_results = iterate_traces(trace_file, arch_file, device_type)

    # Dump the data to a CSV file
    csv_path = os.path.join(RESULT_PATH, f"analytical{device_type}.csv")
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write an optional header
        writer.writerow(["Label", "FinalPerformance"])
        # Write the rows
        writer.writerows(perf_results)    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_file", type=str, required=True)
    parser.add_argument("--arch_file", type=str, required=True)
    parser.add_argument("--device_type", type=int, default=0)
    args = parser.parse_args()

    trace_file: str = args.trace_file
    arch_file: str = args.arch_file
    device_type: int = args.device_type
    get_all_traces_perf(trace_file, arch_file, device_type)