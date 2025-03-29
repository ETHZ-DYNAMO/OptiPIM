from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple
from analytical_modeling.generate_random_traces import generate_random_traces
from analytical_modeling.get_trace_perf import get_all_traces_perf

from rich import print
from tqdm import tqdm

RESULT_PATH = "exp_results/fig9"

ITEM: str = ""


def get_cycle(lines: List[str]) -> int:
    pattern = r"memory_system_cycles:\s*(\d+)"
    for line in lines:
        match = re.search(pattern, line)
        if match is not None:
            cycle = int(match.group(1))
            return cycle

    raise RuntimeError(f"Could not find {ITEM} in the output")


def get_steps(lines: List[str]) -> Tuple[int, int]:
    pattern = r"SimTimeSteps:(\d+), TotalSteps:(\d+)"
    for line in lines:
        match = re.search(pattern, line)
        if match is not None:
            sim_time_steps = int(match.group(1))
            total_steps = int(match.group(2))
            return sim_time_steps, total_steps

    raise RuntimeError(f"Could not find SimTimeSteps in the output")


def get_traces(lines: List[str]) -> Tuple[int, int]:
    pattern = pattern = (
        r"\[Ramulator::PIM Trace\] \[info\] After kernel expansion - (\d+) / (\d+) lines\."
    )
    for line in lines:
        match = re.search(pattern, line)
        if match is not None:
            expanded_lines = int(match.group(1))
            total_lines = int(match.group(2))
            return expanded_lines, total_lines

    raise RuntimeError(f"Could not find trace lines in the output")


def dram_dict2param(dram_dict: Dict[str, int]) -> List[str]:
    ret: List[str] = []
    for key, value in dram_dict.items():
        param: str = f"MemorySystem.DRAM.{key}={value}"
        ret.append("--param")
        ret.append(param)

    return ret


def run_dpsim(
    bin: str,
    config_file: str,
    trace_file: str,
    duplication: str,
    analytical_result: float
) -> int:
    config_list: List[str] = [
        bin,
        "--config_file",
        config_file,
        "--param",
        f"Frontend.path={trace_file}",
        "--param",
        f"Frontend.PimCodeGen.alloc_method={duplication}",
    ]
    # param_list: List[str] = dram_dict2param(dram_dict)
    # arg_list: List[str] = config_list + param_list
    arg_list: List[str] = config_list

    # Execute the binary and capture the output
    result = subprocess.run(
        arg_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode == 0:
        output = result.stdout
        lines = output.splitlines()
        cycle = get_cycle(lines)
        sim_time_steps, total_steps = get_steps(lines)
        expanded_lines, total_lines = get_traces(lines)
        print(
            # f"analytical: {int(analytical_result)}, cycle: {cycle}, sim_time_steps: {sim_time_steps}, total_steps: {total_steps}, expanded_lines: {expanded_lines}, total_lines: {total_lines}"
            f"analytical: {int(analytical_result)}, cycle: {cycle}, trace: {trace_file}"
        )

        step_factor = total_steps / sim_time_steps
        line_factor = total_lines / expanded_lines
        return int(cycle * step_factor * line_factor)

    else:
        print(f"Execution failed with the following: args {arg_list}")
        print(result.stdout)
        print(result.stderr)
        return -1


def run_dpsim_trace(
    bin: str,
    config_file: str,
    trace_file: str,
    duplication: str,
    analytical_result: float
) -> int:

    # for debugging
    # trace_files = trace_files[:2]

    cycle = run_dpsim(bin, config_file, trace_file, duplication, analytical_result)

    return cycle


def read_category_file(category_file: str):
    with open(category_file, "r") as f:
        category: Dict[str, Any] = json.load(f)

    param_names: List[str] = category["param_names"]
    subdir2params: Dict[str, List[Any]] = category["subdirs"]

    return param_names, subdir2params

def read_analytical_file(analytical_file: str, device: str):
    trace_files: List[str] = []
    analytical_results: List[float] = []
    if device == "simdram":
        trace_folder = os.path.join(RESULT_PATH, "traces0")
    else:
        trace_folder = os.path.join(RESULT_PATH, "traces1")
    with open(analytical_file, "r") as f:
        for line_no, line in enumerate(f):
            if line_no != 0:
                str_arr = line.strip().split(",")
                trace_files.append(trace_folder + "/" + str_arr[0] + ".txt")
                analytical_results.append(float(str_arr[1]))
    return trace_files, analytical_results

def main(
    bin: str,
    config_file: str,
    analytical_file: str,
    data_file: str,
    duplication: str,
    device: str
) -> None:
    # read analytical file
    trace_files, analytical_results = read_analytical_file(analytical_file, device)

    pool_list: List[Tuple] = []
    for i, trace_file in enumerate(trace_files):
        pool_list.append((bin, config_file, trace_file, duplication, analytical_results[i]))

    # for debugging
    # pool_list = pool_list[:2]
    # layer_list = layer_list[:2]

    with mp.Pool(processes=(os.cpu_count() - 4) ) as pool:
        results = pool.starmap(run_dpsim_trace, pool_list)

    # results = [run_dpsim_folder(*args) for args in tqdm(pool_list)]

    # write csv
    # with open(data_file, "w") as f:
    #     f.write(",".join(param_names) + ",cycle\n")
    #     for layer, result in zip(layer_list, results):
    #         f.write(f"{','.join(layer)},{result}\n")
    with open(data_file, "w") as f:
        f.write("Trace,Analytical,Simulation\n")
        for i, result in enumerate(results):
            f.write("{},{},{}\n".format(trace_files[i], analytical_results[i], result))

if __name__ == "__main__":
    parser = ArgumentParser()
    # paths
    parser.add_argument("--bin", type=str, help="Path to the binary file")
    parser.add_argument(
        "--config_file", type=str, help="Path to the configuration file"
    )
    parser.add_argument(
        "--analytical_file", type=str, help="File with results of analytical models"
    )
    parser.add_argument(
        "--data_file", type=str, help="Dump the restuls to the data file"
    )

    parser.add_argument(
        "--device", type=str, help="HBM-PIM or SIMDRAM"
    )

    # simulator parameters
    parser.add_argument(
        "--duplication", type=str, help="Duplication type", default="new"
    )

    # # DRAM parameters
    # parser.add_argument("--channel", type=int, help="Number of channels", default=2)
    # parser.add_argument("--rank", type=int, help="Number of ranks", default=2)
    # parser.add_argument(
    #     "--pe_per_bankgroup", type=int, help="Number of PEs per bank group", default=2
    # )
    # parser.add_argument(
    #     "--reg_per_pe", type=int, help="Number of registers per PE", default=1
    # )
    # parser.add_argument("--pe_bits", type=int, help="Number of bits per PE", default=16)

    args = parser.parse_args(sys.argv[1:])

    # paths
    bin: str = args.bin
    config_file: str = args.config_file
    analytical_file: str = args.analytical_file
    data_file: str = args.data_file
    device: str = args.device

    # simulator parameters
    duplication: str = args.duplication

    if device == "hbmpim":
        config_file = "simulator/hbmpim_config.yaml"
        analytical_file = os.path.join(RESULT_PATH, "analytical1.csv")
        data_file = "validation_hbmpim.csv"
        duplication = "new"
        device_tag = 1
    else:
        config_file = "simulator/simdram_config.yaml"
        analytical_file = os.path.join(RESULT_PATH, "analytical0.csv")
        data_file = "validation_simdram.csv"
        duplication = "simdram"
        device_tag = 0
    
    trace_path = os.path.join(RESULT_PATH, f"traces{device_tag}")
    arch_file = "data/dram.json"
    generate_random_traces(32, device_tag, 0, 40, trace_path, arch_file)
    get_all_traces_perf(trace_path, arch_file, device_tag)

    # # DRAM parameters
    # channel: int = args.channel
    # rank: int = args.rank
    # pe_per_bankgroup: int = args.pe_per_bankgroup
    # reg_per_pe: int = args.reg_per_pe
    # pe_bits: int = args.pe_bits
    # dram_dict = {
    #     "channel": channel,
    #     "rank": rank,
    #     "pe_per_bankgroup": pe_per_bankgroup,
    #     "reg_per_pe": reg_per_pe,
    #     "pe_bits": pe_bits,
    # }

    main(bin, config_file, analytical_file, data_file, duplication, device)
