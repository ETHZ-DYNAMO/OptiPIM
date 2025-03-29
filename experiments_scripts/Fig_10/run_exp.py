from multiprocessing import Pool
import subprocess
from tqdm import tqdm
import re
import csv

import os
import subprocess
import time

from parser_trace import *

# Constant Definition
target_device_idx_name_map = {
    0 : "PUM",
    1 : "NBP"
}

storage_method_idx_name_map = {
    0 : "No_output",
    2 : "All"
}

estimation_method_idx_name_map = {
    0 : "our_method",
    1 : "COSA"
}

class OpPerf:
    def __init__(self, 
                 opType,
                 numMulCol = None,
                 numOutCol = None,
                 numFilCol = None,
                 numInCol = None,
                 numRedOutCol = None,
                 numRedCol = None,
                 numColPE = None,
                 numOutPE = None,
                 numInPE = None,
                 numPESys = None,
                 numOutSys = None,
                 numInSys = None):
        # A string store the loop type
        self.opType = opType
        
        # Number of multiplications in a column
        self.numMulCol = numMulCol
        
        # Number of Outputs in a column
        self.numOutCol = numOutCol
        
        # Number of Filters in a column
        self.numFilCol = numFilCol
        
        # Number of Inputs in a column
        self.numInCol = numInCol
        
        # Number of reductions for an output in a column
        self.numRedOutCol = numRedOutCol
        
        # Number of total reductions in a column
        self.numRedCol = numRedCol
        
        # Number of Columns in a PE
        self.numColPE = numColPE
        
        # Number of Outputs per PE
        self.numOutPE = numOutPE
        
        # Number of inputs per PE
        self.numInPE = numInPE
        
        # Number of PEs used in the system
        self.numPESys = numPESys
        
        # Number of output transmissions in the system
        self.numOutSys = numOutSys
        
        # Number of input loading needed in the system
        self.numInSys = numInSys
        
        # Output transmission cost
        self.outTransCost = 0
        
        # Input loading Cost
        self.inLoadingCost = 0
        
        # Filter loading cost for HBM-PIM
        self.numFilPELoading = 0
        
        # Final Overall Performance
        self.finalPerformance = 0
    
    def print_variables(self):
        for key, value in vars(self).items():
            print(f"{key}: {value}")

## Result folder structure

# Environment Setup
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_PATH + "/../.."
NN_MODELS_PATH = f"{PROJECT_ROOT}/nn_models"
MLIR_BIN_PATH = f"{PROJECT_ROOT}/build/bin"


nn_models_list = ["alexnet", "resnet50", "resnet152", "unet", "vgg16", "bert"]
num_banks_list = ["32banks"]
target_device_list = [0, 1]
storage_method_list = [0]
estimation_method_list = [0, 1]

def run_gurobi(args):
    gurobi_cmd, gurobi_workdir = args
    with open(f"{gurobi_workdir}/run_gurobi.log", "w") as output_f:
        try:
            subprocess.run(gurobi_cmd, shell=True, check=True, timeout=500, cwd=gurobi_workdir, stdout=output_f)
            return f"Success: {gurobi_workdir}"
        except subprocess.CalledProcessError as e:
            return f"Failed: {gurobi_workdir}, Error: {str(e)}"


def main_optipim():
    print("#######################################################################")
    print("#####################   OPTIPIM EXPERIMENT Fig10 ######################")
    print("#######################################################################")
    print()
    print("[WARNING] !! CHECK PARAMETERS !!")
    
    gurobi_cmds = []
    gurobi_folder = []
    
    for device_type in target_device_list:
        for nn_model in nn_models_list:
            for num_banks in num_banks_list:
                for sel_storage in storage_method_list:
                    for sel_estimation in estimation_method_list:
                        # nn_model source path
                        model_path = f"{PROJECT_ROOT}/nn_models/{nn_model}/single_layers/{num_banks}"
                        
                        # Get all the layers in the folder
                        layers_to_run = [file for file in os.listdir(model_path)]
                        
                        for sel_layer in layers_to_run:
                            # Create the corresponding folder
                            folder_path = f"{PROJECT_ROOT}/exp_results/fig10/{target_device_idx_name_map[device_type]}/{nn_model}/{num_banks}/{storage_method_idx_name_map[sel_storage]}/{estimation_method_idx_name_map[sel_estimation]}"
                            folder_path = os.path.join(folder_path, sel_layer)
                            

                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)
                                
                            # Get test path
                            nn_layer = os.path.join(model_path, sel_layer)
                            
                            # Check the existence of the result file
                            if not os.path.exists(os.path.join(folder_path, "gurobi_output.txt")):
                                gurobi_cmd = f"{MLIR_BIN_PATH}/pim-opt --data-layout-pass \
                                                --trace-output-path {folder_path}/gurobi_output.txt \
                                                --model-debug-file {folder_path}/milp.lp \
                                                --trans-coeff-method 4 \
                                                --target-device-type {device_type} \
                                                --number-counting-method {sel_estimation} \
                                                --obj-method 0 \
                                                --storage-method {sel_storage} \
                                                --config-arch-path {PROJECT_ROOT}/data/hbm_pim.json \
                                                --config-knobs-path {PROJECT_ROOT}/data/knobs.json \
                                                {nn_layer}"
                                                
                                # Store the command
                                with open(os.path.join(folder_path, "gurobi_cmd.log"), "w") as f:
                                    f.write(gurobi_cmd)
                                
                                gurobi_cmds.append(gurobi_cmd)
                                gurobi_folder.append(folder_path)
    
    assert len(gurobi_cmds) == len(gurobi_folder)
    gurobi_tasks = [(cmd, workdir) for cmd, workdir in zip(gurobi_cmds, gurobi_folder)]
                        
    print(f"[INFO] Total gurobi tasks: {len(gurobi_tasks)}")
    print(f"[INFO] Run Gurobi")
    
    print(f"[INFO] example gurobi work dir", gurobi_folder[0])
    
    with Pool(processes=os.cpu_count() // 4) as pool:
        for _ in tqdm(pool.imap_unordered(run_gurobi, gurobi_tasks), total=len(gurobi_tasks)):
            pass    

def main_cosa():
    print()
    print("[INFO] Run Experiments for COSA!")
    
    gurobi_cmds = []
    gurobi_folder = []
    
    for device_type in target_device_list:
        for nn_model in nn_models_list:
            for num_banks in num_banks_list:
                for sel_storage in storage_method_list:
                    
                    # nn_model source path
                    model_path = f"{PROJECT_ROOT}/nn_models/{nn_model}/single_layers/{num_banks}"
                    
                    # Get all the layers in the folder
                    layers_to_run = [file for file in os.listdir(model_path)]
                    
                    for sel_layer in layers_to_run:
                        # Create the corresponding folder
                        folder_path = f"{PROJECT_ROOT}/exp_results/fig10/{target_device_idx_name_map[device_type]}/{nn_model}/cosa/{num_banks}/{storage_method_idx_name_map[sel_storage]}"
                        folder_path = os.path.join(folder_path, sel_layer)
                        

                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                            
                        # Get test path
                        nn_layer = os.path.join(model_path, sel_layer)
                        
                        # Check the existence of the result file
                        if not os.path.exists(os.path.join(folder_path, "gurobi_output.txt")):
                            gurobi_cmd = f"{MLIR_BIN_PATH}/pim-opt --data-layout-pass \
                                            --trace-output-path {folder_path}/gurobi_output.txt \
                                            --model-debug-file {folder_path}/milp.lp \
                                            --trans-coeff-method 4 \
                                            --target-device-type {device_type} \
                                            --number-counting-method 1 \
                                            --obj-method 1 \
                                            --storage-method {sel_storage} \
                                            --config-arch-path {PROJECT_ROOT}/data/hbm_pim.json \
                                            --config-knobs-path {PROJECT_ROOT}/data/knobs.json \
                                            {nn_layer}"
                                            
                            # Store the command
                            with open(os.path.join(folder_path, "gurobi_cmd.log"), "w") as f:
                                f.write(gurobi_cmd)
                            
                            gurobi_cmds.append(gurobi_cmd)
                            gurobi_folder.append(folder_path)
    
    assert len(gurobi_cmds) == len(gurobi_folder)
    gurobi_tasks = [(cmd, workdir) for cmd, workdir in zip(gurobi_cmds, gurobi_folder)]
                        
    print(f"[INFO] Total gurobi tasks: {len(gurobi_tasks)}")
    print(f"[INFO] Run Gurobi")
    
    print(f"[INFO] example gurobi work dir", gurobi_folder[0])
    
    with Pool(processes=os.cpu_count() // 4) as pool:
        for _ in tqdm(pool.imap_unordered(run_gurobi, gurobi_tasks), total=len(gurobi_tasks)):
            pass
        
def get_first_line(file_path):
    try:
        return open(file_path).readline().strip().lower()
    except(FileNotFoundError, StopIteration):
        print(f"File {file_path} not found or is empty")
        return "Error"
        
def main_baseline():
    print()
    print("[INFO] Run Experiments for Baseline Heuristics!")
    
    baseline_mapping_root = f"{PROJECT_ROOT}/baseline_mappings"
    result_folder = f"{PROJECT_ROOT}/exp_results/fig10/baseline"
    
    heuristic_list = ["heuristic_1", "heuristic_2"]
    device_folder_list = ["simdram_baseline", "hbm_baseline"]
    baseline_nn_models_list = ["Alexnet", "ResNet50", "resnet152", "unet", "vgg16", "bert"]
    baseline_num_banks = ["32_banks"]
    
    # Read the arch info
    path_to_arch_file = f"{PROJECT_ROOT}/data/hbm_pim.json"
    arch_info = ArchInfo(path_to_arch_file)
    
    # Iterate over all trace files in the baseline
    for sel_heuristic in heuristic_list:
        for sel_device_folder in device_folder_list:
            sel_device = 0
            # Check the selected device type
            if (sel_device_folder == "simdram_baseline"):
                sel_device = device_type_idx["PUM"]
            elif (sel_device_folder == "hbm_baseline"):
                sel_device = device_type_idx["NBP"]

            trace_file_folder_base = os.path.join(baseline_mapping_root, sel_heuristic) + f"/{sel_device_folder}"
            
            trace_file_folders = [folder for folder in os.listdir(trace_file_folder_base) if os.path.isdir(os.path.join(trace_file_folder_base,folder))]
            
            # Iterate over all folders in the selected device folder
            for sel_trace_folder in trace_file_folders:
                folder_path = os.path.join(trace_file_folder_base, sel_trace_folder)
                file_path = os.path.join(folder_path, "0.trace")
                
                # Parse the folder name
                name_components = sel_trace_folder.split("-")
                
                num_banks = name_components[1]
                nn_model_name = name_components[2]
                layer_idx = name_components[-1]
                
                if (num_banks in baseline_num_banks and nn_model_name in baseline_nn_models_list):
                    # Check the layer type
                    layer_type = get_first_line(file_path)
                    
                    # Result folder
                    result_folder_base = os.path.join(result_folder, sel_heuristic)
                    result_folder_path = f"{result_folder_base}/{nn_model_name}/{sel_device_folder}/{num_banks}/{layer_idx}"
                    result_file_path = os.path.join(result_folder_path, "perf.log")
                    
                    if not os.path.exists(result_folder_path):
                        os.makedirs(result_folder_path)
                        
                    file_parser = TraceParser(file_path)
                    
                    if (layer_type == "conv2d"):
                        # Conv2D layer
                        tmp_loop_bounds = file_parser.get_loop_bounds()
                        tmp_trans_coeffs = file_parser.get_trans_coeff()
                        stride = file_parser.dilation_stride[2]
                        dilation = file_parser.dilation_stride[0]
                        
                        perf_ins = cal_performance_conv2D(arch_info, tmp_loop_bounds, tmp_trans_coeffs, stride, dilation, sel_device)
                        
                    elif (layer_type == "gemm"):
                        # FC layer
                        tmp_loop_bounds = file_parser.get_loop_bounds()
                        perf_ins = cal_performance_FC(arch_info, tmp_loop_bounds, sel_device)
                        
                    
                    # Store the result log file
                    with open(result_file_path, "w") as f:
                        f.write(perf_ins.get_log())

def custom_sort_key(key):
    match = re.match(r"([a-zA-Z]+)(\d+)", key)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return (prefix, number)
    else:
        return (key, 0) 

def parse_result_optipim():
    print("############################################################################")
    print("##################### OPTIPIM RESULT EXTRACTION Fig 10 #####################")
    print("############################################################################")
    print()
    print("[WARNING] !! CHECK PARAMETERS !!")
    
    # Global information
    infeasible_configs = []
    
    # Iterate over all related sweeping aspects
    for device_type in target_device_list:
        for nn_model in nn_models_list:
            
            # Construct the path to store the output csv file
            folder_path = f"{PROJECT_ROOT}/exp_results/fig10/final_results/optipim/{target_device_idx_name_map[device_type]}/{nn_model}"
            
            # Check and create the corresponding folder
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
            # Iterate over all layers in a model 
            # Store the results of different configurations
            # Format of the final csv file for a single layer
            # Layer Name, 32-our, 32-cosa-est
            ## Iterate over all related configurations
            for num_banks in num_banks_list:
                # Create the storing structure
                layer_results = {}
                
                for sel_storage in storage_method_list:
                    for sel_estimation in estimation_method_list:
                        # Get sweep results folder path
                        sweep_results_folder = f"{PROJECT_ROOT}/exp_results/fig10/{target_device_idx_name_map[device_type]}/{nn_model}/{num_banks}/{storage_method_idx_name_map[sel_storage]}/{estimation_method_idx_name_map[sel_estimation]}"
                        layers_to_parse = [file for file in os.listdir(sweep_results_folder)]
                        
                        # Iterate over all layers
                        for sel_layer in layers_to_parse:
                            # Get the actual result  file path
                            gurobi_log_path = os.path.join(sweep_results_folder, sel_layer)
                            gurobi_log_path = os.path.join(gurobi_log_path, "run_gurobi.log")
                            
                            # Get the actual layer name
                            layer_name = sel_layer.split(".")[-3]
                            
                            # Parse the log file
                            found_optimal = False
                            time_out_flag = False
                            with open(gurobi_log_path, 'r') as file:
                                for line in file:
                                    # If the MILP found the optimal solution
                                    if 'Optimal solution found' in line:
                                        found_optimal = True
                                        continue
                                    
                                    # If the MILP reached time out
                                    if 'Time limit reached' in line:
                                        time_out_flag = True
                                        continue
                                    
                                    # If the model is not feasible
                                    if 'Model is infeasible' in line:
                                        formated_value = "Infeasible"
                                        infeasible_configs.append(gurobi_log_path)
                                        
                                        if layer_name in layer_results:
                                            layer_results[layer_name].append(formated_value)
                                        elif layer_name not in layer_results:
                                            layer_results[layer_name] = ["Infeasible"]
                                        
                                        break
                                    
                                    if (found_optimal or time_out_flag) and 'Best objective' in line:
                                        # Extract the value after "Best Objective"
                                        line = line.strip()
                                        parts = line.split(",")
                                        first_part = parts[0]
                                        tokens = first_part.split()
                                        if len(tokens) >= 3:
                                            value = tokens[2]
                                            # Convert the value to a float and then format it to remove unnecessary zeros
                                            number = float(value)
                                            formated_value = '{:.6e}'.format(number)
                                            
                                            if layer_name in layer_results:
                                                layer_results[layer_name].append(formated_value)
                                            elif layer_name not in layer_results:
                                                layer_results[layer_name] = [formated_value]
                                        
                                        break
                            
                # Generate the final csv file
                csv_file = os.path.join(folder_path, num_banks + ".csv")
                num_bank_value = num_banks.strip("banks")
                header = ["Layer Name", f"{num_bank_value}-optipim", f"{num_bank_value}-cosa_estimation_method"]
                
                with open(csv_file, mode='w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    
                    # Write the header
                    writer.writerow(header)
                    
                    # Write the data
                    for layer_name, values in sorted(layer_results.items(), key=lambda item: custom_sort_key(item[0])):
                        writer.writerow([layer_name] + values)
                        
    print("[INFO] Done")
    
def parse_gurobi_log_file_optipim(file_path):
    # Create the storage structure
    layer_perf = OpPerf("conv")
    
    found_optimal = False
    
    
    # Parse the file
    with open(file_path, 'r') as file:
        for line in file:
            # If the model is infeasible
            if 'Model is infeasible' in line:
                formated_value = "Infeasible"
                layer_perf.finalPerformance = formated_value
                return layer_perf
            
            # If the MILP found the optimal solution
            if 'Optimal solution found' in line:
                found_optimal = True
                continue
            
            # If time out reached
            if 'Time limit reached' in line:
                formated_value = "Timeout"
                layer_perf.finalPerformance = formated_value
                return layer_perf
            
            # If needed value found
            if 'numOutSys' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.outTransCost = formated_number
                
            if 'numInSys' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.inLoadingCost = formated_number
                
            if 'numFilLoading' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.numFilPELoading = formated_number
                
            if (found_optimal) and 'Best objective' in line:
                line = line.strip()
                parts = line.split(",")
                first_part = parts[0]
                tokens = first_part.split()
                if len(tokens) >= 3:
                    value = tokens[2]
                    # Convert the value to a float and then format it to remove unnecessary zeros
                    number = float(value)
                    formated_number = '{:.6e}'.format(number)
                
                    layer_perf.finalPerformance = formated_number
    
    return layer_perf
    
def parse_result_optipim_breakdown():
    print()
    print("[INFO] Extract Detailed Breakdown for OPTIPIM!")
    
    
    # Iterate over all related sweeping aspects
    for device_type in target_device_list:
        for nn_model in nn_models_list:
            
            # Construct the path to store the output csv file
            folder_path = f"{PROJECT_ROOT}/exp_results/fig10/final_results/optipim/{target_device_idx_name_map[device_type]}/{nn_model}"
            
            # Check and create the corresponding folder
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
            # Iterate over all layers in a model 
            # Store the results of different configurations
            # Format of the final csv file for a single layer
            # Layer Name, 32-our-input, 32-our-computation, 32-our-reduction, 32-cosa-input, 32-cosa-computation, 32-cosa-reduction
            ## Iterate over all related configurations
            for num_banks in num_banks_list:
                # Create the storing structure
                layer_results = {}
                header = ["Layer"]
                layer_results["sum"] = [0, 0, 0, 0, 0, 0]
                counter = 0
                
                for sel_storage in storage_method_list:
                    inner_counter = 0
                    for sel_estimation in estimation_method_list:
                        # Get sweep results folder path
                        sweep_results_folder = f"{PROJECT_ROOT}/exp_results/fig10/{target_device_idx_name_map[device_type]}/{nn_model}/{num_banks}/{storage_method_idx_name_map[sel_storage]}/{estimation_method_idx_name_map[sel_estimation]}"
                        layers_to_parse = [file for file in os.listdir(sweep_results_folder)]
                        layer_count = 0
                        
                        # Iterate over all layers
                        for sel_layer in layers_to_parse:
                            # Get the actual result  file path
                            gurobi_log_path = os.path.join(sweep_results_folder, sel_layer)
                            gurobi_log_path = os.path.join(gurobi_log_path, "run_gurobi.log")
                            
                            # Get the actual layer name
                            layer_name = sel_layer.split(".")[-3]
                            
                            # Calculate the layer breakdown
                            tmp_layer_perf = parse_gurobi_log_file_optipim(gurobi_log_path)
                            
                            # Get the desired cost
                            if (tmp_layer_perf.finalPerformance == "Infeasible"):
                                tmp_in_loading = "Infeasible"
                                tmp_out_reduction = "Infeasible"
                                tmp_comp = "Infeasible"
                            elif (tmp_layer_perf.finalPerformance == "Timeout"):
                                tmp_in_loading = "Timeout"
                                tmp_out_reduction = "Timeout"
                                tmp_comp = "Timeout"
                            else:
                                tmp_in_loading = tmp_layer_perf.inLoadingCost
                                tmp_out_reduction = tmp_layer_perf.outTransCost
                                tmp_comp = float(tmp_layer_perf.finalPerformance) - float(tmp_in_loading) - float(tmp_out_reduction)
                            
                            # Parse the gurobi log file
                            if layer_name in layer_results:
                                layer_results[layer_name].append(tmp_in_loading)
                                layer_results[layer_name].append(tmp_comp)
                                layer_results[layer_name].append(tmp_out_reduction)
                            elif layer_name not in layer_results:
                                layer_results[layer_name] = [tmp_in_loading, tmp_comp, tmp_out_reduction]
                            
                            # Update counter
                            if (tmp_in_loading != "Infeasible" and tmp_comp != "Infeasible" and tmp_out_reduction != "Infeasible"):
                                if (tmp_in_loading != "Timeout" and tmp_comp != "Timeout" and tmp_out_reduction != "Timeout"):
                                    
                                    layer_results["sum"][counter * 6 + inner_counter * 3 ] += float(tmp_in_loading)
                                    layer_results["sum"][counter * 6 + inner_counter * 3 + 1] += float(tmp_comp)
                                    layer_results["sum"][counter * 6 + inner_counter * 3 + 2] += float(tmp_out_reduction)
                            
                            layer_count += 1
                        
                        # Update header
                        num_bank_value = num_banks.strip("banks")
                        header.append(f"{num_bank_value}-{estimation_method_idx_name_map[sel_estimation]}-input_loading")
                        header.append(f"{num_bank_value}-{estimation_method_idx_name_map[sel_estimation]}-compute")
                        header.append(f"{num_bank_value}-{estimation_method_idx_name_map[sel_estimation]}-out_reduction")
                        
                        # Update counter
                        inner_counter += 1
                        
                    # Update counter
                    counter += 1
                
                # Generate the final csv file
                csv_file = os.path.join(folder_path, num_banks + "_breakdown.csv")
                
                with open(csv_file, mode='w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    
                    # Add header
                    writer.writerow(header)
                    
                    # Write the data
                    for layer_name, values in sorted(layer_results.items(), key=lambda item: custom_sort_key(item[0])):
                        writer.writerow([layer_name] + values)
                        
    print("[INFO] Done")

def parse_result_cosa():
    print()
    print("[INFO] Extract Results for COSA!")
    
    # Global information
    infeasible_configs = []

    # Iterate over all related sweeping aspects
    for device_type in target_device_list:
        for nn_model in nn_models_list:
            
            # Construct the path to store the output csv file
            folder_path = f"{PROJECT_ROOT}/exp_results/fig10/final_results/cosa/{target_device_idx_name_map[device_type]}/{nn_model}"
            
            # Check and create the corresponding folder
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            ## Iterate over all related configurations
            for num_banks in num_banks_list:
                # Create the storing structure
                layer_results = {}
                last_valid_result = 0
                
                for sel_storage in storage_method_list:
                        # Get sweep results folder path
                        sweep_results_folder = f"{PROJECT_ROOT}/exp_results/fig10/{target_device_idx_name_map[device_type]}/{nn_model}/cosa/{num_banks}/{storage_method_idx_name_map[sel_storage]}"
                        layers_to_parse = [file for file in os.listdir(sweep_results_folder)]
                        
                        # Iterate over all layers
                        for sel_layer in layers_to_parse:
                            # Get the actual result  file path
                            gurobi_log_path = os.path.join(sweep_results_folder, sel_layer)
                            gurobi_log_path = os.path.join(gurobi_log_path, "run_gurobi.log")
                            
                            # Get the actual layer name
                            layer_name = sel_layer.split(".")[-3]
                            
                            # Parse the log file
                            found_optimal = False
                            time_out_flag = False
                            with open(gurobi_log_path, 'r') as file:
                                for line in file:
                                    # If the MILP found the optimal solution
                                    if '[Analytical Perforamnce Modeling Result]' in line:
                                        found_optimal = True
                                        continue
                                    
                                    # If the MILP reached time out
                                    if 'Time limit reached' in line:
                                        time_out_flag = True
                                        continue
                                    
                                    # If the model is not feasible
                                    if 'Model is infeasible' in line:
                                        formated_value = last_valid_result
                                        infeasible_configs.append(gurobi_log_path)
                                        
                                        if layer_name in layer_results:
                                            layer_results[layer_name].append(formated_value)
                                        elif layer_name not in layer_results:
                                            layer_results[layer_name] = [formated_value]
                                        
                                        break
                                    
                                    if (found_optimal or time_out_flag) and 'Real Cost' in line:
                                        # Extract the value after "Best Objective"
                                        line = line.strip()
                                        parts = line.split(":")
                                        second_part = parts[1].strip()
                        
                                        # Convert the value to a float and then format it to remove unnecessary zeros
                                        number = float(second_part)
                                        formated_value = '{:.6e}'.format(number)
                                        last_valid_result = formated_value
                                        
                                        if layer_name in layer_results:
                                            layer_results[layer_name].append(formated_value)
                                        elif layer_name not in layer_results:
                                            layer_results[layer_name] = [formated_value]
                                        
                                        break
                            
                # Generate the final csv file
                csv_file = os.path.join(folder_path, num_banks + ".csv")
                num_bank_value = num_banks.strip("banks")
                header = ["Layer Name", f"{num_bank_value}-cosa"]
                
                with open(csv_file, mode='w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    
                    # Write the header
                    writer.writerow(header)
                    
                    # Write the data
                    for layer_name, values in sorted(layer_results.items(), key=lambda item: custom_sort_key(item[0])):
                        writer.writerow([layer_name] + values)
    print("[INFO] Done")
    
def parse_gurobi_log_file_cosa(file_path):
    # Create the storage structure
    layer_perf = OpPerf("conv")
    
    found_optimal = False
    
    
    # Parse the file
    with open(file_path, 'r') as file:
        for line in file:
            # If the model is infeasible
            if 'Model is infeasible' in line:
                formated_value = "Infeasible"
                layer_perf.finalPerformance = formated_value
                return layer_perf
            
            # If the MILP found the optimal solution
            if 'Optimal solution found' in line:
                found_optimal = True
                continue
            
            # If time out reached
            if 'Time limit reached' in line:
                formated_value = "Timeout"
                layer_perf.finalPerformance = formated_value
                return layer_perf
            
            # If needed value found
            if 'numOutSys' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.outTransCost = formated_number
                
            if 'numInSys' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.inLoadingCost = formated_number
                
            if 'numFilLoading' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.numFilPELoading = formated_number
                
            if (found_optimal) and 'Real Cost' in line:
                line = line.strip()
                parts = line.split(",")
                first_part = parts[0]
                tokens = first_part.split()
                if len(tokens) >= 3:
                    value = tokens[2]
                    # Convert the value to a float and then format it to remove unnecessary zeros
                    number = float(value)
                    formated_number = '{:.6e}'.format(number)
                
                    layer_perf.finalPerformance = formated_number
    
    return layer_perf

def parse_result_cosa_breakdown():
    print()
    print("[INFO] Extract Detailed Breakdown for COSA!")
    
    
    # Iterate over all related sweeping aspects
    for device_type in target_device_list:
        for nn_model in nn_models_list:
            
            # Construct the path to store the output csv file
            folder_path = f"{PROJECT_ROOT}/exp_results/fig10/final_results/cosa/{target_device_idx_name_map[device_type]}/{nn_model}"
            
            # Check and create the corresponding folder
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
            # Iterate over all layers in a model 
            # Store the results of different configurations
            # Format of the final csv file for a single layer
            # Layer Name, 32-cosa-computation, 32-cosa-reduction
            ## Iterate over all related configurations
            for num_banks in num_banks_list:
                # Create the storing structure
                layer_results = {}
                header = ["Layer"]
                layer_results["sum"] = [0, 0, 0, 0]
                counter = 0
                
                for sel_storage in storage_method_list:
                    # Get sweep results folder path
                    sweep_results_folder = f"{PROJECT_ROOT}/exp_results/fig10/{target_device_idx_name_map[device_type]}/{nn_model}/cosa/{num_banks}/{storage_method_idx_name_map[sel_storage]}"
                    layers_to_parse = [file for file in os.listdir(sweep_results_folder)]
                    
                    # Iterate over all layers
                    last_valid_layer = 0
                    for sel_layer in layers_to_parse:
                        # Get the actual result  file path
                        gurobi_log_path = os.path.join(sweep_results_folder, sel_layer)
                        gurobi_log_path = os.path.join(gurobi_log_path, "run_gurobi.log")
                        
                        # Get the actual layer name
                        layer_name = sel_layer.split(".")[-3]
                        
                        # Calculate the layer breakdown
                        tmp_layer_perf = parse_gurobi_log_file_cosa(gurobi_log_path)
                        
                        # Get the desired cost
                        if (tmp_layer_perf.finalPerformance == "Infeasible"):
                            tmp_in_loading = last_valid_layer.inLoadingCost
                            tmp_out_reduction = last_valid_layer.outTransCost
                            tmp_comp = float(last_valid_layer.finalPerformance) - float(tmp_in_loading) - float(tmp_out_reduction)
                            tmp_final = last_valid_layer.finalPerformance
                        elif (tmp_layer_perf.finalPerformance == "Timeout"):
                            tmp_in_loading = "Timeout"
                            tmp_out_reduction = "Timeout"
                            tmp_comp = "Timeout"
                        else:
                            last_valid_layer = tmp_layer_perf
                            tmp_in_loading = tmp_layer_perf.inLoadingCost
                            tmp_out_reduction = tmp_layer_perf.outTransCost
                            tmp_comp = float(tmp_layer_perf.finalPerformance) - float(tmp_in_loading) - float(tmp_out_reduction)
                            tmp_final = tmp_layer_perf.finalPerformance
                        
                        # Parse the gurobi log file
                        if layer_name in layer_results:
                            layer_results[layer_name].append(tmp_in_loading)
                            layer_results[layer_name].append(tmp_comp)
                            layer_results[layer_name].append(tmp_out_reduction)
                            layer_results[layer_name].append(tmp_final)
                        elif layer_name not in layer_results:
                            layer_results[layer_name] = [tmp_in_loading, tmp_comp, tmp_out_reduction, tmp_final]
                            
                        if (tmp_in_loading != "Infeasible" and tmp_comp != "Infeasible" and tmp_out_reduction != "Infeasible"):
                                if (tmp_in_loading != "Timeout" and tmp_comp != "Timeout" and tmp_out_reduction != "Timeout"):
                                    layer_results["sum"][counter * 4 ] += float(tmp_in_loading)
                                    layer_results["sum"][counter * 4 + 1] += float(tmp_comp)
                                    layer_results["sum"][counter * 4  + 2] += float(tmp_out_reduction)
                                    layer_results["sum"][counter * 4  + 3] += float(tmp_final)
                                
                    # Update header
                    num_bank_value = num_banks.strip("banks")
                    header.append(f"{num_bank_value}-cosa-input_loading")
                    header.append(f"{num_bank_value}-cosa-compute")
                    header.append(f"{num_bank_value}-cosa-out_reduction")
                    header.append(f"{num_bank_value}-cosa-overall")
                    
                    # Update counter
                    counter += 1
                            
                # Generate the final csv file
                csv_file = os.path.join(folder_path, num_banks + "break_down.csv")
                
                with open(csv_file, mode='w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    
                    # Write the header
                    writer.writerow(header)
                    
                    # Write the data
                    for layer_name, values in sorted(layer_results.items(), key=lambda item: custom_sort_key(item[0])):
                        writer.writerow([layer_name] + values)
                        
    print("[INFO] Done")

def parse_baseline():
    print()
    print("[INFO] Extract Results for Baseline!")
    
    heuristic_list = ["heuristic_1", "heuristic_2"]
    device_folder_list = ["simdram_baseline", "hbm_baseline"]
    baseline_nn_models_list = ["Alexnet", "ResNet50", "resnet152", "unet", "vgg16", "bert"]
    baseline_num_banks = ["32_banks"]
    
    for sel_heuristic in heuristic_list:
        result_folder = f"{PROJECT_ROOT}/exp_results/fig10/baseline/{sel_heuristic}"
        output_folder = f"{PROJECT_ROOT}/exp_results/fig10/final_results/baseline/{sel_heuristic}"
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for nn_model in baseline_nn_models_list:
            for device_type in device_folder_list:
                # Construct the folder path
                result_folder_path = f"{result_folder}/{nn_model}/{device_type}"
                output_folder_path = f"{output_folder}/{nn_model}/{device_type}"
                
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                
                # Iterate over all num_banks and layers
                for num_banks in baseline_num_banks:
                    layer_results = {}
                    
                    sweep_results_folder = f"{result_folder_path}/{num_banks}"
                    layers_to_parse = [file for file in os.listdir(sweep_results_folder)]
                    
                    for sel_layer in layers_to_parse:
                        log_file_path = os.path.join(sweep_results_folder, sel_layer)
                        log_file_path = os.path.join(log_file_path, "perf.log")
                        
                        with open(log_file_path, 'r') as file:
                            for line in file:
                                if 'finalPerformance' in line:
                                    line = line.strip()
                                    parts = line.split(":")
                                    value = parts[-1].strip()
                                    
                                    number = float(value)
                                    formated_number = '{:.6e}'.format(number)
                                    
                                    layer_results[sel_layer] = formated_number
                                    
                                    break
                                
                    # Generate the final csv file
                    csv_file = os.path.join(output_folder_path, num_banks + ".csv")
                    number_banks_value = num_banks.strip("_banks")
                    header = [f"{number_banks_value}-baseline"]
                    
                    with open(csv_file, mode='w', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        
                        # Write the header
                        writer.writerow(header)
                        
                        for i in range(len(layers_to_parse)):
                            writer.writerow([layer_results[str(i)]])
    print("[INFO] Done")

def parse_gurobi_log_file_baseline(file_path):
    # Create the storage structure
    layer_perf = OpPerf("conv")
    
    found_optimal = False
    
    
    # Parse the file
    with open(file_path, 'r') as file:
        for line in file:
            # If the model is infeasible
            if 'Model is infeasible' in line:
                formated_value = "Infeasible"
                layer_perf.finalPerformance = formated_value
                return layer_perf
            
            # If the MILP found the optimal solution
            if 'Optimal solution found' in line:
                found_optimal = True
                continue
            
            # If time out reached
            if 'Time limit reached' in line:
                formated_value = "Timeout"
                layer_perf.finalPerformance = formated_value
                return layer_perf
            
            # If needed value found
            if 'outTransCost' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.outTransCost = formated_number
                
            if 'inLoadingCost' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.inLoadingCost = formated_number
                
            if 'numFilLoading' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.numFilPELoading = formated_number
                
            if 'finalPerformance' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.finalPerformance = formated_number
    
    return layer_perf

def parse_baseline_breakdown():
    print()
    print("[INFO] Extract Detailed Breakdown for Baseline!")
    
    heuristic_list = ["heuristic_1", "heuristic_2"]
    device_folder_list = ["simdram_baseline", "hbm_baseline"]
    baseline_nn_models_list = ["Alexnet", "ResNet50", "resnet152", "unet", "vgg16", "bert"]
    baseline_num_banks = ["32_banks"]
    
    for sel_heuristic in heuristic_list:
        result_folder = f"{PROJECT_ROOT}/exp_results/fig10/baseline/{sel_heuristic}"
        output_folder = f"{PROJECT_ROOT}/exp_results/fig10/final_results/baseline/{sel_heuristic}"
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for nn_model in baseline_nn_models_list:
            for device_type in device_folder_list:
                # Construct the folder path
                result_folder_path = f"{result_folder}/{nn_model}/{device_type}"
                output_folder_path = f"{output_folder}/{nn_model}/{device_type}"
                
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                
                # Iterate over all num_banks and layers
                for num_banks in baseline_num_banks:
                    layer_results = {}
                
                    #! Add summation
                    layer_results["sum"] = [0, 0, 0]
                    
                    sweep_results_folder = f"{result_folder_path}/{num_banks}"
                    layers_to_parse = [file for file in os.listdir(sweep_results_folder)]
                    
                    for sel_layer in layers_to_parse:
                        log_file_path = os.path.join(sweep_results_folder, sel_layer)
                        log_file_path = os.path.join(log_file_path, "perf.log")
                        
                        # Calculate the layer breakdown
                        tmp_layer_perf = parse_gurobi_log_file_baseline(log_file_path)
                        
                        tmp_in_loading = tmp_layer_perf.inLoadingCost
                        tmp_out_reduction = tmp_layer_perf.outTransCost
                        tmp_comp = float(tmp_layer_perf.finalPerformance) - float(tmp_in_loading) - float(tmp_out_reduction)
                        
                        layer_results[sel_layer] = [tmp_in_loading, tmp_comp, tmp_out_reduction]
                        layer_results["sum"][0] += float(tmp_in_loading)
                        layer_results["sum"][1] += float(tmp_comp)
                        layer_results["sum"][2] += float(tmp_out_reduction)
                                
                    # Generate the final csv file
                    csv_file = os.path.join(output_folder_path, num_banks + "_breakdown.csv")
                    number_banks_value = num_banks.strip("_banks")
                    header = [f"{number_banks_value}-input_loading", f"{number_banks_value}-compute", f"{number_banks_value}-output_reduction",]
                    
                    with open(csv_file, mode='w', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        
                        # Write the header
                        writer.writerow(header)
                        
                        for i in range(len(layers_to_parse)):
                            writer.writerow(layer_results[str(i)])
                            
                        writer.writerow(layer_results["sum"])
    print("[INFO] Done")

if __name__ == "__main__":
    # [Step 1]: Run experiments 
    main_optipim()
    main_cosa()
    main_baseline()
    
    # [Step 2]: Parse all the results
    parse_result_optipim()
    parse_result_optipim_breakdown()
    parse_result_cosa()
    parse_result_cosa_breakdown()
    parse_baseline()
    parse_baseline_breakdown()
