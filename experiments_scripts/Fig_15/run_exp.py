from multiprocessing import Pool
import subprocess
import re
from tqdm import tqdm
import json

import os
import subprocess
import time

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

device_name_holistic = {
    0 : "simdram",
    1 : "hbm"
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

# Environment Setup
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_PATH + "/../.."
NN_MODELS_PATH = f"{PROJECT_ROOT}/nn_models"
MLIR_BIN_PATH = f"{PROJECT_ROOT}/build/bin"

nn_models_list = ["llama3-8B-128","llama3-8B-256","llama3-8B-512","llama3-8B-1024","llama3-14B-128","llama3-14B-256","llama3-14B-512","llama3-14B-1024"]
num_banks_list = [x for x in range(1, 129)]
target_device_list = [0, 1]
PEbandwidth = 256
dataPrecision = 16

def run_gurobi(args):
    gurobi_cmd, gurobi_workdir = args
    with open(f"{gurobi_workdir}/run_gurobi.log", "w") as output_f:
        try:
            subprocess.run(gurobi_cmd, shell=True, check=True, timeout=500, cwd=gurobi_workdir, stdout=output_f)
            return f"Success: {gurobi_workdir}"
        except subprocess.CalledProcessError as e:
            return f"Failed: {gurobi_workdir}, Error: {str(e)}"


def run_exp():
    print("#######################################################################")
    print("#####################   OPTIPIM EXPERIMENT Fig15  #####################")
    print("#######################################################################")
    print()
    print("\n[WARNING] !! CHECK PARAMETERS !!")

    gurobi_cmds = []
    gurobi_folder = []
    
    for nn_model in nn_models_list:
        for device_type in target_device_list:
            # Get the sample layer folder
            sample_layer_folder = f"{NN_MODELS_PATH}/{nn_model}/single_layers/16banks"
            
            # Get all layers in the folder
            layers_to_run = [file for file in os.listdir(sample_layer_folder)]
            
            #
            model_folder_path = f"{NN_MODELS_PATH}/{nn_model}/single_layers/llm_exp"
            
            for sel_layer in layers_to_run:
                for num_banks in num_banks_list:
                    # Get layer name
                    layer_name_parts = sel_layer.split(".")
                    layer_name = layer_name_parts[-3]
                    # Create the corresponding folder
                    folder_path = f"{PROJECT_ROOT}/exp_results/fig15/{nn_model}/{target_device_idx_name_map[device_type]}/{layer_name}"
                    num_bank_str = str(num_banks) + "banks"
                    folder_path = os.path.join(folder_path, num_bank_str)
                    
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                        
                    # Get the mlir file path
                    mlir_file = f"{model_folder_path}/{num_bank_str}/{sel_layer}"
                    
                    # Check the existence of the result file
                    if not os.path.exists(os.path.join(folder_path, "gurobi_output.txt")):
                        gurobi_cmd = f"{MLIR_BIN_PATH}/pim-opt --data-layout-pass \
                                        --trace-output-path {folder_path}/gurobi_output.txt \
                                        --model-debug-file {folder_path}/milp.lp \
                                        --trans-coeff-method 4 \
                                        --target-device-type {device_type} \
                                        --number-counting-method 0 \
                                        --obj-method 0 \
                                        --storage-method 0 \
                                        --config-arch-path {PROJECT_ROOT}/data/hbm_pim.json \
                                        --config-knobs-path {PROJECT_ROOT}/data/knobs.json \
                                        {mlir_file}"
                
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
        
def custom_sort_key(key):
    match = re.match(r"([a-zA-Z]+)(\d+)", key)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return (prefix, number)
    else:
        return (key, 0)  # If no numeric part, treat number as 0
    
def parse_gurobi_log_file(file_path):
    # Create the storage structure
    # The op_type will not be used in this exp, so we set it to "conv"
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
                
            if 'numFilCol' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.numFilCol = formated_number
                
            if 'numColPE' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.numColPE = formated_number
                
            if 'numPESys' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.numPESys = formated_number
                
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

def parse_results():
    print("\n[INFO] Extract Results for Fig15!")
    
    # Final json file
    json_data = {}
    
    # Iterate over all result folders
    for nn_model in nn_models_list:
        # Create the corresponding dict
        json_data[nn_model] = {}
        for device_type in target_device_list:
            # Create new Dict
            json_data[nn_model][device_name_holistic[device_type]] = {}
            
            # Get the base result folder
            result_folder_base = f"{PROJECT_ROOT}/exp_results/fig15/{nn_model}/{target_device_idx_name_map[device_type]}"
            
            layer_folders = [folder for folder in os.listdir(result_folder_base) if os.path.isdir(os.path.join(result_folder_base,folder))]
            
            # Iterate over all layer folders
            for sel_layer in layer_folders:
                # Update the dict
                json_data[nn_model][device_name_holistic[device_type]][sel_layer] = {}
                
                # Track last valid performance & filter_loading
                last_valid_perf = None
                last_valid_loading = None
                
                for num_banks in num_banks_list:
                    num_bank_str = str(num_banks) + "banks"
                    json_data[nn_model][device_name_holistic[device_type]][sel_layer][num_banks] = {}
                    # Get the final result folder
                    result_folder_path = f"{result_folder_base}/{sel_layer}/{num_bank_str}"
                    
                    gurobi_log_path = os.path.join(result_folder_path, "run_gurobi.log")
                    
                    tmp_layer_perf = parse_gurobi_log_file(gurobi_log_path)
                    
                    # Get the desired cost
                    if (tmp_layer_perf.finalPerformance == "Infeasible"):
                        json_data[nn_model][device_name_holistic[device_type]][sel_layer][num_banks]["performance"] = "Infeasible"
                        json_data[nn_model][device_name_holistic[device_type]][sel_layer][num_banks]["filter_loading"] = "Infeasible"
                    elif (tmp_layer_perf.finalPerformance == "Timeout"):
                        # Use the last valid result if available
                        if last_valid_perf is not None and last_valid_loading is not None:
                            json_data[nn_model][device_name_holistic[device_type]][sel_layer][num_banks]["performance"] = last_valid_perf
                            json_data[nn_model][device_name_holistic[device_type]][sel_layer][num_banks]["filter_loading"] = last_valid_loading
                        else:
                            # No valid result yet, so it remains "Timeout"
                            json_data[nn_model][device_name_holistic[device_type]][sel_layer][num_banks]["performance"] = "Timeout"
                            json_data[nn_model][device_name_holistic[device_type]][sel_layer][num_banks]["filter_loading"] = "Timeout"
                    else:
                        # Optimal solution found
                        json_data[nn_model][device_name_holistic[device_type]][sel_layer][num_banks]["performance"] = float(tmp_layer_perf.finalPerformance)
                        
                        # Get the filter loading cost
                        tmp_num_fil_col = float(tmp_layer_perf.numFilCol)
                        tmp_num_col_PE = float(tmp_layer_perf.numColPE)
                        tmp_num_PE_sys = float(tmp_layer_perf.numPESys)
                        
                        tmp_filter_loading_cost = (tmp_num_fil_col * tmp_num_col_PE * float(dataPrecision) * tmp_num_PE_sys) / float(PEbandwidth)
                        json_data[nn_model][device_name_holistic[device_type]][sel_layer][num_banks]["filter_loading"] = float(tmp_filter_loading_cost)
                        
                        # Update the "last valid" values
                        last_valid_perf = float(tmp_layer_perf.finalPerformance)
                        last_valid_loading = float(tmp_filter_loading_cost)
                               
    # Write out the desired json file
    json_file_path = f"{PROJECT_ROOT}/exp_results/fig15/all_layer_llama_ae.json"
    with open(json_file_path, 'w') as f:
        json.dump(json_data, f, indent=4)

if __name__ == "__main__":
    # [Step 1]: Run all experiments
    run_exp()

    # [Step 2]: Extract results
    parse_results()
