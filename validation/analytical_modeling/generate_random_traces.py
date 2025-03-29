# This file generates random traces based on the original loop dimensions

from .analytical import *
from tqdm import tqdm
import random
import multiprocessing as mp
import time
import sys
import os

import pandas as pd

from itertools import product
from pprint import pprint



#===----------------------------------------------------------------------===#
# Class/Function Definition
#===----------------------------------------------------------------------===#
def find_triplets(sel_dimension):
    triplets = []
    for b in range (1, sel_dimension + 1):
        for c in range(1, sel_dimension + 1):
            if b * c > sel_dimension:
                break
            
            if sel_dimension % (b * c) == 0:
                d = sel_dimension // (b * c)
                if d >= 1:
                    triplets.append([b, c, d])       
    return triplets

def get_coeff_combs(sel_loop_bound):
    # sel_loop_bound is the transformed loop bounds of a loop variable
    trans_coeff = []
    
    # Get Comb 0
    trans_coeff.append([sel_loop_bound[1] * sel_loop_bound[2], sel_loop_bound[2], 1])
    
    # Get Comb 1
    trans_coeff.append([sel_loop_bound[2], sel_loop_bound[0] * sel_loop_bound[2], 1])
    
    # Get Comb 2
    trans_coeff.append([1, sel_loop_bound[0] * sel_loop_bound[2], sel_loop_bound[0]])
    
    # Get Comb 3
    trans_coeff.append([1, sel_loop_bound[0], sel_loop_bound[0] * sel_loop_bound[1]])
    
    # Get Comb 4
    trans_coeff.append([sel_loop_bound[1] * sel_loop_bound[2], 1, sel_loop_bound[1]])
    
    # Get Comb 5
    trans_coeff.append([sel_loop_bound[1], 1, sel_loop_bound[1] * sel_loop_bound[0]])
    
    return trans_coeff

def check_num_PE_constraint(loop_bounds, op_type, num_PE):
    used_PE = 1
    
    if (op_type == 0):
        for x in range(0, NUM_BOUND_CONV2D):
            used_PE *= loop_bounds[x][2]
    elif (op_type == 1):
        for x in range(0, NUM_BOUND_FC):
            used_PE *= loop_bounds[x][2]
            
    if (used_PE > num_PE):
        return False
    
    return True

def check_num_Col_constraint(loop_bounds, op_type, device_type, arch_info):
    used_col = 1
    
    if (op_type == 0):
        for x in range(0, NUM_BOUND_CONV2D):
            used_col *= loop_bounds[x][1]
    elif (op_type == 1):
        for x in range(0, NUM_BOUND_FC):
            used_col *= loop_bounds[x][1]
            
    if (device_type == 0):
        # Bit-serial device
        if (used_col > arch_info.numCol):
            return False
    elif (device_type == 1):
        if (used_col * arch_info.dataWidth > arch_info.numCol):
            return False
    
    return True

def check_size_constraint_FC(loop_bounds, arch_info, device_type, storage_method):
    num_col = 1
    num_in = 1
    num_out = 1
    num_fil = 1
    
    # FC
    ## Calculate the number of filters
    num_fil *= loop_bounds[loop_bound_name_FC["N"]][0]
    num_fil *= loop_bounds[loop_bound_name_FC["Q"]][0]
    num_fil *= loop_bounds[loop_bound_name_FC["R"]][0]
    
    ## Calculate the number of outputs
    num_out *= loop_bounds[loop_bound_name_FC["P"]][0]
    num_out *= loop_bounds[loop_bound_name_FC["R"]][0]
    num_out *= loop_bounds[loop_bound_name_FC["N"]][0]
    
    ## Calculate the number of inputs
    num_in *= loop_bounds[loop_bound_name_conv2D["Q"]][0]
    num_in *= loop_bounds[loop_bound_name_conv2D["P"]][0]
    num_in *= loop_bounds[loop_bound_name_conv2D["N"]][0]
    
    ## Calculate the number of columns
    for x in range(0, NUM_BOUND_FC):
        num_col *= loop_bounds[x][1]
            
    ## Check the final condition based on the target device type and storage method
    if (storage_method == 0):
        # Output not stored in the column
        if (device_type == 0):
            # Bit-serial device
            if ((num_in + num_fil) * arch_info.dataWidth > arch_info.numRow):
                return False
        elif (device_type == 1):
            # HBM-PIM
            if ((num_in + num_fil) > arch_info.numRow):
                return False
    elif (storage_method == 1):
        # Store all elements
        if (device_type == 0):
            # Bit-serial device
            if ((num_in + num_fil + num_out) * arch_info.dataWidth > arch_info.numRow):
                return False
        elif (device_type == 1):
            # HBM-PIM
            if ((num_in + num_fil + num_out) > arch_info.numRow):
                return False
            
    return True
    
def check_size_constraint(loop_bounds, op_type, arch_info, device_type, trans_coeff, stride, dilation, storage_method):
    num_col = 1
    num_in = 1
    num_out = 1
    num_fil = 1
    
    if (op_type == 0):
        # Conv2D
        ## Calculate the number of filters
        num_fil *= loop_bounds[loop_bound_name_conv2D["K"]][0]
        num_fil *= loop_bounds[loop_bound_name_conv2D["R"]][0]
        num_fil *= loop_bounds[loop_bound_name_conv2D["S"]][0]
        num_fil *= loop_bounds[loop_bound_name_conv2D["C"]][0]
        
        ## Calculate the number of outputs
        num_out *= loop_bounds[loop_bound_name_conv2D["N"]][0]
        num_out *= loop_bounds[loop_bound_name_conv2D["P"]][0]
        num_out *= loop_bounds[loop_bound_name_conv2D["Q"]][0]
        num_out *= loop_bounds[loop_bound_name_conv2D["K"]][0]
        
        ## Calculate the numebr of inputs
        num_in *= loop_bounds[loop_bound_name_conv2D["N"]][0]
            
        coeff_1 = trans_coeff[loop_bound_name_conv2D["P"]][0] * stride
        coeff_2 = trans_coeff[loop_bound_name_conv2D["R"]][0] * dilation
        in_term_1 = count_unique(coeff_1, coeff_2, loop_bounds[loop_bound_name_conv2D["P"]][0], loop_bounds[loop_bound_name_conv2D["R"]][0])
        num_in *= len(in_term_1)
        
        coeff_3 = trans_coeff[loop_bound_name_conv2D["Q"]][0] * stride
        coeff_4 = trans_coeff[loop_bound_name_conv2D["S"]][0] * dilation
        in_term_2 = count_unique(coeff_3, coeff_4, loop_bounds[loop_bound_name_conv2D["Q"]][0], loop_bounds[loop_bound_name_conv2D["S"]][0])
        num_in *= len(in_term_2)
        num_in *= loop_bounds[loop_bound_name_conv2D["C"]][0]
        
        # Calculate the number of columns
        for x in range(0, NUM_BOUND_CONV2D):
            num_col *= loop_bounds[x][1]
    elif (op_type == 1):
        # FC
        ## Calculate the number of filters
        num_fil *= loop_bounds[loop_bound_name_FC["N"]][0]
        num_fil *= loop_bounds[loop_bound_name_FC["Q"]][0]
        num_fil *= loop_bounds[loop_bound_name_FC["R"]][0]
        
        ## Calculate the number of outputs
        num_out *= loop_bounds[loop_bound_name_FC["P"]][0]
        num_out *= loop_bounds[loop_bound_name_FC["R"]][0]
        num_out *= loop_bounds[loop_bound_name_FC["N"]][0]
        
        ## Calculate the number of inputs
        num_in *= loop_bounds[loop_bound_name_conv2D["Q"]][0]
        num_in *= loop_bounds[loop_bound_name_conv2D["P"]][0]
        num_in *= loop_bounds[loop_bound_name_conv2D["N"]][0]
        
        ## Calculate the numebr of columns
        for x in range(0, NUM_BOUND_FC):
            num_col *= loop_bounds[x][1]
            
    ## Check the final condition based on the target device type and storage method
    if (storage_method == 0):
        # Output not stored in the column
        if (device_type == 0):
            # Bit-serial device
            if ((num_in + num_fil) * arch_info.dataWidth > arch_info.numRow):
                return False
        elif (device_type == 1):
            # HBM-PIM
            if ((num_in + num_fil) > arch_info.numRow):
                return False
    elif (storage_method == 1):
        # Store all elements
        if (device_type == 0):
            # Bit-serial device
            if ((num_in + num_fil + num_out) * arch_info.dataWidth > arch_info.numRow):
                return False
        elif (device_type == 1):
            # HBM-PIM
            if ((num_in + num_fil + num_out) > arch_info.numRow):
                return False
            
    return True

def generate_traces(loop_bounds, 
                    op_type, 
                    arch_file_path, 
                    num_PE, 
                    device_type, 
                    storage_method, 
                    stride, 
                    dilation,
                    num_traces: int,
                    folder_base):
    """_summary_

    This function generate num_traces viable traces for a given layer
    
    loop_bounds: [N, K, P, Q, C, R, S] or [FC_N, FC_K, FC_P, FC_Q, FC_R]
    op_type:
        - 0: Conv2D
        - 1: FC
    device_type:
        - PUM
        - PNM (NBP)
    storage_method:
        - 0: No output
        - 1: All elements
    """
    
    # Record the total number of combinations needed for iteration
    total_combination = 1
    generated_trace = 0
    
    # Get the architecture specification
    arch_info = ArchInfo(arch_file_path)
    
    if (op_type == 0):
        # Conv2D operator
        N_loop_bounds = find_triplets(loop_bounds[loop_bound_name_conv2D["N"]])
        K_loop_bounds = find_triplets(loop_bounds[loop_bound_name_conv2D["K"]])
        P_loop_bounds = find_triplets(loop_bounds[loop_bound_name_conv2D["P"]])
        Q_loop_bounds = find_triplets(loop_bounds[loop_bound_name_conv2D["Q"]])
        C_loop_bounds = find_triplets(loop_bounds[loop_bound_name_conv2D["C"]])
        R_loop_bounds = find_triplets(loop_bounds[loop_bound_name_conv2D["R"]])
        S_loop_bounds = find_triplets(loop_bounds[loop_bound_name_conv2D["S"]])
        
        # Get the total number of combinations
        total_combination *= (len(N_loop_bounds))
        total_combination *= (len(K_loop_bounds))
        total_combination *= (len(P_loop_bounds))
        total_combination *= (len(Q_loop_bounds))
        total_combination *= (len(C_loop_bounds))
        total_combination *= (len(R_loop_bounds))
        total_combination *= (len(S_loop_bounds))
        
        #
        print("[INFO]\t\tTotal Combinations: {}".format(total_combination))
        
        attempts: int = 0
        max_attempts: int = 1000000
        seen_traces = set()     # To store keys of already-generated traces

        while generated_trace < num_traces and attempts < max_attempts:
            attempts += 1

            # Randomly pick one candidate from each dimension
            sel_loop_bounds = [
                random.choice(N_loop_bounds),
                random.choice(K_loop_bounds),
                random.choice(P_loop_bounds),
                random.choice(Q_loop_bounds),
                random.choice(C_loop_bounds),
                random.choice(R_loop_bounds),
                random.choice(S_loop_bounds)
            ]

             # For each loop, pick the transformation coefficients as before.
            # (Here, we continue to always choose the same permutation (index 3) if that is what you desire.)
            sel_trans_coeff = [
                get_coeff_combs(sel_loop_bounds[0])[3],
                get_coeff_combs(sel_loop_bounds[1])[3],
                get_coeff_combs(sel_loop_bounds[2])[3],
                get_coeff_combs(sel_loop_bounds[3])[3],
                get_coeff_combs(sel_loop_bounds[4])[3],
                get_coeff_combs(sel_loop_bounds[5])[3],
                get_coeff_combs(sel_loop_bounds[6])[3]
            ]
                                    
            # Check the criteria
            ## NumPE less than the allocated number
            if (not check_num_PE_constraint(sel_loop_bounds, op_type, num_PE)):
                continue
            
            ## NumCol less than the number of columns in the PE
            if (not check_num_Col_constraint(sel_loop_bounds, op_type, device_type, arch_info)):
                continue
            
            ## Check the size of the element in a column
            if (not check_size_constraint(sel_loop_bounds, op_type, arch_info, device_type, sel_trans_coeff, stride, dilation, storage_method)):
                continue

            # Create a unique key for the trace. You could, for example, use:
            # A tuple that contains all the loop bounds (converted to tuples) and all transformation coefficients.
            trace_key = tuple(tuple(lb) for lb in sel_loop_bounds) + tuple(
                tuple(coeff) for coeff in sel_trans_coeff
            )

            if trace_key in seen_traces:
                # Already generated this trace; skip it.
                continue
            else:
                seen_traces.add(trace_key)

            #NOTE: Viable Trace found
            # Print trace to a file
            # Print the corresponding trace
            file_path = folder_base + "/Traces/trace_" + str(generated_trace) + ".txt"
            print_trace_conv2D(file_path,
                                sel_loop_bounds,
                                loop_bounds,
                                sel_trans_coeff,
                                stride,
                                dilation)                                    
            generated_trace += 1
        
        # Print the results
        print(f"[INFO]\t\t{num_traces} Traces Generated!")
    
    elif (op_type == 1):
        # FC Operator
        N_loop_bounds = find_triplets(loop_bounds[loop_bound_name_FC["N"]])
        K_loop_bounds = find_triplets(loop_bounds[loop_bound_name_FC["K"]])
        P_loop_bounds = find_triplets(loop_bounds[loop_bound_name_FC["P"]])
        Q_loop_bounds = find_triplets(loop_bounds[loop_bound_name_FC["Q"]])
        R_loop_bounds = find_triplets(loop_bounds[loop_bound_name_FC["R"]])
        
        #
        # Get the total number of combinations
        total_combination *= (len(N_loop_bounds))
        total_combination *= (len(K_loop_bounds))
        total_combination *= (len(P_loop_bounds))
        total_combination *= (len(Q_loop_bounds))
        total_combination *= (len(R_loop_bounds))
        
        print("[INFO]\t\tTotal Combinations: {}".format(total_combination))
        
        attempts: int = 0
        max_attempts: int = 1000000
        seen_traces = set()     # To store keys of already-generated traces

        while generated_trace < num_traces and attempts < max_attempts:
            attempts += 1

            # Randomly pick one candidate from each candidate list
            sel_loop_bounds = [
                random.choice(N_loop_bounds),
                random.choice(K_loop_bounds),
                random.choice(P_loop_bounds),
                random.choice(Q_loop_bounds),
                random.choice(R_loop_bounds)
            ]
            
            # Check the constraints for FC
            if not check_num_PE_constraint(sel_loop_bounds, op_type, num_PE):
                continue
            if not check_num_Col_constraint(sel_loop_bounds, op_type, device_type, arch_info):
                continue
            if not check_size_constraint_FC(sel_loop_bounds, arch_info, device_type, storage_method):
                continue

            # Pick the transformation coefficients for each loop variable.
            # (Here we are picking the same permutation (index 3) as before.)
            sel_trans_coeff = [
                get_coeff_combs(sel_loop_bounds[0])[3],
                get_coeff_combs(sel_loop_bounds[1])[3],
                get_coeff_combs(sel_loop_bounds[2])[3],
                get_coeff_combs(sel_loop_bounds[3])[3],
                get_coeff_combs(sel_loop_bounds[4])[3]
            ]

            # Create a unique key for the trace.
            # We use a tuple that contains all loop bound triplets (each converted to a tuple)
            # and all transformation coefficient triplets.
            trace_key = tuple(tuple(lb) for lb in sel_loop_bounds) + tuple(tuple(coeff) for coeff in sel_trans_coeff)
            
            # If the trace is already seen, skip it.
            if trace_key in seen_traces:
                continue
            else:
                seen_traces.add(trace_key)
                            
            #NOTE: Viable Trace found
            # Print trace to a file
            file_path = folder_base + "/Traces/trace_" + str(generated_trace) + ".txt"
            print_trace_FC(file_path,
                            sel_loop_bounds,
                            loop_bounds,
                            sel_trans_coeff)
            generated_trace += 1
        
        # Print the results
        print(f"[INFO]\t\t{num_traces} Traces Generated!")
        # print("Found Best Performance: {}".format(best_performance))
        # print("\tCorresponding Loop Bounds: {}".format(final_loop_bounds))

def print_trace_conv2D(file_path, opt_loop_bounds, ori_loop_bounds, trans_coefficients, stride, dilation):
    """
    This functino print the generated trace of a conv2d layer into a file
    """
    
    # If the file doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    out_string = ""
    out_string += "conv2d\n"
    out_string += "Problem: "
    
    # Print original loop bounds
    for i in range(len(ori_loop_bounds)):
        if i != (len(ori_loop_bounds) - 1):
            out_string += str(ori_loop_bounds[i]) + ","
        else:
            out_string += str(ori_loop_bounds[i]) + "\n"
            
    # Dilations
    out_string += "DilationStride: "
    out_string += (str(dilation) + "," +str(dilation) + "," + str(stride) + "," + str(stride) + "\n")
            
    # The assumed order is : Level 2 -> Level 0 -> Level 1
    out_string += "Loop: N,K,P,Q,C,R,S,N,K,P,Q,C,R,S,N,K,P,Q,C,R,S\n"
    out_string += "Bound: "
    
    # Three levels
    for i in range(NUM_LOOP_LEVEL):
        selected_loop_level = i
        
        if (i == 1):
            selected_loop_level = 0
        elif (i == 0):
            selected_loop_level = 1

        # Iterate over all loop bounds
        for i in range(NUM_BOUND_CONV2D):
            if (selected_loop_level == (NUM_LOOP_LEVEL - 1) and i == (NUM_BOUND_CONV2D - 1)):
                out_string += str(opt_loop_bounds[i][selected_loop_level]) + "\n"
            else:
                out_string += str(opt_loop_bounds[i][selected_loop_level]) + ","
            
    out_string += "Tag: P,P,P,P,P,P,P,T,T,T,T,T,T,T,P,P,P,P,P,P,P\n"
    out_string += "StartBankRow: 0,0\n"
    
    #
    #  Retrieve ALL the transformation coefficients
    #
    # All coefficients are ordered as : Level 2, Level 0, Level 1 to align with the simulator
    # LoopVariable : N
    out_string += "Coeff_N2,N0,N1: "
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["N"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["N"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["N"]][1]) + "\n")
    
    # LoopVariable : K
    out_string += "Coeff_K2,K0,K1: "
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["K"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["K"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["K"]][1]) + "\n")
    # LoopVariable : P
    out_string += "Coeff_P2,P0,P1: "
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["P"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["P"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["P"]][1]) + "\n")
    # LoopVariable : Q
    out_string += "Coeff_Q2,Q0,Q1: "
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["Q"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["Q"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["Q"]][1]) + "\n")
    # LoopVariable : C
    out_string += "Coeff_C2,C0,C1: "
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["C"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["C"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["C"]][1]) + "\n")
    # LoopVariable : R
    out_string += "Coeff_R2,R0,R1: "
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["R"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["R"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["R"]][1]) + "\n")
    # LoopVariable : S
    out_string += "Coeff_S2,S0,S1: "
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["S"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["S"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_conv2D["S"]][1]) + "\n")
    
    out_string += "end\n"

    # Print the final file
    with open(file_path, "w") as f:
        print(out_string, file=f)    
        
def print_trace_FC(file_path, opt_loop_bounds, ori_loop_bounds, trans_coefficients):
    """
    This functino print the generated trace of a FC layer into a file
    """
    # If the file doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    out_string = ""
    out_string += "gemm\n"
    out_string += "Problem: "
    
    # Print original loop bounds
    for i in range(len(ori_loop_bounds)):
        if i != (len(ori_loop_bounds) - 1):
            out_string += str(ori_loop_bounds[i]) + ","
        else:
            out_string += str(ori_loop_bounds[i]) + "\n"
            
    
    out_string += "DilationStride: 1,1,1,1\n"
    out_string += "Loops: N,K,P,Q,R,N,K,P,Q,R,N,K,P,Q,R\n"
    out_string += "Bound: "
    
    # Three levels
    for i in range(NUM_LOOP_LEVEL):
        selected_loop_level = i
        
        if (i == 1):
            selected_loop_level = 0
        elif (i == 0):
            selected_loop_level = 1

        # Iterate over all loop bounds
        for i in range(NUM_BOUND_FC):
            if (selected_loop_level == (NUM_LOOP_LEVEL - 1) and i == (NUM_BOUND_FC - 1)):
                out_string += str(opt_loop_bounds[i][selected_loop_level]) + "\n"
            else:
                out_string += str(opt_loop_bounds[i][selected_loop_level]) + ","
    
    out_string += "Tag: P,P,P,P,P,T,T,T,T,T,P,P,P,P,P\n"
    out_string += "StartBankRow: 0,0\n"
    
    #
    #  Retrieve ALL the transformation coefficients
    #
    # All coefficients are ordered as : Level 2, Level 0, Level 1 to align with the simulator
    # LoopVariable : N
    out_string += "Coeff_N2,N0,N1: "
    out_string += (str(trans_coefficients[loop_bound_name_FC["N"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_FC["N"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_FC["N"]][1]) + "\n")
    
    # LoopVariable : K
    out_string += "Coeff_K2,K0,K1: "
    out_string += (str(trans_coefficients[loop_bound_name_FC["K"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_FC["K"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_FC["K"]][1]) + "\n")
    # LoopVariable : P
    out_string += "Coeff_P2,P0,P1: "
    out_string += (str(trans_coefficients[loop_bound_name_FC["P"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_FC["P"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_FC["P"]][1]) + "\n")
    # LoopVariable : Q
    out_string += "Coeff_Q2,Q0,Q1: "
    out_string += (str(trans_coefficients[loop_bound_name_FC["Q"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_FC["Q"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_FC["Q"]][1]) + "\n")
    # LoopVariable : R
    out_string += "Coeff_R2,R0,R1: "
    out_string += (str(trans_coefficients[loop_bound_name_FC["R"]][2]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_FC["R"]][0]) + ",")
    out_string += (str(trans_coefficients[loop_bound_name_FC["R"]][1]) + "\n")
    
    out_string += "end\n"

    # Print the final file
    with open(file_path, "w") as f:
        print(out_string, file=f)   

#===----------------------------------------------------------------------===#
# User Interfaces
#===----------------------------------------------------------------------===#

def parse_args():
    """
        Parse input arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description='Generate traces for analytical model verification')
    parser.add_argument('--num_PE', type=int, default=16,
                        help='Number of processing elements')
    parser.add_argument('--device_type', type=int, default=0,
                        help='Device type (0: Bit-serial, 1: Bit-parallel)')
    parser.add_argument('--storage_method', type=int, default=0,
                        help='Storage method (0: no-output, 1: all elements)')
    parser.add_argument('--model_params_path', type=str, required=True,
                        help='Path to layer parameters CSV files folder')
    parser.add_argument('--num_traces', type=int, default=30,
                        help="number of traces generated for each layer")
    parser.add_argument('--out_folder', type=str, default=".",
                        help="Path of the output folder")
    parser.add_argument('--arch_file_path', type=str, required=True,
                        help='Path to architecture parameters')
    return parser.parse_args()


def generate_random_traces(num_PE: int, device_type: int, storage_method: int, num_traces: int, out_folder: str, arch_file_path: str):
    # The meaning of each column in the csv file
    columns = [
        'layer_id', 'op_type', 'op_id', 'batch_size', 'output_channel', 
        'output_height', 'output_width', 'input_channel', 'kernel_height',
        'kernel_width', 'dilation_H', 'dilation_W', 'stride_H', 'stride_W'
    ]
    
    # Define all NN models
    nn_models = ["alexnet", "resnet50", "resnet152", "unet", "vgg16"]
    
    time.sleep(2)
    
    for sel_model in nn_models:
        model_param_csv_file = os.path.join("nn_models", sel_model, "layer_params.csv")
        
        print("[INFO] NN Model: {}".format(sel_model))
        print(f"[INFO] Trace generation for all layers in {model_param_csv_file}")
        
        df = pd.read_csv(model_param_csv_file, names=columns, header=None)
        
        # Read the CSV file
        print("[INFO]\tNumber of layers: {}".format(len(df)))
        print(f"[INFO]\tTrace generation for all layers in {model_param_csv_file}")
        
        # For each layer in the csv file
        for i in range(len(df)):
            
            print("[INFO]\t\tLayer: {}".format(i))
            
            layer_row = df[df['layer_id'] == i].iloc[0]
            # Get op_type
            layer_type = layer_row['op_type']
            op_type = 0
            if (layer_type == " CONV2d"):
                op_type = 0
            else:
                op_type = 1
            
            if (op_type == 0):
                # Conv Layer
                loop_bounds = [  # N K P Q C R S
                    layer_row['batch_size'],
                    layer_row['output_channel'], 
                    layer_row['output_height'],
                    layer_row['output_width'],
                    layer_row['input_channel'],
                    layer_row['kernel_height'], 
                    layer_row['kernel_width']
                ]
            else:
                loop_bounds = [  # N K P Q R 
                    layer_row['batch_size'],
                    layer_row['output_channel'], 
                    layer_row['output_height'],
                    layer_row['output_width'],
                    layer_row['kernel_height']
                ]
            
            stride_H = layer_row['stride_H']
            stride_W = layer_row['stride_W']
            dilation_H = layer_row['dilation_H']
            dilation_W = layer_row['dilation_W']
            
            
            
            assert stride_H == stride_W
            assert dilation_H == dilation_W
            
            # Folder structure
            #   NN_model
            #       - Device_type
            #           - Layer index and type
            #               - Traces
            #                   - trace_files
            result_folder = os.path.join(out_folder, sel_model)
            device_folder = os.path.join(result_folder, device_type_idx_name_map[device_type])
            layer_folder = os.path.join(device_folder, "layer_" + str(i) + "_type_" + str(op_type))
            
            # Generate all desired traces
            generate_traces(loop_bounds=loop_bounds, 
                            op_type=op_type, 
                            arch_file_path=arch_file_path, 
                            num_PE=num_PE, 
                            device_type=device_type, 
                            storage_method=storage_method, 
                            stride=stride_H, 
                            dilation=dilation_W, 
                            num_traces=num_traces,
                            folder_base=layer_folder)
    

if __name__ == "__main__":
    """
        Sample usage:
            python generate_random_traces.py --num_PE 32 \
                --device_type 1 \
                --storage_method 0 \
                --model_params_path /home/jianliu/Projects/Datalayout/MLIR-PIM/rebuttal_experiments/model_layer_info \
                --num_traces 40 \
                --out_folder ../traces/
    """
    args = parse_args()

    num_PE: int = args.num_PE
    device_type: int = args.device_type
    storage_method: int = args.storage_method
    num_traces: int = args.num_traces
    out_folder: str = args.out_folder
    arch_file_path: str = args.arch_file_path
    
    generate_random_traces(num_PE, device_type, storage_method, num_traces, out_folder, arch_file_path)
