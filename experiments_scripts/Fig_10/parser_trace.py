import copy
import json

#===----------------------------------------------------------------------===#
# Global Settings
#===----------------------------------------------------------------------===#

NUM_BOUND_CONV2D = 7
NUM_BOUND_FC = 5
NUM_LOOP_LEVEL = 3

loop_bound_name_conv2D = {
    "N" : 0,
    "K" : 1,
    "P" : 2,
    "Q" : 3,
    "C" : 4,
    "R" : 5,
    "S" : 6
}

loop_bound_name_FC = {
    "N" : 0,
    "K" : 1,
    "P" : 2,
    "Q" : 3,
    "R" : 4
}

loop_bound_idx_name_map_conv2D = {
    0 : "N",
    1 : "K",
    2 : "P",
    3 : "Q",
    4 : "C",
    5 : "R",
    6 : "S"
}

loop_bound_idx_name_map_FC = {
    0 : "N",
    1 : "K",
    2 : "P",
    3 : "Q",
    4 : "R"
}

device_type_idx = {
    "PUM" : 0,
    "NBP" : 1
}

#===----------------------------------------------------------------------===#
# Class/Function Definition
#===----------------------------------------------------------------------===#

def count_unique(a, b, c, d):
    "Calculate the unique values of o = a*x + b*y, where x in [0, c) and y in [0, d)"
    o_values = set()
    
    for i in range(c):
        ai = a * i
        for j in range(d):
            temp = ai + b * j
            o_values.add(temp)
    
    return o_values


# Define the class to store all information related to the hardware
class ArchInfo:
    def __init__(self, json_file):
        # Initialize attributes using functions
        data = self._load_data(json_file)
        self._initialize_attributes(data)
    
    def _load_data(self, json_file):
        """Function to read data from the JSON file."""
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"Error: The file '{json_file}' was not found.")
            return {}
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")
            return {}
    
    def _initialize_attributes(self, data):
        """Function to initialize class attributes from data."""
        # Number of bits in a data element
        self.dataWidth = data.get('dataWidth', 0)
        
        # Number of rows in a bank
        self.numRow = data.get('numRow', 0)
        
        # Number of cols in a bank
        self.numCol = data.get('numCol', 0)
        
        # Number of bits/cycle
        self.PEBandWidth = data.get('PEBandWidth', 0)
        
        # Number of bits/cycle
        self.SysBandWidth = data.get('SysBandWidth', 0)
        
        # Number of cycles needed to perform a 16bits multiplication
        self.mulLat = data.get('mulLat', 0)
        
        # Number of cycles needed to perform a 16bits addition
        self.addLat = data.get('addLat', 0)
        
        # Number of cycles needed for a row activation
        self.rowAct = data.get('rowAct', 0)
        
        # Following two parameters are not used
        self.interColTransLat = data.get('interColTransLat', 0)
        self.interPETransLat = data.get('interPETransLat', 0)
    
    def __repr__(self):
        return (f"ArchInfo(dataWidth={self.dataWidth} bits/element, numRow={self.numRow}, numCol={self.numCol}, "
                f"PEBandWidth={self.PEBandWidth} bits/cycle, SysBandWidth={self.SysBandWidth} bits/cycle, mulLat={self.mulLat} cycles, "
                f"addLat={self.addLat} cycles, rowAct={self.rowAct} cycles, interColTransLat={self.interColTransLat}, "
                f"interPETransLat={self.interPETransLat})")
    
    
# Define the class to store the calcualted performance value
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
            
    def get_log(self):
        out_str = ""
        
        for key, value in vars(self).items():
            out_str += f"{key}: {value}\n"
            
        return out_str

def cal_performance_FC(arch_info, loop_bounds, device_type):
    """
        This function calculates the final performance of the given transformed loop
        Needed Inputs:
            - opType: Conv2D or FC
            -loop_bounds:
                loop_bounds infomration for all transformed loop bounds.
                It should be a two dimensional list [x][l], x is the corresponding loop variable
                l is the corresponding loop level. l = 0, loop level 0;
            -trans_coeffs:
                Store the coefficients for different loop variables
                indexed by [x][l]
            -device_type:
                0. PUM --> bit-serial PIM
                1. PNM --> HBM-PIM
        
    """
    # Create the storing structure for the performance value
    op_performance = OpPerf("FC")
    
    # Calculate the number of Multiplications in a column
    num_mul_col = 1
    ## Multiply all transformed loop bounds at level 0
    for x in range(0, NUM_BOUND_FC):
        num_mul_col *= loop_bounds[x][0]
        
    op_performance.numMulCol = num_mul_col
    
    # Calculate the numebr of columns in a PE
    num_col_PE = 1
    ## Multiply all transformed loop bounds at level 1
    for x in range(0, NUM_BOUND_FC):
        num_col_PE *= loop_bounds[x][1]
        
    op_performance.numColPE = num_col_PE
    
    # Calculate the number of PEs in a Sys
    num_PE_Sys = 1
    ## Multiply all transformed loop bounds at level 2
    for x in range(0, NUM_BOUND_FC):
        num_PE_Sys *= loop_bounds[x][2]
    
    num_channel_Sys = int((num_PE_Sys - 1) // 16 + 1)

    ChannelBandWidth = arch_info.SysBandWidth * num_channel_Sys
        
    op_performance.numPESys = num_PE_Sys
        
    # Calculate the number of outputs per column
    # NumOutCol = N * P * R at loop level 0
    num_out_col = 1
    num_out_col *= loop_bounds[loop_bound_name_FC["P"]][0]
    num_out_col *= loop_bounds[loop_bound_name_FC["R"]][0]
    num_out_col *= loop_bounds[loop_bound_name_FC["N"]][0]
    
    op_performance.numOutCol = num_out_col
    
    # Calculate the number of filters per column
    # NumFilCol = Q * R at loop level 0
    num_fil_col = 1
    num_fil_col *= loop_bounds[loop_bound_name_FC["N"]][0]
    num_fil_col *= loop_bounds[loop_bound_name_FC["Q"]][0]
    num_fil_col *= loop_bounds[loop_bound_name_FC["R"]][0]
    
    op_performance.numFilCol = num_fil_col
    
    # Calculate the number of reductions per output in a colun
    # Q - 1; at loop level 0
    num_red_out_col = 1
    num_red_out_col *= loop_bounds[loop_bound_name_FC["Q"]][0]
    num_red_out_col -= 1
    
    op_performance.numRedOutCol = num_red_out_col
    
    # Calculate the number of inputs per column
    # NumFilCol = Q * P * N at loop level 0
    num_in_col = 1
    num_in_col *= loop_bounds[loop_bound_name_conv2D["Q"]][0]
    num_in_col *= loop_bounds[loop_bound_name_conv2D["P"]][0]
    num_in_col *= loop_bounds[loop_bound_name_conv2D["N"]][0]
    

    op_performance.numInCol = num_in_col
    
    # Calculate the total number of reductions in a Column
    op_performance.numRedCol = op_performance.numRedOutCol * op_performance.numOutCol
    
    # Calculate the number of outputs per PE
    op_performance.numOutPE = op_performance.numColPE * op_performance.numOutCol
    
    # Calculate the number of outputs in the entire system
    if (device_type == 0):
        op_performance.numOutSys = op_performance.numOutPE * op_performance.numPESys
    elif (device_type == 1):
        op_performance.numOutSys = op_performance.numPESys * op_performance.numOutCol
    
    ## Calculate the corresponding output transmission cost
    op_performance.outTransCost = (float)(op_performance.numOutSys * arch_info.dataWidth) / (float)(ChannelBandWidth)
    
    # Calculate the number of inputs per PE
    op_performance.numInPE = op_performance.numInCol * op_performance.numColPE
    
    # Calculate the numebr of inputs in the entire system
    op_performance.numInSys = op_performance.numInPE * op_performance.numPESys
    
    ## Calculate the corresponding input loading cost
    op_performance.inLoadingCost = (float)(op_performance.numInSys * arch_info.dataWidth) / (float)(ChannelBandWidth)
    
    # If the target is HBM-PIM
    if (device_type == 1):
        filPELoading = op_performance.numColPE * op_performance.numFilCol
        filPELoading *= (float)(arch_info.dataWidth) / (float)(arch_info.PEBandWidth)
        op_performance.numFilPELoading = filPELoading
    
    # Calculate the final data cost
    if (device_type == 0):
        # Bit-serial PIM
        final_performance = (op_performance.numMulCol * arch_info.mulLat)
        final_performance += op_performance.numRedCol * arch_info.addLat
        final_performance += op_performance.outTransCost
        final_performance += op_performance.inLoadingCost
        op_performance.finalPerformance = final_performance
    elif (device_type == 1) :
        final_performance = op_performance.inLoadingCost
        final_performance += op_performance.numFilCol * arch_info.rowAct
        final_performance += op_performance.numFilPELoading
        final_performance += op_performance.outTransCost
        op_performance.finalPerformance = final_performance
        
    # Print all variables
    # op_performance.print_variables()
        
    return op_performance


def cal_performance_conv2D(arch_info, loop_bounds, trans_coeffs, stride, dilation, device_type):
    """
        This function calculates the final performance of the given transformed loop
        Needed Inputs:
            -loop_bounds:
                loop_bounds infomration for all transformed loop bounds.
                It should be a two dimensional list [x][l], x is the corresponding loop variable
                l is the corresponding loop level. l = 0, loop level 0;
            -trans_coeffs:
                Store the coefficients for different loop variables
                indexed by [x][l]
            -device_type:
                0. PUM --> bit-serial PIM
                1. PNM --> HBM-PIM
        
    """
    # Create the storing structure for the performance value
    op_performance = OpPerf("Conv2D")
    
    # Calculate the number of Multiplications in a column
    num_mul_col = 1
    ## Multiply all transformed loop bounds at level 0
    for x in range(0, NUM_BOUND_CONV2D):
        num_mul_col *= loop_bounds[x][0]
        
    op_performance.numMulCol = num_mul_col
    
    # Calculate the numebr of columns in a PE
    num_col_PE = 1
    ## Multiply all transformed loop bounds at level 1
    for x in range(0, NUM_BOUND_CONV2D):
        num_col_PE *= loop_bounds[x][1]
        
    op_performance.numColPE = num_col_PE
    
    # Calculate the number of PEs in a Sys
    num_PE_Sys = 1
    ## Multiply all transformed loop bounds at level 2
    for x in range(0, NUM_BOUND_CONV2D):
        num_PE_Sys *= loop_bounds[x][2]
        
    op_performance.numPESys = num_PE_Sys

    num_channel_Sys = int((num_PE_Sys - 1) // 16 + 1)
    # print("haha", num_channel_Sys)
    ChannelBandWidth = arch_info.SysBandWidth * num_channel_Sys
        
    # Calculate the number of outputs per column
    # NumOutCol = N * P * Q * K at loop level 0
    num_out_col = 1
    num_out_col *= loop_bounds[loop_bound_name_conv2D["N"]][0]
    num_out_col *= loop_bounds[loop_bound_name_conv2D["P"]][0]
    num_out_col *= loop_bounds[loop_bound_name_conv2D["Q"]][0]
    num_out_col *= loop_bounds[loop_bound_name_conv2D["K"]][0]
    
    op_performance.numOutCol = num_out_col
    
    # Calculate the number of filters per column
    # NumFilCol = K * R * S * C at loop level 0
    num_fil_col = 1
    num_fil_col *= loop_bounds[loop_bound_name_conv2D["K"]][0]
    num_fil_col *= loop_bounds[loop_bound_name_conv2D["R"]][0]
    num_fil_col *= loop_bounds[loop_bound_name_conv2D["S"]][0]
    num_fil_col *= loop_bounds[loop_bound_name_conv2D["C"]][0]
    
    op_performance.numFilCol = num_fil_col
    
    # Calculate the number of reductions per output in a colun
    # R * S * C - 1; at loop level 0
    num_red_out_col = 1
    num_red_out_col *= loop_bounds[loop_bound_name_conv2D["R"]][0]
    num_red_out_col *= loop_bounds[loop_bound_name_conv2D["S"]][0]
    num_red_out_col *= loop_bounds[loop_bound_name_conv2D["C"]][0]
    num_red_out_col -= 1
    
    op_performance.numRedOutCol = num_red_out_col
    
    # Calculate the number of inputs per column
    num_in_col = 1
    num_in_col *= loop_bounds[loop_bound_name_conv2D["N"]][0]
    
    # Calculate the unique number of p * Wstride + r * Wdilation
    coeff_1 = trans_coeffs[loop_bound_name_conv2D["P"]][0] * stride
    coeff_2 = trans_coeffs[loop_bound_name_conv2D["R"]][0] * dilation
    in_term_1 = count_unique(coeff_1, coeff_2, loop_bounds[loop_bound_name_conv2D["P"]][0], loop_bounds[loop_bound_name_conv2D["R"]][0])
    num_in_col *= len(in_term_1)
    
    coeff_3 = trans_coeffs[loop_bound_name_conv2D["Q"]][0] * stride
    coeff_4 = trans_coeffs[loop_bound_name_conv2D["S"]][0] * dilation
    in_term_2 = count_unique(coeff_3, coeff_4, loop_bounds[loop_bound_name_conv2D["Q"]][0], loop_bounds[loop_bound_name_conv2D["S"]][0])
    num_in_col *= len(in_term_2)
    num_in_col *= loop_bounds[loop_bound_name_conv2D["C"]][0]

    op_performance.numInCol = num_in_col
    
    # Calculate the total number of reductions in a Column
    op_performance.numRedCol = op_performance.numRedOutCol * op_performance.numOutCol
    
    # Calculate the number of outputs per PE
    op_performance.numOutPE = op_performance.numColPE * op_performance.numOutCol
    
    # Calculate the number of outputs in the entire system
    if (device_type == 0):
        op_performance.numOutSys = op_performance.numOutPE * op_performance.numPESys
    elif (device_type == 1):
        op_performance.numOutSys = op_performance.numPESys * op_performance.numOutCol
    
    ## Calculate the corresponding output transmission cost
    op_performance.outTransCost = (float)(op_performance.numOutSys * arch_info.dataWidth) / (float)(ChannelBandWidth)
    
    # Calculate the number of inputs per PE
    op_performance.numInPE = op_performance.numInCol * op_performance.numColPE
    
    # Calculate the numebr of inputs in the entire system
    op_performance.numInSys = op_performance.numInPE * op_performance.numPESys
    
    ## Calculate the corresponding input loading cost
    op_performance.inLoadingCost = (float)(op_performance.numInSys * arch_info.dataWidth) / (float)(ChannelBandWidth)
    op_performance.inLoadingCost += op_performance.numInCol * arch_info.rowAct
    
    # If the target is HBM-PIM
    if (device_type == 1):
        # zmac: Filter can be loaded multiple times
        # filPELoading = op_performance.numColPE * op_performance.numFilCol
        filPELoading = op_performance.numColPE * op_performance.numMulCol
        filPELoading *= (float)(arch_info.dataWidth) / (float)(arch_info.PEBandWidth)
        op_performance.numFilPELoading = filPELoading
    
    # Calculate the final data cost
    if (device_type == 0):
        # Bit-serial PIM
        final_performance = (op_performance.numMulCol * arch_info.mulLat)
        final_performance += op_performance.numRedCol * arch_info.addLat
        final_performance += op_performance.outTransCost
        final_performance += op_performance.inLoadingCost
        op_performance.finalPerformance = final_performance
    elif (device_type == 1) :
        final_performance = op_performance.inLoadingCost
        # zmac: Filter can be loaded multiple times
        # final_performance += op_performance.numFilCol * arch_info.rowAct
        final_performance += op_performance.numMulCol * arch_info.rowAct
        final_performance += op_performance.numFilPELoading
        final_performance += op_performance.outTransCost
        op_performance.finalPerformance = final_performance
        
    # Print all variables
    # op_performance.print_variables()
        
    return op_performance

class TraceParser:
    def __init__(self, file_path):
        """
        Initialize the parser with the file path and parse the file.
        
        Parameters:
        - file_path (str): Path to the input file.
        """
        self.file_path = file_path
        self.operation_type = None
        self.problem = []
        self.dilation_stride = []
        self.loop = []
        self.bound = []
        self.tag = []
        self.start_bank_row = []
        self.coefficients = {}
        
        self.parse_file()
        
    def parse_file(self):
        """
        Parse the input file and store the data in class attributes.
        """
        with open(self.file_path, 'r') as file:
            for line in file:
                # Remove leading and trailing whitespace
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check for the end of the file
                if line.lower() == 'end':
                    break
                
                # Parse the operation type (e.g., conv2d)
                if line.lower() == 'conv2d':
                    self.operation_type = line
                    continue
                elif (line.lower() == 'gemm'):
                    self.operation_type = line
                    continue
                
                # Split the line into key and value
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle each key accordingly
                    if key == 'Problem':
                        self.problem = [int(x) for x in value.split(',')]
                    elif key == 'DilationStride':
                        self.dilation_stride = [int(x) for x in value.split(',')]
                    elif key == 'Loop':
                        self.loop = [x.strip() for x in value.split(',')]
                    elif key == 'Bound':
                        self.bound = [int(x) for x in value.split(',')]
                    elif key == 'Tag':
                        self.tag = [x.strip() for x in value.split(',')]
                    elif key == 'StartBankRow':
                        self.start_bank_row = [int(x) for x in value.split(',')]
                    elif key.startswith('Coeff_'):
                        coeff_name = key[len('Coeff_'):]
                        coeff_values = [int(x) for x in value.split(',')]
                        self.coefficients[coeff_name] = coeff_values
                    else:
                        # Handle any other keys if necessary
                        pass
                else:
                    # Handle lines without a colon if necessary
                    pass
                
    def get_loop_bounds(self):
        """
            This function returns the needed loop bound list for the analytical model
            The output file is stored with the following order
            Level 2 -> Level 0 -> Level 1
            We need to convert it back to store it as
            Level 0 -> Level 1 -> Level 2
        """
        loop_bounds = []
        num_loops = 1
        
        if (self.operation_type == "conv2d"):
            # Conv2D operator
            num_loops = 7
        elif (self.operation_type == "gemm"):
            # FC operator
            num_loops = 5
        
        # Iterate over all loop variables
        for i in range(num_loops):
            tmp_loop_bound_list = []
            
            tmp_loop_bound_list.append(self.bound[i + num_loops])
            tmp_loop_bound_list.append(self.bound[i + num_loops * 2])
            tmp_loop_bound_list.append(self.bound[i])
            
            loop_bounds.append(copy.deepcopy(tmp_loop_bound_list))
            
        return loop_bounds
    
    def get_trans_coeff(self):
        """
            This function returns the list of the transcoeff
            The same as above, the order is stored as (order used in the simulator)
                Level 2 -> Level 0 -> Level 1
            We need to convert it back to:
                Level 0 -> Level 1 -> Level 2
        """
        trans_coeff = []
        
        # Check the storing order
        index_list = []
        N_string = list(self.coefficients.keys())[0]
        
        N_list = N_string.split(",")
        
        if (N_list[1] == "N0"):
            index_list = [1, 2, 0]
        elif (N_list[1] == "N1"):
            index_list = [2, 1, 0]
        
        for key, value in self.coefficients.items():
            tmp_trans_coeff = []
            tmp_trans_coeff.append(value[index_list[0]])
            tmp_trans_coeff.append(value[index_list[1]])
            tmp_trans_coeff.append(value[index_list[2]])
            
            trans_coeff.append(copy.deepcopy(tmp_trans_coeff))
            
        return trans_coeff

    def __repr__(self):
        """
        Return a string representation of the parsed data for easy debugging.
        """
        return (
            f"Operation Type: {self.operation_type}\n"
            f"Problem: {self.problem}\n"
            f"DilationStride: {self.dilation_stride}\n"
            f"Loop: {self.loop}\n"
            f"Bound: {self.bound}\n"
            f"Tag: {self.tag}\n"
            f"StartBankRow: {self.start_bank_row}\n"
            f"Coefficients: {self.coefficients}\n"
        )



if __name__ == "__main__":
        
    # Create an instance of the parser with the file path
    parser = TraceParser('/home/jianliu/Projects/Datalayout/MLIR-PIM/Result/Group.txt')

    # Or print all data at once
    print(parser)
    print(parser.get_trans_coeff())
    print(parser.get_loop_bounds())
    print(parser.dilation_stride)

