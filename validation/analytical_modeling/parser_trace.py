# This file defines the parser to parse the trace file
import copy

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
            The same as above, the order is stored as
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

