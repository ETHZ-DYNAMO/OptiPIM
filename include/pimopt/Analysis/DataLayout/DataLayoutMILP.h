#ifndef PIMOPT_DATALAYOUT_MILP
#define PIMOPT_DATALAYOUT_MILP

#include "pimopt/Analysis/DataLayout/DataLayoutSupport.h"

// Check the existence of Gurobi library
#ifndef PIMOPT_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace pim {

//
//  Data structures to store all constant information
//

// Struct that stores all the inforamtion (constants) related to a loop bound in the input nested loop (operator)
struct LoopBoundInfo {
    int32_t loopBoundValue;

    // Vector storing all the divisors of the loopBoundValue
    std::vector<int64_t> divisorsVec;

    // Vector storing the log2() value of all divisors
    std::vector<double_t> divisorsLogVec;

    // =================
    // Helper Functions
    // =================
    void printDetail() {
        llvm::outs() << "\t\tDiv Vector: [ ";

        for (auto divisor : divisorsVec) {
            llvm::outs() << divisor << " ";
        }

        llvm::outs() << "];\n";
    }
};

// Struct storing all constant information related to an operator
struct OpConstInfo {
    int32_t opID;

    // Number of PEs allocated for this op
    int64_t numPE;

    // Op Type
    int32_t opType;

    // Maximum Loop Bound value
    int64_t maxLoopBound;

    // Map from loop bound index to the correponding storing structure
    std::map<int32_t, LoopBoundInfo> loopBoundInfoMap;

    // Flag indicating whether the LUT for GCD values is needed
    std::vector<bool> needGCDLUTs{false, false};

    // LUT for the gcd value of the input array transformation coefficients
    // We ues a vector to represent all possible gcd values
    std::vector<CartesianProduct<int64_t>> inputCoeffCatSets;
    std::vector<std::vector<int64_t>> inputCoeffLUTs;

    // =================
    // Helper Functions
    // =================
    void printDetail() {
        llvm::outs() << "Op ID: " << opID << "\n";

        // Print Workload details
        llvm::outs() << "\t[OP LOOP BOUNDS INFO]\n";
        if (opType == operatorTypesIdx::CONV2D) {
            for (int i = 0; i < NUM_BOUND_CONV2D; i++) {
                llvm::outs() << "\t" << loopBoundNameVerboseConv2D[i] << "(" << loopBoundNameConv2D[i] << "): " << loopBoundInfoMap[i].loopBoundValue << "\n";

                loopBoundInfoMap[i].printDetail();
            }

            llvm::outs() << "\n";

            // IA LUT size
            llvm::outs() << "\t[LOOK UP TABLE SIZE]\n";
            for (int i = 0; i < 2; ++i) {
                llvm::outs() << "\t\t" << inputArrayLUTNameConv2D[i] << " LUT Size: " << inputCoeffLUTs[i].size() << "\n";
            }

            // IA LUT Contents
            llvm::outs() << "\t[LOOK UP TABLE CONTENTS]\n";
            // PR_TABLE
            llvm::outs() << "\t\t" << inputArrayLUTNameConv2D[0] << "[";
            for (int i = 0; i < inputCoeffLUTs[0].size(); i++) {
                llvm::outs() << inputCoeffLUTs[0][i] << " ";
            }
            llvm::outs() << "]\n";
            // QS_TABLE
            llvm::outs() << "\t\t" << inputArrayLUTNameConv2D[1] << "[";
            for (int i = 0; i < inputCoeffLUTs[1].size(); i++) {
                llvm::outs() << inputCoeffLUTs[1][i] << " ";
            }
            llvm::outs() << "]\n";

            // IA LUT Flags
            llvm::outs() << "\t[NEED LOOK UP TABLE FLAGS]\n";
            for (int i = 0; i < needGCDLUTs.size(); ++i) {
                llvm::outs() << "\t\t" << inputArrayLUTNameConv2D[i] << " Flag: " << needGCDLUTs[i] << "\n";
            }

        } else if (opType == operatorTypesIdx::FC) {
            for (int i = 0; i < NUM_BOUND_FC; i++) {
                llvm::outs() << "\t" << loopBoundNameVerboseFC[i] << "(" << loopBoundNameFC[i] << "): " << loopBoundInfoMap[i].loopBoundValue << "\n";

                loopBoundInfoMap[i].printDetail();
            }

            llvm::outs() << "\n";

        }
    }
};


//
//  Data structures to store GRB Variables
//

// Struct to store MILP variables for a single loop bound in the original loop (per original loop bound)
struct LoopBoundVariables {
    // Binary variables indicate which divisor is selected at a specific loop level for a loop bound
    // Represented as a two dimension array
    // E.x. [n][k], loop level n, divisor k
    std::vector<std::vector<GRBVar>> loopBoundBinaryVars;

    // Integer variables represent the actual transformed loop bound value at different loop level
    // Represented as a single vector
    // E.x. [n], loop level n
    std::vector<GRBVar> loopBoundIntVars;
};

// Struct to store MILP variables for the transformation coefficients for a single loop variable
struct TransCoeffVariables {
    // Binary variables indicating which coefficient combination
    // is selected for the corresponding loop variable
    // We will have maximum n! coefficient combinations, where n is the numebr of loop levels
    // The variable is stored in a vector
    // E.x. [f] combination index f
    std::vector<GRBVar> coeffBinaryVars;

    // Integer variables representing the actual value of the corresponding coefficient
    // in the transformation function of a loop variable
    // E.x. [n], transformation coefficient for loop level n
    std::vector<GRBVar> coeffIntVars;
};

// Struct to store all MILP variables for a single Op
struct OpGRBVariables {
    // Vector storing the variables for different loop bounds
    // [d], all related GRB variables for loop bound d
    std::vector<LoopBoundVariables> loopBoundsVars;

    // Vector stroing all variables for the transformation coefficients of different loop variables
    // [m], all related GRB variables for the coefficients of loop variable m
    std::vector<TransCoeffVariables> loopCoeffsVars;

    // Vector storing the LUT selection variable
    std::vector<std::vector<GRBVar>> LUTSelVars;

    //
    // Following variables represent the column level informaiton
    //

    // Variable represent the number of multiplications in a column
    GRBVar numMulCol;

    // Variable represent the number of output elements in a column
    GRBVar numOutCol;

    // Variable represent the number of filter elements in a column
    GRBVar numFilCol;

    // Variable represent the number of input elements in a column
    GRBVar numInCol;

    // Add temp variabel for debugging
    GRBVar numInTerm2Col;

    GRBVar numInTerm3Col;

    // Variable represent the number of reductions for a single output in a column
    GRBVar numRedOutCol;

    // Variable represent the number of reductions in a column
    GRBVar numRedCol;

    //
    // Following variables represent the PE level informaiton
    // 

    // Variable represent the number of columns used in the PE
    GRBVar numColPE;

    // Variable represent the number of Output in a PE
    GRBVar numOutPE;

    // Variable represent the number of Input in a PE
    GRBVar numInPE;

    //
    // Following variables represent the system level informaiton
    // 

    // The number of PEs used by the workload
    GRBVar numPESys;

    // The number of output transmissions in the whole system
    GRBVar numOutSys;

    // The number of input loading in the whole system
    GRBVar numInSys;

    //
    //  HBM-PIM
    //
    GRBVar numFilPELoading;

    //
    //  Variables representing the selected GCD values
    //
    GRBVar PRGCDValue;
    GRBVar QSGCDValue;
};

// Class for the Datalayout MILP modeling
class DataLayoutMILP {
public:

    // Define class constructor
    DataLayoutMILP(GRBEnv &env, std::vector<int32_t> &opGroup,
                    std::map<int32_t, OperatorInfo> &opsInfoDB,
                    const std::string& logPath, const std::string& resultPath,
                    const std::string& archPath, const std::string& knobPath,
                    const int32_t& deviceType, const int32_t& memoryAllocScheme,
                    const int32_t& objMethod, const int32_t& opGroupID,
                    const int32_t& transCoeffMethod, const int32_t& numCountingMethod,
                    const int32_t& storageMethod);    
    
    // =============================================
    //          Overall GRB Construction Func
    // =============================================

    // Top level interface to setup all MILP related configs for Conv2D operator
    GRBLinExpr constructConv2DMILP(const OperatorInfo& selOpInstance, const int32_t& selOpIndex);

    // Top level interface to setup all MILP related configs for FC operator
    GRBLinExpr constructFCMILP(const OperatorInfo& selOpInstance, const int32_t& selOpIndex);

    // =============================================
    //      GRB Variable Construction -- Conv2D
    // =============================================

    // Add all variables related to loop bounds in the resulting nested loop for a single Op
    // Binary Variable naming: b_d_n_k_O_i, which means: loop_bound d, loop level n, divisor k in Operator i
    // Integer Variabel naming: L_d_n_O_i, which means: loop bound d, loop level n in Operator i
    void addLoopBoundVarsConv2D(const int32_t& selOpIdx);

    // Add all variables related to transformation coefficients for conv2D
    // In the Conv2D workload, we need to create the variables for the following 4 loop variables
    // P, R, Q, S
    // 6 binary variabels are needed for each of the loop variable indicating which combination is selected
    // Besides that, we also need multiple integer varibale to indicate the actual value of the transformation coefficient
    void addTransCoeffVarsConv2D(const int32_t& selOpIdx);

    // Add variable representing the numebr of multiplications in a column
    void addNumMulColVarConv2D(const int32_t& selOpIdx);

    // Add variable representing the numebr of columns in a PE
    void addNumColPEVarConv2D(const int32_t& selOpIdx);

    // Add variable representing the numebr of PEs in a system
    void addNumPESysVarConv2D(const int32_t& selOpIdx);

    // Add variable represent the number of outputs per column
    void addNumOutColVarConv2D(const int32_t& selOpIdx);

    // Add variable represent the number of filters per column
    void addNumFilColVarConv2D(const int32_t& selOpIdx);

    // Add variable represent the numebr of reductions per output in a column
    void addNumRedOutColVarConv2D(const int32_t& selOpIdx);

    // Add all related variables and constraints to represent
    // the numebr of inputs in a column
    void addNumInColVarConv2D(const int32_t& selOpIdx);

    // Add variable to represent the number of reductions per column
    void addNumRedColVarConv2D(const int32_t& selOpIdx);

    // Add variable to represent the numebr of output partial sums per PE
    void addNumOutPEVarConv2D(const int32_t& selOpIdx);

    // Add variable to represent the number of output transmissions
    void addNumOutSysVarConv2D(const int32_t& selOpIdx);

    // Add variable to represent the number of inputs in a PE
    void addNumInPEVarConv2D(const int32_t& selOpIdx);

    // Add variable to represent the number of input loading in the system
    void addNumInSysVarConv2D(const int32_t& selOpIdx);

    //
    //  PNM Specific Cost Calculation
    //

    void addFilPELoadingVarConv2D(const int32_t& selOpIdx);

    // =============================================
    //      GRB Constraints Construction -- Conv2D
    // =============================================

    // Common Constraint
    // Add loop bound constraints for all loop bounds and memory level
    // Only one value can be taken for all related binary variables in loop bound d and mem level m
    void addLoopBoundConsConv2D(const int32_t& selLayer);

    // Common Constraint
    // Add transformation coefficient constraints, only one combination can be selected
    void addTransCoeffConsConv2D(const int32_t& selOp);

    // Common Constraint
    // Add number of PE constraint
    // For now we directly use the per layer bank number limit in the input mlir file
    void addNumPEConsConv2D(const int32_t& selOp);

    // Common Constraint
    // Add number of Columns constraint
    // For now we directly use the limit from the arch file
    void addNumColConsConv2D(const int32_t& selOp);

    // Common Constraint
    // Add column size constraint
    // Size of OA + F + IA < size column
    void addColSizeConsConv2D(const int32_t& selOp);

    // =============================================
    //        Print Detailed Output -- Conv2D
    // =============================================

    // Print the details of the transformed nested loop Conv2D
    void printConv2DMILPResult(const int32_t& selOpIndex);

    // Dump the results to a file for a single layer
    std::string traceOutputConv2D(const int32_t& selOpIndex, const OperatorInfo& oriOpInfo);

    // Dump the results to a file for a single layer with a format specific for the simulator
    // Change the position of Level 1 and Level 0
    std::string traceOutputSIMConv2D(const int32_t& selOpIndex, const OperatorInfo& oriOpInfo);

    // =============================================
    //       GRB Variable Construction -- FC
    // =============================================

    // Add all variables related to loop bounds in the resulting nested loop for a single Op
    // Binary Variable naming: b_d_n_k_O_i, which means: loop_bound d, loop level n, divisor k in Operator i
    // Integer Variabel naming: L_d_n_O_i, which means: loop bound d, loop level n in Operator i
    void addLoopBoundVarsFC(const int32_t& selOpIdx);

    // Add variable representing the numebr of multiplications in a column
    void addNumMulColVarFC(const int32_t& selOpIdx);

    // Add variable representing the numebr of columns in a PE
    void addNumColPEVarFC(const int32_t& selOpIdx);

    // Add variable representing the numebr of PEs in a system
    void addNumPESysVarFC(const int32_t& selOpIdx);

    // Add variable represent the number of outputs per column
    void addNumOutColVarFC(const int32_t& selOpIdx);

    // Add variable represent the number of filters per column
    void addNumFilColVarFC(const int32_t& selOpIdx);

    // Add variable represent the numebr of reductions per output in a column
    void addNumRedOutColVarFC(const int32_t& selOpIdx);

    // Add all related variables and constraints to represent
    // the numebr of inputs in a column
    void addNumInColVarFC(const int32_t& selOpIdx);

    // Add variable to represent the number of reductions per column
    void addNumRedColVarFC(const int32_t& selOpIdx);

    // Add variable to represent the numebr of output partial sums per PE
    void addNumOutPEVarFC(const int32_t& selOpIdx);

    // Add variable to represent the number of output transmissions
    void addNumOutSysVarFC(const int32_t& selOpIdx);

    // Add variable to represent the number of inputs in a PE
    void addNumInPEVarFC(const int32_t& selOpIdx);

    // Add variable to represent the number of input loading in the system
    void addNumInSysVarFC(const int32_t& selOpIdx);

    //
    //  PNM Specific Cost Calculation
    //
    void addFilPELoadingVarFC(const int32_t& selOpIdx);

    // =============================================
    //      GRB Constraints Construction -- FC
    // =============================================

    // Common Constraint
    // Add loop bound constraints for all loop bounds and memory level
    // Only one value can be taken for all related binary variables in loop bound d and mem level m
    void addLoopBoundConsFC(const int32_t& selLayer);

    // Common Constraint
    // Add number of PE constraint
    // For now we directly use the per layer bank number limit in the input mlir file
    void addNumPEConsFC(const int32_t& selOp);

    // Common Constraint
    // Add number of Columns constraint
    // For now we directly use the limit from the arch file
    void addNumColConsFC(const int32_t& selOp);

    // Common Constraint
    // Add column size constraint
    // Size of OA + F + IA < size column
    void addColSizeConsFC(const int32_t& selOp);

    // =============================================
    //           Print Detailed Output -- FC
    // =============================================

    // Print the details of the transformed nested loop FC
    void printFCMILPResult(const int32_t& selOpIndex);

    // Dump the results to a file for a single layer
    std::string traceOutputFC(const int32_t& selOpIndex, const OperatorInfo& oriOpInfo);

    // Dump the results to a file for a single layer with a format specific for the simulator
    // Change the position of Level 1 and Level 0
    std::string traceOutputSIMFC(const int32_t& selOpIndex, const OperatorInfo& oriOpInfo);

    // =============================================
    //          Constant Construction Func
    // =============================================
    
    // Function to Extract all constants for a given CONV 2D op
    void buildConv2DOpConstants(const OperatorInfo& selOpIns);

    // Function to Extract all constants for a given FC op
    void buildFCOpConstants(const OperatorInfo& selOpIns);

    // Function returns a vector of all divisors of a given integer
    // For example, if dimension = 64, the vector will be [1, 2, 4, 8, 16, 32, 64]
    std::vector<int64_t> getAllDivisors(const int32_t& loopBound);

    // Function used to calculate the cartesian product
    template<typename T>
    CartesianProduct<T> calCartesian(const std::vector<Set<T>>& sets) {
        CartesianProduct<T> result;
        std::vector<int32_t> indices(sets.size(), 0);

        size_t totalElements = 1;

        // Calculate the total number of elements in the result
        for (const auto& set : sets) {
            totalElements *= set.size();
        }

        result.reserve(totalElements);

        // Calculate all possible combinations
        while (true) {
            // Add current combination
            std::vector<T> tuple(sets.size());
            for (size_t i = 0; i < sets.size(); ++i) {
                tuple[i] = sets[i][indices[i]];
            }
            result.push_back(tuple);

            // Increment indices
            size_t next = sets.size() - 1;
            while (++indices[next] == sets[next].size()) {
                indices[next] = 0;
                if (next == 0) break;
                --next;
            }

            // Check if we have completed all combinations
            if (next == 0 && indices[0] == 0) break;
        }

        return result;
    }

    // Function used to calculate GCD of two integer inputs
    int64_t calGCD(const int64_t& inputOne, const int64_t& inputTwo);
    
    // Function used to calculate the actual GCD value in the LUT table of Conv2D operator
    std::vector<int64_t> calConv2DIALUTValue(const CartesianProduct<int64_t>& selCaPr,
                                                const std::vector<int64_t>& opStride, const std::vector<int64_t>& opDilation);

    // Get the corresponding transformation coefficients for a given loop variable
    std::string getCoeffVar(const int32_t& loopLevel, const int32_t& selLoopVarconst, const int32_t& selOp);

    // =============================================
    //                 Helper Functions
    // =============================================

    // Print all tuples in a given cartesian product set
    template<typename T>
    void printCartesianProduct(const CartesianProduct<T>& selSet) {
        llvm::outs() << "{\n";
        for (size_t i = 0; i < selSet.size(); ++i) {
            llvm::outs() << "\t(";
            for (size_t j = 0; j < selSet[i].size(); ++j) {
                llvm::outs() << selSet[i][j];
                if (j != selSet[i].size() - 1) llvm::outs() << ", ";
            }
            
            llvm::outs() << ")"; 
            if (i != selSet.size() - 1) llvm::outs() << ",\n";
        }

        llvm::outs() << "\n}\n";
    }

    // Write out the MILP model
    // Call this function before model.optimize()
    void writeMILPModel(GRBModel& model, const std::string& logPath, const int32_t& opGroupID);

    // Write out the MILP solution
    // Call this function after model.optimize()
    void writeMILPSol(GRBModel& model, const std::string& logPath, const int32_t& opGroupID);

    // Read the content of the json file into a string, run this before parsing the json file
    std::string readJsonFile(const std::string& filePath);
    
    // Parse the JSON file string
    llvm::Expected<llvm::json::Value> parseJson(const std::string& jsonString);

    // Parse the knobs.json file and construct the Knobs struct 
    void parseKnobsJSON(const std::string& filePath);

    // Parse the architecture.json file
    void parseArchJSON(const std::string& filePath);
    
    // ==========================
    //     Variable Defintion
    // ==========================

    // Map from op index to the storing instances of the op's constants info
    std::map<int32_t, OpConstInfo> opConstInfoMap;
    // Map from op index to variable holding instance
    std::map<int32_t, OpGRBVariables> opVariablesMap;

    // Define the GRBModel
    GRBModel model;

    // Store all Arch info
    ArchInfo archInfo;

    // Store weights for different parts in the final objective function
    Knobs objKnobs;

    // ====================
    //     User Configs
    // ====================

    // Default device type set to bit-serial PUM
    int32_t selDeviceType = deviceTypeIdx::PUM;

    // Memory allocation scheme
    // Only PNM device can use the combined allocation scheme
    int32_t memoryAlloc = memAllocIdx::Exclusive;

    // Type of objective function
    // 0. The cost function we proposed in the paper
    // 1. The cost function used by COSA (with out layout cost)
    int32_t selObjMethod = 0;

    // Transformation coefficient calculation method
    int32_t selTransCoeffMethod = transCoeffMethodIdx::Flexible;

    // number conunting method
    int32_t selNumCountingMethod = 0;

    // Storage Method
    int32_t selStorageMethod = 0;

    // Settings for the gurboi piecewise linearization function
    std::string GurobiApproximationOptions = "FuncPieces=-2 FuncPieceError=0.002";

    // Path to json files: memory architecture and tuning knobs
    std::string selArchPath;
    std::string selKnobPath;

    //
    // Following constants are used for normalization
    //
    // TODO: Need to think about the following normalization, this may not be needed
    double_t ScaleBase = 10000;
    double_t maxMacNum;
    double_t maxMacScaleFactor;
    double_t maxMacScaleFactorLog;
    std::map<int32_t, int32_t> perOpMacValue;
    std::map<int32_t, double_t> perOpMacScaleFactor;
    std::map<int32_t, double_t> perOpMacScaleFactorLog; 

};

}   // namespace pim

#endif // PIMOPT_GUROBI_NOT_INSTALLED

#endif  // PIMOPT_DATALAYOUT_MILP