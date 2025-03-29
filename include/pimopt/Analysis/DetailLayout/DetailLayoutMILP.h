#ifndef PIMOPT_DETAILLAYOUT_MILP
#define PIMOPT_DETAILLAYOUT_MILP

#include "pimopt/Analysis/DetailLayout/DetailLayoutSupport.h"
// #include "pimopt/Support/MILP.h"

// Check the existence of Gurobi library
#ifndef PIMOPT_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace pim {

// Struct that stores all information related to a input workload dimension of a layer
// For example, if the workload dimension C
struct WorkLoadDimInfo {
    int32_t workLoadDim;

    // Vector storing all its divisors
    std::vector<int64_t> divisorsVec;

    // Log format of all divisors
    // TODO: Check the precision, for now we use double, but this may not be necessary
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

        // Log Format
        // llvm::outs() << "\t\tDiv Vector Log2: [ ";

        // for (auto divisor : divisorsLogVec) {
        //     llvm::outs() << divisor << " ";
        // }

        // llvm::outs() << "];\n";
    }
};

// Struct storing all workload dimension info (constants for MILP) related to a single Layer
struct ConstInfo {
    int32_t layerId;

    // Number of banks used by this layer
    int64_t numBanks;

    // Layout Scheme
    int32_t layoutScheme;

    // Layer Type
    int32_t layerType;

    // Map from workload dimension index to the actual storing structure
    std::map<int32_t, WorkLoadDimInfo> workLoadDimMap;

    // Vector of Cartesian Sets for all three tensor types: W, OA and IA
    std::vector<CartesianProduct<int64_t>> tensorCartSets;

    // Vector of all possible values for all three tensor types: W, OA and IA
    // In the same order as the corresponding Cartesian Product Set respectively
    std::vector<std::vector<int64_t>> tensorValues;

    std::vector<int64_t> totalTensorSize; // Vector Storing the tensor size at top memory level -- Constants

    // =================
    // Helper Functions
    // =================
    void printDetail() {
        llvm::outs() << "Layer ID: " << layerId << "\n";

        // Print Workload details
        llvm::outs() << "\t[WORKLOAD DIM INFO]\n";
        if (layerType == layerTypeAll::CONV2D) {
            for (int i = 0; i < NUM_DIM_CONV2D; i++) {
                llvm::outs() << "\t" << WorkLoadDimComNames[i] << "(" << WorkLoadDimSimNames[i] << "): " << workLoadDimMap[i].workLoadDim << "\n";

                workLoadDimMap[i].printDetail();
            }

            llvm::outs() << "\n";

            // Top mem level tensor size
            llvm::outs() << "\t[TOP LEVEL TENSOR SIZES]\n";
            for (int i = 0; i < 3; ++i) {
                llvm::outs() << "\t" << TensorTypeNames[i] << " Top Level Size: " << totalTensorSize[i] << "\n";
            }

            llvm::outs() << "\n";
            
            // tensorCartSets size
            llvm::outs() << "\t[CARTESIAN PRODUCT SIZE]\n";
            for (int i = 0; i < 3; ++i) {
                llvm::outs() << "\t" << TensorTypeNames[i] << " Cartesian Product Size: " << tensorCartSets[i].size() << "\n";
            }
        } else if (layerType == layerTypeAll::FC) {
            for (int i = 0; i < NUM_DIM_FC; i++) {
                llvm::outs() << "\t" << WorkLoadDimComNamesFC[i] << "(" << WorkLoadDimSimNamesFC[i] << "): " << workLoadDimMap[i].workLoadDim << "\n";

                workLoadDimMap[i].printDetail();
            }

            llvm::outs() << "\n";

            // Top mem level tensor size
            llvm::outs() << "\t[TOP LEVEL TENSOR SIZES]\n";
            for (int i = 0; i < 3; ++i) {
                llvm::outs() << "\t" << TensorTypeNamesFC[i] << " Top Level Size: " << totalTensorSize[i] << "\n";
            }

            llvm::outs() << "\n";
            
            // tensorCartSets size
            llvm::outs() << "\t[CARTESIAN PRODUCT SIZE]\n";
            for (int i = 0; i < 3; ++i) {
                llvm::outs() << "\t" << TensorTypeNamesFC[i] << " Cartesian Product Size: " << tensorCartSets[i].size() << "\n";
            }
        }
        
    }
};

// Holds MILP variables related to a single layer 
struct LayerGRBVariables {
    // Output variables -- One hot encoding of the loop bound of each workload dimension at each memory level
    // Represented as a three dimension array
    // E.x. [i][j][k], workload dimension i, memory level j, divisor k
    std::vector<std::vector<std::vector<GRBVar>>> memLoopBoundOneHotVars;

    // Internal Variables -- One hot encoding of the selected value for tensor type i at memory level j.
    // Represented as a three dimension array, Note that we just need to represent Row Buffer and Row Level
    // Bank level information is aleady resolved -- Constants
    // E.x. [i][j][k], mem level i, tensor type j, potential value k
    std::vector<std::vector<std::vector<GRBVar>>> tensorValueOneHotVars;
    
};

// Class for all related information of the DetailLayout MILP problem
class DetailLayoutMILP {
public:
    // Define the constructor
    DetailLayoutMILP(GRBEnv &env, std::vector<int32_t> &layerGroup, 
                     std::map<int32_t, LayerInfo> &layersInfoDB,
                     const std::string& logPath, const std::string& resultPath,
                     const std::string& archPath, const std::string& knobPath,
                     const int32_t& dataLayoutScheme, const int32_t& memoryAllocScheme,
                     const int32_t& objApproMethod, const int32_t& layerGroupID);

    
    // =============================================
    //               Overall GRB Setting
    // =============================================
    GRBLinExpr setupConv2DGRB(const LayerInfo& selLayerInstance, const int32_t& selLayerIndex);

    GRBLinExpr setupFCGRB(const LayerInfo& selLayerInstance, const int32_t& selLayerIndex);
    
    // =============================================
    // GRB Variable Functions Declaration -- Conv2D
    // =============================================

    // Add all variables related to loop bounds in the resulting nested loop for a single layer
    // Variable Naming : L_l_X_i_j_k, which means: workload dimension i, memory level j, divisor k in Layer l
    // All variables are binary -- one hot encoding
    void addLoopBoundVarsConv2D(const int32_t& selLayer);

    // Add all variables related to tensor size one hot encoding for a single layer
    // Variable Naming : L_l_E_i_j_k, which means: mem level i, tensor type j, potential value k in Layer l
    // All variables are binary -- One hot encoding
    void addTensorSizeVarsConv2D(const int32_t& selLayer);

    // ====================================
    //       GRB Constraints -- Conv2D 
    // ====================================

    // Common Constraint
    // Add loop bound constraints for all workload dimension and memory level
    // Only one value can be taken for all variables in workload dim d and mem level m
    void addLoopBoundConsConv2D(const int32_t& selLayer);

    // Common Constraint
    // Add Workload constraints for all loop bound variables at each memory level
    // If we multiple all loop bounds at different mem level for a specific workload dim,
    // this should be equal to the original workload dim value
    void addWorkLoadConsConv2D(const int32_t& selLayer, const double_t& tolerance);

    // Common Constraint
    // Add number of bank constraint
    // For now we directly use the per layer bank number limit in the input mlir file
    void addNumBankConsConv2D(const int32_t& selLayer);

    // Common Constraint
    // Add Tensor value onehot constraint
    // For a given mem level and tensor type, only one value can be selected
    void addTensorValueConsConv2D(const int32_t& selLayer);

    // Common Constraint
    // Add Cartesian Product constraints
    // We need to make sure the Potential value one hot encoding variables aligns with the Loop Bound Variables
    // that means variables X[][][] match with E[][][].
    // The original format of the constraint includes log2 values, and we want the equation to be equal, we may face some precision issues
    // Thus, we relax the equal constraint to two inequality expressions (a <= b + tolerance) and (a >= b - tolerance)
    void addCartesianProductConsConv2D(const int32_t& selLayer, const double_t& tolerance);

    // Layout Scheme 1 Specific Constraint
    //   Total sensor size < Size of all banks
    void addScheme1ConsConv2D(const int32_t& selLayer);

    // Layout Scheme 1 Specific Constraint
    //   Tensor size per bank < Bank size
    void addScheme2ConsConv2D(const int32_t& selLayer);

    // Layout Scheme 1 Specific Constraint
    //   Tensor size in one row < Row Size
    void addScheme3ConsConv2D(const int32_t& selLayer);


    // ====================================
    //       GRB Objectives -- Conv2D
    // ====================================

    // Add computation objective function --> Num of cucles used for computation
    // This part is the same for all schemes
    GRBLinExpr addComputeObjectConv2D(const int32_t& selLayer);

    // Add Scheme 1 Row Act Objective
    GRBLinExpr addScheme1RowActObjectConv2D(const int32_t& selLayer);

    // Add Scheme 1 Cross Bank Loading Objective
    GRBLinExpr addScheme1CrossBankObjConv2D(const int32_t& selLayer);
    
    // Add Scheme 2 Row Act Objective
    GRBLinExpr addScheme2RowActObjectConv2D(const int32_t& selLayer);

    // Add Scheme 2 Cross Bank Loading Objective
    GRBLinExpr addScheme2CrossBankObjConv2D(const int32_t& selLayer);

    // Add Scheme 3 Row Act Objective
    GRBLinExpr addScheme3RowActObjectConv2D(const int32_t& selLayer);

    // Add Scheme 3 Intra Bank Loading Objective
    GRBLinExpr addScheme3IntraBankObjConv2D(const int32_t& selLayer); 

    // Add Scheme 3 Cross Bank Loading Objective
    GRBLinExpr addScheme3CrossBankObjConv2D(const int32_t& selLayer);

    // ====================================
    // GRB Objectives Norm + PWL -- Conv2D
    // ====================================
    // Add computation objective function --> Num of cucles used for computation
    // This part is the same for all schemes
    GRBLinExpr addComputeObjectConv2DPWLNorm(const int32_t& selLayer);

    // Add Scheme 1 Row Act Objective after normalization
    GRBLinExpr addScheme1RowActObjectConv2DPWLNorm(const int32_t& selLayer);

    // Add Scheme 1 Cross Bank Loading Objective after normalization
    GRBLinExpr addScheme1CrossBankObjConv2DPWLNorm(const int32_t& selLayer);
    
    // Add Scheme 2 Row Act Objective after normalization
    GRBLinExpr addScheme2RowActObjectConv2DPWLNorm(const int32_t& selLayer);

    // Add Scheme 2 Cross Bank Loading Objective after normalization
    GRBLinExpr addScheme2CrossBankObjConv2DPWLNorm(const int32_t& selLayer);

    // Add Scheme 3 Row Act Objective after normalization
    GRBLinExpr addScheme3RowActObjectConv2DPWLNorm(const int32_t& selLayer);

    // Add Scheme 3 Intra Bank Loading Objective after normalization
    GRBLinExpr addScheme3IntraBankObjConv2DPWLNorm(const int32_t& selLayer); 

    // =============================================
    // GRB Variable Functions Declaration -- FC
    // =============================================
    // Add all variables related to loop bounds in the resulting nested loop for a single layer
    // Variable Naming : L_l_X_i_j_k, which means: workload dimension i, memory level j, divisor k in Layer l
    // All variables are binary -- one hot encoding
    void addLoopBoundVarsFC(const int32_t& selLayer);

    // Add all variables related to tensor size one hot encoding for a single layer
    // Variable Naming : L_l_E_i_j_k, which means: mem level i, tensor type j, potential value k in Layer l
    // All variables are binary -- One hot encoding
    void addTensorSizeVarsFC(const int32_t& selLayer);
    
    // ====================================
    //         GRB Constraints -- FC 
    // ====================================

    // Common Constraint
    // Add loop bound constraints for all workload dimension and memory level
    // Only one value can be taken for all variables in workload dim d and mem level m
    void addLoopBoundConsFC(const int32_t& selLayer);

    // Common Constraint
    // Add Workload constraints for all loop bound variables at each memory level
    // If we multiple all loop bounds at different mem level for a specific workload dim,
    // this should be equal to the original workload dim value
    void addWorkLoadConsFC(const int32_t& selLayer, const double_t& tolerance);

    // Common Constraint
    // Add number of bank constraint
    // For now we directly use the per layer bank number limit in the input mlir file
    void addNumBankConsFC(const int32_t& selLayer);

    // Common Constraint
    // Add Tensor value onehot constraint
    // For a given mem level and tensor type, only one value can be selected
    void addTensorValueConsFC(const int32_t& selLayer);

    // Common Constraint
    // Add Cartesian Product constraints
    // We need to make sure the Potential value one hot encoding variables aligns with the Loop Bound Variables
    // that means variables X[][][] match with E[][][].
    // The original format of the constraint includes log2 values, and we want the equation to be equal, we may face some precision issues
    // Thus, we relax the equal constraint to two inequality expressions (a <= b + tolerance) and (a >= b - tolerance)
    void addCartesianProductConsFC(const int32_t& selLayer, const double_t& tolerance);

    // Layout Scheme 1 Specific Constraint
    //   Total sensor size < Size of all banks
    void addScheme1ConsFC(const int32_t& selLayer);

    // Layout Scheme 1 Specific Constraint
    //   Tensor size per bank < Bank size
    void addScheme2ConsFC(const int32_t& selLayer);

    // Layout Scheme 1 Specific Constraint
    //   Tensor size in one row < Row Size
    void addScheme3ConsFC(const int32_t& selLayer);

    // ====================================
    //         GRB Objectives -- FC
    // ====================================

    // Add computation objective function --> Num of cucles used for computation
    // This part is the same for all schemes
    GRBLinExpr addComputeObjectFC(const int32_t& selLayer);

    // Add Scheme 1 Row Act Objective
    GRBLinExpr addScheme1RowActObjectFC(const int32_t& selLayer);

    // Add Scheme 1 Cross Bank Loading Objective
    GRBLinExpr addScheme1CrossBankObjFC(const int32_t& selLayer);
    
    // Add Scheme 2 Row Act Objective
    GRBLinExpr addScheme2RowActObjectFC(const int32_t& selLayer);

    // Add Scheme 2 Cross Bank Loading Objective
    GRBLinExpr addScheme2CrossBankObjFC(const int32_t& selLayer);

    // Add Scheme 3 Row Act Objective
    GRBLinExpr addScheme3RowActObjectFC(const int32_t& selLayer);

    // Add Scheme 3 Intra Bank Loading Objective
    GRBLinExpr addScheme3IntraBankObjFC(const int32_t& selLayer); 

    // Add Scheme 3 Cross Bank Loading Objective
    GRBLinExpr addScheme3CrossBankObjFC(const int32_t& selLayer);

    // ====================================
    //    GRB Objectives PWL + Nrom -- FC
    // ====================================

    // Add computation objective function --> Num of cucles used for computation
    // This part is the same for all schemes with Norm and PWL
    GRBLinExpr addComputeObjectFCPWLNorm(const int32_t& selLayer);

    // Add Scheme 1 Row Act Objective with Norm and PWL
    GRBLinExpr addScheme1RowActObjectFCPWLNorm(const int32_t& selLayer);

    // Add Scheme 1 Cross Bank Loading Objective with Norm and PWL
    GRBLinExpr addScheme1CrossBankObjFCPWLNorm(const int32_t& selLayer);
    
    // Add Scheme 2 Row Act Objective with Norm and PWL
    GRBLinExpr addScheme2RowActObjectFCPWLNorm(const int32_t& selLayer);

    // Add Scheme 2 Cross Bank Loading Objective with Norm and PWL
    GRBLinExpr addScheme2CrossBankObjFCPWLNorm(const int32_t& selLayer);

    // Add Scheme 3 Row Act Objective with Norm and PWL 
    GRBLinExpr addScheme3RowActObjectFCPWLNorm(const int32_t& selLayer);

    // Add Scheme 3 Intra Bank Loading Objective with Norm and PWL
    GRBLinExpr addScheme3IntraBankObjFCPWLNorm(const int32_t& selLayer); 

    // Add Scheme 3 Cross Bank Loading Objective with Norm and PWL
    GRBLinExpr addScheme3CrossBankObjFCPWLNorm(const int32_t& selLayer);
    
    // ================================================
    //    GRB Constraints -- Combined Memory Scheme
    // ================================================
    // Add combined memory layout scheme constraints for No-Duplication(Scheme 1)
    void addComMemScheme1Cons(std::vector<int32_t> &layerGroup, const int64_t& numBanks);

    // Add combined memory layout scheme constraints for Bank-Duplication(Scheme 2)
    void addComMemScheme2Cons(std::vector<int32_t> &layerGroup, const int64_t& numBanks);

    // Add combined memory layout scheme constraints for Row-Duplication(Scheme 3)
    void addComMemScheme3Cons(std::vector<int32_t> &layerGroup, const int64_t& numBanks);
    
    // ====================================
    //     GRB Objectives -- Cross Layer
    // ====================================

    // Add Scheme 1 Cross Layer Tensor Dependency Objective -- Conv2D -> Conv2D
    double_t addScheme1CrossLayerObjCC(const int32_t& curLayer, const int32_t& nextLayer);

    // Add Scheme 2 Cross Layer Tensor Dependency Objective
    GRBLinExpr addScheme2CrossLayerObjCC(const int32_t& curLayer, const int32_t& nextLayer);

    // Add Scheme 2 Cross Layer Tensor Dependency Objective
    GRBLinExpr addScheme3CrossLayerObjCC(const int32_t& curLayer, const int32_t& nextLayer);

    // Output Objective
    double_t addScheme1Conv2DOut(const int32_t& curLayer);
    double_t addScheme1FCOut(const int32_t& curLayer);

    GRBLinExpr addScheme2Conv2DOut(const int32_t& curLayer);
    GRBLinExpr addScheme2FCOut(const int32_t& curLayer);

    GRBLinExpr addScheme3Conv2DOut(const int32_t& curLayer);
    GRBLinExpr addScheme3FCOut(const int32_t& curLayer);

    // Inout Objective
    double_t addScheme1Conv2DIn(const int32_t& nextLayer);
    double_t addScheme1FCIn(const int32_t& nextLayer);

    GRBLinExpr addScheme2Conv2DIn(const int32_t& nextLayer);
    GRBLinExpr addScheme2FCIn(const int32_t& nextLayer);

    GRBLinExpr addScheme3Conv2DIn(const int32_t& nextLayer);
    GRBLinExpr addScheme3FCIn(const int32_t& nextLayer);

    // ==============================================
    //   GRB Objectives -- Cross Layer (PWL + Norm)
    // ==============================================
    // Output Objective
    double_t addScheme1Conv2DOutPWLNorm(const int32_t& curLayer);
    double_t addScheme1FCOutPWLNorm(const int32_t& curLayer);

    GRBLinExpr addScheme2Conv2DOutPWLNorm(const int32_t& curLayer);
    GRBLinExpr addScheme2FCOutPWLNorm(const int32_t& curLayer);

    GRBLinExpr addScheme3Conv2DOutPWLNorm(const int32_t& curLayer);
    GRBLinExpr addScheme3FCOutPWLNorm(const int32_t& curLayer);

    // Inout Objective
    double_t addScheme1Conv2DInPWLNorm(const int32_t& nextLayer);
    double_t addScheme1FCInPWLNorm(const int32_t& nextLayer);

    GRBLinExpr addScheme2Conv2DInPWLNorm(const int32_t& nextLayer);
    GRBLinExpr addScheme2FCInPWLNorm(const int32_t& nextLayer);

    GRBLinExpr addScheme3Conv2DInPWLNorm(const int32_t& nextLayer);
    GRBLinExpr addScheme3FCInPWLNorm(const int32_t& nextLayer);
    
    // ====================================
    // Functions for Constants Construction
    // ====================================

    // Function returns a vector of all divisors of a given integer
    // For example, if dimension = 64, the vector will be [1, 2, 4, 8, 16, 32, 64]
    std::vector<int64_t> getAllDivisors(const int32_t& dimension);

    // Function to Extract all constants for a given CONV 2D layer
    void buildConv2DLayerConstants(const LayerInfo& selLayerIns);

    // Function to Extract all constants for a given FC layer
    void buildFCLayerConstants(const LayerInfo& selLayerIns);
    
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

    // Function used to calculate the corresponding potential tensor values from Cartesian Product Set
    std::vector<int64_t> calPTensorValue(const CartesianProduct<int64_t>& selCaPr);

    // Function used to calculate the potentail tensor value of Conv2D input array from Cartesian Product Set
    std::vector<int64_t> calConv2DPIAValue(const CartesianProduct<int64_t>& selCaPr, 
                                           const std::vector<int64_t>& layerStride, const std::vector<int64_t>& layerDialation);
    
    void buildScalingFactor(std::vector<int32_t> &layerGroup, std::map<int32_t, LayerInfo> &layersInfoDB);
    
    // ====================================
    //            Helper Functions
    // ====================================

    // Print all tuples in a given cartesian product set
    template<typename T>
    void printCartesianProduct(const CartesianProduct<T>& selSet);

    // Write out the MILP model
    // Call this function before model.optimize()
    void writeMLIPModel(GRBModel& model, const std::string& logPath, const int32_t& layerGroupID);

    // Write out the MILP solution
    // Call this function after model.optimize()
    void writeMLIPSol(GRBModel& model, const std::string& logPath, const int32_t& layerGroupID);

    // Print out all results, both loop bounds and cartesian product variables
    // This function directly prints out the result of the one-hot encoding
    void printConv2DMILPResult(const int32_t& selLayer);

    void printFCMILPResult(const int32_t& selLayer);
    
    // Read the content of the json file into a string, run this before parsing the json file
    std::string readJsonFile(const std::string& filePath);

    // Parse the JSON file string
    llvm::Expected<llvm::json::Value> parseJson(const std::string& jsonString);

    // Parse the knobs.json file and construct the Knobs struct  -- pass via cmdline
    void readKnobValues(const std::string& filePath);

    // Parse the architecture.json file -- pass via cmdline
    void readArchInfo(const std::string& filePath);

    // Dump the results to a file for a single layer
    std::string traceConv2D(const int32_t& selLayer, const LayerInfo& oriLayerInfo);

    // Dump the results of GEMM to a file
    std::string traceFC(const int32_t& selLayer, const LayerInfo& oriLayerInfo);

    // ==================
    // Variable Defintion
    // ==================
    std::map<int32_t, ConstInfo> layerConstInfo; // Map from layer index to instances that stores all constants info of the layer
    std::map<int32_t, LayerGRBVariables> layerVariables; // Map from layer index to instances holding layer specific variables

    // Define the GRBModel
    GRBModel model;

    // Store all Arch info
    ArchInfo archInfo;

    // Store weights for different parts in the final objective function
    Knobs PerformanceKnobs;

    // ==================
    //  Shared Constants
    // ==================
    // TODO: Need to parse this infomration from the mlir file
    //! Need to expose the interface for the selected scheme
    int32_t selLayoutScheme = LayoutScheme::Scheme_2;

    // Memory layout strategy
    int32_t memoryAlloc = MemoryAllocationStrategy::Exclusive;

    // Objective definition scheme
    int32_t ObjApproScheme = ObjApproximationScheme::Manual_log;

    // Settings for the gurboi piecewise linearization function
    std::string GurobiApproximationOptions = "FuncPieces=-2 FuncPieceError=0.002";

    // Path to json files: memory architecture and tuning knobs
    std::string archPath;
    std::string knobPath;

    //
    // Following constants are used for normalization
    //
    double_t ScaleBase = 10000;
    double_t maxMacNum;
    double_t maxMacScaleFactor;
    double_t maxMacScaleFactorLog;
    std::map<int32_t, int32_t> perLayerMacValue;
    std::map<int32_t, double_t> perLayerMacScaleFactor;
    std::map<int32_t, double_t> perLayerMacScaleFactorLog; 
};

} // namespace pim

#endif // PIMOPT_GUROBI_NOT_INSTALLED

#endif // PIMOPT_DETAILLAYOUT_MILP