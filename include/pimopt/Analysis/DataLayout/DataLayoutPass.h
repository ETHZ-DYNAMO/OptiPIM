#ifndef PIMOPT_DATALAYOUT_PASS
#define PIMOPT_DATALAYOUT_PASS

#include "pimopt/Analysis/DataLayout/DataLayoutMILP.h"

#include <iostream>
#include <vector>

// Check the existence of Gurobi library
#ifndef PIMOPT_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace pim {

// Define the DataLayoutPass, which will do the following:
//      1. Parse the input mlir file
//          a. Get the layer type and layer dimension of each ML operator
//              (Supported: Conv2D, FC, )
//          b. Get the layer group information (which layers are put together) [Tentative]
//          c. Build the mapping structure from the parsed information to the 
//             internal data structures
struct DataLayoutPass: public mlir::PassWrapper<DataLayoutPass, mlir::OperationPass<>> {

    //
    // Functions used to accept all arguments from the command line
    //
    void setOutputPath(const std::string& path) {
        outputPath = path;
    }

    void setDeviceType(const int32_t& selDevice) {
        deviceType = selDevice;
    }

    void setMemAlloc(const int32_t& selMemAlloc) {
        memoryAllocScheme = selMemAlloc;
    }

    void setObjMethod(const int32_t& selLinMethod) {
        objMethod = selLinMethod;
    } 

    void setTransCoeffMethod(const int32_t& selMethod) {
        transCoeffMethod = selMethod;
    }

    void setNumCountingMethod(const int32_t& selMethod) {
        numCountingMethod = selMethod;
    }

    void setStorageMethod(const int32_t& selMethod) {
        storageMethod = selMethod;
    }

    void setMilpPath(const std::string& path) {
        milpModelPath = path;
    }

    void setKnobPath(const std::string& path) {
        knobPath = path;
    }

    void setArchPath(const std::string& path) {
        archPath = path;
    } 

    //
    // Functions used to parse different ML operators
    //
    // Parsing function for Conv2D
    void parseConv2D(mlir::linalg::Conv2DNchwFchwOp op);

    // Parsing function for FC
    void parseFC(mlir::linalg::MatmulOp op);

    // Parsing function for BatchFC
    void parseBatchFC(mlir::linalg::BatchMatmulOp op);

    //
    // Top level function working on different ops in the MLIR file
    //
    void runOnOperation() override;

    llvm::StringRef getArgument() const final {
        return "Input file in linalg.mlir";
    }

    llvm::StringRef getDescription() const final {
        return "PIMopt Detail Data Layout pass";
    }

    //
    // Helper function
    //

    // Print layergroup related information
    void printOpGroupInfo();

    // Insert the key-value pair to the opGroup map
    void insertToOpGroup(std::map<int32_t, std::vector<int32_t>>& opGroupMap, int32_t key, int32_t value) {
        auto it = opGroupMap.find(key);

        if (it != opGroupMap.end()) {
            it->second.push_back(value);
        } else {
            std::vector<int32_t> newOpGroup;
            newOpGroup.push_back(value);
            opGroupMap[key] = newOpGroup;
        }
    }

    // ======================
    //   Internal Variables
    // ======================
    int32_t curOpGroupId = 0;

    // Variable used to store the intermediate resutls during operator group construction
    std::vector<int32_t> tmpOpGroup;

    // Map from global op index to the corresponding storing structure of detailed op info
    std::map<int32_t, OperatorInfo> opsInfo;

    // Map from op group id to the corresponding op group vector
    // E.x. { 0: [0,1,2]; ... }, which means op 0, 1 and 2 are in op group 0
    std::map<int32_t, std::vector<int32_t>> opGroupsInfo;

    // Return a vector containing all related op info in the selected opGroup
    std::vector<OperatorInfo> getOpGroupInfo(std::vector<int32_t> selOpGroup);

    // ==================
    //  Input Constants
    // ==================
    std::string outputPath;     // Path to store the generated optimized nested loop
    std::string milpModelPath;  // Path to store the milp model for debuging
    std::string knobPath;       // Path to the json file storing all the weights(knobs)
    std::string archPath;       // Path to the json file storing all the arch details

    // Transfomration coefficient modeling method
    // 0. Comb 0; 1. Comb 1; 2. Comb 2; 3. Comb 5 
    // 4. Flexible, automatically select
    int32_t transCoeffMethod;
    
    // Targeted device type:
    //  0. In-Memory-Compute devices (i.e. MIMDRAM)
    //  1. Near-Memory-Compute devices (i.e. HBM-PIM)
    int32_t deviceType;

    // Memory allocation scheme:
    //  0. Exclusive
    //  1. Combined
    int32_t memoryAllocScheme;

    // Objective selection method:
    //  0. What we proposed ; 1. COSA's method with out layout cost evaluation;
    int32_t objMethod;

    // Number Estimation Method:
    // 0. Our method; 1. COSA's method
    int32_t numCountingMethod;

    // Storage Method
    // 0. Output Not stored
    // 1. Both Input and Output not stored
    // 2. All stored
    int32_t storageMethod;

};

} // namespace pim

#endif // PIMOPT_GUROBI_NOT_INSTALLED

#endif // PIMOPT_DATALAYOUT_PASS
