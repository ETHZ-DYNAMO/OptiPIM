#ifndef PIMOPT_DETAILLAYOUT_PASS
#define PIMOPT_DETAILLAYOUT_PASS

#include "pimopt/Analysis/DetailLayout/DetailLayoutMILP.h"

#include <iostream>
#include <vector>

// Check the existence of Gurobi library
#ifndef PIMOPT_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace pim {

// Define the DetailLayoutPass, which will do the following:
//      1. Parse the input linalg.mlir file
//          a. Get the layer dimension information for each ML operators
//             (Supported: conv2d, )
//          b. Get the layer group information (which layers are put together)
//          c. Build structures (i.e. maps) to keep the parsed information to create
//             the desired MILP model accordingly.
struct DetailLayoutPass : public mlir::PassWrapper<DetailLayoutPass, mlir::OperationPass<>> {
    // Constructor
    // DetailLayoutPass() : outputPath(outputPath), layoutScheme(layoutScheme), memoryAllocScheme(memoryAllocScheme) {};

    // Set command line input parameters
    void setOutputPath(const std::string& path) {
        outputPath = path;
    }

    void setLayoutScheme(const int32_t& selLayout) {
        layoutScheme = selLayout;
    } 

    void setMemAlloc(const int32_t& selMemAlloc) {
        memoryAllocScheme = selMemAlloc;
    }

    void setObjAppro(const int32_t& selObjAppro) {
        objApproMethod = selObjAppro;
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
    
    // Parse the input file
    void parseConv2D(mlir::linalg::Conv2DNchwFchwOp op);

    void parseFC(mlir::linalg::MatmulOp matMulOp);

    void parseBatchFC(mlir::linalg::BatchMatmulOp batchMatMulOp);

    void runOnOperation() override;

    void printLayerGroupInfo();

    llvm::StringRef getArgument() const final {
        return "Input file in linalg.mlir";
    }

    llvm::StringRef getDescription() const final {
        return "PIMopt Detail Data Layout pass";
    }

    // Helper function to insert the key-value pair to the layerGroup Map
    void insertToLayerGroup(std::map<int32_t, std::vector<int32_t>>& dataMap, int32_t key, int32_t value) {
        auto it = dataMap.find(key);

        if (it != dataMap.end()) {
            it->second.push_back(value);
        } else {
            std::vector<int32_t> newGroup;
            newGroup.push_back(value);
            dataMap[key] = newGroup;
        }
    }

    // ==================
    // Variable Defintion
    // ==================
    int32_t cur_layer_group_id = 0;

    std::vector<int32_t> tmp_layer_group;

    // Map from global layer index to the corresponding storing structure of detailed layer indo
    std::map<int32_t, LayerInfo> layersInfo;

    // Vector storing layer grouping information
    // E.x., [[0,1,2]], which means layer 0, 1, and 2 are in layer group 0
    // std::vector<std::vector<int32_t>> layerGroupsInfo;

    // Map from Layer group id to the corresponding layer group vector
    // E.x. { 0: [0,1,2]; ... }, which means layer 0, 1 and 2 are in layer group 0
    std::map<int32_t, std::vector<int32_t>> layerGroupsInfo;

    // Return a vector containing all related layer info in the selected layergroup
    std::vector<LayerInfo> getLayerGroupInfo(std::vector<int32_t> sel_layer_group);

    // ==================
    //  Input Variables
    // ==================
    std::string outputPath;    // Path to store the generated trace file
    std::string milpModelPath; // Path to store the milp model for debuging
    std::string knobPath;      // Path to tuning knobs json
    std::string archPath;     // Path to architecture json
    int32_t layoutScheme;      // Layout scheme for the MILP optimization process
    int32_t memoryAllocScheme; // Memory allocation scheme: Exclusive or Combined
    int32_t objApproMethod;     // Objective function approximation method: 0. Log; 1. Log + Norm; 2. PWL + Norm
                                                
};

}   // namespace pim

#endif // PIMOPT_GUROBI_NOT_INSTALLED

#endif // PIMOPT_DETAILLAYOUT_PASS