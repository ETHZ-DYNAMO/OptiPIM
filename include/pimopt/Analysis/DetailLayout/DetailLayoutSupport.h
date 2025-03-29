#ifndef PIMOPT_DETAILLAYOUT_SUPPORT
#define PIMOPT_DETAILLAYOUT_SUPPORT

#include "mlir/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/JSON.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#define NUM_DIM_CONV2D 7
#define NUM_DIM_FC 5
#define NUM_MEM_LEVEL  3

namespace pim {

// Define structures for CartesianProduct
template<typename T>
using Set = std::vector<T>;

template<typename T>
using CartesianProduct = std::vector<std::vector<T>>;

enum layerTypeAll {
    CONV2D = 0,
    FC = 1
};

// Define an enum to represent the indices of workload dimensions
enum workLoadDimensionConv2D {
    N = 0,  // Batch Size
    K = 1,  // Output Channel
    P = 2,  // Output Row
    Q = 3,  // Output Col
    C = 4,  // Input Channel
    R = 5,  // Filter Row
    S = 6   // Filter Col
};

enum workLoadDimensionFC {
    FC_N = 0, // Batch Size
    FC_K = 1, // Output Channel
    FC_P = 2, // Row of A
    FC_Q = 3, // Col of A
    FC_R = 4  // Col of B
};

enum memLevel {
    RowBuffer = 0,  // Row Buffer
    Row = 1,        // Row
    Bank = 2        // Bank
};

enum computeTensorType {
    Filter = 0,
    OA = 1,
    IA = 2
};

enum computeTensorTypeFC {
    Mat_A = 0,   // Matrix A
    Mat_B = 1,   // Matrix B
    FC_Out = 2   // Output
};

enum LayoutScheme {
    Scheme_1 = 0,  // No-duplication
    Scheme_2 = 1,  // Bank-duplication
    Scheme_3 = 2   // Row-duplication
};

enum MemoryAllocationStrategy {
    Exclusive = 0,
    Combined = 1
};

enum ObjApproximationScheme {
    Manual_log = 0,
    Manual_norm = 1,
    Gurobi_exp = 2
};

const std::string WorkLoadDimSimNames[] = {
    "N", "K", "P", "Q", "C", "R", "S"
};

const std::string WorkLoadDimSimNamesFC[] = {
    "FC_N", "FC_K", "FC_P", "FC_Q", "FC_R"
};

const std::string WorkLoadDimComNames[] = {
    "Batch Size", 
    "Output Channel", 
    "Output Row",
    "Output Col",
    "Input Channel",
    "Filter Row",
    "Filter Col"
};

const std::string WorkLoadDimComNamesFC[] = {
    "Batch Size", 
    "Output Channel", 
    "Matrix A Row Size",
    "Matrix A Col Size",
    "Matrix B Col Size"
};

const std::string TensorTypeNames[] = {
    "Filter",
    "Output_Array",
    "Input_Array"
};

const std::string TensorTypeNamesFC[] = {
    "Matrix_A",
    "Matrix_B",
    "Output_Array"
};

const std::string LayerTypeNames[] = {
    "2D Convolution Layer",
    "Fully Connected Layer"
};

const std::string MemLevelNames[] = {
    "Row Buffer",
    "Row",
    "Bank"
};

// Struct that stores all dimension information for a layer in the ML model
struct LayerInfo {
    // Layer ID
    int32_t layerId;

    // Layer Group information
    int32_t layerGroupId;

    // Number of banks used by this layer
    int64_t numBanks;

    // Type of the Layer
    int32_t layerType;

    // DataLayout Scheme
    int32_t dataLayoutScheme;

    // Stride
    std::vector<int64_t> stride;

    // Dilation
    std::vector<int64_t> dilation;

    // Workload Dimensions - 7 elements initialized to 1
    std::vector<int64_t> workLoadDimVec{1,1,1,1,1,1,1};

    // Input Channel Depth -- C
    int64_t inChanDepth = 1;

    // Output Channel Depth -- K
    int64_t outChanDepth = 1;

    // Output Row Size -- P
    int64_t outRowSize = 1; 

    // Output Col Size -- Q
    int64_t outColSize = 1;

    // Filter Row Size -- R
    int64_t filterRowSize = 1;

    // Filter Column Size -- S
    int64_t filterColSize = 1;

    // Batch Size -- N
    int64_t batchSize = 1;

    // Helper function definition
    void print_detail() {
        if (layerType == layerTypeAll::CONV2D) {
            llvm::outs() << "Conv2D Layer: (Layer ID: " << layerId << ", Layer Group ID: " << layerGroupId << ")\n";
            llvm::outs() << "\tStride : [ " << stride[0] << ", " << stride[1] << " ];\n";
            llvm::outs() << "\tDilation : [ " << dilation[0] << ", " << dilation[1] << " ];\n";
            llvm::outs() << "\tInput Channel Depth (C) : " << inChanDepth << ";\n";
            llvm::outs() << "\tOutput Channel Depth (K) : " << outChanDepth << ";\n";
            llvm::outs() << "\tOutput Row Size (P) : " << outRowSize << ";\n";
            llvm::outs() << "\tOutput Col Size (Q) : " << outColSize << ";\n";
            llvm::outs() << "\tFilter Row Size (R) : " << filterRowSize << ";\n";
            llvm::outs() << "\tFilter Col Size (S) : " << filterRowSize << ";\n";
            llvm::outs() << "\tBatch Size (N) : " << batchSize << ";\n";
            llvm::outs() << "\tBank Number (N_bank) : " << numBanks << ";\n";
        } else if (layerType == layerTypeAll::FC) {
            llvm::outs() << "FC Layer: (Layer ID: " << layerId << ", Layer Group ID: " << layerGroupId << ")\n";
            llvm::outs() << "\tBatch Size (N) : " << batchSize << ";\n";
            llvm::outs() << "\tOutput Channel Depth (K) : " << outChanDepth << ";\n";
            llvm::outs() << "\tMatrix A Row Size (P) : " << filterRowSize << ";\n";
            llvm::outs() << "\tMatrix A Col Size (Q) : " << filterColSize << ";\n";
            llvm::outs() << "\tMatrix B Col Size (R) : " << outColSize << ";\n";
        }
    }
};

// =================================
//          Constant Values
// =================================

// Constant Relation Matrix for 2D Convolution
// Row indexing: {N:0, K:1, P:2 ,Q:3, C:4, R:5, S:6}
// Col indexing: {Filter:0, Output_Array:1, Input_Array:2}
const int32_t Conv2DRelationMatrix[7][3] = {
    {0, 1, 1},
    {1, 1, 0},
    {0, 1, 1},
    {0, 1, 1},
    {1, 0, 1},
    {1, 0, 1},
    {1, 0, 1}
};

// Mapping matrix between tensor type to the workload dimension position in the cartesian product set
// Row indexing: {N:0, K:1, P:2 ,Q:3, C:4, R:5, S:6}
// Col indexing: {Filter:0, Output_Array:1, Input_Array:2}
const int32_t workLoadToTensorCaPr[7][3] = {
    {-1,  0,  0},
    { 0,  1, -1},
    {-1,  2,  1},
    {-1,  3,  2},
    { 1, -1,  3},
    { 2, -1,  4},
    { 3, -1,  5}
};

// Constant Relation Matrix for Fully Connected Layer
// Row indexing: {N:0, K:1, P:2 ,Q:3, R:4}
// Col indexing: {Matrix_A:0, Matrix_B:1, Out:2}
const int32_t FCRelationMatrix[5][3] = {
    {1, 1, 1},
    {1, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
    {0, 1, 1}
};

// Mapping matrix between tensor type to the workload dimension position in the cartesian product set
// Row indexing: {N:0, K:1, P:2 ,Q:3, R:4}
// Col indexing: {Matrix_A:0, Matrix_B:1, Out:2}
const int32_t FCworkLoadToTensorCaPr[5][3] = {
    { 0,  0,  0},
    { 1,  1,  1},
    { 2, -1,  2},
    { 3,  2, -1},
    {-1,  3,  3}
};

// Architecture Performance Modeling constants
struct ArchInfo {
    // The size of a single data during calculation, in number of bytes
    int32_t dataSize = 4;

    // The size of a single bank, in number of bytes
    int32_t bankSize = 1024 * 512 * 16; // 2^10 * 2^8 * 2^4
    // int32_t bankSize = 1024 * 128; // 2^10 * 2^8

    // The size of a single row, in bytes
    // int32_t rowSize = 1024;
    int32_t rowSize = 512 * 16;

    // Row buffer level processing data size, number bytes
    int32_t dataQueueSizeHBM  = 16; // HBM2 Size
    int32_t dataQueueSizeHBME = 32; // HBM2E Size

    // Time of different operations -- should be in number of clock cycles
    // TODO: Change the following to the number of clock cycles
    double_t rowActTime = 15;     // Row Activation time in ns
    double_t crossBankTime = 150; // Cross Bank Transfer time in ns

    // Bandwidth Info
    double_t HBMBandwidth = 64; // HBM Bandwidth, in bytes

    std::string toString() const {
        std::ostringstream oss;
        oss << "    ArchInfo:\n"
            << "        dataSize: " << dataSize << "\n"
            << "        bankSize: " << bankSize << "\n"
            << "        rowSize: " << rowSize << "\n"
            << "        dataQueueSizeHBM: " << dataQueueSizeHBM << "\n"
            << "        dataQueueSizeHBME: " << dataQueueSizeHBME << "\n"
            << "        rowActTime: " << rowActTime << "\n"
            << "        crossBankTime: " << crossBankTime << "\n"
            << "        HBMBandwidth: " << HBMBandwidth;
        return oss.str();
    }

};

struct Knobs {
    // Tolerance for Workload Constraint
    double_t workLoadTolerance = 0.02;

    // Tolerance for Cartesian Product Constraint
    double_t cartesianProTolerance = 0.05;

    // Weight of computation in the final objective function
    double_t compWeight = 1.0;

    // Weight of Row Activation for Loading latency
    double_t rowActWeight = 4.0;

    // Weight of Cross Bank Loading
    double_t crossBankWeight = 15.0;

    // Weight of intra-bank loading - for scheme 3
    double_t intraBankWeight = 15.0;

    //
    // Cross Layer Objective
    //
    double_t crossLayerWeight = 2.0;

    std::string toString() const {
        std::ostringstream oss;
        oss << "    Knobs:\n"
            << "        workLoadTolerance: " << workLoadTolerance << "\n"
            << "        cartesianProTolerance: " << cartesianProTolerance << "\n"
            << "        compWeight: " << compWeight << "\n"
            << "        rowActWeight: " << rowActWeight << "\n"
            << "        crossBankWeight: " << crossBankWeight << "\n"
            << "        intraBankWeight: " << intraBankWeight << "\n"
            << "        crossLayerWeight: " << crossLayerWeight;
        return oss.str();
    }
};

} // namespace pim

#endif // PIMOPT_DETAILLAYOUT_SUPPORT