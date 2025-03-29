#ifndef PIMOPT_DATALAYOUT_SUPPORT
#define PIMOPT_DATALAYOUT_SUPPORT

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

#define NUM_BOUND_CONV2D 7
#define NUM_BOUND_FC 5
#define NUM_LOOP_LEVEL 3

namespace pim {

// Define structures for CartesianProduct
template<typename T>
using Set = std::vector<T>;

template<typename T>
using CartesianProduct = std::vector<std::vector<T>>;

// =================================
//          Constant Values
// =================================

// Define all possible operator types in our workloads
enum operatorTypesIdx {
    CONV2D = 0,
    FC = 1
};

// Define the indices of all loop bounds in conv2D workload
// Note that the loop bound index is the same as the loop variable index
enum loopBoundIdxConv2D {
    N = 0,  // Batch Size
    K = 1,  // Output Channel
    P = 2,  // Output Width
    Q = 3,  // Output Height
    C = 4,  // Input Channel
    R = 5,  // Filter Width
    S = 6   // Filter Height
};

// Define the indices of all loop bounds in FC workload
// FC is calculated as Out[N][P][R] += In[N][P][Q] * Weight[R][Q]
enum loopBoundIdxFC {
    FC_N = 0,   // Batch Size
    FC_K = 1,   // Output Channel
    FC_P = 2,   // Matrix A rows
    FC_Q = 3,   // Matrix A columns
    FC_R = 4    // Matrix B columns 
};

// Define the indices of all loop levels, inner most is indexed as 0
enum loopLevelIdx {
    LEVEL0 = 0, // Computation Scheduling
    LEVEL1 = 1, // Partition Allocation
    LEVEL2 = 2  // PE Allocation
};

// Define the indices of all tensor types in Conv2D
enum tensorTypeIdxConv2D {
    FA = 0,     // Input Array
    OA = 1,     // Output Array
    IA = 2      // Filter Array
};

// Define the indices of all tensor types in FC
enum tensorTypeIdxFC {
    MatA = 0,   // Matrix A
    MatB = 1,   // Matrix B
    FCOA = 2      // Output Array
};

// Define the index of device types
enum deviceTypeIdx {
    PUM = 0,    // Process-using-memory
    PNM = 1     // Process-near-memory
};

// Define the index of memory allocation strategy
enum memAllocIdx {
    Exclusive = 0,  // Different operators will use different memory resources
    Combined = 1    // Different operators can share the same memory resoruce
};

// Define the indices for the linearization methods
enum linMethodIdx {
    Log = 0,
    LogNorm = 1,
    GurobiExp = 2
};

// Define the indices for the transformation coefficient calculation method
enum transCoeffMethodIdx {
    Comb0 = 0,
    Comb1 = 1,
    Comb2 = 2,
    Comb5 = 3,
    Flexible = 4
};

// Define the indices for the loop variables that need the transformation coefficients
enum transCoeffConv2DIdx {
    Conv2D_P = 0,
    Conv2D_R = 1,
    Conv2D_Q = 2,
    Conv2D_S = 3
};

// Define the map between trans index to the actual loop variable index
const int32_t transToLoopBoundMap[4] = {
    loopBoundIdxConv2D::P,
    loopBoundIdxConv2D::R,
    loopBoundIdxConv2D::Q,
    loopBoundIdxConv2D::S
};

// Define the indices for the input indexing look up table
// PRTable : [P * Wstride + R * Wdilation]
// QSTable : [Q * Hstride + S * Hdilation]
enum inputLutIdxConv2D {
    PRTable = 0,
    QSTable = 1
};

// Index to string converter for related loop variables (Conv2D)
const std::string transCoeffConv2DName[] = {
    "Conv2D_P", "Conv2D_R",  "Conv2D_Q", "Conv2D_S"
};

// Index to string converter for look up table Name (abbreviation) (Conv2D)
const std::string inputArrayLUTNameConv2D[] = {
    "P_R_LUT", "Q_S_LUT"
};

// Index to string converter for loop bounds (abbreviations) (Conv2D)
const std::string loopBoundNameConv2D[] = {
    "N", "K", "P", "Q", "C", "R", "S"
};

// Index to string converter for loop bounds (abbreviations) (FC)
const std::string loopBoundNameFC[] = {
    "FC_N", "FC_K", "FC_P", "FC_Q", "FC_R"
};

// Index to string converter for loop bounds (verbose) (Conv2D)
const std::string loopBoundNameVerboseConv2D[] = {
    "Batch Size", 
    "Output Channel", 
    "Output Width",
    "Output Height",
    "Input Channel",
    "Filter Width",
    "Filter Height"
};

// Index to string converter for loop bounds (verbose) (FC)
const std::string loopBoundNameVerboseFC[] = {
    "Batch Size", 
    "Output Channel", 
    "Matrix A Row",
    "Matrix A Col",
    "Matrix B Col"
};

// Index to string converter for loop bounds (verbose) (FC)
const std::string memoryLevelsName[] = {
    "Level 0 (Inner Most)", 
    "Level 1", 
    "Level 2 (Outter Most)"
};

// Index to string converter for tensor type (Conv2D)
const std::string tensorTypeNameConv2D[] = {
    "Input Array",
    "Output Array",
    "Filter Array"
};

// Index to string converter for tensor type (FC)
const std::string tensorTypeNameFC[] = {
    "Matrix A",
    "Matrix B",
    "Output Array"
};

// Index to Operator Type Name
const std::string operatorTypeName[] = {
    "Convolution 2D Operator",
    "Fully Connected Operator"
};

// Index to Device Type Name
const std::string deviceTypeName[] = {
    "Process-using-memory device",
    "Process-near-memory device"
};

// =================================
//          Operator Info
// =================================

// Struct used to store all information about the original nested loop (input operand)
struct OperatorInfo {
    // Operator ID
    int32_t opID = 0;

    // Operator Group Info
    int32_t opGroupID = 0;

    // Number of PEs assigned to this layer
    int64_t numPE = 1;

    // Type index of the operator
    int32_t opType = 0; 

    // Target device index
    int32_t deviceType = 0;

    // Stride information for the operator
    std::vector<int64_t> stride;

    // Dilation information for the operator
    std::vector<int64_t> dilation;

    // Loop Bound Values - 7 elements initialized to 1
    std::vector<int64_t> loopBoundVec{1,1,1,1,1,1,1};

    // Helper functions
    void print_detail() {
        if (opType == operatorTypesIdx::CONV2D) {
            llvm::outs() << "Conv2D Op: (Op ID: " << opID << ", Op Group ID: " << opGroupID << ")\n";
            llvm::outs() << "\tStride : [ " << stride[0] << ", " << stride[1] << " ];\n";
            llvm::outs() << "\tDilation : [ " << dilation[0] << ", " << dilation[1] << " ];\n";
            llvm::outs() << "\tInput Channel Depth (C) : " << loopBoundVec[loopBoundIdxConv2D::C] << ";\n";
            llvm::outs() << "\tOutput Channel Depth (K) : " << loopBoundVec[loopBoundIdxConv2D::K] << ";\n";
            llvm::outs() << "\tOutput Width (P) : " << loopBoundVec[loopBoundIdxConv2D::P] << ";\n";
            llvm::outs() << "\tOutput Height (Q) : " << loopBoundVec[loopBoundIdxConv2D::Q] << ";\n";
            llvm::outs() << "\tFilter Width (R) : " << loopBoundVec[loopBoundIdxConv2D::R] << ";\n";
            llvm::outs() << "\tFilter Height (S) : " << loopBoundVec[loopBoundIdxConv2D::S] << ";\n";
            llvm::outs() << "\tBatch Size (N) : " << loopBoundVec[loopBoundIdxConv2D::N] << ";\n";
            llvm::outs() << "\tAllocated Number of PEs : " << numPE << ";\n";
        } else if (opType == operatorTypesIdx::FC) {
            llvm::outs() << "FC Layer: (Layer ID: " << opID << ", Layer Group ID: " << opGroupID << ")\n";
            llvm::outs() << "\tBatch Size (N) : " << loopBoundVec[loopBoundIdxFC::FC_N] << ";\n";
            llvm::outs() << "\tOutput Channel Depth (K) : " << loopBoundVec[loopBoundIdxFC::FC_K] << ";\n";
            llvm::outs() << "\tMatrix A Row Size (P) : " << loopBoundVec[loopBoundIdxFC::FC_P] << ";\n";
            llvm::outs() << "\tMatrix A Col Size (Q) : " << loopBoundVec[loopBoundIdxFC::FC_Q] << ";\n";
            llvm::outs() << "\tMatrix B Col Size (R) : " << loopBoundVec[loopBoundIdxFC::FC_R] << ";\n";
        }
    }
};

// =================================
//         Relation Matrics
// =================================



// =================================
//         Arch & Weights
// =================================

// Struct storing architecture-specific information
struct ArchInfo {
    // The bitwidth of a single data element, in number of bits
    int32_t dataWidth = 32;

    // Number of rows in a PE (Column Capacity, number of bits)
    int32_t numRow = 1024;

    // Number of columns in a PE (Row Capacity)
    int32_t numCol = 512;

    // Parallelization levels, in number of bits/s
    int64_t PEBandWidth = 1;     // Inter-Column parallel tranmission block size
    int64_t SysBandWidth = 4;    // Inter-PE parallel tranmission block size

    // Operation latencies
    int64_t mulLat = 10;   // Multiplication latency in bit-serial processing
    int64_t addLat = 4;    // Addition latency in bit-serial processing
    double_t rowAct = 5.0;    // Row Activation Latency for HBM-PIM
    double_t interColTransLat = 5; // Inter-col transmission latency in a PE
    double_t interPETransLat = 15; // Inter-PE transmission latency in the whole system

    // Helper function
    std::string toString() const {
        std::ostringstream oss;
        oss << "    ArchInfo:\n"
            << "        dataWidth: " << dataWidth << " bits/element\n"
            << "        numRow: " << numRow << "\n"
            << "        numCol: " << numCol << "\n"
            << "        PEBandWidth: " << PEBandWidth << " bits/cycle\n"
            << "        SysBandWidth: " << SysBandWidth << " bits/cycle\n"
            << "        mulLat: " << mulLat << " cycles\n"
            << "        addLat: " << addLat << " cycles\n"
            << "        rowAct: " << rowAct << " cycles\n"
            << "        interColTransLat: " << interColTransLat << " s\n"
            << "        interPETransLat: " << interColTransLat;
        return oss.str();
    };
};

// Struct storing all the user inputs for the MILP
struct Knobs {
    // Accuracy tolerance during the MILP calculation
    double_t accTolerance = 0.02;

    // Weight for column multiplication 
    double_t colMulWeight = 1.0;

    // Weight for column addition 
    double_t colAddWeight = 1.0;

    // Weight for output transmission
    double_t outTransWeight = 1.0;

    // Weight for loading
    double_t inLoadingWeight = 1.0;

    // Weight for inter operator loading 
    double_t interOpTransWeight = 1.0;

    // Helper function
    std::string toString() {
        std::ostringstream oss;
        oss << "    Knobs:\n"
            << "        accTolerance: " << accTolerance << "\n"
            << "        colMulWeight: " << colMulWeight << "\n"
            << "        colAddWeight: " << colAddWeight << "\n"
            << "        outTransWeight: " << outTransWeight << "\n"
            << "        inLoadingWeight: " << inLoadingWeight << "\n"
            << "        interOpTransWeight: " << interOpTransWeight;
        return oss.str();
    };
};

}   // namespace pim

#endif // PIMOPT_DATALAYOUT_SUPPORT