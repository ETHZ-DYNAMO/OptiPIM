#include <iostream>
#include <cstring>
#include <stack>
#include <fstream>
#include <stdexcept>

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"

#include "pimopt/Analysis/DataLayout/DataLayoutPass.h"

using namespace mlir;
using namespace pim;

//
//  Parsing Functions
//
void DataLayoutPass::parseConv2D(linalg::Conv2DNchwFchwOp op) {
    // Create a new opInfo instance
    OperatorInfo tmpConv2DInfo;

    // Update op type
    tmpConv2DInfo.opType = operatorTypesIdx::CONV2D;

    // Iterate through all attributes of the operation
    for (auto attr : op->getAttrs()) {
        llvm::StringRef attrName = attr.getName().strref();

        // Check for concerned attributes
        if (attrName == "strides") {
            // Cast the stride information to DenseIntElementsAttr to handle the dense integer vector
            auto strides = attr.getValue().dyn_cast<DenseIntElementsAttr>();

            for (int64_t stride : strides.getValues<int64_t>()) {
                tmpConv2DInfo.stride.push_back(stride);
            }

        } else if (attrName == "dilations") {
            // Cast the dilation information to DenseIntElementsAttr to handle the dense integer vector
            auto dilations = attr.getValue().dyn_cast<DenseIntElementsAttr>();

            for (int64_t dilation : dilations.getValues<int64_t>()) {
                tmpConv2DInfo.dilation.push_back(dilation);
            }
        } else if (attrName == "layer_group") {
            // Get the operator group information
            auto layer_group = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_group.getValue().getSExtValue();
            tmpConv2DInfo.opGroupID = attrValue;
        } else if (attrName == "num_banks") {
            // Get the numPE information
            auto num_banks = attr.getValue().dyn_cast<IntegerAttr>();

            int64_t attrValue = num_banks.getValue().getSExtValue();
            tmpConv2DInfo.numPE = attrValue;
        } else if (attrName == "global_layer_idx") {
            // Get the global operator index info
            auto layer_idx = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_idx.getValue().getSExtValue();
            tmpConv2DInfo.opID = attrValue;
        }
    }

    // Parse all operands of the conv2d operation to get the loop bound information
    // We have three operands for conv2d operation:
    //      1. input featur map shape
    //      2. filter shape
    //      3. output shape
    auto operands = op.getOperands();

    try {
        // Operand 1: Batch size and input feature map
        RankedTensorType input_tensor_type = operands[0].getType().dyn_cast<RankedTensorType>();
        tmpConv2DInfo.loopBoundVec[loopBoundIdxConv2D::N] = input_tensor_type.getShape()[0];

        tmpConv2DInfo.loopBoundVec[loopBoundIdxConv2D::C] = input_tensor_type.getShape()[1];

        // Operand 2: Filter Array
        RankedTensorType filter_tensor_type = operands[1].getType().dyn_cast<RankedTensorType>();
        tmpConv2DInfo.loopBoundVec[loopBoundIdxConv2D::K] = filter_tensor_type.getShape()[0];

        tmpConv2DInfo.loopBoundVec[loopBoundIdxConv2D::R] = filter_tensor_type.getShape()[2];

        tmpConv2DInfo.loopBoundVec[loopBoundIdxConv2D::S] = filter_tensor_type.getShape()[3];

        // Operand 3: Output feature map
        RankedTensorType output_tensor_type = operands[2].getType().dyn_cast<RankedTensorType>();
        tmpConv2DInfo.loopBoundVec[loopBoundIdxConv2D::P] = output_tensor_type.getShape()[2];

        tmpConv2DInfo.loopBoundVec[loopBoundIdxConv2D::Q] = output_tensor_type.getShape()[3];
    } catch (std::exception &e) {
        llvm::errs() << "[ERROR] Linalg.Conv2d does not have all 3 operands desired\n" << e.what();
        exit(-1);
    }

    // Update Op Group
    insertToOpGroup(opGroupsInfo, tmpConv2DInfo.opGroupID, tmpConv2DInfo.opID);

    // Storing the layer info
    // TODO: This storing method may need to be changed if the topological info matters
    opsInfo[tmpConv2DInfo.opID] = tmpConv2DInfo;
}

void DataLayoutPass::parseFC(linalg::MatmulOp matMulOp) {
    // Create a new opInfo instance
    OperatorInfo tmpFCInfo;

    // Update Op Type
    tmpFCInfo.opType = operatorTypesIdx::FC;

    // Iterate through all attributes of the operation
    for (auto attr : matMulOp->getAttrs()) {
        llvm::StringRef attrName = attr.getName().strref();

        // Check for concerned attributes
        if (attrName == "layer_group") {
            // Get the op group information
            auto layer_group = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_group.getValue().getSExtValue();
            tmpFCInfo.opGroupID = attrValue;
        } else if (attrName == "num_banks") {
            // Get the numPE information
            auto num_banks = attr.getValue().dyn_cast<IntegerAttr>();

            int64_t attrValue = num_banks.getValue().getSExtValue();
            tmpFCInfo.numPE = attrValue;
        } else if (attrName == "global_layer_idx") {
            // Get the global op index info
            auto layer_idx = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_idx.getValue().getSExtValue();
            tmpFCInfo.opID = attrValue;
        }
    }

    // Parse all operands of the FC operation 
    // We have three operands for matmul operation:
    //      1. matrix_A
    //      2. matrix_B --> Weight
    //      3. output array
    auto operands = matMulOp.getOperands();

    try {
        // Operand 1: Matrix A shape
        RankedTensorType mat_A_type = operands[0].getType().dyn_cast<RankedTensorType>();
        tmpFCInfo.loopBoundVec[loopBoundIdxFC::FC_P] = mat_A_type.getShape()[0];
        tmpFCInfo.loopBoundVec[loopBoundIdxFC::FC_Q] = mat_A_type.getShape()[1];

        // Operand 2: Matrix B shape
        RankedTensorType mat_B_type = operands[1].getType().dyn_cast<RankedTensorType>();
        tmpFCInfo.loopBoundVec[loopBoundIdxFC::FC_R] = mat_B_type.getShape()[1];

        // Operand 3: Output Shape
        RankedTensorType output_type = operands[2].getType().dyn_cast<RankedTensorType>();

        // Update Batch and output channel information -- Set to 1 for now
        tmpFCInfo.loopBoundVec[loopBoundIdxFC::FC_N] = 1;
        tmpFCInfo.loopBoundVec[loopBoundIdxFC::FC_K] = 1;
    } catch (std::exception &e) {
        llvm::errs() << "[ERROR] Linalg.Matmul PARSING ERROR\n" << e.what();
        exit(-1);
    }

    // Update the layer grouping information
    insertToOpGroup(opGroupsInfo, tmpFCInfo.opGroupID, tmpFCInfo.opID);

    // Storing the layer information
    // TODO: May need to check the following structure to a map, if the topological order matters
    opsInfo[tmpFCInfo.opID] = tmpFCInfo;
}

void DataLayoutPass::parseBatchFC(linalg::BatchMatmulOp batchMatMulOp) {
    // Create a new opInfo instance
    OperatorInfo tmpFCInfo;

    // Update Op Type
    tmpFCInfo.opType = operatorTypesIdx::FC;

    // Iterate through all attributes of the operation
    for (auto attr : batchMatMulOp->getAttrs()) {
        llvm::StringRef attrName = attr.getName().strref();

        // Check for concerned attributes
        if (attrName == "layer_group") {
            // Get the op group information
            auto layer_group = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_group.getValue().getSExtValue();
            tmpFCInfo.opGroupID = attrValue;
        } else if (attrName == "num_banks") {
            // Get the numPE information
            auto num_banks = attr.getValue().dyn_cast<IntegerAttr>();

            int64_t attrValue = num_banks.getValue().getSExtValue();
            tmpFCInfo.numPE = attrValue;
        } else if (attrName == "global_layer_idx") {
            // Get the global op index info
            auto layer_idx = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_idx.getValue().getSExtValue();
            tmpFCInfo.opID = attrValue;
        }
    }

    // Parse all operands of the FC operation 
    // We have three operands for matmul operation:
    //      1. matrix_A
    //      2. matrix_B --> Weight
    //      3. output array
    auto operands = batchMatMulOp.getOperands();

    try {
        // Operand 1: Matrix A shape
        RankedTensorType mat_A_type = operands[0].getType().dyn_cast<RankedTensorType>();
        tmpFCInfo.loopBoundVec[loopBoundIdxFC::FC_N] = mat_A_type.getShape()[0];
        tmpFCInfo.loopBoundVec[loopBoundIdxFC::FC_P] = mat_A_type.getShape()[1];
        tmpFCInfo.loopBoundVec[loopBoundIdxFC::FC_Q] = mat_A_type.getShape()[2];

        // Operand 2: Matrix B shape
        RankedTensorType mat_B_type = operands[1].getType().dyn_cast<RankedTensorType>();
        tmpFCInfo.loopBoundVec[loopBoundIdxFC::FC_R] = mat_B_type.getShape()[2];

        // Operand 3: Output Shape
        RankedTensorType output_type = operands[2].getType().dyn_cast<RankedTensorType>();

        // Update Batch and output channel information -- Set to 1 for now
        
        tmpFCInfo.loopBoundVec[loopBoundIdxFC::FC_K] = 1;
    } catch (std::exception &e) {
        llvm::errs() << "[ERROR] Linalg.Matmul PARSING ERROR\n" << e.what();
        exit(-1);
    }

    // Update the layer grouping information
    insertToOpGroup(opGroupsInfo, tmpFCInfo.opGroupID, tmpFCInfo.opID);

    // Storing the layer information
    // TODO: May need to check the following structure to a map, if the topological order matters
    opsInfo[tmpFCInfo.opID] = tmpFCInfo;
}

//
//  Main runonop function
//

void DataLayoutPass::runOnOperation() {
    // This pass will operate on different types of ML layers,
    Operation *op = getOperation();

    op->walk([&](Operation *op) {
        // Check the type of different operators, BatchFC omitted for now
        if (auto conv2DOp = dyn_cast<linalg::Conv2DNchwFchwOp>(op)) {
            parseConv2D(conv2DOp);
        } else if (auto FCOp = dyn_cast<linalg::MatmulOp>(op)) {
            parseFC(FCOp);
        } else if (auto BatchFCOp = dyn_cast<linalg::BatchMatmulOp>(op)) {
            parseBatchFC(BatchFCOp);
        }
    });

    // ========================
    //    Build MILP Models
    // ========================
    // Create a single Gurobi environment (may need refinement)
    try {

    
        GRBEnv env = GRBEnv(true);

        // Set Gurobi environment parameters
        // env.set(GRB_IntParam_OutputFlag, 0); // Clean Console output
        // For now, we set the timeout of gurobi search process to 400 seconds
        env.set(GRB_DoubleParam_TimeLimit, 400);

        //! Testing, set numerical focus
        env.set(GRB_IntParam_NumericFocus, 3);

        //! Set the MIPFocus
        env.set(GRB_IntParam_MIPFocus, 1);

        env.start(); // Initialize the environment
    

        int32_t opGroupCounter = 0;
        for (auto opGroup : opGroupsInfo) {
            // Check the size of the layer group
            if (opGroup.second.size() < 1) {
                llvm::outs() << "[ERROR] The size of the layer group is 0" << "\n";
                llvm::outs() << "[ERROR] Please use the correct linalg.mlir file!" << "\n";
                exit(-1);
            } else if (opGroup.second[0] <= -5000 || opGroup.second[0] >= 5000) {
                llvm::outs() << "[ERROR] Layer ID: " << opGroup.second[0] << "\n";
                llvm::outs() << "[ERROR] Please use the correct linalg.mlir file!" << "\n";
                exit(-1);
            }

            // Create the DataLayout MILP model
            //TODO: Remove the following error handling after finishing the code
            try {
                DataLayoutMILP dlOpt(env, opGroup.second, opsInfo,
                                milpModelPath, outputPath, archPath,
                                knobPath, deviceType, memoryAllocScheme,
                                objMethod, opGroupCounter, transCoeffMethod,
                                numCountingMethod, storageMethod);
            } catch (GRBException& e) {
                llvm::outs() << "[Construction] Gurobi Exception occurred: " << e.getMessage() << "\n";
            }
            // DataLayoutMILP dlOpt(env, opGroup.second, opsInfo,
            //                      milpModelPath, outputPath, archPath,
            //                      knobPath, deviceType, memoryAllocScheme,
            //                      linMethod, opGroupCounter, transCoeffMethod);

            opGroupCounter++;
        }
    } catch (GRBException& e) {
        llvm::outs() << "[Construction] Gurobi Exception occurred: " << e.getMessage() << "\n";
    }
}

//
//  Helper functions
//

void DataLayoutPass::printOpGroupInfo() {
    int tmp_layer_group_id = 0;

    llvm::outs() << "Layer Group Info: \n";

    for (auto layer_group : opGroupsInfo) {
        llvm::outs() << "\t Group " << tmp_layer_group_id << ": [";

        for (auto tmp_layer_id : layer_group.second) {
            llvm::outs() << tmp_layer_id << " ";
        } 

        llvm::outs() << "];\n";

        tmp_layer_group_id++;
    }
}