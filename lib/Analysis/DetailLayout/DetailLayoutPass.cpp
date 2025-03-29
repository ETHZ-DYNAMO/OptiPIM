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

#include "pimopt/Analysis/DetailLayout/DetailLayoutPass.h"

using namespace mlir;
using namespace pim;


void DetailLayoutPass::parseConv2D(linalg::Conv2DNchwFchwOp op) {
    // Create a new layer info struct instance
    LayerInfo tmp_conv2d_info;

    // Update Layer Type
    tmp_conv2d_info.layerType = layerTypeAll::CONV2D;

    // Iterate through all attributes of the operation
    for (auto attr : op->getAttrs()) {
        llvm::StringRef attrName = attr.getName().strref();

        // Check for concerned attributes
        if (attrName == "strides") {
            // Cast the stride information to DenseIntElementsAttr to handle the dense integer vector
            auto strides = attr.getValue().dyn_cast<DenseIntElementsAttr>();

            for (int64_t stride : strides.getValues<int64_t>()) {
                tmp_conv2d_info.stride.push_back(stride);
            }

        } else if (attrName == "dilations") {
            // Cast the dilation information to DenseIntElementsAttr to handle the dense integer vector
            auto dilations = attr.getValue().dyn_cast<DenseIntElementsAttr>();

            for (int64_t dilation : dilations.getValues<int64_t>()) {
                tmp_conv2d_info.dilation.push_back(dilation);
            }
        } else if (attrName == "layer_group") {
            // Get the layer group information
            auto layer_group = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_group.getValue().getSExtValue();
            tmp_conv2d_info.layerGroupId = attrValue;
        } else if (attrName == "num_banks") {
            // Get the num_bank information
            auto num_banks = attr.getValue().dyn_cast<IntegerAttr>();

            int64_t attrValue = num_banks.getValue().getSExtValue();
            tmp_conv2d_info.numBanks = attrValue;
        } else if (attrName == "global_layer_idx") {
            // Get the global layer index info
            auto layer_idx = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_idx.getValue().getSExtValue();
            tmp_conv2d_info.layerId = attrValue;
        }
    }

    // Parse all operands of the conv2d operation (Error handling omitted)
    // We have three operands for conv2d operation:
    //      1. input featur map shape
    //      2. filter shape
    //      3. output shape
    auto operands = op.getOperands();

    try {
        // Operand 1: Input feature map
        RankedTensorType input_tensor_type = operands[0].getType().dyn_cast<RankedTensorType>();
        tmp_conv2d_info.batchSize = input_tensor_type.getShape()[0];
        tmp_conv2d_info.workLoadDimVec[workLoadDimensionConv2D::N] = input_tensor_type.getShape()[0];

        tmp_conv2d_info.inChanDepth = input_tensor_type.getShape()[1];
        tmp_conv2d_info.workLoadDimVec[workLoadDimensionConv2D::C] = input_tensor_type.getShape()[1];

        // Operand 2: Filter size
        RankedTensorType filter_tensor_type = operands[1].getType().dyn_cast<RankedTensorType>();
        tmp_conv2d_info.outChanDepth = filter_tensor_type.getShape()[0];
        tmp_conv2d_info.workLoadDimVec[workLoadDimensionConv2D::K] = filter_tensor_type.getShape()[0];

        tmp_conv2d_info.filterRowSize = filter_tensor_type.getShape()[2];
        tmp_conv2d_info.workLoadDimVec[workLoadDimensionConv2D::R] = filter_tensor_type.getShape()[2];

        tmp_conv2d_info.filterColSize = filter_tensor_type.getShape()[3];
        tmp_conv2d_info.workLoadDimVec[workLoadDimensionConv2D::S] = filter_tensor_type.getShape()[3];

        // Operand 3: Output feature map
        RankedTensorType output_tensor_type = operands[2].getType().dyn_cast<RankedTensorType>();
        tmp_conv2d_info.outRowSize = output_tensor_type.getShape()[2];
        tmp_conv2d_info.workLoadDimVec[workLoadDimensionConv2D::P] = output_tensor_type.getShape()[2];

        tmp_conv2d_info.outColSize = output_tensor_type.getShape()[3];
        tmp_conv2d_info.workLoadDimVec[workLoadDimensionConv2D::Q] = output_tensor_type.getShape()[3];
    } catch (std::exception &e) {
        llvm::errs() << "[ERROR] Linalg.Conv2d does not have all 3 operands desired\n" << e.what();
        exit(-1);
    }

    // Update layer group
    insertToLayerGroup(layerGroupsInfo, tmp_conv2d_info.layerGroupId, tmp_conv2d_info.layerId);

    // Storing the layer information
    // TODO: May need to check the following structure to a map, if the topological order matters
    layersInfo[tmp_conv2d_info.layerId] = tmp_conv2d_info;
}

// Sample: %208 = linalg.matmul {global_layer_idx = 53 : i32, layer_group = 0 : i32, num_banks = 80 : i32, type_layer_idx = 0 : i32} ins(%collapsed, %205 : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%207 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
void DetailLayoutPass::parseFC(linalg::MatmulOp matMulOp) {
    // Create a new layer info struct instance
    LayerInfo tmpFCInfo;

    // Update Layer Type
    tmpFCInfo.layerType = layerTypeAll::FC;

    // Iterate through all attributes of the operation
    for (auto attr : matMulOp->getAttrs()) {
        llvm::StringRef attrName = attr.getName().strref();

        // Check for concerned attributes
        if (attrName == "layer_group") {
            // Get the layer group information
            auto layer_group = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_group.getValue().getSExtValue();
            tmpFCInfo.layerGroupId = attrValue;
        } else if (attrName == "num_banks") {
            // Get the num_bank information
            auto num_banks = attr.getValue().dyn_cast<IntegerAttr>();

            int64_t attrValue = num_banks.getValue().getSExtValue();
            tmpFCInfo.numBanks = attrValue;
        } else if (attrName == "global_layer_idx") {
            // Get the global layer index info
            auto layer_idx = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_idx.getValue().getSExtValue();
            tmpFCInfo.layerId = attrValue;
        }
    }

    // Parse all operands of the FC operation (Error handling omitted)
    // We have three operands for conv2d operation:
    //      1. matrix_A
    //      2. matrix_B
    //      3. output size
    auto operands = matMulOp.getOperands();

    try {
        // Operand 1: Matrix A shape
        RankedTensorType mat_A_type = operands[0].getType().dyn_cast<RankedTensorType>();
        tmpFCInfo.filterRowSize = mat_A_type.getShape()[0];
        tmpFCInfo.filterColSize = mat_A_type.getShape()[1];
        tmpFCInfo.workLoadDimVec[workLoadDimensionFC::FC_P] = mat_A_type.getShape()[0];
        tmpFCInfo.workLoadDimVec[workLoadDimensionFC::FC_Q] = mat_A_type.getShape()[1];

        // Operand 2: Matrix B shape
        RankedTensorType mat_B_type = operands[1].getType().dyn_cast<RankedTensorType>();
        tmpFCInfo.outColSize = mat_B_type.getShape()[1];
        tmpFCInfo.workLoadDimVec[workLoadDimensionFC::FC_R] = mat_B_type.getShape()[1];

        // Operand 3: Output Shape
        RankedTensorType output_type = operands[2].getType().dyn_cast<RankedTensorType>();

        // Update Batch and output channel information -- Set to 1 for now
        // TODO: Check this out
        tmpFCInfo.batchSize = 1;
        tmpFCInfo.workLoadDimVec[workLoadDimensionFC::FC_N] = 1;

        tmpFCInfo.outChanDepth = 1;
        tmpFCInfo.workLoadDimVec[workLoadDimensionFC::FC_K] = 1;
    } catch (std::exception &e) {
        llvm::errs() << "[ERROR] Linalg.Matmul PARSING ERROR\n" << e.what();
        exit(-1);
    }

    // Update the layer grouping information
    insertToLayerGroup(layerGroupsInfo, tmpFCInfo.layerGroupId, tmpFCInfo.layerId);

    // Storing the layer information
    // TODO: May need to check the following structure to a map, if the topological order matters
    layersInfo[tmpFCInfo.layerId] = tmpFCInfo;
}

void DetailLayoutPass::parseBatchFC(mlir::linalg::BatchMatmulOp batchMatMulOp) {
    // Create a new layer info struct instance
    LayerInfo tmpFCInfo;

    // Update Layer Type
    tmpFCInfo.layerType = layerTypeAll::FC;

    // Iterate through all attributes of the operation
    for (auto attr : batchMatMulOp->getAttrs()) {
        llvm::StringRef attrName = attr.getName().strref();

        // Check for concerned attributes
        if (attrName == "layer_group") {
            // Get the layer group information
            auto layer_group = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_group.getValue().getSExtValue();
            tmpFCInfo.layerGroupId = attrValue;
        } else if (attrName == "num_banks") {
            // Get the num_bank information
            auto num_banks = attr.getValue().dyn_cast<IntegerAttr>();

            int64_t attrValue = num_banks.getValue().getSExtValue();
            tmpFCInfo.numBanks = attrValue;
        } else if (attrName == "global_layer_idx") {
            // Get the global layer index info
            auto layer_idx = attr.getValue().dyn_cast<IntegerAttr>();

            int32_t attrValue = layer_idx.getValue().getSExtValue();
            tmpFCInfo.layerId = attrValue;
        }
    }

    // Parse all operands of the FC operation (Error handling omitted)
    // We have three operands for conv2d operation:
    //      1. matrix_A
    //      2. matrix_B
    //      3. output size
    auto operands = batchMatMulOp.getOperands();

    try {
        // Operand 1: Matrix A shape
        RankedTensorType mat_A_type = operands[0].getType().dyn_cast<RankedTensorType>();
        tmpFCInfo.batchSize = mat_A_type.getShape()[0];
        tmpFCInfo.filterRowSize = mat_A_type.getShape()[1];
        tmpFCInfo.filterColSize = mat_A_type.getShape()[2];

        tmpFCInfo.workLoadDimVec[workLoadDimensionFC::FC_N] = mat_A_type.getShape()[0];
        tmpFCInfo.workLoadDimVec[workLoadDimensionFC::FC_P] = mat_A_type.getShape()[1];
        tmpFCInfo.workLoadDimVec[workLoadDimensionFC::FC_Q] = mat_A_type.getShape()[2];

        // Operand 2: Matrix B shape
        RankedTensorType mat_B_type = operands[1].getType().dyn_cast<RankedTensorType>();
        tmpFCInfo.outColSize = mat_B_type.getShape()[1];
        tmpFCInfo.workLoadDimVec[workLoadDimensionFC::FC_R] = mat_B_type.getShape()[1];

        // Operand 3: Output Shape
        RankedTensorType output_type = operands[2].getType().dyn_cast<RankedTensorType>();

        // Update Batch and output channel information -- Set to 1 for now
        // TODO: Check this out
        tmpFCInfo.outChanDepth = 1;
        tmpFCInfo.workLoadDimVec[workLoadDimensionFC::FC_K] = 1;
    } catch (std::exception &e) {
        llvm::errs() << "[ERROR] Linalg.Matmul PARSING ERROR\n" << e.what();
        exit(-1);
    }

    // Update the layer grouping information
    insertToLayerGroup(layerGroupsInfo, tmpFCInfo.layerGroupId, tmpFCInfo.layerId);

    // Storing the layer information
    // TODO: May need to change the following structure to a map, if the topological order matters
    layersInfo[tmpFCInfo.layerId] = tmpFCInfo;
}

void DetailLayoutPass::runOnOperation() {
    // This pass will operate on different types of ML layers,
    Operation *op = getOperation();

    op->walk([&](Operation *op) {
        // Check different types of layers
        if (auto conv2DOp = dyn_cast<linalg::Conv2DNchwFchwOp>(op)) {
            parseConv2D(conv2DOp);
        } else if (auto FCOp = dyn_cast<linalg::MatmulOp>(op)) {
            parseFC(FCOp);
        } else if (auto BatchFC = dyn_cast<linalg::BatchMatmulOp>(op)) {
            parseBatchFC(BatchFC);
        }
    });

    // Testing
    printLayerGroupInfo();

    // ==================
    // Build MILP Models
    // ==================
    // Create a single Gurobi environment (may need refinement)
    GRBEnv env = GRBEnv(true);

    // Set Gurobi environment parameters
    // env.set(GRB_IntParam_OutputFlag, 0); // Clean Console output
    // For now, we set the timeout of gurobi search process to 180 seconds
    env.set(GRB_DoubleParam_TimeLimit, 400);

    env.start(); // Initialize the environment

    int32_t layerGroupCounter = 0;
    // Version 1: For now, we optimize the layer one by one
    for (auto layer_group : layerGroupsInfo) {
        // Create DetailLayout MILP instance, different models will share the same env
        // llvm::outs() << "DEBUG IN detailLayoutPass.cpp PATH: " << archPath << " " << knobPath << "\n";  //! this is fine
        
        // Check the size of the layergroup
        if (layer_group.second.size() < 1) {
            llvm::outs() << "[ERROR] The size of the layer group is 0" << "\n";
            llvm::outs() << "[ERROR] Please use the correct linalg.mlir file!" << "\n";
            exit(-1);
        } else if (layer_group.second[0] <= -5000 || layer_group.second[0] >= 5000) {
            llvm::outs() << "[ERROR] Layer ID: " << layer_group.second[0] << "\n";
            llvm::outs() << "[ERROR] Please use the correct linalg.mlir file!" << "\n";
            exit(-1);
        }
        
        DetailLayoutMILP dl_opt(env, layer_group.second, layersInfo, 
                                milpModelPath, outputPath, archPath, 
                                knobPath, layoutScheme, memoryAllocScheme, 
                                objApproMethod, layerGroupCounter);
    
        layerGroupCounter++;
    }
}

void DetailLayoutPass::printLayerGroupInfo() {
    int tmp_layer_group_id = 0;

    llvm::outs() << "Layer Group Info: \n";

    for (auto layer_group : layerGroupsInfo) {
        llvm::outs() << "\t Group " << tmp_layer_group_id << ": [";

        for (auto tmp_layer_id : layer_group.second) {
            llvm::outs() << tmp_layer_id << " ";
        } 

        llvm::outs() << "];\n";

        tmp_layer_group_id++;
    }
}

std::vector<LayerInfo> DetailLayoutPass::getLayerGroupInfo(std::vector<int32_t> sel_layer_group) {
    std::vector<LayerInfo> sel_layers;

    // Iterate through all layers in the layer group
    for (auto sel_layer : sel_layer_group) {
        sel_layers.push_back(layersInfo[sel_layer]);

        // Testing
        // layersInfo[sel_layer].print_detail();
    }

    return sel_layers;
}
