#include <iostream>
#include <cstring>
#include <stack>
#include <fstream>
#include <stdexcept>

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

#include "pimopt/Analysis/ExtractOpParamsPass.h"

using namespace mlir;
using namespace pim;

void ExtractOpParamsPass::parseConv(linalg::Conv2DNchwFchwOp conv_op, mlir::MLIRContext &context,
                                    int32_t type_id, int32_t layer_id) {
  auto stride_attr = conv_op->getAttrOfType<DenseIntElementsAttr>("strides");
  auto dilation_attr = conv_op->getAttrOfType<DenseIntElementsAttr>("dilations");

  auto operands = conv_op.getOperands();
  auto numOperands = std::distance(operands.begin(), operands.end());

  mlir::RankedTensorType input_tensor_type;
  mlir::RankedTensorType filter_tensor_type;
  mlir::RankedTensorType output_tensor_type;
  
  try {
    input_tensor_type  = operands[0].getType().dyn_cast<mlir::RankedTensorType>();
    filter_tensor_type = operands[1].getType().dyn_cast<mlir::RankedTensorType>();
    output_tensor_type = operands[2].getType().dyn_cast<mlir::RankedTensorType>();
  } catch (std::exception &e) {
    llvm::errs() << "[ERROR] conv2d does not have 3 operands\n" << e.what();
    exit(-1);
  }

  // Example implementation of layer group
  mlir::Builder builder(conv_op.getContext());
  if (type_id < 25) {
    conv_op->setAttr("layer_group", builder.getI32IntegerAttr(0));
  } else {
    conv_op->setAttr("layer_group", builder.getI32IntegerAttr(1));
  }

  // Example num_banks per op
  conv_op->setAttr("num_banks", builder.getI32IntegerAttr(128));
  conv_op->setAttr("type_layer_idx", builder.getI32IntegerAttr(type_id));
  conv_op->setAttr("global_layer_idx", builder.getI32IntegerAttr(layer_id));


  int32_t N, K, P, Q, C, R, S;
  int32_t stride_H, stride_W, dilation_H, dilation_W;

  N = input_tensor_type.getShape()[0];
  C = input_tensor_type.getShape()[1];
  R = filter_tensor_type.getShape()[2];
  S = filter_tensor_type.getShape()[3];
  K = output_tensor_type.getShape()[1];
  P = output_tensor_type.getShape()[2];
  Q = output_tensor_type.getShape()[3];
  
  for (auto attr : conv_op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().strref();

    // Check for concerned attributes
    if (attrName == "strides") {
      auto strides = attr.getValue().dyn_cast<DenseIntElementsAttr>();

      for (int64_t stride : strides.getValues<int64_t>()) {
        stride_H = stride_W = stride;
      }

    } else if (attrName == "dilations") {
      auto dilations = attr.getValue().dyn_cast<DenseIntElementsAttr>();

      for (int64_t dilation : dilations.getValues<int64_t>()) {
        dilation_H = dilation_W = dilation;
      }
    } 
  }

  std::ofstream outputFile;
  if (layer_id == 0) {  // refresh file content each run 
    outputFile = std::ofstream("layer_params.csv");
  } else {
    outputFile = std::ofstream("layer_params.csv", std::ios::app);
  }

  outputFile << layer_id << ", CONV2d, ";
  outputFile << type_id << ", " << N << ", " << K << ", " << P << ", " << Q << ", " << C << ", " << R << ", " << S;
  outputFile << ", " << stride_H << ", " << stride_W << ", " << dilation_H << ", " << dilation_W << "\n";
  outputFile.close();
}

void ExtractOpParamsPass::parseFC(linalg::MatmulOp matmul_op, mlir::MLIRContext &context, int32_t type_id, int32_t layer_id) {
  mlir::RankedTensorType mat_A_type;
  mlir::RankedTensorType mat_B_type;
  mlir::RankedTensorType output_tensor_type;

  auto operands = matmul_op.getOperands();
  auto numOperands = std::distance(operands.begin(), operands.end());
  
  try {
    mat_A_type = operands[0].getType().dyn_cast<mlir::RankedTensorType>();
    mat_B_type = operands[1].getType().dyn_cast<mlir::RankedTensorType>();
    output_tensor_type = operands[2].getType().dyn_cast<mlir::RankedTensorType>();
  } catch (std::exception &e) {
    llvm::errs() << "[ERROR] parsing fully connected op\n" << e.what();
    exit(-1);
  }

  int32_t N = 1, K = 1, P, Q, C = -1, R, S = -1;  // K is H (head)
  P = mat_A_type.getShape()[0];
  Q = mat_A_type.getShape()[1];
  R = mat_B_type.getShape()[1];

  std::ofstream outputFile;
  if (layer_id == 0) {  // refresh file content each run 
    outputFile = std::ofstream("layer_params.csv");
  } else {
    outputFile = std::ofstream("layer_params.csv", std::ios::app);
  }

  mlir::Builder builder(matmul_op.getContext());
  if (type_id < 25) {
    matmul_op->setAttr("layer_group", builder.getI32IntegerAttr(0));
  } else {
    matmul_op->setAttr("layer_group", builder.getI32IntegerAttr(1));
  }

  // Example num_banks per op
  matmul_op->setAttr("num_banks", builder.getI32IntegerAttr(128));
  matmul_op->setAttr("type_layer_idx", builder.getI32IntegerAttr(type_id));
  matmul_op->setAttr("global_layer_idx", builder.getI32IntegerAttr(layer_id));

  outputFile << layer_id << ", FC, ";
  outputFile << type_id << ", " << N << ", " << K << ", " << P << ", " << Q << ", " << C << ", " << R << ", " << S;
  outputFile << ", " << -1 << ", " << -1 << ", " << -1 << ", " << -1 << "\n";
  outputFile.close();
}

void ExtractOpParamsPass::parseBatchMatmul(linalg::BatchMatmulOp batch_matmul_op, mlir::MLIRContext &context, int32_t type_id, int32_t layer_id) {
  mlir::RankedTensorType mat_A_type;
  mlir::RankedTensorType mat_B_type;
  mlir::RankedTensorType output_tensor_type;

  auto operands = batch_matmul_op.getOperands();
  auto numOperands = std::distance(operands.begin(), operands.end());
  
  try {
    mat_A_type = operands[0].getType().dyn_cast<mlir::RankedTensorType>();
    mat_B_type = operands[1].getType().dyn_cast<mlir::RankedTensorType>();
    output_tensor_type = operands[2].getType().dyn_cast<mlir::RankedTensorType>();
  } catch (std::exception &e) {
    llvm::errs() << "[ERROR] parsing fully connected op\n" << e.what();
    exit(-1);
  }

  mlir::Builder builder(batch_matmul_op.getContext());
  if (type_id < 25) {
    batch_matmul_op->setAttr("layer_group", builder.getI32IntegerAttr(0));
  } else {
    batch_matmul_op->setAttr("layer_group", builder.getI32IntegerAttr(1));
  }

  // Example num_banks per op
  batch_matmul_op->setAttr("num_banks", builder.getI32IntegerAttr(128));
  batch_matmul_op->setAttr("type_layer_idx", builder.getI32IntegerAttr(type_id));
  batch_matmul_op->setAttr("global_layer_idx", builder.getI32IntegerAttr(layer_id));
  

  int32_t N = 1, K = -1, P, Q, C = -1, R, S = -1;  // K is H (head)
  N = mat_A_type.getShape()[0];
  P = mat_A_type.getShape()[1];
  Q = mat_A_type.getShape()[2];
  R = mat_B_type.getShape()[1];

  std::ofstream outputFile;
  if (layer_id == 0) {  // refresh file content each run 
    outputFile = std::ofstream("layer_params.csv");
  } else {
    outputFile = std::ofstream("layer_params.csv", std::ios::app);
  }

  outputFile << layer_id << ", batch_matmul, ";
  outputFile << type_id << ", " << N << ", " << K << ", " << P << ", " << Q << ", " << C << ", " << R << ", " << S;
  outputFile << ", " << -1 << ", " << -1 << ", " << -1 << ", " << -1 << "\n";
  outputFile.close();
}

void ExtractOpParamsPass::runOnOperation() {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);

  Operation *op = getOperation();
  int32_t layer_id = 0, matmul_id = 0, conv_id = 0;
  op->walk([&](Operation *op) {
    if (auto conv_op = dyn_cast<linalg::Conv2DNchwFchwOp>(op)) {
      parseConv(conv_op, context, conv_id, layer_id);
      conv_id++;
      layer_id++;
    }

    // regular fc / matmul
    if (auto fc_op = dyn_cast<linalg::MatmulOp>(op)) {
      parseFC(fc_op, context, matmul_id, layer_id);
      matmul_id++;
      layer_id++;
    }

    if (auto batch_matmul_op = dyn_cast<linalg::BatchMatmulOp>(op)) {
      parseBatchMatmul(batch_matmul_op, context, matmul_id, layer_id);
      matmul_id++;
      layer_id++;
    }
  });
}