#ifndef ANALYSIS_EXTRACTOPPARAMS
#define ANALYSIS_EXTRACTOPPARAMS

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

namespace pim {
    
struct ExtractOpParamsPass : public mlir::PassWrapper<ExtractOpParamsPass, mlir::OperationPass<>> {

  void runOnOperation() override;

  void parseConv(mlir::linalg::Conv2DNchwFchwOp op, mlir::MLIRContext &context, int32_t conv_id, int32_t layer_id);

  void parseFC(mlir::linalg::MatmulOp, mlir::MLIRContext &context, int32_t fc_id, int32_t layer_id);  // linear / FC

  void parseBatchMatmul(mlir::linalg::BatchMatmulOp, mlir::MLIRContext &contect, int32_t matmul_id, int32_t layer_id);
  
  llvm::StringRef getArgument() const final {
    return "Extract Op Params";
  }

  llvm::StringRef getDescription() const final {
    return "Extract Op Params from conv2d, matmul, and batch_matmul ops in linalg";
  }

  int conv_id = 0;
};

} // namespace pim

#endif // ANALYSIS_EXTRACTOPPARAMS