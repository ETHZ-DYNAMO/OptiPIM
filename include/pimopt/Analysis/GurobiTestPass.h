#ifndef ANALYSIS_GUROBITEST
#define ANALYSIS_GUROBITEST

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace pim {

struct GurobiTestPass : public mlir::PassWrapper<GurobiTestPass, mlir::OperationPass<>> {
  // Implemented in GurobiTestPass.cpp
  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "Input file in affine.mlir"; }

  llvm::StringRef getDescription() const final { return "Testing the Gurobi Solver"; }
};

// Add the pass registration function
// void registerGurobiTestPass();

} // namespace pim

#endif // ANALYSIS_GUROBITEST