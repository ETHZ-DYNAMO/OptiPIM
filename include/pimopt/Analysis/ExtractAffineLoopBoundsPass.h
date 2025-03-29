#ifndef ANALYSIS_LOOPBOUND
#define ANALYSIS_LOOPBOUND

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace pim {

constexpr char LOOP_SYMBOL[] = {'N', 'K', 'P', 'Q', 'C', 'R', 'S'};
// batch, output channel, output H, output W, input channel, kernel H, kernel W

// Define the Pass class
struct ExtractAffineLoopBoundsPass : public mlir::PassWrapper<ExtractAffineLoopBoundsPass, mlir::OperationPass<>> {
    void parseNestedLoop(mlir::affine::AffineForOp &forOp);

    void runOnOperation() override;

    llvm::StringRef getArgument() const final {
        return "Input file in affine.mlir";
    }

    llvm::StringRef getDescription() const final { return "Testing the Gurobi Solver"; }
};

} // namespace pimopt

#endif // ANALYSIS_LOOPBOUND