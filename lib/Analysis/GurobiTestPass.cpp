#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <cstring>
#include <stack>
#include <fstream>

#include "pimopt/Analysis/GurobiTestPass.h"

using namespace mlir;
using namespace pim;

// Check the existence of Gurobi library
#ifndef PIMOPT_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

void GurobiTestPass::runOnOperation() {
  Operation *rootOp = getOperation();
    // op->walk() is post order
    // operate only on func::return
    rootOp->walk([&](Operation *op) {
      if (auto forOp = dyn_cast<func::ReturnOp>(op)) {
        // This is a return op
        //! We test a simple mip here, just for testing, has nothing to do with our project
        // maximize x + y + 2 z
        // subject to : x + 2 y + 3 z <= 4
        //              x + y         >= 1
        //              x, y, z binary

        GRBEnv env = GRBEnv(true);
        env.start();

        // Create an empty model
        GRBModel model = GRBModel(env);

        // Create variables
        GRBVar x = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x");
        GRBVar y = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "y");
        GRBVar z = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "z");

        // Set Objective: maximize x + y + 2 z
        model.setObjective(x + y + 2 * z, GRB_MAXIMIZE);

        // Add constraint
        model.addConstr(x + 2 * y + 3 * z <= 4, "c0");

        model.addConstr(x + y >= 1, "c1");

        // Optimize model
        model.optimize();

        // Generate output
        llvm::outs() << x.get(GRB_StringAttr_VarName) << " "
                     << x.get(GRB_DoubleAttr_X) << "\n";
        
        llvm::outs() << y.get(GRB_StringAttr_VarName) << " "
                     << y.get(GRB_DoubleAttr_X) << "\n";

        llvm::outs() << z.get(GRB_StringAttr_VarName) << " "
                     << z.get(GRB_DoubleAttr_X) << "\n";

        llvm::outs() << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << "\n";
      }
    });
}

// void registerGurobiTestPass() {
//   PassRegistration<GurobiTestPass>();
// }

#endif // PIMOPT_GUROBI_NOT_INSTALLED