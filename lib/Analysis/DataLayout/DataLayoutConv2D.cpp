//===----------------------------------------------------------------------===//
//
// This file implements all MILP modeling related funcs for CONV2D.
//
//===----------------------------------------------------------------------===//
#include "pimopt/Analysis/DataLayout/DataLayoutMILP.h"

// Check the existence of Gurobi library
#ifndef PIMOPT_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace mlir;
using namespace pim;

// =============================================
//          Overall GRB Construction Func
// =============================================

GRBLinExpr DataLayoutMILP::constructConv2DMILP(const OperatorInfo& selOpInstance, const int32_t& selOpIndex) {
    // Return expression
    GRBLinExpr outputExpr;

    //
    // Step 1: Construct all the constant values for the layer
    //
    buildConv2DOpConstants(selOpInstance);

    //
    // Step 2: Create layer specific variables
    //
    OpGRBVariables tmpOpVariables;
    opVariablesMap[selOpIndex] = tmpOpVariables;

    // Step 2.1: Add all loop bound variables
    addLoopBoundVarsConv2D(selOpIndex);

    // Step 2.2 Add all transformation coefficient variables
    addTransCoeffVarsConv2D(selOpIndex);

    // Step 2.3 Add all related variables and constraints for
    //          the number of multiplications in a column in the PE
    addNumMulColVarConv2D(selOpIndex);

    // Step 2.4 Add variable for the number of columns used in the PE
    addNumColPEVarConv2D(selOpIndex);

    // Step 2.5 Add variable for the number of PEs used in the system
    addNumPESysVarConv2D(selOpIndex);

    // Step 2.6 Add variable for the number of output elements per column
    addNumOutColVarConv2D(selOpIndex);
    
    // Step 2.7 Add variable for the number of filter elements per column
    addNumFilColVarConv2D(selOpIndex);

    // Step 2.8 Add variable for the number of reductions per output in a column
    addNumRedOutColVarConv2D(selOpIndex);

    // Step 2.9 Add variable for the number of inputs in a column
    addNumInColVarConv2D(selOpIndex);

    // Step 2.10 Add Variable for the numebr of reductions per column
    addNumRedColVarConv2D(selOpIndex);

    // Step 2.11 Add Variable for the numebr of output partial sums per PE
    addNumOutPEVarConv2D(selOpIndex);

    // Step 2.12 Add variable to represent the number of output transmissin in the workload
    addNumOutSysVarConv2D(selOpIndex);

    // Step 2.13 Add variable to represent the number of inputs in a PE
    addNumInPEVarConv2D(selOpIndex);

    // Step 2.14 Add variable to represent the loading cost
    addNumInSysVarConv2D(selOpIndex);

    // If the target device is PNM
    // Add the filter loading cost
    if (selDeviceType == deviceTypeIdx::PNM) {
        addFilPELoadingVarConv2D(selOpIndex);
    }


    //
    // Step 3: Add all related constraints
    //
    // Step 3.1: Add loop bound constraints
    addLoopBoundConsConv2D(selOpIndex);

    // Step 3.2: Add transformation coefficient constraints
    addTransCoeffConsConv2D(selOpIndex);

    // Step 3.3: Add NumPE constraint
    addNumPEConsConv2D(selOpIndex);

    // Step 3.4 Add NumCol Constraint
    addNumColConsConv2D(selOpIndex);

    // Step 3.5 Add ColSize Constraint
    addColSizeConsConv2D(selOpIndex);

    //
    // Step 4: Add the final objective
    //

    // Select based on targeted device
    if (selDeviceType == deviceTypeIdx::PUM) {
        // Check the intended cost function
        if (selObjMethod == 0) {
            // Our Cost function
            // Cost 1: Get the col multiplication cost
            outputExpr += (opVariablesMap[selOpIndex].numMulCol * archInfo.mulLat) * objKnobs.colMulWeight;

            // Cost 2: Get the col reduction cost
            outputExpr += (opVariablesMap[selOpIndex].numRedCol * archInfo.addLat) * objKnobs.colAddWeight;
            
            // Cost 3: Get the output transmission cost
            outputExpr += opVariablesMap[selOpIndex].numOutSys * objKnobs.outTransWeight;

            // Cost 4: Get the input loading cost
            outputExpr += opVariablesMap[selOpIndex].numInSys * objKnobs.inLoadingWeight;
        } else if (selObjMethod == 1) {
            // COSA's COST Function
            // Cost 1: Get the col multiplication cost
            outputExpr += (opVariablesMap[selOpIndex].numMulCol * archInfo.mulLat) * objKnobs.colMulWeight;

            // Cost 2: Get the col reduction cost
            outputExpr += (opVariablesMap[selOpIndex].numRedCol * archInfo.addLat) * objKnobs.colAddWeight;
        }
        
    } else if (selDeviceType == deviceTypeIdx::PNM) {
        // Check the intended cost function
        if (selObjMethod == 0) {
            // Cost 1: Get the input loading cost
            outputExpr += opVariablesMap[selOpIndex].numInSys * objKnobs.inLoadingWeight;

            // Cost 2: Computation Row Activation Cost
            outputExpr += opVariablesMap[selOpIndex].numFilCol * archInfo.rowAct * objKnobs.colMulWeight;

            // Cost 3: Bank Filter loading cost
            outputExpr += opVariablesMap[selOpIndex].numFilPELoading * objKnobs.colAddWeight;

            // Cost 4: Get the output transmission cost
            outputExpr += opVariablesMap[selOpIndex].numOutSys * objKnobs.outTransWeight;
        } else if (selObjMethod == 1) {
            // COSA's cost function
            // Cost 2: Computation Row Activation Cost
            outputExpr += opVariablesMap[selOpIndex].numFilCol * archInfo.rowAct * objKnobs.colMulWeight;

            // Cost 3: Bank Filter loading cost
            outputExpr += opVariablesMap[selOpIndex].numFilPELoading * objKnobs.colAddWeight;
        }
        
    }

    // Final return
    return outputExpr;
}

// =============================================
//      GRB Variable Construction -- Conv2D
// =============================================

void DataLayoutMILP::addLoopBoundVarsConv2D(const int32_t& selOpIdx) {
    
    // Create a Gurobi binary variable of the given name and type for all loop bound in resulting nested loop
    auto createBinaryVar = [&](const std::string &name, char type) {
        return model.addVar(0, 1, 0.0, type, name);
    };

    // Get the constant info of this Op
    OpConstInfo& selOpConstInfo = opConstInfoMap[selOpIdx];

    // Iterate over all loop bounds d
    std::vector<LoopBoundVariables> tmpLoopBoundVars;
    for (int d = 0; d < NUM_BOUND_CONV2D; ++d) {
        LoopBoundVariables tmpSingleBoundVars;

        // Iterate over all loop levels n
        for (int n = 0; n < NUM_LOOP_LEVEL; n++) {
            // Create variables for all divisors
            std::vector<GRBVar> tmpDivsorVars;

            // Create tmp LinExpr to represent the final loop bound integer value
            GRBLinExpr tmpLoopBoundIntValue;

            // Iterate overall divisors k
            for (int k = 0; k < selOpConstInfo.loopBoundInfoMap[d].divisorsVec.size(); k++) {
                // Construct the variable name
                std::string tmpVarName = "b_" + std::to_string(d) + "_" +
                                            std::to_string(n) + "_" + std::to_string(k) +
                                            "_O_" + std::to_string(selOpIdx);

                // Create and store the binary variable --> b_d_n_k
                GRBVar tmpBinaryVar = createBinaryVar(tmpVarName, GRB_BINARY);
                tmpDivsorVars.push_back(tmpBinaryVar);

                // Updatae the linear expression
                tmpLoopBoundIntValue += tmpBinaryVar * selOpConstInfo.loopBoundInfoMap[d].divisorsVec[k];
            }

            // Create Integer variable --> L_d_n
            std::string tmpIntegerVarName = "L_" + std::to_string(d) + "_" +
                                                std::to_string(n) + "_O_" + std::to_string(selOpIdx);
            int64_t tmpUpperBound = selOpConstInfo.loopBoundInfoMap[d].divisorsVec.back();
            GRBVar tmpLoopBoundIntVar = model.addVar(1, tmpUpperBound, 0.0, GRB_INTEGER, tmpIntegerVarName);

            // Add constraints for the created integer variable
            // It must eqal the linear expression composed by all binary variables
            model.addConstr(tmpLoopBoundIntVar == tmpLoopBoundIntValue, "Integer_Loop_Bound_Var_Cons_" + tmpIntegerVarName);

            // Store the integer variable
            tmpSingleBoundVars.loopBoundIntVars.push_back(tmpLoopBoundIntVar);
            
            // Store binary variables
            tmpSingleBoundVars.loopBoundBinaryVars.push_back(tmpDivsorVars);
        }

        // Store the loop bound variables
        opVariablesMap[selOpIdx].loopBoundsVars.push_back(tmpSingleBoundVars);
    }

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addTransCoeffVarsConv2D(const int32_t& selOpIdx) {
    // Create a Gurobi binary variable of the given name and type for all loop bound in resulting nested loop
    auto createBinaryVar = [&](const std::string &name, char type) {
        return model.addVar(0, 1, 0.0, type, name);
    };

    // For conv2D workload, we need to create transfomration coefficient variables
    // for 4 loop variables, let's first create the binary variables
    for (int32_t m = 0; m < 4; m++) {
        TransCoeffVariables tmpTransVars;

        // Step 1: Create the temporary corresponding integer variables to indicate the actual value
        // We know that for each related loop variable, we just need to consider the coefficient
        // at level 0, thus we need one extra GRB variable for each loop variable to indicate the 
        // multiplication of the corresponding loop bounds at level 1 and level 2 : L_m_1 * L_m_2
        int32_t tmpLoopVarIdx = transToLoopBoundMap[m];
        int32_t tmpMaxVarValue = opConstInfoMap[selOpIdx].loopBoundInfoMap[tmpLoopVarIdx].loopBoundValue;
        
        // Step 1.1: Create the tmp variable to represent the multiplication
        GRBVar tmpLevelOneLoopBoundVar = opVariablesMap[selOpIdx].loopBoundsVars[tmpLoopVarIdx].loopBoundIntVars[1];
        GRBVar tmpLevelTwoLoopBoundVar = opVariablesMap[selOpIdx].loopBoundsVars[tmpLoopVarIdx].loopBoundIntVars[2];

        GRBVar tmpLoopBoundMultVar = model.addVar(1, tmpMaxVarValue, 0.0, GRB_INTEGER, "Coeff_" + std::to_string(m) + "_0_0_O_" + std::to_string(selOpIdx));
        model.addQConstr(tmpLoopBoundMultVar == tmpLevelOneLoopBoundVar * tmpLevelTwoLoopBoundVar, "Coeff_Comb_0_Var_" + std::to_string(m) + "_O_" + std::to_string(selOpIdx) + "_product_constraint");
        
        // Step 2: Create the needed binary variables and the final 
        // delta_m_f_O_i, combination index f for loop variable m in Op i 
        std::vector<GRBVar> tmpCoeffBinaryVars;
        std::vector<GRBVar> tmpCoeffIntVars;
        GRBLinExpr tmpCoeffIntValue;

        for (int32_t f = 0; f < 6; f++) {
            std::string tmpVarName = "delta_" + std::to_string(m) + "_" +
                                            std::to_string(f) + 
                                            "_O_" + std::to_string(selOpIdx);

            // Create and store the binary variable --> b_d_n_k
            GRBVar tmpBinaryVar = createBinaryVar(tmpVarName, GRB_BINARY);
            tmpCoeffBinaryVars.push_back(tmpBinaryVar);
        }

        // Step 2.1 Create the temporary variable to represent the selection process for each combination
        // TODO: This can be simplified to 4 combinations, if we only care about the coefficient at level 0
        // Combination 0 : f = a * b
        GRBVar tmpCoeffComb0Var = model.addVar(0, tmpMaxVarValue, 0.0, GRB_INTEGER, "Coeff_" + std::to_string(m) + "_Comb_0_O_" + std::to_string(selOpIdx));
        model.addQConstr(tmpCoeffComb0Var == tmpLoopBoundMultVar * tmpCoeffBinaryVars[0], "Coeff_" + std::to_string(m) + "_Comb_0_O_" + std::to_string(selOpIdx) + "_Constraint");
        tmpCoeffIntValue += tmpCoeffComb0Var;

        // Combination 1 : f = a
        GRBVar tmpCoeffComb1Var = model.addVar(0, tmpMaxVarValue, 0.0, GRB_INTEGER, "Coeff_" + std::to_string(m) + "_Comb_1_O_" + std::to_string(selOpIdx));
        model.addQConstr(tmpCoeffComb1Var == tmpLevelTwoLoopBoundVar * tmpCoeffBinaryVars[1], "Coeff_" + std::to_string(m) + "_Comb_1_O_" + std::to_string(selOpIdx) + "_Constraint");
        tmpCoeffIntValue += tmpCoeffComb1Var;

        // Combination 2 : f = 1
        tmpCoeffIntValue += tmpCoeffBinaryVars[2];

        // Combination 3 : f = 1
        tmpCoeffIntValue += tmpCoeffBinaryVars[3];

        // Combination 4 : f = a * b
        GRBVar tmpCoeffComb4Var = model.addVar(0, tmpMaxVarValue, 0.0, GRB_INTEGER, "Coeff_" + std::to_string(m) + "_Comb_4_O_" + std::to_string(selOpIdx));
        model.addQConstr(tmpCoeffComb4Var == tmpLoopBoundMultVar * tmpCoeffBinaryVars[4], "Coeff_" + std::to_string(m) + "_Comb_4_O_" + std::to_string(selOpIdx) + "_Constraint");
        tmpCoeffIntValue += tmpCoeffComb4Var;

        // Combination 5 : f = b
        GRBVar tmpCoeffComb5Var = model.addVar(0, tmpMaxVarValue, 0.0, GRB_INTEGER, "Coeff_" + std::to_string(m) + "_Comb_5_O_" + std::to_string(selOpIdx));
        model.addQConstr(tmpCoeffComb5Var == tmpLevelOneLoopBoundVar * tmpCoeffBinaryVars[5], "Coeff_" + std::to_string(m) + "_Comb_5_O_" + std::to_string(selOpIdx) + "_Constraint");
        tmpCoeffIntValue += tmpCoeffComb5Var;

        // Final Coeff variable --> C_m_n
        GRBVar tmpCoeffIntVar = model.addVar(0, tmpMaxVarValue, 0.0, GRB_INTEGER, "C_" + std::to_string(m) + "_0_O_" + std::to_string(selOpIdx));
        // Select the final transformation coefficient
        if (selTransCoeffMethod == transCoeffMethodIdx::Comb0) {
            model.addConstr(tmpCoeffIntVar == tmpLoopBoundMultVar, "C_" + std::to_string(m) + "_0_O_" + std::to_string(selOpIdx) + "_Constraint");
        } else if (selTransCoeffMethod == transCoeffMethodIdx::Comb1) {
            model.addConstr(tmpCoeffIntVar == tmpLevelTwoLoopBoundVar, "C_" + std::to_string(m) + "_0_O_" + std::to_string(selOpIdx) + "_Constraint");
        } else if (selTransCoeffMethod == transCoeffMethodIdx::Comb2) {
            model.addConstr(tmpCoeffIntVar == 1, "C_" + std::to_string(m) + "_0_O_" + std::to_string(selOpIdx) + "_Constraint");
        } else if (selTransCoeffMethod == transCoeffMethodIdx::Comb5) {
            model.addConstr(tmpCoeffIntVar == tmpLevelOneLoopBoundVar, "C_" + std::to_string(m) + "_0_O_" + std::to_string(selOpIdx) + "_Constraint");
        } else if (selTransCoeffMethod == transCoeffMethodIdx::Flexible) {
            model.addConstr(tmpCoeffIntVar == tmpCoeffIntValue, "C_" + std::to_string(m) + "_0_O_" + std::to_string(selOpIdx) + "_Constraint");
        }
        tmpCoeffIntVars.push_back(tmpCoeffIntVar);

        // Store the binary variabls
        tmpTransVars.coeffBinaryVars = tmpCoeffBinaryVars;
        tmpTransVars.coeffIntVars = tmpCoeffIntVars;
        opVariablesMap[selOpIdx].loopCoeffsVars.push_back(tmpTransVars);
    }

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumMulColVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumMulColVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "num_Mul_Col_Var");

    // To calculate the number of multiplications in a col
    // We need to multiply all the loop bounds at level 0
    
    // Multiplication 1
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::N].loopBoundIntVars[loopLevelIdx::LEVEL0];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::K].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::N].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::K].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] tmpNum_Mul_Col_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "] tmpNum_Mul_Col_Var_0_constraint");

    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::P].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue1 = tmpMaxValue0 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::P].loopBoundValue;
    GRBVar tmpMul1 = model.addVar(1, tmpMaxValue1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] tmpNum_Mul_Col_Var_1");
    model.addQConstr(tmpMul1 == tmpMul0 * tmpLoopBound2, "[Op_" + std::to_string(selOpIdx) + "] tmpNum_Mul_Col_Var_1_constraint");

    // Multiplication 3
    GRBVar tmpLoopBound3 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::Q].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue2 = tmpMaxValue1 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::Q].loopBoundValue;
    GRBVar tmpMul2 = model.addVar(1, tmpMaxValue2, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] tmpNum_Mul_Col_Var_2");
    model.addQConstr(tmpMul2 == tmpMul1 * tmpLoopBound3, "[Op_" + std::to_string(selOpIdx) + "] tmpNum_Mul_Col_Var_2_constraint");

    // Multiplication 4
    GRBVar tmpLoopBound4 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::C].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue3 = tmpMaxValue2 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::C].loopBoundValue;
    GRBVar tmpMul3 = model.addVar(1, tmpMaxValue3, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] tmpNum_Mul_Col_Var_3");
    model.addQConstr(tmpMul3 == tmpMul2 * tmpLoopBound4, "[Op_" + std::to_string(selOpIdx) + "] tmpNum_Mul_Col_Var_3_constraint");

    // Multiplication 5
    GRBVar tmpLoopBound5 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::R].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue4 = tmpMaxValue3 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::R].loopBoundValue;
    GRBVar tmpMul4 = model.addVar(1, tmpMaxValue4, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] tmpNum_Mul_Col_Var_4");
    model.addQConstr(tmpMul4 == tmpMul3 * tmpLoopBound5, "[Op_" + std::to_string(selOpIdx) + "] tmpNum_Mul_Col_Var_4_constraint");

    // Multiplication 6
    GRBVar tmpLoopBound6 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::S].loopBoundIntVars[loopLevelIdx::LEVEL0];
    model.addQConstr(finalNumMulColVar == tmpMul4 * tmpLoopBound6, "[Op_" + std::to_string(selOpIdx) + "] num_Mul_Col_Var_cons");

    // Store the variable
    tmpOpVars.numMulCol = finalNumMulColVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumColPEVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumColPEVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "num_Col_PE_Var");

    // To calculate the number of multiplications in a col
    // We need to multiply all the loop bounds at level 1
    
    // Multiplication 1
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::N].loopBoundIntVars[loopLevelIdx::LEVEL1];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::K].loopBoundIntVars[loopLevelIdx::LEVEL1];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::N].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::K].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_Col_PE_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "] num_Col_PE_Var_0_constraint");

    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::P].loopBoundIntVars[loopLevelIdx::LEVEL1];
    int64_t tmpMaxValue1 = tmpMaxValue0 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::P].loopBoundValue;
    GRBVar tmpMul1 = model.addVar(1, tmpMaxValue1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_Col_PE_Var_1");
    model.addQConstr(tmpMul1 == tmpMul0 * tmpLoopBound2, "[Op_" + std::to_string(selOpIdx) + "] num_Col_PE_Var_1_constraint");

    // Multiplication 3
    GRBVar tmpLoopBound3 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::Q].loopBoundIntVars[loopLevelIdx::LEVEL1];
    int64_t tmpMaxValue2 = tmpMaxValue1 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::Q].loopBoundValue;
    GRBVar tmpMul2 = model.addVar(1, tmpMaxValue2, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_Col_PE_Var_2");
    model.addQConstr(tmpMul2 == tmpMul1 * tmpLoopBound3, "[Op_" + std::to_string(selOpIdx) + "] num_Col_PE_Var_2_constraint");

    // Multiplication 4
    GRBVar tmpLoopBound4 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::C].loopBoundIntVars[loopLevelIdx::LEVEL1];
    int64_t tmpMaxValue3 = tmpMaxValue2 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::C].loopBoundValue;
    GRBVar tmpMul3 = model.addVar(1, tmpMaxValue3, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_Col_PE_Var_3");
    model.addQConstr(tmpMul3 == tmpMul2 * tmpLoopBound4, "[Op_" + std::to_string(selOpIdx) + "] num_Col_PE_Var_3_constraint");

    // Multiplication 5
    GRBVar tmpLoopBound5 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::R].loopBoundIntVars[loopLevelIdx::LEVEL1];
    int64_t tmpMaxValue4 = tmpMaxValue3 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::R].loopBoundValue;
    GRBVar tmpMul4 = model.addVar(1, tmpMaxValue4, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_Col_PE_Var_4");
    model.addQConstr(tmpMul4 == tmpMul3 * tmpLoopBound5, "[Op_" + std::to_string(selOpIdx) + "] num_Col_PE_Var_4_constraint");

    // Multiplication 6
    GRBVar tmpLoopBound6 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::S].loopBoundIntVars[loopLevelIdx::LEVEL1];
    model.addQConstr(finalNumColPEVar == tmpMul4 * tmpLoopBound6, "[Op_" + std::to_string(selOpIdx) + "] num_Col_PE_Var_cons");

    // Store the variable
    tmpOpVars.numColPE = finalNumColPEVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumPESysVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumPESysVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "num_PE_Sys_Var");

    // To calculate the number of multiplications in a col
    // We need to multiply all the loop bounds at level 2
    
    // Multiplication 1
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::N].loopBoundIntVars[loopLevelIdx::LEVEL2];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::K].loopBoundIntVars[loopLevelIdx::LEVEL2];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::N].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::K].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_PE_Sys_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "] num_PE_Sys_Var_0_constraint");

    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::P].loopBoundIntVars[loopLevelIdx::LEVEL2];
    int64_t tmpMaxValue1 = tmpMaxValue0 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::P].loopBoundValue;
    GRBVar tmpMul1 = model.addVar(1, tmpMaxValue1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_PE_Sys_Var_1");
    model.addQConstr(tmpMul1 == tmpMul0 * tmpLoopBound2, "[Op_" + std::to_string(selOpIdx) + "] num_PE_Sys_Var_1_constraint");

    // Multiplication 3
    GRBVar tmpLoopBound3 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::Q].loopBoundIntVars[loopLevelIdx::LEVEL2];
    int64_t tmpMaxValue2 = tmpMaxValue1 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::Q].loopBoundValue;
    GRBVar tmpMul2 = model.addVar(1, tmpMaxValue2, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_PE_Sys_Var_2");
    model.addQConstr(tmpMul2 == tmpMul1 * tmpLoopBound3, "[Op_" + std::to_string(selOpIdx) + "] num_PE_Sys_Var_2_constraint");

    // Multiplication 4
    GRBVar tmpLoopBound4 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::C].loopBoundIntVars[loopLevelIdx::LEVEL2];
    int64_t tmpMaxValue3 = tmpMaxValue2 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::C].loopBoundValue;
    GRBVar tmpMul3 = model.addVar(1, tmpMaxValue3, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_PE_Sys_Var_3");
    model.addQConstr(tmpMul3 == tmpMul2 * tmpLoopBound4, "[Op_" + std::to_string(selOpIdx) + "] num_PE_Sys_Var_3_constraint");

    // Multiplication 5
    GRBVar tmpLoopBound5 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::R].loopBoundIntVars[loopLevelIdx::LEVEL2];
    int64_t tmpMaxValue4 = tmpMaxValue3 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::R].loopBoundValue;
    GRBVar tmpMul4 = model.addVar(1, tmpMaxValue4, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_PE_Sys_Var_4");
    model.addQConstr(tmpMul4 == tmpMul3 * tmpLoopBound5, "[Op_" + std::to_string(selOpIdx) + "] num_PE_Sys_Var_4_constraint");

    // Multiplication 6
    GRBVar tmpLoopBound6 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::S].loopBoundIntVars[loopLevelIdx::LEVEL2];
    model.addQConstr(finalNumPESysVar == tmpMul4 * tmpLoopBound6, "[Op_" + std::to_string(selOpIdx) + "] num_PE_Sys_Var_cons");

    // Store the variable
    tmpOpVars.numPESys = finalNumPESysVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumOutColVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = 1;

    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::N].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::P].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::Q].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::K].loopBoundValue;

    // Create the final variable
    GRBVar finalNumOutColVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "num_Out_Col_Var");


    // The number of outputs in a column can be calculated as
    // NumOutCol = N * P * Q * K at loop level 0
    // Multiplication 1
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::N].loopBoundIntVars[loopLevelIdx::LEVEL0];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::P].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::N].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::P].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_Out_Col_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "] num_Out_Col_Var_0_constraint");

    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::Q].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue1 = tmpMaxValue0 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::Q].loopBoundValue;
    GRBVar tmpMul1 = model.addVar(1, tmpMaxValue1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_Out_Col_Var_1");
    model.addQConstr(tmpMul1 == tmpMul0 * tmpLoopBound2, "[Op_" + std::to_string(selOpIdx) + "] num_Out_Col_Var_1_constraint");

    // Multiplication 3
    GRBVar tmpLoopBound3 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::K].loopBoundIntVars[loopLevelIdx::LEVEL0];
    model.addQConstr(finalNumOutColVar == tmpMul1 * tmpLoopBound3, "[Op_" + std::to_string(selOpIdx) + "] num_Out_Col_Var_constraint");

    // Store the variable
    tmpOpVars.numOutCol = finalNumOutColVar;


    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumFilColVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = 1;

    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::K].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::R].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::S].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::C].loopBoundValue;

    // Create the final variable
    GRBVar finalNumFilColVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "num_Fil_Col_Var");

    // The number of outputs in a column can be calculated as
    // NumOutCol = K * R * S * C at loop level 0
    // Multiplication 1
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::K].loopBoundIntVars[loopLevelIdx::LEVEL0];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::R].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::K].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::R].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_Fil_Col_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "] num_Fil_Col_Var_0_constraint");

    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::S].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue1 = tmpMaxValue0 * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::S].loopBoundValue;
    GRBVar tmpMul1 = model.addVar(1, tmpMaxValue1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_Fil_Col_Var_1");
    model.addQConstr(tmpMul1 == tmpMul0 * tmpLoopBound2, "[Op_" + std::to_string(selOpIdx) + "] num_Fil_Col_Var_1_constraint");

    // Multiplication 3
    GRBVar tmpLoopBound3 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::C].loopBoundIntVars[loopLevelIdx::LEVEL0];
    model.addQConstr(finalNumFilColVar == tmpMul1 * tmpLoopBound3, "[Op_" + std::to_string(selOpIdx) + "] num_Fil_Col_Var_constraint");

    // Store the variable
    tmpOpVars.numFilCol = finalNumFilColVar;


    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumRedOutColVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // The number of reduction per output in a column can be calculated as
    // numRedOutCol = R * S * C - 1; at loop level 0

    // Maximum value bound
    int64_t maxValue = 1;

    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::R].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::S].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::C].loopBoundValue;

    // Create the final variable
    GRBVar finalNumRedOutColVar = model.addVar(0, maxValue, 0.0, GRB_INTEGER, "num_red_out_col_Var");

    // Multiplication 1
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::R].loopBoundIntVars[loopLevelIdx::LEVEL0];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::S].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::R].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::S].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_red_out_col_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "] num_red_out_col_Var_0_constraint");

    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::C].loopBoundIntVars[loopLevelIdx::LEVEL0];
    model.addQConstr(finalNumRedOutColVar == tmpMul0 * tmpLoopBound2 - 1, "[Op_" + std::to_string(selOpIdx) + "] num_red_out_col_Var_constraint");

    // Store the variable
    tmpOpVars.numRedOutCol = finalNumRedOutColVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumInColVarConv2D(const int32_t& selOpIdx) {
    
    // Create a Gurobi binary variable of the given name and type for all loop bound in resulting nested loop
    auto createBinaryVar = [&](const std::string &name, char type) {
        return model.addVar(0, 1, 0.0, type, name);
    };

    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Create the final numInColVar
    int64_t maxValue = 1;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::N].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::P].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::R].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::Q].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::S].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::C].loopBoundValue;

    GRBVar finalNumInColVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "num_in_col_Var");

    //
    // Step 1: Add all needed binary variables to represent the selected gcd pair
    //

    // We need to account for both the PR_TABLE and QS_TABLE
    for (int i = 0; i < 2; i++) {

        // Construct the lhs of the constraint
        GRBLinExpr singleSelectionCons;
        GRBLinExpr coeffOneEqualCons;
        GRBLinExpr coeffTwoEqualCons;
        GRBLinExpr finalGCDValue;

        // Tmp Variable List
        std::vector<GRBVar> tmpBinarVars;

        // Iterate over all possible gcd pairs for a single LUT
        for (int j = 0; j < selOpInfo.inputCoeffLUTs[i].size(); j++) {
            // Construct the variable name
            std::string tmpVarName = inputArrayLUTNameConv2D[i] + "_Sel_" +
                                        std::to_string(j) + "_O_" + std::to_string(selOpIdx);

            // Create the variable
            GRBVar tmpBinaryVar = createBinaryVar(tmpVarName, GRB_BINARY);
            tmpBinarVars.push_back(tmpBinaryVar);

            // Update the corresponding constraints
            // Cons 1: Only one binary variable can be selected
            singleSelectionCons += tmpBinaryVar;

            // Cons 2: Coeff 1 must equal to the acutal coefficient in the transformed loop
            coeffOneEqualCons += tmpBinaryVar * selOpInfo.inputCoeffCatSets[i][j][0];

            // Cons 3: Coeff 2 must equal to the acutal coefficient in the transformed loop
            coeffTwoEqualCons += tmpBinaryVar * selOpInfo.inputCoeffCatSets[i][j][1];

            // Update the final gcd value
            double_t tmpInverseGCDValue = 1.0 / static_cast<double_t>(selOpInfo.inputCoeffLUTs[i][j]);
            finalGCDValue += tmpBinaryVar * tmpInverseGCDValue;
        }

        // Store the tmp binary variable
        tmpOpVars.LUTSelVars.push_back(tmpBinarVars);

        // Create the final constraints
        // Cons 1: Only one binary variable can be selected
        model.addConstr(singleSelectionCons == 1, inputArrayLUTNameConv2D[i] + "_single_sel_constraint");

        if (i == inputLutIdxConv2D::QSTable) {
            // Cons 2: Coeff 1 must equal to the acutal coefficient in the transformed loop
            model.addConstr(coeffOneEqualCons == tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_Q].coeffIntVars[0], inputArrayLUTNameConv2D[i] + "_Coeff_One_constraint");

            // Cons 3: Coeff 2 must equal to the acutal coefficient in the transformed loop
            model.addConstr(coeffTwoEqualCons == tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_S].coeffIntVars[0], inputArrayLUTNameConv2D[i] + "_Coeff_Two_constraint");
        
            // Update the tmp GCD value
            GRBVar tmpQSGCDValue = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "QS_Sel_GCD_Value");
            model.addConstr(tmpQSGCDValue == finalGCDValue);

            // Store the variable
            tmpOpVars.QSGCDValue = tmpQSGCDValue;
        } else if (i == inputLutIdxConv2D::PRTable) {
            // Cons 2: Coeff 1 must equal to the acutal coefficient in the transformed loop
            model.addConstr(coeffOneEqualCons == tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_P].coeffIntVars[0], inputArrayLUTNameConv2D[i] + "_Coeff_One_constraint");

            // Cons 3: Coeff 2 must equal to the acutal coefficient in the transformed loop
            model.addConstr(coeffTwoEqualCons == tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_R].coeffIntVars[0], inputArrayLUTNameConv2D[i] + "_Coeff_Two_constraint");
        
            // Update the tmp GCD value
            GRBVar tmpPRGCDValue = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "PR_Sel_GCD_Value");
            model.addConstr(tmpPRGCDValue == finalGCDValue);

            // Store the variable
            tmpOpVars.PRGCDValue = tmpPRGCDValue;
        }
    }

    //
    //  Calculate the actual number of input elements in a column
    //      Input indexing: [N][P * Wstride + R * Wdilation][Q * Hstride + S * Hdilation][C] at level 0
    //

    // Get the second term in the input indexing fucntion
    // P * Wstride + R * Wdilation

    // Calculate P * R
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::P].loopBoundIntVars[loopLevelIdx::LEVEL0];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::R].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::P].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::R].loopBoundValue;
    GRBVar tmpTerm2Second = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_input_Col_Term2_Term2");
    model.addQConstr(tmpTerm2Second == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "] num_input_Col_Term2_Term2_constraint");

    // Calculate Coeff_P * (P - 1))
    GRBVar tmpPCoeff = tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_P].coeffIntVars[0];
    GRBVar tmpMaxPCValue = model.addVar(0, tmpMaxValue0 * 4, 0.0, GRB_INTEGER, "tmp_maxP_C_value_var");
    model.addQConstr(tmpMaxPCValue == tmpPCoeff * (tmpLoopBound0 - 1));

    // Calculate Coeff_R * (R - 1))
    GRBVar tmpRCoeff = tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_R].coeffIntVars[0];
    GRBVar tmpMaxRCValue = model.addVar(0, tmpMaxValue0 * 3, 0.0, GRB_INTEGER, "tmp_maxR_C_value_var");
    model.addQConstr(tmpMaxRCValue == tmpRCoeff * (tmpLoopBound1 - 1));

    //
    GRBVar tmpTerm2OnePartial = model.addVar(1, tmpMaxValue0 * 2, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_input_Col_Term2_Term1_partial");
    model.addConstr(tmpTerm2OnePartial == tmpMaxRCValue + tmpMaxPCValue);
    GRBVar tmpTerm2One = model.addVar(1, tmpMaxValue0 * 2, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_input_Col_Term2_Term1");
    model.addQConstr(tmpTerm2One == ((tmpTerm2OnePartial * tmpOpVars.PRGCDValue) + 1));

    // Final Term 2
    // Create the needed GRB Vector
    GRBVar finalTerm2 = model.addVar(1, tmpMaxValue0 + 1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_inputs_col_term2");
    GRBVar term2Vars[] = {tmpTerm2One, tmpTerm2Second};

    // Select the estiamtion method
    if (selNumCountingMethod == 1) {
        // COSA's method
        model.addQConstr(finalTerm2 == tmpTerm2Second, "num_inputs_col_constraint");
    } else if (selNumCountingMethod == 0) {
        model.addGenConstrMin(finalTerm2, term2Vars, 2);
    }

    // Get the third term in the input indexing fucntion
    // Q * Hstride + S * Hdilation

    // Calculate P * R
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::Q].loopBoundIntVars[loopLevelIdx::LEVEL0];
    GRBVar tmpLoopBound3 = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::S].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue1 = selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::Q].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxConv2D::S].loopBoundValue;
    GRBVar tmpTerm3Second = model.addVar(1, tmpMaxValue1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_input_Col_Term3_Term2");
    model.addQConstr(tmpTerm3Second == tmpLoopBound2 * tmpLoopBound3, "[Op_" + std::to_string(selOpIdx) + "] num_input_Col_Term3_Term2_constraint");

    // Calculate Coeff_Q * (Q - 1))
    GRBVar tmpQCoeff = tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_Q].coeffIntVars[0];
    GRBVar tmpMaxQCValue = model.addVar(0, tmpMaxValue1 + 1, 0.0, GRB_INTEGER, "tmp_maxQ_C_value_var");
    model.addQConstr(tmpMaxQCValue == tmpQCoeff * (tmpLoopBound2 - 1));

    // Calculate Coeff_R * (R - 1))
    GRBVar tmpSCoeff = tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_S].coeffIntVars[0];
    GRBVar tmpMaxSCValue = model.addVar(0, tmpMaxValue1 + 1, 0.0, GRB_INTEGER, "tmp_maxS_C_value_var");
    model.addQConstr(tmpMaxSCValue == tmpSCoeff * (tmpLoopBound3 - 1));

    //
    GRBVar tmpTerm3OnePartial = model.addVar(1, tmpMaxValue1 * 2, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_input_Col_Term3_Term1_partial");
    model.addConstr(tmpTerm3OnePartial == tmpMaxQCValue + tmpMaxSCValue);
    GRBVar tmpTerm3One = model.addVar(1, tmpMaxValue1 + 1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_input_Col_Term3_Term1");
    model.addQConstr(tmpTerm3One == ((tmpTerm3OnePartial * tmpOpVars.QSGCDValue) + 1));

    // Final Term 2
    // Create the needed GRB Vector
    GRBVar finalTerm3 = model.addVar(1, tmpMaxValue1 + 1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_inputs_col_term3");
    GRBVar term3Vars[] = {tmpTerm3One, tmpTerm3Second};

    // Select the estiamtion method
    if (selNumCountingMethod == 1) {
        model.addQConstr(finalTerm3 == tmpTerm3Second, "num_inputs_col_constraint");
    } else if (selNumCountingMethod == 0) {
        // COSA's method
        model.addGenConstrMin(finalTerm3, term3Vars, 2);
    }

    // Calculate multiplication of N * term 2
    GRBVar tmpFinalMul0 = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_inputs_col_tmp_mul0");
    GRBVar tmpTerm1Var = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::N].loopBoundIntVars[loopLevelIdx::LEVEL0];
    model.addQConstr(tmpFinalMul0 == tmpTerm1Var * finalTerm2);

    // Calculate multiplication of tmpFinalMul0 * term 3
    GRBVar tmpFinalMul1 = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "] num_inputs_col_tmp_mul1");
    model.addQConstr(tmpFinalMul1 == tmpFinalMul0 * finalTerm3);

    // Calculate multiplication of tmpFinalMul1 * C
    GRBVar tmpTerm4Var = tmpOpVars.loopBoundsVars[loopBoundIdxConv2D::C].loopBoundIntVars[loopLevelIdx::LEVEL0];
    
    // Select the estimation method
    model.addQConstr(finalNumInColVar == tmpFinalMul1 * tmpTerm4Var, "num_inputs_col_constraint");

    // Store the constructed variable
    tmpOpVars.numInCol = finalNumInColVar;
    tmpOpVars.numInTerm2Col = finalTerm2;
    tmpOpVars.numInTerm3Col = finalTerm3;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumRedColVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumRedColVar = model.addVar(0, maxValue, 0.0, GRB_INTEGER, "num_Red_Col_Var");

    // Get the desired variables
    GRBVar numRedOut = tmpOpVars.numRedOutCol;
    GRBVar numOutCol = tmpOpVars.numOutCol;

    // Get the desired variable
    model.addQConstr(finalNumRedColVar == numRedOut * numOutCol, "num_Red_Col_Var_constraint");

    // Store the variable
    tmpOpVars.numRedCol = finalNumRedColVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumOutPEVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumOutPEVar = model.addVar(1, maxValue * maxValue, 0.0, GRB_INTEGER, "num_Out_PE_Var");

    // Get the desired variables
    GRBVar numColPE = tmpOpVars.numColPE;
    GRBVar numOutCol = tmpOpVars.numOutCol;

    // Get the desired variable
    model.addQConstr(finalNumOutPEVar == numColPE * numOutCol, "num_Out_PE_Var_constraint");

    // Store the variable
    tmpOpVars.numOutPE = finalNumOutPEVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumOutSysVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumOutSysVar = model.addVar(0, maxValue * maxValue, 0.0, GRB_CONTINUOUS, "num_Out_Sys_Var");

    // Get the desired variables
    GRBVar numPESys = tmpOpVars.numPESys;
    GRBVar numOutPE = tmpOpVars.numOutPE;
    GRBVar numOutCol = tmpOpVars.numOutCol;

    // Get the bandwidth
    double_t tmpBandWidth = 1.0 / static_cast<double_t>(archInfo.SysBandWidth);
    double_t tmpConstant = tmpBandWidth * archInfo.dataWidth;

    // Get the final variable
    // Check the device type
    if (selDeviceType == deviceTypeIdx::PUM) {
        model.addQConstr(finalNumOutSysVar == numPESys * numOutPE * tmpConstant, "num_Out_Sys_Var_constraint");
    } else if (selDeviceType == deviceTypeIdx::PNM) {
        model.addQConstr(finalNumOutSysVar == numPESys * numOutCol * tmpConstant, "num_Out_Sys_Var_constraint");
    }
    

    // Store the variable
    tmpOpVars.numOutSys = finalNumOutSysVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumInPEVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumInPEVar = model.addVar(1, maxValue * maxValue, 0.0, GRB_INTEGER, "num_In_PE_Var");

    // Get the desired variables
    GRBVar numColPE = tmpOpVars.numColPE;
    GRBVar numInCol = tmpOpVars.numInCol;

    // Get the desired variable
    model.addQConstr(finalNumInPEVar == numColPE * numInCol, "num_In_PE_Var_constraint");

    // Store the variable
    tmpOpVars.numInPE = finalNumInPEVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumInSysVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumInSysVar = model.addVar(0, maxValue * maxValue, 0.0, GRB_CONTINUOUS, "num_In_Sys_Var");

    // Get the desired variables
    GRBVar numPESys = tmpOpVars.numPESys;
    GRBVar numInPE = tmpOpVars.numInPE;

    // Get the bandwidth
    double_t tmpBandWidth = 1.0 / static_cast<double_t>(archInfo.SysBandWidth);
    double_t tmpConstant = tmpBandWidth * archInfo.dataWidth;

    // Get the final variable
    model.addQConstr(finalNumInSysVar == numPESys * numInPE * tmpConstant, "num_In_Sys_Var_constraint");

    // Store the variable
    tmpOpVars.numInSys = finalNumInSysVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addFilPELoadingVarConv2D(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalFilPELoadingVar = model.addVar(0, maxValue * maxValue, 0.0, GRB_CONTINUOUS, "num_Fil_Loading_Var");

    // Get the desired variables
    GRBVar numColPE = tmpOpVars.numColPE;
    GRBVar numFilCol = tmpOpVars.numFilCol;

    // Get the bandwidth
    double_t tmpBandWidth = 1.0 / static_cast<double_t>(archInfo.PEBandWidth);
    double_t tmpConstant = tmpBandWidth * archInfo.dataWidth;

    // Get the final variable
    model.addQConstr(finalFilPELoadingVar == numColPE * numFilCol * tmpConstant, "num_Fil_Loading_Var_constraint");

    // Store the variable
    tmpOpVars.numFilPELoading = finalFilPELoadingVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

// =============================================
//      GRB Constraints Construction -- Conv2D
// =============================================

void DataLayoutMILP::addLoopBoundConsConv2D(const int32_t& selLayer) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selLayer];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selLayer];

    // Iterate over all loop bounds d
    for (int d = 0; d < NUM_BOUND_CONV2D; d++) {

        // Get loop bound variables
        LoopBoundVariables& tmpLoopBoundVars = tmpOpVars.loopBoundsVars[d];

        // Constraint for the mulitplication of transfromded loop bound at different levels
        // a * b * c = N
        GRBLinExpr tmpLoopBoundIntValueConstraint;

        // Iterate over all loop levels
        for (int n = 0; n < NUM_LOOP_LEVEL; n++) {
            // Sum over all divisor binary variables
            GRBLinExpr tmpLoopBoundConstraint;

            // Int Constraint
            GRBLinExpr tmpLoopBoundMulConstraint;

            // Iterate over all divisors k
            for (int k = 0; k < selOpInfo.loopBoundInfoMap[d].divisorsVec.size(); k++) {
                tmpLoopBoundConstraint += tmpLoopBoundVars.loopBoundBinaryVars[n][k];

                // Get the log value of the transformed loop bound
                tmpLoopBoundMulConstraint += selOpInfo.loopBoundInfoMap[d].divisorsLogVec[k]
                                            * tmpLoopBoundVars.loopBoundBinaryVars[n][k];
            }

            tmpLoopBoundIntValueConstraint += tmpLoopBoundMulConstraint;

            // Constraints 1: Only one value can be selected
            std::string tmpConsName = "[Op" + std::to_string(selLayer) + "]_" + 
                                        "LoopBoundBinaryConstraint_" + std::to_string(d) + "_" + std::to_string(n);

            // Add Constraint
            model.addConstr(tmpLoopBoundConstraint == 1, tmpConsName);
        }

        // Construct the constraint name
        std::string tmpConsNameUpper = "[Op" + std::to_string(selLayer) + "]_" + 
                                            "LoopBoundMul_" + std::to_string(d) + "_constraint";

        // Log2 value of the original loop bound
        double_t rhs = std::log2(static_cast<double_t>(selOpInfo.loopBoundInfoMap[d].loopBoundValue));

        model.addConstr(tmpLoopBoundIntValueConstraint == rhs, tmpConsNameUpper);
    }
}

void DataLayoutMILP::addTransCoeffConsConv2D(const int32_t& selOp) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOp];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOp];

    // For Conv2D we just need to iterate over 4 loop variables
    for (int m = 0; m < 4; m++) {
        GRBLinExpr tmpCoeffConstraint;

        // Iterate over all possible combinations
        // TODO: Change the hard coded number of combinations
        for (int f = 0; f < 6; f++) {
            tmpCoeffConstraint += tmpOpVars.loopCoeffsVars[m].coeffBinaryVars[f];
        }

        // Construct the constraint Name
        std::string tmpConsName = "[Op" + std::to_string(selOp) + "]_" +
                                        "TransCoeff_" + std::to_string(m) + "_constraint";

        // Add the constraint
        model.addConstr(tmpCoeffConstraint == 1, tmpConsName);
    }   
}

void DataLayoutMILP::addNumPEConsConv2D(const int32_t& selOp) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOp];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOp];

    // For this constraint we only care about the loop bounds at level 2
    // GRBLinExpr numPEConstraint;
    // for (int d = 0; d < NUM_BOUND_CONV2D; d++) {
    //     // Sum up all diviosrs in each loop bound
    //     GRBLinExpr loopBoundLogValue;

    //     for (int k = 0; k < selOpInfo.loopBoundInfoMap[d].divisorsLogVec.size(); k++) {
    //         double_t tmpConstant = selOpInfo.loopBoundInfoMap[d].divisorsLogVec[k];

    //         loopBoundLogValue += tmpConstant * tmpOpVars.loopBoundsVars[d].loopBoundBinaryVars[loopLevelIdx::LEVEL2][k];
    //     }

    //     numPEConstraint += loopBoundLogValue;
    // }

    // // Construct the constraint name
    // std::string tmpConName = "[Op" + std::to_string(selOp) + "]_" + "NumPEConstraint";
    // double_t rhs = std::log2(static_cast<double_t>(selOpInfo.numPE));

    // // Add the constraint in the model and catch error
    // try {
    //     model.addConstr(numPEConstraint <= rhs, tmpConName);
    // } catch (GRBException& e) {
    //     llvm::errs() << "Gurobi Exception occurred: " << e.getMessage() << "\n";
    // }

    // Use the variable directly
    model.addConstr(tmpOpVars.numPESys <= selOpInfo.numPE, "num_PE_Constraint");
}

void DataLayoutMILP::addNumColConsConv2D(const int32_t& selOp) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOp];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOp];

    // For this constraint we only care about the loop bounds at level 1
    // GRBLinExpr numColConstraint;
    // for (int d = 0; d < NUM_BOUND_CONV2D; d++) {
    //     // Sum up all diviosrs in each loop bound
    //     GRBLinExpr loopBoundLogValue;

    //     for (int k = 0; k < selOpInfo.loopBoundInfoMap[d].divisorsLogVec.size(); k++) {
    //         double_t tmpConstant = selOpInfo.loopBoundInfoMap[d].divisorsLogVec[k];

    //         loopBoundLogValue += tmpConstant * tmpOpVars.loopBoundsVars[d].loopBoundBinaryVars[loopLevelIdx::LEVEL1][k];
    //     }

    //     numColConstraint += loopBoundLogValue;
    // }

    // // Construct the constraint name
    // std::string tmpConName = "[Op" + std::to_string(selOp) + "]_" + "NumColConstraint";
    // double_t rhs = std::log2(static_cast<double_t>(archInfo.numCol));

    // // Add the constraint in the model and catch error
    // try {
    //     model.addConstr(numColConstraint <= rhs, tmpConName);
    // } catch (GRBException& e) {
    //     llvm::errs() << "Gurobi Exception occurred: " << e.getMessage() << "\n";
    // }

    // Use the variable directly
    // Add the constraint based on the selected device
    if (selDeviceType == deviceTypeIdx::PUM) {
        model.addConstr(tmpOpVars.numColPE <= archInfo.numCol, "num_Col_Constraint");
    } else if (selDeviceType == deviceTypeIdx::PNM) {
        // We need to further differentiate between Our cost function and cosa's cost function
        if (selObjMethod == 0) {
            // Our cost function
            model.addConstr(tmpOpVars.numColPE * archInfo.dataWidth <= archInfo.numCol, "num_Col_Constraint");
        } else if (selObjMethod == 1) {
            // COSA's cost function 
            // The parallelization is 256 bits for cosa
            model.addConstr(tmpOpVars.numColPE * archInfo.dataWidth <= 256, "num_Col_Constraint");
        }
        
    }
}

void DataLayoutMILP::addColSizeConsConv2D(const int32_t& selOp) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOp];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOp];

    // Get all desired variables
    GRBVar numOutCol = tmpOpVars.numOutCol;
    GRBVar numFilCol = tmpOpVars.numFilCol;
    GRBVar numInCol = tmpOpVars.numInCol;

    // Use the variable directly
    // Change the constraint based on the configuration
    if (selDeviceType == deviceTypeIdx::PUM) {
        if (selStorageMethod == 0) {
            if (selNumCountingMethod == 1) {
                // COSA's method
                model.addConstr( (numFilCol) * archInfo.dataWidth <= archInfo.numRow / 2, "Col_Size_Constraint_Fil_COSA");
                model.addConstr( (numInCol) * archInfo.dataWidth <= archInfo.numRow / 2, "Col_Size_Constraint_In_COSA");
            } else if (selNumCountingMethod == 0) {
                model.addConstr( (numFilCol + numInCol) * archInfo.dataWidth <= archInfo.numRow, "Col_Size_Constraint");
            }
        } else if (selStorageMethod == 1) {
            model.addConstr( (numFilCol) * archInfo.dataWidth <= archInfo.numRow, "Col_Size_Constraint");
        } else if (selStorageMethod == 2) {
            if (selNumCountingMethod == 1) {
                // COSA's Method
                model.addConstr( (numFilCol) * archInfo.dataWidth <= archInfo.numRow / 3, "Col_Size_Constraint_Fil_COSA");
                model.addConstr( (numInCol) * archInfo.dataWidth <= archInfo.numRow / 3, "Col_Size_Constraint_In_COSA");
                model.addConstr( (numOutCol) * archInfo.dataWidth <= archInfo.numRow / 3, "Col_Size_Constraint_Out_COSA");
            } else if (selNumCountingMethod == 0) {
                // Out Method
                model.addConstr( (numOutCol + numFilCol + numInCol) * archInfo.dataWidth <= archInfo.numRow, "Col_Size_Constraint");
            }
        }
    } else if (selDeviceType == deviceTypeIdx::PNM) {
        if (selStorageMethod == 0) {
            if (selNumCountingMethod == 1) {
                // COSA's method
                // Check whether we are using COSA's cost function or not
                if (selObjMethod == 0) {
                    // Our cost function
                    model.addConstr( (numFilCol) <= archInfo.numRow / 2, "Col_Size_Constraint_Fil_COSA");
                    model.addConstr( (numInCol) <= archInfo.numRow / 2, "Col_Size_Constraint_In_COSA");
                } else if (selObjMethod == 1) {
                    // Calculate the costant for Cosa
                    double_t tmpCOSACons = archInfo.numCol / 256;

                    // COSA's cost function
                    model.addConstr( (numFilCol) <= (archInfo.numRow * tmpCOSACons) / 2, "Col_Size_Constraint_Fil_COSA");
                    model.addConstr( (numInCol) <= (archInfo.numRow * tmpCOSACons) / 2, "Col_Size_Constraint_In_COSA");
                }
                
            } else if (selNumCountingMethod == 0) {
                // Check whether we are using COSA's cost function or not
                if (selObjMethod == 0) {
                    // Our cost function
                    model.addConstr( (numFilCol + numInCol) <= archInfo.numRow, "Col_Size_Constraint");
                } else if (selObjMethod == 1) {
                    // Calculate the costant for Cosa
                    double_t tmpCOSACons = archInfo.numCol / 256;

                    model.addConstr( (numFilCol + numInCol) <= archInfo.numRow * tmpCOSACons, "Col_Size_Constraint");
                }
            }
        } else if (selStorageMethod == 1) {
            model.addConstr( (numFilCol) <= archInfo.numRow, "Col_Size_Constraint");

            //
            llvm::outs() << "[WARNING] YOU SHOULD NOT USE THIS STORAGE METHOD!!! CAN'T JUST STORE THE FILTERS IN A COL!!\n";
            exit(-1);
        } else if (selStorageMethod == 2) {
            if (selNumCountingMethod == 1) {
                // COSA's Method
                if (selObjMethod == 0) {
                    model.addConstr( (numFilCol) <= archInfo.numRow / 3, "Col_Size_Constraint_Fil_COSA");
                    model.addConstr( (numInCol) <= archInfo.numRow / 3, "Col_Size_Constraint_In_COSA");
                    model.addConstr( (numOutCol) <= archInfo.numRow / 3, "Col_Size_Constraint_Out_COSA");
                } else if (selObjMethod == 1) {
                    // COSA's cost function
                    double_t tmpCOSACons = archInfo.numCol / 256;

                    model.addConstr( (numFilCol) <= (archInfo.numRow * tmpCOSACons) / 3, "Col_Size_Constraint_Fil_COSA");
                    model.addConstr( (numInCol) <= (archInfo.numRow * tmpCOSACons) / 3, "Col_Size_Constraint_In_COSA");
                    model.addConstr( (numOutCol) <= (archInfo.numRow * tmpCOSACons) / 3, "Col_Size_Constraint_Out_COSA");
                }
                
            } else if (selNumCountingMethod == 0) {
                // Out Method
                if (selObjMethod == 0) {
                    // Our cost function
                    model.addConstr( (numOutCol + numFilCol + numInCol) <= archInfo.numRow, "Col_Size_Constraint");
                } else if (selObjMethod == 1) {
                    // COSA's Cost function
                    double_t tmpCOSACons = archInfo.numCol / 256;

                    model.addConstr( (numOutCol + numFilCol + numInCol) <= (archInfo.numRow * tmpCOSACons), "Col_Size_Constraint");
                }
            }
        }
    }
}

// =============================================
//          Constant Construction Func
// =============================================
void DataLayoutMILP::buildConv2DOpConstants(const OperatorInfo& selOpIns) {
    // Build the constant storing struct
    OpConstInfo tmpConstInfo;
    tmpConstInfo.opID = selOpIns.opID;
    tmpConstInfo.opType = selOpIns.opType;
    tmpConstInfo.numPE = selOpIns.numPE;

    int64_t tmpMaxLoopBound = 1;

    // Build hte storing struct for all loop bounds' divisor lists
    for (int i = 0; i < NUM_BOUND_CONV2D; i++) {
        LoopBoundInfo tmpLoopBoundInfo;

        tmpLoopBoundInfo.loopBoundValue = selOpIns.loopBoundVec[i];
        tmpLoopBoundInfo.divisorsVec = getAllDivisors(selOpIns.loopBoundVec[i]);

        // Update the maximum loop bound
        tmpMaxLoopBound *= selOpIns.loopBoundVec[i];

        // Get the log2 value of all divisorsVec
        std::vector<double_t> tmpDivisorsLogVec(tmpLoopBoundInfo.divisorsVec.size());

        std::transform(tmpLoopBoundInfo.divisorsVec.begin(), tmpLoopBoundInfo.divisorsVec.end(), tmpDivisorsLogVec.begin(),
                            [](int64_t num) { return std::log2(static_cast<double_t>(num)); });

        tmpLoopBoundInfo.divisorsLogVec = tmpDivisorsLogVec;

        // Store the struct in the constinfo map
        tmpConstInfo.loopBoundInfoMap[i] = tmpLoopBoundInfo;
    }

    tmpConstInfo.maxLoopBound = tmpMaxLoopBound;

    // Build the loopup table for the single operator input array size estimation
    // In Conv2D we have the following indexing pattern for the input array
    // Inputs[N][P * Wstride + R * Wdilation][Q * Hstride + S * Hdilation][C]
    // Thus, we need to build look up tables for the second and third term
    // as they are not fully independent.
    for (int inputIdxFunc = 0; inputIdxFunc < 2; inputIdxFunc++) {
        std::vector<Set<int64_t>> tmpSets;

        // Construct the related sets for the LUT
        if (inputIdxFunc == inputLutIdxConv2D::PRTable) {
            tmpSets.push_back(tmpConstInfo.loopBoundInfoMap[loopBoundIdxConv2D::P].divisorsVec);
            tmpSets.push_back(tmpConstInfo.loopBoundInfoMap[loopBoundIdxConv2D::R].divisorsVec);
        } else if (inputIdxFunc == inputLutIdxConv2D::QSTable) {
            tmpSets.push_back(tmpConstInfo.loopBoundInfoMap[loopBoundIdxConv2D::Q].divisorsVec);
            tmpSets.push_back(tmpConstInfo.loopBoundInfoMap[loopBoundIdxConv2D::S].divisorsVec);
        }

        // Get the IA Coefficient Cartesian Product
        CartesianProduct<int64_t> IACaPr = calCartesian(tmpSets); 

        // Get the actual LUT that stores the GCD value of the transformation coefficients
        std::vector<int64_t> IACoeffGCDValue = calConv2DIALUTValue(IACaPr, selOpIns.stride, selOpIns.dilation);

        // Check the maximum GCD value
        auto tmpMaxGCDValue = std::max_element(IACoeffGCDValue.begin(), IACoeffGCDValue.end());

        if (*tmpMaxGCDValue > 1) tmpConstInfo.needGCDLUTs[inputIdxFunc] = true;
        
        // Store the calculated information
        tmpConstInfo.inputCoeffCatSets.push_back(IACaPr);
        tmpConstInfo.inputCoeffLUTs.push_back(IACoeffGCDValue);
    }

    // [Final] : Store all calcualted constants
    // Construct the opConstInfo map
    opConstInfoMap[selOpIns.opID] = tmpConstInfo;

    //! Testing
    tmpConstInfo.printDetail();
    // printCartesianProduct(tmpConstInfo.inputCoeffCatSets[0]);
    // printCartesianProduct(tmpConstInfo.inputCoeffCatSets[1]);
}

std::vector<int64_t> DataLayoutMILP::getAllDivisors(const int32_t& loopBound) {
    // We assume the input dimension is positive
    std::vector<int64_t> tmpDivisors;

    // Calculate divisors up to the square root of n
    for (int i = 1; i <= std::sqrt(loopBound); i++) {
        if (loopBound % i == 0) {
            tmpDivisors.push_back(i);

            if (i != (loopBound / i)) {
                tmpDivisors.push_back(loopBound / i);
            }
        }
    }

    // Sort the obtained list in ascending order
    std::sort(tmpDivisors.begin(), tmpDivisors.end());

    return tmpDivisors;
}

// Function used to calculate GCD of two integer inputs
int64_t DataLayoutMILP::calGCD(const int64_t& inputOne, const int64_t& inputTwo) {
    // Find Minimum of a and b
    int64_t res = std::min(inputOne, inputTwo);

    int64_t a = inputOne;
    int64_t b = inputTwo;

    while (res > 1) {
        if (a % res == 0 && b % res == 0)
            break;
        res--;
    }
    return res;
}

std::vector<int64_t> DataLayoutMILP::calConv2DIALUTValue(const CartesianProduct<int64_t>& selCaPr,
                                                const std::vector<int64_t>& opStride, const std::vector<int64_t>& opDilation) {
    //! Dimension check is omitted for this version, Jiantao 17/10/2024
    // For now, we assume that stride is the same for width and height
    std::vector<int64_t> result;

    for (size_t i = 0; i < selCaPr.size(); i++) {
        // The final transformation coefficients are calculated as
        // P * Wstride + R * Wdilation and Q * Hstride + S * Hdilation
        int64_t coeffOne = selCaPr[i][0] * opStride[0];
        int64_t coeffTwo = selCaPr[i][1] * opDilation[0];

        int64_t tmpGCDResult = calGCD(coeffOne, coeffTwo);

        //! Testing
        // llvm::outs() << "[Testing] Temp GCD value of (" << coeffOne << ", " << coeffTwo <<") is " << tmpGCDResult << "\n";

        result.push_back(tmpGCDResult);
    }

    return result;
}



#endif  // PIMOPT_GUROBI_NOT_INSTALLED
