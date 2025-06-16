//===----------------------------------------------------------------------===//
//
// This file implements all MILP modeling related funcs for FC.
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

GRBLinExpr DataLayoutMILP::constructFCMILP(const OperatorInfo& selOpInstance, const int32_t& selOpIndex) {
    // Return expression
    GRBLinExpr outputExpr;

    //
    // Step 1: Construct all the constant values for the layer
    //
    buildFCOpConstants(selOpInstance);

    //
    // Step 2: Create layer specific variables
    //
    OpGRBVariables tmpOpVariables;
    opVariablesMap[selOpIndex] = tmpOpVariables;

    // Step 2.1: Add all loop bound variables
    addLoopBoundVarsFC(selOpIndex);

    // Step 2.2 Add all related variables and constraints for
    //          the number of multiplications in a column in the PE
    addNumMulColVarFC(selOpIndex);

    // Step 2.3 Add variable for the number of columns used in the PE
    addNumColPEVarFC(selOpIndex);

    // Step 2.4 Add variable for the number of PEs used in the system
    addNumPESysVarFC(selOpIndex);

    // Step 2.5 Add variable for the number of output elements per column
    addNumOutColVarFC(selOpIndex);
    
    // Step 2.6 Add variable for the number of filter elements per column
    addNumFilColVarFC(selOpIndex);

    // Step 2.7 Add variable for the number of reductions per output in a column
    addNumRedOutColVarFC(selOpIndex);

    // Step 2.8 Add variable for the number of inputs in a column
    addNumInColVarFC(selOpIndex);

    // Step 2.9 Add Variable for the numebr of reductions per column
    addNumRedColVarFC(selOpIndex);

    // Step 2.10 Add Variable for the numebr of output partial sums per PE
    addNumOutPEVarFC(selOpIndex);

    // Step 2.12 Add variable to represent the number of output transmissin in the workload
    addNumOutSysVarFC(selOpIndex);

    // Step 2.13 Add variable to represent the number of inputs in a PE
    addNumInPEVarFC(selOpIndex);

    // Step 2.14 Add variable to represent the loading cost
    addNumInSysVarFC(selOpIndex);

    // If the target device is PNM
    // Add the filter loading cost
    if (selDeviceType == deviceTypeIdx::PNM) {
        addFilPELoadingVarFC(selOpIndex);
    }

    //
    // Step 3: Add all related constraints
    //

    // Step 3.1: Add loop bound constraints
    addLoopBoundConsFC(selOpIndex);

    // Step 3.2: Add NumPE constraint
    addNumPEConsFC(selOpIndex);

    // Step 3.3 Add NumCol Constraint
    addNumColConsFC(selOpIndex);

    // Step 3.4 Add ColSize Constraint
    addColSizeConsFC(selOpIndex);

    //
    // Step 4: Add the final objective
    //

    // Select based on targeted device
    if (selDeviceType == deviceTypeIdx::PUM) {
        // Check the intended cost function
        if (selObjMethod == 0) {
            // Cost 1: Get the col multiplication cost
            outputExpr += (opVariablesMap[selOpIndex].numMulCol * archInfo.mulLat) * objKnobs.colMulWeight;

            // Cost 2: Get the col reduction cost
            outputExpr += (opVariablesMap[selOpIndex].numRedCol * archInfo.addLat) * objKnobs.colAddWeight;
            
            // Cost 3: Get the output transmission cost
            outputExpr += opVariablesMap[selOpIndex].finalOutCostVar * objKnobs.outTransWeight;

            // Cost 4: Get the input loading cost
            outputExpr += opVariablesMap[selOpIndex].finalInCostVar * objKnobs.inLoadingWeight;
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
            outputExpr += opVariablesMap[selOpIndex].finalInCostVar * objKnobs.inLoadingWeight;

            // Cost 2: Computation Row Activation Cost, We store it in numMulCol
            outputExpr += opVariablesMap[selOpIndex].numFilCol * archInfo.rowAct * objKnobs.colMulWeight;

            // Cost 3: Bank Filter loading cost
            outputExpr += opVariablesMap[selOpIndex].numFilPELoading * objKnobs.colAddWeight;

            // Cost 4: Get the output transmission cost
            outputExpr += opVariablesMap[selOpIndex].finalOutCostVar * objKnobs.outTransWeight;
        } else if (selObjMethod == 1) {
            // COSA's cost function
            // Cost 2: Computation Row Activation Cost, We store it in numMulCol
            outputExpr += opVariablesMap[selOpIndex].numFilCol * archInfo.rowAct * objKnobs.colMulWeight;

            // Cost 3: Bank Filter loading cost
            outputExpr += opVariablesMap[selOpIndex].numFilPELoading * objKnobs.colAddWeight;
        }
        
    }

    return outputExpr;
}

// =============================================
//       GRB Variable Construction -- FC
// =============================================

void DataLayoutMILP::addLoopBoundVarsFC(const int32_t& selOpIdx) {

    // Create a Gurobi binary variable of the given name and type for all loop bound in resulting nested loop
    auto createBinaryVar = [&](const std::string &name, char type) {
        return model.addVar(0, 1, 0.0, type, name);
    };

    // Get the constant info of this Op
    OpConstInfo& selOpConstInfo = opConstInfoMap[selOpIdx];

    // Iterate over all loop bounds d
    std::vector<LoopBoundVariables> tmpLoopBoundVars;
    for (int d = 0; d < NUM_BOUND_FC; ++d) {
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


void DataLayoutMILP::addNumMulColVarFC(const int32_t& selOpIdx) {
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
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_K].loopBoundIntVars[loopLevelIdx::LEVEL0];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_N].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_K].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_N].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_tmpNum_Mul_Col_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "]_tmpNum_Mul_Col_Var_0_constraint");

    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_P].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue1 = tmpMaxValue0 * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_P].loopBoundValue;
    GRBVar tmpMul1 = model.addVar(1, tmpMaxValue1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_tmpNum_Mul_Col_Var_1");
    model.addQConstr(tmpMul1 == tmpMul0 * tmpLoopBound2, "[Op_" + std::to_string(selOpIdx) + "]_tmpNum_Mul_Col_Var_1_constraint");

    // Multiplication 3
    GRBVar tmpLoopBound3 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_Q].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue2 = tmpMaxValue1 * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_Q].loopBoundValue;
    GRBVar tmpMul2 = model.addVar(1, tmpMaxValue2, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_tmpNum_Mul_Col_Var_2");
    model.addQConstr(tmpMul2 == tmpMul1 * tmpLoopBound3, "[Op_" + std::to_string(selOpIdx) + "]_tmpNum_Mul_Col_Var_2_constraint");

    // Multiplication 4
    GRBVar tmpLoopBound4 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_R].loopBoundIntVars[loopLevelIdx::LEVEL0];
    model.addQConstr(finalNumMulColVar == tmpMul2 * tmpLoopBound4, "[Op_" + std::to_string(selOpIdx) + "]_num_Mul_Col_Var_cons");

    // Store the variable
    tmpOpVars.numMulCol = finalNumMulColVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumColPEVarFC(const int32_t& selOpIdx) {
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
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_N].loopBoundIntVars[loopLevelIdx::LEVEL1];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_K].loopBoundIntVars[loopLevelIdx::LEVEL1];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_N].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_K].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_num_Col_PE_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "]_num_Col_PE_Var_0_constraint");

    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_P].loopBoundIntVars[loopLevelIdx::LEVEL1];
    int64_t tmpMaxValue1 = tmpMaxValue0 * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_P].loopBoundValue;
    GRBVar tmpMul1 = model.addVar(1, tmpMaxValue1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_num_Col_PE_Var_1");
    model.addQConstr(tmpMul1 == tmpMul0 * tmpLoopBound2, "[Op_" + std::to_string(selOpIdx) + "]_num_Col_PE_Var_1_constraint");

    // Multiplication 3
    GRBVar tmpLoopBound3 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_Q].loopBoundIntVars[loopLevelIdx::LEVEL1];
    int64_t tmpMaxValue2 = tmpMaxValue1 * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_Q].loopBoundValue;
    GRBVar tmpMul2 = model.addVar(1, tmpMaxValue2, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_num_Col_PE_Var_2");
    model.addQConstr(tmpMul2 == tmpMul1 * tmpLoopBound3, "[Op_" + std::to_string(selOpIdx) + "]_num_Col_PE_Var_2_constraint");

    // Multiplication 6
    GRBVar tmpLoopBound4 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_R].loopBoundIntVars[loopLevelIdx::LEVEL1];
    model.addQConstr(finalNumColPEVar == tmpMul2 * tmpLoopBound4, "[Op_" + std::to_string(selOpIdx) + "]_num_Col_PE_Var_cons");

    // Store the variable
    tmpOpVars.numColPE = finalNumColPEVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumPESysVarFC(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumPESysVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "num_PE_Sys_Var");

    // Set the lower bound to 1, as we need at least one PE
    finalNumPESysVar.set(GRB_DoubleAttr_LB, 1);

    // To calculate the number of multiplications in a col
    // We need to multiply all the loop bounds at level 2
    
    // Multiplication 1
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_N].loopBoundIntVars[loopLevelIdx::LEVEL2];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_K].loopBoundIntVars[loopLevelIdx::LEVEL2];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_N].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_K].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_num_PE_Sys_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "]_num_PE_Sys_Var_0_constraint");

    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_P].loopBoundIntVars[loopLevelIdx::LEVEL2];
    int64_t tmpMaxValue1 = tmpMaxValue0 * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_P].loopBoundValue;
    GRBVar tmpMul1 = model.addVar(1, tmpMaxValue1, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_num_PE_Sys_Var_1");
    model.addQConstr(tmpMul1 == tmpMul0 * tmpLoopBound2, "[Op_" + std::to_string(selOpIdx) + "]_num_PE_Sys_Var_1_constraint");

    // Multiplication 3
    GRBVar tmpLoopBound3 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_Q].loopBoundIntVars[loopLevelIdx::LEVEL2];
    int64_t tmpMaxValue2 = tmpMaxValue1 * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_Q].loopBoundValue;
    GRBVar tmpMul2 = model.addVar(1, tmpMaxValue2, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_num_PE_Sys_Var_2");
    model.addQConstr(tmpMul2 == tmpMul1 * tmpLoopBound3, "[Op_" + std::to_string(selOpIdx) + "]_num_PE_Sys_Var_2_constraint");

    // Multiplication 4
    GRBVar tmpLoopBound4 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_R].loopBoundIntVars[loopLevelIdx::LEVEL2];
    model.addQConstr(finalNumPESysVar == tmpMul2 * tmpLoopBound4, "[Op_" + std::to_string(selOpIdx) + "]_num_PE_Sys_Var_cons");

    // Calculate the needed bandwidth
    // Calculate the number of Channels in the system
    GRBVar numChannelsSysVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "num_Channels_Sys_Var");
    model.addConstr(finalNumPESysVar >= 16 * numChannelsSysVar - 15, "[Op] numChan_lower");
    model.addConstr(finalNumPESysVar <= 16 * numChannelsSysVar, "[Op] numChan_upper");

    // Set the lower bound to 1, as we need at least one channel
    numChannelsSysVar.set(GRB_DoubleAttr_LB, 1);
    
    // Calculate the channel bandwidth
    GRBVar channelBandwidthVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "channel_Bandwidth_Var");
    model.addConstr(channelBandwidthVar == numChannelsSysVar * archInfo.SysBandWidth, "[Op_" + std::to_string(selOpIdx) + "] channel_Bandwidth_Var_constraint");

    // Store the variable
    tmpOpVars.numPESys = finalNumPESysVar;
    tmpOpVars.numChannelsSys = numChannelsSysVar;
    tmpOpVars.channelLevelBandwidth = channelBandwidthVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumOutColVarFC(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = 1;

    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_P].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_R].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_N].loopBoundValue;
    
    // Create the final variable
    GRBVar finalNumOutColVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "num_Out_Col_Var");

    // The number of outputs in a column can be calculated as
    // NumOutCol = N * P * R at loop level 0
    // Multiplication 1
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_P].loopBoundIntVars[loopLevelIdx::LEVEL0];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_R].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_P].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_R].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_num_Out_Col_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "]_num_Out_Col_Var_0_constraint");

    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_N].loopBoundIntVars[loopLevelIdx::LEVEL0];
    model.addQConstr(finalNumOutColVar == tmpMul0 * tmpLoopBound2, "[Op_" + std::to_string(selOpIdx) + "]_num_Out_Col_Var_constraint");

    // Store the variable
    tmpOpVars.numOutCol = finalNumOutColVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumFilColVarFC(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = 1;

    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_N].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_R].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_Q].loopBoundValue;
    
    // Create the final variable
    GRBVar finalNumFilColVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "num_Fil_Col_Var");

    // The number of outputs in a column can be calculated as
    // NumFilCol = N * R * Q at loop level 0
    // Multiplication 1
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_R].loopBoundIntVars[loopLevelIdx::LEVEL0];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_Q].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_R].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_Q].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_num_Fil_Col_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "]_num_Fil_Col_Var_0_constraint");

    
    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_N].loopBoundIntVars[loopLevelIdx::LEVEL0];
    model.addQConstr(finalNumFilColVar == tmpMul0 * tmpLoopBound2, "[Op_" + std::to_string(selOpIdx) + "]_num_Fil_Col_Var_constraint");

    // Store the variable
    tmpOpVars.numFilCol = finalNumFilColVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumRedOutColVarFC(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // The number of reduction per output in a column can be calculated as
    // numRedOutCol = Q - 1; at loop level 0

    // Maximum value bound
    int64_t maxValue = selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_Q].loopBoundValue;

    // Create the final variable
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_Q].loopBoundIntVars[loopLevelIdx::LEVEL0];
    GRBVar finalNumRedOutColVar = model.addVar(0, maxValue, 0.0, GRB_INTEGER, "num_red_out_col_Var");
    model.addConstr(finalNumRedOutColVar == tmpLoopBound0 - 1, "[Op_" + std::to_string(selOpIdx) + "]_num_red_out_col_Var_constraint");

    // Store the variable
    tmpOpVars.numRedOutCol = finalNumRedOutColVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumInColVarFC(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = 1;

    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_Q].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_P].loopBoundValue;
    maxValue *= selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_N].loopBoundValue;
    
    // Create the final variable
    GRBVar finalNumInColVar = model.addVar(1, maxValue, 0.0, GRB_INTEGER, "num_In_Col_Var");

    // The number of outputs in a column can be calculated as
    // NumInCol = N * P * Q at loop level 0
    // Multiplication 1
    GRBVar tmpLoopBound0 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_Q].loopBoundIntVars[loopLevelIdx::LEVEL0];
    GRBVar tmpLoopBound1 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_P].loopBoundIntVars[loopLevelIdx::LEVEL0];
    int64_t tmpMaxValue0 = selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_P].loopBoundValue * selOpInfo.loopBoundInfoMap[loopBoundIdxFC::FC_Q].loopBoundValue;
    GRBVar tmpMul0 = model.addVar(1, tmpMaxValue0, 0.0, GRB_INTEGER, "[Op_" + std::to_string(selOpIdx) + "]_num_In_Col_Var_0");
    model.addQConstr(tmpMul0 == tmpLoopBound0 * tmpLoopBound1, "[Op_" + std::to_string(selOpIdx) + "]_num_In_Col_Var_0_constraint");
    
    // Multiplication 2
    GRBVar tmpLoopBound2 = tmpOpVars.loopBoundsVars[loopBoundIdxFC::FC_N].loopBoundIntVars[loopLevelIdx::LEVEL0];
    model.addQConstr(finalNumInColVar == tmpMul0 * tmpLoopBound2, "[Op_" + std::to_string(selOpIdx) + "]_num_In_Col_Var_constraint");

    // Store the variable
    tmpOpVars.numInCol = finalNumInColVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumRedColVarFC(const int32_t& selOpIdx) {
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

void DataLayoutMILP::addNumOutPEVarFC(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumOutPEVar = model.addVar(0, maxValue * archInfo.numCol, 0.0, GRB_INTEGER, "num_Out_PE_Var");

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

void DataLayoutMILP::addNumOutSysVarFC(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumOutSysVar = model.addVar(0, maxValue * archInfo.numCol, 0.0, GRB_CONTINUOUS, "num_Out_Sys_Var");
    GRBVar finalOutCostVar = model.addVar(0, maxValue * maxValue, 0.0, GRB_CONTINUOUS, "Out_Cost_Sys_Var");

    // Get the desired variables
    GRBVar numPESys = tmpOpVars.numPESys;
    GRBVar numOutPE = tmpOpVars.numOutPE;
    GRBVar numOutCol = tmpOpVars.numOutCol;

    // Get the bandwidth
    double_t tmpBandWidth = 1.0 / static_cast<double_t>(archInfo.SysBandWidth);
    double_t tmpConstant = tmpBandWidth * archInfo.dataWidth;

    // Get the final variable
    if (selDeviceType == deviceTypeIdx::PUM) {
        model.addQConstr(finalNumOutSysVar == numPESys * numOutPE, "num_Out_Sys_Var_constraint");
    } else if (selDeviceType == deviceTypeIdx::PNM) {
        model.addQConstr(finalNumOutSysVar == numPESys * numOutCol, "num_Out_Sys_Var_constraint");
    }

    // Get the real cost of output transmission
    model.addQConstr(finalOutCostVar * tmpOpVars.channelLevelBandwidth == finalNumOutSysVar * archInfo.dataWidth, "Out_Cost_Sys_Var_constraint");
    
    // Store the variable
    tmpOpVars.numOutSys = finalNumOutSysVar;
    tmpOpVars.finalOutCostVar = finalOutCostVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addNumInPEVarFC(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumInPEVar = model.addVar(1, maxValue * archInfo.numCol, 0.0, GRB_INTEGER, "num_In_PE_Var");

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

void DataLayoutMILP::addNumInSysVarFC(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalNumInSysVar = model.addVar(0, maxValue * archInfo.numCol, 0.0, GRB_CONTINUOUS, "num_In_Sys_Var");
    GRBVar finalInCostVar = model.addVar(0, maxValue * maxValue, 0.0, GRB_CONTINUOUS, "In_Cost_Sys_Var");


    // Get the desired variables
    GRBVar numPESys = tmpOpVars.numPESys;
    GRBVar numInPE = tmpOpVars.numInPE;

    // Get the bandwidth
    double_t tmpBandWidth = 1.0 / static_cast<double_t>(archInfo.SysBandWidth);
    double_t tmpConstant = tmpBandWidth * archInfo.dataWidth;

    // Get the final variable
    model.addQConstr(finalNumInSysVar == numPESys * numInPE * tmpConstant, "num_In_Sys_Var_constraint");
    model.addQConstr(finalInCostVar * tmpOpVars.channelLevelBandwidth == finalNumInSysVar * archInfo.dataWidth, "tmp_In_Cost_Sys_Var_constraint");

    // Store the variable
    tmpOpVars.numInSys = finalNumInSysVar;
    tmpOpVars.finalInCostVar = finalInCostVar;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DataLayoutMILP::addFilPELoadingVarFC(const int32_t& selOpIdx) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIdx];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIdx];

    // Maximum value bound
    int64_t maxValue = selOpInfo.maxLoopBound;

    // Define the final variable
    GRBVar finalFilPELoadingVar = model.addVar(0, maxValue * archInfo.numCol, 0.0, GRB_CONTINUOUS, "num_Fil_Loading_Var");

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
//      GRB Constraints Construction -- FC
// =============================================

void DataLayoutMILP::addLoopBoundConsFC(const int32_t& selLayer) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selLayer];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selLayer];

    // Iterate over all loop bounds d
    for (int d = 0; d < NUM_BOUND_FC; d++) {
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

void DataLayoutMILP::addNumPEConsFC(const int32_t& selOp) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOp];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOp];

    // Use the variable directly
    model.addConstr(tmpOpVars.numPESys <= selOpInfo.numPE, "num_PE_Constraint");
    // model.addConstr(tmpOpVars.numPESys == selOpInfo.numPE, "num_PE_Constraint");
}

void DataLayoutMILP::addNumColConsFC(const int32_t& selOp) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOp];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOp];

    // Use the variable directly
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

void DataLayoutMILP::addColSizeConsFC(const int32_t& selOp) {
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
            model.addConstr( (numFilCol + numInCol) * archInfo.dataWidth <= archInfo.numRow, "Col_Size_Constraint");
        } else if (selStorageMethod == 1) {
            model.addConstr( (numFilCol) * archInfo.dataWidth <= archInfo.numRow, "Col_Size_Constraint");
        } else if (selStorageMethod == 2) {
            model.addConstr( (numOutCol + numFilCol + numInCol) * archInfo.dataWidth <= archInfo.numRow, "Col_Size_Constraint");
        }
    } else if (selDeviceType == deviceTypeIdx::PNM) {
        if (selStorageMethod == 0) {
            // Check the number counting method
            if (selNumCountingMethod == 1) {
                // Cosa's storage constriant
                // Check whether we are using COSA's cost function or not
                if (selObjMethod == 0) {
                    // Our Cost function
                    model.addConstr( (numFilCol) <= archInfo.numRow / 2, "Col_Size_Constraint");
                    model.addConstr(numInCol <= archInfo.numRow / 2, "Col_Size_Constraint_In_COSA");
                } else if (selObjMethod == 1) {
                    // COSA's cost function
                    double_t tmpCOSACons = archInfo.numCol / 256;

                    // COSA's cost function
                    model.addConstr( (numFilCol) <= (archInfo.numRow * tmpCOSACons) / 2, "Col_Size_Constraint_Fil_COSA");
                    model.addConstr( (numInCol) <= (archInfo.numRow * tmpCOSACons) / 2, "Col_Size_Constraint_In_COSA");
                }
            } else if (selNumCountingMethod == 0) {
                // Our constraint
                // Check whether we are using COSA's cost function or not
                if (selObjMethod == 0) {
                    // Our cost function
                    model.addConstr( (numFilCol + numInCol) <= archInfo.numRow, "Col_Size_Constraint");
                } else if (selObjMethod == 1) {
                    // Cosa's cost function
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
                // COSA's storage constraint
                // Check the selected objective function
                if (selObjMethod == 0) {
                    // Our cost function
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
                // Our sotrage constraint
                // Check the selected objective function
                if (selObjMethod == 0) {
                    // Our cost function
                    model.addConstr( (numOutCol + numFilCol + numInCol) <= archInfo.numRow, "Col_Size_Constraint");
                } else if (selObjMethod == 1) {
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
void DataLayoutMILP::buildFCOpConstants(const OperatorInfo& selOpIns) {
    // Build the constant storing struct
    OpConstInfo tmpConstInfo;
    tmpConstInfo.opID = selOpIns.opID;
    tmpConstInfo.opType = selOpIns.opType;
    tmpConstInfo.numPE = selOpIns.numPE;

    int64_t tmpMaxLoopBound = 1;

    // Build hte storing struct for all loop bounds' divisor lists
    for (int i = 0; i < NUM_BOUND_FC; i++) {
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

    // For FC we don't need the Coefficient LUT
    // [Final] : Store all calcualted constants
    // Construct the opConstInfo map
    opConstInfoMap[selOpIns.opID] = tmpConstInfo;

    //! Testing
    tmpConstInfo.printDetail();
}




#endif  // PIMOPT_GUROBI_NOT_INSTALLED
