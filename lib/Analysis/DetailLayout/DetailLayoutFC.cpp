//===----------------------------------------------------------------------===//
//
// This file implements functions for Fully Connected Layer in the DetailLayout.
//
//===----------------------------------------------------------------------===//

#include "pimopt/Analysis/DetailLayout/DetailLayoutMILP.h"

// Check the existence of Gurobi library
#ifndef PIMOPT_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace mlir;
using namespace pim;

// ====================================
//           Overall GRB Setup
// ====================================

GRBLinExpr DetailLayoutMILP::setupFCGRB(const LayerInfo& selLayerInstance, const int32_t& selLayerIndex) {
    //
    GRBLinExpr outputExpr;

    //
    // Step 1: Build the storing structure for a single layer
    //
    buildFCLayerConstants(selLayerInstance);

    //
    // Step 2: Create layer specific variables
    //
    LayerGRBVariables tmpLayerVariables;
    layerVariables[selLayerIndex] = tmpLayerVariables;

    // Step 2.1: Add all loop bound variables
    addLoopBoundVarsFC(selLayerIndex);

    // Step 2.2: Add tensor size variables, for W, OA and IA
    addTensorSizeVarsFC(selLayerIndex);

    //
    // Step 3: Add Constraints
    //
    // Step 3.1: Add Loop Bound Constraint
    addLoopBoundConsFC(selLayerIndex);

    // Step 3.2: Add WorkLoad Constraint
    addWorkLoadConsFC(selLayerIndex, PerformanceKnobs.workLoadTolerance);

    // Step 3.3: Add Number of Banks Constraint
    addNumBankConsFC(selLayerIndex);

    // Step 3.4: Add Cartesian Product Constraint
    addTensorValueConsFC(selLayerIndex);
    addCartesianProductConsFC(selLayerIndex, PerformanceKnobs.cartesianProTolerance);

    //
    // Step 3.5: Add Scheme Specific Constraints
    //
    // Check the specified layout scheme
    if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_1) {
        addScheme1ConsFC(selLayerIndex);
    } else if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_2) {
        addScheme2ConsFC(selLayerIndex);
    } else {
        addScheme3ConsFC(selLayerIndex);
    }

    //
    // Step 4: Add Objectives (Scheme specific)
    //

    // Step 4.1 : Add compute latency objective
    GRBLinExpr computeObj;
    GRBLinExpr rowActObj;
    GRBLinExpr crossBankObj;
    GRBLinExpr intraBankObj;

    // Step 4.2 : Add Scheme Specific Data Transfer Latency Obj
    if (ObjApproScheme == ObjApproximationScheme::Manual_log) {
        computeObj = addComputeObjectFC(selLayerIndex);
        if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_1) {
            rowActObj += addScheme1RowActObjectFC(selLayerIndex);
            crossBankObj += addScheme1CrossBankObjFC(selLayerIndex);
        } else if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_2) {
            rowActObj += addScheme2RowActObjectFC(selLayerIndex);
            crossBankObj += addScheme2CrossBankObjFC(selLayerIndex);
        } else {
            rowActObj += addScheme3RowActObjectFC(selLayerIndex);

            if (PerformanceKnobs.intraBankWeight > 0) intraBankObj += addScheme3IntraBankObjConv2D(selLayerIndex);
        }
    } else if (ObjApproScheme == ObjApproximationScheme::Manual_norm) {
        computeObj = (addComputeObjectFC(selLayerIndex) + maxMacScaleFactorLog);
        if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_1) {
            rowActObj += (addScheme1RowActObjectFC(selLayerIndex) + maxMacScaleFactorLog);
            crossBankObj += (addScheme1CrossBankObjFC(selLayerIndex) + maxMacScaleFactorLog);
        } else if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_2) {
            rowActObj += (addScheme2RowActObjectFC(selLayerIndex) + maxMacScaleFactorLog);
            crossBankObj += (addScheme2CrossBankObjFC(selLayerIndex) + maxMacScaleFactorLog);
        } else {
            rowActObj += (addScheme3RowActObjectFC(selLayerIndex) + maxMacScaleFactorLog);

            if (PerformanceKnobs.intraBankWeight > 0) intraBankObj += (addScheme3IntraBankObjConv2D(selLayerIndex) + maxMacScaleFactorLog);
        }
    } else if (ObjApproScheme == ObjApproximationScheme::Gurobi_exp) {
        computeObj = addComputeObjectFCPWLNorm(selLayerIndex);
        if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_1) {
            rowActObj += addScheme1RowActObjectFCPWLNorm(selLayerIndex);
            crossBankObj += addScheme1CrossBankObjFCPWLNorm(selLayerIndex);
        } else if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_2) {
            rowActObj += addScheme2RowActObjectFCPWLNorm(selLayerIndex);
            crossBankObj += addScheme2CrossBankObjFCPWLNorm(selLayerIndex);
        } else {
            rowActObj += addScheme3RowActObjectFCPWLNorm(selLayerIndex);

            if (PerformanceKnobs.intraBankWeight > 0) intraBankObj += addScheme3IntraBankObjConv2DPWLNorm(selLayerIndex);
        }
    }
    

    // [Output] Update LayerGroup Objective function
    outputExpr += PerformanceKnobs.compWeight * computeObj;
    outputExpr += PerformanceKnobs.rowActWeight * rowActObj;
    outputExpr += PerformanceKnobs.crossBankWeight * crossBankObj;
    outputExpr += PerformanceKnobs.intraBankWeight * intraBankObj;

    return outputExpr;
}


// =============================================
//         GRB Variables Creation -- FC
// =============================================

void DetailLayoutMILP::addLoopBoundVarsFC(const int32_t& selLayer) {
    // Create a Gurobi variable of the given name and type for all loop bound in resulting nested loop
    auto createVar = [&](const std::string &name, char type) {
        return model.addVar(0, 1, 0.0, type, name);
    };

    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Iterate through all workload dimensions
    std::vector<std::vector<std::vector<GRBVar>>> tmpWorkLoadVars;
    for (int i = 0; i < NUM_DIM_FC; ++i) {
        std::vector<std::vector<GRBVar>> tmpMemLevelVars;

        // Iterate through all mem levels
        for (int j = 0; j < NUM_MEM_LEVEL; ++j) {
            std::vector<GRBVar> tmpDivOneHotVar;

            // Iterate through all divisors
            for (int k = 0; k < selLayerInfo.workLoadDimMap[i].divisorsVec.size(); ++k) {
                // Construct Variable name
                std::string tmpVarName = "L_" + std::to_string(selLayer) + "_" + 
                                            "X_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k);

                // Create and push back this variable
                tmpDivOneHotVar.push_back(createVar(tmpVarName, GRB_BINARY));
            }

            tmpMemLevelVars.push_back(tmpDivOneHotVar);
        }

        tmpWorkLoadVars.push_back(tmpMemLevelVars);
    }

    // Store all created variables
    layerVariables[selLayer].memLoopBoundOneHotVars = tmpWorkLoadVars;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

void DetailLayoutMILP::addTensorSizeVarsFC(const int32_t& selLayer) {
    // Create a Gurobi variable of the given name and type for all potential values in resulting nested loop
    auto createVar = [&](const std::string &name, char type) {
        return model.addVar(0, 1, 0.0, type, name);
    };

    // Get the needed layer info 
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Iterate through all mem levels
    std::vector<std::vector<std::vector<GRBVar>>> tmpMemLevelVars;
    for (int i = 0; i < (NUM_MEM_LEVEL - 1); ++i) {
        std::vector<std::vector<GRBVar>> tmpTensorLevelVars;
        
        // Iterate through all tensor types
        for (int j = 0; j < 3; ++j) {
            std::vector<GRBVar> tmpCartesianProductVars;

            for (int k = 0; k < selLayerInfo.tensorValues[j].size(); ++k) {
                // Construct the variable name
                std::string tmpVarName = "L_" + std::to_string(selLayer) + "_" + 
                                            "E_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k);

                //! Testing
                // llvm::outs() << tmpVarName << "\n";

                // Create and push back this variable
                tmpCartesianProductVars.push_back(createVar(tmpVarName, GRB_BINARY));
            }

            tmpTensorLevelVars.push_back(tmpCartesianProductVars);
        }

        tmpMemLevelVars.push_back(tmpTensorLevelVars);
    }

    // Store all created variables
    layerVariables[selLayer].tensorValueOneHotVars = tmpMemLevelVars;

    // Update the model before returning so that these variables can be referenced
    // safely during the rest of model creation
    model.update();
}

// ====================================
//          Constants Creation
// ====================================

void DetailLayoutMILP::buildFCLayerConstants(const LayerInfo& selLayerIns) {
    // Get the layer id
    int32_t tmp_layer_id = selLayerIns.layerId;

    // Get Layer Type
    int32_t tmp_layer_type = selLayerIns.layerType;

    // Create structs for stroing layer constants information
    ConstInfo tmp_const_info;
    tmp_const_info.layerId = tmp_layer_id;
    tmp_const_info.layerType = tmp_layer_type;
    tmp_const_info.numBanks = selLayerIns.numBanks;

    // Layer scheme info
    tmp_const_info.layoutScheme = selLayoutScheme;

    // Build the storing struct for all workload dimension divisor lists
    for (int i = 0; i < NUM_DIM_FC; i++) {
        WorkLoadDimInfo tmp_workLoadDimInfo;

        tmp_workLoadDimInfo.workLoadDim = selLayerIns.workLoadDimVec[i];
        tmp_workLoadDimInfo.divisorsVec = getAllDivisors(selLayerIns.workLoadDimVec[i]);

        // Get the log2 value of all divisorsVec
        std::vector<double_t> tmpDivisorsLogVec(tmp_workLoadDimInfo.divisorsVec.size());

        std::transform(tmp_workLoadDimInfo.divisorsVec.begin(), tmp_workLoadDimInfo.divisorsVec.end(), tmpDivisorsLogVec.begin(),
                            [](int32_t num) { return std::log2(static_cast<double_t>(num)); });

        tmp_workLoadDimInfo.divisorsLogVec = tmpDivisorsLogVec;

        // Store the struct in const_info map
        tmp_const_info.workLoadDimMap[i] = tmp_workLoadDimInfo;
    }

    // Build the Cartesian Product sets for all three data tensors
    // With the following order:
    // 0 : Matrix A;
    // 1 : Matrix B;
    // 2 : Output
    for (int tensor_type = 0; tensor_type < 3; ++tensor_type) {
        std::vector<Set<int64_t>> tempSets;

        // Related sets are identified by the relation matrix
        for (int dimIndex = 0; dimIndex < NUM_DIM_FC; ++dimIndex) {
            if (FCRelationMatrix[dimIndex][tensor_type] > 0) {
                tempSets.push_back(tmp_const_info.workLoadDimMap[dimIndex].divisorsVec);
            }
        }

        // Get the corresponding Cartesian Product
        CartesianProduct<int64_t> tmpCaPr = calCartesian(tempSets);

        // Get the all potentail tensor values from the obtained CaPr
        std::vector<int64_t> tmpTensorValues = calPTensorValue(tmpCaPr);

        // Store back
        tmp_const_info.tensorCartSets.push_back(tmpCaPr);
        tmp_const_info.tensorValues.push_back(tmpTensorValues);
    }

    // Calculate top mem level tensor sizes
    for (int tensor_type = 0; tensor_type < 3; ++tensor_type) {
        int64_t tmpResult = 1;

        for (int dimIndex = 0; dimIndex < NUM_DIM_FC; ++dimIndex) {
            if (FCRelationMatrix[dimIndex][tensor_type] > 0) {
                tmpResult *= selLayerIns.workLoadDimVec[dimIndex];
            }
        }

        tmp_const_info.totalTensorSize.push_back(tmpResult);
    }

    // [Final] : Store all calcualted constants
    // Construct the layerConstInfo map
    layerConstInfo[tmp_layer_id] = tmp_const_info;

    //! Testing
    // tmp_const_info.printDetail();
}


// ====================================
//       GRB Constraints -- FC 
// ====================================

void DetailLayoutMILP::addLoopBoundConsFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Iterate through all workload dim
    for (int i = 0; i < NUM_DIM_FC; ++i) {
        // Iterate through all mem level 
        for (int j = 0; j < NUM_MEM_LEVEL; j++) {
            // Sum up all one hot encoding variables for a given workload i and mem level j
            GRBLinExpr loopBoundConstraint;

            for (int k = 0; k < selLayerInfo.workLoadDimMap[i].divisorsVec.size(); ++k) {
                loopBoundConstraint += layerVariables[selLayer].memLoopBoundOneHotVars[i][j][k];
            }

            // Define Constraint Name
            // [Layer l] LoopBound_i_j, Work Load i, Mem Level j in layer l
            std::string tmpConsName = "[Layer_" + std::to_string(selLayer) + "]_" + 
                                        "LoopBoundConstraint_" + std::to_string(i) + "_" + std::to_string(j);

            // Add constraint
            model.addConstr(loopBoundConstraint == 1, tmpConsName);
        }
    }
}

void DetailLayoutMILP::addWorkLoadConsFC(const int32_t& selLayer, const double_t& tolerance) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Iterate through all workload dim
    for (int i = 0; i < NUM_DIM_FC; ++i) {
        // Sum up all loop bound related terms
        GRBLinExpr workLoadConstraint;

        // Iterate through all memory levels
        for (int j = 0; j < NUM_MEM_LEVEL; ++j) {
            GRBLinExpr memLevelConstraint;
            // Iterate through all loop bound one hot encoding vars
            for (int k = 0; k < selLayerInfo.workLoadDimMap[i].divisorsVec.size(); ++k) {
                memLevelConstraint += selLayerInfo.workLoadDimMap[i].divisorsLogVec[k] 
                                        * layerVariables[selLayer].memLoopBoundOneHotVars[i][j][k];
            }

            //
            workLoadConstraint += memLevelConstraint;
        }

        // Construct Constraint Name
        // [Layer l] WorkLoadConstraint_i_upper, workload dim i, upper bound of the constraint
        std::string tmpConsNameUpper = "[Layer_" + std::to_string(selLayer) + "]_" + 
                                            "WorkLoadConstraint_" + std::to_string(i) + "_upper";

        // [Layer l] WorkLoadConstraint_i_lower, workload dim i, lower bound of the constraint
        std::string tmpConsNameLower = "[Layer_" + std::to_string(selLayer) + "]_" + 
                                            "WorkLoadConstraint_" + std::to_string(i) + "_lower";

        double_t rhs = std::log2(static_cast<double_t>(selLayerInfo.workLoadDimMap[i].workLoadDim));

        // Add the constraints to the model
        if (memoryAlloc == MemoryAllocationStrategy::Exclusive) {
            model.addConstr(workLoadConstraint <= (rhs + tolerance), tmpConsNameUpper);
            model.addConstr(workLoadConstraint >= (rhs - tolerance), tmpConsNameLower);
        } else if (memoryAlloc == MemoryAllocationStrategy::Combined) {
            model.addConstr(workLoadConstraint == rhs, tmpConsNameLower);
        }
    }
}

void DetailLayoutMILP::addNumBankConsFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // For this constraint, we only care about loop bounds at bank level
    // Iterate through all workload dims
    GRBLinExpr bankNumConstraint;
    for (int i = 0; i < NUM_DIM_FC; ++i) {
        // Sum up all bank number
        GRBLinExpr divisorConstraint;

        for (int k = 0; k < selLayerInfo.workLoadDimMap[i].divisorsVec.size(); ++k) {
            double_t tmpConstant = selLayerInfo.workLoadDimMap[i].divisorsLogVec[k];

            // Testing
            // llvm::outs() << tmpConstant << "\n";

            divisorConstraint += tmpConstant * layerVariables[selLayer].memLoopBoundOneHotVars[i][memLevel::Bank][k];
        }

        bankNumConstraint += divisorConstraint;
    }

    // Construct constraint name
    std::string tmpConName = "[Layer_" + std::to_string(selLayer) + "]_" + "NumBankConstraint";
    double_t rhs = std::log2(static_cast<double_t>(selLayerInfo.numBanks));

    // Add the constraint in the model and catch error
    try {
        model.addConstr(bankNumConstraint <= rhs, tmpConName);
    } catch (GRBException& e) {
        llvm::errs() << "Gurobi Exception occurred: " << e.getMessage() << "\n";
    }
}

void DetailLayoutMILP::addTensorValueConsFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Iterate through all mem levels below bank
    for (int i = 0; i < memLevel::Bank; ++i) {
        // Iterate through all tensor types
        for (int j = 0; j < 3; ++j) {
            // Sum up all one hot encoding variables for a specifc mem level i and tensor type j
            GRBLinExpr tensorOneHotCon;

            // Iterate over all potential values for given i, j
            for (int k = 0; k < selLayerInfo.tensorValues[j].size(); ++k) {
                tensorOneHotCon += layerVariables[selLayer].tensorValueOneHotVars[i][j][k];
            }

            // Construct constraint name
            // [Layer l] tensorValueConstraint_i_j, mem level i, tensor type j
            std::string tmpConName = "[Layer_" + std::to_string(selLayer) + "]_" + 
                                        "tensorValueConstraint_" + std::to_string(i) + "_" + std::to_string(j);

            // Add to the model
            model.addConstr(tensorOneHotCon == 1, tmpConName);
        }
    }
}

void DetailLayoutMILP::addCartesianProductConsFC(const int32_t& selLayer, const double_t& tolerance) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // We need to make sure that e(m,v,h) aligns with x(d,m,t), which means the product of one workload type below a certain memory level
    // which is selected by e(m,v,h) equals to the product of that specific loop bounds below that mem level (determined by x(d,m,t))
    // We relax these constraints with an upper and lower bound according to the specified tolerance.
    // Iterate over all tensor types -- v
    for (int tensorType = 0; tensorType < 3; ++tensorType) {

        // Iterate over all workload dim -- d
        for (int workLoadDim = 0; workLoadDim < NUM_DIM_FC; ++workLoadDim) {
            // If this workLoadDim is related to the tensorType
            if (FCRelationMatrix[workLoadDim][tensorType] > 0) {

                // Iterate over all memory levels -- m
                for (int memLevel = 0; memLevel < memLevel::Bank; ++memLevel) {
                    // Construct the left-hand side of the constraint
                    GRBLinExpr LHSConstraint;

                    for (int oneHotIndex = 0; oneHotIndex < selLayerInfo.tensorValues[tensorType].size(); ++oneHotIndex) {
                        // Get the selected workload dimension specific value
                        double_t workLoadFactorLog = std::log2(static_cast<double_t>(selLayerInfo.tensorCartSets[tensorType][oneHotIndex][FCworkLoadToTensorCaPr[workLoadDim][tensorType]]));

                        LHSConstraint += layerVariables[selLayer].tensorValueOneHotVars[memLevel][tensorType][oneHotIndex] * workLoadFactorLog;
                    }

                    // Construct the right hand side
                    GRBLinExpr RHSConstraint;

                    // Traverse all mem levels below
                    for (int m = 0; m <= memLevel; ++m) {
                        // Iterate over all divisors
                        for (int d = 0; d < selLayerInfo.workLoadDimMap[workLoadDim].divisorsVec.size(); ++d) {
                            RHSConstraint += selLayerInfo.workLoadDimMap[workLoadDim].divisorsLogVec[d] * layerVariables[selLayer].memLoopBoundOneHotVars[workLoadDim][m][d];
                        }
                    }

                    // Construct the name of the constraint
                    // Format: "[Layer l] OneHotLoopBondCon<tensorType>_<workLoadDim>_<memLevel>"
                    std::string tmpConNameUpper = "[Layer_" + std::to_string(selLayer) + "]_" + 
                                                        "OneHotLoopBoundCon_" + std::to_string(tensorType) + "_" + std::to_string(workLoadDim) + "_" + std::to_string(memLevel) + "_Upper";
                    std::string tmpConNameLower = "[Layer_" + std::to_string(selLayer) + "]_" + 
                                                        "OneHotLoopBoundCon_" + std::to_string(tensorType) + "_" + std::to_string(workLoadDim) + "_" + std::to_string(memLevel) + "_Lower";

                    // Add the constraint to model
                    if (memoryAlloc == MemoryAllocationStrategy::Exclusive) {
                        model.addConstr(LHSConstraint <= RHSConstraint + tolerance, tmpConNameUpper);
                        model.addConstr(LHSConstraint >= RHSConstraint - tolerance, tmpConNameLower);
                    } else if (memoryAlloc == MemoryAllocationStrategy::Combined) {
                        model.addConstr(LHSConstraint == RHSConstraint, tmpConNameLower);
                    }
                }
            }

        }
    }
}

void DetailLayoutMILP::addScheme1ConsFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // For Scheme 1, we have the following constraint:
    //      Total sensor size < all-bank size
    int64_t totalTensorSize = 0;
    for (auto tensorSize: selLayerInfo.totalTensorSize) totalTensorSize += (tensorSize * archInfo.dataSize);

    double_t NumBank = static_cast<double_t>(totalTensorSize) / archInfo.bankSize;
    double_t totalTensorSizeLog = std::log2(NumBank);

    // Construct the RHS
    GRBLinExpr scheme1Con;

    // Iterate over all workload Dim
    for (int i = 0; i < NUM_DIM_FC; ++i) {

        // Iterate over all Loop Bound onehot encoding variable
        for (int j = 0; j < selLayerInfo.workLoadDimMap[i].divisorsVec.size(); ++j) {
            scheme1Con += selLayerInfo.workLoadDimMap[i].divisorsLogVec[j] * layerVariables[selLayer].memLoopBoundOneHotVars[i][memLevel::Bank][j];
        }
    }

    // Name
    std::string tmpConName = "[Layer_" + std::to_string(selLayer) + "]_" + 
                                "Scheme_1_Constraint";
    model.addConstr(scheme1Con >= totalTensorSizeLog, tmpConName);
}

void DetailLayoutMILP::addScheme2ConsFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Get size of Input at row level
    GRBLinExpr totalTensorSize;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < selLayerInfo.tensorValues[i].size(); ++j) {
            totalTensorSize += (selLayerInfo.tensorValues[i][j] * layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][i][j] * archInfo.dataSize);
        }
    }

    // Get size of Input at row level
    std::string tmpConName = "[Layer_" + std::to_string(selLayer) + "]_" + 
                                "Scheme_2_Constraint";
    model.addConstr(totalTensorSize <= archInfo.bankSize, tmpConName);
}

void DetailLayoutMILP::addScheme3ConsFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer]; 

    // Get size of Input at Row Buffer Level
    GRBLinExpr totalTensorSize;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < selLayerInfo.tensorValues[i].size(); ++j) {
            totalTensorSize += (selLayerInfo.tensorValues[i][j] * layerVariables[selLayer].tensorValueOneHotVars[memLevel::RowBuffer][i][j] * archInfo.dataSize);
        }
    }

    // Get the number of rows
    //! I add the following constraint to make sure the number of rows used in one bank is smaller than the maximum number of rows in a bank
    GRBLinExpr totalRowNumber;
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all divs
        for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            totalRowNumber += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] * 
                                layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
        }
    }

    // Add Tensor Size Constraint
    std::string tmpConName = "[Layer_" + std::to_string(selLayer) + "]_" + 
                                "Scheme_3_Constraint_RowBuffer";
    model.addConstr(totalTensorSize <= archInfo.rowSize, tmpConName);

    // Add Row number constraint
    std::string tmpConName2 = "[Layer_" + std::to_string(selLayer) + "]_" + 
                                    "Scheme_3_Constraint_RowNumber";

    double_t num_rows = static_cast<double_t>(archInfo.bankSize) / archInfo.rowSize;
    double_t num_rows_log = std::log2(num_rows);

    model.addConstr(totalRowNumber <= num_rows_log, tmpConName2);
}


// ====================================
//       GRB Objectives -- FC
// ====================================
GRBLinExpr DetailLayoutMILP::addComputeObjectFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr computeObj;

    // Iterate over all workload dimension
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all divisors
        for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            // Compute Number of Sequence
            computeObj += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] * 
                            layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];

            // Compute #cycles per sequence
            // double_t tmpConstant = selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div];
            double_t tmpConstant = selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div];

            computeObj += (tmpConstant * 
                            layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div]);
        }
    }

    double_t tmpCoEff =  static_cast<double_t>(archInfo.dataSize) / archInfo.dataQueueSizeHBM;

    computeObj += std::log2(tmpCoEff);

    return computeObj; 
}

GRBLinExpr DetailLayoutMILP::addScheme1RowActObjectFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme1RAObj;

    // Get Total Tensor Size
    int64_t totalTensorSize = 0;
    for (auto tensorSize: selLayerInfo.totalTensorSize) totalTensorSize += (tensorSize * archInfo.dataSize);

    double_t tmpConstant = static_cast<double_t>(totalTensorSize) / archInfo.rowSize;
    double_t tmpConstantLog = std::log2(tmpConstant);

    scheme1RAObj += tmpConstantLog;

    // Iterate over all workLoad
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all divisors
        for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {
            scheme1RAObj -= selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] * 
                                layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    }

    // Add the final log2(T_act)
    double_t tmpConstantLog2 = std::log2(archInfo.rowActTime);

    scheme1RAObj += tmpConstantLog2;

    return scheme1RAObj;
}

GRBLinExpr DetailLayoutMILP::addScheme1CrossBankObjFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme1Crossobj;

    // We need to calculate cross bank latency for all three tensor types
    // Iterate over all tensor types
    for (int tensorType = 0; tensorType < 3; ++tensorType) {
        
        // Iterate over all potential values
        for (int value = 0; value < selLayerInfo.tensorValues[tensorType].size(); ++value) {

            // Get the constant
            double_t innerConstant = std::log2((static_cast<double_t>(selLayerInfo.tensorValues[tensorType][value])));

            scheme1Crossobj += layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][tensorType][value] *
                                    innerConstant;
        }



        // Iterate over all workLoad Dims
        for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

            // Iterate over all divisors
            for (int div = 0;  div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {
                
                scheme1Crossobj += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                    layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
            }
        }
    }

    // Add the constants
    double_t tmpConstant = (static_cast<double_t>(archInfo.dataSize) / archInfo.dataQueueSizeHBM) * archInfo.crossBankTime;
    double_t tmpConstantLog = std::log2(tmpConstant);

    scheme1Crossobj += 3 * tmpConstantLog;

    return scheme1Crossobj;
}

GRBLinExpr DetailLayoutMILP::addScheme2RowActObjectFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme2RAObj;

    // Iterate over all tensor types
    for (int tensorType = 0; tensorType < 3; ++tensorType) {

        // Iterate over all possible values
        for (int value = 0; value < selLayerInfo.tensorValues[tensorType].size(); ++value) {

            double_t tmpConstant = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[tensorType][value]));

            scheme2RAObj += tmpConstant *
                                layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][tensorType][value];
        }
    }

    // Add final constants
    double_t Scheme2Constant = (static_cast<double_t>(archInfo.dataSize) / archInfo.rowSize) * archInfo.rowActTime;
    scheme2RAObj += 3 * Scheme2Constant;

    return scheme2RAObj;
}

// Pending Implementation
GRBLinExpr DetailLayoutMILP::addScheme2CrossBankObjFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme2CrossObj;

    // Iterate over all potential values of u_oa_1
    for (int value = 0; value < selLayerInfo.tensorValues[computeTensorTypeFC::FC_Out].size(); ++value) {

        // Get the constants
        double_t outputUseConstant = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[computeTensorTypeFC::FC_Out][value]));

        scheme2CrossObj += 2 * layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][computeTensorTypeFC::FC_Out][value] * 
                            outputUseConstant;
    }

    // Get the number of banks used by the layer
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all potential divisior
        for (int divisor = 0; divisor < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++divisor) {
            scheme2CrossObj += 2 * selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[divisor] *
                                    layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][divisor];
        }
    }

    // Get the final constant
    double_t tempCons = (static_cast<double_t>(archInfo.dataSize) * archInfo.crossBankTime) / static_cast<double_t>(archInfo.dataQueueSizeHBM);
    double_t tempConsLog = std::log2(tempCons / selLayerInfo.totalTensorSize[computeTensorTypeFC::FC_Out]);

    scheme2CrossObj += tempConsLog;


    return scheme2CrossObj;
}

GRBLinExpr DetailLayoutMILP::addScheme3RowActObjectFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme3RAObj;

    // Iterate over all workLoad dim
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all divisors
        for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++ div) {
            scheme3RAObj += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
        }
    }

    scheme3RAObj += std::log2(static_cast<double_t>(archInfo.rowActTime));

    return scheme3RAObj;
}

// Pending Implementation
GRBLinExpr DetailLayoutMILP::addScheme3IntraBankObjFC(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme3IntraObj;

    // Iterate over all workLoad
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all divs
        for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3IntraObj += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] * 
                                layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
        }
    }
    
    // Define row level size variable
    int64_t upperBound = selLayerInfo.totalTensorSize[0] + selLayerInfo.totalTensorSize[1] + selLayerInfo.totalTensorSize[2];
    double_t upperBoundLog = std::log2(static_cast<double_t>(upperBound));

    GRBVar rowLevelSizeVar = model.addVar(1, upperBound, 0.0, GRB_CONTINUOUS, "Layer_" + std::to_string(selLayer) +"_Scheme2RowLevevlTensorSize");
    GRBLinExpr rowLevelTensorSizes;
    
    // Iterate over all tensor types
    for (int tensorType = 0; tensorType < 3; ++tensorType) {

        // Iterate over all potential values
        for (int value = 0; value < selLayerInfo.tensorValues[tensorType].size(); ++value) {
            double_t tmpLogValue = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[tensorType][value]));

            rowLevelTensorSizes += tmpLogValue *
                                layerVariables[selLayer].tensorValueOneHotVars[memLevel::RowBuffer][tensorType][value];

            if (tensorType == computeTensorTypeFC::FC_Out) {
                scheme3IntraObj -= tmpLogValue *
                                    layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][tensorType][value];
            }
        }
    }

    model.addConstr(rowLevelSizeVar == rowLevelTensorSizes);

    // Define a new variable to represent the Log of tensor sizes at rowbuffer level
    GRBVar rowLevelSizeLogVar = model.addVar(0, upperBoundLog, 0.0, GRB_CONTINUOUS, "Layer_" + std::to_string(selLayer) +"_Scheme2RowLevevlTensorSizeLog");
    model.addGenConstrLogA(rowLevelSizeVar, rowLevelSizeLogVar, 2.0, "Layer_" + std::to_string(selLayer) +"_Scheme2RowLevevlTensorSizeLog_Cons", GurobiApproximationOptions);

    
    scheme3IntraObj += rowLevelSizeLogVar;
    scheme3IntraObj += std::log2(static_cast<double_t>(archInfo.rowActTime));

    return scheme3IntraObj;
}

// ====================================
//    GRB Objectives PWL + Nrom -- FC
// ====================================

// This part is the same for all schemes with Norm and PWL
GRBLinExpr DetailLayoutMILP::addComputeObjectFCPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr computeObj;

    GRBLinExpr FinalObj;

    // Iterate over all workload dimension
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all divisors
        for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            // Compute Number of Sequence
            computeObj += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] * 
                            layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];

            // Compute #cycles per sequence
            // double_t tmpConstant = selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div];
            double_t tmpConstant = selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div];

            computeObj += (tmpConstant * 
                            layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div]);
        }
    }

    double_t tmpCoEff =  static_cast<double_t>(archInfo.dataSize) / archInfo.dataQueueSizeHBM;

    computeObj += std::log2(tmpCoEff);

    // Scale it
    computeObj += perLayerMacScaleFactorLog[selLayer];

    // Get the original value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[selLayer], std::log2(1.02 * perLayerMacValue[selLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_Log_Compute_Obj");
    model.addConstr(compObjLog == computeObj, "L_" + std::to_string(selLayer) + "_Compute_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[selLayer], 1.02 * perLayerMacValue[selLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_Compute_Obj");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(selLayer) + "_Compute_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[selLayer]) * maxMacScaleFactor;

    return FinalObj;
}

// Add Scheme 1 Row Act Objective with Norm and PWL
GRBLinExpr DetailLayoutMILP::addScheme1RowActObjectFCPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme1RAObj;
    GRBLinExpr FinalObj;

    // Get Total Tensor Size
    int64_t totalTensorSize = 0;
    for (auto tensorSize: selLayerInfo.totalTensorSize) totalTensorSize += (tensorSize * archInfo.dataSize);

    double_t tmpConstant = static_cast<double_t>(totalTensorSize) / archInfo.rowSize;
    double_t tmpConstantLog = std::log2(tmpConstant);

    scheme1RAObj += tmpConstantLog;

    // Iterate over all workLoad
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all divisors
        for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {
            scheme1RAObj -= selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] * 
                                layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    }

    // Add the final log2(T_act)
    double_t tmpConstantLog2 = std::log2(archInfo.rowActTime);

    scheme1RAObj += tmpConstantLog2;

    // Scalse it
    scheme1RAObj += perLayerMacScaleFactorLog[selLayer];

    // Get the original value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[selLayer], std::log2(1.02 * perLayerMacValue[selLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_Log_RA_Obj");
    model.addConstr(compObjLog == scheme1RAObj, "L_" + std::to_string(selLayer) + "_RA_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[selLayer], 1.02 * perLayerMacValue[selLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_RA_Obj");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(selLayer) + "_RA_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[selLayer]) * maxMacScaleFactor;

    return FinalObj;
}

// Add Scheme 1 Cross Bank Loading Objective with Norm and PWL
GRBLinExpr DetailLayoutMILP::addScheme1CrossBankObjFCPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr FinalObj;

    // Add the constants
    double_t tmpConstant = (static_cast<double_t>(archInfo.dataSize) / archInfo.dataQueueSizeHBM) * archInfo.crossBankTime;
    double_t tmpConstantLog = std::log2(tmpConstant);

    // We need to calculate cross bank latency for all three tensor types
    // Iterate over all tensor types
    for (int tensorType = 0; tensorType < 3; ++tensorType) {
        GRBLinExpr scheme1Crossobj;
        
        // Iterate over all potential values
        for (int value = 0; value < selLayerInfo.tensorValues[tensorType].size(); ++value) {

            // Get the constant
            double_t innerConstant = std::log2((static_cast<double_t>(selLayerInfo.tensorValues[tensorType][value])));

            scheme1Crossobj += layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][tensorType][value] *
                                    innerConstant;
        }



        // Iterate over all workLoad Dims
        for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

            // Iterate over all divisors
            for (int div = 0;  div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {
                
                scheme1Crossobj += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                    layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
            }
        }

        scheme1Crossobj += tmpConstantLog;

        // Scalse it
        scheme1Crossobj += perLayerMacScaleFactorLog[selLayer];

        // Get the original value -- exp function
        GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[selLayer], std::log2(1.02 * perLayerMacValue[selLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_Log_CR_Obj_" + TensorTypeNames[tensorType]);
        model.addConstr(compObjLog == scheme1Crossobj, "L_" + std::to_string(selLayer) + "_CR_Log_Cons_" + TensorTypeNames[tensorType]);

        GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[selLayer], 1.02 * perLayerMacValue[selLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_CR_Obj_" + TensorTypeNames[tensorType]);
        model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(selLayer) + "_CR_Cons", GurobiApproximationOptions);

        FinalObj += (compObjOri / perLayerMacScaleFactor[selLayer]) * maxMacScaleFactor;

    }    

    return FinalObj;
}

// Add Scheme 2 Row Act Objective with Norm and PWL
GRBLinExpr DetailLayoutMILP::addScheme2RowActObjectFCPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr FinalObj;

    // Add final constants
    double_t Scheme2Constant = (static_cast<double_t>(archInfo.dataSize) / archInfo.rowSize) * archInfo.rowActTime;

    // Iterate over all tensor types
    for (int tensorType = 0; tensorType < 3; ++tensorType) {
        GRBLinExpr scheme2RAObj;

        // Iterate over all possible values
        for (int value = 0; value < selLayerInfo.tensorValues[tensorType].size(); ++value) {

            double_t tmpConstant = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[tensorType][value]));

            scheme2RAObj += tmpConstant *
                                layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][tensorType][value];
        }

        scheme2RAObj += Scheme2Constant;

        // Scalse it
        scheme2RAObj += perLayerMacScaleFactorLog[selLayer];

        // Get the original value -- exp function
        GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[selLayer], std::log2(1.02 * perLayerMacValue[selLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_Log_RA_Obj_" + TensorTypeNames[tensorType]);
        model.addConstr(compObjLog == scheme2RAObj, "L_" + std::to_string(selLayer) + "_RA_Log_Cons_" + TensorTypeNames[tensorType]);

        GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[selLayer], 1.02 * perLayerMacValue[selLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_RA_Obj_" + TensorTypeNames[tensorType]);
        model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(selLayer) + "_RA_Cons" +  TensorTypeNames[tensorType], GurobiApproximationOptions);

        FinalObj += (compObjOri / perLayerMacScaleFactor[selLayer]) * maxMacScaleFactor;
    }

    return FinalObj;
}

// Add Scheme 2 Cross Bank Loading Objective with Norm and PWL
GRBLinExpr DetailLayoutMILP::addScheme2CrossBankObjFCPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme2CrossObj;
    GRBLinExpr FinalObj;

    // Iterate over all potential values of u_oa_1
    for (int value = 0; value < selLayerInfo.tensorValues[computeTensorTypeFC::FC_Out].size(); ++value) {

        // Get the constants
        double_t outputUseConstant = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[computeTensorTypeFC::FC_Out][value]));

        scheme2CrossObj += 2 * layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][computeTensorTypeFC::FC_Out][value] * 
                            outputUseConstant;
    }

    // Get the number of banks used by the layer
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all potential divisior
        for (int divisor = 0; divisor < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++divisor) {
            scheme2CrossObj += 2 * selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[divisor] *
                                    layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][divisor];
        }
    }

    // Get the final constant
    double_t tempCons = (static_cast<double_t>(archInfo.dataSize) * archInfo.crossBankTime) / static_cast<double_t>(archInfo.dataQueueSizeHBM);
    double_t tempConsLog = std::log2(tempCons / selLayerInfo.totalTensorSize[computeTensorTypeFC::FC_Out]);

    scheme2CrossObj += tempConsLog;

    // Scale it
    scheme2CrossObj += perLayerMacScaleFactorLog[selLayer];

    // Get the original value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[selLayer], std::log2(1.02 * perLayerMacValue[selLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_Log_CR_Obj");
    model.addConstr(compObjLog == scheme2CrossObj, "L_" + std::to_string(selLayer) + "_CR_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[selLayer], 1.02 * perLayerMacValue[selLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_CR_Obj");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(selLayer) + "_CR_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[selLayer]) * maxMacScaleFactor;

    return FinalObj;
}

// Add Scheme 3 Row Act Objective with Norm and PWL 
GRBLinExpr DetailLayoutMILP::addScheme3RowActObjectFCPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr FinalObj;
    GRBLinExpr scheme3RAObj;

    // Iterate over all workLoad dim
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all divisors
        for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++ div) {
            scheme3RAObj += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
        }
    }

    scheme3RAObj += std::log2(static_cast<double_t>(archInfo.rowActTime));

    // Scalse it
    scheme3RAObj += perLayerMacScaleFactorLog[selLayer];

    // Get the original value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[selLayer], std::log2(1.02 * perLayerMacValue[selLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_Log_RA_Obj");
    model.addConstr(compObjLog == scheme3RAObj, "L_" + std::to_string(selLayer) + "_RA_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[selLayer], 1.02 * perLayerMacValue[selLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_RA_Obj");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(selLayer) + "_RA_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[selLayer]) * maxMacScaleFactor;

    return FinalObj;
}

// Add Scheme 3 Intra Bank Loading Objective with Norm and PWL
GRBLinExpr DetailLayoutMILP::addScheme3IntraBankObjFCPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme3IntraObj;
    GRBLinExpr FinalObj;

    // Iterate over all workLoad -- Get number of rows
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all divs
        for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3IntraObj += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] * 
                                layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
        }
    }

    GRBLinExpr rowBufferOutputSize;

    for (int j = 0; j < selLayerInfo.tensorValues[computeTensorTypeFC::FC_Out].size(); ++j) {
        double_t tmpLogValue = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[computeTensorTypeFC::FC_Out][j]));

        rowBufferOutputSize -= tmpLogValue *
                                    layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][computeTensorTypeFC::FC_Out][j];
    }
    
    
    // Iterate over all tensor types
    for (int tensorType = 0; tensorType < 3; ++tensorType) {
        GRBLinExpr RowTensorSize;

        // Iterate over all potential values
        for (int value = 0; value < selLayerInfo.tensorValues[tensorType].size(); ++value) {
            double_t tmpLogValue = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[tensorType][value]));

            RowTensorSize += tmpLogValue *
                                layerVariables[selLayer].tensorValueOneHotVars[memLevel::RowBuffer][tensorType][value];

            
        }

        RowTensorSize += std::log2(static_cast<double_t>(archInfo.rowActTime));
        RowTensorSize += scheme3IntraObj;
        RowTensorSize += rowBufferOutputSize;

        // Scalse it
        RowTensorSize += perLayerMacScaleFactorLog[selLayer];

        // Get the original value
        GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[selLayer], std::log2(1.02 * perLayerMacValue[selLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_Log_IB_Obj");
        model.addConstr(compObjLog == RowTensorSize, "L_" + std::to_string(selLayer) + "_IB_Log_Cons");

        GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[selLayer], 1.02 * perLayerMacValue[selLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_IB_Obj");
        model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(selLayer) + "_IB_Cons", GurobiApproximationOptions);

        FinalObj += (compObjOri / perLayerMacScaleFactor[selLayer]) * maxMacScaleFactor;

        
    }

    return FinalObj;
}


#endif  // PIMOPT_GUROBI_NOT_INSTALLED