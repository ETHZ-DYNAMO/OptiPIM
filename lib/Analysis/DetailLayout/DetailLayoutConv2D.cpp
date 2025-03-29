//===----------------------------------------------------------------------===//
//
// This file implements functions for Conv2D Layer in the DetailLayout.
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
GRBLinExpr DetailLayoutMILP::setupConv2DGRB(const LayerInfo& selLayerInstance, const int32_t& selLayerIndex) {
    //
    GRBLinExpr outputExpr;
    
    //
    // Step 1: Build the storing structure for a single layer
    //
    buildConv2DLayerConstants(selLayerInstance);

    //
    // Step 2: Create layer specific variables
    //
    LayerGRBVariables tmpLayerVariables;
    layerVariables[selLayerIndex] = tmpLayerVariables;

    // Step 2.1: Add all loop bound variables
    addLoopBoundVarsConv2D(selLayerIndex);

    // Step 2.2: Add tensor size variables, for W, OA and IA
    addTensorSizeVarsConv2D(selLayerIndex);

    //
    // Step 3: Add Constraints
    //
    // Step 3.1: Add Loop Bound Constraint
    addLoopBoundConsConv2D(selLayerIndex);

    // Step 3.2: Add WorkLoad Constraint
    addWorkLoadConsConv2D(selLayerIndex, PerformanceKnobs.workLoadTolerance);

    // Step 3.3: Add Number of Banks Constraint
    addNumBankConsConv2D(selLayerIndex);

    // Step 3.4: Add Cartesian Product Constraint
    addTensorValueConsConv2D(selLayerIndex);
    addCartesianProductConsConv2D(selLayerIndex, PerformanceKnobs.cartesianProTolerance);

    //
    // Step 3.5: Add Scheme Specific Constraints
    //
    // Check the specified layout scheme
    if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_1) {
        addScheme1ConsConv2D(selLayerIndex);
    } else if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_2) {
        addScheme2ConsConv2D(selLayerIndex);
    } else {
        addScheme3ConsConv2D(selLayerIndex);
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
        computeObj += addComputeObjectConv2D(selLayerIndex);

        if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_1) {
            rowActObj += addScheme1RowActObjectConv2D(selLayerIndex);
            crossBankObj += addScheme1CrossBankObjConv2D(selLayerIndex);
        } else if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_2) {

            rowActObj += addScheme2RowActObjectConv2D(selLayerIndex);
            crossBankObj += addScheme2CrossBankObjConv2D(selLayerIndex);
        } else {
            rowActObj += addScheme3RowActObjectConv2D(selLayerIndex);
            // crossBankObj += addScheme3CrossBankObjConv2D(selLayerIndex);
            if (PerformanceKnobs.intraBankWeight > 0) {
                intraBankObj += addScheme3IntraBankObjConv2D(selLayerIndex);
            }
        }
    } else if (ObjApproScheme == ObjApproximationScheme::Manual_norm) {
        computeObj += (addComputeObjectConv2D(selLayerIndex) + maxMacScaleFactorLog); 
        if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_1) {
            rowActObj += (addScheme1RowActObjectConv2D(selLayerIndex) + maxMacScaleFactorLog);
            crossBankObj += (addScheme1CrossBankObjConv2D(selLayerIndex) + maxMacScaleFactorLog);
        } else if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_2) {
            rowActObj += (addScheme2RowActObjectConv2D(selLayerIndex) + maxMacScaleFactorLog);
            crossBankObj += (addScheme2CrossBankObjConv2D(selLayerIndex) + maxMacScaleFactorLog);
        } else {
            rowActObj += (addScheme3RowActObjectConv2D(selLayerIndex) + maxMacScaleFactorLog);
            // crossBankObj += addScheme3CrossBankObjConv2D(selLayerIndex);
            if (PerformanceKnobs.intraBankWeight > 0) {
                intraBankObj += (addScheme3IntraBankObjConv2D(selLayerIndex) + maxMacScaleFactorLog);
            }
        }
    } else if (ObjApproScheme == ObjApproximationScheme::Gurobi_exp) {
        computeObj += addComputeObjectConv2DPWLNorm(selLayerIndex);

        if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_1) {
            rowActObj += addScheme1RowActObjectConv2DPWLNorm(selLayerIndex);
            crossBankObj += addScheme1CrossBankObjConv2DPWLNorm(selLayerIndex);
        } else if (layerConstInfo[selLayerIndex].layoutScheme == LayoutScheme::Scheme_2) {
            rowActObj += addScheme2RowActObjectConv2DPWLNorm(selLayerIndex);
            crossBankObj += addScheme2CrossBankObjConv2DPWLNorm(selLayerIndex);
        } else {
            rowActObj += addScheme3RowActObjectConv2DPWLNorm(selLayerIndex);
            // crossBankObj += addScheme3CrossBankObjConv2D(selLayerIndex);
            if (PerformanceKnobs.intraBankWeight > 0) {
                intraBankObj += addScheme3IntraBankObjConv2DPWLNorm(selLayerIndex);
            }
        }
    }
    

    // [Output] Update LayerGroup Objective function
    outputExpr += PerformanceKnobs.compWeight * computeObj;
    outputExpr += PerformanceKnobs.rowActWeight * rowActObj;
    outputExpr += PerformanceKnobs.crossBankWeight * crossBankObj;
    outputExpr += PerformanceKnobs.intraBankWeight * intraBankObj;

    return outputExpr;
}


// ====================================
//          Constants Creation
// ====================================

void DetailLayoutMILP::buildConv2DLayerConstants(const LayerInfo& selLayerIns) {
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
    for (int i = 0; i < NUM_DIM_CONV2D; i++) {
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
    //    0 : W;
    //    1 : OA;
    for (int tensor_type = 0; tensor_type < 2; ++tensor_type) {
        std::vector<Set<int64_t>> tempSets;
        
        // Related sets are identified by the relation matrix
        for (int dimIndex = 0; dimIndex < NUM_DIM_CONV2D; ++dimIndex) {
            if (Conv2DRelationMatrix[dimIndex][tensor_type] > 0) {
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

    // Input Array needs special processing for building the Cartesian product set
    // Input Array sizes are related to 6 workload dimensions, ordered as {N, K, P, Q, R, S}
    std::vector<Set<int64_t>> IASets;
    for (int dimIndex = 0; dimIndex < NUM_DIM_CONV2D; ++dimIndex) {
        if (Conv2DRelationMatrix[dimIndex][2] > 0) {
            IASets.push_back(tmp_const_info.workLoadDimMap[dimIndex].divisorsVec);
        }
    }

    // Get the corresponding IA Cartesian Product
    CartesianProduct<int64_t> IACaPr = calCartesian(IASets);

    // Get the all potentail tensor values from the obtained CaPr
    std::vector<int64_t> IATensorValues = calConv2DPIAValue(IACaPr, selLayerIns.stride, selLayerIns.dilation);

    // Store the information related to IA
    tmp_const_info.tensorCartSets.push_back(IACaPr);
    tmp_const_info.tensorValues.push_back(IATensorValues);

    // Calculate top mem level tensor sizes
    // W and OA
    for (int tensor_type = 0; tensor_type < 2; ++tensor_type) {
        int64_t tmpResult = 1;

        for (int dimIndex = 0; dimIndex < NUM_DIM_CONV2D; ++dimIndex) {
            if (Conv2DRelationMatrix[dimIndex][tensor_type] > 0) {
                tmpResult *= selLayerIns.workLoadDimVec[dimIndex];
            }
        }

        tmp_const_info.totalTensorSize.push_back(tmpResult);
    }

    // Calculate size for IA
    int64_t topMemCN = selLayerIns.workLoadDimVec[workLoadDimensionConv2D::C] * selLayerIns.workLoadDimVec[workLoadDimensionConv2D::N];
    int64_t topMemW = selLayerIns.stride[0] * (selLayerIns.workLoadDimVec[workLoadDimensionConv2D::Q] - 1) +
                        selLayerIns.dilation[0] * (selLayerIns.workLoadDimVec[workLoadDimensionConv2D::S] - 1) + 1;
    int64_t topMemH = selLayerIns.stride[1] * (selLayerIns.workLoadDimVec[workLoadDimensionConv2D::P] - 1) +
                        selLayerIns.dilation[1] * (selLayerIns.workLoadDimVec[workLoadDimensionConv2D::R] - 1) + 1;
    tmp_const_info.totalTensorSize.push_back(topMemCN * topMemH * topMemW);
    
    // [Final] : Store all calcualted constants
    // Construct the layerConstInfo map
    layerConstInfo[tmp_layer_id] = tmp_const_info;

    //! Testing
    // tmp_const_info.printDetail();
}

std::vector<int64_t> DetailLayoutMILP::calConv2DPIAValue(const CartesianProduct<int64_t>& selCaPr, 
                                           const std::vector<int64_t>& layerStride, const std::vector<int64_t>& layerDialation) {
    //! Dimension check is omitted for this version, Jiantao 11/06/2024
    // For now, we assume that stride is the same for width and height
    std::vector<int64_t> result;

    for (size_t i = 0; i < selCaPr.size(); ++i) {
        int64_t tmpResult = 1;

        // Input size is calculated as: C * N * W * H,
        // Where, W = Stride * (Q - 1) + Dilation * (S - 1) + 1
        // H = Stride * (P - 1) + Dilation * (R - 1) + 1
        tmpResult *= (selCaPr[i][0] * selCaPr[i][3]); // N * C

        int64_t tmpWidth  = layerStride[0] * (selCaPr[i][2] - 1) + layerDialation[0] * (selCaPr[i][5] - 1) + 1;
        int64_t tmpHeight = layerStride[1] * (selCaPr[i][1] - 1) + layerDialation[1] * (selCaPr[i][4] - 1) + 1;

        tmpResult *= (tmpWidth * tmpHeight);

        result.push_back(tmpResult);
    }

    return result;
}

// =============================================
//        GRB Variables Creation -- Conv2D
// =============================================

void DetailLayoutMILP::addLoopBoundVarsConv2D(const int32_t& selLayer) {
    
    // Create a Gurobi variable of the given name and type for all loop bound in resulting nested loop
    auto createVar = [&](const std::string &name, char type) {
        return model.addVar(0, 1, 0.0, type, name);
    };

    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Iterate through all workload dimensions
    std::vector<std::vector<std::vector<GRBVar>>> tmpWorkLoadVars;
    for (int i = 0; i < NUM_DIM_CONV2D; ++i) {
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

void DetailLayoutMILP::addTensorSizeVarsConv2D(const int32_t& selLayer) {
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
//       GRB Constraints -- Conv2D 
// ====================================

void DetailLayoutMILP::addLoopBoundConsConv2D(const int32_t& selLayer) {
    // TODO: Maybe this should be merged with the variable creation part
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];
    
    // Iterate through all workload dim
    for (int i = 0; i < NUM_DIM_CONV2D; ++i) {
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

void DetailLayoutMILP::addWorkLoadConsConv2D(const int32_t& selLayer, const double_t& tolerance) {
    // TODO: This can be merged with the loopbound constraint defintion
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Iterate through all workload dim
    for (int i = 0; i < NUM_DIM_CONV2D; ++i) {
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

void DetailLayoutMILP::addNumBankConsConv2D(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // For this constraint, we only care about loop bounds at bank level
    // Iterate through all workload dims
    GRBLinExpr bankNumConstraint;
    for (int i = 0; i < NUM_DIM_CONV2D; ++i) {
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

void DetailLayoutMILP::addTensorValueConsConv2D(const int32_t& selLayer) {
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

void DetailLayoutMILP::addCartesianProductConsConv2D(const int32_t& selLayer, const double_t& tolerance) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // We need to make sure that e(m,v,h) aligns with x(d,m,t), which means the product of one workload type below a certain memory level
    // which is selected by e(m,v,h) equals to the product of that specific loop bounds below that mem level (determined by x(d,m,t))
    // We relax these constraints with an upper and lower bound according to the specified tolerance.
    // Iterate over all tensor types -- v
    for (int tensorType = 0; tensorType < 3; ++tensorType) {

        // Iterate over all workload dim -- d
        for (int workLoadDim = 0; workLoadDim < NUM_DIM_CONV2D; ++workLoadDim) {
            // If this workLoadDim is related to the tensorType
            if (Conv2DRelationMatrix[workLoadDim][tensorType] > 0) {

                // Iterate over all memory levels -- m
                for (int memLevel = 0; memLevel < memLevel::Bank; ++memLevel) {
                    // Construct the left-hand side of the constraint
                    GRBLinExpr LHSConstraint;

                    for (int oneHotIndex = 0; oneHotIndex < selLayerInfo.tensorValues[tensorType].size(); ++oneHotIndex) {
                        // Get the selected workload dimension specific value
                        double_t workLoadFactorLog = std::log2(static_cast<double_t>(selLayerInfo.tensorCartSets[tensorType][oneHotIndex][workLoadToTensorCaPr[workLoadDim][tensorType]]));

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
                    // Add the constraints to the model
                    if (memoryAlloc == MemoryAllocationStrategy::Exclusive) {
                        // llvm::outs() << "Relaxed\n";
                        model.addConstr(LHSConstraint <= RHSConstraint + tolerance, tmpConNameUpper);
                        model.addConstr(LHSConstraint >= RHSConstraint - tolerance, tmpConNameLower);
                    } else if (memoryAlloc == MemoryAllocationStrategy::Combined) {
                        model.addConstr(LHSConstraint == RHSConstraint, tmpConNameLower);
                    }
                    // model.addConstr(LHSConstraint == RHSConstraint, tmpConNameLower);
                }
            }

        }
    }
}

void DetailLayoutMILP::addScheme1ConsConv2D(const int32_t& selLayer) {
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
    for (int i = 0; i < NUM_DIM_CONV2D; ++i) {

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

void DetailLayoutMILP::addScheme2ConsConv2D(const int32_t& selLayer) {
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

void DetailLayoutMILP::addScheme3ConsConv2D(const int32_t& selLayer) {
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
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

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
//       GRB Objectives -- Conv2D
// ====================================

GRBLinExpr DetailLayoutMILP::addComputeObjectConv2D(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr computeObj;

    // Iterate over all workload dimension
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

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

GRBLinExpr DetailLayoutMILP::addScheme1RowActObjectConv2D(const int32_t& selLayer) {
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
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

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

GRBLinExpr DetailLayoutMILP::addScheme1CrossBankObjConv2D(const int32_t& selLayer) {
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
            double_t innerConstant = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[tensorType][value]));

            scheme1Crossobj += layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][tensorType][value] *
                                    innerConstant;
        }



        // Iterate over all workLoad Dims
        for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

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

GRBLinExpr DetailLayoutMILP::addScheme2RowActObjectConv2D(const int32_t& selLayer){
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
GRBLinExpr DetailLayoutMILP::addScheme2CrossBankObjConv2D(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme2CrossObj;

    // Iterate over all potential values of u_oa_1
    for (int value = 0; value < selLayerInfo.tensorValues[computeTensorType::OA].size(); ++value) {

        // Get the constants
        double_t outputUseConstant = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[computeTensorType::OA][value]));

        scheme2CrossObj += 2 * layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][computeTensorType::OA][value] * 
                            outputUseConstant;
    }

    // Get the number of banks used by the layer
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all potential divisior
        for (int divisor = 0; divisor < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++divisor) {
            scheme2CrossObj += 2 * selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[divisor] *
                                    layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][divisor];
        }
    }

    // Get the final constant
    double_t tempCons = (static_cast<double_t>(archInfo.dataSize) * archInfo.crossBankTime) / static_cast<double_t>(archInfo.dataQueueSizeHBM);
    double_t tempConsLog = std::log2(tempCons / selLayerInfo.totalTensorSize[computeTensorType::OA]);

    scheme2CrossObj += tempConsLog;


    return scheme2CrossObj;
}

// Add Scheme 3 Row Act Objective
GRBLinExpr DetailLayoutMILP::addScheme3RowActObjectConv2D(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme3RAObj;

    // Iterate over all workLoad dim
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all divisors
        for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++ div) {
            scheme3RAObj += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
        }
    }

    scheme3RAObj += std::log2(static_cast<double_t>(archInfo.rowActTime));

    return scheme3RAObj;
}

// Add Scheme 3 Intra Bank Loading Objective
GRBLinExpr DetailLayoutMILP::addScheme3IntraBankObjConv2D(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme3IntraObj;

    // Iterate over all workLoad
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

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

            if (tensorType == computeTensorType::OA) {
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

// Add Scheme 3 Cross Bank Loading Objective -- Pending implementation
GRBLinExpr DetailLayoutMILP::addScheme3CrossBankObjConv2D(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme3CrossObj;


    return scheme3CrossObj;
}

// ====================================
// GRB Objectives Norm + PWL -- Conv2D
// ====================================

// This part is the same for all schemes
GRBLinExpr DetailLayoutMILP::addComputeObjectConv2DPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr computeObj;

    // Final Obj
    GRBLinExpr FinalObj;

    // Iterate over all workload dimension
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

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

// Add Scheme 1 Row Act Objective after normalization
GRBLinExpr DetailLayoutMILP::addScheme1RowActObjectConv2DPWLNorm(const int32_t& selLayer) {
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
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

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

// Add Scheme 1 Cross Bank Loading Objective after normalization
GRBLinExpr DetailLayoutMILP::addScheme1CrossBankObjConv2DPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr FinalObj;

    // Define the constants
    double_t tmpConstant = (static_cast<double_t>(archInfo.dataSize) / archInfo.dataQueueSizeHBM) * archInfo.crossBankTime;
    double_t tmpConstantLog = std::log2(tmpConstant);

    // We need to calculate cross bank latency for all three tensor types
    // Iterate over all tensor types
    for (int tensorType = 0; tensorType < 3; ++tensorType) {
        GRBLinExpr scheme1Crossobj;

        // Iterate over all potential values
        for (int value = 0; value < selLayerInfo.tensorValues[tensorType].size(); ++value) {

            // Get the constant
            double_t innerConstant = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[tensorType][value]));

            scheme1Crossobj += layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][tensorType][value] *
                                    innerConstant;
        }

        // Iterate over all workLoad Dims
        for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

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

// Add Scheme 2 Row Act Objective after normalization
GRBLinExpr DetailLayoutMILP::addScheme2RowActObjectConv2DPWLNorm(const int32_t& selLayer) {
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

// Add Scheme 2 Cross Bank Loading Objective after normalization
GRBLinExpr DetailLayoutMILP::addScheme2CrossBankObjConv2DPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme2CrossObj;
    GRBLinExpr FinalObj;

    // Iterate over all potential values of u_oa_1
    for (int value = 0; value < selLayerInfo.tensorValues[computeTensorType::OA].size(); ++value) {

        // Get the constants
        double_t outputUseConstant = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[computeTensorType::OA][value]));

        scheme2CrossObj += 2 * layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][computeTensorType::OA][value] * 
                            outputUseConstant;
    }

    // Get the number of banks used by the layer
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all potential divisior
        for (int divisor = 0; divisor < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++divisor) {
            scheme2CrossObj += 2 * selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[divisor] *
                                    layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][divisor];
        }
    }

    // Get the final constant
    double_t tempCons = (static_cast<double_t>(archInfo.dataSize) * archInfo.crossBankTime) / static_cast<double_t>(archInfo.dataQueueSizeHBM);
    double_t tempConsLog = std::log2(tempCons / selLayerInfo.totalTensorSize[computeTensorType::OA]);

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

// Add Scheme 3 Row Act Objective after normalization
GRBLinExpr DetailLayoutMILP::addScheme3RowActObjectConv2DPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme3RAObj;
    GRBLinExpr FinalObj;

    // Iterate over all workLoad dim
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

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

// Add Scheme 3 Intra Bank Loading Objective after normalization
GRBLinExpr DetailLayoutMILP::addScheme3IntraBankObjConv2DPWLNorm(const int32_t& selLayer) {
    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];

    // Create the GRB expression
    GRBLinExpr scheme3IntraObj;
    GRBLinExpr FinalObj;

    // Iterate over all workLoad -- Get number of rows
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all divs
        for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3IntraObj += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] * 
                                layerVariables[selLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
        }
    }

    GRBLinExpr rowBufferOutputSize;

    for (int j = 0; j < selLayerInfo.tensorValues[computeTensorType::OA].size(); ++j) {
        double_t tmpLogValue = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[computeTensorType::OA][j]));

        rowBufferOutputSize -= tmpLogValue *
                                    layerVariables[selLayer].tensorValueOneHotVars[memLevel::Row][computeTensorType::OA][j];
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
        GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[selLayer], std::log2(1.02 * perLayerMacValue[selLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_Log_IB_Obj_" + std::to_string(tensorType));
        model.addConstr(compObjLog == RowTensorSize, "L_" + std::to_string(selLayer) + "_IB_Log_Cons_" + std::to_string(tensorType));

        GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[selLayer], 1.02 * perLayerMacValue[selLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(selLayer) + "_IB_Obj" + std::to_string(tensorType));
        model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(selLayer) + "_IB_Cons" + std::to_string(tensorType), GurobiApproximationOptions);

        FinalObj += (compObjOri / perLayerMacScaleFactor[selLayer]) * maxMacScaleFactor;
    }

    return FinalObj;
}


// ===========================================
//       GRB Objectives -- Conv2D Cross Layer
// ===========================================

// Add Scheme 1 Cross Layer Tensor Dependency Objective -- Conv2D -> Conv2D
double_t DetailLayoutMILP::addScheme1CrossLayerObjCC(const int32_t& curLayer, const int32_t& nextLayer) {
    // Get the needed layer const
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    //
    // GRBLinExpr scheme1CrossLayerObj;

    int64_t curOutputSize = curLayerInfo.totalTensorSize[computeTensorType::OA] * archInfo.dataSize;
    int64_t nextInputSize = nextLayerInfo.totalTensorSize[computeTensorType::IA] * archInfo.dataSize;

    double_t totalTensor = static_cast<double_t>(curOutputSize) + nextInputSize;
    double_t reverseBandwidth = (1.0 / archInfo.HBMBandwidth);


    return totalTensor * reverseBandwidth;
}


// Add Scheme 2 Cross Layer Tensor Dependency Objective
GRBLinExpr DetailLayoutMILP::addScheme2CrossLayerObjCC(const int32_t& curLayer, const int32_t& nextLayer) {
    // Get the needed layer const
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    //
    GRBLinExpr scheme2CrossLayerObj;

    // Part 1: Calculat the output size of the current layer
    // Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < curLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme2CrossLayerObj += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    } 

    // Get output size per bank
    // Iterate over all potential values for output size of Row level
    for (int value = 0; value < curLayerInfo.tensorValues[computeTensorType::OA].size(); ++value) {
        
        double_t tmpConst = std::log2(static_cast<double_t>(curLayerInfo.tensorValues[computeTensorType::OA][value]));
        
        scheme2CrossLayerObj += tmpConst *
                                    layerVariables[curLayer].tensorValueOneHotVars[memLevel::Row][computeTensorType::OA][value];
    }

    scheme2CrossLayerObj += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    // Part II :: Calculate the input size of next Layer
    // Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < nextLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme2CrossLayerObj += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    } 

    // Get input size per bank
    // Iterate over all potential values for output size of Row level
    for (int value = 0; value < nextLayerInfo.tensorValues.size(); ++value) {
        
        double_t tmpConst = std::log2(static_cast<double_t>(nextLayerInfo.tensorValues[computeTensorType::IA][value]));
        
        scheme2CrossLayerObj += tmpConst *
                                    layerVariables[nextLayer].tensorValueOneHotVars[memLevel::Row][computeTensorType::IA][value];
    }

    scheme2CrossLayerObj += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    return scheme2CrossLayerObj;
}

// Add Scheme 3 Cross Layer Tensor Dependency Objective
GRBLinExpr DetailLayoutMILP::addScheme3CrossLayerObjCC(const int32_t& curLayer, const int32_t& nextLayer) {
    // Get the needed layer const
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    //
    GRBLinExpr scheme3CrossLayerObj;

    // Part 1: Calculat the output size of the current layer
    // 1.1 :Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < curLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3CrossLayerObj += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];

            // 1.2: Get the number of outputs
            if (Conv2DRelationMatrix[workLoad][computeTensorType::OA] > 0) {
                scheme3CrossLayerObj += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                            layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
            }

            // 1.3: Get the size of the output
            scheme3CrossLayerObj += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div];
        }
    }

    scheme3CrossLayerObj += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    // Part II :: Calculate the input size of next Layer
    // 2.1 Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < nextLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3CrossLayerObj += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];

            // 2.2: Get the number of inputs
            if (Conv2DRelationMatrix[workLoad][computeTensorType::IA] > 0) {
                scheme3CrossLayerObj += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                            layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
            }

            // 2.3: Get the size of the input
            scheme3CrossLayerObj += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div];
        }
    } 

    scheme3CrossLayerObj += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    return scheme3CrossLayerObj;
}

#endif  // PIMOPT_GUROBI_NOT_INSTALLED
