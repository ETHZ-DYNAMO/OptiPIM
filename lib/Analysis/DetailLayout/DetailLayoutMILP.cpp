/*
██████╗ ██╗███╗   ███╗ ██████╗ ██████╗ ████████╗
██╔══██╗██║████╗ ████║██╔═══██╗██╔══██╗╚══██╔══╝
██████╔╝██║██╔████╔██║██║   ██║██████╔╝   ██║   
██╔═══╝ ██║██║╚██╔╝██║██║   ██║██╔═══╝    ██║   
██║     ██║██║ ╚═╝ ██║╚██████╔╝██║        ██║   
╚═╝     ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝        ╚═╝                                                 
*/

//===----------------------------------------------------------------------===//
//
// This file implements main interfaces for the DetailLayoutPass.
//
//===----------------------------------------------------------------------===//

#include "pimopt/Analysis/DetailLayout/DetailLayoutMILP.h"

// Check the existence of Gurobi library
#ifndef PIMOPT_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

// Add Source Definition for Con2D related functions
// #include "DetailLayoutConv2D.cpp"

using namespace mlir;
using namespace pim;

DetailLayoutMILP::DetailLayoutMILP(GRBEnv &env, std::vector<int32_t> &layerGroup, 
                    std::map<int32_t, LayerInfo> &layersInfoDB,
                     const std::string& logPath, const std::string& resultPath,
                     const std::string& _archPath, const std::string& _knobPath,
                     const int32_t& dataLayoutScheme, const int32_t& memoryAllocScheme,
                     const int32_t& objApproMethod, const int32_t& layerGroupID): model(GRBModel(env)) {

    int32_t layer_group_id = layersInfoDB[layerGroup[0]].layerGroupId;

    // SET THE LAYOUT,  MEMORY ALLOCATION SCHEME AND OBJ APPROXIMATION METHOD
    selLayoutScheme = dataLayoutScheme;

    memoryAlloc = memoryAllocScheme;

    ObjApproScheme = objApproMethod;

    // Final Objective Expression -- for the whole layer group
    GRBLinExpr FinalObj;

    // set arch and knob path 
    archPath = _archPath;
    knobPath = _knobPath;

    // Step 0: Parse Knob values and arch info, passed in from command line
    llvm::outs() << "[debug] arch and knob paths: " << archPath << " " << knobPath << "\n";
    readKnobValues(knobPath);
    readArchInfo(archPath);

    // Step 0: Build Scaling Factor
    buildScalingFactor(layerGroup, layersInfoDB);

    // Total Number of banks
    int32_t layerGroupBanks = layersInfoDB[layerGroup[0]].numBanks;

    // Iterate through all layers in the layer group vector
    // and construct all needed constants
    //      1. Divs lists for all workload dimensions
    //      2. Cartesian product sets for W, OA and IA
    //      3. Possible values from Cartesian Product Sets
    for (auto layerIndex : layerGroup) {
        // Get Layer type
        int32_t selLayerType = layersInfoDB[layerIndex].layerType;

        // Set up layer-specific GRB constants, variables, constraints and objectives based on Layer type
        if (selLayerType == layerTypeAll::CONV2D) {
            FinalObj += setupConv2DGRB(layersInfoDB[layerIndex], layerIndex);
        } else if (selLayerType == layerTypeAll::FC) {
            FinalObj += setupFCGRB(layersInfoDB[layerIndex], layerIndex);
        }
    }

    //
    //  Add Comined Memory Constraints
    //
    if (memoryAlloc == MemoryAllocationStrategy::Combined) {
        if (selLayoutScheme == LayoutScheme::Scheme_1) {
            // Scheme 1 Selected
            addComMemScheme1Cons(layerGroup, layerGroupBanks);
        } else if (selLayoutScheme == LayoutScheme::Scheme_2) {
            addComMemScheme2Cons(layerGroup, layerGroupBanks);
        } else {
            addComMemScheme3Cons(layerGroup, layerGroupBanks);
        }
    }

    //
    //  Add Cross Layer objectives
    //
    // For now we assume a linear dependency between consecutive layers
    // TODO: Need to parse the connection information from the linalg.mlir file
    GRBLinExpr crossLayerObj;
    for (int preLayerIndex = 0; preLayerIndex < layerGroup.size() - 1; ++preLayerIndex) {
        // Get current and next Layer index
        int32_t curLayerIndex = layerGroup[preLayerIndex];
        int32_t nextLayerIndex = layerGroup[preLayerIndex + 1];

        // Get the Layer type
        int32_t curLayerType = layerConstInfo[preLayerIndex].layerType;
        int32_t nextLayerType = layerConstInfo[preLayerIndex + 1].layerType;

        if (ObjApproScheme == ObjApproximationScheme::Manual_log) {
            // Add cross Layer dependency objective based on the scheme
            if (layerConstInfo[curLayerIndex].layoutScheme == LayoutScheme::Scheme_1) {
                if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme1Conv2DOut(curLayerIndex) + addScheme1Conv2DIn(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme1Conv2DOut(curLayerIndex) + addScheme1FCIn(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme1FCOut(curLayerIndex) + addScheme1FCIn(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme1FCOut(curLayerIndex) + addScheme1Conv2DIn(nextLayerIndex);
                }
                // crossLayerObj += addScheme1CrossLayerObjCC(curLayerIndex, nextLayerIndex);
            } else if (layerConstInfo[curLayerIndex].layoutScheme == LayoutScheme::Scheme_2) {
                if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme2Conv2DOut(curLayerIndex) + addScheme2Conv2DIn(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme2Conv2DOut(curLayerIndex) + addScheme2FCIn(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme2FCOut(curLayerIndex) + addScheme2FCIn(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme2FCOut(curLayerIndex) + addScheme2Conv2DIn(nextLayerIndex);
                }
                // crossLayerObj += addScheme2CrossLayerObjCC(curLayerIndex, nextLayerIndex);
            } else {
                if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme3Conv2DOut(curLayerIndex) + addScheme3Conv2DIn(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme3Conv2DOut(curLayerIndex) + addScheme3FCIn(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme3FCOut(curLayerIndex) + addScheme3FCIn(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme3FCOut(curLayerIndex) + addScheme3Conv2DIn(nextLayerIndex);
                }
                // crossLayerObj += addScheme3CrossLayerObjCC(curLayerIndex, nextLayerIndex);
            }
        } else if (ObjApproScheme == ObjApproximationScheme::Manual_norm) {
            // Add cross Layer dependency objective based on the scheme
            if (layerConstInfo[curLayerIndex].layoutScheme == LayoutScheme::Scheme_1) {
                if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme1Conv2DOut(curLayerIndex) + addScheme1Conv2DIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                } else if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme1Conv2DOut(curLayerIndex) + addScheme1FCIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme1FCOut(curLayerIndex) + addScheme1FCIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme1FCOut(curLayerIndex) + addScheme1Conv2DIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                }
                // crossLayerObj += addScheme1CrossLayerObjCC(curLayerIndex, nextLayerIndex);
            } else if (layerConstInfo[curLayerIndex].layoutScheme == LayoutScheme::Scheme_2) {
                if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme2Conv2DOut(curLayerIndex) + addScheme2Conv2DIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                } else if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme2Conv2DOut(curLayerIndex) + addScheme2FCIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme2FCOut(curLayerIndex) + addScheme2FCIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme2FCOut(curLayerIndex) + addScheme2Conv2DIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                }
                // crossLayerObj += addScheme2CrossLayerObjCC(curLayerIndex, nextLayerIndex);
            } else {
                if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme3Conv2DOut(curLayerIndex) + addScheme3Conv2DIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                } else if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme3Conv2DOut(curLayerIndex) + addScheme3FCIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme3FCOut(curLayerIndex) + addScheme3FCIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme3FCOut(curLayerIndex) + addScheme3Conv2DIn(nextLayerIndex) + (2 * maxMacScaleFactorLog);
                }
                // crossLayerObj += addScheme3CrossLayerObjCC(curLayerIndex, nextLayerIndex);
            }
        } else if (ObjApproScheme == ObjApproximationScheme::Gurobi_exp) {
            // Add cross Layer dependency objective based on the scheme
            if (layerConstInfo[curLayerIndex].layoutScheme == LayoutScheme::Scheme_1) {
                if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme1Conv2DOutPWLNorm(curLayerIndex) + addScheme1Conv2DInPWLNorm(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme1Conv2DOutPWLNorm(curLayerIndex) + addScheme1FCInPWLNorm(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme1FCOutPWLNorm(curLayerIndex) + addScheme1FCInPWLNorm(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme1FCOutPWLNorm(curLayerIndex) + addScheme1Conv2DInPWLNorm(nextLayerIndex);
                }
                // crossLayerObj += addScheme1CrossLayerObjCC(curLayerIndex, nextLayerIndex);
            } else if (layerConstInfo[curLayerIndex].layoutScheme == LayoutScheme::Scheme_2) {
                if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme2Conv2DOutPWLNorm(curLayerIndex) + addScheme2Conv2DInPWLNorm(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme2Conv2DOutPWLNorm(curLayerIndex) + addScheme2FCInPWLNorm(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme2FCOutPWLNorm(curLayerIndex) + addScheme2FCInPWLNorm(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme2FCOutPWLNorm(curLayerIndex) + addScheme2Conv2DInPWLNorm(nextLayerIndex);
                }
                // crossLayerObj += addScheme2CrossLayerObjCC(curLayerIndex, nextLayerIndex);
            } else {
                if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme3Conv2DOutPWLNorm(curLayerIndex) + addScheme3Conv2DInPWLNorm(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::CONV2D && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme3Conv2DOutPWLNorm(curLayerIndex) + addScheme3FCInPWLNorm(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::FC) {
                    crossLayerObj += addScheme3FCOutPWLNorm(curLayerIndex) + addScheme3FCInPWLNorm(nextLayerIndex);
                } else if (curLayerType == layerTypeAll::FC && nextLayerType == layerTypeAll::CONV2D) {
                    crossLayerObj += addScheme3FCOutPWLNorm(curLayerIndex) + addScheme3Conv2DInPWLNorm(nextLayerIndex);
                }
                // crossLayerObj += addScheme3CrossLayerObjCC(curLayerIndex, nextLayerIndex);
            }
        }
    }

    // [FINAL] Update LayerGroup Objective function -- Cross Layer parts
    FinalObj += PerformanceKnobs.crossLayerWeight * crossLayerObj;

    // Sets the MILP Objective
    try {
        model.setObjective(FinalObj, GRB_MINIMIZE);
    } catch (GRBException& e) {
        llvm::errs() << "[Objective Function] Gurobi Exception occurred: " << e.getMessage() << "\n";
    }

    // Debug, write out the model
    if (logPath != "") {
        try {
            writeMLIPModel(model, logPath, layer_group_id);
        } catch (GRBException& e) {
            llvm::errs() << "[Model Logging] Gurobi Exception occurred: " << e.getMessage() << "\n";
        }
    }

    //
    // Optimize the model
    //
    try {
        model.optimize();

        if (selLayoutScheme == LayoutScheme::Scheme_2 && memoryAlloc == MemoryAllocationStrategy::Combined) {
            // Relax the constraint
            if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
                model.feasRelax(0, false, true, true);
                model.optimize();
            }
        }

    } catch (GRBException& e) {
        llvm::errs() << "[Optimization] Gurobi Exception occurred: " << e.getMessage() << "\n";
    }
    
    // Check the status of the optimized model
    if (int status = model.get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
        llvm::errs() << "Optimization Failed, No Feasible solution found, Please increase the num_banks value in the .linalg.mlir file\n";

        // TODO: Need to make the logging more flexible
        std::string resultTracePath = resultPath + "/Group_" + std::to_string(layerGroupID) + ".txt";
        std::ofstream layerGroupOutFile = std::ofstream(resultTracePath);

        // Print out results -- Extract All Results
        for (auto layerIndex : layerGroup) {
            layerConstInfo[layerIndex].printDetail();

            if (layerConstInfo[layerIndex].layerType == layerTypeAll::CONV2D) {
                // Print the results to terminal
                printConv2DMILPResult(layerIndex);

                // Dump the results to a file
                layerGroupOutFile << traceConv2D(layerIndex, layersInfoDB[layerIndex]);
            } else if (layerConstInfo[layerIndex].layerType == layerTypeAll::FC) {
                // Print the results to terminal
                printFCMILPResult(layerIndex);

                // Dump the results to a file
                layerGroupOutFile << traceFC(layerIndex, layersInfoDB[layerIndex]);
            }
        }

    } else {
        // Create output file
        // TODO: Need to make the logging more flexible
        std::string resultTracePath = resultPath + "/Group_" + std::to_string(layerGroupID) + ".txt";
        std::ofstream layerGroupOutFile = std::ofstream(resultTracePath);

        // Print out results -- Extract All Results
        for (auto layerIndex : layerGroup) {
            layerConstInfo[layerIndex].printDetail();

            if (layerConstInfo[layerIndex].layerType == layerTypeAll::CONV2D) {
                // Print the results to terminal
                printConv2DMILPResult(layerIndex);

                // Dump the results to a file
                layerGroupOutFile << traceConv2D(layerIndex, layersInfoDB[layerIndex]);
            } else if (layerConstInfo[layerIndex].layerType == layerTypeAll::FC) {
                // Print the results to terminal
                printFCMILPResult(layerIndex);

                // Dump the results to a file
                layerGroupOutFile << traceFC(layerIndex, layersInfoDB[layerIndex]);
            }
        }
    }
}

// =================================================
//     GRB Objectives -- Cross Layer/Different Type
// =================================================

double_t DetailLayoutMILP::addScheme1Conv2DOut(const int32_t& curLayer) {
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    int64_t curOutputSize = curLayerInfo.totalTensorSize[computeTensorType::OA] * archInfo.dataSize;

    double_t result = std::log2(static_cast<double_t>(curOutputSize) / archInfo.HBMBandwidth);

    return result;
}

double_t DetailLayoutMILP::addScheme1FCOut(const int32_t& curLayer) {
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    int64_t curOutputSize = curLayerInfo.totalTensorSize[computeTensorTypeFC::FC_Out] * archInfo.dataSize;

    double_t result = std::log2(static_cast<double_t>(curOutputSize) / archInfo.HBMBandwidth);

    return result;
}

GRBLinExpr DetailLayoutMILP::addScheme2Conv2DOut(const int32_t& curLayer) {
    // Get the needed layer const
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    //
    GRBLinExpr scheme2Conv2DOut;

    // Part 1: Calculat the output size of the current layer
    // Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < curLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme2Conv2DOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    } 

    // Get output size per bank
    // Iterate over all potential values for output size of Row level
    for (int value = 0; value < curLayerInfo.tensorValues.size(); ++value) {
        
        double_t tmpConst = std::log2(static_cast<double_t>(curLayerInfo.tensorValues[computeTensorType::OA][value]));
        
        scheme2Conv2DOut += tmpConst *
                                    layerVariables[curLayer].tensorValueOneHotVars[memLevel::Row][computeTensorType::OA][value];
    }

    scheme2Conv2DOut += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    return scheme2Conv2DOut;
}

GRBLinExpr DetailLayoutMILP::addScheme2FCOut(const int32_t& curLayer) {
    // Get the needed layer const
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    //
    GRBLinExpr scheme2FCOut;

    // Part 1: Calculat the output size of the current layer
    // Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < curLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme2FCOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    } 

    // Get output size per bank
    // Iterate over all potential values for output size of Row level
    for (int value = 0; value < curLayerInfo.tensorValues.size(); ++value) {
        
        double_t tmpConst = std::log2(static_cast<double_t>(curLayerInfo.tensorValues[computeTensorTypeFC::FC_Out][value]));
        
        scheme2FCOut += tmpConst *
                                    layerVariables[curLayer].tensorValueOneHotVars[memLevel::Row][computeTensorTypeFC::FC_Out][value];
    }

    scheme2FCOut += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    return scheme2FCOut;
}

GRBLinExpr DetailLayoutMILP::addScheme3Conv2DOut(const int32_t& curLayer) {
    // Get the needed layer const
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    //
    GRBLinExpr scheme3Conv2DOut;

    // Part 1: Calculat the output size of the current layer
    // 1.1 :Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < curLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3Conv2DOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];

            // 1.2: Get the number of outputs
            if (Conv2DRelationMatrix[workLoad][computeTensorType::OA] > 0) {
                scheme3Conv2DOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                            layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
            }

            // 1.3: Get the size of the output
            scheme3Conv2DOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div];
        }
    }

    scheme3Conv2DOut += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    return scheme3Conv2DOut;
}

GRBLinExpr DetailLayoutMILP::addScheme3FCOut(const int32_t& curLayer) {
    // Get the needed layer const
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    //
    GRBLinExpr scheme3FCOut;

    // Part 1: Calculat the output size of the current layer
    // 1.1 :Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < curLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3FCOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];

            // 1.2: Get the number of outputs
            if (FCRelationMatrix[workLoad][computeTensorTypeFC::FC_Out] > 0) {
                scheme3FCOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                            layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
            }

            // 1.3: Get the size of the output
            scheme3FCOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div];
        }
    }

    scheme3FCOut += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    return scheme3FCOut;
}

// Inout Objective
double_t DetailLayoutMILP::addScheme1Conv2DIn(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    int64_t nextInputSize = nextLayerInfo.totalTensorSize[computeTensorType::IA] * archInfo.dataSize;

    double_t result = static_cast<double_t>(nextInputSize) / archInfo.HBMBandwidth;

    return result;
}

double_t DetailLayoutMILP::addScheme1FCIn(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    int64_t nextInputSize = nextLayerInfo.totalTensorSize[computeTensorTypeFC::Mat_A] * archInfo.dataSize;

    double_t result = static_cast<double_t>(nextInputSize) / archInfo.HBMBandwidth;

    return result;
}

GRBLinExpr DetailLayoutMILP::addScheme2Conv2DIn(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    //
    GRBLinExpr scheme2Conv2DIn;

    // Part II :: Calculate the input size of next Layer
    // Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < nextLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme2Conv2DIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    } 

    // Get input size per bank
    // Iterate over all potential values for output size of Row level
    for (int value = 0; value < nextLayerInfo.tensorValues.size(); ++value) {
        
        double_t tmpConst = std::log2(static_cast<double_t>(nextLayerInfo.tensorValues[computeTensorType::IA][value]));
        
        scheme2Conv2DIn += tmpConst *
                                    layerVariables[nextLayer].tensorValueOneHotVars[memLevel::Row][computeTensorType::IA][value];
    }

    scheme2Conv2DIn += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    return scheme2Conv2DIn;
}

GRBLinExpr DetailLayoutMILP::addScheme2FCIn(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    //
    GRBLinExpr scheme2FCIn;

    // Part II :: Calculate the input size of next Layer
    // Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < nextLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme2FCIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    } 

    // Get input size per bank
    // Iterate over all potential values for output size of Row level
    for (int value = 0; value < nextLayerInfo.tensorValues.size(); ++value) {
        
        double_t tmpConst = std::log2(static_cast<double_t>(nextLayerInfo.tensorValues[computeTensorTypeFC::Mat_A][value]));
        
        scheme2FCIn += tmpConst *
                            layerVariables[nextLayer].tensorValueOneHotVars[memLevel::Row][computeTensorTypeFC::Mat_A][value];
    }

    scheme2FCIn += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    return scheme2FCIn;
}

GRBLinExpr DetailLayoutMILP::addScheme3Conv2DIn(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    //
    GRBLinExpr scheme3Conv2DIn;

    // Part II :: Calculate the input size of next Layer
    // 2.1 Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < nextLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3Conv2DIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];

            // 2.2: Get the number of inputs
            if (Conv2DRelationMatrix[workLoad][computeTensorType::IA] > 0) {
                scheme3Conv2DIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                            layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
            }

            // 2.3: Get the size of the input
            scheme3Conv2DIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div];
        }
    } 

    scheme3Conv2DIn += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    return scheme3Conv2DIn;

}

GRBLinExpr DetailLayoutMILP::addScheme3FCIn(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    //
    GRBLinExpr scheme3FCIn;

    // Part II :: Calculate the input size of next Layer
    // 2.1 Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < nextLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3FCIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];

            // 2.2: Get the number of inputs
            if (FCRelationMatrix[workLoad][computeTensorTypeFC::Mat_A] > 0) {
                scheme3FCIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                            layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
            }

            // 2.3: Get the size of the input
            scheme3FCIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div];
        }
    } 

    scheme3FCIn += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    return scheme3FCIn;
}


// ==============================================
//   GRB Objectives -- Cross Layer (PWL + Norm)
// ==============================================
// Output Objective
double_t DetailLayoutMILP::addScheme1Conv2DOutPWLNorm(const int32_t& curLayer) {
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    int64_t curOutputSize = curLayerInfo.totalTensorSize[computeTensorType::OA] * archInfo.dataSize;

    double_t result = static_cast<double_t>(curOutputSize) / archInfo.HBMBandwidth;

    return result * maxMacScaleFactor;
}

double_t DetailLayoutMILP::addScheme1FCOutPWLNorm(const int32_t& curLayer) {
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    int64_t curOutputSize = curLayerInfo.totalTensorSize[computeTensorTypeFC::FC_Out] * archInfo.dataSize;

    double_t result = static_cast<double_t>(curOutputSize) / archInfo.HBMBandwidth;

    return result * maxMacScaleFactor;
}

GRBLinExpr DetailLayoutMILP::addScheme2Conv2DOutPWLNorm(const int32_t& curLayer) {
    // Get the needed layer const
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    //
    GRBLinExpr scheme2Conv2DOut;
    GRBLinExpr FinalObj;

    // Part 1: Calculat the output size of the current layer
    // Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < curLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme2Conv2DOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    } 

    // Get output size per bank
    // Iterate over all potential values for output size of Row level
    for (int value = 0; value < curLayerInfo.tensorValues[computeTensorType::OA].size(); ++value) {
        
        double_t tmpConst = std::log2(static_cast<double_t>(curLayerInfo.tensorValues[computeTensorType::OA][value]));
        
        scheme2Conv2DOut += tmpConst *
                                    layerVariables[curLayer].tensorValueOneHotVars[memLevel::Row][computeTensorType::OA][value];
    }

    scheme2Conv2DOut += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    // Scale it
    scheme2Conv2DOut += perLayerMacScaleFactorLog[curLayer];

    // Get the original Value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[curLayer], std::log2(1.02 * perLayerMacValue[curLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(curLayer) + "_Scheme2_Output_Log");
    model.addConstr(compObjLog == scheme2Conv2DOut, "L_" + std::to_string(curLayer) + "_Scheme2_Output_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[curLayer], 1.02 * perLayerMacValue[curLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(curLayer) + "_Scheme2_Output");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(curLayer) + "_Scheme2_Output_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[curLayer]) * maxMacScaleFactor;

    return FinalObj;
}

GRBLinExpr DetailLayoutMILP::addScheme2FCOutPWLNorm(const int32_t& curLayer) {
    // Get the needed layer const
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    //
    GRBLinExpr scheme2Conv2DOut;
    GRBLinExpr FinalObj;

    // Part 1: Calculat the output size of the current layer
    // Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < curLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme2Conv2DOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    } 

    // Get output size per bank
    // Iterate over all potential values for output size of Row level
    for (int value = 0; value < curLayerInfo.tensorValues[computeTensorTypeFC::FC_Out].size(); ++value) {
        
        double_t tmpConst = std::log2(static_cast<double_t>(curLayerInfo.tensorValues[computeTensorTypeFC::FC_Out][value]));
        
        scheme2Conv2DOut += tmpConst *
                                    layerVariables[curLayer].tensorValueOneHotVars[memLevel::Row][computeTensorTypeFC::FC_Out][value];
    }

    scheme2Conv2DOut += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    // Scale it
    scheme2Conv2DOut += perLayerMacScaleFactorLog[curLayer];

    // Get the original Value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[curLayer], std::log2(1.02 * perLayerMacValue[curLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(curLayer) + "_Scheme2_Output_Log");
    model.addConstr(compObjLog == scheme2Conv2DOut, "L_" + std::to_string(curLayer) + "_Scheme2_Output_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[curLayer], 1.02 * perLayerMacValue[curLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(curLayer) + "_Scheme2_Output");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(curLayer) + "_Scheme2_Output_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[curLayer]) * maxMacScaleFactor;

    return FinalObj;
}

GRBLinExpr DetailLayoutMILP::addScheme3Conv2DOutPWLNorm(const int32_t& curLayer) {
    // Get the needed layer const
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    //
    GRBLinExpr scheme3Conv2DOut;
    GRBLinExpr FinalObj;

    // Part 1: Calculat the output size of the current layer
    // 1.1 :Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < curLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3Conv2DOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];

            // 1.2: Get the number of outputs
            if (Conv2DRelationMatrix[workLoad][computeTensorType::OA] > 0) {
                scheme3Conv2DOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                            layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
            }

            // 1.3: Get the size of the output
            scheme3Conv2DOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div];
        }
    }

    scheme3Conv2DOut += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    // Scale it
    scheme3Conv2DOut += perLayerMacScaleFactorLog[curLayer];

    // Get the original Value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[curLayer], std::log2(1.02 * perLayerMacValue[curLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(curLayer) + "_Scheme3_Output_Log");
    model.addConstr(compObjLog == scheme3Conv2DOut, "L_" + std::to_string(curLayer) + "_Scheme2_Output_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[curLayer], 1.02 * perLayerMacValue[curLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(curLayer) + "_Scheme3_Output");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(curLayer) + "_Scheme2_Output_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[curLayer]) * maxMacScaleFactor;


    return FinalObj;
}

GRBLinExpr DetailLayoutMILP::addScheme3FCOutPWLNorm(const int32_t& curLayer) {
    // Get the needed layer const
    ConstInfo& curLayerInfo = layerConstInfo[curLayer];

    //
    GRBLinExpr scheme3FCOut;
    GRBLinExpr FinalObj;

    // Part 1: Calculat the output size of the current layer
    // 1.1 :Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < curLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3FCOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];

            // 1.2: Get the number of outputs
            if (FCRelationMatrix[workLoad][computeTensorTypeFC::FC_Out] > 0) {
                scheme3FCOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                            layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
            }

            // 1.3: Get the size of the output
            scheme3FCOut += curLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[curLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div];
        }
    }

    scheme3FCOut += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    // Scale it
    scheme3FCOut += perLayerMacScaleFactorLog[curLayer];

    // Get the original Value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[curLayer], std::log2(1.02 * perLayerMacValue[curLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(curLayer) + "_Scheme3_Output_Log");
    model.addConstr(compObjLog == scheme3FCOut, "L_" + std::to_string(curLayer) + "_Scheme2_Output_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[curLayer], 1.02 * perLayerMacValue[curLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(curLayer) + "_Scheme3_Output");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(curLayer) + "_Scheme2_Output_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[curLayer]) * maxMacScaleFactor;

    return FinalObj;
}

// Inout Objective
double_t DetailLayoutMILP::addScheme1Conv2DInPWLNorm(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    int64_t nextInputSize = nextLayerInfo.totalTensorSize[computeTensorType::IA] * archInfo.dataSize;

    double_t result = static_cast<double_t>(nextInputSize) / archInfo.HBMBandwidth;

    return result * maxMacScaleFactor;
}

double_t DetailLayoutMILP::addScheme1FCInPWLNorm(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    int64_t nextInputSize = nextLayerInfo.totalTensorSize[computeTensorTypeFC::Mat_A] * archInfo.dataSize;

    nextInputSize += nextLayerInfo.totalTensorSize[computeTensorTypeFC::Mat_B] * archInfo.dataSize;

    double_t result = static_cast<double_t>(nextInputSize) / archInfo.HBMBandwidth;

    return result * maxMacScaleFactor;
}

GRBLinExpr DetailLayoutMILP::addScheme2Conv2DInPWLNorm(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    //
    GRBLinExpr scheme2Conv2DIn;
    GRBLinExpr FinalObj;

    // Part II :: Calculate the input size of next Layer
    // Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < nextLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme2Conv2DIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    } 

    // Get input size per bank
    // Iterate over all potential values for output size of Row level
    for (int value = 0; value < nextLayerInfo.tensorValues[computeTensorType::IA].size(); ++value) {
        
        double_t tmpConst = std::log2(static_cast<double_t>(nextLayerInfo.tensorValues[computeTensorType::IA][value]));
        
        scheme2Conv2DIn += tmpConst *
                                    layerVariables[nextLayer].tensorValueOneHotVars[memLevel::Row][computeTensorType::IA][value];
    }

    scheme2Conv2DIn += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    // Scale it
    scheme2Conv2DIn += perLayerMacScaleFactorLog[nextLayer];

    // Get the original Value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[nextLayer], std::log2(1.02 * perLayerMacValue[nextLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(nextLayer) + "_Scheme2_Input_Log");
    model.addConstr(compObjLog == scheme2Conv2DIn, "L_" + std::to_string(nextLayer) + "_Scheme2_Input_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[nextLayer], 1.02 * perLayerMacValue[nextLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(nextLayer) + "_Scheme2_Input");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(nextLayer) + "_Scheme2_Input_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[nextLayer]) * maxMacScaleFactor;

    return FinalObj;
}

GRBLinExpr DetailLayoutMILP::addScheme2FCInPWLNorm(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    //
    GRBLinExpr scheme2Conv2DIn;
    GRBLinExpr FinalObj;

    // Part II :: Calculate the input size of next Layer
    // Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < nextLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme2Conv2DIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
        }
    } 

    // Get input size per bank
    // Iterate over all potential values for output size of Row level
    for (int value = 0; value < nextLayerInfo.tensorValues[computeTensorTypeFC::Mat_A].size(); ++value) {
        
        double_t tmpConst = std::log2(static_cast<double_t>(nextLayerInfo.tensorValues[computeTensorTypeFC::Mat_A][value]));
        
        scheme2Conv2DIn += tmpConst *
                                    layerVariables[nextLayer].tensorValueOneHotVars[memLevel::Row][computeTensorTypeFC::Mat_A][value];
    }

    // Iterate over all potential values for output size of Row level
    for (int value = 0; value < nextLayerInfo.tensorValues[computeTensorTypeFC::Mat_B].size(); ++value) {
        
        double_t tmpConst = std::log2(static_cast<double_t>(nextLayerInfo.tensorValues[computeTensorTypeFC::Mat_B][value]));
        
        scheme2Conv2DIn += tmpConst *
                                    layerVariables[nextLayer].tensorValueOneHotVars[memLevel::Row][computeTensorTypeFC::Mat_B][value];
    }

    scheme2Conv2DIn += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    // Scale it
    scheme2Conv2DIn += perLayerMacScaleFactorLog[nextLayer];

    // Get the original Value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[nextLayer], std::log2(1.02 * perLayerMacValue[nextLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(nextLayer) + "_Scheme2_Input_Log");
    model.addConstr(compObjLog == scheme2Conv2DIn, "L_" + std::to_string(nextLayer) + "_Scheme2_Input_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[nextLayer], 1.02 * perLayerMacValue[nextLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(nextLayer) + "_Scheme2_Input");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(nextLayer) + "_Scheme2_Input_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[nextLayer]) * maxMacScaleFactor;

    return FinalObj;
}

GRBLinExpr DetailLayoutMILP::addScheme3Conv2DInPWLNorm(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    //
    GRBLinExpr scheme3Conv2DIn;
    GRBLinExpr FinalObj;

    // Part II :: Calculate the input size of next Layer
    // 2.1 Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < nextLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3Conv2DIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];

            // 2.2: Get the number of inputs
            if (Conv2DRelationMatrix[workLoad][computeTensorType::IA] > 0) {
                scheme3Conv2DIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                            layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
            }

            // 2.3: Get the size of the input
            scheme3Conv2DIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div];
        }
    } 

    scheme3Conv2DIn += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    // Scale it
    scheme3Conv2DIn += perLayerMacScaleFactorLog[nextLayer];

    // Get the original Value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[nextLayer], std::log2(1.02 * perLayerMacValue[nextLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(nextLayer) + "_Scheme3_Input_Log");
    model.addConstr(compObjLog == scheme3Conv2DIn, "L_" + std::to_string(nextLayer) + "_Scheme3_Input_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[nextLayer], 1.02 * perLayerMacValue[nextLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(nextLayer) + "_Scheme3_Input");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(nextLayer) + "_Scheme3_Input_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[nextLayer]) * maxMacScaleFactor;


    return FinalObj;
}

GRBLinExpr DetailLayoutMILP::addScheme3FCInPWLNorm(const int32_t& nextLayer) {
    ConstInfo& nextLayerInfo = layerConstInfo[nextLayer];

    //
    GRBLinExpr scheme3FCIn;
    GRBLinExpr FinalObj;

    // Part II :: Calculate the input size of next Layer
    // 2.1 Iterate over all workLoad -- Calculate number of banks
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {

        // Iterate over all diviosrs
        for (int div = 0; div < nextLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {

            scheme3FCIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];

            // 2.2: Get the number of inputs
            if (FCRelationMatrix[workLoad][computeTensorTypeFC::Mat_A] > 0) {
                scheme3FCIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                            layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
            }

            // 2.3: Get the size of the input
            scheme3FCIn += nextLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div] *
                                        layerVariables[nextLayer].memLoopBoundOneHotVars[workLoad][memLevel::RowBuffer][div];
        }
    } 

    scheme3FCIn += std::log2((static_cast<double_t>(archInfo.dataSize) / archInfo.HBMBandwidth));

    // Scale it
    scheme3FCIn += perLayerMacScaleFactorLog[nextLayer];

    // Get the original Value
    GRBVar compObjLog = model.addVar(perLayerMacScaleFactorLog[nextLayer], std::log2(1.02 * perLayerMacValue[nextLayer]), 0.0, GRB_CONTINUOUS, "L_" + std::to_string(nextLayer) + "_Scheme3_Input_Log");
    model.addConstr(compObjLog == scheme3FCIn, "L_" + std::to_string(nextLayer) + "_Scheme3_Input_Log_Cons");

    GRBVar compObjOri = model.addVar(perLayerMacScaleFactor[nextLayer], 1.02 * perLayerMacValue[nextLayer], 0.0, GRB_CONTINUOUS, "L_" + std::to_string(nextLayer) + "_Scheme3_Input");
    model.addGenConstrExpA(compObjLog, compObjOri, 2.0, "L_" + std::to_string(nextLayer) + "_Scheme3_Input_Cons", GurobiApproximationOptions);

    FinalObj += (compObjOri / perLayerMacScaleFactor[nextLayer]) * maxMacScaleFactor;

    return FinalObj;
}

// ================================================
//          Combined Memory Constraints
// ================================================
void DetailLayoutMILP::addComMemScheme1Cons(std::vector<int32_t> &layerGroup, const int64_t& numBanks) {
    // This is actually very simple, it should be satisfied automatically
    int64_t allLayerTensorSize = 0;

    for (auto layerIndex: layerGroup) {
        for (int i = 0; i < 3; ++i) {
            allLayerTensorSize += (layerConstInfo[layerIndex].totalTensorSize[i] * archInfo.dataSize);

            //! Testing
            llvm::outs() << allLayerTensorSize << "\n";
        }
        //! Testing
        layerConstInfo[layerIndex].printDetail();
    }

    int64_t totalBankSize = static_cast<int64_t>(numBanks) * archInfo.bankSize;

    if (allLayerTensorSize > totalBankSize) {
        llvm::outs() << "[ERROR] Num Banks: " << numBanks << "\n";
        llvm::outs() << "[ERROR] Bank Size: " << archInfo.bankSize << "\n";
        llvm::outs() << "[ERROR] Total Bank Size: " << numBanks * archInfo.bankSize << "\n";
        llvm::outs() << "[ERROR] Total Tensor Size: " << allLayerTensorSize << "\n";
        llvm::outs() << "[ERROR] Combined memory size constraint for Scheme can't be satisfied, allocate more banks to the bank group\n";
        exit(-1);
    }
}

void DetailLayoutMILP::addComMemScheme2Cons(std::vector<int32_t> &layerGroup, const int64_t& num_banks) {
    // We need to do piecewise linearization for this constraint
    GRBQuadExpr FinalCons;

    // Iterate overall all layers and update the constraints based on layer types
    for (auto layerIndex: layerGroup) {
        // Get the layer type
        int32_t selLayerType = layerConstInfo[layerIndex].layerType;
        ConstInfo& selLayerInfo = layerConstInfo[layerIndex];
        int32_t numDimensions = 0;

        // Check the layer type
        if (selLayerType == layerTypeAll::CONV2D) {
            numDimensions = NUM_DIM_CONV2D;
        } else if (selLayerType == layerTypeAll::FC) {
            numDimensions = NUM_DIM_FC;
        }

        selLayerInfo.printDetail();

        // We first calculate the log format of the tensor in one layer
        GRBLinExpr numBanksLog;
        // Iterate over all workload dimension
        for (int workLoad = 0; workLoad < numDimensions; ++workLoad) {

            // Iterate over all divisors
            for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {
                numBanksLog += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div]
                                * layerVariables[layerIndex].memLoopBoundOneHotVars[workLoad][memLevel::Bank][div];
            }
        }

        // Get the original bank value
        GRBVar numBanksLogVar = model.addVar(0, std::log2(static_cast<double_t>(num_banks)), 0.0, GRB_CONTINUOUS, "Layer_" + std::to_string(layerIndex) + "_originalBankValueLog");
        model.addConstr(numBanksLogVar == numBanksLog);
        GRBVar numBanksOriVar = model.addVar(0, num_banks, 0.0, GRB_CONTINUOUS, "Layer_" + std::to_string(layerIndex) + "_originalBankValue");
        model.addGenConstrExpA(numBanksLogVar, numBanksOriVar, 2.0, "Layer_" + std::to_string(layerIndex) + "_originalBankValueCons", "FuncPieces=-2 FuncPieceError=0.00002");

        // Get the tensor sizes
        GRBLinExpr tmpTensorSizeRow;

        for (int tensor = 0; tensor < 3; ++tensor) {
            for (int value = 0; value < selLayerInfo.tensorValues[tensor].size(); ++value) {
                // double_t potentialValue = std::log2(static_cast<double_t>(selLayerInfo.tensorValues[tensor][value]));

                tmpTensorSizeRow += (layerVariables[layerIndex].tensorValueOneHotVars[memLevel::Row][tensor][value] *
                                    selLayerInfo.tensorValues[tensor][value] * archInfo.dataSize);
            }
        }

        // Get the tensorSize limit
        double_t tensorSizeUpper = maxMacNum * archInfo.dataSize;
        GRBVar rowTensorSizeVar = model.addVar(0, tensorSizeUpper, 0.0, GRB_CONTINUOUS, "Layer_" + std::to_string(layerIndex) + "_rowTensorSizeVar");
        model.addConstr(rowTensorSizeVar == tmpTensorSizeRow);

        FinalCons += rowTensorSizeVar * numBanksOriVar;
    }

    // Add Scheme 2 constraint
    try {
        model.addQConstr(FinalCons <= num_banks * static_cast<int64_t>(archInfo.bankSize), "Scheme_2_CominedMemory_Constraint");
    } catch (GRBException& e) {
        llvm::errs() << "[Model Logging] Gurobi Exception occurred: " << e.getMessage() << "\n";
    }
    
}

void DetailLayoutMILP::addComMemScheme3Cons(std::vector<int32_t> &layerGroup, const int64_t& numBanks) {
    // We need to do piecewise linearization for this constraint
    GRBLinExpr FinalCons;
    double_t maxNumRowsPerBank = static_cast<double_t>(archInfo.bankSize) / archInfo.rowSize;
    double_t LogMaxNumRows = std::log2(maxNumRowsPerBank);

    // Iterate over all layers
    for (auto layerIndex: layerGroup) {
        // Get the layer type
        int32_t selLayerType = layerConstInfo[layerIndex].layerType;
        ConstInfo& selLayerInfo = layerConstInfo[layerIndex];
        int32_t numDimensions = 0;

        // Check the layer type
        if (selLayerType == layerTypeAll::CONV2D) {
            numDimensions = NUM_DIM_CONV2D;
        } else if (selLayerType == layerTypeAll::FC) {
            numDimensions = NUM_DIM_FC;
        }

        // Get the logarithmic format of number of rows
        GRBLinExpr numRows;

        // Iterate over all workload dimension
        for (int workLoad = 0; workLoad < numDimensions; ++workLoad) {

            // Iterate over all divisors
            for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {
                numRows += selLayerInfo.workLoadDimMap[workLoad].divisorsLogVec[div]
                                * layerVariables[layerIndex].memLoopBoundOneHotVars[workLoad][memLevel::Row][div];
            }
        }

        // Get the original value through exp function

        GRBVar numRowsLogVars = model.addVar(0, LogMaxNumRows, 0.0, GRB_CONTINUOUS, "Layer_" + std::to_string(layerIndex) + "_Scheme3NumRows_Log");
        model.addConstr(numRowsLogVars == numRows);

        GRBVar numRowsVar = model.addVar(1, maxNumRowsPerBank, 0.0, GRB_CONTINUOUS, "Layer_" + std::to_string(layerIndex) + "_Scheme3NumRows");
        model.addGenConstrExpA(numRowsLogVars, numRowsVar, 2.0, "Layer_" + std::to_string(layerIndex) + "_Scheme3NumRows_Cons", GurobiApproximationOptions);

        FinalCons += numRowsVar;
    }

    // Add Scheme 3 Constraint
    model.addConstr(FinalCons <= maxNumRowsPerBank * numBanks, "Scheme_3_CominedMemory_Constraint");
}


// ====================================
// Functions for Constants Construction
// ====================================
std::vector<int64_t> DetailLayoutMILP::getAllDivisors(const int32_t& dimension) {
    // We assume the input dimension is positive
    std::vector<int64_t> tmp_divisor;

    // Calculate divisors up to the square root of n
    for (int i = 1; i <= std::sqrt(dimension); i++) {
        if (dimension % i == 0) {
            tmp_divisor.push_back(i);

            if (i != (dimension / i)) {
                tmp_divisor.push_back(dimension / i);
            }
        }
    }

    // Sort the obtained list in ascending order
    std::sort(tmp_divisor.begin(), tmp_divisor.end());

    return tmp_divisor;
}

std::vector<int64_t> DetailLayoutMILP::calPTensorValue(const CartesianProduct<int64_t>& selCaPr) {
    std::vector<int64_t> result;

    for (size_t i = 0; i < selCaPr.size(); ++i) {
        int64_t tmpResult = 1;

        for (size_t j = 0; j < selCaPr[i].size(); ++j) {
            tmpResult *= selCaPr[i][j];
        }

        result.push_back(tmpResult);
    }

    return result;
}

void DetailLayoutMILP::buildScalingFactor(std::vector<int32_t> &layerGroup, std::map<int32_t, LayerInfo> &layersInfoDB) {
    double_t max_Mac_Number = 1.0;

    // Iterate over all layers to build the scaling factor
    for (auto layer: layerGroup) {
        int32_t iterationNum = 0;

        if (layersInfoDB[layer].layerType == layerTypeAll::CONV2D) {
            iterationNum = NUM_DIM_CONV2D;
        } else if (layersInfoDB[layer].layerType == layerTypeAll::FC) {
            iterationNum = NUM_DIM_FC;
        }

        double_t layerMac = 1.0;

        // Conv2D layer
        for (int i = 0; i < iterationNum; ++i) {
            layerMac *= layersInfoDB[layer].workLoadDimVec[i];
        }

        if (layerMac > max_Mac_Number) max_Mac_Number = layerMac;

        perLayerMacValue[layer] = layerMac;

        double_t tmpLayerScaling = static_cast<double_t>(ScaleBase) / (1.02 * layerMac);
        perLayerMacScaleFactor[layer] = tmpLayerScaling;

        double_t tmpLayerScalingLog = std::log2(tmpLayerScaling);
        perLayerMacScaleFactorLog[layer] = tmpLayerScalingLog;
    }

    maxMacNum = max_Mac_Number;
    maxMacScaleFactor = static_cast<double_t>(ScaleBase) / (1.02 * max_Mac_Number);
    maxMacScaleFactorLog = std::log2(maxMacScaleFactor);
}

// ====================================
//            Helper Functions
// ====================================
template<typename T>
void DetailLayoutMILP::printCartesianProduct(const CartesianProduct<T>& selSet) {
    llvm::outs() << "{\n";
    for (size_t i = 0; i < selSet.size(); ++i) {
        llvm::outs() << "\t(";
        for (size_t j = 0; j < selSet[i].size(); ++j) {
            llvm::outs() << selSet[i][j];
            if (j != selSet[i].size() - 1) llvm::outs() << ", ";
        }
        
        llvm::outs() << ")"; 
        if (i != selSet.size() - 1) llvm::outs() << ",\n";
    }

    llvm::outs() << "\n}\n";
}

void DetailLayoutMILP::writeMLIPModel(GRBModel& model, const std::string& logPath, const int32_t& layerGroupID) {
    if (!logPath.empty())
        model.write(logPath);
}

void DetailLayoutMILP::writeMLIPSol(GRBModel& model, const std::string& logPath, const int32_t& layerGroupID) {
    if (!logPath.empty())
        model.write(logPath + "LayerGroup_" + std::to_string(layerGroupID) + "_solutions.json");
}

void DetailLayoutMILP::printConv2DMILPResult(const int32_t& selLayer) {
    // Get the needed layer const and variables
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];
    LayerGRBVariables& selLayerVars = layerVariables[selLayer];

    //
    llvm::outs() << "\n\n[OPTIMIZED LAYER RESULTS]\n";
    llvm::outs() << "Layer ID: " << selLayerInfo.layerId << "; Type: Conv2D\n";

    // Iterate through all mem levels
    for (int memLevel = 0; memLevel < NUM_MEM_LEVEL; ++memLevel) {

        llvm::outs() << "\t[" << MemLevelNames[memLevel] << "]:\n";

        // Iterate through all Loop Bound Encoding variables
        for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {
            
            llvm::outs() << "\t\t" << WorkLoadDimSimNames[workLoad] << ": ";

            int32_t counter = 0;

            // Iterate through all possible divisors
            for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {
                // Get Variable value
                double tmpVariableValue = selLayerVars.memLoopBoundOneHotVars[workLoad][memLevel][div].get(GRB_DoubleAttr_X);

                if (tmpVariableValue > 0) {
                    counter++;
                    if (counter < 2) {
                        llvm::outs() << selLayerInfo.workLoadDimMap[workLoad].divisorsVec[div] << "\n"; 
                    } else {
                        llvm::outs() << "\t\t\t[ERROR] More than 1 value available for " << WorkLoadDimSimNames[workLoad] << ": " << selLayerInfo.workLoadDimMap[workLoad].divisorsVec[div] << "; Variable Value: " << std::to_string(tmpVariableValue) << "\n";
                    }
                }
            }
        }

        // Print tensor sizes
        if (memLevel < 2) {
            // Iterate through all tensor Types
            for (int tensorType = 0; tensorType < 3; ++tensorType) {

                llvm::outs() << "\t\t" << TensorTypeNames[tensorType] << " Size: ";
                // Iterate through all potential values

                for (int value = 0; value < selLayerInfo.tensorValues[tensorType].size(); ++value) {
                    // Get Variable value
                    double tmpVariableValueT = selLayerVars.tensorValueOneHotVars[memLevel][tensorType][value].get(GRB_DoubleAttr_X);

                    if (tmpVariableValueT > 0) {
                        llvm::outs() << selLayerInfo.tensorValues[tensorType][value] << "\n";
                    }
                }
            }
        }
    }
    
}

void DetailLayoutMILP::printFCMILPResult(const int32_t& selLayer) {
    // Get the needed layer const and variables
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];
    LayerGRBVariables& selLayerVars = layerVariables[selLayer];

    //
    llvm::outs() << "\n\n[OPTIMIZED LAYER RESULTS]\n";
    llvm::outs() << "Layer ID: " << selLayerInfo.layerId << "; Type: FC\n";

    // Iterate through all mem levels
    for (int memLevel = 0; memLevel < NUM_MEM_LEVEL; ++memLevel) {

        llvm::outs() << "\t[" << MemLevelNames[memLevel] << "]:\n";

        // Iterate through all Loop Bound Encoding variables
        for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {
            
            llvm::outs() << "\t\t" << WorkLoadDimSimNamesFC[workLoad] << ": ";

            // Iterate through all possible divisors
            for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {
                // Get Variable value
                double tmpVariableValue = selLayerVars.memLoopBoundOneHotVars[workLoad][memLevel][div].get(GRB_DoubleAttr_X);

                if (tmpVariableValue > 0) {
                    llvm::outs() << selLayerInfo.workLoadDimMap[workLoad].divisorsVec[div] << "\n"; 
                }
            }
        }

        // Print tensor sizes
        if (memLevel < 2) {
            // Iterate through all tensor Types
            for (int tensorType = 0; tensorType < 3; ++tensorType) {

                llvm::outs() << "\t\t" << TensorTypeNamesFC[tensorType] << " Size: ";
                // Iterate through all potential values

                for (int value = 0; value < selLayerInfo.tensorValues[tensorType].size(); ++value) {
                    // Get Variable value
                    double tmpVariableValueT = selLayerVars.tensorValueOneHotVars[memLevel][tensorType][value].get(GRB_DoubleAttr_X);

                    if (tmpVariableValueT > 0) {
                        llvm::outs() << selLayerInfo.tensorValues[tensorType][value] << "\n";
                    }
                }
            }
        }

    }
}

std::string DetailLayoutMILP::readJsonFile(const std::string& filePath) {
    std::ifstream jsonFile(filePath);

    if (!jsonFile.is_open()) {
        llvm::errs() << "Failed to open file: " << filePath << "\n";
        return "";
    }

    std::stringstream fileBuffer;
    fileBuffer << jsonFile.rdbuf();

    return fileBuffer.str();
}

llvm::Expected<llvm::json::Value> DetailLayoutMILP::parseJson(const std::string& jsonString) {
    // Create the json value instance
    llvm::Expected<llvm::json::Value> jsonValue = llvm::json::parse(jsonString);

    // Error Handling
    if (!jsonValue) {
        llvm::Error parseError = jsonValue.takeError();
        llvm::errs() << "JSON Parsing failed: " << llvm::toString(std::move(parseError)) << "\n";
        return llvm::make_error<llvm::StringError>("Failed to parse JSON\n", llvm::inconvertibleErrorCode());
    }

    return jsonValue;
}

void DetailLayoutMILP::readKnobValues(const std::string& filePath) {
    // Get the json file content
    std::string jsonContent = readJsonFile(filePath);

    // Get the json Value
    auto maybeJsonValue = parseJson(jsonContent);

    // Error Handling
    if (maybeJsonValue) {
        // Get the actual json value
        llvm::json::Value &jsonValue = *maybeJsonValue;

        // Parse the contents
        if (const auto *obj = jsonValue.getAsObject()) {
            double_t workLoadTolerance = *(obj->getNumber("workLoadTolerance"));
            double_t cartesianProTolerance = *(obj->getNumber("cartesianProTolerance"));
            double_t compWeight = *(obj->getNumber("compWeight"));
            double_t rowActWeight = *(obj->getNumber("rowActWeight"));
            double_t crossBankWeight = *(obj->getNumber("crossBankWeight"));
            double_t intraBankWeight = *(obj->getNumber("intraBankWeight"));
            double_t crossLayerWeight = *(obj->getNumber("crossLayerWeight"));
            
            // Update the performance knobs
            PerformanceKnobs.workLoadTolerance = workLoadTolerance;
            PerformanceKnobs.cartesianProTolerance = cartesianProTolerance;
            PerformanceKnobs.compWeight = compWeight;
            PerformanceKnobs.rowActWeight = rowActWeight;
            PerformanceKnobs.crossBankWeight = crossBankWeight;
            PerformanceKnobs.intraBankWeight = intraBankWeight;
            PerformanceKnobs.crossLayerWeight = crossLayerWeight;
        } else {
            llvm::errs() << "Expected a JSON object\n";
        }
        
    } else {
        llvm::errs() << "Failed to parse the JSON file in: " << filePath << "\n";
    }

    llvm::outs() << "Successfully loaded tuning knob file " << filePath << "\n";
    llvm::outs() << PerformanceKnobs.toString() << "\n";
}

void DetailLayoutMILP::readArchInfo(const std::string& filePath) {
    // Get the json file content
    std::string jsonContent = readJsonFile(filePath);

    // Get the json Value
    auto maybeJsonValue = parseJson(jsonContent);

    // Error Handling
    if (maybeJsonValue) {
        // Get the actual json value
        llvm::json::Value &jsonValue = *maybeJsonValue;

        // Parse the contents
        if (const auto *obj = jsonValue.getAsObject()) {
            int32_t dataSize = *(obj->getNumber("dataSize"));
            int32_t bankSize = *(obj->getNumber("bankSize"));
            int32_t rowSize = *(obj->getNumber("rowSize"));
            int32_t HBM2 = *(obj->getNumber("HBM2"));
            int32_t HBM2E = *(obj->getNumber("HBM2E"));
            int32_t rowActTime = *(obj->getNumber("rowActTime"));
            int32_t crossBankTime = *(obj->getNumber("crossBankTime"));
            int32_t HBMBandwidth = *(obj->getNumber("HBMBandwidth"));

            
            // Update the Architecture info
            archInfo.dataSize = dataSize;
            archInfo.bankSize = bankSize;
            archInfo.rowSize = rowSize;
            archInfo.dataQueueSizeHBM = HBM2;
            archInfo.dataQueueSizeHBME = HBM2E;
            archInfo.rowActTime = rowActTime;
            archInfo.crossBankTime = crossBankTime;
            archInfo.HBMBandwidth = HBMBandwidth;
        } else {
            llvm::errs() << "Expected a JSON object\n";
        }
        
    } else {
        llvm::errs() << "Failed to parse the JSON file in: " << filePath << "\n";
    }

    llvm::outs() << "Successfully loaded architecture file " << filePath << "\n";
    llvm::outs() << archInfo.toString() << "\n";
}

std::string DetailLayoutMILP::traceConv2D(const int32_t& selLayer, const LayerInfo& oriLayerInfo) {
    std::string outputString;

    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];
    LayerGRBVariables& selLayerVars = layerVariables[selLayer];

    outputString += "conv2d\n";

    outputString += "Problem: ";
    for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {
        if (workLoad != NUM_DIM_CONV2D - 1) {
            outputString += (std::to_string(selLayerInfo.workLoadDimMap[workLoad].workLoadDim) + ",");
        } else {
            outputString += (std::to_string(selLayerInfo.workLoadDimMap[workLoad].workLoadDim) + "\n");
        }
    }

    outputString += "DilationStride: ";
    outputString += (std::to_string(oriLayerInfo.dilation[0]) + ",");
    outputString += (std::to_string(oriLayerInfo.dilation[1]) + ",");
    outputString += (std::to_string(oriLayerInfo.stride[0]) + ",");
    outputString += (std::to_string(oriLayerInfo.stride[1]) + "\n");

    outputString += "Loop: N,K,P,Q,C,R,S,N,K,P,Q,C,R,S,N,K,P,Q,C,R,S\n";

    outputString += "Bound: ";
    // Iterate through all mem levels
    for (int memLevel = memLevel::Bank; memLevel >= 0; --memLevel) {
        // Iterate through all Loop Bound Encoding variables
        for (int workLoad = 0; workLoad < NUM_DIM_CONV2D; ++workLoad) {
            // Iterate through all possible divisors
            for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {
                // Get Variable value
                double tmpVariableValue = selLayerVars.memLoopBoundOneHotVars[workLoad][memLevel][div].get(GRB_DoubleAttr_X);
                if (tmpVariableValue > 0) {
                    if (memLevel == 0 && (workLoad == (NUM_DIM_CONV2D - 1))) {
                        outputString += (std::to_string(selLayerInfo.workLoadDimMap[workLoad].divisorsVec[div]) + "\n");
                    } else {
                        outputString += (std::to_string(selLayerInfo.workLoadDimMap[workLoad].divisorsVec[div]) + ",");
                    }
                }
            }
        }
    }

    outputString += "Tag: P,P,P,P,P,P,P,T,T,T,T,T,T,T,P,P,P,P,P,P,P\n";
    outputString += "StartBankRow: 0,0\n";
    outputString += "end\n";

    return outputString;
}

std::string DetailLayoutMILP::traceFC(const int32_t& selLayer, const LayerInfo& oriLayerInfo) {
    std::string outputString;

    // Get the needed layer const
    ConstInfo& selLayerInfo = layerConstInfo[selLayer];
    LayerGRBVariables& selLayerVars = layerVariables[selLayer];

    outputString += "gemm\n";

    outputString += "Problem: ";
    for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {
        if (workLoad != NUM_DIM_FC - 1) {
            outputString += (std::to_string(selLayerInfo.workLoadDimMap[workLoad].workLoadDim) + ",");
        } else {
            outputString += (std::to_string(selLayerInfo.workLoadDimMap[workLoad].workLoadDim) + "\n");
        }
    }

    outputString += "DilationStride: 1,1,1,1\n";

    outputString += "Loops: N,H,P,Q,R,N,H,P,Q,R,N,H,P,Q,R\n";

    outputString += "Bound: ";
    // Iterate through all mem levels
    for (int memLevel = memLevel::Bank; memLevel >= 0; --memLevel) {
        // Iterate through all Loop Bound Encoding variables
        for (int workLoad = 0; workLoad < NUM_DIM_FC; ++workLoad) {
            // Iterate through all possible divisors
            for (int div = 0; div < selLayerInfo.workLoadDimMap[workLoad].divisorsVec.size(); ++div) {
                // Get Variable value
                double tmpVariableValue = selLayerVars.memLoopBoundOneHotVars[workLoad][memLevel][div].get(GRB_DoubleAttr_X);
                if (tmpVariableValue > 0) {
                    if (memLevel == 0 && (workLoad == (NUM_DIM_FC - 1))) {
                        outputString += (std::to_string(selLayerInfo.workLoadDimMap[workLoad].divisorsVec[div]) + "\n");
                    } else {
                        outputString += (std::to_string(selLayerInfo.workLoadDimMap[workLoad].divisorsVec[div]) + ",");
                    }
                }
            }
        }
    }

    outputString += "Tag: P,P,P,P,P,T,T,T,T,T,P,P,P,P,P\n";
    outputString += "StartBankRow: 0,0\n";
    outputString += "end\n";

    return outputString;
}

#endif // PIMOPT_GUROBI_NOT_INSTALLED
