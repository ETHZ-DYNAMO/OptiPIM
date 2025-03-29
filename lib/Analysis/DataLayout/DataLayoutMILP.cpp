/*
 ██████╗ ██████╗ ████████╗██╗██████╗ ██╗███╗   ███╗
██╔═══██╗██╔══██╗╚══██╔══╝██║██╔══██╗██║████╗ ████║
██║   ██║██████╔╝   ██║   ██║██████╔╝██║██╔████╔██║
██║   ██║██╔═══╝    ██║   ██║██╔═══╝ ██║██║╚██╔╝██║
╚██████╔╝██║        ██║   ██║██║     ██║██║ ╚═╝ ██║
 ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝                                                 
*/

//===----------------------------------------------------------------------===//
//
// This file implements main interfaces for the DataLayoutPass.
//
//===----------------------------------------------------------------------===//

#include "pimopt/Analysis/DataLayout/DataLayoutMILP.h"

// Check the existence of Gurobi library
#ifndef PIMOPT_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace mlir;
using namespace pim;

// =============================================
//                   Constructor
// =============================================

DataLayoutMILP::DataLayoutMILP(GRBEnv &env, std::vector<int32_t> &opGroup,
                    std::map<int32_t, OperatorInfo> &opsInfoDB,
                    const std::string& logPath, const std::string& resultPath,
                    const std::string& archPath, const std::string& knobPath,
                    const int32_t& deviceType, const int32_t& memoryAllocScheme,
                    const int32_t& objMethod, const int32_t& opGroupID,
                    const int32_t& transCoeffMethod, const int32_t& numCountingMethod,
                    const int32_t& storageMethod) : model(GRBModel(env)) {

    // Get the op group id
    int32_t selOpGroupID = opsInfoDB[opGroup[0]].opGroupID;

    // Set all user configs
    selDeviceType = deviceType;
    memoryAlloc = memoryAllocScheme;
    selObjMethod = objMethod;
    selTransCoeffMethod = transCoeffMethod;
    selNumCountingMethod = numCountingMethod;
    selStorageMethod = storageMethod;

    // Check the validity of user specifications
    if (memoryAlloc == memAllocIdx::Combined) {
        // Error
        llvm::outs() << "[ERROR] The combined memory allocation scheme is still under construction" << "\n";
        exit(-1);
    }

    // Final Objective Expression -- for the whole layer group
    GRBLinExpr finalObj;

    // set arch and knob path 
    selArchPath = archPath;
    selKnobPath = knobPath;

    // Step 0: Parse Knob values and arch info, passed from command line
    llvm::outs() << "[Debug] arch and knob paths: " << selArchPath << " " << selKnobPath << "\n";
    parseKnobsJSON(selKnobPath);
    parseArchJSON(selArchPath);

    // Potential Normalization Process -- Skipped for now
    // TODO: Check the necessity of the normalization process
    // Build the Scaling and normalization factors
    // buildScalingFactor(layerGroup, layersInfoDB);

    // Total Number of PEs
    int32_t opGroupPEs = opsInfoDB[opGroup[0]].numPE;

    // Iterate through all ops in the op group vector
    // and construct all needed constants
    //      1. Divs lists for all workload dimensions
    for (auto opIndex : opGroup) {
        // Get op type
        int32_t selOpType = opsInfoDB[opIndex].opType;

        // Construct the layer-specific MILP Model
        if (selOpType == operatorTypesIdx::CONV2D) {
            finalObj += constructConv2DMILP(opsInfoDB[opIndex], opIndex);
        } else if (selOpType == operatorTypesIdx::FC) {
            finalObj += constructFCMILP(opsInfoDB[opIndex], opIndex);
        }
    }

    // Sets the MILP Objective
    try {
        model.setObjective(finalObj, GRB_MINIMIZE);
    } catch (GRBException& e) {
        llvm::errs() << "[Objective Function] Gurobi Exception occurred: " << e.getMessage() << "\n";
    }

    // Debug, write out the model
    if (logPath != "") {
        try {
            writeMILPModel(model, logPath, selOpGroupID);
        } catch (GRBException& e) {
            llvm::errs() << "[Model Logging] Gurobi Exception occurred: " << e.getMessage() << "\n";
        }
    }

    //
    // Optimize the model
    //
    try {
        model.optimize();

        
    } catch (GRBException& e) {
        llvm::errs() << "[Optimization] Gurobi Exception occurred: " << e.getMessage() << "\n";
    }

    // Check the status of the optimized model
    if (int status = model.get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
        llvm::errs() << "Skipping this run, expected behavior. \n";

    } else {
        // Create output file
        // TODO: Need to make the logging more flexible
        // std::string resultTracePath = resultPath + "/Group_" + std::to_string(selOpGroupID) + ".txt";
        std::ofstream layerGroupOutFile = std::ofstream(resultPath);

        // Print out results -- Extract All Results
        for (auto layerIndex : opGroup) {
            opConstInfoMap[layerIndex].printDetail();

            if (opConstInfoMap[layerIndex].opType == operatorTypesIdx::CONV2D) {
                // Print the results to terminal
                printConv2DMILPResult(layerIndex);

                // Dump the results to a file
                layerGroupOutFile << traceOutputSIMConv2D(layerIndex, opsInfoDB[layerIndex]);
            } else if (opConstInfoMap[layerIndex].opType == operatorTypesIdx::FC) {
                // Print the results to terminal
                printFCMILPResult(layerIndex);

                // Dump the results to a file
                layerGroupOutFile << traceOutputSIMFC(layerIndex, opsInfoDB[layerIndex]);
            }
        }
    }
}


// =============================================
//                 Helper Functions
// =============================================

void DataLayoutMILP::writeMILPModel(GRBModel& model, const std::string& logPath, const int32_t& opGroupID) {
    if (!logPath.empty())
        model.write(logPath);
}

void DataLayoutMILP::writeMILPSol(GRBModel& model, const std::string& logPath, const int32_t& opGroupID) {
    if (!logPath.empty())
        model.write(logPath + "OpGroup_" + std::to_string(opGroupID) + "_solutions.json");
}

std::string DataLayoutMILP::readJsonFile(const std::string& filePath) {
    std::ifstream jsonFile(filePath);

    if (!jsonFile.is_open()) {
        llvm::errs() << "Failed to open file: " << filePath << "\n";
        return "";
    }

    std::stringstream fileBuffer;
    fileBuffer << jsonFile.rdbuf();

    return fileBuffer.str();
}

llvm::Expected<llvm::json::Value> DataLayoutMILP::parseJson(const std::string& jsonString) {
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

void DataLayoutMILP::parseKnobsJSON(const std::string& filePath) {
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
            double_t accTolerance = *(obj->getNumber("accTolerance"));
            double_t colMulWeight = *(obj->getNumber("colMulWeight"));
            double_t colAddWeight = *(obj->getNumber("colAddWeight"));
            double_t outTransWeight = *(obj->getNumber("outTransWeight"));
            double_t inLoadingWeight = *(obj->getNumber("inLoadingWeight"));
            double_t interOpTransWeight = *(obj->getNumber("interOpTransWeight"));
            
            // Update the performance knobs
            objKnobs.accTolerance = accTolerance;
            objKnobs.colMulWeight = colMulWeight;
            objKnobs.colAddWeight = colAddWeight;
            objKnobs.outTransWeight = outTransWeight;
            objKnobs.inLoadingWeight = inLoadingWeight;
            objKnobs.interOpTransWeight = interOpTransWeight;
        } else {
            llvm::errs() << "Expected a JSON object\n";
        }
        
    } else {
        llvm::errs() << "Failed to parse the JSON file in: " << filePath << "\n";
    }

    llvm::outs() << "Successfully loaded tuning knob file " << filePath << "\n";
    llvm::outs() << objKnobs.toString() << "\n";
}

void DataLayoutMILP::parseArchJSON(const std::string& filePath) {
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
            int32_t dataWidth = *(obj->getNumber("dataWidth"));
            int32_t numRow = *(obj->getNumber("numRow"));
            int32_t numCol = *(obj->getNumber("numCol"));
            int64_t PEBandWidth = *(obj->getNumber("PEBandWidth"));
            int64_t SysBandWidth = *(obj->getNumber("SysBandWidth"));
            int64_t mulLat = *(obj->getNumber("mulLat"));
            int64_t addLat = *(obj->getNumber("addLat"));
            int64_t rowAct = *(obj->getNumber("rowAct"));
            int32_t interColTransLat = *(obj->getNumber("interColTransLat"));
            int32_t interPETransLat = *(obj->getNumber("interPETransLat"));

            
            // Update the Architecture info
            archInfo.dataWidth = dataWidth;
            archInfo.numRow = numRow;
            archInfo.numCol = numCol;
            archInfo.PEBandWidth = PEBandWidth;
            archInfo.SysBandWidth = SysBandWidth;
            archInfo.mulLat = mulLat;
            archInfo.addLat = addLat;
            archInfo.rowAct = rowAct;
            archInfo.interColTransLat = interColTransLat;
            archInfo.interPETransLat = interPETransLat;
        } else {
            llvm::errs() << "Expected a JSON object\n";
        }
        
    } else {
        llvm::errs() << "Failed to parse the JSON file in: " << filePath << "\n";
    }

    llvm::outs() << "Successfully loaded architecture file " << filePath << "\n";
    llvm::outs() << archInfo.toString() << "\n";
}

// =============================================
//        Print Detailed Output -- Conv2D
// =============================================

void DataLayoutMILP::printConv2DMILPResult(const int32_t& selOpIndex) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIndex];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIndex];

    //
    llvm::outs() << "\n\n[OPTIMIZED OPERATOR RESULTS]\n";
    llvm::outs() << "\tOp ID: " << selOpIndex << "; Type: Conv2D; Target Device: " << deviceTypeName[selDeviceType] << ";\n";

    //
    //  Retrieve Transformed loop bounds
    //
    // Iterate over all loop levels
    for (int n = 0; n < NUM_LOOP_LEVEL; n++) {

        llvm::outs() << "\t[" << memoryLevelsName[n] << "]:\n";

        // Iterate over all loop bounds
        for (int d = 0; d < NUM_BOUND_CONV2D; d++) {

            llvm::outs() << "\t\t" << loopBoundNameConv2D[d] << ": ";

            // Get the actual loop bounds
            int64_t tmpLoopBound = static_cast<int64_t>(tmpOpVars.loopBoundsVars[d].loopBoundIntVars[n].get(GRB_DoubleAttr_X));

            llvm::outs() << tmpLoopBound << "\n";

            // Check the validity of the obtained solution
            // Iterate through all possible diviosrs
            int32_t counter = 0;
            for (int k = 0; k < selOpInfo.loopBoundInfoMap[d].divisorsVec.size(); k++) {
                // Get Variable Value
                double tmpLoopBoundDivVar = tmpOpVars.loopBoundsVars[d].loopBoundBinaryVars[n][k].get(GRB_DoubleAttr_X);

                if (tmpLoopBoundDivVar > 0) {
                    counter++;

                    if (counter > 1) {
                        llvm::outs() << "\t\t\t[ERROR] More than 1 divisor selected for " << loopBoundNameConv2D[d] << ": " << selOpInfo.loopBoundInfoMap[d].divisorsVec[k] << "; Variable Value: " << tmpLoopBoundDivVar << "\n";
                    }
                }
            }
        }
    }

    //
    // Retrieve all needed transformation coefficients
    //
    llvm::outs() << "\t[Transformation Coefficients]:\n";

    // Iterate over all related loop variables
    for (int m = 0; m < 4; m++) {
        //
        llvm::outs() << "\t\t" << transCoeffConv2DName[m] << ": ";

        // Get the actual coefficient value
        int64_t tmpCoeffValue = static_cast<int64_t>(tmpOpVars.loopCoeffsVars[m].coeffIntVars[0].get(GRB_DoubleAttr_X));

        llvm::outs() << tmpCoeffValue << "\n";

        // Check the validity of the obtained solution
        // Iterate through all possible diviosrs
        int32_t counter = 0;
        for (int f = 0; f < 6; f++) {
            // Get Variable Value
            double tmpCoeffCombVar = tmpOpVars.loopCoeffsVars[m].coeffBinaryVars[f].get(GRB_DoubleAttr_X);

            if (tmpCoeffCombVar > 0) {
                counter++;
                
                llvm::outs() << "\t\t\t" << "Trans Comb " << f << " selected;\n";

                if (counter > 1) {
                    llvm::outs() << "\t\t\t[ERROR] More than 1 tans comb selected for " << transCoeffConv2DName[m] << ": " << f << "; Variable Value: " << tmpCoeffCombVar << "\n";
                }
            }
        }
    }

    //
    // Retrieve the selected LUT pair
    //

    for (int i = 0; i < 2; i++) {
        llvm::outs() << "\t\t" << inputArrayLUTNameConv2D[i] << " selected pair: (";

        for (int j = 0; j < tmpOpVars.LUTSelVars[i].size(); j++) {
            // Get Variable Value
            double tmpLUTVar = tmpOpVars.LUTSelVars[i][j].get(GRB_DoubleAttr_X);

            if (tmpLUTVar > 0) {
                // Found the selected value
                llvm::outs() << selOpInfo.inputCoeffCatSets[i][j][0] << ", " << selOpInfo.inputCoeffCatSets[i][j][1] << ")\n";
                llvm::outs() << "\t\t\tCorresponding GCD Value: " << selOpInfo.inputCoeffLUTs[i][j] << "; Idx: " << j << "\n";
            }
        }
    }
   


    //
    // Retrieve all cost related variables
    //
    llvm::outs() << "\t[Perfromance modeling Variables]:\n";

    // PRGCDValue
    double tmpPRGCDValue = tmpOpVars.PRGCDValue.get(GRB_DoubleAttr_X);
    llvm::outs() << "\t\tPRGCDValue: " << tmpPRGCDValue << "\n ";

    // QSGCDValue
    double tmpQSGCDValue = tmpOpVars.QSGCDValue.get(GRB_DoubleAttr_X);
    llvm::outs() << "\t\tQSGCDValue: " << tmpQSGCDValue << "\n\n ";
    
    // numMulCol
    int64_t tmpnumMulCol = static_cast<int64_t>(tmpOpVars.numMulCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tNumMulCol: " << tmpnumMulCol << "\n ";

    // numOutCol
    int64_t tmpnumOutCol = static_cast<int64_t>(tmpOpVars.numOutCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumOutCol: " << tmpnumOutCol << "\n ";

    // numFilCol
    int64_t tmpnumFilCol = static_cast<int64_t>(tmpOpVars.numFilCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumFilCol: " << tmpnumFilCol << "\n ";

    // numInCol
    int64_t tmpnumInCol = static_cast<int64_t>(tmpOpVars.numInCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumInCol: " << tmpnumInCol << "\n ";

    // numInColTerm2
    int64_t tmpnumInTerm2Col = static_cast<int64_t>(tmpOpVars.numInTerm2Col.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumInTerm2Col: " << tmpnumInTerm2Col << "\n ";

    // numInColTerm3
    int64_t tmpnumInTerm3Col = static_cast<int64_t>(tmpOpVars.numInTerm3Col.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumInTerm3Col: " << tmpnumInTerm3Col << "\n ";

    // numRedOutCol
    int64_t tmpnumRedOutCol = static_cast<int64_t>(tmpOpVars.numRedOutCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumRedOutCol: " << tmpnumRedOutCol << "\n ";

    // numRedCol
    int64_t tmpnumRedCol = static_cast<int64_t>(tmpOpVars.numRedCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumRedCol: " << tmpnumRedCol << "\n ";

    // numColPE
    int64_t tmpnumColPE = static_cast<int64_t>(tmpOpVars.numColPE.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumColPE: " << tmpnumColPE << "\n ";

    // numOutPE
    int64_t tmpnumOutPE = static_cast<int64_t>(tmpOpVars.numOutPE.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumOutPE: " << tmpnumOutPE << "\n ";

    // numInPE
    int64_t tmpnumInPE = static_cast<int64_t>(tmpOpVars.numInPE.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumInPE: " << tmpnumInPE << "\n ";

    // numPESys
    int64_t tmpnumPESys = static_cast<int64_t>(tmpOpVars.numPESys.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumPESys: " << tmpnumPESys << "\n ";

    // numOutSys
    double tmpnumOutSys = tmpOpVars.numOutSys.get(GRB_DoubleAttr_X);
    llvm::outs() << "\t\tnumOutSys: " << tmpnumOutSys << "\n ";

    // numInSys
    double tmpnumInSys = tmpOpVars.numInSys.get(GRB_DoubleAttr_X);
    llvm::outs() << "\t\tnumInSys: " << tmpnumInSys << "\n ";

    // If the target is PNM
    double tmpNumFilPELoading = 1;
    if (selDeviceType == deviceTypeIdx::PNM) {
        tmpNumFilPELoading = tmpOpVars.numFilPELoading.get(GRB_DoubleAttr_X);
        llvm::outs() << "\t\tnumFilLoading: " << tmpNumFilPELoading << "\n ";
    }

    // Final Performance
    llvm::outs() << "\t[Analytical Perforamnce Modeling Result]:\n";
    double finalPerfResult = 0.0;
    if (selDeviceType == deviceTypeIdx::PUM) {
        finalPerfResult += (tmpnumMulCol * archInfo.mulLat);
        finalPerfResult += tmpnumRedCol * archInfo.addLat;
        finalPerfResult += tmpnumOutSys;
        finalPerfResult += tmpnumInSys;
    } else if (selDeviceType == deviceTypeIdx::PNM) {
        finalPerfResult += tmpnumOutSys;
        finalPerfResult += tmpnumInSys;
        finalPerfResult += tmpNumFilPELoading;
        finalPerfResult += tmpnumFilCol * archInfo.rowAct;
    }

    llvm::outs() << "\t\t Real Cost: " << finalPerfResult << "\n";

}

std::string DataLayoutMILP::traceOutputConv2D(const int32_t& selOpIndex, const OperatorInfo& oriOpInfo) {
    // Define the output string
    std::string outputString;

    // Get the needed Op const
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIndex];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIndex];

    outputString += "conv2d\n";

    outputString += "Problem: ";

    // Print the original loop bounds
    for (int x = 0; x < NUM_BOUND_CONV2D; x++) {
        if (x != NUM_BOUND_CONV2D - 1) {
            outputString += (std::to_string(selOpInfo.loopBoundInfoMap[x].loopBoundValue) + ",");
        } else {
            outputString += (std::to_string(selOpInfo.loopBoundInfoMap[x].loopBoundValue) + "\n");
        }
    }

    outputString += "DilationStride: ";
    outputString += (std::to_string(oriOpInfo.dilation[0]) + ",");
    outputString += (std::to_string(oriOpInfo.dilation[1]) + ",");
    outputString += (std::to_string(oriOpInfo.stride[0]) + ",");
    outputString += (std::to_string(oriOpInfo.stride[1]) + "\n");
    

    // The assumed order is : Level 2 -> Level 1 -> Level 0
    outputString += "Loop: N,K,P,Q,C,R,S,N,K,P,Q,C,R,S,N,K,P,Q,C,R,S\n";

    outputString += "Bound: ";
    
    //
    //  Retrieve Transformed loop bounds
    //
    // Iterate over all loop levels
    for (int n = loopLevelIdx::LEVEL2; n >= 0; n--) {
        // Iterate over all loop bounds
        for (int d = 0; d < NUM_BOUND_CONV2D; d++) {
            // Get the actual loop bounds
            int64_t tmpLoopBound = static_cast<int64_t>(tmpOpVars.loopBoundsVars[d].loopBoundIntVars[n].get(GRB_DoubleAttr_X));

            if (n == 0 && (d == (NUM_BOUND_CONV2D - 1))) {
                outputString += (std::to_string(tmpLoopBound) + "\n");
            } else {
                outputString += (std::to_string(tmpLoopBound) + ",");
            }
        }
    }

    outputString += "Tag: P,P,P,P,P,P,P,P,P,P,P,P,P,P,T,T,T,T,T,T,T\n";
    outputString += "StartBankRow: 0,0\n";
    

    //
    //  Retrieve ALL the transformation coefficients
    //
    // All coefficients are ordered as : Level 2, Level 1, Level 0
    // LoopVariable : N
    outputString += "N2,N1,N0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::N, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::N, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::N, selOpIndex) + "\n");
    // LoopVariable : K
    outputString += "K2,K1,K0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::K, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::K, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::K, selOpIndex) + "\n");
    // LoopVariable : P
    outputString += "P2,P1,P0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::P, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::P, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::P, selOpIndex) + "\n");
    // LoopVariable : Q
    outputString += "Q2,Q1,Q0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::Q, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::Q, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::Q, selOpIndex) + "\n");
    // LoopVariable : C
    outputString += "C2,C1,C0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::C, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::C, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::C, selOpIndex) + "\n");
    // LoopVariable : R
    outputString += "R2,R1,R0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::R, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::R, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::R, selOpIndex) + "\n");
    // LoopVariable : S
    outputString += "S2,S1,S0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::S, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::S, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::S, selOpIndex) + "\n");

    outputString += "end\n";

    return outputString;
}

std::string DataLayoutMILP::traceOutputSIMConv2D(const int32_t& selOpIndex, const OperatorInfo& oriOpInfo) {
    // Define the output string
    std::string outputString;

    // Get the needed Op const
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIndex];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIndex];

    outputString += "conv2d\n";

    outputString += "Problem: ";

    // Print the original loop bounds
    for (int x = 0; x < NUM_BOUND_CONV2D; x++) {
        if (x != NUM_BOUND_CONV2D - 1) {
            outputString += (std::to_string(selOpInfo.loopBoundInfoMap[x].loopBoundValue) + ",");
        } else {
            outputString += (std::to_string(selOpInfo.loopBoundInfoMap[x].loopBoundValue) + "\n");
        }
    }

    outputString += "DilationStride: ";
    outputString += (std::to_string(oriOpInfo.dilation[0]) + ",");
    outputString += (std::to_string(oriOpInfo.dilation[1]) + ",");
    outputString += (std::to_string(oriOpInfo.stride[0]) + ",");
    outputString += (std::to_string(oriOpInfo.stride[1]) + "\n");
    

    // The assumed order is : Level 2 -> Level 0 -> Level 1
    outputString += "Loop: N,K,P,Q,C,R,S,N,K,P,Q,C,R,S,N,K,P,Q,C,R,S\n";

    outputString += "Bound: ";
    
    //
    //  Retrieve Transformed loop bounds
    //
    // Iterate over all loop levels
    for (int n = loopLevelIdx::LEVEL2; n >= 0; n--) {
        // Get the inner Loop level index, change the position of level 1 and level 0 to align with the simulator
        int tmpLoopLevel = n;

        if (n == 1) {
            tmpLoopLevel = loopLevelIdx::LEVEL0;
        } else if (n == 0) {
            tmpLoopLevel = loopLevelIdx::LEVEL1;
        }
        
        // Iterate over all loop bounds
        for (int d = 0; d < NUM_BOUND_CONV2D; d++) {
            // Get the actual loop bounds
            int64_t tmpLoopBound = static_cast<int64_t>(tmpOpVars.loopBoundsVars[d].loopBoundIntVars[tmpLoopLevel].get(GRB_DoubleAttr_X));

            if (n == 0 && (d == (NUM_BOUND_CONV2D - 1))) {
                outputString += (std::to_string(tmpLoopBound) + "\n");
            } else {
                outputString += (std::to_string(tmpLoopBound) + ",");
            }
        }
    }

    outputString += "Tag: P,P,P,P,P,P,P,T,T,T,T,T,T,T,P,P,P,P,P,P,P\n";
    outputString += "StartBankRow: 0,0\n";
    

    //
    //  Retrieve ALL the transformation coefficients
    //
    // All coefficients are ordered as : Level 2, Level 0, Level 1 to align with the simulator
    // LoopVariable : N
    outputString += "Coeff_N2,N0,N1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::N, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::N, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::N, selOpIndex) + "\n");
    // LoopVariable : K
    outputString += "Coeff_K2,K0,K1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::K, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::K, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::K, selOpIndex) + "\n");
    // LoopVariable : P
    outputString += "Coeff_P2,P0,P1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::P, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::P, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::P, selOpIndex) + "\n");
    // LoopVariable : Q
    outputString += "Coeff_Q2,Q0,Q1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::Q, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::Q, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::Q, selOpIndex) + "\n");
    // LoopVariable : C
    outputString += "Coeff_C2,C0,C1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::C, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::C, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::C, selOpIndex) + "\n");
    // LoopVariable : R
    outputString += "Coeff_R2,R0,R1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::R, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::R, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::R, selOpIndex) + "\n");
    // LoopVariable : S
    outputString += "Coeff_S2,S0,S1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxConv2D::S, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxConv2D::S, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxConv2D::S, selOpIndex) + "\n");

    outputString += "end\n";

    return outputString;
}

// =============================================
//          Print Detailed Output -- FC
// =============================================

void DataLayoutMILP::printFCMILPResult(const int32_t& selOpIndex) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIndex];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIndex];

    //
    llvm::outs() << "\n\n[OPTIMIZED OPERATOR RESULTS]\n";
    llvm::outs() << "\tOp ID: " << selOpIndex << "; Type: FC; Target Device: " << deviceTypeName[selDeviceType] << ";\n";

    //
    //  Retrieve Transformed loop bounds
    //
    // Iterate over all loop levels
    for (int n = 0; n < NUM_LOOP_LEVEL; n++) {

        llvm::outs() << "\t[" << memoryLevelsName[n] << "]:\n";

        // Iterate over all loop bounds
        for (int d = 0; d < NUM_BOUND_FC; d++) {

            llvm::outs() << "\t\t" << loopBoundNameFC[d] << ": ";

            // Get the actual loop bounds
            int64_t tmpLoopBound = static_cast<int64_t>(tmpOpVars.loopBoundsVars[d].loopBoundIntVars[n].get(GRB_DoubleAttr_X));

            llvm::outs() << tmpLoopBound << "\n";

            // Check the validity of the obtained solution
            // Iterate through all possible diviosrs
            int32_t counter = 0;
            for (int k = 0; k < selOpInfo.loopBoundInfoMap[d].divisorsVec.size(); k++) {
                // Get Variable Value
                double tmpLoopBoundDivVar = tmpOpVars.loopBoundsVars[d].loopBoundBinaryVars[n][k].get(GRB_DoubleAttr_X);

                if (tmpLoopBoundDivVar > 0) {
                    counter++;

                    if (counter > 1) {
                        llvm::outs() << "\t\t\t[ERROR] More than 1 divisor selected for " << loopBoundNameConv2D[d] << ": " << selOpInfo.loopBoundInfoMap[d].divisorsVec[k] << "; Variable Value: " << tmpLoopBoundDivVar << "\n";
                    }
                }
            }
        }
    }

    //
    // Retrieve all cost related variables
    //
    llvm::outs() << "\t[Perfromance modeling Variables]:\n";
    
    // numMulCol
    int64_t tmpnumMulCol = static_cast<int64_t>(tmpOpVars.numMulCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tNumMulCol: " << tmpnumMulCol << "\n ";

    // numOutCol
    int64_t tmpnumOutCol = static_cast<int64_t>(tmpOpVars.numOutCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumOutCol: " << tmpnumOutCol << "\n ";

    // numFilCol
    int64_t tmpnumFilCol = static_cast<int64_t>(tmpOpVars.numFilCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumFilCol: " << tmpnumFilCol << "\n ";

    // numInCol
    int64_t tmpnumInCol = static_cast<int64_t>(tmpOpVars.numInCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumInCol: " << tmpnumInCol << "\n ";

    // numRedOutCol
    int64_t tmpnumRedOutCol = static_cast<int64_t>(tmpOpVars.numRedOutCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumRedOutCol: " << tmpnumRedOutCol << "\n ";

    // numRedCol
    int64_t tmpnumRedCol = static_cast<int64_t>(tmpOpVars.numRedCol.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumRedCol: " << tmpnumRedCol << "\n ";

    // numColPE
    int64_t tmpnumColPE = static_cast<int64_t>(tmpOpVars.numColPE.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumColPE: " << tmpnumColPE << "\n ";

    // numOutPE
    int64_t tmpnumOutPE = static_cast<int64_t>(tmpOpVars.numOutPE.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumOutPE: " << tmpnumOutPE << "\n ";

    // numInPE
    int64_t tmpnumInPE = static_cast<int64_t>(tmpOpVars.numInPE.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumInPE: " << tmpnumInPE << "\n ";

    // numPESys
    int64_t tmpnumPESys = static_cast<int64_t>(tmpOpVars.numPESys.get(GRB_DoubleAttr_X));
    llvm::outs() << "\t\tnumPESys: " << tmpnumPESys << "\n ";

    // numOutSys
    double tmpnumOutSys = tmpOpVars.numOutSys.get(GRB_DoubleAttr_X);
    llvm::outs() << "\t\tnumOutSys: " << tmpnumOutSys << "\n ";

    // numInSys
    double tmpnumInSys = tmpOpVars.numInSys.get(GRB_DoubleAttr_X);
    llvm::outs() << "\t\tnumInSys: " << tmpnumInSys << "\n ";

    // If the target is PNM
    double tmpNumFilPELoading = 1.0;
    if (selDeviceType == deviceTypeIdx::PNM) {
        tmpNumFilPELoading = tmpOpVars.numFilPELoading.get(GRB_DoubleAttr_X);
        llvm::outs() << "\t\tnumFilLoading: " << tmpNumFilPELoading << "\n ";
    }

    // Final Performance
    llvm::outs() << "\t[Analytical Perforamnce Modeling Result]:\n";
    double finalPerfResult = 0.0;
    if (selDeviceType == deviceTypeIdx::PUM) {
        finalPerfResult += (tmpnumMulCol * archInfo.mulLat);
        finalPerfResult += tmpnumRedCol * archInfo.addLat;
        finalPerfResult += tmpnumOutSys;
        finalPerfResult += tmpnumInSys;
    } else if (selDeviceType == deviceTypeIdx::PNM) {
        finalPerfResult += tmpnumFilCol * archInfo.rowAct;
        finalPerfResult += tmpNumFilPELoading;
        finalPerfResult += tmpnumOutSys;
        finalPerfResult += tmpnumInSys;
    }

    llvm::outs() << "\t\t Real Cost: " << finalPerfResult << "\n";
}

std::string DataLayoutMILP::traceOutputFC(const int32_t& selOpIndex, const OperatorInfo& oriOpInfo) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIndex];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIndex];

    std::string outputString;

    outputString += "gemm\n";

    outputString += "Problem: ";
    for (int x = 0; x < NUM_BOUND_FC; ++x) {
        if (x != NUM_BOUND_FC - 1) {
            outputString += (std::to_string(selOpInfo.loopBoundInfoMap[x].loopBoundValue) + ",");
        } else {
            outputString += (std::to_string(selOpInfo.loopBoundInfoMap[x].loopBoundValue) + "\n");
        }
    }

    outputString += "DilationStride: 1,1,1,1\n";

    outputString += "Loops: N,K,P,Q,R,N,K,P,Q,R,N,K,P,Q,R\n";

    outputString += "Bound: ";

    for (int n = loopLevelIdx::LEVEL2; n >= 0; --n) {

        // Iterate through all Loop Bound Encoding variables
        for (int d = 0; d < NUM_BOUND_FC; ++d) {
            // Get the actual loop bounds
            int64_t tmpLoopBound = static_cast<int64_t>(tmpOpVars.loopBoundsVars[d].loopBoundIntVars[n].get(GRB_DoubleAttr_X));

            if (n == 0 && (d == (NUM_BOUND_FC - 1))) {
                outputString += (std::to_string(tmpLoopBound) + "\n");
            } else {
                outputString += (std::to_string(tmpLoopBound) + ",");
            }
        }
    }

    outputString += "Tag: P,P,P,P,P,P,P,P,P,P,T,T,T,T,T\n";
    outputString += "StartBankRow: 0,0\n";

    //
    //  Retrieve ALL the transformation coefficients
    //
    // All coefficients are ordered as : Level 2, Level 1, Level 0
    // LoopVariable : N
    outputString += "N2,N1,N0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxFC::FC_N, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxFC::FC_N, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxFC::FC_N, selOpIndex) + "\n");
    // LoopVariable : K
    outputString += "K2,K1,K0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxFC::FC_K, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxFC::FC_K, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxFC::FC_K, selOpIndex) + "\n");
    // LoopVariable : P
    outputString += "P2,P1,P0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxFC::FC_P, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxFC::FC_P, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxFC::FC_P, selOpIndex) + "\n");
    // LoopVariable : Q
    outputString += "Q2,Q1,Q0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxFC::FC_Q, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxFC::FC_Q, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxFC::FC_Q, selOpIndex) + "\n");
    // LoopVariable : R
    outputString += "R2,R1,R0: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxFC::FC_R, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxFC::FC_R, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxFC::FC_R, selOpIndex) + "\n");

    outputString += "end\n";

    return outputString;
}

std::string DataLayoutMILP::traceOutputSIMFC(const int32_t& selOpIndex, const OperatorInfo& oriOpInfo) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOpIndex];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOpIndex];

    std::string outputString;

    outputString += "gemm\n";

    outputString += "Problem: ";
    for (int x = 0; x < NUM_BOUND_FC; ++x) {
        if (x != NUM_BOUND_FC - 1) {
            outputString += (std::to_string(selOpInfo.loopBoundInfoMap[x].loopBoundValue) + ",");
        } else {
            outputString += (std::to_string(selOpInfo.loopBoundInfoMap[x].loopBoundValue) + "\n");
        }
    }

    outputString += "DilationStride: 1,1,1,1\n";

    outputString += "Loops: N,K,P,Q,R,N,K,P,Q,R,N,K,P,Q,R\n";

    outputString += "Bound: ";

    for (int n = loopLevelIdx::LEVEL2; n >= 0; --n) {
        // Get the inner Loop level index, change the position of level 1 and level 0 to align with the simulator
        int tmpLoopLevel = n;

        if (n == 1) {
            tmpLoopLevel = loopLevelIdx::LEVEL0;
        } else if (n == 0) {
            tmpLoopLevel = loopLevelIdx::LEVEL1;
        }

        // Iterate through all Loop Bound Encoding variables
        for (int d = 0; d < NUM_BOUND_FC; ++d) {
            // Get the actual loop bounds
            int64_t tmpLoopBound = static_cast<int64_t>(tmpOpVars.loopBoundsVars[d].loopBoundIntVars[tmpLoopLevel].get(GRB_DoubleAttr_X));

            if (n == 0 && (d == (NUM_BOUND_FC - 1))) {
                outputString += (std::to_string(tmpLoopBound) + "\n");
            } else {
                outputString += (std::to_string(tmpLoopBound) + ",");
            }
        }
    }

    outputString += "Tag: P,P,P,P,P,T,T,T,T,T,P,P,P,P,P\n";
    outputString += "StartBankRow: 0,0\n";

    //
    //  Retrieve ALL the transformation coefficients
    //
    // All coefficients are ordered as : Level 2, Level 0, Level 1
    // LoopVariable : N
    outputString += "Coeff_N2,N0,N1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxFC::FC_N, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxFC::FC_N, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxFC::FC_N, selOpIndex) + "\n");
    // LoopVariable : K
    outputString += "Coeff_K2,K0,K1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxFC::FC_K, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxFC::FC_K, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxFC::FC_K, selOpIndex) + "\n");
    // LoopVariable : P
    outputString += "Coeff_P2,P0,P1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxFC::FC_P, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxFC::FC_P, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxFC::FC_P, selOpIndex) + "\n");
    // LoopVariable : Q
    outputString += "Coeff_Q2,Q0,Q1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxFC::FC_Q, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxFC::FC_Q, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxFC::FC_Q, selOpIndex) + "\n");
    // LoopVariable : R
    outputString += "Coeff_R2,R0,R1: ";
    outputString += (getCoeffVar(loopLevelIdx::LEVEL2, loopBoundIdxFC::FC_R, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL0, loopBoundIdxFC::FC_R, selOpIndex) + ",");
    outputString += (getCoeffVar(loopLevelIdx::LEVEL1, loopBoundIdxFC::FC_R, selOpIndex) + "\n");

    outputString += "end\n";

    return outputString;
}

// =============================================
//                 Helper Functions
// =============================================

std::string DataLayoutMILP::getCoeffVar(const int32_t& loopLevel, const int32_t& selLoopVar, const int32_t& selOp) {
    // Get the needed op const storing structure
    OpConstInfo& selOpInfo = opConstInfoMap[selOp];

    // Get the variable storing structure
    OpGRBVariables& tmpOpVars = opVariablesMap[selOp];

    // Define the vector
    std::vector<std::vector<int64_t>> combs;

    // Get the corresponding loop bounds
    int64_t tmpA = static_cast<int64_t>(tmpOpVars.loopBoundsVars[selLoopVar].loopBoundIntVars[2].get(GRB_DoubleAttr_X));
    int64_t tmpB = static_cast<int64_t>(tmpOpVars.loopBoundsVars[selLoopVar].loopBoundIntVars[1].get(GRB_DoubleAttr_X));
    int64_t tmpJ = static_cast<int64_t>(tmpOpVars.loopBoundsVars[selLoopVar].loopBoundIntVars[0].get(GRB_DoubleAttr_X));

    // Calculating the corresponding transformation coefficient
    // Comb 0: a * b, a, 1
    std::vector<int64_t> tmpComb0Coeffs;
    tmpComb0Coeffs.push_back(tmpA * tmpB);
    tmpComb0Coeffs.push_back(tmpA);
    tmpComb0Coeffs.push_back(1);
    combs.push_back(tmpComb0Coeffs);

    // Comb 1: a, a * j, 1
    std::vector<int64_t> tmpComb1Coeffs;
    tmpComb1Coeffs.push_back(tmpA);
    tmpComb1Coeffs.push_back(tmpA * tmpJ);
    tmpComb1Coeffs.push_back(1);
    combs.push_back(tmpComb1Coeffs);

    // Comb 2: 1, a * j, j
    std::vector<int64_t> tmpComb2Coeffs;
    tmpComb2Coeffs.push_back(1);
    tmpComb2Coeffs.push_back(tmpA * tmpJ);
    tmpComb2Coeffs.push_back(tmpJ);
    combs.push_back(tmpComb2Coeffs);

    // Comb 3: 1, j, b * j
    std::vector<int64_t> tmpComb3Coeffs;
    tmpComb3Coeffs.push_back(1);
    tmpComb3Coeffs.push_back(tmpJ);
    tmpComb3Coeffs.push_back(tmpB * tmpJ);
    combs.push_back(tmpComb3Coeffs);

    // Comb 4: a * b, 1, b
    std::vector<int64_t> tmpComb4Coeffs;
    tmpComb4Coeffs.push_back(tmpA * tmpB);
    tmpComb4Coeffs.push_back(1);
    tmpComb4Coeffs.push_back(tmpB);
    combs.push_back(tmpComb4Coeffs);

    // Comb 5: b, 1, b * j
    std::vector<int64_t> tmpComb5Coeffs;
    tmpComb5Coeffs.push_back(tmpB);
    tmpComb5Coeffs.push_back(1);
    tmpComb5Coeffs.push_back(tmpB * tmpJ);
    combs.push_back(tmpComb5Coeffs);

    // Get the desired coefficient
    if (selOpInfo.opType == operatorTypesIdx::FC) {
        // If this is FC , then we don't care about the transcoefficient, we directly use comb3
        return std::to_string(combs[3][loopLevel]);
    } else if (selOpInfo.opType == operatorTypesIdx::CONV2D) {
        if (selTransCoeffMethod == transCoeffMethodIdx::Comb0) {
            return std::to_string(combs[0][loopLevel]);
        } else if (selTransCoeffMethod == transCoeffMethodIdx::Comb1) {
            return std::to_string(combs[1][loopLevel]);
        } else if (selTransCoeffMethod == transCoeffMethodIdx::Comb2) {
            return std::to_string(combs[2][loopLevel]);
        } else if (selTransCoeffMethod == transCoeffMethodIdx::Comb5) {
            return std::to_string(combs[5][loopLevel]);
        } else if (selTransCoeffMethod == transCoeffMethodIdx::Flexible) {

            if (selLoopVar == loopBoundIdxConv2D::C) {
                return std::to_string(combs[3][loopLevel]);
            } else if (selLoopVar == loopBoundIdxConv2D::K) {
                return std::to_string(combs[3][loopLevel]);
            } else if (selLoopVar == loopBoundIdxConv2D::N) {
                return std::to_string(combs[3][loopLevel]);
            } else if (selLoopVar == loopBoundIdxConv2D::P) {
                // Get the selected Combination
                for (int f = 0; f < 6; f++) {
                    // Get Variable Value
                    double tmpCoeffCombVar = tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_P].coeffBinaryVars[f].get(GRB_DoubleAttr_X);

                    if (tmpCoeffCombVar > 0) {
                        
                        return std::to_string(combs[f][loopLevel]);

                    }    
                }
            } else if (selLoopVar == loopBoundIdxConv2D::R) {
                // Get the selected Combination
                for (int f = 0; f < 6; f++) {
                    // Get Variable Value
                    double tmpCoeffCombVar = tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_R].coeffBinaryVars[f].get(GRB_DoubleAttr_X);

                    if (tmpCoeffCombVar > 0) {
                        
                        return std::to_string(combs[f][loopLevel]);

                    }    
                }
            } else if (selLoopVar == loopBoundIdxConv2D::Q) {
                // Get the selected Combination
                for (int f = 0; f < 6; f++) {
                    // Get Variable Value
                    double tmpCoeffCombVar = tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_Q].coeffBinaryVars[f].get(GRB_DoubleAttr_X);

                    if (tmpCoeffCombVar > 0) {
                        
                        return std::to_string(combs[f][loopLevel]);

                    }    
                }
            } else if (selLoopVar == loopBoundIdxConv2D::S) {
                // Get the selected Combination
                for (int f = 0; f < 6; f++) {
                    // Get Variable Value
                    double tmpCoeffCombVar = tmpOpVars.loopCoeffsVars[transCoeffConv2DIdx::Conv2D_S].coeffBinaryVars[f].get(GRB_DoubleAttr_X);

                    if (tmpCoeffCombVar > 0) {
                        
                        return std::to_string(combs[f][loopLevel]);

                    }    
                }
            } 
        }

    }
}

#endif // PIMOPT_GUROBI_NOT_INSTALLED
