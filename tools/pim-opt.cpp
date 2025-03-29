/*
 ██████╗ ██████╗ ████████╗██╗██████╗ ██╗███╗   ███╗
██╔═══██╗██╔══██╗╚══██╔══╝██║██╔══██╗██║████╗ ████║
██║   ██║██████╔╝   ██║   ██║██████╔╝██║██╔████╔██║
██║   ██║██╔═══╝    ██║   ██║██╔═══╝ ██║██║╚██╔╝██║
╚██████╔╝██║        ██║   ██║██║     ██║██║ ╚═╝ ██║
 ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝                                                
*/

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/FileSystem.h"

// Include the passes -- need refinement
#include "pimopt/Analysis/DataLayout/DataLayoutPass.h"
#include "pimopt/Analysis/ExtractOpParamsPass.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                        llvm::cl::desc("<input file>"),
                        llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Output filename"),
                         llvm::cl::value_desc("filename"),
                         llvm::cl::init("-"));

static llvm::cl::opt<bool> useDataLayoutPass("data-layout-pass",
                       llvm::cl::desc("Use Data Layout Pass"),
                       llvm::cl::init(false));

static llvm::cl::opt<bool> useExtractOpParamsPass("extract-op-params-pass",
                       llvm::cl::desc("Use Extract Op Params Pass"),
                       llvm::cl::init(false));

static llvm::cl::opt<std::string> traceOutputPath("trace-output-path",
                                             llvm::cl::desc("Specify the output path for the generated simulation trace file"),
                                             llvm::cl::value_desc("path"),
                                             llvm::cl::init("./Result/Layer_group_0_results.txt"));

static llvm::cl::opt<std::string> modelDebugPath("model-debug-file",
                                             llvm::cl::desc("Specify the output path to store the milp Model"),
                                             llvm::cl::value_desc("path-milp"),
                                             llvm::cl::init(""));

static llvm::cl::opt<int> deviceType("target-device-type",
                                    llvm::cl::desc("Specify the target device type"),
                                    llvm::cl::value_desc("devicetype"),
                                    llvm::cl::init(0));  

static llvm::cl::opt<int> memAlloScheme("memory-allocation-scheme",
                                    llvm::cl::desc("Specify the memory allocation scheme for MILP optimization"),
                                    llvm::cl::value_desc("memory-alloc"),
                                    llvm::cl::init(0));

static llvm::cl::opt<int> transCoeffMethod("trans-coeff-method",
                                    llvm::cl::desc("Specify the assumption made for the transformation coefficients for a single loop variable"),
                                    llvm::cl::value_desc("trans-coeff"),
                                    llvm::cl::init(4));

static llvm::cl::opt<int> numCountingMethod("number-counting-method",
                                    llvm::cl::desc("Specify the estimation methods for the input number"),
                                    llvm::cl::value_desc("number-counting"),
                                    llvm::cl::init(0));

static llvm::cl::opt<int> storageMethod("storage-method",
                                    llvm::cl::desc("Specify the storage method for the column"),
                                    llvm::cl::value_desc("storage-method"),
                                    llvm::cl::init(1));

static llvm::cl::opt<int> objMethod("obj-method",
                                    llvm::cl::desc("Specify the objective function for the milp"),
                                    llvm::cl::value_desc("obj-approx"),
                                    llvm::cl::init(0));

static llvm::cl::opt<std::string> configArchPath("config-arch-path",
                                      llvm::cl::desc("Specify path to architecture json file"),
                                      llvm::cl::value_desc("arch-json"),
                                      llvm::cl::init("")); 

static llvm::cl::opt<std::string> configKnobPath("config-knobs-path",
                                      llvm::cl::desc("Specify path to tuninig knob json file"),
                                      llvm::cl::value_desc("knob-json"),
                                      llvm::cl::init("")); 

int main(int argc, char **argv) {
  //! Note: This is only a temporary solution for registring all dialects we used
  // DialectRegistry registry;

  // // Add all dialects we care about
  // registry.insert<LLVM::LLVMDialect, affine::AffineDialect,
  //                 scf::SCFDialect, func::FuncDialect,
  //                 linalg::LinalgDialect>();
  // registerAllDialects(registry);

  // Register GurobiTestPass
  PassRegistration<pim::DataLayoutPass>();
  PassRegistration<pim::ExtractOpParamsPass>();

  llvm::cl::ParseCommandLineOptions(argc, argv, "PIM Optimization Pass");

  std::string errorMessage;
  auto inputBuffer = openInputFile(inputFilename, &errorMessage);
  if (!inputBuffer) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Create an MLIR context and parse the input file
  //! adding all dialects as a temporary solution
  MLIRContext context;
  context.getOrLoadDialect<mlir::ml_program::MLProgramDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputBuffer), llvm::SMLoc());
  
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  // auto module = parseSourceFile(*inputBuffer, &context);
  if (!module) {
    llvm::errs() << "Error: failed to parse the input MLIR file.\n";
    return 1;
  }

  PassManager pm(&context);
  if (useDataLayoutPass) {
    auto dataLayoutPass = std::make_unique<pim::DataLayoutPass>();

    dataLayoutPass->setOutputPath(traceOutputPath);
    dataLayoutPass->setDeviceType(deviceType);
    dataLayoutPass->setMemAlloc(memAlloScheme);
    dataLayoutPass->setTransCoeffMethod(transCoeffMethod);
    dataLayoutPass->setNumCountingMethod(numCountingMethod);
    dataLayoutPass->setStorageMethod(storageMethod);
    dataLayoutPass->setMilpPath(modelDebugPath);
    dataLayoutPass->setArchPath(configArchPath);
    dataLayoutPass->setKnobPath(configKnobPath);
    dataLayoutPass->setObjMethod(objMethod);

    
    pm.addPass(std::move(dataLayoutPass));
  }

  if (useExtractOpParamsPass) {
    pm.addPass(std::make_unique<pim::ExtractOpParamsPass>());
  }

  if (failed(pm.run(*module))) {
    llvm::errs() << "Error: pass manager failed.\n";
    return 1;
  }

  if (useExtractOpParamsPass) {
    module->print(output->os());
  }

  output->keep();

  return 0;
}
