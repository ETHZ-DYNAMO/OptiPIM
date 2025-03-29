set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(pim-opt
  pim-opt.cpp
)

llvm_update_compile_flags(pim-opt)
target_link_libraries(pim-opt
  PRIVATE
    PIMAnalysis
    PIMDataLayout
    PIMDetailLayout
    PIMSupport
    MLIRIR
    MLIROptLib
    MLIRParser
    MLIRSupport

    MLIRTransforms
    MLIRAffineTransforms
    MLIRArithTransforms
    MLIRSCFTransforms
    MLIRMemRefTransforms

    MLIRLLVMDialect
    MLIRBufferizationDialect
    MLIRMLProgramDialect
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRTensorDialect
)