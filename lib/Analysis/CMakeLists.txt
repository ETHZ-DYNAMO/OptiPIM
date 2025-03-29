add_library(PIMAnalysis SHARED
    ExtractOpParamsPass.cpp
)

target_link_libraries(PIMAnalysis
  PRIVATE
    MLIRPass
    LLVMPasses
    MLIRAffineDialect
    MLIRParser
    MLIRSupport
    MLIRIR
    MLIRBufferizationDialect
    MLIRMLProgramDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
)

# Include the public include directory
target_include_directories(PIMAnalysis
  PRIVATE
    ${PIMOPT_INCLUDE_DIR}
)

# Link Gurobi
if (GUROBI_FOUND)
  target_link_libraries(PIMAnalysis
      PUBLIC
      ${GUROBI_CXX_LIBRARY}
      ${GUROBI_LIBRARY}
  )
endif()

# Include the folder for detail layout
add_subdirectory(DetailLayout)
add_subdirectory(DataLayout)