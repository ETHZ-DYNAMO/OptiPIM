add_library(PIMDataLayout SHARED
    DataLayoutPass.cpp
    DataLayoutMILP.cpp
    DataLayoutConv2D.cpp
    DataLayoutFC.cpp
)

target_link_libraries(PIMDataLayout
  PRIVATE
    MLIRPass
    LLVMPasses
    MLIRAffineDialect
    MLIRParser
    MLIRSupport
    MLIRIR
    MLIRBufferizationDialect
    MLIRAffineDialect
    MLIRMLProgramDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
)

# Include the public include directory
target_include_directories(PIMDataLayout
  PRIVATE
    ${PIMOPT_INCLUDE_DIR}
)

# Link Gurobi
if (GUROBI_FOUND)
  target_link_libraries(PIMDataLayout
      PUBLIC
      ${GUROBI_CXX_LIBRARY}
      ${GUROBI_LIBRARY}
  )
endif()

