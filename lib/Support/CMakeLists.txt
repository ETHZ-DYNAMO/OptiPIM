add_library(PIMSupport SHARED
    MILP.cpp
)

target_link_libraries(PIMSupport
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
target_include_directories(PIMSupport
  PRIVATE
    ${PIMOPT_INCLUDE_DIR}
)

# Link Gurobi
if (GUROBI_FOUND)
  target_link_libraries(PIMSupport
      PUBLIC
      ${GUROBI_CXX_LIBRARY}
      ${GUROBI_LIBRARY}
  )
endif()