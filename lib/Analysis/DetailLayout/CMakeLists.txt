add_library(PIMDetailLayout SHARED
    DetailLayoutPass.cpp
    DetailLayoutMILP.cpp
    DetailLayoutConv2D.cpp
    DetailLayoutFC.cpp
)

target_link_libraries(PIMDetailLayout
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
target_include_directories(PIMDetailLayout
  PRIVATE
    ${PIMOPT_INCLUDE_DIR}
)

# Link Gurobi
if (GUROBI_FOUND)
  target_link_libraries(PIMDetailLayout
      PUBLIC
      ${GUROBI_CXX_LIBRARY}
      ${GUROBI_LIBRARY}
  )
endif()