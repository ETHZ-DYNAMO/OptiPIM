add_library(pim-codegen OBJECT)

target_sources(
  pim-codegen PRIVATE
  codegen.h 
  layout.h
  pim_ir.h
  data_space.h
  nest_analysis.h

  impl/fimdram.cpp
  impl/fimdram.h
  impl/simdram.cpp
  impl/simdram.h
  layout.cpp
)

target_link_libraries(
  ramulator
  PRIVATE
  pim-codegen
)