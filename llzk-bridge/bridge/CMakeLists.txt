# May change after this bridge actually depends on LLZK
cmake_minimum_required(VERSION 3.29) 

project(LLZKRustBridge)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(GNUInstallDirs)
set(LLZK_BRIDGE_EXPORT_TARGETS "LlzkBridgeTargets")
include(Dependencies.cmake)
llzkbridge_setup_dependencies(${LLZK_BRIDGE_EXPORT_TARGETS})

add_library(llzkbridge STATIC)
target_include_directories(llzkbridge 
  PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llzkbridge INTERFACE LLVMHeaders MLIRHeaders)
target_link_libraries(llzkbridge PRIVATE 
  LLZK::LLZKDialectHeaders 
  MLIRSupport 
  LLZK::LLZKDialect 
  MLIRIR 
  MLIRDebug
  LLVMSupport
)

add_subdirectory(lib)

# Install the library in the root of the output directory 
# since that is the place rustc will look for it.
install(TARGETS llzkbridge DESTINATION .)
