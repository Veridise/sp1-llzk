# May change after this bridge actually depends on LLZK
cmake_minimum_required(VERSION 3.29) 

project(LLZKRustBridge)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(llzkbridge)

target_include_directories(llzkbridge 
  PRIVATE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_subdirectory(lib)

# Install the library in the root of the output directory 
# since that is the place rustc will look for it.
install(TARGETS llzkbridge DESTINATION .)
