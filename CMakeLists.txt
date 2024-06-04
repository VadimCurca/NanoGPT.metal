cmake_minimum_required(VERSION 3.14)
project(NetGen)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra")
add_link_options("-fuse-ld=lld")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# This is needed to force de Xcode generator to comply with the CMAKE_BUILD_TYPE
# variable. Didn't search for a better solution.
set(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE})

# Torch

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Metal and NetGen

if("${METAL_CPP_DIR}" STREQUAL "")
  message(
    FATAL_ERROR "METAL_CPP_DIR cmake variable must be set to metal-cpp path.")
endif()

include_directories(include)
include_directories(${METAL_CPP_DIR})

file(GLOB src_metal_kernels "kernels/*.metal")
file(GLOB src_netgen "src/*.cpp")
set_source_files_properties(
  ${src_metal_kernels} PROPERTIES LANGUAGE METAL COMPILE_OPTIONS
                                                 "-I${PROJECT_SOURCE_DIR}")

add_library(NetGen SHARED ${src_netgen} ${src_metal_kernels})

target_link_libraries(NetGen PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(NetGen PUBLIC "-framework Foundation")
target_link_libraries(NetGen PUBLIC "-framework QuartzCore")
target_link_libraries(NetGen PUBLIC "-framework Metal")
target_link_libraries(NetGen PUBLIC "-framework MetalKit")
target_compile_definitions(
  NetGen
  PUBLIC
    METAL_DEFAULT_LIBRARY="${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/default.metallib"
)

add_executable(nanoGPT NanoGPT.cpp)
target_link_libraries(nanoGPT NetGen)

add_subdirectory(test)
