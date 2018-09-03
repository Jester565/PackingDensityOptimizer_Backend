cmake_minimum_required(VERSION 3.5)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set (PROJECT_NAME CudaCircles)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(${PROJECT_NAME})

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/csrc/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/*.h")

message(STATUS ${header_files})

include_directories(${AWS_SDK}/include)
include_directories(${CMAKE_SOURCE_DIR}/${PROJECT_NAME})

find_package(Threads)
add_executable(${PROJECT_NAME} ${header_files} ${cpu_source_files} ${gpu_source_files})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA 7.5)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60; -std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35;-std=c++11)
CUDA_ADD_LIBRARY(${PROJECT_NAME}_gpu ${header_files} ${gpu_source_files})
target_link_libraries(${PROJECT_NAME}_gpu
  cudart
  cudadevrt)
#set_target_properties(${PROJECT_NAME}_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} 
  ${PROJECT_NAME}_gpu
  aws-cpp-sdk-dynamodb
  aws-cpp-sdk-sqs
  aws-cpp-sdk-core
  cudart
  cudadevrt
  ${CMAKE_THREAD_LIBS_INIT})
add_definitions(-DUSE_CUDA)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME}_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
