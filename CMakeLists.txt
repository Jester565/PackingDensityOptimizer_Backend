cmake_minimum_required(VERSION 3.5)

set (PROJECT_NAME CudaCircles)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

project(${PROJECT_NAME})

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/csrc/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/*.h")

message(STATUS ${header_files})

include_directories(${AWS_SDK}/include)
include_directories(${CMAKE_SOURCE_DIR}/${PROJECT_NAME})

find_package(Threads)
add_executable(${PROJECT_NAME} ${header_files} ${cpu_source_files} ${gpu_source_files})

find_package(CUDA 7.5)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE;-D_FORCE_INLINES")
CUDA_ADD_LIBRARY(${PROJECT_NAME}_gpu ${header_files} ${gpu_source_files})
target_link_libraries(${PROJECT_NAME} 
${PROJECT_NAME}_gpu
aws-cpp-sdk-sqs
aws-cpp-sdk-core
${CMAKE_THREAD_LIBS_INIT})

add_definitions(-DUSE_CUDA)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED 11)
