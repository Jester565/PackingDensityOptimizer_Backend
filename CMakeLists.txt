cmake_minimum_required(VERSION 3.8)

set (PROJECT_NAME CudaCircles)

project(${PROJECT_NAME})

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/csrc/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/*.h")

include_directories(${AWS_SDK}/include)

find_package(Threads)
add_executable(${PROJECT_NAME} ${header_files} ${cpu_source_files} ${gpu_source_files})

find_package(CUDA 9.1)
CUDA_ADD_LIBRARY(${PROJECT_NAME}_gpu  ${header_files} ${gpu_source_files})
target_link_libraries(${PROJECT_NAME} 
${PROJECT_NAME}_gpu
${CMAKE_THREAD_LIBS_INIT} 
${AWS_SDK}/lib/x64/Release/v140/aws-cpp-sdk-core.lib
${AWS_SDK}/lib/x64/Release/v140/aws-cpp-sdk-sqs.lib)

add_definitions(-DUSE_CUDA)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED 11)