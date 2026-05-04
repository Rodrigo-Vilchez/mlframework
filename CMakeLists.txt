cmake_minimum_required(VERSION 3.25)
project(mlframework VERSION 0.5.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

find_package(BLAS REQUIRED)

find_library(CUDART_LIB cudart HINTS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
find_library(CUBLAS_LIB cublas HINTS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
find_path(CUDA_INCLUDE_DIR cuda_runtime.h HINTS /usr/local/cuda/include)

add_library(mlframework_core
    src/tensor.cpp
    src/ops.cpp
    src/optimizer.cpp
    src/module.cpp
    src/layer.cpp
    src/loss.cpp
    src/dataloader.cpp
    src/mlp.cpp
    src/cnn.cpp
    src/model_io.cpp
    src/cuda/tensor_cuda.cu
    src/cuda/ops_cuda.cu
)

target_include_directories(mlframework_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIR}
)

target_link_libraries(mlframework_core PUBLIC
    BLAS::BLAS
    ${CUDART_LIB}
    ${CUBLAS_LIB}
)

set_target_properties(mlframework_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(mlframework cli/main.cpp)
target_link_libraries(mlframework PRIVATE mlframework_core)

add_executable(tests tests/test_tensor.cpp)
target_link_libraries(tests PRIVATE mlframework_core)