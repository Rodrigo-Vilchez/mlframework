cmake_minimum_required(VERSION 3.25)
project(mlframework VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Core library
add_library(mlframework_core
    src/tensor.cpp
    src/ops.cpp
    src/optimizer.cpp
    src/layer.cpp
    src/loss.cpp
    src/dataloader.cpp
)

target_include_directories(mlframework_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# CLI
add_executable(mlframework cli/main.cpp)
target_link_libraries(mlframework PRIVATE mlframework_core)

# Tests
add_executable(tests tests/test_tensor.cpp)
target_link_libraries(tests PRIVATE mlframework_core)