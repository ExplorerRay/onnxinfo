cmake_minimum_required(VERSION 3.25)
project(onnxinfo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(protobuf_BUILD_TESTS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# protobuf
include(FetchContent)
FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG e6ab258b7ca407ed1bad8a2f04971e72b16f5409 # v28.2
)
FetchContent_MakeAvailable(protobuf)

# generate onnx.proto3.pb.h and onnx.proto3.pb.cc
if (NOT EXISTS ${onnxinfo_SOURCE_DIR}/third_party/onnx/onnx.proto3)
    file(MAKE_DIRECTORY ${onnxinfo_SOURCE_DIR}/third_party/onnx)
    execute_process(
        COMMAND wget "https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto3" -O ${onnxinfo_SOURCE_DIR}/third_party/onnx/onnx.proto3
    )
endif()
add_custom_command(
    OUTPUT ${onnxinfo_SOURCE_DIR}/third_party/onnx/onnx.proto3.pb.h
           ${onnxinfo_SOURCE_DIR}/third_party/onnx/onnx.proto3.pb.cc
    COMMAND protobuf::protoc --cpp_out=${onnxinfo_SOURCE_DIR}/third_party/onnx -I${onnxinfo_SOURCE_DIR}/third_party/onnx onnx.proto3
)

# pybind11
find_package(pybind11 CONFIG)

# start building onnxinfo library for pybind11
include_directories(
    ${protobuf_SOURCE_DIR}/third_party/abseil-cpp
    ${protobuf_SOURCE_DIR}
    ${pybind11_SOURCE_DIR}/include
    ${onnxinfo_SOURCE_DIR}/include
    ${onnxinfo_SOURCE_DIR}/third_party/onnx
)

link_directories(
    ${protobuf_BINARY_DIR}
    ${protobuf_BINARY_DIR}/third_party/abseil-cpp/absl
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fPIC -flto=auto")
file(GLOB_RECURSE SOURCES "${onnxinfo_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM SOURCES "${onnxinfo_SOURCE_DIR}/src/main.cpp")

add_library(
    ${PROJECT_NAME} STATIC

    ${SOURCES}
    ${onnxinfo_SOURCE_DIR}/third_party/onnx/onnx.proto3.pb.cc
)
add_dependencies(${PROJECT_NAME} protobuf::protoc)
target_link_libraries(${PROJECT_NAME} protobuf::libprotobuf)

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${onnxinfo_SOURCE_DIR}/tests)
# pybind11_add_module(_onnxinfo src/main.cpp)
# target_link_libraries(_onnxinfo PRIVATE ${PROJECT_NAME})
