cmake_minimum_required(VERSION 3.25)
project(onnxinfo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(protobuf_BUILD_TESTS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

# protobuf
include(FetchContent)
FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG e6ab258b7ca407ed1bad8a2f04971e72b16f5409 # v28.2
)
FetchContent_MakeAvailable(protobuf)

# pybind11
# FetchContent_Declare(
#     pybind11
#     GIT_REPOSITORY https://github.com/pybind/pybind11.git
#     GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4 # v2.13.6
# )
# FetchContent_MakeAvailable(pybind11)

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

include_directories(
    ${protobuf_SOURCE_DIR}/third_party/abseil-cpp
    ${protobuf_SOURCE_DIR}
    ${onnxinfo_SOURCE_DIR}/include
    ${onnxinfo_SOURCE_DIR}/third_party/onnx
)

link_directories(
    ${protobuf_BINARY_DIR}
    ${protobuf_BINARY_DIR}/third_party/abseil-cpp/absl
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(
    ${PROJECT_NAME}

    ${SOURCES}
    ${onnxinfo_SOURCE_DIR}/third_party/onnx/onnx.proto3.pb.cc
)
add_dependencies(${PROJECT_NAME} protobuf::protoc)

target_link_libraries(${PROJECT_NAME} protobuf::libprotobuf-lite)
