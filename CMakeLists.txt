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

# generate onnx.proto3.pb.h and onnx.proto3.pb.cc
if (NOT EXISTS ${onnxinfo_SOURCE_DIR}/third_party/onnx/onnx.proto3)
    file(MAKE_DIRECTORY ${onnxinfo_SOURCE_DIR}/third_party/onnx)
    execute_process(
        COMMAND wget "https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto3" -O ${onnxinfo_SOURCE_DIR}/third_party/onnx/onnx.proto3
    )
endif()
add_custom_command(
    OUTPUT ${onnxinfo_SOURCE_DIR}/include/onnx.proto3.pb.h
    COMMAND protobuf::protoc --cpp_out=${onnxinfo_SOURCE_DIR}/include -I${onnxinfo_SOURCE_DIR}/third_party/onnx onnx.proto3
)

include_directories(
    ${protobuf_SOURCE_DIR}/third_party/abseil-cpp
    ${protobuf_SOURCE_DIR}
    ${onnxinfo_SOURCE_DIR}/include
)

link_directories(
    ${protobuf_BINARY_DIR}
    ${protobuf_BINARY_DIR}/third_party/abseil-cpp/absl
)

add_executable(
    ${PROJECT_NAME}

    src/main.cpp
    ${onnxinfo_SOURCE_DIR}/include/onnx.proto3.pb.cc
    ${onnxinfo_SOURCE_DIR}/include/onnx.proto3.pb.h
)
add_dependencies(${PROJECT_NAME} protobuf::protoc)

target_link_libraries(${PROJECT_NAME} protobuf::libprotobuf-lite)

# add_subdirectory(third_party/pybind11)
