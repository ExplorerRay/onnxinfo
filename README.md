# onnxinfo
A tool to show ONNX model summary like torchinfo

## Requirements
* cmake (install by yourself)
* [Protobuf](https://github.com/protocolbuffers/protobuf)

## Build
1. `git clone` this repository
2. `cd onnxinfo`
3. `cmake -S . -B build/`
4. `cmake --build build/ [--parallel <thread number>]` to build dependency and onnxinfo
