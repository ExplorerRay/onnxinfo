# onnxinfo
A tool to show ONNX model summary like torchinfo

## Requirements
* cmake (install by yourself)
* [Protobuf](https://github.com/protocolbuffers/protobuf)
* [Pybind11](https://github.com/pybind/pybind11)

## Build
1. `git clone` this repository
2. `cd onnxinfo`
3. `cmake -S . -B build/`
4. `cmake --build build/ [--parallel <thread number>]` to build dependency and onnxinfo

## Test
`python3 -m pytest -v`

Use model(resnet18_Opset16.onnx) from [ONNX Model Zoo](https://github.com/onnx/models/tree/main) to test.

## Docker
* Run `docker build -t onnxinfo -f docker/Dockerfile .` first.
    * You can type `docker run onnxinfo` to run tests.
    * Or type `docker run -it onnxinfo bash` to enter the environment which has onnxinfo.
