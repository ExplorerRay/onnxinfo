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

## Usage
Your Python file should be in the same directory as the `.so` file built by CMake.
```python
import onnxinfo

onnxinfo.summary('<onnx model path>')
```

## Functions
### Shape Inference
Support node types so far: Conv, Relu, MaxPool, AveragePool, Add, GlobalAveragePool, Flatten, Gemm

### Static Analysis
Will be run when shape inferencing by default.
<!-- You can use `infer_shapes(analyze = false)` to run shape inference only. -->

#### MACs
Doesn't count for non MACs operations like `Relu`, `MaxPool` and so on.

#### Parameters
Calculate trainable parameters for each node.

#### Memory (WIP)
Aims to calculate the memory usage of each node when input and output. (Bytes)

## Test
`python3 -m pytest -v`

Use model(resnet18_Opset16.onnx) from [ONNX Model Zoo](https://github.com/onnx/models/tree/main) to test.

## Docker
* Run `docker build -t onnxinfo -f docker/Dockerfile .` first.
    * You can type `docker run onnxinfo` to run tests.
    * Or type `docker run -it onnxinfo bash` to enter the environment which has onnxinfo.

## Limited Support
There are many node types in ONNX, but I only support some of them now.
