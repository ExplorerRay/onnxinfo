FROM explorerray/onnxinfo:latest

COPY . /onnxinfo

WORKDIR /onnxinfo

RUN wget "https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto3" -O ./third_party/onnx/onnx.proto3

RUN mkdir models \
 && wget "https://github.com/onnx/models/raw/refs/heads/main/Computer_Vision/resnet18_Opset16_timm/resnet18_Opset16.onnx" -O ./models/resnet18_Opset16.onnx

RUN mv docker/CMakeLists.txt . \
 && cmake --build build/ --parallel 4

# CMD ["bash"]
CMD ["python3", "-m", "pytest", "tests/", "-v"]
