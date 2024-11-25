FROM debian:12-slim

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y \
 	build-essential cmake wget git \
  	pybind11-dev python3.11-dev python3.11 \
	python3-onnx python3-pytest \
 && apt-get clean autoclean \
 && apt-get autoremove --yes \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /onnxinfo

COPY docker/original/CMakeLists.txt /onnxinfo

WORKDIR /onnxinfo

RUN cmake -S . -B build/ \
 && cmake --build build/ --parallel 6

CMD ["bash"]
### original Dockerfile for explorerray/onnxinfo:latest
