# FROM debian:12-slim

# RUN apt-get update \
#  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
#  	build-essential cmake pybind11-dev python3.11-dev \
# 	git python3.11 python3-pytest \
#  && apt-get clean autoclean \
#  && apt-get autoremove --yes \
#  && rm -rf /var/lib/apt/lists/*

# RUN mkdir /onnxinfo

# COPY docker/CMakeLists.txt /onnxinfo

# WORKDIR /onnxinfo

# RUN cmake -S . -B build/
#  && cmake --build build/ --parallel 6

# CMD ["bash"]
### original Dockerfile for explorerray/onnxinfo:latest

FROM explorerray/onnxinfo:latest

COPY . /onnxinfo

WORKDIR /onnxinfo

RUN cmake --build build/ --parallel 6

# CMD ["bash"]
CMD ["python3", "-m", "pytest", "tests/", "-v"]
