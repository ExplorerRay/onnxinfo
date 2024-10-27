#include "utils.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <onnx-model-path>\n";
    return 1;
  }

  onnx::ModelProto model = read_onnx(argv[1]);

  onnx::GraphProto graph = model.graph();

  iterate_graph(graph);

  return 0;
}

PYBIND11_MODULE(_onnxinfo, m) {
    m.doc() = "pybind11 onnxinfo module"; // optional module docstring

    m.def("read_onnx", &read_onnx, "A C++ function that read ONNX model");
    m.def("iterate_graph", &iterate_graph, "A C++ function that iterate ONNX graph");
}
