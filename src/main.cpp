#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils.hpp"

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <onnx-model-path>\n";
    return 1;
  }

  // onnx::ModelProto model = read_onnx(argv[1]);
  // onnx::GraphProto graph = model.graph();
  // iterate_graph(graph);
  iterate_graph(read_onnx(argv[1]));

  return 0;
}

PYBIND11_MODULE(_onnxinfo, m) {
  m.doc() = "pybind11 onnxinfo module"; // optional module docstring

  m.def("read_onnx", &read_onnx, "A C++ function that read ONNX model");
  m.def("iterate_graph", &iterate_graph, "A C++ function that iterate ONNX graph");

  // pybind11::class_<onnx::ModelProto>(m, "ModelProto")
  //   .def(pybind11::init())
  //   .def_readonly("graph", &onnx::ModelProto::graph);
  pybind11::class_<onnx::GraphProto>(m, "GraphProto");
}
