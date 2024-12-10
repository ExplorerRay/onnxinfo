#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils.hpp"
#include "InferShape.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_onnxinfo, m) {
  m.doc() = "pybind11 onnxinfo module"; // optional module docstring

  m.def("read_onnx", &read_onnx, "A C++ function that read ONNX model");
  // m.def("iterate_graph", &iterate_graph, "A C++ function that iterate ONNX graph");

  py::class_<InferShapeImpl>(m, "InferShapeImpl")
    .def(py::init<const onnx::GraphProto &>())
    .def("set_io_iniz_shape_to_map", &InferShapeImpl::set_io_iniz_shape_to_map)
    .def("infer_shapes", py::overload_cast<>(&InferShapeImpl::infer_shapes))
    .def("print_summary", &InferShapeImpl::print_summary)
    .def("get_ndname_to_shape", &InferShapeImpl::get_ndname_to_shape);

  py::class_<onnx::ModelProto>(m, "ModelProto")
    .def("graph", &onnx::ModelProto::graph);
  py::class_<onnx::GraphProto>(m, "GraphProto");
}
