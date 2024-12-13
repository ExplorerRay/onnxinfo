#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils.hpp"
#include "InferShape.hpp"

void summary(std::string model_path) {
  onnx::ModelProto model_proto = read_onnx(model_path);
  InferShapeImpl infer_shape_impl(model_proto.graph());
  infer_shape_impl.infer_shapes();
  infer_shape_impl.print_summary();
}

namespace py = pybind11;

PYBIND11_MODULE(onnxinfo, m) {
  m.doc() = "pybind11 onnxinfo module"; // optional module docstring

  m.def("read_onnx", &read_onnx, "A C++ function that read ONNX model");
  m.def("summary", &summary, "A C++ function that print summary of ONNX model");

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
