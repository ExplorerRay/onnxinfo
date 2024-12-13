#pragma once

#include <unordered_map>
#include "onnx.proto3.pb.h"
#include "AttrInfo.hpp"
#include "ModelAnalysis.hpp"

constexpr size_t INDENT = 30;
constexpr size_t TP_IND = 20;
constexpr size_t SP_IND = 23;
constexpr size_t DT_IND = 16;

class InferShapeImpl {
public:
  InferShapeImpl() = default;
  InferShapeImpl(const onnx::GraphProto &in_graph) : graph(in_graph) {}

  ~InferShapeImpl() = default;

  void set_io_iniz_shape_to_map(bool analyze);
  void infer_shapes(bool analyze);
  void infer_shapes();
  void print_summary();

  const str_shape_map_t get_ndname_to_shape() {
    return this->ndname_to_shape;
  }

private:
  onnx::GraphProto graph;
  str_shape_map_t ndname_to_shape;
  std::unordered_map<std::string, struct AnalyzeData> ndname_to_anal_data;
  str_sz_map_t ndname_to_dtype_size;

  // TODO: more op types
  void infer_shapes_Conv(onnx::NodeProto &node);
  void infer_shapes_Relu(onnx::NodeProto &node);
  void infer_shapes_MaxPool(onnx::NodeProto &node);
  void infer_shapes_Add(onnx::NodeProto &node);
  void infer_shapes_GlobalAveragePool(onnx::NodeProto &node);
  void infer_shapes_Flatten(onnx::NodeProto &node);
  void infer_shapes_Gemm(onnx::NodeProto &node);
};
