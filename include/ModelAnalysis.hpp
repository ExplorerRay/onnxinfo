#pragma once

#include "InferShape.hpp"
#include "onnx.proto3.pb.h"

constexpr int64_t MUL_MACS = 1;
constexpr int64_t ADD_MACS = 1;
// refers to onnx-tool/node.py
constexpr int64_t DIV_MACS = 4;

struct AnalyzeData {
  int64_t mac = 0;
  int64_t param = 0;
  int64_t mem = 0;
};

struct NodeAnalArgs {
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<int64_t> output_shape;
  std::unordered_map<std::string, size_t> ndname_to_size;
};

int64_t get_prod(std::vector<int64_t> &vec);
NodeAnalArgs get_anal_args(onnx::NodeProto &node, const std::unordered_map<std::string, std::vector<int64_t>> &ndname_to_shape, const std::unordered_map<std::string, size_t> &ndname_to_dtype_size);

AnalyzeData analyze_node_Conv(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size);
AnalyzeData analyze_node_Relu(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size);
AnalyzeData analyze_node_MaxPool(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size);
AnalyzeData analyze_node_Add(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size);
AnalyzeData analyze_node_GlobalAveragePool(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size);
AnalyzeData analyze_node_Flatten(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size);
AnalyzeData analyze_node_Gemm(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size);

AnalyzeData analyze_node(onnx::NodeProto &node, const std::unordered_map<std::string, std::vector<int64_t>> &ndname_to_shape, const std::unordered_map<std::string, size_t> &ndname_to_dtype_size);
