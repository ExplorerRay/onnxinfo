#pragma once

#include "onnx.proto3.pb.h"

using str_sz_map_t = std::unordered_map<std::string, size_t>;
using str_shape_map_t = std::unordered_map<std::string, std::vector<int64_t>>;

constexpr int64_t MUL_FLOPS = 1;
constexpr int64_t ADD_FLOPS = 1;
constexpr int64_t CMP_FLOPS = 1;
constexpr int64_t DIV_FLOPS = 1;

struct AnalyzeData {
  int64_t flop = 0;
  int64_t param = 0;
  int64_t mem = 0;
};

struct NodeAnalArgs {
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<int64_t> output_shape;
  str_sz_map_t ndname_to_size;
};

int64_t get_prod(std::vector<int64_t> &vec);
NodeAnalArgs get_anal_args(onnx::NodeProto &node,
  const str_shape_map_t &shape_map,
  const str_sz_map_t &sz_map);

class AnalyzeImpl {
public:
  AnalyzeImpl() = default;
  ~AnalyzeImpl() = default;
  AnalyzeImpl(const AnalyzeImpl &other) : m_name_to_count(other.m_name_to_count) {}
  AnalyzeImpl(AnalyzeImpl &&other) noexcept : m_name_to_count(std::move(other.m_name_to_count)) {}
  AnalyzeImpl& operator=(const AnalyzeImpl &other) {
    if (this != &other) {
      this->m_name_to_count = other.m_name_to_count;
    }
    return *this;
  }
  AnalyzeImpl& operator=(AnalyzeImpl &&other) noexcept {
    if (this != &other) {
      this->m_name_to_count = std::move(other.m_name_to_count);
    }
    return *this;
  }

  AnalyzeData analyze_node(onnx::NodeProto &node,
    const str_shape_map_t &ndname_to_shape,
    const str_sz_map_t &ndname_to_dtype_size);

  bool need_mem_add(const std::string &name);
  void set_iniz_checked(const std::string &name);

private:
  str_sz_map_t m_name_to_count;

  // TODO: more op types
  AnalyzeData analyze_node_Conv(onnx::NodeProto &node, NodeAnalArgs &anal_args);
  AnalyzeData analyze_node_Relu(onnx::NodeProto &node, NodeAnalArgs &anal_args);
  AnalyzeData analyze_node_MaxPool(onnx::NodeProto &node, NodeAnalArgs &anal_args);
  AnalyzeData analyze_node_AveragePool(onnx::NodeProto &node, NodeAnalArgs &anal_args);
  AnalyzeData analyze_node_Add(onnx::NodeProto &node, NodeAnalArgs &anal_args);
  AnalyzeData analyze_node_GlobalAveragePool(onnx::NodeProto &node, NodeAnalArgs &anal_args);
  AnalyzeData analyze_node_Flatten(onnx::NodeProto &node, NodeAnalArgs &anal_args);
  AnalyzeData analyze_node_Gemm(onnx::NodeProto &node, NodeAnalArgs &anal_args);
};
