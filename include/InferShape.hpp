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
  InferShapeImpl(const onnx::GraphProto &in_graph) : m_graph(in_graph) {}
  InferShapeImpl(const InferShapeImpl &other) : m_graph(other.m_graph) {}
  InferShapeImpl(InferShapeImpl&& other) noexcept : m_graph(std::move(other.m_graph)) {}
  InferShapeImpl& operator=(const InferShapeImpl &other) {
    if (this != &other) {
      this->m_graph = other.m_graph;
      this->m_name_to_shape = other.m_name_to_shape;
      this->m_name_to_anal_data = other.m_name_to_anal_data;
      this->m_name_to_dtsize = other.m_name_to_dtsize;
      this->m_analyzer = other.m_analyzer;
    }
    return *this;
  }
  InferShapeImpl& operator=(InferShapeImpl&& other) noexcept {
    if (this != &other) {
      this->m_graph = std::move(other.m_graph);
      this->m_name_to_shape = std::move(other.m_name_to_shape);
      this->m_name_to_anal_data = std::move(other.m_name_to_anal_data);
      this->m_name_to_dtsize = std::move(other.m_name_to_dtsize);
      this->m_analyzer = std::move(other.m_analyzer);
    }
    return *this;
  }

  ~InferShapeImpl() = default;

  void set_io_iniz_shape_to_map(bool analyze);
  void infer_shapes(bool analyze);
  void infer_shapes();
  void print_summary();

  const str_shape_map_t get_ndname_to_shape() {
    return this->m_name_to_shape;
  }

private:
  onnx::GraphProto m_graph;
  str_shape_map_t m_name_to_shape;
  std::unordered_map<std::string, struct AnalyzeData> m_name_to_anal_data;
  str_sz_map_t m_name_to_dtsize;
  AnalyzeImpl m_analyzer;

  // TODO: more op types
  void infer_shapes_Conv(onnx::NodeProto &node);
  void infer_shapes_Relu(onnx::NodeProto &node);
  void infer_shapes_MaxPool(onnx::NodeProto &node);
  void infer_shapes_Add(onnx::NodeProto &node);
  void infer_shapes_AveragePool(onnx::NodeProto &node);
  void infer_shapes_GlobalAveragePool(onnx::NodeProto &node);
  void infer_shapes_Flatten(onnx::NodeProto &node);
  void infer_shapes_Gemm(onnx::NodeProto &node);
};
