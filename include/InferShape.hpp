#pragma once

#include <unordered_map>
#include "onnx.proto3.pb.h"
#include "AttrInfo.hpp"

#define INDENT 30

class InferShapeImpl {
public:
    InferShapeImpl() = default;
    InferShapeImpl(const onnx::GraphProto &in_graph) : graph(in_graph) {}

    ~InferShapeImpl() = default;

    void set_io_iniz_shape_to_map();
    void infer_shapes();
    void print_summary();

    std::unordered_map<std::string, std::vector<int64_t>> get_ndname_to_shape() {
        return this->ndname_to_shape;
    }

private:
    onnx::GraphProto graph;
    std::unordered_map<std::string, std::vector<int64_t>> ndname_to_shape;

    // TODO: check dimensions & attributes needed or default
    void infer_shapes_Conv(onnx::NodeProto &node);
    void infer_shapes_Relu(onnx::NodeProto &node);
    void infer_shapes_MaxPool(onnx::NodeProto &node);
    void infer_shapes_Add(onnx::NodeProto &node);
    void infer_shapes_GlobalAveragePool(onnx::NodeProto &node);
    void infer_shapes_Flatten(onnx::NodeProto &node);
    void infer_shapes_Gemm(onnx::NodeProto &node);
};
