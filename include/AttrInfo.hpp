#pragma once

#include <vector>
#include "onnx.proto3.pb.h"

struct AttrInfo_Conv {
    std::vector<int64_t> kernel_shape; // got from Weight
    std::vector<int64_t> strides = {1, 1, 1}; // default 1
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0}; // default 0
    std::vector<int64_t> dilations = {1, 1, 1}; // default 1

    int64_t group = 1; // not used now
};

struct AttrInfo_MaxPool {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> strides = {1, 1, 1}; // default 1
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0}; // default 0
    std::vector<int64_t> dilations = {1, 1, 1}; // default 1

    int64_t ceil_mode = 0; // not used now
    int64_t storage_order = 0; // not used now
};

struct AttrInfo_Gemm {
    bool transA = false;
    bool transB = false;
};
