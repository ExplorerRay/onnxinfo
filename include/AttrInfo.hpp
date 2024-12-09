#pragma once

#include <vector>
#include "onnx.proto3.pb.h"

struct AttrInfo_Conv {
    std::vector<int64_t> kernel_shape; // got from Weight
    std::vector<int64_t> strides; // default 1
    std::vector<int64_t> pads; // default 0
    std::vector<int64_t> dilations; // default 1

    int64_t group = 1; // not used now

    void set_default_attr(size_t shape_len) {
        for (size_t i = 0; i < shape_len - 2; ++i) {
            strides.emplace_back(1);
            pads.emplace_back(0);
            pads.emplace_back(0);
            dilations.emplace_back(1);
        }
    }
};

struct AttrInfo_MaxPool {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> strides; // default 1
    std::vector<int64_t> pads; // default 0
    std::vector<int64_t> dilations; // default 1

    int64_t ceil_mode = 0; // not used now
    int64_t storage_order = 0; // not used now

    void set_default_attr(size_t shape_len) {
        for (size_t i = 0; i < shape_len - 2; ++i) {
            strides.emplace_back(1);
            pads.emplace_back(0);
            pads.emplace_back(0);
            dilations.emplace_back(1);
        }
    }
};

struct AttrInfo_Gemm {
    bool transA = false;
    bool transB = false;
};
