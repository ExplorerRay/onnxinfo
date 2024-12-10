#pragma once

#include <fstream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include "onnx.proto3.pb.h"

onnx::ModelProto read_onnx(const std::string &filename);
void print_dim(const ::onnx::TensorShapeProto_Dimension &dim);
void print_dims_vec(const std::vector<int64_t> &dims);
std::string dims_vec_to_str(const std::vector<int64_t> &dims);
void set_vec_to_shape(onnx::ValueInfoProto *val_info, const std::vector<int64_t> &dims);
std::string string_trimmer(const std::string &inputString, const size_t maxLen);
std::string int64_to_str(int64_t num);
