#pragma once

#include <fstream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include "onnx.proto3.pb.h"

onnx::ModelProto read_onnx(const std::string &filename);

void iterate_graph(const ::onnx::GraphProto &graph);
