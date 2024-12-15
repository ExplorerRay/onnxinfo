#include "utils.hpp"

onnx::ModelProto read_onnx(const std::string &filename) {
  // open file and move current position in file to the end
  std::ifstream input(filename, std::ios::ate | std::ios::binary);

  // check file existence
  if(!input.good()) {
    throw std::invalid_argument("File not found: " + filename);
  }

  std::streamsize size = input.tellg();  // get current position in file for file size
  input.seekg(0, std::ios::beg);  // move to start of file

  std::vector<char> buffer(size);
  input.read(buffer.data(), size);  // read raw data

  onnx::ModelProto model;
  model.ParseFromArray(buffer.data(), size);  // parse protobuf

  return model;
}

std::string dims_vec_to_str(const std::vector<int64_t> &dims) {
  std::string str = "[";
  for (size_t i = 0; i < dims.size() - 1; ++i) {
    str += std::to_string(dims[i]) + ", ";
  }
  str += std::to_string(dims.back()) + "]";

  return str;
}

void set_vec_to_shape(onnx::ValueInfoProto *val_info, const std::vector<int64_t> &dims) {
  auto shape = val_info->mutable_type()->mutable_tensor_type()->mutable_shape();
  for (auto dim: dims) {
    auto new_dim = shape->add_dim();
    new_dim->set_dim_value(dim);
  }
}

std::string string_trimmer(const std::string &inputString, const size_t maxLen) {
  std::string trimmedString = inputString;

  if (trimmedString.length() > maxLen) {
    trimmedString = trimmedString.substr(0, maxLen - 3) + "...";
  }

  return trimmedString;
}

std::string int64_to_str(int64_t num) {
  std::string str = std::to_string(num);
  // add comma
  for (int i = str.length() - 3; i > 0; i -= 3) {
    str.insert(i, ",");
  }
  return str;
}
