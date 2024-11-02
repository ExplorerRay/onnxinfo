#include "utils.hpp"

onnx::ModelProto read_onnx(const std::string &filename) {
  // open file and move current position in file to the end
  std::ifstream input(filename, std::ios::ate | std::ios::binary);

  // check file existence
  if(!input.good()) {
    throw std::invalid_argument("File not found: " + filename);
  }

  std::streamsize size = input.tellg(); // get current position in file for file size
  input.seekg(0, std::ios::beg); // move to start of file

  std::vector<char> buffer(size);
  input.read(buffer.data(), size); // read raw data

  onnx::ModelProto model;
  model.ParseFromArray(buffer.data(), size); // parse protobuf

  return model;
}

void iterate_graph(const ::onnx::GraphProto &graph) {
  onnx::NodeProto node;
  for (int i = 0; i < graph.node_size(); i++) {
    std::cout << "node NO." << i << '\n';
    node = graph.node(i);

    // input & output
    std::cout << "  inputs:\n";
    for (auto input: node.input()) {
      std::cout << "    " << input << '\n';
    }
    std::cout << "  outputs:\n";
    for (auto output: node.output()) {
      std::cout << "    " << output << '\n';
    }
  }
}
