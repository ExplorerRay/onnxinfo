#include "utils.hpp"

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <onnx-model-path>\n";
    return 1;
  }

  onnx::ModelProto model = read_onnx(argv[1]);

  onnx::GraphProto graph = model.graph();

  iterate_graph(graph);

  return 0;
}
