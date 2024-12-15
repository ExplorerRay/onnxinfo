#include "ModelAnalysis.hpp"
#include "utils.hpp"

int64_t get_prod(std::vector<int64_t> &vec) {
  int64_t prod = 1;
  for (int64_t v : vec) {
    prod *= v;
  }
  return prod;
}

NodeAnalArgs get_anal_args(onnx::NodeProto &node,
  const str_shape_map_t &shape_map,
  const str_sz_map_t &sz_map)
{
  NodeAnalArgs anal_args;
  std::vector<std::vector<int64_t>> input_shapes;
  str_sz_map_t ndname_to_size;
  for (auto input : node.input()) {
    input_shapes.emplace_back(shape_map.at(input));
    ndname_to_size[input] = sz_map.at(input);
  }
  std::vector<int64_t> output_shape = shape_map.at(node.output(0));
  ndname_to_size[node.output(0)] = sz_map.at(node.output(0));

  anal_args.input_shapes = input_shapes;
  anal_args.output_shape = output_shape;
  anal_args.ndname_to_size = ndname_to_size;

  return anal_args;
}

bool AnalyzeImpl::need_mem_add(const std::string &name) {
  // check is initializer or not, and check is visited or not
  bool in_map = this->m_name_to_count.find(name) != this->m_name_to_count.end();
  return (in_map && this->m_name_to_count[name] == 0) || !in_map;
}

void AnalyzeImpl::set_iniz_checked(const std::string &name) {
  this->m_name_to_count[name] = 0;
}

AnalyzeData AnalyzeImpl::analyze_node_Conv(onnx::NodeProto &node, NodeAnalArgs &anal_args) {
  AnalyzeData data;
  std::vector<std::vector<int64_t>> input_shapes = anal_args.input_shapes;
  std::vector<int64_t> output_shape = anal_args.output_shape;
  str_sz_map_t ndname_to_size = anal_args.ndname_to_size;

  // MACs
  std::vector<int64_t> kernel_shape = input_shapes[1];
  std::vector<int64_t> reduce_shape(kernel_shape.begin() + 1, kernel_shape.end());
  int64_t out_prod = get_prod(output_shape);
  data.flop = out_prod * get_prod(reduce_shape) * MUL_FLOPS;

  if (node.input_size() == 3) {  // with bias
    data.flop += out_prod * ADD_FLOPS;
  }

  // Parameters
  for (size_t i = 1; i < input_shapes.size(); ++i) {
    data.param += get_prod(input_shapes[i]);
  }

  return data;
}

AnalyzeData AnalyzeImpl::analyze_node_Relu(onnx::NodeProto &node, NodeAnalArgs &anal_args) {
  AnalyzeData data;
  std::vector<std::vector<int64_t>> input_shapes = anal_args.input_shapes;
  std::vector<int64_t> output_shape = anal_args.output_shape;
  str_sz_map_t ndname_to_size = anal_args.ndname_to_size;

  data.param = 0;  // no trainable parameters

  // MACs
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    data.flop += get_prod(input_shapes[i]) * CMP_FLOPS;
  }

  return data;
}

AnalyzeData AnalyzeImpl::analyze_node_MaxPool(onnx::NodeProto &node, NodeAnalArgs &anal_args) {
  AnalyzeData data;
  std::vector<std::vector<int64_t>> input_shapes = anal_args.input_shapes;
  std::vector<int64_t> output_shape = anal_args.output_shape;
  str_sz_map_t ndname_to_size = anal_args.ndname_to_size;

  // MACs
  std::vector<int64_t> kernel_shape;
  for (auto attr : node.attribute()) {
    if (attr.name() == "kernel_shape") {
      for (int i = 0; i < attr.ints_size(); ++i) {
        kernel_shape.emplace_back(attr.ints(i));
      }
    }
  }
  data.flop = get_prod(output_shape) * get_prod(kernel_shape) * CMP_FLOPS;

  // Parameters
  data.param = 0;  // no trainable parameters

  return data;
}

AnalyzeData AnalyzeImpl::analyze_node_AveragePool(onnx::NodeProto &node, NodeAnalArgs &anal_args) {
  AnalyzeData data;
  std::vector<std::vector<int64_t>> input_shapes = anal_args.input_shapes;
  std::vector<int64_t> output_shape = anal_args.output_shape;
  str_sz_map_t ndname_to_size = anal_args.ndname_to_size;

  // MACs
  std::vector<int64_t> kernel_shape;
  for (auto attr : node.attribute()) {
    if (attr.name() == "kernel_shape") {
      for (int i = 0; i < attr.ints_size(); ++i) {
        kernel_shape.emplace_back(attr.ints(i));
      }
    }
  }
  data.flop = get_prod(output_shape) * get_prod(kernel_shape) * CMP_FLOPS;

  // Parameters
  data.param = 0;  // no trainable parameters

  return data;
}

AnalyzeData AnalyzeImpl::analyze_node_Add(onnx::NodeProto &node, NodeAnalArgs &anal_args) {
  AnalyzeData data;
  std::vector<std::vector<int64_t>> input_shapes = anal_args.input_shapes;
  std::vector<int64_t> output_shape = anal_args.output_shape;
  str_sz_map_t ndname_to_size = anal_args.ndname_to_size;

  // MACs
  data.flop += get_prod(output_shape) * ADD_FLOPS;


  // Parameters
  data.param = 0;  // no trainable parameters

  return data;
}

AnalyzeData AnalyzeImpl::analyze_node_GlobalAveragePool(onnx::NodeProto &node, NodeAnalArgs &anal_args) {
  AnalyzeData data;
  std::vector<std::vector<int64_t>> input_shapes = anal_args.input_shapes;
  std::vector<int64_t> output_shape = anal_args.output_shape;
  str_sz_map_t ndname_to_size = anal_args.ndname_to_size;

  // MACs
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    data.flop += get_prod(input_shapes[i]) * ADD_FLOPS;
  }
  data.flop += get_prod(output_shape) * DIV_FLOPS;

  // Parameters
  data.param = 0;  // no trainable parameters

  return data;
}

AnalyzeData AnalyzeImpl::analyze_node_Flatten(onnx::NodeProto &node, NodeAnalArgs &anal_args) {
  AnalyzeData data;
  std::vector<std::vector<int64_t>> input_shapes = anal_args.input_shapes;
  std::vector<int64_t> output_shape = anal_args.output_shape;
  str_sz_map_t ndname_to_size = anal_args.ndname_to_size;

  // MACs
  data.flop = 0;  // non MACs operation

  // Parameters
  data.param = 0;  // no trainable parameters

  return data;
}

AnalyzeData AnalyzeImpl::analyze_node_Gemm(onnx::NodeProto &node, NodeAnalArgs &anal_args) {
  AnalyzeData data;
  std::vector<std::vector<int64_t>> input_shapes = anal_args.input_shapes;
  std::vector<int64_t> output_shape = anal_args.output_shape;
  str_sz_map_t ndname_to_size = anal_args.ndname_to_size;

  // MACs
  data.flop = get_prod(input_shapes[0]) * get_prod(output_shape) * MUL_FLOPS;
  if (node.input_size() == 3) {  // with bias
    data.flop += get_prod(output_shape) * ADD_FLOPS;
  }

  // Parameters
  data.param = get_prod(input_shapes[0]) * get_prod(output_shape) + get_prod(output_shape);

  return data;
}

AnalyzeData AnalyzeImpl::analyze_node(onnx::NodeProto &node,
  const str_shape_map_t &ndname_to_shape,
  const str_sz_map_t &ndname_to_dtype_size)
{
  AnalyzeData data;
  NodeAnalArgs anal_args = get_anal_args(node, ndname_to_shape, ndname_to_dtype_size);
  if (node.op_type() == "Conv") {
    data = analyze_node_Conv(node, anal_args);
  }
  else if (node.op_type() == "Relu") {
    data = analyze_node_Relu(node, anal_args);
  }
  else if (node.op_type() == "MaxPool") {
    data = analyze_node_MaxPool(node, anal_args);
  }
  else if (node.op_type() == "AveragePool") {
    data = analyze_node_AveragePool(node, anal_args);
  }
  else if (node.op_type() == "Add") {
    data = analyze_node_Add(node, anal_args);
  }
  else if (node.op_type() == "GlobalAveragePool") {
    data = analyze_node_GlobalAveragePool(node, anal_args);
  }
  else if (node.op_type() == "Flatten") {
    data = analyze_node_Flatten(node, anal_args);
  }
  else if (node.op_type() == "Gemm") {
    data = analyze_node_Gemm(node, anal_args);
  }
  else {
    std::cerr << "Error: " << node.op_type() << " not supported now\n";
    exit(1);
  }

  // Memory
  for (int i = 0; i < node.input_size(); ++i) {
    if (need_mem_add(node.input(i))) {
      std::vector<int64_t> shape = ndname_to_shape.at(node.input(i));
      size_t sz = ndname_to_dtype_size.at(node.input(i));
      data.mem += get_prod(shape) * sz;
      this->m_name_to_count[node.input(i)] = 1;
    }
  }
  for (int i = 0; i < node.output_size(); ++i) {
    std::vector<int64_t> shape = ndname_to_shape.at(node.output(i));
    size_t sz = ndname_to_dtype_size.at(node.output(i));
    data.mem += get_prod(shape) * sz;
  }

  return data;
}
