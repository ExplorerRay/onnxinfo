#include <math.h>
#include <iomanip>
#include "InferShape.hpp"
#include "utils.hpp"

void InferShapeImpl::set_io_iniz_shape_to_map(bool analyze) {
  for (auto input : this->m_graph.input()) {
    auto shape = input.type().tensor_type().shape();
    std::vector<int64_t> shape_vec = {};
    for (int i = 0; i < shape.dim_size(); ++i) {
      auto dim = shape.dim(i);
      if (dim.has_dim_value()) {
        shape_vec.emplace_back(dim.dim_value());
      }
    }
    this->m_name_to_shape[input.name()] = shape_vec;

    // get dtype size
    if (analyze) {
      if (input.type().tensor_type().elem_type() == 1) {
        // float32
        this->m_name_to_dtsize[input.name()] = 4;
      }
    }
  }

  for (auto initializer : this->m_graph.initializer()) {
    auto shape = initializer.dims();
    std::vector<int64_t> shape_vec = {};
    for (int i = 0; i < shape.size(); ++i) {
      shape_vec.emplace_back(shape.Get(i));
    }
    this->m_name_to_shape[initializer.name()] = shape_vec;

    if (analyze) {
      this->m_analyzer.set_iniz_checked(initializer.name());

      // get dtype size
      if (initializer.data_type() == 1) {
        this->m_name_to_dtsize[initializer.name()] = 4;
      }
    }
  }

  for (auto output : this->m_graph.output()) {
    auto shape = output.type().tensor_type().shape();
    std::vector<int64_t> shape_vec = {};
    for (int i = 0; i < shape.dim_size(); ++i) {
      auto dim = shape.dim(i);
      if (dim.has_dim_value()) {
        shape_vec.emplace_back(dim.dim_value());
      }
    }
    this->m_name_to_shape[output.name()] = shape_vec;

    // get dtype size
    if (analyze) {
      if (output.type().tensor_type().elem_type() == 1) {
        this->m_name_to_dtsize[output.name()] = 4;
      }
      else {
        std::cerr << "Error: dtype not supported now\n";
        exit(1);
      }
    }
  }
}

void InferShapeImpl::print_summary() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(0);

  if (this->m_name_to_anal_data.empty()) {
    size_t TOTAL_IND = INDENT + TP_IND + SP_IND * 2;
    std::cout << std::left << std::setw(INDENT) << "Name"
        << std::left << std::setw(TP_IND) << "Type"
        << std::left << std::setw(SP_IND) << "Input Shape"
        << std::left << std::setw(SP_IND) << "Output Shape" << '\n';
    std::cout << std::string(TOTAL_IND, '-') << '\n';

    for (auto node : m_graph.node()) {
      std::cout << std::left << std::setw(INDENT) << string_trimmer(node.name(), INDENT-5);

      std::cout << std::left << std::setw(TP_IND) << node.op_type();
      std::cout << std::setw(SP_IND) << dims_vec_to_str(this->m_name_to_shape[node.input(0)]);
      std::cout << std::setw(SP_IND) << dims_vec_to_str(this->m_name_to_shape[node.output(0)]);
      std::cout << '\n';
    }
  }
  else {
    size_t TOTAL_IND = INDENT + TP_IND + SP_IND * 2 + DT_IND * 3;
    std::cout << std::left << std::setw(INDENT) << "Name"
        << std::left << std::setw(TP_IND) << "Type"
        << std::left << std::setw(SP_IND) << "Input Shape"
        << std::left << std::setw(SP_IND) << "Output Shape"
        << std::left << std::setw(DT_IND) << "FLOPs"
        << std::left << std::setw(DT_IND) << "Params"
        << std::left << std::setw(DT_IND) << "Memory" << '\n';
    std::cout << std::string(TOTAL_IND, '-') << '\n';

    AnalyzeData total_data;
    for (auto node : m_graph.node()) {
      total_data.flop += this->m_name_to_anal_data[node.name()].flop;
      total_data.param += this->m_name_to_anal_data[node.name()].param;
      total_data.mem += this->m_name_to_anal_data[node.name()].mem;

      std::cout << std::left << std::setw(INDENT) << string_trimmer(node.name(), INDENT-5);

      std::cout << std::left << std::setw(TP_IND) << node.op_type();
      std::cout << std::setw(SP_IND) << dims_vec_to_str(this->m_name_to_shape[node.input(0)]);
      std::cout << std::setw(SP_IND) << dims_vec_to_str(this->m_name_to_shape[node.output(0)]);
      std::cout << std::setw(DT_IND) << int64_to_str(this->m_name_to_anal_data[node.name()].flop);
      std::cout << std::setw(DT_IND) << int64_to_str(this->m_name_to_anal_data[node.name()].param);
      std::cout << std::setw(DT_IND) << int64_to_str(this->m_name_to_anal_data[node.name()].mem);
      std::cout << '\n';
    }
    std::cout << std::left << std::setw(INDENT) << "Total";
    std::cout << std::left << std::setw(TP_IND) << "-";
    std::cout << std::setw(SP_IND) << dims_vec_to_str(this->m_name_to_shape[m_graph.input(0).name()]);
    std::cout << std::setw(SP_IND) << dims_vec_to_str(this->m_name_to_shape[m_graph.output(0).name()]);
    std::cout << std::setw(DT_IND) << int64_to_str(total_data.flop);
    std::cout << std::setw(DT_IND) << int64_to_str(total_data.param);
    std::cout << std::setw(DT_IND) << int64_to_str(total_data.mem);
    std::cout << '\n';
  }
}

void InferShapeImpl::infer_shapes_Conv(onnx::NodeProto &node) {
  struct AttrInfo_Conv attr_info;

  // get node input shapes
  std::vector<std::vector<int64_t>> input_shapes;
  for (int i = 0; i < node.input_size(); ++i) {
    auto ndinput = node.input(i);
    if (this->m_name_to_shape.find(ndinput) == this->m_name_to_shape.end()) {
      std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
      exit(1);
    }
    else {
      // get shape from ndname_to_shape
      std::vector<int64_t> shape = this->m_name_to_shape[ndinput];
      if (i == 0) attr_info.set_default_attr(shape.size());
      input_shapes.emplace_back(shape);
    }
  }

  // get attributes (kernel_shape, strides, pads, dilations, group)
  for (auto attr : node.attribute()) {
    if (attr.name() == "kernel_shape") {
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.kernel_shape.emplace_back(attr.ints(i));
      }
    }
    else if (attr.name() == "strides") {
      attr_info.strides.clear();
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.strides.emplace_back(attr.ints(i));
      }
    }
    else if (attr.name() == "pads") {
      attr_info.pads.clear();
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.pads.emplace_back(attr.ints(i));
      }
    }
    else if (attr.name() == "dilations") {
      attr_info.dilations.clear();
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.dilations.emplace_back(attr.ints(i));
      }
    }
  }

  // calculate output shape after convolution
  auto input_shape = input_shapes[0];
  auto weight_shape = input_shapes[1];
  std::vector<int64_t> output_shape;
  output_shape.emplace_back(input_shape[0]);  // batch size
  output_shape.emplace_back(weight_shape[0]);  // number of channels
  for (size_t i = 0; i < attr_info.kernel_shape.size(); ++i) {
    int64_t pad = attr_info.pads[i];
    int64_t dilation = attr_info.dilations[i];
    int64_t kernel = attr_info.kernel_shape[i];
    int64_t stride = attr_info.strides[i];

    output_shape.emplace_back(
      (input_shape[i + 2] + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1
    );
  }

  // set value_info and update ndname_to_shape
  onnx::ValueInfoProto *val_info = this->m_graph.add_value_info();
  val_info->set_name(node.name());
  set_vec_to_shape(val_info, output_shape);

  this->m_name_to_shape[node.output(0)] = output_shape;
  this->m_name_to_dtsize[node.output(0)] = this->m_name_to_dtsize[node.input(0)];
}

void InferShapeImpl::infer_shapes_Relu(onnx::NodeProto &node) {
  // get node input shapes
  std::vector<std::vector<int64_t>> input_shapes;
  for (auto ndinput : node.input()) {
    if (this->m_name_to_shape.find(ndinput) == this->m_name_to_shape.end()) {
      std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
      return;
    }
    else {
      // get shape from ndname_to_shape
      input_shapes.emplace_back(this->m_name_to_shape[ndinput]);
    }
  }

  // set value_info and update ndname_to_shape
  onnx::ValueInfoProto *val_info = this->m_graph.add_value_info();
  val_info->set_name(node.name());
  set_vec_to_shape(val_info, input_shapes[0]);

  this->m_name_to_shape[node.output(0)] = input_shapes[0];
  this->m_name_to_dtsize[node.output(0)] = this->m_name_to_dtsize[node.input(0)];
}

void InferShapeImpl::infer_shapes_MaxPool(onnx::NodeProto &node) {
  struct AttrInfo_MaxPool attr_info;

  auto input_shape = this->m_name_to_shape[node.input(0)];

  attr_info.set_default_attr(input_shape.size());
  for (auto attr : node.attribute()) {
    // std::cout << "attr name: " << attr.name() << '\n';
    if (attr.name() == "kernel_shape") {  // required
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.kernel_shape.emplace_back(attr.ints(i));
      }
    }
    else if (attr.name() == "strides") {
      attr_info.strides.clear();
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.strides.emplace_back(attr.ints(i));
      }
    }
    else if (attr.name() == "pads") {
      attr_info.pads.clear();
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.pads.emplace_back(attr.ints(i));
      }
    }
    else if (attr.name() == "ceil_mode") {
      attr_info.ceil_mode = attr.i();
    }
    else if (attr.name() == "dilations") {
      attr_info.dilations.clear();
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.dilations.emplace_back(attr.ints(i));
      }
    }
  }

  // calculate output shape after maxpool
  std::vector<int64_t> output_shape;
  output_shape.emplace_back(input_shape[0]);  // batch size
  output_shape.emplace_back(input_shape[1]);  // number of channels
  for (size_t i = 0; i < attr_info.kernel_shape.size(); ++i) {
    int64_t pad = attr_info.pads[i];
    int64_t dilation = attr_info.dilations[i];
    int64_t kernel = attr_info.kernel_shape[i];
    int64_t stride = attr_info.strides[i];

    output_shape.emplace_back(
      (input_shape[i + 2] + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1
    );
  }

  // set value_info and update ndname_to_shape
  onnx::ValueInfoProto *val_info = this->m_graph.add_value_info();
  val_info->set_name(node.name());
  set_vec_to_shape(val_info, output_shape);

  this->m_name_to_shape[node.output(0)] = output_shape;
  this->m_name_to_dtsize[node.output(0)] = this->m_name_to_dtsize[node.input(0)];
}

void InferShapeImpl::infer_shapes_Add(onnx::NodeProto &node) {
  // get node input shapes
  std::vector<std::vector<int64_t>> input_shapes;
  for (auto ndinput : node.input()) {
    if (this->m_name_to_shape.find(ndinput) == this->m_name_to_shape.end()) {
      std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
      return;
    }
    else {
      // get shape from ndname_to_shape
      input_shapes.emplace_back(this->m_name_to_shape[ndinput]);
    }
  }

  // set value_info and update ndname_to_shape
  onnx::ValueInfoProto *val_info = this->m_graph.add_value_info();
  val_info->set_name(node.name());
  set_vec_to_shape(val_info, input_shapes[0]);

  this->m_name_to_shape[node.output(0)] = input_shapes[0];
  this->m_name_to_dtsize[node.output(0)] = this->m_name_to_dtsize[node.input(0)];
}

void InferShapeImpl::infer_shapes_AveragePool(onnx::NodeProto &node) {
  struct AttrInfo_MaxPool attr_info;

  auto input_shape = this->m_name_to_shape[node.input(0)];

  attr_info.set_default_attr(input_shape.size());
  for (auto attr : node.attribute()) {
    if (attr.name() == "kernel_shape") {  // required
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.kernel_shape.emplace_back(attr.ints(i));
      }
    }
    else if (attr.name() == "strides") {
      attr_info.strides.clear();
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.strides.emplace_back(attr.ints(i));
      }
    }
    else if (attr.name() == "pads") {
      attr_info.pads.clear();
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.pads.emplace_back(attr.ints(i));
      }
    }
    else if (attr.name() == "ceil_mode") {
      attr_info.ceil_mode = attr.i();
    }
    else if (attr.name() == "dilations") {
      attr_info.dilations.clear();
      for (int i = 0; i < attr.ints_size(); ++i) {
        attr_info.dilations.emplace_back(attr.ints(i));
      }
    }
  }

  // calculate output shape after averagepool
  std::vector<int64_t> output_shape;
  output_shape.emplace_back(input_shape[0]);  // batch size
  output_shape.emplace_back(input_shape[1]);  // number of channels
  for (size_t i = 0; i < attr_info.kernel_shape.size(); ++i) {
    int64_t pad = attr_info.pads[i];
    int64_t dilation = attr_info.dilations[i];
    int64_t kernel = attr_info.kernel_shape[i];
    int64_t stride = attr_info.strides[i];

    output_shape.emplace_back(
      (input_shape[i + 2] + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1
    );
  }

  // set value_info and update ndname_to_shape
  onnx::ValueInfoProto *val_info = this->m_graph.add_value_info();
  val_info->set_name(node.name());
  set_vec_to_shape(val_info, output_shape);

  this->m_name_to_shape[node.output(0)] = output_shape;
  this->m_name_to_dtsize[node.output(0)] = this->m_name_to_dtsize[node.input(0)];
}

void InferShapeImpl::infer_shapes_GlobalAveragePool(onnx::NodeProto &node) {
  // get node input shapes
  std::vector<std::vector<int64_t>> input_shapes;
  for (auto ndinput : node.input()) {
    if (this->m_name_to_shape.find(ndinput) == this->m_name_to_shape.end()) {
      std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
      return;
    }
    else {
      // get shape from ndname_to_shape
      input_shapes.emplace_back(this->m_name_to_shape[ndinput]);
    }
  }

  // calculate output shape after globalaveragepool
  std::vector<int64_t> output_shape = {
    input_shapes[0][0], // batch size
    input_shapes[0][1] // number of channels
  };
  for (size_t i = 2; i < input_shapes[0].size(); ++i) {
    output_shape.emplace_back(1);
  }

  // set value_info and update ndname_to_shape
  onnx::ValueInfoProto *val_info = this->m_graph.add_value_info();
  val_info->set_name(node.name());
  set_vec_to_shape(val_info, output_shape);

  this->m_name_to_shape[node.output(0)] = output_shape;
  this->m_name_to_dtsize[node.output(0)] = this->m_name_to_dtsize[node.input(0)];
}

void InferShapeImpl::infer_shapes_Flatten(onnx::NodeProto &node) {
  // attribute axis is not used now

  // get node input shapes
  std::vector<std::vector<int64_t>> input_shapes;
  for (auto ndinput : node.input()) {
    if (this->m_name_to_shape.find(ndinput) == this->m_name_to_shape.end()) {
      std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
      return;
    }
    else {
      // get shape from ndname_to_shape
      input_shapes.emplace_back(this->m_name_to_shape[ndinput]);
    }
  }

  // calculate output shape after flatten
  std::vector<int64_t> output_shape;
  int64_t flatten_dim = 1;
  for (size_t i = 2; i < input_shapes[0].size(); ++i) {
    flatten_dim *= input_shapes[0][i];
  }

  if (flatten_dim == 1) {
    output_shape = {
      input_shapes[0][0], // batch size
      input_shapes[0][1] // number of channels
    };
  }
  else {
    output_shape = {
      input_shapes[0][0], // batch size
      input_shapes[0][1], // number of channels
      flatten_dim // flatten_dim
    };
  }

  // set value_info and update ndname_to_shape
  onnx::ValueInfoProto *val_info = this->m_graph.add_value_info();
  val_info->set_name(node.name());
  set_vec_to_shape(val_info, output_shape);

  this->m_name_to_shape[node.output(0)] = output_shape;
  this->m_name_to_dtsize[node.output(0)] = this->m_name_to_dtsize[node.input(0)];
}

void InferShapeImpl::infer_shapes_Gemm(onnx::NodeProto &node) {
  // get attributes
  AttrInfo_Gemm attr_info;
  for (auto attr : node.attribute()) {
    if (attr.name() == "transA") {
      if (attr.i() == 1) attr_info.transA = true;
    }
    else if (attr.name() == "transB") {
      if (attr.i() == 1) attr_info.transB = true;
    }
  }

  // get node input shapes (A and B)
  std::vector<std::vector<int64_t>> input_shapes;
  for (size_t num = 0; num < 2; ++num) {
    auto ndinput = node.input(num);
    if (this->m_name_to_shape.find(ndinput) == this->m_name_to_shape.end()) {
      std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
      return;
    }
    else {
      // get shape from ndname_to_shape
      input_shapes.emplace_back(this->m_name_to_shape[ndinput]);
      if (num == 0 && attr_info.transA) {
        std::reverse(input_shapes[num].begin(), input_shapes[num].end());
      }
      else if (num == 1 && attr_info.transB) {
        std::reverse(input_shapes[num].begin(), input_shapes[num].end());
      }
    }
  }

  // calculate output shape after gemm
  // [M, K] * [K, N] = [M, N]
  std::vector<int64_t> output_shape;
  output_shape = {
    input_shapes[0][0],
    input_shapes[1][1]
  };

  // set value_info and update ndname_to_shape
  onnx::ValueInfoProto *val_info = this->m_graph.add_value_info();
  val_info->set_name(node.name());
  set_vec_to_shape(val_info, output_shape);

  this->m_name_to_shape[node.output(0)] = output_shape;
  this->m_name_to_dtsize[node.output(0)] = this->m_name_to_dtsize[node.input(0)];
}

void InferShapeImpl::infer_shapes(bool analyze) {
  this->set_io_iniz_shape_to_map(analyze);

  // infer shape for each node and store to value_info
  for (auto node : m_graph.node()) {
    if (node.op_type() == "Conv") {
      this->infer_shapes_Conv(node);
    }
    else if (node.op_type() == "Relu") {
      this->infer_shapes_Relu(node);
    }
    else if (node.op_type() == "MaxPool") {
      this->infer_shapes_MaxPool(node);
    }
    else if (node.op_type() == "AveragePool") {
      this->infer_shapes_AveragePool(node);
    }
    else if (node.op_type() == "Add") {
      this->infer_shapes_Add(node);
    }
    else if (node.op_type() == "GlobalAveragePool") {
      this->infer_shapes_GlobalAveragePool(node);
    }
    else if (node.op_type() == "Flatten") {
      this->infer_shapes_Flatten(node);
    }
    else if (node.op_type() == "Gemm") {
      this->infer_shapes_Gemm(node);
    }
    else {
      std::cerr << "Error: " << node.op_type() << " not supported now\n";
      exit(1);
    }

    if (analyze) {
      // analyze node and store to ndname_to_anal_data
      this->m_name_to_anal_data[node.name()] =
        this->m_analyzer.analyze_node(
          node, this->m_name_to_shape, this->m_name_to_dtsize);
    }
  }
}

void InferShapeImpl::infer_shapes() {
  this->infer_shapes(true);
}
