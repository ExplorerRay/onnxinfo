#include <math.h>
#include <iomanip>
#include "InferShape.hpp"
#include "utils.hpp"
#include "AttrInfo.hpp"

void InferShapeImpl::set_io_iniz_shape_to_map() {
    for (auto input : this->graph.input()) {
        auto shape = input.type().tensor_type().shape();
        std::vector<int64_t> shape_vec = {};
        for (int i = 0; i < shape.dim_size(); ++i) {
            auto dim = shape.dim(i);
            if (dim.has_dim_value()) {
                shape_vec.emplace_back(dim.dim_value());
            }
        }
        this->ndname_to_shape[input.name()] = shape_vec;
    }

    for (auto initializer : this->graph.initializer()) {
        auto shape = initializer.dims();
        std::vector<int64_t> shape_vec = {};
        for (int i = 0; i < shape.size(); ++i) {
            shape_vec.emplace_back(shape.Get(i));
        }
        this->ndname_to_shape[initializer.name()] = shape_vec;
    }

    for (auto output : this->graph.output()) {
        auto shape = output.type().tensor_type().shape();
        std::vector<int64_t> shape_vec = {};
        for (int i = 0; i < shape.dim_size(); ++i) {
            auto dim = shape.dim(i);
            if (dim.has_dim_value()) {
                shape_vec.emplace_back(dim.dim_value());
            }
        }
        this->ndname_to_shape[output.name()] = shape_vec;
    }
}

void InferShapeImpl::print_summary() {
    std::cout << std::left << std::setw(INDENT) << "Name"
              << std::left << std::setw(INDENT) << "Type"
              << std::left << std::setw(INDENT) << "Input Shape"
              << std::left << std::setw(INDENT) << "Output Shape" << '\n';
    std::cout << std::string(INDENT * 4, '-') << '\n';

    for (auto node : graph.node()) {
        std::cout << std::left << std::setw(INDENT) << string_trimmer(node.name(), INDENT-5);

        std::cout << std::left << std::setw(INDENT) << node.op_type();
        std::cout << std::setw(INDENT) << dims_vec_to_str(this->get_ndname_to_shape()[node.input(0)]);
        std::cout << std::setw(INDENT) << dims_vec_to_str(this->get_ndname_to_shape()[node.output(0)]);
        std::cout << '\n';
    }
}

void InferShapeImpl::infer_shapes_Conv(onnx::NodeProto &node) {
    // get attributes (kernel_shape, strides, pads, dilations, group)
    struct AttrInfo_Conv attr_info;
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

    // get node input shapes
    std::vector<std::vector<int64_t>> input_shapes;
    for (auto ndinput : node.input()) {
        if (this->get_ndname_to_shape().find(ndinput) == this->get_ndname_to_shape().end()) {
            std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
            return;
            exit(1);
        }
        else {
            // get shape from ndname_to_shape
            input_shapes.emplace_back(this->get_ndname_to_shape()[ndinput]);
        }
    }

    // calculate output shape after convolution
    auto input_shape = input_shapes[0];
    auto weight_shape = input_shapes[1];
    std::vector<int64_t> output_shape = {
        input_shape[0], // batch size
        weight_shape[0], // number of channels
        (input_shape[2] + 2 * attr_info.pads[0] - attr_info.dilations[0] * (weight_shape[2] - 1) - 1) / attr_info.strides[0] + 1, // height
        (input_shape[3] + 2 * attr_info.pads[1] - attr_info.dilations[1] * (weight_shape[3] - 1) - 1) / attr_info.strides[1] + 1 // width
    };

    // set value_info and update ndname_to_shape
    onnx::ValueInfoProto *val_info = this->graph.add_value_info();
    val_info->set_name(node.name());
    set_vec_to_shape(val_info, output_shape);

    this->ndname_to_shape[node.output(0)] = output_shape;
}

void InferShapeImpl::infer_shapes_Relu(onnx::NodeProto &node) {
    // get node input shapes
    std::vector<std::vector<int64_t>> input_shapes;
    for (auto ndinput : node.input()) {
        if (this->get_ndname_to_shape().find(ndinput) == this->get_ndname_to_shape().end()) {
            std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
            return;
        }
        else {
            // get shape from ndname_to_shape
            input_shapes.emplace_back(this->get_ndname_to_shape()[ndinput]);
        }
    }

    // set value_info and update ndname_to_shape
    onnx::ValueInfoProto *val_info = this->graph.add_value_info();
    val_info->set_name(node.name());
    set_vec_to_shape(val_info, input_shapes[0]);

    this->ndname_to_shape[node.output(0)] = input_shapes[0];
}

void InferShapeImpl::infer_shapes_MaxPool(onnx::NodeProto &node) {
    struct AttrInfo_MaxPool attr_info;
    for (auto attr : node.attribute()) {
        // std::cout << "attr name: " << attr.name() << '\n';
        if (attr.name() == "kernel_shape") { // required
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
            // std::cout << "ceil_mode: " << attr_info.ceil_mode << '\n';
        }
        else if (attr.name() == "dilations") {
            attr_info.dilations.clear();
            for (int i = 0; i < attr.ints_size(); ++i) {
                attr_info.dilations.emplace_back(attr.ints(i));
            }
        }
    }

    // calculate output shape after maxpool
    auto input_shape = this->get_ndname_to_shape()[node.input(0)];
    std::vector<int64_t> output_shape = {
        input_shape[0], // batch size
        input_shape[1], // number of channels
        (input_shape[2] + 2 * attr_info.pads[0] - attr_info.dilations[0] * (attr_info.kernel_shape[0] - 1) - 1) / attr_info.strides[0] + 1, // height
        (input_shape[3] + 2 * attr_info.pads[1] - attr_info.dilations[1] * (attr_info.kernel_shape[1] - 1) - 1) / attr_info.strides[1] + 1 // width
    };

    // set value_info and update ndname_to_shape
    onnx::ValueInfoProto *val_info = this->graph.add_value_info();
    val_info->set_name(node.name());
    set_vec_to_shape(val_info, output_shape);

    this->ndname_to_shape[node.output(0)] = output_shape;
}

void InferShapeImpl::infer_shapes_Add(onnx::NodeProto &node) {
    // get node input shapes
    std::vector<std::vector<int64_t>> input_shapes;
    for (auto ndinput : node.input()) {
        if (this->get_ndname_to_shape().find(ndinput) == this->get_ndname_to_shape().end()) {
            std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
            return;
        }
        else {
            // get shape from ndname_to_shape
            input_shapes.emplace_back(this->get_ndname_to_shape()[ndinput]);
        }
    }

    // set value_info and update ndname_to_shape
    onnx::ValueInfoProto *val_info = this->graph.add_value_info();
    val_info->set_name(node.name());
    set_vec_to_shape(val_info, input_shapes[0]);

    this->ndname_to_shape[node.output(0)] = input_shapes[0];
}

void InferShapeImpl::infer_shapes_GlobalAveragePool(onnx::NodeProto &node) {
    // get node input shapes
    std::vector<std::vector<int64_t>> input_shapes;
    for (auto ndinput : node.input()) {
        if (this->get_ndname_to_shape().find(ndinput) == this->get_ndname_to_shape().end()) {
            std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
            return;
        }
        else {
            // get shape from ndname_to_shape
            input_shapes.emplace_back(this->get_ndname_to_shape()[ndinput]);
        }
    }

    // calculate output shape after globalaveragepool
    std::vector<int64_t> output_shape = {
        input_shapes[0][0], // batch size
        input_shapes[0][1], // number of channels
        1, // height
        1 // width
    };

    // set value_info and update ndname_to_shape
    onnx::ValueInfoProto *val_info = this->graph.add_value_info();
    val_info->set_name(node.name());
    set_vec_to_shape(val_info, output_shape);

    this->ndname_to_shape[node.output(0)] = output_shape;
}

void InferShapeImpl::infer_shapes_Flatten(onnx::NodeProto &node) {
    // for (auto attr : node.attribute()) {
    //     if (attr.name() == "axis") {
    //         std::cout << "axis: " << attr.i() << '\n';
    //     }
    // }

    // get node input shapes
    std::vector<std::vector<int64_t>> input_shapes;
    for (auto ndinput : node.input()) {
        if (this->get_ndname_to_shape().find(ndinput) == this->get_ndname_to_shape().end()) {
            std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
            return;
        }
        else {
            // get shape from ndname_to_shape
            input_shapes.emplace_back(this->get_ndname_to_shape()[ndinput]);
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
    onnx::ValueInfoProto *val_info = this->graph.add_value_info();
    val_info->set_name(node.name());
    set_vec_to_shape(val_info, output_shape);

    this->ndname_to_shape[node.output(0)] = output_shape;
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

    // get node input shapes
    std::vector<std::vector<int64_t>> input_shapes;
    for (size_t num = 0; num < 2; ++num) {
        auto ndinput = node.input(num);
        if (this->get_ndname_to_shape().find(ndinput) == this->get_ndname_to_shape().end()) {
            std::cerr << "Error: " << ndinput << " not found in ndname_to_shape\n";
            return;
        }
        else {
            // get shape from ndname_to_shape
            input_shapes.emplace_back(this->get_ndname_to_shape()[ndinput]);
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
    onnx::ValueInfoProto *val_info = this->graph.add_value_info();
    val_info->set_name(node.name());
    set_vec_to_shape(val_info, output_shape);

    this->ndname_to_shape[node.output(0)] = output_shape;
}

void InferShapeImpl::infer_shapes() {
    this->set_io_iniz_shape_to_map();

    // infer shape for each node and store to value_info
    for (auto node : graph.node()) {
        // std::cout << "Node: " << node.name() << '\n';
        if (node.op_type() == "Conv") {
            this->infer_shapes_Conv(node);
        }
        else if (node.op_type() == "Relu") {
            this->infer_shapes_Relu(node);
        }
        else if (node.op_type() == "MaxPool") {
            this->infer_shapes_MaxPool(node);
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
            return;
        }
    }
}
