#include "ModelAnalysis.hpp"
#include "utils.hpp"

int64_t get_prod(std::vector<int64_t> &vec) {
    int64_t prod = 1;
    for (int64_t v : vec) {
        prod *= v;
    }
    return prod;
}

AnalyzeData analyze_node_Conv(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size) {
    AnalyzeData data;

    // MACs
    std::vector<int64_t> kernel_shape = input_shapes[1];
    std::vector<int64_t> reduce_shape(kernel_shape.begin() + 1, kernel_shape.end());
    int64_t out_prod = get_prod(output_shape);
    data.mac = out_prod * get_prod(reduce_shape) * MUL_MACS;

    if (node.input_size() == 3) { // with bias
        data.mac += out_prod * ADD_MACS;
    }

    // Parameters & Memory
    for (size_t i = 1; i < input_shapes.size(); ++i) {
        data.param += get_prod(input_shapes[i]);
        data.mem += get_prod(input_shapes[i]) * ndname_to_size[node.input(i)];
    }
    data.mem += get_prod(input_shapes[0]) * ndname_to_size[node.input(0)];
    data.mem += out_prod * ndname_to_size[node.output(0)];

    return data;
}

AnalyzeData analyze_node_Relu(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size) {
    AnalyzeData data;

    data.param = 0; // no trainable parameters
    data.mac = 0; // non MACs operation

    // Memory
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        data.mem += get_prod(input_shapes[i]) * ndname_to_size[node.input(i)];
    }
    data.mem += get_prod(output_shape) * ndname_to_size[node.output(0)];

    return data;
}

AnalyzeData analyze_node_MaxPool(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size) {
    AnalyzeData data;

    // MACs
    data.mac = 0; // non MACs operation

    // Parameters & Memory
    data.param = 0; // no trainable parameters
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        data.mem += get_prod(input_shapes[i]) * ndname_to_size[node.input(i)];
    }
    data.mem += get_prod(output_shape) * ndname_to_size[node.output(0)];

    return data;
}

AnalyzeData analyze_node_Add(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size) {
    AnalyzeData data;

    // MACs
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        data.mac += get_prod(input_shapes[i]) * ADD_MACS;
    }

    // Parameters & Memory
    data.param = 0; // no trainable parameters
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        data.mem += get_prod(input_shapes[i]) * ndname_to_size[node.input(i)];
    }
    data.mem += get_prod(output_shape) * ndname_to_size[node.output(0)];

    return data;
}

AnalyzeData analyze_node_GlobalAveragePool(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size) {
    AnalyzeData data;

    // MACs
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        data.mac += get_prod(input_shapes[i]) * ADD_MACS;
    }
    data.mac += get_prod(output_shape) * DIV_MACS;

    // Parameters & Memory
    data.param = 0; // no trainable parameters
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        data.mem += get_prod(input_shapes[i]) * ndname_to_size[node.input(i)];
    }
    data.mem += get_prod(output_shape) * ndname_to_size[node.output(0)];

    return data;
}

AnalyzeData analyze_node_Flatten(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size) {
    AnalyzeData data;

    // MACs
    data.mac = 0; // non MACs operation

    // Parameters & Memory
    data.param = 0; // no trainable parameters
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        data.mem += get_prod(input_shapes[i]) * ndname_to_size[node.input(i)];
    }
    data.mem += get_prod(output_shape) * ndname_to_size[node.output(0)];

    return data;
}

AnalyzeData analyze_node_Gemm(onnx::NodeProto &node, std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> &output_shape, std::unordered_map<std::string, size_t> &ndname_to_size) {
    AnalyzeData data;

    // MACs
    data.mac = get_prod(input_shapes[0]) * get_prod(output_shape) * MUL_MACS;

    // Parameters & Memory
    data.param = get_prod(input_shapes[0]) * get_prod(output_shape) + get_prod(output_shape);
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        data.mem += get_prod(input_shapes[i]) * ndname_to_size[node.input(i)];
    }
    data.mem += get_prod(output_shape) * ndname_to_size[node.output(0)];

    return data;
}

AnalyzeData analyze_node(onnx::NodeProto &node, const std::unordered_map<std::string, std::vector<int64_t>> &ndname_to_shape, const std::unordered_map<std::string, size_t> &ndname_to_dtype_size) {
    AnalyzeData data;
    if (node.op_type() == "Conv") {
        std::unordered_map<std::string, size_t> ndname_to_size;
        std::vector<std::vector<int64_t>> input_shapes;
        for (auto input : node.input()) {
            ndname_to_size[input] = ndname_to_dtype_size.at(input);
            input_shapes.emplace_back(ndname_to_shape.at(input));
        }
        std::vector<int64_t> output_shape = ndname_to_shape.at(node.output(0));
        ndname_to_size[node.output(0)] = ndname_to_dtype_size.at(node.output(0));

        data = analyze_node_Conv(node, input_shapes, output_shape, ndname_to_size);
    }
    else if (node.op_type() == "Relu") {
        std::unordered_map<std::string, size_t> ndname_to_size;
        std::vector<std::vector<int64_t>> input_shapes;
        for (auto input : node.input()) {
            ndname_to_size[input] = ndname_to_dtype_size.at(input);
            input_shapes.emplace_back(ndname_to_shape.at(input));
        }
        std::vector<int64_t> output_shape = ndname_to_shape.at(node.output(0));
        ndname_to_size[node.output(0)] = ndname_to_dtype_size.at(node.output(0));

        data = analyze_node_Relu(node, input_shapes, output_shape, ndname_to_size);
    }
    else if (node.op_type() == "MaxPool") {
        std::unordered_map<std::string, size_t> ndname_to_size;
        std::vector<std::vector<int64_t>> input_shapes;
        for (auto input : node.input()) {
            ndname_to_size[input] = ndname_to_dtype_size.at(input);
            input_shapes.emplace_back(ndname_to_shape.at(input));
        }
        std::vector<int64_t> output_shape = ndname_to_shape.at(node.output(0));
        ndname_to_size[node.output(0)] = ndname_to_dtype_size.at(node.output(0));

        data = analyze_node_MaxPool(node, input_shapes, output_shape, ndname_to_size);
    }
    else if (node.op_type() == "Add") {
        std::unordered_map<std::string, size_t> ndname_to_size;
        std::vector<std::vector<int64_t>> input_shapes;
        for (auto input : node.input()) {
            ndname_to_size[input] = ndname_to_dtype_size.at(input);
            input_shapes.emplace_back(ndname_to_shape.at(input));
        }
        std::vector<int64_t> output_shape = ndname_to_shape.at(node.output(0));
        ndname_to_size[node.output(0)] = ndname_to_dtype_size.at(node.output(0));

        data = analyze_node_Add(node, input_shapes, output_shape, ndname_to_size);
    }
    else if (node.op_type() == "GlobalAveragePool") {
        std::unordered_map<std::string, size_t> ndname_to_size;
        std::vector<std::vector<int64_t>> input_shapes;
        for (auto input : node.input()) {
            ndname_to_size[input] = ndname_to_dtype_size.at(input);
            input_shapes.emplace_back(ndname_to_shape.at(input));
        }
        std::vector<int64_t> output_shape = ndname_to_shape.at(node.output(0));
        ndname_to_size[node.output(0)] = ndname_to_dtype_size.at(node.output(0));

        data = analyze_node_GlobalAveragePool(node, input_shapes, output_shape, ndname_to_size);
    }
    else if (node.op_type() == "Flatten") {
        std::unordered_map<std::string, size_t> ndname_to_size;
        std::vector<std::vector<int64_t>> input_shapes;
        for (auto input : node.input()) {
            ndname_to_size[input] = ndname_to_dtype_size.at(input);
            input_shapes.emplace_back(ndname_to_shape.at(input));
        }
        std::vector<int64_t> output_shape = ndname_to_shape.at(node.output(0));
        ndname_to_size[node.output(0)] = ndname_to_dtype_size.at(node.output(0));

        data = analyze_node_Flatten(node, input_shapes, output_shape, ndname_to_size);
    }
    else if (node.op_type() == "Gemm") {
        std::unordered_map<std::string, size_t> ndname_to_size;
        std::vector<std::vector<int64_t>> input_shapes;
        for (auto input : node.input()) {
            ndname_to_size[input] = ndname_to_dtype_size.at(input);
            input_shapes.emplace_back(ndname_to_shape.at(input));
        }
        std::vector<int64_t> output_shape = ndname_to_shape.at(node.output(0));
        ndname_to_size[node.output(0)] = ndname_to_dtype_size.at(node.output(0));

        data = analyze_node_Gemm(node, input_shapes, output_shape, ndname_to_size);
    }
    else {
        std::cerr << "Error: " << node.op_type() << " not supported now\n";
        exit(1);
    }

    return data;
}
