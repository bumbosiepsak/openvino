// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_28 {

ov::OutputVector swiglu(const ov::frontend::onnx::Node& node) {
    // Operator definition: https://onnx.ai/onnx/operators/onnx__SwiGLU.html
    //   Y = Swish(A, alpha) * B = A * Sigmoid(alpha * A) * B
    common::default_op_checks(node, 2);
    const auto inputs = node.get_ov_inputs();
    const auto& gate = inputs[0];
    const auto& value = inputs[1];

    // The specification requires identical shapes and disallows broadcasting.
    CHECK_VALID_NODE(node,
                     gate.get_partial_shape().compatible(value.get_partial_shape()),
                     "SwiGLU expects inputs A and B to have identical shapes, got: ",
                     gate.get_partial_shape(),
                     " and ",
                     value.get_partial_shape());

    std::shared_ptr<ov::Node> swish;
    if (node.get_attribute_value<float>("alpha", 1.0f) == 1.0f) {
        // Single-input Swish is the form expected by GLU and MoE fusions.
        swish = std::make_shared<v4::Swish>(gate);
    } else {
        ov::Output<ov::Node> alpha = node.get_attribute_as_constant<float>("alpha", 1.0f, gate.get_element_type());
        if (gate.get_element_type().is_dynamic()) {
            alpha = std::make_shared<v1::ConvertLike>(alpha, gate);
        }
        swish = std::make_shared<v4::Swish>(gate, alpha);
    }
    return {std::make_shared<v1::Multiply>(swish, value)};
}

// SwiGLU is a part of the standard ONNX domain since opset 28, but it is registered since opset 1,
// because OperatorsBridge requires a translator for every version imported by a model.
ONNX_OP("SwiGLU", OPSET_SINCE(1), ai_onnx::opset_28::swiglu);

}  // namespace ov::frontend::onnx::ai_onnx::opset_28
