#ifndef VGARD_GRAPH_H_
#define VGARD_GRAPH_H_

#include <functional>

#include "tensor.h"

namespace vgrad {

template <IsNode InNode, IsShape _OutShape, Number _DType>
    requires std::is_same_v<typename InNode::DType, _DType>
struct UnaryOpNode {
    static constexpr bool is_node = true;
    static constexpr bool is_unary_node = true;

    using DType = _DType;
    using InShape = InNode::OutShape;
    using InTensor = Tensor<InShape, DType>;
    using OutShape = _OutShape;
    using OutTensor = Tensor<OutShape, DType>;
    using GradFn = std::function<InTensor(const OutTensor&)>;

    const std::shared_ptr<InNode> in_node;
    const GradFn grad_fn;
};

template <IsNode InNode1, IsNode InNode2, IsShape _OutShape, Number _DType>
    requires std::is_same_v<typename InNode1::DType, _DType> && std::is_same_v<typename InNode2::DType, _DType>
struct BinaryOpNode {
    static constexpr bool is_node = true;
    static constexpr bool is_binary_node = true;

    using DType = _DType;
    using InShape1 = InNode1::OutShape;
    using InShape2 = InNode2::OutShape;
    using InTensor1 = Tensor<InShape1, DType>;
    using InTensor2 = Tensor<InShape2, DType>;
    using OutShape = _OutShape;
    using OutTensor = Tensor<OutShape, DType>;
    using GradFn = std::function<std::pair<InTensor1, InTensor2>(const OutTensor&)>;

    const std::shared_ptr<InNode1> in_node1;
    const std::shared_ptr<InNode2> in_node2;
    const GradFn grad_fn;
};

}  // namespace vgrad

#endif  // VGARD_GRAPH_H_