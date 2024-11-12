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

}  // namespace vgrad

#endif  // VGARD_GRAPH_H_