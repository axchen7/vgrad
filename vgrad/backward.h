#ifndef VGRAD_BACKWARD_H_
#define VGRAD_BACKWARD_H_

#include <tuple>

#include "create_tensor.h"
#include "graph.h"

namespace vgrad {

template <IsTensor T>
struct GradientHolder {
    const T tensor;
    std::optional<typename T::Detached> gradient;
};

template <IsNode Node, IsTensor Param>
void backward_single_rec(const std::shared_ptr<Node>& node,
                         const Tensor<typename Node::OutShape, typename Node::DType>& dl_df,
                         GradientHolder<Param>& grad_holder) {
    if constexpr (std::is_same_v<Node, typename Param::Node>) {
        if (node == grad_holder.tensor.get_node()) {
            grad_holder.gradient = dl_df;
            return;
        }
    }

    if constexpr (IsUnaryNode<Node>) {
        auto dl_da = node->grad_fn(dl_df);
        backward_single_rec(node->in_node, dl_da, grad_holder);
    } else if constexpr (IsBinaryNode<Node>) {
        auto [dl_da1, dl_da2] = node->grad_fn(dl_df);
        backward_single_rec(node->in_node1, dl_da1, grad_holder);
        backward_single_rec(node->in_node2, dl_da2, grad_holder);
    }
}

template <IsScalarTensor RootTensor, IsTensor Param>
auto backward_single(const RootTensor& out, const Param& param) {
    const auto node = out.get_node();
    auto grad_holder = GradientHolder{param, std::nullopt};
    backward_single_rec(node, ones_like(out), grad_holder);
    if (grad_holder.gradient.has_value()) {
        return grad_holder.gradient.value();
    } else {
        return zeros_like(param);
    }
}

template <IsScalarTensor RootTensor, IsTensor... Params>
    requires IsFloatTensor<RootTensor> && (IsFloatTensor<Params> && ...)
auto backward(const RootTensor& out, const Params&... params) {
    return std::tuple{backward_single(out, params)...};
}

}  // namespace vgrad

#endif  // VGRAD_BACKWARD_H_