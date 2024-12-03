#ifndef VGRAD_BACKWARD_H_
#define VGRAD_BACKWARD_H_

#include <tuple>

#include "create_tensor.h"
#include "graph.h"

namespace vgrad {

template <IsTensor T>
struct GradientHolder {
    const T tensor;
    typename T::Detached gradient;

    GradientHolder(const T& tensor) : tensor{tensor}, gradient{zeros_like(tensor)} {}
};

template <IsNode Node, IsTensor Param>
auto accumulate_grad(const std::shared_ptr<Node> node,
                     const Tensor<typename Node::OutShape, typename Node::DType>& d_loss_d_out,
                     GradientHolder<Param>& grad_holder) {
    if constexpr (std::is_same_v<typename Param::Node, Node>) {
        if (grad_holder.tensor.get_node() == node)
            grad_holder.gradient = (grad_holder.gradient + d_loss_d_out).detach();
    }
}

template <IsNode Node, IsTensor... Params>
void backward_rec(const std::shared_ptr<Node> node,
                  const Tensor<typename Node::OutShape, typename Node::DType>& d_loss_d_out,
                  GradientHolder<Params>&... grad_holders) {
    (accumulate_grad(node, d_loss_d_out, grad_holders), ...);

    if constexpr (IsUnaryNode<Node>) {
        auto d_loss_d_in = node->grad_fn(d_loss_d_out);
        backward_rec(node->in_node, d_loss_d_in, grad_holders...);
    } else if constexpr (IsBinaryNode<Node>) {
        auto [d_loss_d_in1, d_loss_d_in2] = node->grad_fn(d_loss_d_out);
        backward_rec(node->in_node1, d_loss_d_in1, grad_holders...);
        backward_rec(node->in_node2, d_loss_d_in2, grad_holders...);
    }
}

template <IsScalarTensor RootTensor, IsTensor... Params>
    requires IsFloatTensor<RootTensor> && (IsFloatTensor<Params> && ...)
auto backward(const RootTensor& out, const Params&... params) {
    PROFILE_SCOPE("backward");
    auto grad_holders = std::make_tuple(GradientHolder{params}...);
    std::apply([&](auto&... grad_holders) { backward_rec(out.get_node(), ones_like(out), grad_holders...); },
               grad_holders);
    return std::apply([](auto&... grad_holders) { return std::make_tuple(grad_holders.gradient...); }, grad_holders);
}

}  // namespace vgrad

#endif  // VGRAD_BACKWARD_H_