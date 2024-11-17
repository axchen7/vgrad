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

    GradientHolder(const T& tensor) : tensor{tensor}, gradient{zeros_like(T)} {}
};

template <IsTensor T>
void backward_rec(const std::shared_pointer<T::Node> node, const GradientHolder<T>& grad_holder,
                  const T::Detached& d_loss_d_out) {
    if (grad_holder.tensor.get_node() == node) {
        grad_holder.gradient = (grad_holder.gradient + d_loss_d_out).detach();
        return;
    }

    if constexpr (T::is_unary_node) {
        auto d_loss_d_in = node->grad_fn(d_loss_d_out);
        backward_rec(node->in_node, grad_holder, d_loss_d_in);
    }
}

template <IsScalarTensor RootTensor, IsTensor Param>
    requires IsFloatTensor<RootTensor> && IsFloatTensor<Param>
auto backward(const RootTensor& out, const Param& param) {
    GradientHolder<RootTensor> grad{};
    backward_rec(out);
}

template <IsScalarTensor RootTensor, IsTensor... Params>
    requires IsFloatTensor<RootTensor> && (IsFloatTensor<Params> && ...)
auto backward(const RootTensor& out, const Params&... params) {}

}  // namespace vgrad

#endif  // VGRAD_BACKWARD_H_