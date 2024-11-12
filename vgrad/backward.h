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

template <IsScalarTensor RootTensor, IsTensor... Params>
    requires IsFloatTensor<RootTensor> && (IsFloatTensor<Params> && ...)
auto backward(const RootTensor& out, const Params&... params) {
    // TODO
}

}  // namespace vgrad

#endif  // VGRAD_BACKWARD_H_