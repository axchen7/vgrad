#ifndef VGRAD_OPTIMIZERS_H_
#define VGRAD_OPTIMIZERS_H_

#include "backward.h"

namespace vgrad {

template <IsTensor... Params>
    requires(IsFloatTensor<Params> && ...)
class SGD {
   public:
    SGD(float learning_rate, Params&... params) : learning_rate_{learning_rate}, params_{params...} {}

    template <IsScalarTensor Loss>
        requires IsFloatTensor<Loss>
    void step(const Loss& loss) {
        auto grads_tuple = std::apply([&loss](auto&... params) { return backward(loss, params...); }, params_);

        std::apply(
            [&](auto&... params) {
                std::apply([&](auto&... grads) { ((params = (params - learning_rate_ * grads).detach()), ...); },
                           grads_tuple);
            },
            params_);
    }

   private:
    const float learning_rate_;
    std::tuple<Params&...> params_;
};

}  // namespace vgrad

#endif  // VGRAD_OPTIMIZERS_H_