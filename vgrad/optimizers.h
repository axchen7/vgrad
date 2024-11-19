#ifndef VGRAD_OPTIMIZERS_H_
#define VGRAD_OPTIMIZERS_H_

#include "backward.h"

namespace vgrad::optim {

template <IsTensor... Params>
    requires(IsFloatTensor<Params> && ...)
class SGD {
   public:
    SGD(float lr, Params&... params) : lr_{lr}, params_{params...} {}

    template <IsScalarTensor Loss>
        requires IsFloatTensor<Loss>
    void step(const Loss& loss) {
        auto grads_tuple = std::apply([&loss](auto&... params) { return backward(loss, params...); }, params_);

        std::apply(
            [&](auto&... params) {
                std::apply([&](auto&... grads) { ((params = (params - lr_ * grads).detach()), ...); }, grads_tuple);
            },
            params_);
    }

   private:
    const float lr_;
    std::tuple<Params&...> params_;
};

}  // namespace vgrad::optim

#endif  // VGRAD_OPTIMIZERS_H_