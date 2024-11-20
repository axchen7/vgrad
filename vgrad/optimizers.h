#ifndef VGRAD_OPTIMIZERS_H_
#define VGRAD_OPTIMIZERS_H_

#include "backward.h"

namespace vgrad::optim {

template <IsTensor... Params>
    requires(IsFloatTensor<Params> && ...)
class SGD {
   public:
    SGD(const float lr, Params&... params) : SGD(lr, std::make_tuple(std::ref(params)...)) {}

    SGD(const float lr, std::tuple<Params&...> params) : lr_{lr}, params_{params} {}

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

template <IsTensor... Params>
    requires(IsFloatTensor<Params> && ...)
class Adam {
   public:
    Adam(const float lr, Params&... params) : Adam(lr, std::make_tuple(std::ref(params)...)) {}

    // default parameters from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    Adam(const float lr, std::tuple<Params&...> params) : Adam(lr, 0.9, 0.999, 1e-8, params) {}

    Adam(const float lr, const float beta1, const float beta2, const float eps, Params&... params)
        : Adam(lr, beta1, beta2, eps, std::make_tuple(std::ref(params)...)) {}

    Adam(const float lr, const float beta1, const float beta2, const float eps, std::tuple<Params&...> params)
        : lr_{lr},
          beta1_{beta1},
          beta2_{beta2},
          eps_{eps},
          params_{params},
          t_{1},
          m_{std::apply([](auto&... params) { return std::make_tuple((zeros_like(params))...); }, params_)},
          v_{std::apply([](auto&... params) { return std::make_tuple((zeros_like(params))...); }, params_)} {}

    template <IsScalarTensor Loss>
        requires IsFloatTensor<Loss>
    void step(const Loss& loss) {
        // implementation of https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

        // g <- dL/dw
        auto g_ = std::apply([&loss](auto&... params) { return backward(loss, params...); }, params_);

        // m <- beta1 * m + (1 - beta1) * g
        std::apply(
            [&](auto&... m) {
                std::apply([&](auto&... g) { ((m = (beta1_ * m + (1 - beta1_) * g).detach()), ...); }, g_);
            },
            m_);

        // v <- beta2 * v + (1 - beta2) * g^2
        std::apply(
            [&](auto&... v) {
                std::apply([&](auto&... g) { ((v = (beta2_ * v + (1 - beta2_) * g * g).detach()), ...); }, g_);
            },
            v_);

        // m_hat <- m / (1 - beta1^t)
        auto m_hat = std::apply(
            [&](auto&... m) {
                auto denom = 1 - std::pow(beta1_, t_);
                return std::make_tuple((m / denom).detach()...);
            },
            m_);

        // v_hat <- v / (1 - beta2^t)
        auto v_hat = std::apply(
            [&](auto&... v) {
                auto denom = 1 - std::pow(beta2_, t_);
                return std::make_tuple((v / denom).detach()...);
            },
            v_);

        // w <- w - lr * m_hat / (sqrt(v_hat) + eps)
        std::apply(
            [&](auto&... params) {
                std::apply(
                    [&](auto&... m_hat) {
                        std::apply(
                            [&](auto&... v_hat) {
                                ((params = (params - lr_ * m_hat / (sqrt(v_hat) + eps_)).detach()), ...);
                            },
                            v_hat);
                    },
                    m_hat);
            },
            params_);

        t_++;
    }

   private:
    const float lr_;
    const float beta1_;
    const float beta2_;
    const float eps_;
    std::tuple<Params&...> params_;

    // buffers

    // iteration counter
    int t_;
    // first moment
    std::tuple<typename Params::Detached...> m_;
    // second moment
    std::tuple<typename Params::Detached...> v_;
};

}  // namespace vgrad::optim

#endif  // VGRAD_OPTIMIZERS_H_