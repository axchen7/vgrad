#ifndef VGRAD_MODULE_H_
#define VGRAD_MODULE_H_

#include "tensor.h"

namespace vgrad {

template <typename T>
concept IsModule = requires(T t) {
    { t.params() };
};

auto unpack_params(IsModule auto& module) { return module.params(); }
auto unpack_params(IsTensor auto& tensor) { return std::make_tuple(std::ref(tensor)); }

template <typename... Params>
    requires((IsModule<Params> || IsTensor<Params>) && ...)
auto make_params(Params&... params) {
    return std::tuple_cat(unpack_params(params)...);
}

template <IsDimension In, IsDimension Out, Number DType>
class Linear {
   public:
    auto operator()(const auto& x) const {
        PROFILE_SCOPE("Linear::forward");
        return matmul(x, w) + b;
    }
    auto params() { return make_params(w, b); }

   private:
    using WShape = MakeShape<In, Out>;
    using BShape = MakeShape<Out>;
    Tensor<WShape, DType> w = randn<DType, WShape>();
    Tensor<BShape, DType> b = randn<DType, BShape>();
};

}  // namespace vgrad

#endif  // VGRAD_MODULE_H_