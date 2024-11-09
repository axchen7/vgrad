#ifndef VGRAD_OPS_H_
#define VGRAD_OPS_H_

#include "tensor.h"

namespace vgrad {

template <typename A, typename B>
concept TensorBinaryOpCompatible = TensorDTypeCompatible<A, B> && TensorShapeCompatible<A, B>;

template <IsTensor A>
auto unary_op(const A& a, auto op) {
    Tensor<typename A::Shape, typename A::DType> result;
    for (Size i = 0; i < a.data_->size(); ++i) {
        (*result.data_)[i] = op((*a.data_)[i]);
    }
    return result;
}

template <IsTensor A>
auto operator-(const A& a) {
    return unary_op(a, [](auto x) { return -x; });
}

template <IsFloatTensor A>
auto exp(const A& a) {
    return unary_op(a, [](auto x) { return std::exp(x); });
}

template <IsFloatTensor A>
auto log(const A& a) {
    return unary_op(a, [](auto x) { return std::log(x); });
}

template <IsFloatTensor A>
auto relu(const A& a) {
    return unary_op(a, [](auto x) { return x > 0 ? x : 0; });
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto binary_op(const A& a, const B& b, auto op) {
    Tensor<typename A::Shape, typename A::DType> result;
    for (Size i = 0; i < a.data_->size(); ++i) {
        (*result.data_)[i] = op((*a.data_)[i], (*b.data_)[i]);
    }
    return result;
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator+(const A& a, const B& b) {
    return binary_op(a, b, [](auto x, auto y) { return x + y; });
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator-(const A& a, const B& b) {
    return binary_op(a, b, [](auto x, auto y) { return x - y; });
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator*(const A& a, const B& b) {
    return binary_op(a, b, [](auto x, auto y) { return x * y; });
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator/(const A& a, const B& b) {
    return binary_op(a, b, [](auto x, auto y) { return x / y; });
}

template <IsTensor A>
auto operator+(const A& a, typename A::DType b) {
    return unary_op(a, [b](auto x) { return x + b; });
}

template <IsTensor B>
auto operator+(typename B::DType a, const B& b) {
    return unary_op(b, [a](auto x) { return a + x; });
}

template <IsTensor A>
auto operator-(const A& a, typename A::DType b) {
    return unary_op(a, [b](auto x) { return x - b; });
}

template <IsTensor B>
auto operator-(typename B::DType a, const B& b) {
    return unary_op(b, [a](auto x) { return a - x; });
}

template <IsTensor A>
auto operator*(const A& a, typename A::DType b) {
    return unary_op(a, [b](auto x) { return x * b; });
}

template <IsTensor B>
auto operator*(typename B::DType a, const B& b) {
    return unary_op(b, [a](auto x) { return a * x; });
}

template <IsTensor A>
auto operator/(const A& a, typename A::DType b) {
    return unary_op(a, [b](auto x) { return x / b; });
}

template <IsTensor B>
auto operator/(typename B::DType a, const B& b) {
    return unary_op(b, [a](auto x) { return a / x; });
}

}  // namespace vgrad

#endif  // VGRAD_OPS_H_