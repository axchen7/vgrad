#ifndef VGRAD_OPS_H_
#define VGRAD_OPS_H_

#include <span>

#include "tensor.h"

namespace vgrad {

template <typename A, typename B>
concept TensorBinaryOpCompatible = TensorDTypeCompatible<A, B> && TensorShapeCompatible<A, B>;

template <IsTensor A>
auto unary_op(const A& a, auto op) {
    Tensor<typename A::Shape, typename A::DType> result;
    for (Size i = 0; i < a.data_->size(); i++) {
        (*result.data_)[i] = op((*a.data_)[i]);
    }
    return result;
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto binary_op(const A& a, const B& b, auto op) {
    Tensor<typename A::Shape, typename A::DType> result;
    for (Size i = 0; i < a.data_->size(); i++) {
        (*result.data_)[i] = op((*a.data_)[i], (*b.data_)[i]);
    }
    return result;
}

template <IsTensor A, Index I1, Index I2>
auto transpose(const A& a) {
    using NewShape = typename A::Shape::template Transpose<I1, I2>;
    Tensor<NewShape, typename A::DType> result;

    constexpr auto original_strides = A::Shape::strides();
    constexpr auto new_strides = NewShape::strides();

    for (Size i = 0; i < a.data_->size(); i++) {
        std::array<Size, A::Shape::rank> indices;

        Size orig_idx = i;
        for (Size j = 0; j < indices.size(); j++) {
            indices[j] = orig_idx / original_strides[j];
            orig_idx %= original_strides[j];
        }
        std::swap(indices[I1], indices[I2]);

        Size new_idx = 0;
        for (Size j = 0; j < indices.size(); j++) {
            new_idx += indices[j] * new_strides[j];
        }

        (*result.data_)[new_idx] = (*a.data_)[i];
    }

    return result;
}

template <IsTensor A>
auto reduce_last_dim(const A& a, auto op) {
    using NewShape = typename A::Shape::template Remove<-1>;
    Tensor<NewShape, typename A::DType> result;

    constexpr auto last_dim = A::Shape::flat_size / NewShape::flat_size;

    for (Size i = 0; i < NewShape::flat_size; i++) {
        std::span<const typename A::DType> slice{a.data_->begin() + i * last_dim, last_dim};
        (*result.data_)[i] = op(slice);
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