#ifndef VGRAD_OPS_H_
#define VGRAD_OPS_H_

#include <span>

#include "tensor.h"

namespace vgrad {

template <typename A, typename B>
concept TensorBinaryOpCompatible = TensorDTypeCompatible<A, B> && TensorShapeCompatible<A, B>;

using OneDimension = Dimension<1>;  // for unsqueeze

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

template <Index I1, Index I2, IsTensor A>
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

template <Index I, IsTensor A>
    requires(A::Shape::template At<I>::value == 1)
auto squeeze(const A& a) {
    using NewShape = typename A::Shape::template Remove<I>;
    return Tensor<NewShape, typename A::DType>(a.data_);
}

template <Index I, IsTensor A>
auto unsqueeze(const A& a) {
    using NewShape = typename A::Shape::template Insert<I, OneDimension>;
    return Tensor<NewShape, typename A::DType>(a.data_);
}

// Reduce the last dimension of a tensor.
template <IsTensor A>
auto reduce(const A& a, auto op) {
    using LastDim = typename A::Shape::template At<-1>;
    using NewShape = typename A::Shape::template Remove<-1>;
    Tensor<NewShape, typename A::DType> result;

    for (Size i = 0; i < NewShape::flat_size; i++) {
        std::span<const typename A::DType> slice{a.data_->begin() + i * LastDim::value, LastDim::value};
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

template <IsFloatTensor A, IsFloatTensor B>
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

template <IsFloatTensor A>
auto operator/(const A& a, typename A::DType b) {
    return unary_op(a, [b](auto x) { return x / b; });
}

template <IsFloatTensor B>
auto operator/(typename B::DType a, const B& b) {
    return unary_op(b, [a](auto x) { return a / x; });
}

template <IsTensor A>
auto sum(const A& a) {
    return reduce(a, [](std::span<const typename A::DType> x) {
        typename A::DType sum = 0;
        for (auto i : x) sum += i;
        return sum;
    });
}

template <IsTensor A>
auto prod(const A& a) {
    return reduce(a, [](std::span<const typename A::DType> x) {
        typename A::DType prod = 1;
        for (auto i : x) prod *= i;
        return prod;
    });
}

template <IsTensor A>
auto mean(const A& a) {
    using LastDim = typename A::Shape::template At<-1>;
    return sum(a) / LastDim::value;
}

}  // namespace vgrad

#endif  // VGRAD_OPS_H_