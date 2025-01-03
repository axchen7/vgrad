#ifndef VGRAD_CREATE_TENSOR_H_
#define VGRAD_CREATE_TENSOR_H_

#include <random>

#include "tensor.h"

namespace vgrad {

std::default_random_engine eng(std::random_device{}());

template <typename DType, IsDimension Dim>
constexpr auto eye() {
    PROFILE_SCOPE("eye");
    Dim dim;
    auto shape = make_shape(dim, dim);
    Tensor<decltype(shape), DType> result;
    for (Size i = 0; i < Dim::value; i++) {
        result._flat_data()[i * Dim::value + i] = 1;
    }
    return result;
}

template <typename DType, IsShape Shape>
constexpr auto full(DType value) {
    PROFILE_SCOPE("full");
    Tensor<Shape, DType> result;
    result._flat_data().fill(value);
    return result;
}

template <typename DType, IsShape Shape>
constexpr auto zeros() {
    PROFILE_SCOPE("zeros");
    return full<DType, Shape>(0);
}

template <typename DType, IsShape Shape>
constexpr auto ones() {
    PROFILE_SCOPE("ones");
    return full<DType, Shape>(1);
}

template <typename DType, IsDimension Dim>
constexpr auto arange() {
    PROFILE_SCOPE("arange");
    Dim dim;
    auto shape = make_shape(dim);
    Tensor<decltype(shape), DType> result;
    for (Size i = 0; i < Dim::value; i++) {
        result._flat_data()[i] = i;
    }
    return result;
}

template <typename DType, IsShape Shape>
auto randn() {
    PROFILE_SCOPE("randn");
    std::normal_distribution<DType> dist(0, 1);

    Tensor<Shape, DType> result;
    for (Size i = 0; i < Shape::flat_size; i++) {
        result._flat_data()[i] = dist(eng);
    }
    return result;
}

template <IsTensor T>
constexpr auto full_like(const T& tensor, typename T::DType value) {
    PROFILE_SCOPE("full_like");
    return full<typename T::DType, typename T::Shape>(value);
}

template <IsTensor T>
constexpr auto zeros_like(const T& tensor) {
    PROFILE_SCOPE("zeros_like");
    return zeros<typename T::DType, typename T::Shape>();
}

template <IsTensor T>
constexpr auto ones_like(const T& tensor) {
    PROFILE_SCOPE("ones_like");
    return ones<typename T::DType, typename T::Shape>();
}

template <IsTensor T>
auto randn_like(const T& tensor) {
    PROFILE_SCOPE("randn_like");
    return randn<typename T::DType, typename T::Shape>();
}

}  // namespace vgrad

#endif  // VGRAD_CREATE_TENSOR_H_