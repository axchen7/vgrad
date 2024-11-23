#ifndef VGRAD_CREATE_TENSOR_H_
#define VGRAD_CREATE_TENSOR_H_

#include <random>

#include "tensor.h"

namespace vgrad {

std::default_random_engine eng(std::random_device{}());

template <typename DType, IsDimension Dim>
constexpr auto eye() {
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
    Tensor<Shape, DType> result;
    result._flat_data().fill(value);
    return result;
}

template <typename DType, IsShape Shape>
constexpr auto zeros() {
    return full<DType, Shape>(0);
}

template <typename DType, IsShape Shape>
constexpr auto ones() {
    return full<DType, Shape>(1);
}

template <IsTensor T>
constexpr auto full_like(const T& tensor, typename T::DType value) {
    return full<typename T::DType, typename T::Shape>(value);
}

template <IsTensor T>
constexpr auto zeros_like(const T& tensor) {
    return zeros<typename T::DType, typename T::Shape>();
}

template <IsTensor T>
constexpr auto ones_like(const T& tensor) {
    return ones<typename T::DType, typename T::Shape>();
}

template <typename DType, IsShape Shape>
auto randn() {
    std::normal_distribution<DType> dist(0, 1);

    Tensor<Shape, DType> result;
    for (Size i = 0; i < Shape::flat_size; i++) {
        result._flat_data()[i] = dist(eng);
    }
    return result;
}

}  // namespace vgrad

#endif  // VGRAD_CREATE_TENSOR_H_