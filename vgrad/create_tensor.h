#ifndef VGRAD_CREATE_TENSOR_H_
#define VGRAD_CREATE_TENSOR_H_

#include <random>

#include "tensor.h"

namespace vgrad {

template <typename DType, IsDimension Dim>
constexpr auto eye() {
    Dim dim;
    auto shape = make_shape(dim, dim);
    Tensor<decltype(shape), DType> result;
    for (Size i = 0; i < Dim::value; i++) {
        for (Size j = 0; j < Dim::value; j++) {
            result[i][j] = i == j ? 1 : 0;
        }
    }
    return result;
}

template <typename DType, IsShape Shape>
constexpr auto zeros() {
    Tensor<Shape, DType> result;
    for (Size i = 0; i < Shape::flat_size; i++) {
        (*result.data_)[i] = 0;
    }
    return result;
}

template <typename DType, IsShape Shape>
constexpr auto ones() {
    Tensor<Shape, DType> result;
    for (Size i = 0; i < Shape::flat_size; i++) {
        (*result.data_)[i] = 1;
    }
    return result;
}

template <typename DType, IsShape Shape>
auto randn() {
    std::default_random_engine eng;
    std::normal_distribution<DType> dist(0, 1);

    Tensor<Shape, DType> result;
    for (Size i = 0; i < Shape::flat_size; i++) {
        (*result.data_)[i] = dist(eng);
    }
    return result;
}

}  // namespace vgrad

#endif  // VGRAD_CREATE_TENSOR_H_