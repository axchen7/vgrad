#ifndef VGRAD_TENSOR_H_
#define VGRAD_TENSOR_H_

#include <array>
#include <memory>

#include "shape.h"

namespace vgrad {

template <typename T>
concept Number = std::is_arithmetic_v<T>;

template <IsShape Shape, Number DType>
struct NestedArrayHelper;

template <IsShape Shape, Number DType>
    requires(Shape::rank == 0)
struct NestedArrayHelper<Shape, DType> {
    using type = DType;
};

template <IsShape Shape, Number DType>
    requires(Shape::rank > 0)
struct NestedArrayHelper<Shape, DType> {
    using type = std::array<typename NestedArrayHelper<decltype(Shape::inner), DType>::type, Shape::outer.value>;
};

template <IsShape Shape, Number DType>
using NestedArray = typename NestedArrayHelper<Shape, DType>::type;

template <IsShape Shape_, Number DType_>
class Tensor {
   public:
    using Shape = Shape_;
    using DType = DType_;
    using FlatData = std::array<DType, Shape::flat_size>;

    std::shared_ptr<FlatData> data_;

    Tensor() : data_(std::make_shared<FlatData>()) {}
    Tensor(NestedArray<Shape, DType> data) : data_(std::make_shared<FlatData>(flatten(data))) {}

    NestedArray<Shape, DType>& data() const { return *reinterpret_cast<NestedArray<Shape, DType>*>(data_->data()); }

   private:
    static FlatData flatten(NestedArray<Shape, DType>& data) { return *reinterpret_cast<FlatData*>(data.data()); }
};

template <typename T>
concept IsTensor = std::is_same_v<T, Tensor<typename T::Shape, typename T::DType>>;

template <typename A, typename B>
concept TensorShapeCompatible = IsTensor<A> && IsTensor<B> && std::is_same_v<typename A::Shape, typename B::Shape>;

template <typename A, typename B>
concept TensorMultiplicationCompatible = IsTensor<A> && IsTensor<B> && A::Shape::rank >= 2 && B::Shape::rank >= 2 &&
                                         A::Shape::template dim<-1>() == B::Shape::template dim<-2>();

}  // namespace vgrad

#endif  // VGRAD_TENSOR_H_