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
    using NestedData = NestedArray<Shape, DType>;
    using RawData = std::vector<DType>;

    Tensor() : data_(std::make_shared<RawData>(Shape::flat_size, DType{})) {}

    Tensor(const NestedData& data) : data_(std::make_shared<RawData>(Shape::flat_size)) {
        auto flat_data = reinterpret_cast<const DType*>(data.data());
        std::copy(flat_data, flat_data + Shape::flat_size, data_->begin());
    }

    Tensor(const std::shared_ptr<RawData>& data) : data_(data) {}

    template <IsShape NewShape>
        requires(NewShape::flat_size == Shape::flat_size)
    auto reshape() const {
        return Tensor<NewShape, DType>{data_};
    }

    const FlatData& flat_view() const { return *reinterpret_cast<const FlatData*>(data_->data()); }

    const NestedData& nested_view() const
        requires(Shape::rank > 0)
    {
        return *reinterpret_cast<const NestedData*>(data_->data());
    }

    const auto value() const
        requires(Shape::rank == 0)
    {
        return flat_view()[0];
    }

    const auto& operator[](Size index) const
        requires(Shape::rank > 0)
    {
        return nested_view()[index];
    }

    void _init_entry(Size index, DType value) { (*data_)[index] = value; }

   private:
    const std::shared_ptr<RawData> data_;
};

template <typename T>
concept IsTensor = std::is_same_v<T, Tensor<typename T::Shape, typename T::DType>>;

template <typename A>
concept IsFloatTensor = IsTensor<A> && std::is_floating_point_v<typename A::DType>;

template <typename A, typename B>
concept TensorDTypeCompatible = IsTensor<A> && IsTensor<B> && std::is_same_v<typename A::DType, typename B::DType>;

template <typename A, typename B>
concept TensorShapeCompatible = IsTensor<A> && IsTensor<B> && std::is_same_v<typename A::Shape, typename B::Shape>;

template <typename A, typename B>
concept TensorMatmulCompatible = IsTensor<A> && IsTensor<B> && A::Shape::rank >= 2 && B::Shape::rank >= 2 &&
                                 A::Shape::template At<-1>::value == B::Shape::template At<-2>::value;

}  // namespace vgrad

#endif  // VGRAD_TENSOR_H_