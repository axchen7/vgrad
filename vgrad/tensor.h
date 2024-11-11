#ifndef VGRAD_TENSOR_H_
#define VGRAD_TENSOR_H_

#include <array>
#include <cassert>
#include <memory>

#include "shape.h"

namespace vgrad {

template <IsShape _OutShape, Number _DType>
struct LeafNode {
    static constexpr bool is_node = true;
    using DType = _DType;
    using OutShape = _OutShape;
};

template <IsShape _Shape, Number _DType, IsNode _Node = LeafNode<_Shape, _DType>>
    requires std::is_same_v<typename _Node::OutShape, _Shape> && std::is_same_v<typename _Node::DType, _DType>
class Tensor {
   public:
    using Shape = _Shape;
    using DType = _DType;
    using Node = _Node;
    using FlatData = std::array<DType, Shape::flat_size>;
    using NestedData = NestedArray<Shape, DType>;
    using RawData = std::vector<DType>;  // TODO use static-length std::array (no longer need raw data)
    using Detached = Tensor<Shape, DType>;

    Tensor(Node&& node = Node{})
        : data_{std::make_shared<RawData>(Shape::flat_size, DType{})}, node_{std::make_shared<Node>(node)} {}

    Tensor(const NestedData& data, Node&& node = Node{})
        : data_{std::make_shared<RawData>(Shape::flat_size)}, node_{std::make_shared<Node>(node)} {
        if constexpr (Shape::rank == 0) {
            (*data_)[0] = data;
        } else {
            auto flat_data = reinterpret_cast<const DType*>(data.data());
            std::copy(flat_data, flat_data + Shape::flat_size, data_->begin());
        }
    }

    Tensor(const std::shared_ptr<RawData>& data, Node&& node = Node{})
        : data_{data}, node_{std::make_shared<Node>(node)} {
        assert(data_->size() == Shape::flat_size);
    }

    template <IsShape NewShape>
        requires(NewShape::flat_size == Shape::flat_size)
    auto reshape() const {
        // TODO: backprop
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

    auto detach() const { return Detached{data_}; }

    const auto get_node() const { return node_; }

    void _init_entry(Size index, DType value) { (*data_)[index] = value; }

   private:
    std::shared_ptr<RawData> data_;
    std::shared_ptr<Node> node_;
};

template <typename T>
concept IsTensor = std::is_same_v<T, Tensor<typename T::Shape, typename T::DType, typename T::Node>>;

template <typename A>
concept IsFloatTensor = IsTensor<A> && std::is_floating_point_v<typename A::DType>;

template <typename A>
concept IsScalarTensor = IsTensor<A> && A::Shape::rank == 0;

template <typename A, typename B>
concept TensorDTypeCompatible = IsTensor<A> && IsTensor<B> && std::is_same_v<typename A::DType, typename B::DType>;

template <typename A, typename B>
concept TensorShapeCompatible = IsTensor<A> && IsTensor<B> && std::is_same_v<typename A::Shape, typename B::Shape>;

template <typename A, typename B>
concept TensorMatmulCompatible = IsTensor<A> && IsTensor<B> && A::Shape::rank >= 2 && B::Shape::rank >= 2 &&
                                 A::Shape::template At<-1>::value == B::Shape::template At<-2>::value;

}  // namespace vgrad

#endif  // VGRAD_TENSOR_H_