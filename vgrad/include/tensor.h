#ifndef VGRAD_TENSOR_H_
#define VGRAD_TENSOR_H_

#include <array>
#include <cassert>
#include <iostream>
#include <memory>

#include "complexity.h"
#include "profile.h"
#include "shape.h"

namespace vgrad {

template <typename DType>
constexpr std::string dtype_to_string() {
    if constexpr (std::is_same_v<DType, float>) {
        return "float";
    } else if constexpr (std::is_same_v<DType, double>) {
        return "double";
    } else if constexpr (std::is_same_v<DType, int32_t>) {
        return "int32";
    } else if constexpr (std::is_same_v<DType, int64_t>) {
        return "int64";
    } else {
        return "unknown";
    }
}

template <Number DType>
using MemoryConstant = cx::Constant<sizeof(DType), "B">;

using TimeConstant = cx::Constant<1, "ops">;

template <IsShape _OutShape, Number _DType>
struct LeafNode {
    static constexpr bool is_node = true;
    static constexpr bool is_leaf_node = true;
    using DType = _DType;
    using OutShape = _OutShape;

    using Cx = cx::ProductTermFromShape<OutShape>;
    using TotalMemoryComplexity = cx::MakeComplexity<cx::ConstProductTerm<MemoryConstant<DType>, Cx>>;
    using TotalTimeComplexity = cx::MakeComplexity<cx::ConstProductTerm<TimeConstant, Cx>>;
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
    using Detached = Tensor<Shape, DType>;

    static constexpr auto mem_complexity = typename Node::TotalMemoryComplexity{};
    static constexpr auto time_complexity = typename Node::TotalTimeComplexity{};

    // data is initialized to zeros
    Tensor(Node&& node = Node{}) : data_{std::make_shared<FlatData>()}, node_{std::make_shared<Node>(node)} {}

    Tensor(const NestedData& data, Node&& node = Node{})
        : data_{std::make_shared<FlatData>()}, node_{std::make_shared<Node>(node)} {
        if constexpr (Shape::rank == 0) {
            (*data_)[0] = data;
        } else {
            auto flat_data = reinterpret_cast<const DType*>(data.data());
            std::copy(flat_data, flat_data + Shape::flat_size, data_->begin());
        }
    }

    Tensor(const std::shared_ptr<FlatData>& data, Node&& node = Node{})
        : data_{data}, node_{std::make_shared<Node>(node)} {
        assert(data_->size() == Shape::flat_size);
    }

    const FlatData& flat_view() const { return *data_; }

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

    template <typename T>
    auto& operator-=(const T& other)
        requires IsLeafNode<Node>
    {
        auto result = *this - other;
        this->data_ = std::make_shared<FlatData>(result.flat_view());
        return *this;
    }

    auto detach() const { return Detached{data_}; }

    const auto get_data() const { return data_; }

    const auto get_node() const { return node_; }

    void _init_entry(Size index, DType value) { (*data_)[index] = value; }

    // Escape hatch for mutating the data on init. If only viewing, use
    // flat_view() instead.
    FlatData& _flat_data() { return *data_; }

    auto& bind_profile(profile::ProfileNode& profile_node) const {
        profile_node.add_hook([](profile::ProfileHookDuration duration, std::ostream& os) {
            double duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
            double per_op_duration_ns = duration_ns / time_complexity.total.value;
            os << per_op_duration_ns << "ns" << " / " << time_complexity.total.unit;
        });
        profile_node.add_hook([](profile::ProfileHookDuration duration, std::ostream& os) {
            os << time_complexity.total.typehint_type() << " total";
        });
        profile_node.add_hook(
            [](profile::ProfileHookDuration duration, std::ostream& os) { os << "shape: " << Shape::typehint_type(); });
        return *this;
    }

    static constexpr auto typehint_type() {
        auto shape = Shape::typehint_type();
        auto type = dtype_to_string<DType>();
        return shape + ", " + type;
    }

   private:
    std::shared_ptr<FlatData> data_;
    std::shared_ptr<Node> node_;
};

template <IsShape Shape, Number DType, typename Node>
    requires(Shape::rank == 0)
std::ostream& operator<<(std::ostream& os, const Tensor<Shape, DType, Node>& tensor) {
    os << tensor.value();
    return os;
}

template <IsShape Shape, Number DType, typename Node>
    requires(Shape::rank == 1)
std::ostream& operator<<(std::ostream& os, const Tensor<Shape, DType, Node>& tensor) {
    const auto& data = tensor.nested_view();
    os << "[ ";
    for (Size i = 0; i < Shape::flat_size; i++) {
        os << data[i] << ' ';
    }
    os << "]";
    return os;
}

template <IsShape Shape, Number DType, typename Node>
    requires(Shape::rank == 2)
std::ostream& operator<<(std::ostream& os, const Tensor<Shape, DType, Node>& tensor) {
    const auto& data = tensor.nested_view();
    os << "[\n";
    for (const auto& row : data) {
        for (const auto& elem : row) {
            os << elem << ' ';
        }
        os << '\n';
    }
    os << "]";
    return os;
}

template <typename T>
concept IsTensor = std::is_same_v<T, Tensor<typename T::Shape, typename T::DType, typename T::Node>>;

template <typename A>
concept IsIntegralTensor = IsTensor<A> && std::is_integral_v<typename A::DType>;

template <typename A>
concept IsFloatTensor = IsTensor<A> && std::is_floating_point_v<typename A::DType>;

template <typename A>
concept IsScalarTensor = IsTensor<A> && A::Shape::rank == 0;

template <typename A, typename B>
concept TensorDTypeCompatible = IsTensor<A> && IsTensor<B> && std::is_same_v<typename A::DType, typename B::DType>;

template <typename A, typename B>
concept TensorShapeCompatible = IsTensor<A> && IsTensor<B> && std::is_same_v<typename A::Shape, typename B::Shape>;

template <typename A, typename B>
concept TensorShapeBroadcastCompatible =
    IsTensor<A> && IsTensor<B> &&
    std::is_same_v<typename A::Shape::template Last<std::min(A::Shape::rank, B::Shape::rank)>,
                   typename B::Shape::template Last<std::min(A::Shape::rank, B::Shape::rank)>>;

template <typename A, typename B>
concept TensorMatmulCompatible = IsTensor<A> && IsTensor<B> && A::Shape::rank >= 2 && B::Shape::rank >= 2 &&
                                 A::Shape::template At<-1>::value == B::Shape::template At<-2>::value;

}  // namespace vgrad

#endif  // VGRAD_TENSOR_H_