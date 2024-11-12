#ifndef VGRAD_CONCEPTS_H_
#define VGRAD_CONCEPTS_H_

#include <concepts>

namespace vgrad {

using Size = unsigned int;
using Index = int;  // allow negative indexing

template <typename T>
concept IsDimension = requires {
    { T::value } -> std::same_as<const Size&>;
} && (T::value > 0);

template <typename T>
concept IsShape = requires {
    { T::is_shape } -> std::same_as<const bool&>;
} && T::is_shape;

template <typename T>
concept IsNode = requires {
    { T::is_node } -> std::same_as<const bool&>;
} && T::is_node;

template <typename T>
concept IsUnaryNode = IsNode<T> && requires {
    { T::is_unary_node } -> std::same_as<const bool&>;
} && T::is_unary_node;

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

}  // namespace vgrad

#endif  // VGRAD_CONCEPTS_H_