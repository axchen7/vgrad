#ifndef VGRAD_OPS_H_
#define VGRAD_OPS_H_

#include <span>

#include "graph.h"
#include "tensor.h"

namespace vgrad {

template <typename A, typename B>
concept TensorBinaryOpCompatible = TensorDTypeCompatible<A, B> && TensorShapeCompatible<A, B>;

using OneDimension = Dimension<1>;  // for unsqueeze

template <IsTensor A>
auto unary_op(const A& a, auto forward, auto backward) {
    using Node = UnaryOpNode<typename A::Node, typename A::Shape, typename A::DType>;

    Tensor<typename A::Shape, typename A::DType, Node> result{
        Node{a.get_node(), [a, backward](const auto& dl_df) {
                 Tensor<typename A::Shape, typename A::DType> df_da;
                 for (Size i = 0; i < A::Shape::flat_size; i++) {
                     df_da._init_entry(i, backward(a.flat_view()[i]));
                 }
                 auto dl_da = dl_df * df_da;
                 return dl_da.detach();
             }}};

    for (Size i = 0; i < A::Shape::flat_size; i++) {
        result._init_entry(i, forward(a.flat_view()[i]));
    }
    return result;
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto binary_op(const A& a, const B& b, auto op) {
    Tensor<typename A::Shape, typename A::DType> result;
    for (Size i = 0; i < A::Shape::flat_size; i++) {
        result._init_entry(i, op(a.flat_view()[i], b.flat_view()[i]));
    }
    return result;
}

template <Index I1, Index I2, IsTensor A>
auto transpose(const A& a) {
    constexpr auto idx1 = A::Shape::template normalize_index<I1>();
    constexpr auto idx2 = A::Shape::template normalize_index<I2>();

    using NewShape = typename A::Shape::template Transpose<I1, I2>;
    Tensor<NewShape, typename A::DType> result;

    for (Size i = 0; i < A::Shape::flat_size; i++) {
        auto indices = A::Shape::to_indices(i);
        std::swap(indices[idx1], indices[idx2]);
        auto new_idx = NewShape::to_flat_index(indices);
        result._init_entry(new_idx, a.flat_view()[i]);
    }

    return result;
}

// Remove a singleton dimension from a tensor.
template <Index I, IsTensor A>
    requires(A::Shape::template At<I>::value == 1)
auto squeeze(const A& a) {
    using NewShape = typename A::Shape::template Remove<I>;
    return a.template reshape<NewShape>();
}

template <Index I, IsTensor A>
auto unsqueeze(const A& a) {
    using NewShape = typename A::Shape::template Insert<I, OneDimension>;
    return a.template reshape<NewShape>();
}

// Repeat a singleton dimension of a tensor to make it of dimension Dim.
template <Index I, IsDimension Dim, IsTensor A>
    requires(A::Shape::template At<I>::value == 1)
auto repeat(const A& a) {
    constexpr auto idx = A::Shape::template normalize_index<I>();

    using NewShape = typename A::Shape::template Remove<idx>::template Insert<idx, Dim>;
    Tensor<NewShape, typename A::DType> result;

    for (Size i = 0; i < NewShape::flat_size; i++) {
        auto indices = NewShape::to_indices(i);
        indices[idx] = 0;
        auto flat_idx = A::Shape::to_flat_index(indices);
        result._init_entry(i, a.flat_view()[flat_idx]);
    }

    return result;
}

// Reduce the last dimension of a tensor.
template <IsTensor A>
auto reduce(const A& a, auto op) {
    using LastDim = typename A::Shape::template At<-1>;
    using NewShape = typename A::Shape::template Remove<-1>;
    Tensor<NewShape, typename A::DType> result;

    for (Size i = 0; i < NewShape::flat_size; i++) {
        std::span<const typename A::DType> slice{a.flat_view().begin() + i * LastDim::value, LastDim::value};
        result._init_entry(i, op(slice));
    }
    return result;
}

template <IsTensor A, IsTensor B>
    requires TensorDTypeCompatible<A, B> && TensorMatmulCompatible<A, B>
auto matmul(const A& a, const B& b) {
    // A has shape .. x M x N
    // B has shape .. x N x P
    using M = typename A::Shape::template At<-2>;
    using N = typename A::Shape::template At<-1>;
    using P = typename B::Shape::template At<-1>;

    // expand A to .. x M x 1 x N
    auto c = unsqueeze<-1>(a);
    // expand A to .. x M x P x N
    auto d = repeat<-2, P>(c);

    // transpose B to .. x P x N
    auto e = transpose<-2, -1>(b);
    // expand B to .. x 1 x P x N
    auto f = unsqueeze<-2>(e);
    // expand B to .. x M x P x N
    auto g = repeat<-3, M>(f);

    // element-wise multiplication, still .. x M x P x N
    auto h = d * g;
    // collapse the last dimension, yielding .. x M x P
    return sum(h);
}

template <IsTensor A>
auto operator-(const A& a) {
    return unary_op(a, [](auto x) { return -x; }, [](auto x) { return -x; });
}

template <IsFloatTensor A>
auto exp(const A& a) {
    return unary_op(a, [](auto x) { return std::exp(x); }, [](auto x) { return std::exp(x); });
}

template <IsFloatTensor A>
auto log(const A& a) {
    return unary_op(a, [](auto x) { return std::log(x); }, [](auto x) { return 1 / x; });
}

template <IsFloatTensor A>
auto relu(const A& a) {
    return unary_op(a, [](auto x) { return x > 0 ? x : 0; }, [](auto x) { return x > 0 ? 1 : 0; });
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
    return unary_op(a, [b](auto x) { return x + b; }, [](auto x) { return 1; });
}

template <IsTensor B>
auto operator+(typename B::DType a, const B& b) {
    return unary_op(b, [a](auto x) { return a + x; }, [](auto x) { return 1; });
}

template <IsTensor A>
auto operator-(const A& a, typename A::DType b) {
    return unary_op(a, [b](auto x) { return x - b; }, [](auto x) { return 1; });
}

template <IsTensor B>
auto operator-(typename B::DType a, const B& b) {
    return unary_op(b, [a](auto x) { return a - x; }, [](auto x) { return -1; });
}

template <IsTensor A>
auto operator*(const A& a, typename A::DType b) {
    return unary_op(a, [b](auto x) { return x * b; }, [b](auto x) { return b; });
}

template <IsTensor B>
auto operator*(typename B::DType a, const B& b) {
    return unary_op(b, [a](auto x) { return a * x; }, [a](auto x) { return a; });
}

template <IsFloatTensor A>
auto operator/(const A& a, typename A::DType b) {
    return unary_op(a, [b](auto x) { return x / b; }, [b](auto x) { return 1 / b; });
}

template <IsFloatTensor B>
auto operator/(typename B::DType a, const B& b) {
    return unary_op(b, [a](auto x) { return a / x; }, [a](auto x) { return -a / (x * x); });
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