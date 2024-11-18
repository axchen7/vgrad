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
auto _unary_op(const A& a, auto forward, auto backward) {
    using Node = UnaryOpNode<typename A::Node, typename A::Shape, typename A::DType>;

    Tensor<typename A::Shape, typename A::DType, Node> result{Node{
        a.get_node(),
        [a, backward](const auto& dl_df) {
            Tensor<typename A::Shape, typename A::DType> dl_da;
            for (Size i = 0; i < A::Shape::flat_size; i++) {
                auto df_da = backward(a.flat_view()[i]);
                dl_da._init_entry(i, dl_df.flat_view()[i] * df_da);
            }
            return dl_da;
        },
    }};

    for (Size i = 0; i < A::Shape::flat_size; i++) {
        result._init_entry(i, forward(a.flat_view()[i]));
    }
    return result;
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto _binary_op(const A& a, const B& b, auto forward, auto backward) {
    using Node = BinaryOpNode<typename A::Node, typename B::Node, typename A::Shape, typename A::DType>;

    Tensor<typename A::Shape, typename A::DType, Node> result{Node{
        a.get_node(),
        b.get_node(),
        [a, b, backward](const auto& dl_df) {
            Tensor<typename A::Shape, typename A::DType> dl_da;
            Tensor<typename B::Shape, typename B::DType> dl_db;
            for (Size i = 0; i < A::Shape::flat_size; i++) {
                auto [df_da, df_db] = backward(a.flat_view()[i], b.flat_view()[i]);

                dl_da._init_entry(i, dl_df.flat_view()[i] * df_da);
                dl_db._init_entry(i, dl_df.flat_view()[i] * df_db);
            }

            return std::make_pair(dl_da, dl_db);
        },
    }};

    for (Size i = 0; i < A::Shape::flat_size; i++) {
        result._init_entry(i, forward(a.flat_view()[i], b.flat_view()[i]));
    }
    return result;
}

template <Index I1, Index I2, IsTensor A>
auto _transpose_no_grad(const A& a) {
    using NewShape = typename A::Shape::template Transpose<I1, I2>;

    if constexpr (std::is_same_v<NewShape, typename A::Shape>) {
        return a;
    }

    Tensor<NewShape, typename A::DType> result;

    constexpr auto idx1 = A::Shape::template normalize_index<I1>();
    constexpr auto idx2 = A::Shape::template normalize_index<I2>();

    for (Size i = 0; i < A::Shape::flat_size; i++) {
        auto indices = A::Shape::to_indices(i);
        std::swap(indices[idx1], indices[idx2]);
        auto new_idx = NewShape::to_flat_index(indices);
        result._init_entry(new_idx, a.flat_view()[i]);
    }
    return result;
}

template <Index I1, Index I2, IsTensor A>
auto transpose(const A& a) {
    auto raw_result = _transpose_no_grad<I1, I2>(a);

    using NewShape = typename decltype(raw_result)::Shape;
    using Node = UnaryOpNode<typename A::Node, NewShape, typename A::DType>;

    return Tensor<NewShape, typename A::DType, Node>{
        raw_result.get_data(),
        Node{
            a.get_node(),
            [](const auto& dl_df) { return _transpose_no_grad<I1, I2>(dl_df); },
        },
    };
}

// Remove a singleton dimension from a tensor.
template <Index I, IsTensor A>
    requires(A::Shape::template At<I>::value == 1)
auto squeeze(const A& a) {
    using NewShape = typename A::Shape::template Remove<I>;
    using Node = UnaryOpNode<typename A::Node, NewShape, typename A::DType>;

    return Tensor<NewShape, typename A::DType, Node>{
        a.get_data(),
        Node{
            a.get_node(),
            [](const auto& dl_df) { return Tensor<typename A::Shape, typename A::DType>{dl_df.get_data()}; },
        },
    };
}

template <Index I, IsTensor A>
auto unsqueeze(const A& a) {
    using NewShape = typename A::Shape::template Insert<I, OneDimension>;
    using Node = UnaryOpNode<typename A::Node, NewShape, typename A::DType>;

    return Tensor<NewShape, typename A::DType, Node>{
        a.get_data(),
        Node{
            a.get_node(),
            [](const auto& dl_df) { return Tensor<typename A::Shape, typename A::DType>{dl_df.get_data()}; },
        },
    };
}

template <IsTensor A>
auto _reduce_last(const A& a, auto forward, auto backward) {
    using LastDim = typename A::Shape::template At<-1>;
    using NewShape = typename A::Shape::template Remove<-1>;
    using Node = UnaryOpNode<typename A::Node, NewShape, typename A::DType>;

    Tensor<NewShape, typename A::DType, Node> result{Node{
        a.get_node(),
        [a, backward](const auto& dl_df) {
            Tensor<typename A::Shape, typename A::DType> dl_da;

            for (Size i = 0; i < NewShape::flat_size; i++) {
                std::span<const typename A::DType> slice{a.flat_view().begin() + i * LastDim::value, LastDim::value};
                auto df_da = backward(slice);  // holds a row
                for (Size j = 0; j < LastDim::value; j++) {
                    dl_da._init_entry(i * LastDim::value + j, dl_df.flat_view()[i] * df_da[j]);
                }
            }
            return dl_da;
        },
    }};

    for (Size i = 0; i < NewShape::flat_size; i++) {
        std::span<const typename A::DType> slice{a.flat_view().begin() + i * LastDim::value, LastDim::value};
        result._init_entry(i, forward(slice));
    }
    return result;
}

template <Index I, IsTensor A>
auto _reduce(const A& a, auto forward, auto backward) {
    // pivot index I to the last dimension
    // (must use normalized idx because we change the rank)
    constexpr auto idx = A::Shape::template normalize_index<I>();
    auto b = squeeze<idx>(transpose<idx, -1>(unsqueeze<A::Shape::rank>(a)));

    return _reduce_last(b, forward, backward);
}

// Repeat a singleton dimension of a tensor to make it of dimension Dim.
template <Index I, IsDimension Dim, IsTensor A>
    requires(A::Shape::template At<I>::value == 1)
auto repeat(const A& a) {
    // (must use normalized idx because we change the rank)
    constexpr auto idx = A::Shape::template normalize_index<I>();

    using NewShape = typename A::Shape::template Remove<I>::template Insert<idx, Dim>;
    using Node = UnaryOpNode<typename A::Node, NewShape, typename A::DType>;

    Tensor<NewShape, typename A::DType, Node> result{Node{
        a.get_node(),
        [a](const auto& dl_df) {
            Tensor<typename A::Shape, typename A::DType> dl_da;

            for (Size i = 0; i < A::Shape::flat_size; i++) {
                typename A::DType dl_da_val = 0;

                auto indices = A::Shape::to_indices(i);
                for (Size j = 0; j < Dim::value; j++) {
                    indices[idx] = j;
                    auto flat_idx = NewShape::to_flat_index(indices);
                    dl_da_val += dl_df.flat_view()[flat_idx];
                }
                dl_da._init_entry(i, dl_da_val);
            }
            return dl_da;
        },
    }};

    for (Size i = 0; i < NewShape::flat_size; i++) {
        auto indices = NewShape::to_indices(i);
        indices[idx] = 0;
        auto flat_idx = A::Shape::to_flat_index(indices);
        result._init_entry(i, a.flat_view()[flat_idx]);
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
    return _unary_op(a, [](auto x) { return -x; }, [](auto x) { return -x; });
}

template <IsFloatTensor A>
auto exp(const A& a) {
    return _unary_op(a, [](auto x) { return std::exp(x); }, [](auto x) { return std::exp(x); });
}

template <IsFloatTensor A>
auto log(const A& a) {
    return _unary_op(a, [](auto x) { return std::log(x); }, [](auto x) { return 1 / x; });
}

template <IsTensor A>
auto pow(const A& a, typename A::DType b) {
    return _unary_op(a, [b](auto x) { return std::pow(x, b); }, [b](auto x) { return b * std::pow(x, b - 1); });
}

template <IsFloatTensor A>
auto relu(const A& a) {
    return _unary_op(a, [](auto x) { return x > 0 ? x : 0; }, [](auto x) { return x > 0 ? 1 : 0; });
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator+(const A& a, const B& b) {
    return _binary_op(a, b, [](auto x, auto y) { return x + y; }, [](auto x, auto y) { return std::make_pair(1, 1); });
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator-(const A& a, const B& b) {
    return _binary_op(a, b, [](auto x, auto y) { return x - y; }, [](auto x, auto y) { return std::make_pair(1, -1); });
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator*(const A& a, const B& b) {
    return _binary_op(a, b, [](auto x, auto y) { return x * y; }, [](auto x, auto y) { return std::make_pair(y, x); });
}

template <IsFloatTensor A, IsFloatTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator/(const A& a, const B& b) {
    return _binary_op(
        a, b, [](auto x, auto y) { return x / y; }, [](auto x, auto y) { return std::make_pair(1 / y, -x / (y * y)); });
}

template <IsTensor A>
auto operator+(const A& a, typename A::DType b) {
    return _unary_op(a, [b](auto x) { return x + b; }, [](auto x) { return 1; });
}

template <IsTensor B>
auto operator+(typename B::DType a, const B& b) {
    return _unary_op(b, [a](auto x) { return a + x; }, [](auto x) { return 1; });
}

template <IsTensor A>
auto operator-(const A& a, typename A::DType b) {
    return _unary_op(a, [b](auto x) { return x - b; }, [](auto x) { return 1; });
}

template <IsTensor B>
auto operator-(typename B::DType a, const B& b) {
    return _unary_op(b, [a](auto x) { return a - x; }, [](auto x) { return -1; });
}

template <IsTensor A>
auto operator*(const A& a, typename A::DType b) {
    return _unary_op(a, [b](auto x) { return x * b; }, [b](auto x) { return b; });
}

template <IsTensor B>
auto operator*(typename B::DType a, const B& b) {
    return _unary_op(b, [a](auto x) { return a * x; }, [a](auto x) { return a; });
}

template <IsFloatTensor A>
auto operator/(const A& a, typename A::DType b) {
    return _unary_op(a, [b](auto x) { return x / b; }, [b](auto x) { return 1 / b; });
}

template <IsFloatTensor B>
auto operator/(typename B::DType a, const B& b) {
    return _unary_op(b, [a](auto x) { return a / x; }, [a](auto x) { return -a / (x * x); });
}

template <Index I = -1, IsTensor A>
    requires IsValidIndex<typename A::Shape, I>
auto sum(const A& a) {
    return _reduce<I>(
        a,
        [](auto x) {
            typename A::DType sum = 0;
            for (auto i : x) sum += i;
            return sum;
        },
        [](auto x) {
            using Dim = typename A::Shape::template At<I>;
            std::array<typename A::DType, Dim::value> row;
            row.fill(1);
            return row;
        });
}

template <Index I = -1, IsTensor A>
    requires IsValidIndex<typename A::Shape, I>
auto prod(const A& a) {
    return _reduce<I>(
        a,
        [](std::span<const typename A::DType> x) {
            typename A::DType prod = 1;
            for (auto i : x) prod *= i;
            return prod;
        },
        [](auto x) {
            using Dim = typename A::Shape::template At<I>;
            typename A::DType prod = 1;
            for (auto i : x) prod *= i;
            std::array<typename A::DType, Dim::value> row;
            for (Size i = 0; i < Dim::value; i++) {
                row[i] = prod / x[i];
            }
            return row;
        });
}

template <IsTensor A>
auto mean(const A& a) {
    using LastDim = typename A::Shape::template At<-1>;
    return sum(a) / LastDim::value;
}

}  // namespace vgrad

#endif  // VGRAD_OPS_H_