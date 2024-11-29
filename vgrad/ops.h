#ifndef VGRAD_OPS_H_
#define VGRAD_OPS_H_

#include <omp.h>

#include <span>
#include <stdexcept>

#include "graph.h"
#include "tensor.h"

namespace vgrad {

template <typename A, typename B>
concept TensorBinaryOpCompatible = TensorDTypeCompatible<A, B> && TensorShapeBroadcastCompatible<A, B>;

using OneDimension = Dimension<1>;  // for unsqueeze

template <IsShape NewShape, IsTensor A>
    requires(A::Shape::flat_size == NewShape::flat_size)
auto reshape(const A& a) {
    PROFILE_SCOPE("reshape");
    using Node = UnaryOpNode<typename A::Node, NewShape, typename A::DType, cx::ProductTerm<cx::ZeroPolyTerm>>;

    return Tensor<NewShape, typename A::DType, Node>{
        a.get_data(),
        Node{
            a.get_node(),
            [](const auto& dl_df) {
                PROFILE_SCOPE("reshape::grad");
                return Tensor<typename A::Shape, typename A::DType>{dl_df.get_data()};
            },
        },
    };
}

template <IsShape NewShape, IsTensor B>
    requires TensorShapeBroadcastCompatible<Tensor<NewShape, typename B::DType>, B> &&
             (NewShape::rank >= B::Shape::rank)
auto broadcast(const B& b) {
    PROFILE_SCOPE("broadcast");
    if constexpr (NewShape::rank == B::Shape::rank) {
        return b;
    } else {
        using Repeat = Dimension<NewShape::flat_size / B::Shape::flat_size>;
        return reshape<NewShape>(repeat<0, Repeat>(unsqueeze<0>(b)));
    }
}

template <IsTensor A>
auto _unary_op(const A& a, auto forward, auto backward) {
    PROFILE_SCOPE("_unary_op");
    using Node = UnaryOpNode<typename A::Node, typename A::Shape, typename A::DType,
                             cx::ProductTermFromShape<typename A::Shape>>;

    Tensor<typename A::Shape, typename A::DType, Node> result{Node{
        a.get_node(),
        [a, backward](const auto& dl_df) {
            PROFILE_SCOPE("_unary_op::grad");
            Tensor<typename A::Shape, typename A::DType> dl_da;

#pragma omp parallel for
            for (Size i = 0; i < A::Shape::flat_size; i++) {
                auto df_da = backward(a.flat_view()[i]);
                dl_da._flat_data()[i] = dl_df.flat_view()[i] * df_da;
            }

            return dl_da;
        },
    }};

#pragma omp parallel for
    for (Size i = 0; i < A::Shape::flat_size; i++) {
        result._flat_data()[i] = forward(a.flat_view()[i]);
    }

    return result;
}

template <IsTensor A, IsTensor B>
auto _binary_op_same_shape(const A& a, const B& b, auto forward, auto backward_a, auto backward_b) {
    PROFILE_SCOPE("_binary_op_same_shape");
    using Node = BinaryOpNode<typename A::Node, typename B::Node, typename A::Shape, typename A::DType,
                              cx::ProductTermFromShape<typename A::Shape>>;

    Tensor<typename A::Shape, typename A::DType, Node> result{Node{
        a.get_node(),
        b.get_node(),
        [a, b, backward_a, backward_b](const auto& dl_df) {
            PROFILE_SCOPE("_binary_op_same_shape::grad");
            Tensor<typename A::Shape, typename A::DType> dl_da;
            Tensor<typename B::Shape, typename B::DType> dl_db;

#pragma omp parallel
            for (Size i = 0; i < A::Shape::flat_size; i++) {
                auto df_da = backward_a(a.flat_view()[i], b.flat_view()[i]);
                auto df_db = backward_b(a.flat_view()[i], b.flat_view()[i]);
                dl_da._flat_data()[i] = dl_df.flat_view()[i] * df_da;
                dl_db._flat_data()[i] = dl_df.flat_view()[i] * df_db;
            }

            return std::make_pair(dl_da, dl_db);
        },
    }};

#pragma omp parallel
    for (Size i = 0; i < A::Shape::flat_size; i++) {
        result._flat_data()[i] = forward(a.flat_view()[i], b.flat_view()[i]);
    }

    return result;
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto _binary_op(const A& a, const B& b, auto forward, auto backward_a, auto backward_b) {
    PROFILE_SCOPE("_binary_op");
    if constexpr (A::Shape::rank > B::Shape::rank) {
        return _binary_op_same_shape(a, broadcast<typename A::Shape>(b), forward, backward_a, backward_b);
    } else {
        return _binary_op_same_shape(broadcast<typename B::Shape>(a), b, forward, backward_a, backward_b);
    }
}

template <Index I1, Index I2, IsTensor A>
auto _transpose_no_grad(const A& a) {
    PROFILE_SCOPE("_transpose_no_grad");
    using NewShape = typename A::Shape::template Transpose<I1, I2>;

    if constexpr (std::is_same_v<NewShape, typename A::Shape>) {
        return a;
    }

    Tensor<NewShape, typename A::DType> result;

    constexpr auto idx1 = A::Shape::template normalize_index<I1>();
    constexpr auto idx2 = A::Shape::template normalize_index<I2>();

#pragma omp parallel for
    for (Size i = 0; i < A::Shape::flat_size; i++) {
        auto indices = A::Shape::to_indices(i);
        std::swap(indices[idx1], indices[idx2]);
        auto new_idx = NewShape::to_flat_index(indices);
        result._flat_data()[new_idx] = a.flat_view()[i];
    }

    return result;
}

template <Index I1, Index I2, IsTensor A>
auto transpose(const A& a) {
    PROFILE_SCOPE("transpose");
    auto raw_result = _transpose_no_grad<I1, I2>(a);

    using NewShape = typename decltype(raw_result)::Shape;
    using Node =
        UnaryOpNode<typename A::Node, NewShape, typename A::DType, cx::ProductTermFromShape<typename A::Shape>>;

    return Tensor<NewShape, typename A::DType, Node>{
        raw_result.get_data(),
        Node{
            a.get_node(),
            [](const auto& dl_df) {
                PROFILE_SCOPE("transpose::grad");
                return _transpose_no_grad<I1, I2>(dl_df);
            },
        },
    };
}

// Remove a singleton dimension from a tensor.
template <Index I, IsTensor A>
    requires(A::Shape::template At<I>::value == 1)
auto squeeze(const A& a) {
    PROFILE_SCOPE("squeeze");
    using NewShape = typename A::Shape::template Remove<I>;
    return reshape<NewShape>(a);
}

template <Index I, IsTensor A>
auto unsqueeze(const A& a) {
    PROFILE_SCOPE("unsqueeze");
    using NewShape = typename A::Shape::template Insert<I, OneDimension>;
    return reshape<NewShape>(a);
}

template <IsTensor A>
auto _reduce_last(const A& a, auto forward, auto backward) {
    PROFILE_SCOPE("_reduce_last");
    using LastDim = typename A::Shape::template At<-1>;
    using NewShape = typename A::Shape::template Remove<-1>;
    using Node =
        UnaryOpNode<typename A::Node, NewShape, typename A::DType, cx::ProductTermFromShape<typename A::Shape>>;

    Tensor<NewShape, typename A::DType, Node> result{Node{
        a.get_node(),
        [a, backward](const auto& dl_df) {
            PROFILE_SCOPE("_reduce_last::grad");
            Tensor<typename A::Shape, typename A::DType> dl_da;

#pragma omp parallel for
            for (Size i = 0; i < NewShape::flat_size; i++) {
                std::span<const typename A::DType> slice{a.flat_view().begin() + i * LastDim::value, LastDim::value};
                auto df_da = backward(slice);  // holds a row
                for (Size j = 0; j < LastDim::value; j++) {
                    dl_da._flat_data()[i * LastDim::value + j] = dl_df.flat_view()[i] * df_da[j];
                }
            }

            return dl_da;
        },
    }};

#pragma omp parallel for
    for (Size i = 0; i < NewShape::flat_size; i++) {
        std::span<const typename A::DType> slice{a.flat_view().begin() + i * LastDim::value, LastDim::value};
        result._flat_data()[i] = forward(slice);
    }

    return result;
}

template <Index I, bool KeepDim, IsTensor A>
    requires IsValidIndex<typename A::Shape, I>
auto _reduce(const A& a, auto forward, auto backward) {
    PROFILE_SCOPE("_reduce");
    // pivot index I to the last dimension
    // (must use normalized idx because we change the rank)
    constexpr auto idx = A::Shape::template normalize_index<I>();
    auto b = squeeze<idx>(transpose<idx, -1>(unsqueeze<A::Shape::rank>(a)));

    auto reduced = _reduce_last(b, forward, backward);

    if constexpr (KeepDim) {
        using RemovedDim = typename A::Shape::template At<idx>;
        return repeat<idx, RemovedDim>(unsqueeze<idx>(reduced));
    } else {
        return reduced;
    }
}

// Repeat a singleton dimension of a tensor to make it of dimension Dim.
template <Index I, IsDimension Dim, IsTensor A>
    requires(A::Shape::template At<I>::value == 1)
auto repeat(const A& a) {
    PROFILE_SCOPE("repeat");
    // (must use normalized idx because we change the rank)
    constexpr auto idx = A::Shape::template normalize_index<I>();

    using NewShape = typename A::Shape::template Remove<I>::template Insert<idx, Dim>;
    using Node = UnaryOpNode<typename A::Node, NewShape, typename A::DType, cx::ProductTermFromShape<NewShape>>;

    Tensor<NewShape, typename A::DType, Node> result{Node{
        a.get_node(),
        [a, idx](const auto& dl_df) {
            PROFILE_SCOPE("repeat::grad");
            Tensor<typename A::Shape, typename A::DType> dl_da;

#pragma omp parallel for
            for (Size i = 0; i < A::Shape::flat_size; i++) {
                typename A::DType dl_da_val = 0;

                auto indices = A::Shape::to_indices(i);
                for (Size j = 0; j < Dim::value; j++) {
                    indices[idx] = j;
                    auto flat_idx = NewShape::to_flat_index(indices);
                    dl_da_val += dl_df.flat_view()[flat_idx];
                }
                dl_da._flat_data()[i] = dl_da_val;
            }

            return dl_da;
        },
    }};

#pragma omp parallel for
    for (Size i = 0; i < NewShape::flat_size; i++) {
        auto indices = NewShape::to_indices(i);
        indices[idx] = 0;
        auto flat_idx = A::Shape::to_flat_index(indices);
        result._flat_data()[i] = a.flat_view()[flat_idx];
    }

    return result;
}

template <IsTensor A, IsTensor B>
    requires TensorDTypeCompatible<A, B> && TensorMatmulCompatible<A, B>
auto matmul(const A& a, const B& b) {
    PROFILE_SCOPE("matmul");
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
    PROFILE_SCOPE("operator-::unary");
    return _unary_op(a, [](auto x) { return -x; }, [](auto x) { return -1; });
}

template <IsFloatTensor A>
auto exp(const A& a) {
    PROFILE_SCOPE("exp");
    return _unary_op(a, [](auto x) { return std::exp(x); }, [](auto x) { return std::exp(x); });
}

template <IsFloatTensor A>
auto log(const A& a) {
    PROFILE_SCOPE("log");
    return _unary_op(a, [](auto x) { return std::log(x); }, [](auto x) { return 1 / x; });
}

template <IsTensor A>
auto pow(const A& a, typename A::DType b) {
    PROFILE_SCOPE("pow");
    return _unary_op(a, [b](auto x) { return std::pow(x, b); }, [b](auto x) { return b * std::pow(x, b - 1); });
}

template <IsTensor A>
auto sqrt(const A& a) {
    PROFILE_SCOPE("sqrt");
    return pow(a, 0.5);
}

template <IsFloatTensor A>
auto relu(const A& a) {
    PROFILE_SCOPE("relu");
    return _unary_op(a, [](auto x) { return x > 0 ? x : 0; }, [](auto x) { return x > 0 ? 1 : 0; });
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator+(const A& a, const B& b) {
    PROFILE_SCOPE("operator+::tensor_tensor");
    return _binary_op(
        a, b, [](auto x, auto y) { return x + y; }, [](auto x, auto y) { return 1; }, [](auto x, auto y) { return 1; });
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator-(const A& a, const B& b) {
    PROFILE_SCOPE("operator-::tensor_tensor");
    return _binary_op(
        a, b, [](auto x, auto y) { return x - y; }, [](auto x, auto y) { return 1; },
        [](auto x, auto y) { return -1; });
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator*(const A& a, const B& b) {
    PROFILE_SCOPE("operator*::tensor_tensor");
    return _binary_op(
        a, b, [](auto x, auto y) { return x * y; }, [](auto x, auto y) { return y; }, [](auto x, auto y) { return x; });
}

template <IsFloatTensor A, IsFloatTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator/(const A& a, const B& b) {
    PROFILE_SCOPE("operator/::tensor_tensor");
    return _binary_op(
        a, b, [](auto x, auto y) { return x / y; }, [](auto x, auto y) { return 1 / y; },
        [](auto x, auto y) { return -x / (y * y); });
}

template <IsTensor A, IsTensor B>
    requires TensorBinaryOpCompatible<A, B>
auto operator==(const A& a, const B& b) {
    PROFILE_SCOPE("operator==");
    // backward is never used
    auto result = _binary_op(
        a, b, [](auto x, auto y) { return x == y ? 1 : 0; }, [](auto x, auto y) { return 0; },
        [](auto x, auto y) { return 0; });
    return result.detach();
}

template <IsTensor A>
auto operator+(const A& a, typename A::DType b) {
    PROFILE_SCOPE("operator+::tensor_scalar");
    return _unary_op(a, [b](auto x) { return x + b; }, [](auto x) { return 1; });
}

template <IsTensor B>
auto operator+(typename B::DType a, const B& b) {
    PROFILE_SCOPE("operator+::scalar_tensor");
    return _unary_op(b, [a](auto x) { return a + x; }, [](auto x) { return 1; });
}

template <IsTensor A>
auto operator-(const A& a, typename A::DType b) {
    PROFILE_SCOPE("operator-::tensor_scalar");
    return _unary_op(a, [b](auto x) { return x - b; }, [](auto x) { return 1; });
}

template <IsTensor B>
auto operator-(typename B::DType a, const B& b) {
    PROFILE_SCOPE("operator-::scalar_tensor");
    return _unary_op(b, [a](auto x) { return a - x; }, [](auto x) { return -1; });
}

template <IsTensor A>
auto operator*(const A& a, typename A::DType b) {
    PROFILE_SCOPE("operator*::tensor_scalar");
    return _unary_op(a, [b](auto x) { return x * b; }, [b](auto x) { return b; });
}

template <IsTensor B>
auto operator*(typename B::DType a, const B& b) {
    PROFILE_SCOPE("operator*::scalar_tensor");
    return _unary_op(b, [a](auto x) { return a * x; }, [a](auto x) { return a; });
}

template <IsFloatTensor A>
auto operator/(const A& a, typename A::DType b) {
    PROFILE_SCOPE("operator/::tensor_scalar");
    return _unary_op(a, [b](auto x) { return x / b; }, [b](auto x) { return 1 / b; });
}

template <IsFloatTensor B>
auto operator/(typename B::DType a, const B& b) {
    PROFILE_SCOPE("operator/::scalar_tensor");
    return _unary_op(b, [a](auto x) { return a / x; }, [a](auto x) { return -a / (x * x); });
}

template <Index I = -1, bool KeepDim = false, IsTensor A>
    requires IsValidIndex<typename A::Shape, I>
auto sum(const A& a) {
    PROFILE_SCOPE("sum");
    return _reduce<I, KeepDim>(
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

template <Index I = -1, bool KeepDim = false, IsTensor A>
    requires IsValidIndex<typename A::Shape, I>
auto prod(const A& a) {
    PROFILE_SCOPE("prod");
    return _reduce<I, KeepDim>(
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

// numerically stable, unlike log(sum(exp(x))).
// implementation follows
// https://github.com/pytorch/pytorch/blob/01d98c7cfb2031fd5ab4148444a2d34a171e700c/aten/src/ATen/native/ReduceOps.cpp#L333
template <Index I = -1, bool KeepDim = false, IsTensor A>
    requires IsValidIndex<typename A::Shape, I>
auto logsumexp(const A& a) {
    PROFILE_SCOPE("logsumexp");
    // must avoid exp() of a large input
    // log(sum(exp(a))) = log(sum(exp(a-max[a])) * exp(max[a])) = log(sum(exp(a-max[a]))) + max[a]
    auto max_a = max<I, KeepDim>(a).detach();
    auto max_a_expanded = max<I, true>(a).detach();
    return log(sum<I, KeepDim>(exp(a - max_a_expanded))) + max_a;
}

template <Index I = -1, bool KeepDim = false, IsTensor A>
auto mean(const A& a) {
    PROFILE_SCOPE("mean");
    using SumDim = typename A::Shape::template At<I>;
    return sum<I, KeepDim>(a) / SumDim::value;
}

template <Index I = -1, bool KeepDim = false, IsTensor A>
    requires IsValidIndex<typename A::Shape, I>
auto max(const A& a) {
    PROFILE_SCOPE("max");
    return _reduce<I, KeepDim>(
        a, [](auto x) { return *std::max_element(x.begin(), x.end()); },
        [](auto x) {
            // all 0s except 1 at the max element
            using Dim = typename A::Shape::template At<I>;
            std::array<typename A::DType, Dim::value> row;
            row.fill(0);
            auto max_it = std::max_element(x.begin(), x.end());
            row[std::distance(x.begin(), max_it)] = 1;
            return row;
        });
}

template <Index I = -1, bool KeepDim = false, IsTensor A>
    requires IsValidIndex<typename A::Shape, I>
auto min(const A& a) {
    PROFILE_SCOPE("min");
    return -max<I, KeepDim>(-a);
}

template <Number DType, IsTensor A>
auto _argmax_last(const A& a) {
    PROFILE_SCOPE("_argmax_last");
    using LastDim = typename A::Shape::template At<-1>;
    using NewShape = typename A::Shape::template Remove<-1>;

    Tensor<NewShape, DType> result;

#pragma omp parallel for
    for (Size i = 0; i < NewShape::flat_size; i++) {
        std::span<const typename A::DType> slice{a.flat_view().begin() + i * LastDim::value, LastDim::value};
        auto max_it = std::max_element(slice.begin(), slice.end());
        result._flat_data()[i] = std::distance(slice.begin(), max_it);
    }

    return result;
}

template <Index I = -1, IsTensor A, Number DType = typename A::DType>
    requires IsValidIndex<typename A::Shape, I>
auto argmax(const A& a) {
    PROFILE_SCOPE("argmax");
    // pivot index I to the last dimension
    // (must use normalized idx because we change the rank)
    constexpr auto idx = A::Shape::template normalize_index<I>();
    auto b = squeeze<idx>(transpose<idx, -1>(unsqueeze<A::Shape::rank>(a)));
    return _argmax_last<DType>(b);
}

template <Index I = -1, IsTensor A, Number DType = typename A::DType>
    requires IsValidIndex<typename A::Shape, I>
auto argmin(const A& a) {
    PROFILE_SCOPE("argmin");
    return argmax<I, A, DType>(-a);
}

template <IsDimension Classes, IsTensor A, Number DType = typename A::DType>
    requires IsIntegralTensor<A>
auto one_hot(const A& a) {
    PROFILE_SCOPE("one_hot");
    using NewShape = typename A::Shape::template Insert<A::Shape::rank, Classes>;

    Tensor<NewShape, DType> result;

#pragma omp parallel for
    for (Size i = 0; i < A::Shape::flat_size; i++) {
        auto cur_class = a.flat_view()[i];
        if (cur_class >= Classes::value) {
            throw std::invalid_argument("class index out of range");
        }
        result._flat_data()[i * Classes::value + cur_class] = 1;
    }

    return result;
}

template <Index I = -1, IsTensor A>
auto softmax(const A& a) {
    PROFILE_SCOPE("softmax");
    // must avoid exp() of a large input
    // exp(a_i)/sum(exp(a)) = exp(a_i-max[a])/sum(exp(a-max[a]))
    auto max_a = max<I, true>(a).detach();
    auto exp_a = exp(a - max_a);
    return exp_a / sum<I, true>(exp_a);
}

// numerically stable, unlike log(softmax(x)).
// derived by manipulating formula in
// https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax
template <Index I = -1, IsTensor A>
auto log_softmax(const A& a) {
    PROFILE_SCOPE("log_softmax");
    // log(exp(a_i)/sum(exp(a))) = a_i - log(sum(exp(a)))
    return a - logsumexp<I, true>(a);
}

template <IsTensor Logits, IsTensor Target>
auto cross_entropy(const Logits& logits, const Target& target) {
    PROFILE_SCOPE("cross_entropy");
    // from https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    using Classes = typename Logits::Shape::template At<-1>;
    auto log_probs = log_softmax(logits);
    auto one_hot_target = one_hot<Classes, Target, typename Logits::DType>(target);
    auto per_input_cross_entropy = -sum(one_hot_target * log_probs);
    return mean(per_input_cross_entropy);
}

}  // namespace vgrad

#endif  // VGRAD_OPS_H_