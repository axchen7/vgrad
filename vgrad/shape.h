#ifndef VGRAD_SHAPE_H_
#define VGRAD_SHAPE_H_

#include <array>

#include "typehint.h"
#include "types.h"

namespace vgrad {

template <Size V
#ifdef __APPLE__
          ,
          typehint::StringLiteral Name = ""
#endif
          >
    requires(V > 0)
struct Dimension {
    static constexpr Size value = V;

#ifdef __APPLE__
    static constexpr auto typehint_type() {
        if constexpr (Name.value[0] == '\0') {
            return typehint::to_string(value);
        } else {
            return std::string{Name.value};
        }
    }
#endif
};

template <typename A, Index I>
concept IsValidIndex =
    IsShape<A> &&
    ((I >= 0 && I < static_cast<Index>(A::rank)) ||
     ((I + static_cast<Index>(A::rank)) >= 0 && (I + static_cast<Index>(A::rank)) < static_cast<Index>(A::rank)));

template <IsDimension Outer, IsShape Inner>
struct Shape {
   public:
    static constexpr bool is_shape = true;

    static constexpr Outer outer;
    static constexpr Inner inner;

    static constexpr Size rank = 1 + Inner::rank;
    static constexpr Size flat_size = Outer::value * Inner::flat_size;

    static constexpr auto compute_strides() {
        std::array<Size, rank> result{};
        result[0] = inner.flat_size;
        if constexpr (rank > 1) {
            for (Size i = 0; i < inner.rank; i++) {
                result[i + 1] = inner.strides[i];
            }
        }
        return result;
    }

    static constexpr auto strides = compute_strides();

    template <Index I>
        requires IsValidIndex<Shape<Outer, Inner>, I>
    static constexpr Size normalize_index() {
        return I < 0 ? rank + I : I;
    }

    template <Index I>
    static constexpr auto at() {
        constexpr auto i = normalize_index<I>();
        if constexpr (i == 0) {
            return outer;
        } else {
            return inner.template at<i - 1>();
        }
    }

    template <Index I>
    using At = decltype(at<I>());

    template <Index I>
    static constexpr auto remove() {
        constexpr auto i = normalize_index<I>();
        if constexpr (i == 0) {
            return inner;
        } else {
            return Shape<Outer, decltype(inner.template remove<i - 1>())>{};
        }
    }

    template <Index I>
    using Remove = decltype(remove<I>());

    template <Index I, IsDimension Dim>
    static constexpr auto insert() {
        if constexpr (I == rank && Inner::rank == 0) {
            return Shape<Outer, Shape<Dim, Inner>>{};
        } else if constexpr (I == rank) {
            return Shape<Outer, decltype(inner.template insert<I - 1, Dim>())>{};
        } else {
            constexpr auto i = normalize_index<I>();
            if constexpr (i == 0) {
                return Shape<Dim, Shape<Outer, Inner>>{};
            } else {
                return Shape<Outer, decltype(inner.template insert<i - 1, Dim>())>{};
            }
        }
    }

    template <Index I, IsDimension Dim>
    using Insert = decltype(insert<I, Dim>());

    template <Index I1, Index I2>
    static constexpr auto transpose() {
        // (must use normalized idx because we change the rank)
        constexpr auto i1 = normalize_index<I1>();
        constexpr auto i2 = normalize_index<I2>();
        using D1 = At<i1>;
        using D2 = At<i2>;
        return Shape<Outer, Inner>{}
            .template remove<i1>()
            .template insert<i1, D2>()
            .template remove<i2>()
            .template insert<i2, D1>();
    }

    template <Index I1, Index I2>
    using Transpose = decltype(transpose<I1, I2>());

    template <Size Count>
    static constexpr auto last() {
        static_assert(Count <= rank);
        if constexpr (Count == rank) {
            return Shape<Outer, Inner>{};
        } else {
            return inner.template last<Count>();
        }
    }

    template <Size Count>
    using Last = decltype(last<Count>());

    static constexpr auto to_indices(Size flat_index) {
        std::array<Size, rank> result{};
        for (Size i = 0; i < rank; i++) {
            result[i] = flat_index / strides[i];
            flat_index %= strides[i];
        }
        return result;
    }

    static constexpr auto to_flat_index(std::array<Size, rank> indices) {
        Size result = 0;
        for (Size i = 0; i < rank; i++) {
            result += indices[i] * strides[i];
        }
        return result;
    }

#ifdef __APPLE__
    static constexpr auto typehint_type() {
        auto result = Outer::typehint_type();
        if constexpr (Inner::rank > 0) {
            result += " x " + Inner::typehint_type();
        }
        return result;
    }
#endif
};

struct ScalarShape {
    static constexpr bool is_shape = true;
    static constexpr Size rank = 0;
    static constexpr Size flat_size = 1;

    template <Index I, IsDimension Dim>
        requires(I == 0)
    static constexpr auto insert() {
        return Shape<Dim, ScalarShape>{};
    }

    template <Index I, IsDimension Dim>
    using Insert = decltype(insert<I, Dim>());

    template <Size Count>
    static constexpr auto last() {
        static_assert(Count == 0);
        return ScalarShape{};
    }

    template <Size Count>
    using Last = decltype(last<Count>());

#ifdef __APPLE__
    static constexpr auto typehint_type() { return std::string{"scalar"}; }
#endif
};

constexpr auto make_shape() { return ScalarShape{}; }

template <IsDimension Outer, IsDimension... Rest>
constexpr auto make_shape(Outer outer, Rest... rest) {
    return Shape<Outer, decltype(make_shape(rest...))>{};
}

template <IsDimension Outer, IsDimension... Rest>
using MakeShape = decltype(make_shape(Outer{}, Rest{}...));

}  // namespace vgrad

#endif  // VGRAD_SHAPE_H_