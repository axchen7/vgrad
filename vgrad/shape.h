#ifndef VGRAD_SHAPE_H_
#define VGRAD_SHAPE_H_

#include <concepts>
#include <type_traits>

namespace vgrad {

using Size = unsigned int;
using Index = int;  // allow negative indexing

template <typename T>
concept IsDimension = requires {
    { T::value } -> std::same_as<const Size&>;
};

template <typename T>
concept IsShape = requires {
    { T::is_shape } -> std::same_as<const bool&>;
} && T::is_shape;

template <Size V>
struct Dimension {
    static constexpr Size value = V;
};

template <typename Outer, typename Inner>
struct Shape {};

template <IsDimension Outer, IsShape Inner>
class Shape<Outer, Inner> {
   public:
    static constexpr bool is_shape = true;

    static constexpr Outer outer;
    static constexpr Inner inner;

    static constexpr Size rank = 1 + Inner::rank;
    static constexpr Size flat_size = Outer::value * Inner::flat_size;

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

   private:
    template <Index I>
    static constexpr Size normalize_index() {
        if constexpr (I < 0) {
            constexpr auto i = rank + I;
            static_assert(i >= 0 && i < rank, "Invalid index");
            return i;
        } else {
            static_assert(I >= 0 && I < rank, "Invalid index");
            return I;
        }
    }
};

struct EmptyShape {
    static constexpr bool is_shape = true;
    static constexpr Size rank = 0;
    static constexpr Size flat_size = 1;
};

constexpr auto make_shape() { return EmptyShape{}; }

template <IsDimension Outer, IsDimension... Rest>
constexpr auto make_shape(Outer outer, Rest... rest) {
    return Shape<Outer, decltype(make_shape(rest...))>{};
}

template <IsDimension Outer, IsDimension... Rest>
using MakeShape = decltype(make_shape(Outer{}, Rest{}...));

}  // namespace vgrad

#endif  // VGRAD_SHAPE_H_