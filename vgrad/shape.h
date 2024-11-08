#include <concepts>
#include <type_traits>

template <typename T>
concept IsDimension = requires {
    { T::value } -> std::same_as<const int&>;
};

template <typename T>
concept IsShape = requires {
    { T::is_shape } -> std::same_as<const bool&>;
} && T::is_shape;

template <int V>
struct Dimension {
    static constexpr int value = V;
};

template <typename Outer, typename Inner>
struct Shape {};

template <IsDimension Outer, IsShape Inner>
struct Shape<Outer, Inner> {
    static constexpr bool is_shape = true;

    static constexpr Outer outer;
    static constexpr Inner inner;

    template <int I>
    static constexpr auto at() {
        if constexpr (I == 0) {
            return outer;
        } else {
            return inner.template at<I - 1>();
        }
    }

    template <int I>
    static constexpr auto squeeze() {
        if constexpr (I == 0) {
            return inner;
        } else {
            return Shape<Outer, decltype(inner.template squeeze<I - 1>())>{};
        }
    }

    template <int I, IsDimension D>
    static constexpr auto unsqueeze(D dim) {
        if constexpr (I == 0) {
            return Shape<D, Shape<Outer, Inner>>{};
        } else {
            return Shape<Outer, decltype(inner.template unsqueeze<I - 1>(dim))>{};
        }
    }
};

struct EmptyShape {
    static constexpr bool is_shape = true;

    template <int I, IsDimension D>
    static constexpr auto unsqueeze(D dim) {
        static_assert(I == 0, "Invalid index");
        return Shape<D, EmptyShape>{};
    }
};