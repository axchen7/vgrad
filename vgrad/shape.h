#include <concepts>
#include <type_traits>

namespace vgrad {

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
class Shape<Outer, Inner> {
   public:
    static constexpr bool is_shape = true;

    static constexpr Outer outer;
    static constexpr Inner inner;

    static constexpr int rank = 1 + Inner::rank;
    static constexpr int flat_size = Inner::rank == 0 ? Outer::value : Outer::value * Inner::flat_size;

    template <int I>
    static constexpr auto at() {
        constexpr int i = normalize_index<I>();
        if constexpr (i == 0) {
            return outer;
        } else {
            return inner.template at<i - 1>();
        }
    }

    template <int I>
    static constexpr auto squeeze() {
        constexpr int i = normalize_index<I>();
        if constexpr (i == 0) {
            return inner;
        } else {
            return Shape<Outer, decltype(inner.template squeeze<i - 1>())>{};
        }
    }

    template <int I, IsDimension D>
    static constexpr auto unsqueeze(D dim) {
        if constexpr (I == rank) {
            return Shape<Outer, Shape<Inner, D>>{};
        } else {
            constexpr int i = normalize_index<I>();
            if constexpr (i == 0) {
                return Shape<D, Shape<Outer, Inner>>{};
            } else {
                return Shape<Outer, decltype(inner.template unsqueeze<i - 1>(dim))>{};
            }
        }
    }

    template <int I1, int I2>
    static constexpr auto transpose() {
        constexpr int i1 = normalize_index<I1>();
        constexpr int i2 = normalize_index<I2>();
        auto d1 = at<i1>();
        auto d2 = at<i2>();
        return Shape<Outer, Inner>{}
            .template squeeze<i1>()
            .template unsqueeze<i1>(d2)
            .template squeeze<i2>()
            .template unsqueeze<i2>(d1);
    }

   private:
    template <int I>
    static constexpr int normalize_index() {
        if constexpr (I < 0) {
            constexpr int i = rank + I;
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
    static constexpr int rank = 0;
    static constexpr int flat_size = 0;
};

constexpr auto make_shape() { return EmptyShape{}; }

template <IsDimension Outer, IsDimension... Rest>
constexpr auto make_shape(Outer outer, Rest... rest) {
    return Shape<Outer, decltype(make_shape(rest...))>{};
}

}  // namespace vgrad