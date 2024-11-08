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

struct BaseShape {
    static constexpr bool is_shape = true;
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
};
