#include <concepts>
#include <type_traits>

template <typename T>
concept IsDimension = requires {
    { T::value } -> std::same_as<const int&>;
};

template <int V>
struct Dimension {
    static constexpr int value = V;
};

struct BaseShape {};

template <typename T>
concept IsShape = std::is_base_of_v<BaseShape, T>;

template <typename Outer, typename Inner>
struct Shape {};

template <IsDimension Outer, IsShape Inner>
struct Shape<Outer, Inner> : BaseShape {
    static constexpr Outer outer;
    static constexpr Inner inner;
};
