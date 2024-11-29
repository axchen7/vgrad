#ifndef VGRAD_RTIME_H_
#define VGRAD_RTIME_H_

#include "shape.h"
#include "typehint.h"

namespace vgrad::rtime {

using ConstantValue = long long;

template <typename T>
concept IsConstant = requires {
    { T::is_const_term } -> std::same_as<const bool&>;
};

template <typename T>
concept IsPolyTerm = requires {
    { T::is_poly_term } -> std::same_as<const bool&>;
};

template <typename T>
concept IsProductTerm = requires {
    { T::is_product_term } -> std::same_as<const bool&>;
};

template <typename T>
concept IsConstProductTerm = requires {
    { T::is_const_product_term } -> std::same_as<const bool&>;
};

template <typename T>
concept IsRunningTime = requires {
    { T::is_running_time } -> std::same_as<const bool&>;
};

template <ConstantValue Value, typehint::StringLiteral Unit>
struct Constant {
    static constexpr bool is_const_term = true;

    static constexpr ConstantValue value = Value;
    static constexpr typehint::StringLiteral unit = Unit;

    static constexpr auto typehint_type() { return typehint::to_string(value) + " " + std::string{unit.value}; }
};

template <typename Const1, typename Const2>
concept ConstantUnitsMatch =
    IsConstant<Const1> && IsConstant<Const2> &&
    typehint::string_compare(std::string(Const1::unit.value), std::string(Const2::unit.value)) == 0;

template <IsConstant Const1, IsConstant Const2>
    requires(ConstantUnitsMatch<Const1, Const2>)
constexpr auto add_const_terms(Const1, Const2) {
    return Constant<Const1::value + Const2::value, Const1::unit>{};
}

template <IsConstant Const1, IsConstant Const2>
using AddConstants = decltype(add_const_terms(Const1{}, Const2{}));

template <IsDimension _Dim, int _power>
struct PolyTerm {
    static constexpr bool is_poly_term = true;

    using Dim = _Dim;
    static constexpr Size power = _power;

    static constexpr auto typehint_type() { return Dim::typehint_type() + "^" + typehint::to_string(power); }
};

struct EmptyProductTerm {
    static constexpr bool is_product_term = true;
};

template <IsPolyTerm _Outer, IsProductTerm _Inner = EmptyProductTerm>
struct ProductTerm {
    static constexpr bool is_product_term = true;

    using Outer = _Outer;
    using Inner = _Inner;

    // de-dupe and sort
    static constexpr auto normalized() {
        if constexpr (std::is_same_v<Inner, EmptyProductTerm>) {
            return ProductTerm<Outer>{};
        } else {
            using InnerNormalized = decltype(Inner::normalized());

            if constexpr (std::is_same_v<typename Outer::Dim, typename InnerNormalized::Outer::Dim>) {
                using NewOuter = PolyTerm<typename Outer::Dim, Outer::power + InnerNormalized::Outer::power>;
                return ProductTerm<NewOuter, typename InnerNormalized::Inner>{};
            } else if constexpr (typehint::string_compare(Outer::Dim::typehint_type(),
                                                          InnerNormalized::Outer::Dim::typehint_type()) > 0) {
                return ProductTerm<typename InnerNormalized::Outer,
                                   decltype(ProductTerm<Outer, typename InnerNormalized::Inner>::normalized())>{};
            } else {
                return ProductTerm<Outer, InnerNormalized>{};
            }
        }
    }

    static constexpr auto sorted_typehint_type_() {
        if constexpr (std::is_same_v<Inner, EmptyProductTerm>) {
            return Outer::typehint_type();
        } else {
            return Outer::typehint_type() + " x " + Inner::sorted_typehint_type_();
        }
    }

    static constexpr auto typehint_type() {
        using Normalized = decltype(normalized());
        return Normalized::sorted_typehint_type_();
    }
};

template <IsConstant _Constant, IsProductTerm _Product>
struct ConstProductTerm {
    static constexpr bool is_const_product_term = true;

    using Constant = _Constant;
    using Product = _Product;

    static constexpr auto typehint_type() { return Constant::typehint_type() + " x " + Product::typehint_type(); }
};

struct EmptyRunningTime {
    static constexpr bool is_running_time = true;
};

// sum of products
template <IsConstProductTerm _Outer, IsRunningTime _Inner>
struct RunningTime {
    static constexpr bool is_running_time = true;

    using Outer = _Outer;
    using Inner = _Inner;

    // de-dupe and sort
    static constexpr auto normalized() {
        if constexpr (std::is_same_v<Inner, EmptyRunningTime>) {
            return RunningTime<Outer, Inner>{};
        } else {
            using InnerNormalized = decltype(Inner::normalized());

            if constexpr (std::is_same_v<typename Outer::Product, typename InnerNormalized::Outer::Product>) {
                using NewConstant = AddConstants<typename Outer::Constant, typename InnerNormalized::Outer::Constant>;
                using NewConstProduct = ConstProductTerm<NewConstant, typename Outer::Product>;
                return RunningTime<NewConstProduct, typename InnerNormalized::Inner>{};
            } else if constexpr (typehint::string_compare(Outer::Product::typehint_type(),
                                                          InnerNormalized::Outer::Product::typehint_type()) > 0) {
                return RunningTime<typename InnerNormalized::Outer,
                                   decltype(RunningTime<Outer, typename InnerNormalized::Inner>::normalized())>{};
            } else {
                return RunningTime<Outer, InnerNormalized>{};
            }
        }
    }

    static constexpr auto sorted_typehint_type_() {
        if constexpr (std::is_same_v<Inner, EmptyRunningTime>) {
            return Outer::typehint_type();
        } else {
            return Outer::typehint_type() + " + " + Inner::sorted_typehint_type_();
        }
    }

    static constexpr auto typehint_type() {
        using Normalized = decltype(normalized());
        return Normalized::sorted_typehint_type_();
    }
};

constexpr auto make_product_term() { return EmptyProductTerm{}; }

template <IsPolyTerm Outer, IsPolyTerm... Rest>
constexpr auto make_product_term(Outer outer, Rest... rest) {
    return ProductTerm<Outer, decltype(make_product_term(rest...))>{};
}

template <IsPolyTerm Outer, IsPolyTerm... Rest>
using MakeProductTerm = decltype(make_product_term(Outer{}, Rest{}...));

template <IsShape S>
constexpr auto product_term_from_shape(S shape) {
    if constexpr (std::is_same_v<S, ScalarShape>) {
        return EmptyProductTerm{};
    } else {
        constexpr auto outer = PolyTerm<decltype(shape.outer), 1>{};
        return ProductTerm<decltype(outer), decltype(product_term_from_shape(shape.inner))>{};
    }
}

template <IsShape S>
using ProductTermFromShape = decltype(product_term_from_shape(S{}));

constexpr auto make_running_time() { return EmptyRunningTime{}; }

template <IsConstProductTerm Outer, IsConstProductTerm... Rest>
constexpr auto make_running_time(Outer outer, Rest... rest) {
    return RunningTime<Outer, decltype(make_running_time(rest...))>{};
}

template <IsConstProductTerm Outer, IsConstProductTerm... Rest>
using MakeRunningTime = decltype(make_running_time(Outer{}, Rest{}...));

template <IsRunningTime RTime1, IsRunningTime RTime2>
constexpr auto add_running_times(RTime1, RTime2) {
    if constexpr (std::is_same_v<RTime1, EmptyRunningTime>) {
        return RTime2{};
    } else {
        constexpr auto new_inner = add_running_times(typename RTime1::Inner{}, RTime2{});
        return RunningTime<typename RTime1::Outer, decltype(new_inner)>{};
    }
}

template <IsRunningTime RTime1, IsRunningTime RTime2>
using AddRunningTimes = decltype(add_running_times(RTime1{}, RTime2{}));

}  // namespace vgrad::rtime

#endif  // VGRAD_RTIME_H_