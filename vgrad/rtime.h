#ifndef VGRAD_RTIME_H_
#define VGRAD_RTIME_H_

#include "shape.h"
#include "typehint.h"

namespace vgrad::rtime {

using ConstantType = long long;

template <typename T>
concept IsConstTerm = requires {
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

template <ConstantType _ns_constant>
struct ConstTerm {
    static constexpr bool is_const_term = true;

    static constexpr ConstantType ns_constant = _ns_constant;

    static constexpr auto typehint_type() { return typehint::to_string(ns_constant) + "ns"; }
};

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

    // de-dupe and sort, such that outer dimensions < inner dimensions
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

constexpr auto make_product_term() { return EmptyProductTerm{}; }

template <IsPolyTerm Outer, IsPolyTerm... Rest>
constexpr auto make_product_term(Outer outer, Rest... rest) {
    return ProductTerm<Outer, decltype(make_product_term(rest...))>{};
}

template <IsPolyTerm Outer, IsPolyTerm... Rest>
using MakeProductTerm = decltype(make_product_term(Outer{}, Rest{}...));

}  // namespace vgrad::rtime

#endif  // VGRAD_RTIME_H_