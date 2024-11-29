#ifndef VGRAD_COMPLEXITY_H_
#define VGRAD_COMPLEXITY_H_

#include "shape.h"
#include "typehint.h"

namespace vgrad::cx {  // cx -> complexity

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
concept IsComplexity = requires {
    { T::is_complexity } -> std::same_as<const bool&>;
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

    static constexpr auto typehint_type() {
        if constexpr (power == 1) {
            return Dim::typehint_type();
        } else {
            return Dim::typehint_type() + "^" + typehint::to_string(power);
        }
    }
};

struct EmptyProductTerm {
    static constexpr bool is_product_term = true;

    static constexpr std::string typehint_type() { return "0"; }
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

struct EmptyComplexity {
    static constexpr bool is_complexity = true;

    static constexpr auto normalized() { return EmptyComplexity{}; }
};

// sum of products
template <IsConstProductTerm _Outer, IsComplexity _Inner>
struct Complexity {
    static constexpr bool is_complexity = true;

    using Outer = _Outer;
    using Inner = _Inner;

    // de-dupe and sort
    static constexpr auto normalized() {
        using InnerNormalized = decltype(Inner::normalized());

        if constexpr (std::is_same_v<InnerNormalized, EmptyComplexity>) {
            return Complexity<Outer, InnerNormalized>{};
        } else {
            if constexpr (std::is_same_v<typename Outer::Product, typename InnerNormalized::Outer::Product>) {
                using NewConstant = AddConstants<typename Outer::Constant, typename InnerNormalized::Outer::Constant>;
                using NewConstProduct = ConstProductTerm<NewConstant, typename Outer::Product>;
                return Complexity<NewConstProduct, typename InnerNormalized::Inner>{};
            } else if constexpr (typehint::string_compare(Outer::Product::typehint_type(),
                                                          InnerNormalized::Outer::Product::typehint_type()) > 0) {
                return Complexity<typename InnerNormalized::Outer,
                                  decltype(Complexity<Outer, typename InnerNormalized::Inner>::normalized())>{};
            } else {
                return Complexity<Outer, InnerNormalized>{};
            }
        }
    }

    static constexpr auto sorted_typehint_type_() {
        if constexpr (std::is_same_v<Inner, EmptyComplexity>) {
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

constexpr auto make_complexity() { return EmptyComplexity{}; }

template <IsConstProductTerm Outer, IsConstProductTerm... Rest>
constexpr auto make_complexity(Outer outer, Rest... rest) {
    return Complexity<Outer, decltype(make_complexity(rest...))>{};
}

template <IsConstProductTerm Outer, IsConstProductTerm... Rest>
using MakeComplexity = decltype(make_complexity(Outer{}, Rest{}...));

template <IsComplexity Cx1, IsComplexity Cx2>
constexpr auto add_complexities(Cx1, Cx2) {
    if constexpr (std::is_same_v<Cx1, EmptyComplexity>) {
        return Cx2{};
    } else {
        constexpr auto new_inner = add_complexities(typename Cx1::Inner{}, Cx2{});
        return Complexity<typename Cx1::Outer, decltype(new_inner)>{};
    }
}

template <IsComplexity Cx1, IsComplexity Cx2>
using AddComplexities = decltype(add_complexities(Cx1{}, Cx2{}));

}  // namespace vgrad::cx

#endif  // VGRAD_COMPLEXITY_H_