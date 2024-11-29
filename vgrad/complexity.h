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
    TYPEHINT_PASSTHROUGH_CALL
    static constexpr bool is_const_term = true;

    static constexpr ConstantValue value = Value;
    static constexpr typehint::StringLiteral unit = Unit;

    static constexpr bool is_zero = value == 0 && typehint::string_compare(unit.value, "") == 0;

    static constexpr auto typehint_type() {
        if constexpr (is_zero) {
            return "0";
        } else {
            return typehint::to_string(value) + " " + std::string{unit.value};
        }
    }
};

template <typename Const1, typename Const2>
concept CanAddConstants =
    IsConstant<Const1> && IsConstant<Const2> &&
    (Const1::is_zero || Const2::is_zero ||
     typehint::string_compare(std::string(Const1::unit.value), std::string(Const2::unit.value)) == 0);

template <IsConstant Const1, IsConstant Const2>
    requires(CanAddConstants<Const1, Const2>)
constexpr auto add_constants(Const1, Const2) {
    if constexpr (Const1::is_zero) {
        return Const2{};
    } else if constexpr (Const2::is_zero) {
        return Const1{};
    } else {
        return Constant<Const1::value + Const2::value, Const1::unit>{};
    }
}

template <IsConstant Const1, IsConstant Const2>
using AddConstants = decltype(add_constants(Const1{}, Const2{}));

static constexpr std::string product_typehint_(const std::string a, const std::string b) {
    if (typehint::string_compare(a, "0") == 0 || typehint::string_compare(b, "0") == 0) {
        return "0";
    } else if (typehint::string_compare(a, "1") == 0) {
        return b;
    } else if (typehint::string_compare(b, "1") == 0) {
        return a;
    } else {
        return a + " x " + b;
    }
}

static constexpr std::string sum_typehint_(const std::string a, const std::string b) {
    if (typehint::string_compare(a, "0") == 0) {
        return b;
    } else if (typehint::string_compare(b, "0") == 0) {
        return a;
    } else {
        return a + " + " + b;
    }
}

struct ZeroPolyTerm {
    TYPEHINT_PASSTHROUGH_CALL
    static constexpr bool is_poly_term = true;

    static constexpr ConstantValue total = 0;

    static constexpr auto typehint_type() { return "0"; }
};

template <IsDimension _Dim, int _power>
    requires(_power >= 0)
struct PolyTerm {
    TYPEHINT_PASSTHROUGH_CALL
    static constexpr bool is_poly_term = true;

    using Dim = _Dim;
    static constexpr Size power = _power;

    static constexpr ConstantValue total_() {
        ConstantValue result = 1;
        for (int i = 0; i < power; i++) {
            result *= Dim::value;
        }
        return result;
    }

    static constexpr ConstantValue total = total_();

    static constexpr auto typehint_type() {
        if constexpr (power == 1) {
            return Dim::typehint_type();
        } else {
            return Dim::typehint_type() + "^" + typehint::to_string(power);
        }
    }
};

struct EmptyProductTerm {
    TYPEHINT_PASSTHROUGH_CALL
    static constexpr bool is_product_term = true;

    static constexpr ConstantValue total = 1;

    static constexpr std::string typehint_type() { return "1"; }
};

template <IsPolyTerm _Outer, IsProductTerm _Inner = EmptyProductTerm>
struct ProductTerm {
    TYPEHINT_PASSTHROUGH_CALL
    static constexpr bool is_product_term = true;

    using Outer = _Outer;
    using Inner = _Inner;

    static constexpr ConstantValue total = Outer::total * Inner::total;

    // de-dupe and sort
    static constexpr auto normalized() {
        if constexpr (std::is_same_v<Inner, EmptyProductTerm>) {
            return ProductTerm<Outer>{};
        } else if constexpr (Outer::total == 0) {
            return ProductTerm<ZeroPolyTerm>{};
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
            return product_typehint_(Outer::typehint_type(), Inner::sorted_typehint_type_());
        }
    }

    static constexpr auto typehint_type() {
        using Normalized = decltype(normalized());
        return Normalized::sorted_typehint_type_();
    }
};

template <IsConstant _C, IsProductTerm _Product>
struct ConstProductTerm {
    TYPEHINT_PASSTHROUGH_CALL
    static constexpr bool is_const_product_term = true;

    using C = _C;
    using Product = _Product;

    using Total = Constant<C::value * Product::total, C::unit>;

    static constexpr auto typehint_type() { return product_typehint_(C::typehint_type(), Product::typehint_type()); }
};

struct EmptyComplexity {
    TYPEHINT_PASSTHROUGH_CALL
    static constexpr bool is_complexity = true;

    using Total = Constant<0, "">;
    static constexpr Total total{};

    static constexpr auto normalized() { return EmptyComplexity{}; }

    static constexpr std::string typehint_type() { return "0"; }
};

// sum of products
template <IsConstProductTerm _Outer, IsComplexity _Inner>
struct Complexity {
    TYPEHINT_PASSTHROUGH_CALL
    static constexpr bool is_complexity = true;

    using Outer = _Outer;
    using Inner = _Inner;

    using Total = AddConstants<typename Outer::Total, typename Inner::Total>;
    static constexpr Total total{};

    // de-dupe and sort
    static constexpr auto normalized() {
        using InnerNormalized = decltype(Inner::normalized());

        if constexpr (std::is_same_v<InnerNormalized, EmptyComplexity>) {
            return Complexity<Outer, InnerNormalized>{};
        } else {
            static_assert(CanAddConstants<typename Outer::C, typename InnerNormalized::Outer::C>);

            if constexpr (typehint::string_compare(Outer::Product::typehint_type(),
                                                   InnerNormalized::Outer::Product::typehint_type()) == 0) {
                using NewConstant = AddConstants<typename Outer::C, typename InnerNormalized::Outer::C>;
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
            // return Outer::typehint_type() + " + " + Inner::sorted_typehint_type_();
            return sum_typehint_(Outer::typehint_type(), Inner::sorted_typehint_type_());
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

template <IsComplexity Cx, IsConstant Bound>
    requires CanAddConstants<typename Cx::Total, Bound>
struct UpperBoundCheck {
    static constexpr auto typehint_type() {
        if constexpr (Cx::Total::value <= Bound::value) {
            return "OK: " + Cx::Total::typehint_type() + " <= " + Bound::typehint_type();
        } else {
            return "ERROR: " + Cx::Total::typehint_type() + " > " + Bound::typehint_type();
        }
    }
};

template <IsComplexity Cx, IsConstant Bound>
    requires CanAddConstants<typename Cx::Total, Bound>
constexpr auto assert_upper_bound(Cx cx, Bound bound) {
    static_assert(Cx::Total::value <= Bound::value, "Complexity exceeds bound");
    return UpperBoundCheck<Cx, Bound>{};
}

template <IsComplexity Cx, IsConstant Bound>
    requires CanAddConstants<typename Cx::Total, Bound>
constexpr auto check_upper_bound(Cx cx, Bound bound) {
    return UpperBoundCheck<Cx, Bound>{};
}

}  // namespace vgrad::cx

#endif  // VGRAD_COMPLEXITY_H_