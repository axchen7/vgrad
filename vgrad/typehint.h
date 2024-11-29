#ifdef __APPLE__
#ifndef TYPEHINT_TYPEHINT_H_
#define TYPEHINT_TYPEHINT_H_

#include <charconv>
#include <string>

#define TYPEHINT_PRINT_VAL_TYPE(label, rval)                                                                           \
    rval;                                                                                                              \
    typehint::static_print<label,                                                                                      \
                           typehint::to_string_literal<decltype(typehint::passthrough(rval))::typehint_type().size()>( \
                               decltype(typehint::passthrough(rval))::typehint_type())>()

#define TYPEHINT_PRINT_USING_TYPE(label, Type) \
    Type;                                      \
    typehint::static_print<label, typehint::to_string_literal<Type::typehint_type().size()>(Type::typehint_type())>()

// to get typehint via () without assigning to a variable
#define TYPEHINT_PASSTHROUGH_CALL \
    constexpr auto operator()() const { return *this; }

template <typename _T>
struct TYPEHINT_TYPE_PASSTHROUGH {
    using T = _T;
};

namespace typehint {

auto passthrough(auto x) { return x; }

// taken from: https://ctrpeach.io/posts/cpp20-string-literal-template-parameters/
template <std::size_t N>
struct StringLiteral {
    constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }
    char value[N] = {};
};

template <std::size_t Size>
constexpr auto to_string_literal(const std::string str) {
    constexpr auto arr_len = Size + 1;
    char arr[arr_len] = {};
    std::copy_n(str.c_str(), arr_len, arr);
    return StringLiteral<arr_len>{arr};
}

// taken from: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3438r0.pdf
constexpr std::string to_string(long long v) {
    // +1 for \0, +1 for minus sign
    constexpr size_t bufsize = std::numeric_limits<long long>::digits10 + 2;
    char buf[bufsize];
    const auto res = std::to_chars(buf, buf + bufsize, v);
    return std::string(buf, res.ptr);
}

// -1, 0, 1 for less, equal, greater
constexpr int string_compare(const std::string a, const std::string b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

// inspired by: https://stackoverflow.com/a/58834326
template <StringLiteral Label, StringLiteral str>
constexpr void static_print() {
    auto unused = Label;
};

}  // namespace typehint

#endif  // TYPEHINT_TYPEHINT_H_
#endif  // __APPLE__