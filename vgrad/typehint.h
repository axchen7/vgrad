#ifndef TYPEHINT_TYPEHINT_H_
#define TYPEHINT_TYPEHINT_H_

#include <charconv>
#include <string>

#define TYPEHINT_PRINT_TYPE(label, val)                                                               \
    typehint::static_print<label, typehint::to_string_literal<decltype(val)::typehint_type().size()>( \
                                      decltype(val)::typehint_type())>()

namespace typehint {

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
constexpr std::string to_string(int v) {
    // +1 for \0, +1 for minus sign
    constexpr size_t bufsize = std::numeric_limits<int>::digits10 + 2;
    char buf[bufsize];
    const auto res = std::to_chars(buf, buf + bufsize, v);
    return std::string(buf, res.ptr);
}

// inspired by: https://stackoverflow.com/a/58834326
template <auto Label, StringLiteral str>
constexpr void static_print() {
    auto unused = Label;
};

}  // namespace typehint

#endif  // TYPEHINT_TYPEHINT_H_