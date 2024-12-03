#define PRINT_PROFILE_ON_EXIT

#include "vgrad.h"

using namespace vgrad;

template <Size N>
auto dumb_transpose(const auto& mat) {
    PROFILE_SCOPE("dumb_transpose");
    auto res = zeros_like(mat);
    auto& mat_data = mat.flat_view();
    auto& res_data = res._flat_data();
#pragma omp parallel for
    for (Size i = 0; i < N; i++) {
        for (Size j = 0; j < N; j++) {
            res_data[j * N + i] = mat_data[i * N + j];
        }
    }
    return res;
}

template <Size N>
void measure() {
    using Dim = Dimension<N>;
    auto mat = zeros<float, MakeShape<Dim, Dim>>();
    transpose<0, 1>(mat);
    dumb_transpose<N>(mat);
}

int main() {
    measure<1000>();
    measure<2000>();
    measure<3000>();
    measure<4000>();
    measure<5000>();
    measure<6000>();
    measure<7000>();
    measure<8000>();
    measure<9000>();
    measure<10000>();
}