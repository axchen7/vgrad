#define PRINT_PROFILE_ON_EXIT

#include "vgrad.h"

using namespace vgrad;

template <Size N>
auto dumb_matmul(const auto& mat) {
    PROFILE_SCOPE("dumb_matmul");
    auto res = zeros_like(mat);
    auto& mat_data = mat.flat_view();
    auto& res_data = res._flat_data();
#pragma omp parallel for
    for (Size i = 0; i < N; i++) {
        for (Size j = 0; j < N; j++) {
            for (Size k = 0; k < N; k++) {
                res_data[i * N + j] += mat_data[i * N + k] * mat_data[k * N + j];
            }
        }
    }
    return res;
}

template <Size N>
void measure() {
    using Dim = Dimension<N>;
    auto mat = randn<float, MakeShape<Dim, Dim>>();
    matmul(mat, mat);
    dumb_matmul<N>(mat);
}

int main() {
    measure<100>();
    measure<200>();
    measure<300>();
    measure<400>();
    measure<500>();
    measure<600>();
    measure<700>();
    measure<800>();
    measure<900>();
    measure<1000>();
}