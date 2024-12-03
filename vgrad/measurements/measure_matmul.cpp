#define PRINT_PROFILE_ON_EXIT

#include "vgrad.h"

using namespace vgrad;

template <Size N>
void measure() {
    using Dim = Dimension<N>;
    auto mat = randn<float, MakeShape<Dim, Dim>>();
    matmul(mat, mat);
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