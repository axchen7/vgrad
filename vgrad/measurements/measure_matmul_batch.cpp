#define PRINT_PROFILE_ON_EXIT

#include "vgrad.h"

using namespace vgrad;

template <Size B>
void measure() {
    using Dim = Dimension<16>;
    using Batch = Dimension<B>;
    auto mat = randn<float, MakeShape<Batch, Dim, Dim>>();
    matmul(mat, mat);
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