#define PRINT_PROFILE_ON_EXIT

#include "vgrad.h"

using namespace vgrad;

template <Size N>
void measure() {
    using Dim = Dimension<N>;
    auto mat = zeros<float, MakeShape<Dim, Dim>>();
    transpose<0, 1>(mat);
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