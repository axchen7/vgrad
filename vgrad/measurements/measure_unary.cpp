#define PRINT_PROFILE_ON_EXIT

#include "vgrad.h"

using namespace vgrad;

template <Size N>
void measure() {
    using Dim = Dimension<N>;
    auto vec = randn<float, MakeShape<Dim>>();
    vec + 1;
}

int main() {
    measure<10'000'000>();
    measure<20'000'000>();
    measure<30'000'000>();
    measure<40'000'000>();
    measure<50'000'000>();
    measure<60'000'000>();
    measure<70'000'000>();
    measure<80'000'000>();
    measure<90'000'000>();
    measure<100'000'000>();
}