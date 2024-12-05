#include "vgrad.h"

using namespace vgrad;
using namespace cx;

int main() {
    using D1 = Dimension<100>;       // 🔍 [100]
    using D2 = Dimension<200, "N">;  // 🔍 [N]

    using Shape1 = MakeShape<D1, D2>;      // 🔍 [100 x N]
    using Shape2 = MakeShape<D2, D2, D1>;  // 🔍 [N x N x 100]
    using Shape3 = ScalarShape;            // 🔍 [scalar]

    auto m1 = eye<float, D1>();        // 🔍 [100 x 100, float]
    auto m2 = full<float, Shape1>(3);  // 🔍 [100 x N, float]

    // symbolic memory complexity
    m1.mem_complexity();  // 🔍 [4 B x 100^2]
    // total memory complexity
    m1.mem_complexity.total();  // 🔍 [40000 B]

    // impose 2GB memory bound
    auto bound = Constant<2'000'000, "B">{};
    // pretty-print whether the memory complexity is within the bound
    check_upper_bound(m1.mem_complexity, bound);  // 🔍 [OK: 40000 B <= 2000000 B]
    // static_assert if the memory complexity exceeds the bound
    assert_upper_bound(m1.mem_complexity, bound);
}