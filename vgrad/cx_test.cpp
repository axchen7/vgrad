#include "complexity.h"

using namespace vgrad;
using namespace vgrad::cx;

int main() {
    using Dim1 = Dimension<10, "D">;
    using Dim2 = Dimension<20, "E">;

    using T1 = PolyTerm<Dim1, 2>;  // typehint: [D^2]
    using T2 = PolyTerm<Dim2, 3>;  // typehint: [E^3]

    using Const1 = Constant<100, "ns">;                      // typehint: [100 ns]
    using ProdTerm1 = MakeProductTerm<T1, T2>;               // typehint: [D^2 x E^3]
    using CProdTerm1 = ConstProductTerm<Const1, ProdTerm1>;  // typehint: [100 ns x D^2 x E^3]

    using Const2 = Constant<200, "ns">;                      // typehint: [200 ns]
    using ProdTerm2 = MakeProductTerm<T1, T2>;               // typehint: [D^2 x E^3]
    using CProdTerm2 = ConstProductTerm<Const2, ProdTerm2>;  // typehint: [200 ns x D^2 x E^3]

    using Const3 = Constant<50, "ns">;                       // typehint: [50 ns]
    using ProdTerm3 = MakeProductTerm<T1>;                   // typehint: [D^2]
    using CProdTerm3 = ConstProductTerm<Const3, ProdTerm3>;  // typehint: [50 ns x D^2]

    using _RunningTime1 = MakeComplexity<CProdTerm1, CProdTerm2, CProdTerm3>;
    using RunningTime1 = _RunningTime1;  // typehint: [50 ns x D^2 + 300 ns x D^2 x E^3]

    using RunningTime2 = AddComplexities<RunningTime1, RunningTime1>;  // typehint: [100 ns x D^2 + 600 ns x D^2 x E^3]

    using Foo = RunningTime2::TotalValue;  // typehint: [480010000 ns]
}