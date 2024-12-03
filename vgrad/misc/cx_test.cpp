#include "complexity.h"
#include "create_tensor.h"
#include "ops.h"

using namespace vgrad;
using namespace vgrad::cx;

template <Size N>
void benchmark_matmul() {
    using Dim = Dimension<N>;
    auto mat = randn<float, MakeShape<Dim, Dim>>();
    auto x = matmul(mat, mat);
    x.mem_complexity.total();  // 🔍 [20120000 B] [160480000 B] [541080000 B] [1281920000 B] [2503000000 B] [4324320000 B] [6865880000 B] [10247680000 B] [14589720000 B] [20012000000 B]
                               // B] [6865880000 B] [10247680000 B] [14589720000 B] [20012000000 B]
}

int main() {
    benchmark_matmul<100>();
    benchmark_matmul<200>();
    benchmark_matmul<300>();
    benchmark_matmul<400>();
    benchmark_matmul<500>();
    benchmark_matmul<600>();
    benchmark_matmul<700>();
    benchmark_matmul<800>();
    benchmark_matmul<900>();
    benchmark_matmul<1000>();
}