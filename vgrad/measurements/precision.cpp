#include "vgrad.h"

using namespace vgrad;

int main() {
    using D = Dimension<100>;
    auto x = import_vgtensor<float, MakeShape<D, D>>("data/rand_matrix.vgtensor");

    for (int i = 0; i < 10; i++) {
        x = (matmul(x, x) / std::sqrt(D::value)).detach();
        auto m = mean(mean(x));
        auto var = mean(mean(pow(x - m, 2)));
        std::cout << "Variance: " << var.value() << std::endl;
    }
}