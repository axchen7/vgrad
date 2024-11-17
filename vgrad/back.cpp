#include <iostream>

#include "backward.h"
#include "create_tensor.h"
#include "ops.h"
#include "shape.h"
#include "tensor.h"

using namespace vgrad;

template <IsTensor T>
void print_mat(T mat) {
    typename T::Shape shape;
    static_assert(shape.rank == 2);
    for (Size i = 0; i < shape.template at<0>().value; i++) {
        for (Size j = 0; j < shape.template at<1>().value; j++) {
            std::cout << mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

auto f(auto x, auto y) {
    // f(x) = (x-2)^2 + (y-3)^2
    return pow(x - 2, 2) + pow(y - 3, 2);
}

int main() {
    // auto x = Tensor<ScalarShape, float>{3};
    // auto y = 2 * pow(x, 2) + 9 * pow(x, 7);
    // auto dy_dx = backward(y, x);
    // std::cout << "y: " << y.value() << std::endl;
    // std::cout << "dy/dx: " << dy_dx.value() << std::endl;

    auto x = randn<float, MakeShape<Dimension<2>, Dimension<2>>>();
    auto y = transpose<0, 1>(x);
    print_mat(x);
    print_mat(y);
}