#include <iostream>

#include "backward.h"
#include "create_tensor.h"
#include "ops.h"
#include "shape.h"
#include "tensor.h"

using namespace vgrad;

template <IsTensor T>
    requires(T::Shape::rank == 2)
void print_mat(T mat) {
    typename T::Shape shape;
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

    // auto x = randn<float, MakeShape<Dimension<2>, Dimension<2>>>();
    // auto y = unsqueeze<0>(x);
    // auto z = squeeze<0>(y);
    // auto l = sum<0>(sum<0>(z));

    // auto f = l.value();

    // print_mat(x);
    // print_mat(z);
    // std::cout << l.value() << std::endl;

    // auto dl_dz = backward(l, z);
    // auto dl_dx = backward(l, x);

    // print_mat(dl_dz);
    // print_mat(dl_dx);

    // auto x = randn<double, MakeShape<Dimension<2>, Dimension<2>>>();
    // auto y = prod(x);
    // auto z = prod(y);

    // auto dz_dx = backward(z, x);
    // print_mat(dz_dx);

    // auto x = Tensor<ScalarShape, float>{1};
    // auto y = repeat<0, Dimension<3>>(unsqueeze<0>(x));
    // auto z = repeat<0, Dimension<3>>(unsqueeze<0>(y));
    // auto l = sum(sum(z));

    // std::cout << l.value() << std::endl;

    // auto dl_dx = backward(l, x);
    // auto dl_dy = backward(l, y);

    // std::cout << dl_dx.value() << std::endl;
    // std::cout << dl_dy[0] << std::endl;

    using D = Dimension<1000>;

    auto x = ones<float, MakeShape<D, D>>();
    auto y = eye<float, D>();

    // auto z = x * y;
    auto z = matmul(x, y);

    auto l = sum(sum(z));

    std::cout << l.value() << std::endl;

    auto dl_dx = backward(l, x);

    // print_mat(dl_dx);
    std::cout << sum(sum(dl_dx)).value() << std::endl;
}