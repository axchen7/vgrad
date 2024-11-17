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
    auto x = Tensor<ScalarShape, float>{0.0};
    auto y = Tensor<ScalarShape, float>{0.0};

    int epochs = 100;
    double lr = 0.1;

    for (int i = 0; i < epochs; i++) {
        auto z = f(x, y);
        auto dz_dx = backward(z, x);
        auto dz_dy = backward(z, y);
        x = (x - lr * dz_dx).detach();
        y = (y - lr * dz_dy).detach();
    }

    std::cout << "x: " << x.value() << std::endl;
    std::cout << "y: " << y.value() << std::endl;
}