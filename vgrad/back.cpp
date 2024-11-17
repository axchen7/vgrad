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

auto f(auto x) {
    // f(x) = (x-2)^2
    return pow((x - 2), 2);
}

int main() {
    auto x = Tensor<ScalarShape, float>{0.0};

    int epochs = 100;
    double lr = 0.1;

    for (int i = 0; i < epochs; i++) {
        auto y = f(x);
        auto dy_dx = backward(y, x);
        x = x - lr * dy_dx;
    }

    std::cout << x.value() << std::endl;
}