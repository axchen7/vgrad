#include "vgrad.h"

using namespace vgrad;

const float epochs = 1000;
const float lr = 0.1;  // learning rate

auto f(const auto x) { return pow(x - 3, 2); }  // f(x) = (x - 3)^2

int main() {
    Tensor<ScalarShape, float> x{0};

    for (int i = 0; i < epochs; i++) {
        auto y = f(x);
        auto [dy_dx] = backward(y, x);
        x -= lr * dy_dx;
    }

    std::cout << "x: " << x << std::endl;
}