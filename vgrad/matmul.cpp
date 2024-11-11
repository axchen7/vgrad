#include <iostream>

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

int main() {
    using Dim = Dimension<100>;

    auto A = vgrad::eye<float, Dim>();
    auto B = A + A + 5;

    auto C = vgrad::matmul(B, B);  // shape is 3x3

    // print_mat(C);

    auto top_left = C[0][0];

    std::cout << top_left << std::endl;
}