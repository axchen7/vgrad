#include <iostream>

#include "ops.h"
#include "shape.h"
#include "tensor.h"

int main() {
    using namespace vgrad;

    Dimension<2> img_size;
    auto img_shape = make_shape(img_size, img_size);

    std::array<std::array<float, 2>, 2> img_data{{{1, 2}, {3, 4}}};

    Tensor<decltype(img_shape), float> img1{img_data};
    Tensor<decltype(img_shape), float> img2{img_data};

    auto sum = img1 + img2;

    // for (size_t i = 0; i < sum.data_->size(); ++i) {
    //     std::cout << (*sum.data_)[i] << " ";
    // }
    // std::cout << std::endl;

    sum = vgrad::log(sum);
    sum = vgrad::exp(sum);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << sum[i][j] << " ";
        }
        std::cout << std::endl;
    }
}