//#include <iostream>
//
//#include "create_tensor.h"
//#include "ops.h"
//#include "shape.h"
//#include "tensor.h"
//
//using namespace vgrad;
//
//template <IsTensor T>
//void print_mat(T mat) {
//    typename T::Shape shape;
//    static_assert(shape.rank == 2);
//    for (Size i = 0; i < shape.template at<0>().value; i++) {
//        for (Size j = 0; j < shape.template at<1>().value; j++) {
//            std::cout << mat[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//}
//
//int main() {
//    // ----- STUFF 1 -----
//
//    using ImgSize = Dimension<2>;
//
//    auto img1 = vgrad::randn<float, MakeShape<ImgSize, ImgSize>>();
//    auto img2 = vgrad::eye<float, ImgSize>();
//
//    auto sum = img1 + img2;
//
//    for (int i = 0; i < 1000; i++) {
//        sum = sum + vgrad::randn<float, MakeShape<ImgSize, ImgSize>>();
//    }
//
//    sum = sum / (1000 * vgrad::ones_like(sum));
//
//    sum = vgrad::log(sum);
//    sum = vgrad::exp(sum);
//
//    print_mat(sum);

    // ----- STUFF 2 -----

    // using D1 = Dimension<2>;
    // using D2 = Dimension<3>;

    // std::array<std::array<float, 3>, 2> data = {{{1, 2, 3}, {4, 5, 7}}};

    // Tensor<MakeShape<D1, D2>, float> mat1{data};
    // print_mat(mat1);

    // auto mat2 = transpose<0, 1>(mat1);
    // print_mat(mat2);

    // auto mat3 = mean(mat1);

    // auto mat4 = unsqueeze<1>(mat3);
    // print_mat(mat4);

    // auto mat5 = repeat<1, D2>(mat4);
    // print_mat(mat5);

    // ----- STUFF 3 -----

    // using D1 = Dimension<2>;
    // using D2 = Dimension<3>;

    // std::array<std::array<float, 3>, 2> data = {{{1, 2, 3}, {4, 5, 7}}};

    // auto mat1 = Tensor<MakeShape<D1, D2>, float>{data};
    // auto mat2 = transpose<0, 1>(mat1);
    // auto mat3 = matmul(mat1, mat2);

    // std::cout << "mat1:" << std::endl;
    // print_mat(mat1);

    // std::cout << "mat2:" << std::endl;
    // print_mat(mat2);

    // std::cout << "mat3:" << std::endl;
    // print_mat(mat3);

    // ----- STUFF 4 -----

    // using BigDim = Dimension<1000>;
    // auto mat1 = randn<float, MakeShape<BigDim, BigDim>>();
    // auto mat2 = randn<float, MakeShape<BigDim, BigDim>>();
    // auto mat3 = matmul(mat1, mat2);
    // std::cout << sum(sum(mat3)).value() << std::endl;
}