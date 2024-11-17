//#include <iostream>
//
//#include "shape.h"
//#include "tensor.h"
//
//int main() {
//    using namespace vgrad;
//
//    using D1 = Dimension<1>;
//    using D2 = Dimension<2>;
//    using D3 = Dimension<3>;
//
//    using Shape1 = Shape<D1, ScalarShape>;
//    using Shape2 = Shape<D2, Shape1>;
//
//    auto x = Shape1::at<0>();
//    auto y = Shape2::at<1>();
//
//    using Shape3 = Shape2::Remove<0>;
//    using Shape4 = Shape2::Remove<1>::Remove<0>();
//    using Shape5 = Shape2::Insert<2, D3>;
//
//    using Shape6 = MakeShape<D1, D2, D3>;
//    using Shape7 = Shape6::Transpose<0, 1>;
//
//    auto DTest = Shape7::at<-1>();
//
//    using Shape8 = MakeShape<D1, D2>;
//
//    Tensor<Shape8, int> tensor1;
//    auto tensor2 = tensor1;
//
//    std::cout << (*tensor1.data_)[0] << std::endl;
//    std::cout << (*tensor1.data_)[1] << std::endl;
//}