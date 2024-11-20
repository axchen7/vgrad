#include "vgtensor.h"

#include <tuple>

#include "create_tensor.h"
#include "ops.h"
#include "optimizers.h"

using namespace vgrad;

template <IsDimension In, IsDimension Out, Number DType>
class Linear {
   public:
    auto forward(const auto x) const { return matmul(x, w) + b; }

    auto params() { return std::make_tuple(std::ref(w), std::ref(b)); }

   private:
    using WShape = MakeShape<In, Out>;
    using BShape = MakeShape<Out>;
    Tensor<WShape, DType> w = randn<DType, WShape>();
    Tensor<BShape, DType> b = randn<DType, BShape>();
};

template <IsDimension In, IsDimension Out, Number DType, IsDimension Inner>
class Model {
   public:
    auto forward(auto x) const {
        auto o1 = layer1.forward(x);   // typehint: [10000 x 128, float]
        auto o2 = relu(o1);            // typehint: [10000 x 128, float]
        auto o3 = layer2.forward(o2);  // typehint: [10000 x 10, float]
        return o3;
    }

    auto params() {
        auto [w1, b1] = layer1.params();
        auto [w2, b2] = layer2.params();
        return std::make_tuple(std::ref(w1), std::ref(b1), std::ref(w2), std::ref(b2));
    }

   private:
    Linear<In, Inner, DType> layer1;
    Linear<Inner, Out, DType> layer2;
};

int main() {
    using Inner = Dimension<128>;  // typehint: [128]

    using TrainBatch = Dimension<10'000>;  // typehint: [10000]
    using TestBatch = Dimension<1'000>;    // typehint: [1000]

    using ImgSize = Dimension<28>;                                // typehint: [28]
    using FlatSize = Dimension<ImgSize::value * ImgSize::value>;  // typehint: [784]

    using Classes = Dimension<10>;  // typehint: [10]

    auto train_imgs = import_vgtensor<float, MakeShape<TrainBatch, ImgSize, ImgSize>>("../torch/train_images.vgtensor");
    auto test_imgs = import_vgtensor<float, MakeShape<TestBatch, ImgSize, ImgSize>>("../torch/test_images.vgtensor");

    auto train_labels = import_vgtensor<int32_t, MakeShape<TrainBatch>>("../torch/train_labels.vgtensor");
    auto test_labels = import_vgtensor<int32_t, MakeShape<TestBatch>>("../torch/test_labels.vgtensor");

    auto train_flat = reshape<MakeShape<TrainBatch, FlatSize>>(train_imgs);  // typehint: [10000 x 784, float]
    auto test_flat = reshape<MakeShape<TestBatch, FlatSize>>(test_imgs);     // typehint: [1000 x 784, float]

    Model<FlatSize, Classes, float, Inner> model;

    const float lr = 0.01;
    const int epochs = 10;

    optim::Adam optimizer{lr, model.params()};

    for (int i = 0; i < epochs; i++) {
        auto out = model.forward(train_flat);          // typehint: [10000 x 10, float]
        auto loss = cross_entropy(out, train_labels);  // typehint: [scalar, float]
        optimizer.step(loss);

        std::cout << "Epoch " << i << " loss: " << loss.value() << std::endl;
    }
}