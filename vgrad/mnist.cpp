#include <tuple>

#include "create_tensor.h"
#include "ops.h"
#include "optimizers.h"
#include "vgtensor.h"

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
        auto o1 = layer1.forward(x);
        auto o2 = relu(o1);
        auto o3 = layer2.forward(o2);
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
    using Inner = Dimension<128>;

    using TrainBatch = Dimension<10'000>;
    using TestBatch = Dimension<1'000>;

    using ImgSize = Dimension<28>;
    using FlatSize = Dimension<ImgSize::value * ImgSize::value>;

    using Classes = Dimension<10>;

    auto train_imgs = import_vgtensor<float, MakeShape<TrainBatch, ImgSize, ImgSize>>("../torch/train_images.vgtensor");
    auto test_imgs = import_vgtensor<float, MakeShape<TestBatch, ImgSize, ImgSize>>("../torch/test_images.vgtensor");

    auto train_labels = import_vgtensor<int32_t, MakeShape<TrainBatch>>("../torch/train_labels.vgtensor");
    auto test_labels = import_vgtensor<int32_t, MakeShape<TestBatch>>("../torch/test_labels.vgtensor");

    auto train_flat = reshape<MakeShape<TrainBatch, FlatSize>>(train_imgs);
    auto test_flat = reshape<MakeShape<TestBatch, FlatSize>>(test_imgs);

    Model<FlatSize, Classes, float, Inner> model;

    const float lr = 0.1;
    const int epochs = 100;

    optim::Adam optimizer{lr, model.params()};

    for (int i = 0; i < epochs; i++) {
        auto out = model.forward(train_flat);
        auto loss = cross_entropy(out, train_labels);
        optimizer.step(loss);

        std::cout << "Epoch " << i << " loss: " << loss.value() << std::endl;
    }
}