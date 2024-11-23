#include <tuple>

#include "create_tensor.h"
#include "module.h"
#include "ops.h"
#include "optimizers.h"
#include "vgtensor.h"

using namespace vgrad;

template <IsDimension In, IsDimension Out, Number DType, IsDimension Inner>
class Model {
   public:
    auto forward(const auto& x) const {
        auto o1 = layer1.forward(x);
        auto o2 = relu(o1);
        auto o3 = layer2.forward(o2);
        return o3;
    }

    auto params() { return make_params(layer1, layer2); }

   private:
    Linear<In, Inner, DType> layer1;
    Linear<Inner, Out, DType> layer2;
};

template <IsTensor Out, IsTensor Labels>
auto compute_accuracy(const Out& out, const Labels& labels) {
    auto predictions = argmax<-1, Out, int32_t>(out);
    float matches = sum(predictions == labels).value();
    return matches / Labels::Shape::flat_size;
}

int main() {
    using Inner = Dimension<16>;

    using TrainBatch = Dimension<10000>;
    using TestBatch = Dimension<500>;

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
    const int epochs = 400;

    optim::Adam optimizer{lr, model.params()};

    for (int epoch = 0; epoch < epochs; epoch++) {
        auto train_out = model.forward(train_flat);
        auto train_loss = cross_entropy(train_out, train_labels);
        optimizer.step(train_loss);

        auto test_out = model.forward(test_flat);
        auto test_loss = cross_entropy(test_out, test_labels);

        auto test_acc = compute_accuracy(test_out, test_labels);

        std::cout << "Epoch: " << epoch << "\ttrain loss: " << train_loss.value()
                  << "\ttest loss: " << test_loss.value() << "\ttest acc: " << test_acc << std::endl;
    }
}
