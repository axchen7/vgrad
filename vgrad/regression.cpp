#include <tuple>

#include "create_tensor.h"
#include "module.h"
#include "ops.h"
#include "optimizers.h"
#include "vgtensor.h"

using namespace vgrad;

template <Number DType>
class Model {
   public:
    Tensor<ScalarShape, DType> a = randn<DType, ScalarShape>();
    Tensor<ScalarShape, DType> b = randn<DType, ScalarShape>();

    template <IsTensor X>
    auto forward(const X& x) const {
        PROFILE_SCOPE("Model::forward");

        using Dim = typename X::Shape::template At<0>;

        auto range = arange<DType, Dim>();
        auto error = x - (a * range + b);
        auto loss = sum(pow(error, 2));
        return loss;
    }

    auto params() { return make_params(a, b); }
};

int main() {
    using Dim = Dimension<100>;
    using DType = float;

    // auto x = zeros<DType, MakeShape<Dim>>();
    auto x = arange<DType, Dim>() + randn<DType, MakeShape<Dim>>() * 0.1;

    Model<DType> model;

    const float lr = 0.1;
    const int epochs = 1000;

    optim::Adam optimizer{lr, model.params()};

    for (int epoch = 0; epoch < epochs; epoch++) {
        PROFILE_SCOPE("epoch");

        auto loss = model.forward(x);
        optimizer.step(loss);

        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << "\tLoss: " << loss.value() << std::endl;
        }
    }

    std::cout << "a: " << model.a.value() << "\tb: " << model.b.value() << std::endl;
}
