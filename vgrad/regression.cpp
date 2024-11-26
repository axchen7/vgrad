#include <tuple>

#include "create_tensor.h"
#include "module.h"
#include "ops.h"
#include "optimizers.h"
#include "vgtensor.h"

using namespace vgrad;

template <Number DType>
class LinearModel {
   public:
    template <IsTensor X>
    auto forward(const X& x) const {
        PROFILE_SCOPE("LinearModel::forward");
        using Dim = typename X::Shape::template At<0>;
        auto range = arange<DType, Dim>();
        return a * range + b;
    }

    auto params() { return make_params(a, b); }

    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const LinearModel<T>& model) {
        os << model.a.value() << "x + " << model.b.value();
        return os;
    }

   private:
    Tensor<ScalarShape, DType> a = randn<DType, ScalarShape>();
    Tensor<ScalarShape, DType> b = randn<DType, ScalarShape>();
};

auto loss(const auto& model, const auto& x) {
    PROFILE_SCOPE("loss");
    auto y = model.forward(x);
    auto loss = sum(pow(x - y, 2));
    return loss;
}

int main() {
    using Dim = Dimension<100>;
    using DType = float;

    // auto x = zeros<DType, MakeShape<Dim>>();
    auto x = arange<DType, Dim>() + randn<DType, MakeShape<Dim>>() * 0.1;

    LinearModel<DType> model;

    const float lr = 0.1;
    const int epochs = 1000;

    optim::Adam optimizer{lr, model.params()};

    for (int epoch = 0; epoch < epochs; epoch++) {
        PROFILE_SCOPE("epoch");

        auto l = loss(model, x);
        optimizer.step(l);

        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << "\tLoss: " << l.value() << std::endl;
        }
    }

    std::cout << "Model: " << model << std::endl;
}
