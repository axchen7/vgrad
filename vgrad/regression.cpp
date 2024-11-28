#include <tuple>

#include "create_tensor.h"
#include "module.h"
#include "ops.h"
#include "optimizers.h"
#include "vgtensor.h"

using namespace vgrad;

template <Number DType>
class ScalarModel {
   public:
    auto forward(const auto& x) const { return coeff; }
    auto params() { return make_params(coeff); }

    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const ScalarModel<T>& model) {
        os << model.coeff;
        return os;
    }

   private:
    Tensor<ScalarShape, DType> coeff = randn<DType, ScalarShape>();
};

template <Number DType, int degree>
class PolynomialModel {
   public:
    template <IsTensor X>
    auto forward(const X& x) const {
        using Dim = typename X::Shape::template At<0>;
        auto range = arange<DType, Dim>();
        return coeff * pow(range, degree) + next.forward(x);
    }

    auto params() { return make_params(coeff, next); }

    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const PolynomialModel<T, degree>& model) {
        os << model.coeff << "x^" << degree << " + " << model.next;
        return os;
    }

   private:
    Tensor<ScalarShape, DType> coeff = randn<DType, ScalarShape>();

    using NextModel = std::conditional_t<degree - 1 == 0, ScalarModel<DType>, PolynomialModel<DType, degree - 1>>;
    NextModel next;
};

template <Number DType>
using LinearModel = PolynomialModel<DType, 1>;

auto loss(const auto& model, const auto& x) {
    PROFILE_SCOPE("loss");
    auto y = model.forward(x);
    auto loss = sum(pow(x - y, 2));
    return loss;
}

int main() {
    using Dim = Dimension<100>;
    using DType = float;

    auto r = arange<DType, Dim>();
    auto noise = randn_like(r);

    auto x = 5 * pow(r, 2) + 3 * pow(r, 1) + 2 + 0.1 * noise;

    PolynomialModel<DType, 2> model;

    const float lr = 0.1;
    const int epochs = 10'000;

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
