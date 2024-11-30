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
        return coeff * pow(x, degree) + next.forward(x);
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

// A sin(Bx + C)
template <Number DType>
class SinusoidalModel {
   public:
    auto forward(const auto& x) const { return A * sin(B * x + C); }

    auto params() { return make_params(A, B, C); }

    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const SinusoidalModel<T>& model) {
        os << model.A << "sin(" << model.B << "x + " << model.C << ")";
        return os;
    }

   private:
    Tensor<ScalarShape, DType> A = randn<DType, ScalarShape>();
    Tensor<ScalarShape, DType> B = randn<DType, ScalarShape>();
    Tensor<ScalarShape, DType> C = randn<DType, ScalarShape>();
};

auto loss(const auto& model, const auto& x, const auto& y) {
    PROFILE_SCOPE("loss");
    auto y_hat = model.forward(x);
    auto loss = sum(pow(y_hat - y, 2));
    return loss;
}

int main() {
    using Dim = Dimension<100>;
    using DType = float;

    auto x = arange<DType, Dim>() / 10;
    auto noise = randn_like(x);

    // auto y = 5 * pow(x, 2) + 3 * pow(x, 1) + 2 + 0.1 * noise;
    auto y = sin(x);

    // PolynomialModel<DType, 2> model;
    SinusoidalModel<DType> model;

    const float lr = 0.1;
    const int epochs = 2'000;

    optim::Adam optimizer{lr, model.params()};

    for (int epoch = 0; epoch < epochs; epoch++) {
        PROFILE_SCOPE("epoch");

        auto l = loss(model, x, y);
        optimizer.step(l);

        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << "\tLoss: " << l.value() << std::endl;
        }
    }

    std::cout << "Model: " << model << std::endl;
}
