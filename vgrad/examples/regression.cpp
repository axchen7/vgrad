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
    auto forward(const auto& x) const { return coeff * pow(x, degree) + next.forward(x); }

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
    SinusoidalModel() {}
    SinusoidalModel(DType initial_freq) : B{initial_freq} {}

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

template <Number DType>
class DoubleNoiseModel {
   public:
    DoubleNoiseModel(DType initial_freq) : noise1_model{initial_freq}, noise2_model{initial_freq} {}

    auto forward(const auto& x, const auto& y) const {
        auto y_hat1 = baseline_model.forward(x) + noise1_model.forward(x);
        auto y_hat2 = baseline_model.forward(x) + noise2_model.forward(x);

        auto diff1 = pow(y_hat1 - y, 2);
        auto diff2 = pow(y_hat2 - y, 2);

        auto y_hat = where(diff1 < diff2, y_hat1, y_hat2);
        return y_hat;
    }

    auto params() { return make_params(baseline_model, noise1_model, noise2_model); }

    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const DoubleNoiseModel<T>& model) {
        os << "(" << model.baseline_model << ") + [" << model.noise1_model << " | " << model.noise2_model << "]";
        return os;
    }

   private:
    LinearModel<DType> baseline_model;
    SinusoidalModel<DType> noise1_model;
    SinusoidalModel<DType> noise2_model;
};

auto loss(const auto& y, const auto& y_hat) {
    PROFILE_SCOPE("loss");
    auto loss = sum(pow(y_hat - y, 2));
    return loss;
}

void shift_and_add(auto& tensor, auto val) {
    PROFILE_SCOPE("shift_and_add");
    auto& data = tensor._flat_data();
    for (int i = 0; i < data.size() - 1; i++) {
        data[i] = data[i + 1];
    }
    data[data.size() - 1] = val;
}

// return false on EOF
template <typename DType>
bool read_csv_line(auto& x, auto& y) {
    std::string line;
    if (!std::getline(std::cin, line)) {
        return false;
    }

    std::stringstream ss{line};
    std::string token;

    DType x_val;
    DType y_val;

    std::getline(ss, token, ',');
    x_val = std::stof(token);

    std::getline(ss, token, ',');
    y_val = std::stof(token);

    if (ss >> token) {
        throw std::runtime_error("Extra values in line");
    }

    shift_and_add(x, x_val);
    shift_and_add(y, y_val);
    return true;
}

int main() {
    using Dim = Dimension<100>;
    using DType = float;

    auto x = zeros<DType, MakeShape<Dim>>();
    auto y = zeros<DType, MakeShape<Dim>>();
    size_t num_read = 0;

    DType initial_freq = 20;

    DoubleNoiseModel<DType> model{initial_freq};

    const float lr = 0.1;
    const int initial_epochs = 500;
    const int refine_epochs = 50;

    optim::Adam optimizer{lr, model.params()};

    // throw out first input line (possibly csv header)
    std::string _header;
    std::getline(std::cin, _header);

    // read data in a loop
    while (true) {
        PROFILE_SCOPE("read_data");

        if (!read_csv_line<DType>(x, y)) {
            break;
        }
        num_read++;

        int epochs = 0;

        if (num_read == Dim::value) {
            epochs = initial_epochs;
        } else if (num_read > Dim::value) {
            epochs = refine_epochs;
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            PROFILE_SCOPE("epoch");

            auto y_hat = model.forward(x, y);
            auto l = loss(y, y_hat);
            optimizer.step(l);

            if (epoch % 20 == 0) {
                std::cout << "Epoch " << epoch << "\tLoss: " << l.value() << std::endl;
            }
        }
    }

    // std::cout << "Model: f(x) = " << model << std::endl;
}
