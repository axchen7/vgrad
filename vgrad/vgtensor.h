#ifndef VGRAD_VGTENSOR_H_
#define VGRAD_VGTENSOR_H_

#include <fstream>

#include "tensor.h"

namespace vgrad {

// import a tensor saved via numpy .tobytes()
template <typename DType, IsShape Shape>
auto import_vgtensor(std::string filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file");
    }

    if (file.tellg() != sizeof(DType) * Shape::flat_size) {
        throw std::runtime_error("File size does not match expected tensor size");
    }

    file.seekg(0, std::ios::beg);

    Tensor<Shape, DType> result;

    for (Size i = 0; i < Shape::flat_size; ++i) {
        DType value;
        file.read(reinterpret_cast<char*>(&value), sizeof(DType));
        if (!file) {
            throw std::runtime_error("Failed to read data from file");
        }
        result._init_entry(i, value);
    }

    return result;
}

}  // namespace vgrad

#endif  // VGRAD_VGTENSOR_H_