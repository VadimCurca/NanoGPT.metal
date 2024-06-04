#pragma once

#include "tensor.h"
#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

namespace nt {

inline Tensor ntTensorfromTorchTensor(const torch::Tensor &torchTensor) {
    torch::IntArrayRef shape = torchTensor.sizes();
    const nt::Shape ntShape(shape.begin(), shape.end());
    return {ntShape, static_cast<float *>(torchTensor.data_ptr())};
}

inline torch::Tensor torchTensorFromNtTensor(const Tensor &ntTensor) {
    const auto shape = ntTensor.getShape();
    const std::vector<int64_t> shapeI64(shape.begin(), shape.end());
    const torch::Tensor torchTensor =
        at::empty(shapeI64, at::TensorOptions(at::ScalarType::Float));
    std::memcpy(torchTensor.data_ptr(),
                static_cast<void const *>(ntTensor.begin()),
                shape.numel() * sizeof(float));
    return torchTensor;
}

const double defaultRtol = 1e-05;
const double defaultAtol = 1e-08;

inline bool allclose(const torch::Tensor &torchTensor,
                     const Tensor &netGenTensor, double rtol = defaultRtol,
                     double atol = defaultAtol) {
    return torch::allclose(torchTensor, torchTensorFromNtTensor(netGenTensor),
                           rtol, atol);
}

template <class T>
inline torch::Tensor generateUniformTorchTensor(const std::vector<T> &shape,
                                                double from = 0.0,
                                                double to = 1.0) {
    const std::vector<int64_t> shapeI64(shape.begin(), shape.end());
    return at::empty(shapeI64, at::TensorOptions(at::ScalarType::Float))
        .uniform_(from, to);
}

} // namespace nt
