#include "tensor.h"
#include <sstream>

namespace nt {

// The function implementation is a bit messy and surely can be optimised, but
// it is not a priority right now.
std::string Tensor::toString(size_t dim, size_t offset) const {
    std::stringstream out;
    if (dim == _shape.dim() - 1) {
        const auto data = span().subspan(offset);
        size_t lastDimSize = _shape[_shape.dim() - 1];

        out << '[' << data[0];
        for (size_t i = 1; i < lastDimSize; i++) {
            out << ' ' << data[i];
        }
        out << "]";
        return out.str();
    }

    const auto strides = getStrides();
    out << '[' << toString(dim + 1, offset);
    for (size_t i = 1; i < _shape[dim]; i++) {
        const size_t localOffset = offset + i * strides[dim];
        out << '\n' << toString(dim + 1, localOffset);
    }
    out << "]";
    return out.str();
}

Tensor &Tensor::reshape(Shape shape) {
    if (std::any_of(shape.begin(), shape.end(),
                    [](size_t dim) { return dim == 0; })) {
        throw std::runtime_error("Invalid new shape");
    }
    if (shape.numel() != _shape.numel()) {
        throw std::runtime_error(
            "Tensor reshape operation expects the same number of elements "
            "as the current shape.");
    }

    _shape = shape;
    return *this;
}

Tensor &Tensor::reshape(std::vector<int64_t> shape) {
    std::optional<int64_t> inferedIdx{};
    int64_t numel = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] == -1) {
            if (!inferedIdx.has_value()) {
                inferedIdx = i;
            } else {
                throw std::runtime_error("Only one dimension can be inferred.");
            }
        } else if (shape[i] <= 0) {
            throw std::runtime_error("Invalid new shape");
        } else {
            numel *= shape[i];
        }
    }
    if (inferedIdx.has_value()) {
        if ((_shape.numel() % numel) != 0U) {
            throw std::runtime_error(
                "New shape is invalid for current shape size.");
        }
        const auto inferedValue = _shape.numel() / numel;
        shape[inferedIdx.value()] = static_cast<int64_t>(inferedValue);
    }
    const nt::Shape newShape(shape.begin(), shape.end());
    return reshape(newShape);
}

Tensor Tensor::arrange(size_t n) {
    Tensor output(Shape{n});
    auto outputArr = output.span();
    for (size_t i = 0; i < n; i++) {
        outputArr[i] = static_cast<float>(i);
    }
    return output;
}

} // namespace nt
