#include "utils.h"
#include "shape.h"
#include <cassert>
#include <stdexcept>

namespace nt {

Shape broadcastShapes(const Shape &shape1, const Shape &shape2) {
    if (shape1 == shape2) {
        return shape1;
    }

    Shape outputShape = shape2;
    ;
    size_t minDim = shape1.dim();

    if (shape1.dim() > shape2.dim()) {
        outputShape = shape1;
        minDim = shape2.dim();
    }

    for (size_t i = 0; i < minDim; i++) {
        auto dim1 = shape1[shape1.dim() - 1 - i];
        auto dim2 = shape2[shape2.dim() - 1 - i];
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            throw std::runtime_error("Shapes can not be broadcasted.");
        }

        outputShape[outputShape.dim() - 1 - i] = dim1 == 1 ? dim2 : dim1;
    }

    return outputShape;
}

bool increaseCoord(std::vector<size_t> &coords, const Shape &dims,
                   std::optional<int64_t> beginCoord) {
    assert(coords.size() == dims.dim());
    if (dims.dim() == 0) {
        return false;
    }

    int64_t beginCoordValue = beginCoord.value_or(coords.size() - 1);
    if (beginCoordValue < 0 ||
        beginCoordValue > static_cast<int64_t>(coords.size()) - 1) {
        return false;
    }

    for (int64_t i = beginCoordValue; i >= 0; i--) {
        if (coords[i] < dims[i] - 1) {
            coords[i]++;
            return true;
        }
        coords[i] = 0;
    }

    return false;
}

std::vector<size_t> clampCoordsToShape(const std::vector<size_t> &coords,
                                       const Shape &shape) {
    assert(coords.size() == shape.dim());
    std::vector<size_t> outCoords(coords.size());

    for (size_t i = 0; i < outCoords.size(); i++) {
        outCoords[outCoords.size() - 1 - i] = std::min(
            coords[coords.size() - 1 - i], shape[shape.dim() - 1 - i] - 1);
    }

    return outCoords;
}

size_t coordsToOffset(const std::vector<size_t> &coords,
                      const std::vector<size_t> &strides) {
    assert(coords.size() == strides.size());

    size_t offset = 0;
    for (size_t i = 0; i < coords.size(); i++) {
        offset += coords[i] * strides[i];
    }

    return offset;
}

size_t roundUpTo(size_t num, size_t roundTo) {
    if ((num % roundTo) != 0 || num == 0) {
        num += roundTo - (num % roundTo);
    }
    return num;
}

} // namespace nt
