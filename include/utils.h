#pragma once

#include "shape.h"

#define DO_PRAGMA(x) _Pragma(#x);
#define NOWARN(warnoption, ...)                                                \
    DO_PRAGMA(clang diagnostic push)                                           \
    DO_PRAGMA(clang diagnostic ignored warnoption)                             \
    __VA_ARGS__                                                                \
    DO_PRAGMA(clang diagnostic pop)

namespace nt {

// Broadcast shapes, it throws an error if the shapes are not broadcastable.
Shape broadcastShapes(const Shape &shape1, const Shape &shape2);

// Increase the coords acording to dims, starting from beginCoord and going to
// the outermost dimensions. If beginCoord is not set, then the incrementation
// will start from `coords.size() - 1`.
bool increaseCoord(std::vector<size_t> &coords, const Shape &dims,
                   std::optional<int64_t> beginCoord = {});

// Clamp coords to shape. Expects both inputs to be of the same size.
std::vector<size_t> clampCoordsToShape(const std::vector<size_t> &coords,
                                       const Shape &shape);

// Computes offset from coords and strides. Expects both inputs to be of the
// same size.
size_t coordsToOffset(const std::vector<size_t> &coords,
                      const std::vector<size_t> &strides);

size_t roundUpTo(size_t num, size_t roundTo);

} // namespace nt
