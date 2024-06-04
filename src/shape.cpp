#include "shape.h"
#include <ostream>

namespace nt {

std::ostream &operator<<(std::ostream &stream, const nt::Shape &shape) {
    stream << "[";

    if (shape.dim() > 0) {
        stream << shape[0];
    }
    for (size_t i = 1; i < shape.dim(); i++) {
        stream << ' ' << shape[i];
    }
    stream << "]\n";
    return stream;
}

} // namespace nt
