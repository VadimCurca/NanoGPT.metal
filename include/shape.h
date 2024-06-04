#pragma once

#include <numeric>
#include <vector>

namespace nt {

using value_type = size_t;
using Base = std::vector<value_type>;

class Shape : public Base {
  public:
    using Base::vector;
    [[nodiscard]] static Shape fromVector(const Base &vector) {
        return Shape(vector);
    }

  private:
    // Conversion constructor from base class
    explicit Shape(const Base &other) : Base(other) {};

  public:
    // Get the number of dimensions.
    [[nodiscard]] size_t dim() const { return size(); }

    // Get the number of the elements defined by shape.
    [[nodiscard]] value_type numel() const {
        if (size() == 0) {
            return 0;
        }

        return std::accumulate(begin(), end(), 1, std::multiplies());
    }

    friend std::ostream &operator<<(std::ostream &stream,
                                    const nt::Shape &shape);
};

} // namespace nt
