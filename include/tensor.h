#pragma once

#include "shape.h"
#include "utils.h"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace nt {

class Tensor {
  public:
    // Default constructor.
    Tensor() : _data(nullptr) {}

    // Allocates unitialized memory.
    explicit Tensor(const Shape &shape)
        : _shape(shape), _data(tensorAlloc(shape.numel())) {}

    // Allocates memory and copies that ammount of data from pointer.
    Tensor(const Shape &shape, const float *data)
        : _shape(shape), _data(tensorAlloc(shape.numel())) {
        std::memcpy(_data.get(), data, sizeof(float) * shape.numel());
    }

    // Allocates memory and initialize it with a constant.
    Tensor(const Shape &shape, float data)
        : _shape(shape), _data(tensorAlloc(shape.numel())) {
        std::fill_n(_data.get(), shape.numel(), data);
    }

    // Allocates memory and copies a scalar.
    explicit Tensor(float data) : _shape({1}), _data(tensorAlloc(1)) {
        *_data = data;
    }

    Tensor(const Shape &shape, std::initializer_list<float> data)
        : _shape(shape), _data(tensorAlloc(shape.numel())) {
        std::copy(data.begin(), data.end(), _data.get());
    }

    template <class T>
    explicit Tensor(std::vector<T> data)
        : _shape({data.size()}), _data(tensorAlloc(data.size())) {
        std::copy(data.begin(), data.end(), _data.get());
    }

    template <class T> std::vector<T> toVector() const {
        return std::vector<T>(begin(), end());
    }

    // Copy constructor does a deep copy of t.
    Tensor(const Tensor &t)
        : _shape(t._shape), _data(tensorAlloc(_shape.numel())) {
        memcpy(_data.get(), t._data.get(), _shape.numel() * sizeof(float));
    }

    // Move constructor does a shallow copy of t, the data of t is
    // de-referenced.
    Tensor(Tensor &&t) noexcept
        : _shape(std::move(t._shape)), _data(std::move(t._data)) {}

    // Default destructor.
    ~Tensor() = default;

    // Assignment operator releases its allocated data and does a deep copy of
    // t. It does nothing in case of self-assignment.
    Tensor &operator=(const nt::Tensor &t) {
        if (&t == this) {
            return *this;
        }
        _shape = t._shape;
        _data = tensorAlloc(_shape.numel());
        memcpy(_data.get(), t._data.get(), _shape.numel() * sizeof(float));

        return *this;
    }

    // Move assignment operator does a shallow copy of t, the data of t is
    // de-referenced.
    Tensor &operator=(Tensor &&t) noexcept {
        _shape = std::move(t._shape);
        _data = std::move(t._data);

        return *this;
    }

    // Get the element from coordinates specified by indices. If the number of
    // elements in indices is different from Shape dimensions, a runtime_error
    // will be triggered.
    float &operator()(std::initializer_list<size_t> indices) {
        if (indices.size() != _shape.dim()) {
            throw std::runtime_error("");
        }

        size_t offset = 0;
        int idx = 0;
        for (auto indice : indices) {
            if (indice < 0) {
                indice += _shape[idx];
            }
            offset += indice * getStrides()[idx++];
        }

        return span()[offset];
    }

    [[nodiscard]] std::span<const float> span() const {
        return {_data.get(), _shape.numel()};
    }
    [[nodiscard]] std::span<float> span() {
        return {_data.get(), _shape.numel()};
    }
    // Get raw pointers to data.
    [[deprecated]] [[nodiscard]] float const *raw() const { return begin(); }
    [[nodiscard]] float const *begin() const { return span().begin().base(); }
    [[nodiscard]] float const *end() const { return span().end().base(); }

    [[deprecated]] [[nodiscard]] float *raw() { return begin(); }
    // NOLINTBEGIN(readability-make-member-function-const)
    [[nodiscard]] float *begin() { return span().begin().base(); }
    [[nodiscard]] float *end() { return span().end().base(); }
    // NOLINTEND(readability-make-member-function-const)

    // Get data shape.
    [[nodiscard]] Shape getShape() const { return _shape; }

    // De-reference the tensor data.
    void release() { _data.reset(); }

    // Return true if the tensor is not initialized.
    [[nodiscard]] bool empty() const { return _data == nullptr; }

    // Get the strides of the tensor. For a shape = {2, 3, 4}, assuming
    // the data is contiguous, the strides will be {12, 4, 1}.
    [[nodiscard]] std::vector<size_t> getStrides() const {
        if (_shape.dim() == 0) {
            return {};
        }

        std::vector<size_t> strides(_shape.dim());
        *strides.rbegin() = 1;
        for (int i = static_cast<int>(strides.size()) - 2; i >= 0; i--) {
            strides[i] = _shape[i + 1] * strides[i + 1];
        }
        return strides;
    }

    // Changes the shape of the tensor without data permutation. Returns a
    // reference to *this.
    Tensor &reshape(Shape shape);

    // Changes the shape of the tensor without data permutation. Supports shape
    // inference. Returns a reference to *this.
    Tensor &reshape(std::vector<int64_t> shape);

    // Fill a 1D tensor with numbers from 0 to n-1.
    [[nodiscard]] static Tensor arrange(size_t n);

    friend std::ostream &operator<<(std::ostream &stream,
                                    const nt::Tensor &tensor) {
        stream << tensor.toString(0, 0);
        return stream;
    }

  private:
    Shape _shape;
    std::shared_ptr<float> _data;

    // Return string representation of tensor.
    [[nodiscard]] std::string toString(size_t dim, size_t offset) const;

    [[nodiscard]] static std::unique_ptr<float> tensorAlloc(size_t numel) {
        // Make the number of elements divisible by 4 to avoid out of bounds
        // acces when using float4 or int4.
        numel = roundUpTo(numel, 4);

        // Align the buffer to float4 or int4 size.
        return std::unique_ptr<float>(static_cast<float *>(aligned_alloc(
            4 * sizeof(float), roundUpTo(numel, 4) * sizeof(float))));
    }
};

} // namespace nt
