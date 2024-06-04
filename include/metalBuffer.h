#include <iostream>
#include <utility>

#include "Metal/MTLCommandBuffer.hpp"
#include "metal_device_resources.h"
#include "tensor.h"

namespace nt {

class MetalBuffer {
  public:
    MetalBuffer() = default;
    explicit MetalBuffer(Shape shape) : _shape(std::move(shape)) {
        MetalDeviceResources &mdr = MetalDeviceResources::getInstance();
        // Make the number of elements divisible by 4 to avoid out of bounds
        // acces when using float4 or int4.
        const auto numel = roundUpTo(_shape.numel(), 4);
        _buffer = mdr.newBuffer(numel * sizeof(float));
    }
    explicit MetalBuffer(const Tensor &tensor) : _shape(tensor.getShape()) {
        MetalDeviceResources &mdr = MetalDeviceResources::getInstance();
        const auto numel = roundUpTo(_shape.numel(), 4);
        _buffer = mdr.newBufferWithBytes(tensor.begin(), numel * sizeof(float));
    }

    MetalBuffer(const MetalBuffer &) = delete;
    MetalBuffer &operator=(const MetalBuffer &) = delete;

    MetalBuffer(MetalBuffer &&b) noexcept
        : _commandBuffer(b._commandBuffer), _buffer(b._buffer),
          _shape(std::move(b._shape)) {
        b._buffer = nullptr;
        b._commandBuffer = nullptr;
    }

    MetalBuffer &operator=(MetalBuffer &&b) noexcept {
        if (_buffer != nullptr) {
            _buffer->release();
        }

        _buffer = b._buffer;
        b._buffer = nullptr;

        _commandBuffer = b._commandBuffer;
        b._commandBuffer = nullptr;

        _shape = std::move(b._shape);
        return *this;
    }

    [[nodiscard]] Tensor toTensor() const {
        throwIfCommandBufferUncompleted();
        assert(_buffer != nullptr);
        return {_shape, static_cast<float *>(_buffer->contents())};
    }

    [[nodiscard]] MTL::Buffer *raw() const { return _buffer; }
    [[nodiscard]] Shape getShape() const { return _shape; }

    void attachCommandBuffer(MTL::CommandBuffer *commandBuffer) {
        _commandBuffer = commandBuffer;
    }

    void detachCommandBuffer() {
        throwIfCommandBufferUncompleted();
        _commandBuffer = nullptr;
    }

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

    ~MetalBuffer() {
        if (_buffer != nullptr) {
            _buffer->release();
            _buffer = nullptr;
        }
    }

  private:
    // _commandBuffer is not owned by MetalBuffer
    MTL::CommandBuffer *_commandBuffer = nullptr;
    // _buffer is owned by MetalBuffer
    MTL::Buffer *_buffer = nullptr;
    Shape _shape;

    void throwIfCommandBufferUncompleted() const {
        if (_commandBuffer != nullptr) {
            if (_commandBuffer->status() >
                    MTL::CommandBufferStatus::CommandBufferStatusNotEnqueued &&
                _commandBuffer->status() <
                    MTL::CommandBufferStatus::CommandBufferStatusCompleted) {
                std::cout << _commandBuffer->status() << '\n';
                throw std::runtime_error("This MetalBuffer was produced by an "
                                         "operation that is in progress");
            }
        }
    }
};

} // namespace nt
