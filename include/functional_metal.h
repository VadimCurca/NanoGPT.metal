#pragma once

#include "Metal/MTLCommandBuffer.hpp"
#include "metalBuffer.h"
#include <string>

namespace nt::functional::metal {

class Add {
  public:
    static MetalBuffer encode(const MetalBuffer &input,
                              const MetalBuffer &other,
                              MTL::CommandBuffer *commandBuffer);
    static Tensor forward(const Tensor &input, const Tensor &other);

  private:
    static std::string getAddVecToVecKernelName() { return {"addVecToVec"}; }
    static std::string getAddVecToSclKernelName() { return {"addVecToScl"}; }
    static std::string getAddScltoVecKernelName() { return {"addSclToVec"}; }
};

class Multiply {
  public:
    static MetalBuffer encode(const MetalBuffer &input,
                              const MetalBuffer &other,
                              MTL::CommandBuffer *commandBuffer);
    static Tensor forward(const Tensor &input, const Tensor &other);

  private:
    static std::string getMultiplyVecToVecKernelName() {
        return {"multiplyVecToVec"};
    }
    static std::string getMultiplyVecToSclKernelName() {
        return {"multiplyVecToScl"};
    }
    static std::string getMultiplySclToVecKernelName() {
        return {"multiplySclToVec"};
    }
};

class Linear {
  public:
    static MetalBuffer encode(const MetalBuffer &input,
                              const MetalBuffer &weight,
                              MTL::CommandBuffer *commandBuffer,
                              const std::optional<MetalBuffer> &bias = {},
                              const std::optional<float> &triangleFill = {});

    static Tensor forward(const Tensor &input, const Tensor &weight,
                          const std::optional<Tensor> &bias = {},
                          const std::optional<float> &triangleFill = {});

  private:
    static std::string getKernelName() { return {"linear"}; }
    static std::string getKernelNameTriangleFill() {
        return {"linearTriangleFill"};
    }
};

class MatMul {
  public:
    static MetalBuffer encode(const MetalBuffer &input,
                              const MetalBuffer &other,
                              MTL::CommandBuffer *commandBuffer);

    static Tensor forward(const Tensor &input, const Tensor &other);

  private:
    static std::string getKernelName() { return {"matmul"}; }
};

class Relu {
  public:
    static MetalBuffer encode(const MetalBuffer &input,
                              MTL::CommandBuffer *commandBuffer);
    static Tensor forward(const Tensor &input);

  private:
    static std::string getKernelName() { return {"relu"}; }
};

class Softmax {
  public:
    static MetalBuffer encode(const MetalBuffer &input, int64_t dim,
                              MTL::CommandBuffer *commandBuffer);
    static Tensor forward(const Tensor &input, int64_t dim);

  private:
    static std::string getKernelName() { return {"softmax"}; }
};

class Concat {
  public:
    static MetalBuffer encode(const std::vector<MetalBuffer> &inputs,
                              int64_t dim, MTL::CommandBuffer *commandBuffer);
    static Tensor forward(const std::vector<Tensor> &inputs, int64_t dim);
};

class LayerNormalization {
  public:
    // NOLINTBEGIN(*-magic-numbers)
    MetalBuffer encode(const MetalBuffer &input, int64_t normalizedShape,
                       const MetalBuffer &weight, const MetalBuffer &bias,
                       MTL::CommandBuffer *commandBuffer, float eps = 1e-5);
    static Tensor forward(const Tensor &input, int64_t normalizedShape,
                          const Tensor &weight, const Tensor &bias,
                          float eps = 1e-5);
    // NOLINTEND(*-magic-numbers)

  private:
    static std::string getMeanVarKernelName() { return {"meanVar"}; }
    static std::string getLayerNormKernelName() { return {"layerNorm"}; }
    MetalBuffer mean;
    MetalBuffer var;
};

} // namespace nt::functional::metal
