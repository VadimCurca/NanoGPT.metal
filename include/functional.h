#pragma once

#include "tensor.h"
#include <cstdint>
#include <random>
#include <vector>

namespace nt::functional::cpu {

class Softmax {
  public:
    static Tensor forward(const Tensor &input, int64_t dim);
};

class Split {
  public:
    static std::vector<Tensor>
    forward(const Tensor &input, const std::vector<int64_t> &splitSections,
            int64_t dim);
};

class Concat {
  public:
    static Tensor forward(const std::vector<Tensor> &inputs, int64_t dim);
};

class ArgMax {
  public:
    static Tensor forward(const Tensor &input, int64_t dim, bool keepDim);
};

class Embedding {
  public:
    static Tensor forward(const Tensor &input, const Tensor &weight);
};

class DiscreteDistribution {
  public:
    static Tensor forward(const Tensor &input,
                          std::optional<std::mt19937> gen = std::nullopt);
};

} // namespace nt::functional::cpu
