#include "functional.h"
#include "functional_metal.h"
#include "tensor.h"
#include "test_utils.h"
#include "gtest/gtest.h"
#include <ATen/ops/batch_norm.h>
#include <ATen/ops/layer_norm.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace {

using activationParams = std::tuple<std::vector<size_t>>;
using softmaxParams = std::tuple<std::vector<size_t>, int64_t>;
using argMaxParams = std::tuple<std::vector<size_t>, int64_t, bool>;

class ActivationTest {
  protected:
    void generateInputs(const std::vector<size_t> &shape) {
        const int rangeMin = -10;
        const int rangeMax = -10;
        inTorch = nt::generateUniformTorchTensor(shape, rangeMin, rangeMax);
        inNetGen = nt::ntTensorfromTorchTensor(inTorch);
    }

    // NOLINTBEGIN
    std::vector<int64_t> shapeInt64_t;
    torch::Tensor inTorch;
    nt::Tensor inNetGen;
    // NOLINTEND
};

class ActivationReluTest : public ActivationTest,
                           public testing::TestWithParam<activationParams> {
  protected:
    void SetUp() override {
        const auto &[shape] = GetParam();
        generateInputs(shape);
    }
};

TEST_P(ActivationReluTest, Metal) {
    const torch::Tensor outTorchReference = at::relu(inTorch);
    const nt::Tensor outNetGen = nt::functional::metal::Relu::forward(inNetGen);
    EXPECT_TRUE(nt::allclose(outTorchReference, outNetGen));
}

class ActivationSoftmaxTest : public ActivationTest,
                              public testing::TestWithParam<softmaxParams> {
  protected:
    void SetUp() override {
        const auto &[shape, dim] = GetParam();
        generateInputs(shape);
        this->dim = dim;
    }

    int64_t dim{}; // NOLINT
};

TEST_P(ActivationSoftmaxTest, compareWithTorchReferenceFP32) {
    const torch::Tensor outTorchReference = at::softmax(inTorch, dim);
    const nt::Tensor outNetGen =
        nt::functional::cpu::Softmax::forward(inNetGen, dim);
    EXPECT_TRUE(nt::allclose(outTorchReference, outNetGen));
}

TEST_P(ActivationSoftmaxTest, Metal) {
    const torch::Tensor outTorchReference = at::softmax(inTorch, dim);
    const nt::Tensor outNetGen =
        nt::functional::metal::Softmax::forward(inNetGen, dim);
    EXPECT_TRUE(nt::allclose(outTorchReference, outNetGen));
}

class ActivationArgMaxTest : public ActivationTest,
                             public testing::TestWithParam<argMaxParams> {
  protected:
    void SetUp() override {
        const auto &[shape, dim, keepDims] = GetParam();
        generateInputs(shape);
        this->dim = dim;
        this->keepDims = keepDims;
    }

    // NOLINTBEGIN
    int64_t dim{};
    bool keepDims{};
    // NOLINTEND
};

TEST_P(ActivationArgMaxTest, compareWithTorchReferenceFP32) {
    const torch::Tensor outTorchReference =
        at::argmax(inTorch, dim, keepDims).toType(at::ScalarType::Float);
    const nt::Tensor outNetGen =
        nt::functional::cpu::ArgMax::forward(inNetGen, dim, keepDims);
    EXPECT_TRUE(nt::allclose(outTorchReference, outNetGen));
}

// NOLINTBEGIN(cert-err58-cpp)
const std::vector<std::vector<size_t>> shapes = {
    {1}, {1, 1}, {1, 2, 3}, {10, 20, 30}, {32, 16, 8}};

INSTANTIATE_TEST_SUITE_P(ActivationReluTest, ActivationReluTest,
                         ::testing::Combine(::testing::ValuesIn(shapes)));

const std::vector<std::vector<size_t>> shapes1D{
    std::vector<size_t>{1}, {10}, {20}};
const std::vector<int64_t> axesFor1D{0};

const std::vector<std::vector<size_t>> shapes2D{
    std::vector<size_t>{1, 1}, {5, 1}, {1, 5}, {12, 20}};
const std::vector<int64_t> axesFor2D{-2, -1, 0, 1};

const std::vector<std::vector<size_t>> shapes4D{std::vector<size_t>{1, 1, 1, 1},
                                                {5, 1, 1, 1},
                                                {1, 5, 1, 1},
                                                {1, 1, 5, 1},
                                                {1, 1, 1, 5},
                                                {4, 8, 16, 32},
                                                {30, 15, 7, 3}};
const std::vector<int64_t> axesFor4D{-2, -1, 0, 1, 2, 3};
// NOLINTEND(cert-err58-cpp)

INSTANTIATE_TEST_SUITE_P(ActivationSoftmaxTest1D, ActivationSoftmaxTest,
                         ::testing::Combine(::testing::ValuesIn(shapes1D),
                                            ::testing::ValuesIn(axesFor1D)));

INSTANTIATE_TEST_SUITE_P(ActivationSoftmaxTest2D, ActivationSoftmaxTest,
                         ::testing::Combine(::testing::ValuesIn(shapes2D),
                                            ::testing::ValuesIn(axesFor2D)));

INSTANTIATE_TEST_SUITE_P(ActivationSoftmaxTest4D, ActivationSoftmaxTest,
                         ::testing::Combine(::testing::ValuesIn(shapes4D),
                                            ::testing::ValuesIn(axesFor4D)));

INSTANTIATE_TEST_SUITE_P(
    ActivationArgMaxTest1D, ActivationArgMaxTest,
    ::testing::Combine(::testing::ValuesIn(shapes1D),
                       ::testing::ValuesIn(axesFor1D),
                       ::testing::ValuesIn({true}) // keepDims
                       ));

INSTANTIATE_TEST_SUITE_P(
    ActivationArgMaxTest2D, ActivationArgMaxTest,
    ::testing::Combine(::testing::ValuesIn(shapes2D),
                       ::testing::ValuesIn(axesFor2D),
                       ::testing::ValuesIn({true, false}) // keepDims
                       ));

INSTANTIATE_TEST_SUITE_P(
    ActivationArgMaxTest4D, ActivationArgMaxTest,
    ::testing::Combine(::testing::ValuesIn(shapes4D),
                       ::testing::ValuesIn(axesFor4D),
                       ::testing::ValuesIn({true, false}) // keepDims
                       ));

} // namespace
