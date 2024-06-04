#include "functional.h"
#include "functional_metal.h"
#include "tensor.h"
#include "test_utils.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace {

using concatParams = std::tuple<std::vector<std::vector<size_t>>, int64_t>;

class ConcatTest : public testing::TestWithParam<concatParams> {};

TEST_P(ConcatTest, CPU) {
    auto [inputShapes, dim] = GetParam();

    const size_t numInputs = inputShapes.size();
    std::vector<torch::Tensor> inputsTorch(numInputs);
    std::vector<nt::Tensor> inputsNetGen(numInputs);

    const int rangeMin = -10;
    const int rangeMax = 10;

    for (size_t idx = 0; const auto &shape : inputShapes) {
        inputsTorch[idx] =
            nt::generateUniformTorchTensor(shape, rangeMin, rangeMax);
        inputsNetGen[idx] = nt::ntTensorfromTorchTensor(inputsTorch[idx]);

        idx++;
    }

    torch::Tensor outTorchReference = at::concat(inputsTorch, dim);

    nt::Tensor outNetGen =
        nt::functional::cpu::Concat::forward(inputsNetGen, dim);

    EXPECT_TRUE(nt::allclose(outTorchReference, outNetGen));
}

TEST_P(ConcatTest, Metal) {
    auto [inputShapes, dim] = GetParam();

    const size_t numInputs = inputShapes.size();
    std::vector<torch::Tensor> inputsTorch(numInputs);
    std::vector<nt::Tensor> inputsNetGen(numInputs);

    const int rangeMin = -10;
    const int rangeMax = 10;

    for (size_t idx = 0; const auto &shape : inputShapes) {
        inputsTorch[idx] =
            nt::generateUniformTorchTensor(shape, rangeMin, rangeMax);
        inputsNetGen[idx] = nt::ntTensorfromTorchTensor(inputsTorch[idx]);

        idx++;
    }

    torch::Tensor outTorchReference = at::concat(inputsTorch, dim);

    nt::Tensor outNetGen =
        nt::functional::metal::Concat::forward(inputsNetGen, dim);

    EXPECT_TRUE(nt::allclose(outTorchReference, outNetGen));
}

// NOLINTBEGIN
const std::vector<std::vector<std::vector<size_t>>> inputShapes0 = {
    {{2, 3, 4}, {5, 3, 4}, {1, 3, 4}, {3, 3, 4}}, {{2, 3, 4}}};
const std::vector<std::vector<std::vector<size_t>>> inputShapes1 = {
    {{2, 3, 4}, {2, 5, 4}, {2, 1, 4}, {2, 3, 4}}};
const std::vector<std::vector<std::vector<size_t>>> inputShapes2 = {
    {{2, 3, 4}, {2, 3, 5}, {2, 3, 1}, {2, 3, 3}}};
// NOLINTEND

INSTANTIATE_TEST_SUITE_P(ConcatTest0, ConcatTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes0),
                                            ::testing::Values(0) // dim
                                            ));

INSTANTIATE_TEST_SUITE_P(ConcatTest1, ConcatTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes1),
                                            ::testing::Values(1) // dim
                                            ));

INSTANTIATE_TEST_SUITE_P(
    ConcatTest2, ConcatTest,
    ::testing::Combine(::testing::ValuesIn(inputShapes2),
                       ::testing::ValuesIn(std::vector<int64_t>{2, -1}) // dim
                       ));

} // namespace
