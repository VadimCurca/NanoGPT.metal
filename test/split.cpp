#include "functional.h"
#include "tensor.h"
#include "test_utils.h"
#include "gtest/gtest.h"
#include <ATen/core/ATen_fwd.h>
#include <cstdint>
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace {

using splitParams =
    std::tuple<std::vector<size_t>, std::vector<int64_t>, int64_t>;

class SplitTest : public testing::TestWithParam<splitParams> {};

TEST_P(SplitTest, CPU) {
    const auto &[inputShape, sections, dim] = GetParam();

    const int rangeMin = -10;
    const int rangeMax = -10;
    const torch::Tensor inTorch =
        nt::generateUniformTorchTensor(inputShape, rangeMin, rangeMax);
    const at::IntArrayRef sectionsTorch(sections);

    const nt::Tensor inNetGen = nt::ntTensorfromTorchTensor(inTorch);

    const std::vector<torch::Tensor> outTorchReference =
        at::split(inTorch, sectionsTorch, dim);

    const std::vector<nt::Tensor> outNetGen =
        nt::functional::cpu::Split::forward(inNetGen, sections, dim);

    const double relTol = 1e-03;
    const double absTol = 1e-05;
    for (size_t i = 0; i < outNetGen.size(); i++) {
        EXPECT_TRUE(
            nt::allclose(outTorchReference[i], outNetGen[i], relTol, absTol));
    }
}

// NOLINTBEGIN(cert-err58-cpp)
const std::vector<std::vector<size_t>> inputShapes = {{1, 16, 8, 128}};

const std::vector<std::vector<int64_t>> sectionsDim0 = {{1}};
const std::vector<std::vector<int64_t>> sectionsDim1 = {{16}, {8, 4, 4}};
const std::vector<std::vector<int64_t>> sectionsDim2 = {{2, 1, 1, 4}};
const std::vector<std::vector<int64_t>> sectionsDim3 = {
    {64, 32, 32}, {1, 127}, {127, 1}, {64, 64}};
// NOLINTEND(cert-err58-cpp)

INSTANTIATE_TEST_SUITE_P(SplitTest0, SplitTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(sectionsDim0),
                                            ::testing::Values(0) // dim
                                            ));
INSTANTIATE_TEST_SUITE_P(SplitTest1, SplitTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(sectionsDim1),
                                            ::testing::Values(1) // dim
                                            ));
INSTANTIATE_TEST_SUITE_P(SplitTest2, SplitTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(sectionsDim2),
                                            ::testing::Values(2) // dim
                                            ));
INSTANTIATE_TEST_SUITE_P(SplitTest3, SplitTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(sectionsDim3),
                                            ::testing::Values(3) // dim
                                            ));

} // namespace
