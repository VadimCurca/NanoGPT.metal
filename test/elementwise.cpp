#include "functional_metal.h"
#include "tensor.h"
#include "test_utils.h"
#include "gtest/gtest.h"
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace {

using elementwiseParams = std::tuple<std::vector<size_t>, std::vector<size_t>>;

class ElementwiseTest : public testing::TestWithParam<elementwiseParams> {
  protected:
    void SetUp() override {
        const auto &[inputShape, otherShape] = GetParam();

        const int rangeMin = -10;
        const int rangeMax = -10;

        inTorch1 =
            nt::generateUniformTorchTensor(inputShape, rangeMin, rangeMax);
        inTorch2 =
            nt::generateUniformTorchTensor(otherShape, rangeMin, rangeMax);

        inNetGen1 = nt::ntTensorfromTorchTensor(inTorch1);
        inNetGen2 = nt::ntTensorfromTorchTensor(inTorch2);
    }

    // NOLINTBEGIN
    std::vector<int64_t> inputShapeInt64_t;
    std::vector<int64_t> otherShapeInt64_t;
    torch::Tensor inTorch1;
    torch::Tensor inTorch2;

    nt::Tensor inNetGen1;
    nt::Tensor inNetGen2;
    // NOLINTEND
};

class ElementwiseAddTest : public ElementwiseTest {};
class ElementwiseMultiplyTest : public ElementwiseTest {};

TEST_P(ElementwiseAddTest, Metal) {
    torch::Tensor outTorchReference = inTorch1 + inTorch2;

    const auto outNetGen =
        nt::functional::metal::Add::forward(inNetGen1, inNetGen2);

    EXPECT_TRUE(nt::allclose(outTorchReference, outNetGen));
}

TEST_P(ElementwiseMultiplyTest, MetalMultiply) {
    auto outTorchReference = inTorch1.mul(inTorch2);

    nt::Tensor outNetGen =
        nt::functional::metal::Multiply::forward(inNetGen1, inNetGen2);

    EXPECT_TRUE(nt::allclose(outTorchReference, outNetGen));
}

// NOLINTBEGIN(cert-err58-cpp)
const std::vector<size_t> shape0 = {32, 16, 8};

const std::vector<std::vector<size_t>> shapes1 = {
    {1}, {1, 1}, {1, 2, 3}, {10, 20, 30}};

const std::vector<std::vector<size_t>> shapes2 = {{1}};
// NOLINTEND(cert-err58-cpp)

INSTANTIATE_TEST_SUITE_P(ElementwiseAddTest0, ElementwiseAddTest,
                         ::testing::Combine(::testing::Values(shape0),
                                            ::testing::Values(shape0)));

INSTANTIATE_TEST_SUITE_P(ElementwiseAddTest1, ElementwiseAddTest,
                         ::testing::Combine(::testing::ValuesIn(shapes1),
                                            ::testing::ValuesIn(shapes2)));

INSTANTIATE_TEST_SUITE_P(ElementwiseAddTest2, ElementwiseAddTest,
                         ::testing::Combine(::testing::ValuesIn(shapes2),
                                            ::testing::ValuesIn(shapes1)));

INSTANTIATE_TEST_SUITE_P(ElementwiseMultiplyTest1, ElementwiseMultiplyTest,
                         ::testing::Combine(::testing::ValuesIn(shapes1),
                                            ::testing::ValuesIn(shapes2)));

INSTANTIATE_TEST_SUITE_P(ElementwiseMultiplyTest2, ElementwiseMultiplyTest,
                         ::testing::Combine(::testing::ValuesIn(shapes2),
                                            ::testing::ValuesIn(shapes1)));

} // namespace
