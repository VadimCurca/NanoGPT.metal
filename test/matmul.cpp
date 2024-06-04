#include "functional_metal.h"
#include "tensor.h"
#include "test_utils.h"
#include "gtest/gtest.h"
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace {

using matMulParams = std::tuple<std::vector<size_t>, std::vector<size_t>,
                                size_t, size_t, size_t>;

class MatMulTest : public testing::TestWithParam<matMulParams> {};

TEST_P(MatMulTest, Metal) {
    const auto &[inputBatchDims, otherBatchDims, m, n, p] = GetParam();

    if (inputBatchDims.size() != 0 || otherBatchDims.size() != 0) {
        // Silently skip unsupported case
        return;
    }

    auto inputShape = inputBatchDims;
    inputShape.push_back(m);
    inputShape.push_back(n);

    auto otherShape = otherBatchDims;
    otherShape.push_back(n);
    otherShape.push_back(p);

    const torch::Tensor inTorch1 =
        nt::generateUniformTorchTensor(inputShape, -2, 2);
    const torch::Tensor inTorch2 =
        nt::generateUniformTorchTensor(otherShape, -2, 2);

    const nt::Tensor inNetGen1 = nt::ntTensorfromTorchTensor(inTorch1);
    const nt::Tensor inNetGen2 = nt::ntTensorfromTorchTensor(inTorch2);

    const torch::Tensor outTorchReference = at::matmul(inTorch1, inTorch2);

    const nt::Tensor outNetGen =
        nt::functional::metal::MatMul::forward(inNetGen1, inNetGen2);

    const double relTol = 1e-03;
    const double absTol = 1e-05;
    EXPECT_TRUE(nt::allclose(outTorchReference, outNetGen, relTol, absTol));
}

// NOLINTBEGIN(cert-err58-cpp)
const std::vector<std::vector<size_t>> inputBatchDims = {{1, 2, 1}};
const std::vector<std::vector<size_t>> otherBatchDims = {{}, {4}, {2, 1, 3}};
const std::vector<size_t> sizes = {1, 2, 30, 60};
// NOLINTEND(cert-err58-cpp)

INSTANTIATE_TEST_SUITE_P(MatMulTest1, MatMulTest,
                         ::testing::Combine(::testing::ValuesIn(inputBatchDims),
                                            ::testing::ValuesIn(otherBatchDims),
                                            ::testing::ValuesIn(sizes), // m
                                            ::testing::ValuesIn(sizes), // n
                                            ::testing::ValuesIn(sizes)  // p
                                            ));

INSTANTIATE_TEST_SUITE_P(MatMulTest2, MatMulTest,
                         ::testing::Combine(::testing::ValuesIn(otherBatchDims),
                                            ::testing::ValuesIn(inputBatchDims),
                                            ::testing::ValuesIn(sizes), // m
                                            ::testing::ValuesIn(sizes), // n
                                            ::testing::ValuesIn(sizes)  // p
                                            ));

} // namespace
