#include "functional_metal.h"
#include "tensor.h"
#include "test_utils.h"
#include "gtest/gtest.h"
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace {

using linearParams = std::tuple<std::vector<size_t>, size_t, size_t, bool>;

class LinearTest : public testing::TestWithParam<linearParams> {};

TEST_P(LinearTest, Metal) {
    const auto &[inputBatchDims, inFeatures, outFeatures, hasBias] = GetParam();

    auto inputShape = inputBatchDims;
    inputShape.push_back(inFeatures);

    const std::vector<size_t> weightsShape{outFeatures, inFeatures};
    const std::vector<size_t> biasShape{outFeatures};

    const torch::Tensor inTorch =
        nt::generateUniformTorchTensor(inputShape, -2, 2);
    const torch::Tensor weightsTorch =
        nt::generateUniformTorchTensor(weightsShape, -2, 2);
    std::optional<torch::Tensor> biasTorch{};
    if (hasBias) {
        biasTorch = nt::generateUniformTorchTensor(biasShape, -2, 2);
    }

    const nt::Tensor inNetGen = nt::ntTensorfromTorchTensor(inTorch);
    const nt::Tensor weightsNetGen = nt::ntTensorfromTorchTensor(weightsTorch);
    std::optional<nt::Tensor> biasNetGen{};
    if (hasBias) {
        biasNetGen = nt::ntTensorfromTorchTensor(biasTorch.value());
    }

    const torch::Tensor outTorchReference =
        at::linear(inTorch, weightsTorch, biasTorch);

    const nt::Tensor outNetGen = nt::functional::metal::Linear::forward(
        inNetGen, weightsNetGen, biasNetGen);

    const double relTol = 1e-03;
    const double absTol = 1e-05;
    EXPECT_TRUE(nt::allclose(outTorchReference, outNetGen, relTol, absTol));
}

// NOLINTBEGIN(cert-err58-cpp)
const std::vector<std::vector<size_t>> inputBatchDims = {{1, 2, 1}};
const std::vector<size_t> inFeatures = {2, 30};
const std::vector<size_t> outFeatures = {4, 60};
const std::vector<bool> hasBias = {true, false};
// NOLINTEND(cert-err58-cpp)

INSTANTIATE_TEST_SUITE_P(LinearTest1, LinearTest,
                         ::testing::Combine(::testing::ValuesIn(inputBatchDims),
                                            ::testing::ValuesIn(inFeatures),
                                            ::testing::ValuesIn(outFeatures),
                                            ::testing::ValuesIn(hasBias)));

} // namespace
