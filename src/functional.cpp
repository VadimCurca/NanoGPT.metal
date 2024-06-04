#include "functional.h"
#include "tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace nt::functional::cpu {

Tensor Softmax::forward(const Tensor &input, int64_t dim) {
    const auto inputShape = input.getShape();
    const auto inputStrides = nt::Shape::fromVector(input.getStrides());

    if (dim < 0) {
        dim += static_cast<int64_t>(inputShape.dim());
    }

    if (dim < 0 || dim >= static_cast<int64_t>(inputShape.dim())) {
        throw std::runtime_error("Softmax dim is out of expected range "
                                 "[-inputShape.dim(), inputShape.dim() - 1].");
    }

    Tensor output(inputShape);

    const auto inputArr = input.span();
    auto outputArr = output.span();

    const auto adjustedDim = static_cast<size_t>(dim);
    size_t outerDim = 1;
    size_t innerDim = 1;

    for (size_t i = 0; i < adjustedDim; i++) {
        outerDim *= inputShape[i];
    }
    for (size_t i = adjustedDim + 1; i < inputShape.dim(); i++) {
        innerDim *= inputShape[i];
    }

    const size_t outerDimStride =
        adjustedDim == 0 ? 0 : inputStrides[adjustedDim - 1];
    const size_t innerDimStride = (adjustedDim == inputShape.dim() - 1) ? 0 : 1;

    for (size_t i = 0; i < outerDim; i++) {
        for (size_t j = 0; j < innerDim; j++) {
            const size_t dimOffset = outerDimStride * i + innerDimStride * j;
            float max = inputArr[dimOffset];
            for (size_t k = 0, localOffset = dimOffset;
                 k < static_cast<size_t>(inputShape[adjustedDim]); k++) {
                max = std::max(max, inputArr[localOffset]);
                localOffset += inputStrides[adjustedDim];
            }

            float sum = 0;
            for (size_t k = 0, localOffset = dimOffset;
                 k < static_cast<size_t>(inputShape[adjustedDim]); k++) {
                outputArr[localOffset] = std::exp(inputArr[localOffset] - max);
                sum += outputArr[localOffset];
                localOffset += inputStrides[adjustedDim];
            }

            for (size_t k = 0, localOffset = dimOffset;
                 k < static_cast<size_t>(inputShape[adjustedDim]); k++) {
                outputArr[localOffset] = outputArr[localOffset] / sum;
                localOffset += inputStrides[adjustedDim];
            }
        }
    }

    return output;
}

Tensor ArgMax::forward(const Tensor &input, int64_t dim, bool keepDim) {
    const auto inputShape = input.getShape();
    const auto inputStrides = input.getStrides();
    assert(keepDim != false || inputShape.dim() != 1);

    if (dim < 0) {
        dim += static_cast<int64_t>(inputShape.dim());
    }
    const auto adjustedDim = static_cast<size_t>(dim);
    assert(dim >= 0 && dim < static_cast<int64_t>(inputShape.dim()));

    nt::Shape outputShape = inputShape;
    if (keepDim) {
        outputShape[adjustedDim] = 1;
    } else {
        outputShape.erase(outputShape.begin() +
                          static_cast<int64_t>(adjustedDim));
    }

    nt::Tensor output(outputShape);

    const auto inputArr = input.span();
    auto outputArr = output.span();

    size_t outerDim = 1;
    size_t innerDim = 1;

    for (size_t i = 0; i < adjustedDim; i++) {
        outerDim *= inputShape[i];
    }
    for (size_t i = adjustedDim + 1; i < inputShape.dim(); i++) {
        innerDim *= inputShape[i];
    }

    const size_t outerDimStride =
        adjustedDim == 0 ? 0 : inputStrides[adjustedDim - 1];
    const size_t innerDimStride = (adjustedDim == inputShape.dim() - 1) ? 0 : 1;

    for (size_t i = 0, outputArrIdx = 0; i < outerDim; i++) {
        for (size_t j = 0; j < innerDim; j++) {
            size_t inDimOffset = outerDimStride * i + innerDimStride * j;
            float max = inputArr[inDimOffset];
            size_t maxIdx = 0;

            for (size_t k = 0, localOffset = inDimOffset;
                 k < static_cast<size_t>(inputShape[adjustedDim]); k++) {
                if (max < inputArr[localOffset]) {
                    max = inputArr[localOffset];
                    maxIdx = k;
                }
                localOffset += inputStrides[adjustedDim];
            }

            outputArr[outputArrIdx++] = static_cast<float>(maxIdx);
        }
    }

    return output;
}

std::vector<Tensor> Split::forward(const Tensor &input,
                                   const std::vector<int64_t> &splitSections,
                                   int64_t dim) {
    const auto inputShape = input.getShape();
    const auto inputStrides = input.getStrides();

    if (dim < 0) {
        dim += static_cast<int64_t>(inputShape.dim());
    }
    const auto adjustedDim = static_cast<size_t>(dim);
    assert(dim >= 0 && dim < static_cast<int64_t>(inputShape.dim()));

    const size_t sectionsSum = std::accumulate(
        splitSections.begin(), splitSections.end(), 0, std::plus<>());
    assert(sectionsSum == inputShape[adjustedDim]);
    (void)sectionsSum; // may be unused

    const auto inputArr = input.span();
    std::vector<Tensor> outputs(splitSections.size());

    size_t outerDim = 1;
    size_t innerDim = 1;

    for (size_t i = 0; i < adjustedDim; i++) {
        outerDim *= inputShape[i];
    }
    for (size_t i = adjustedDim + 1; i < inputShape.dim(); i++) {
        innerDim *= inputShape[i];
    }

    const size_t outerDimStride =
        adjustedDim == 0 ? 0 : inputStrides[adjustedDim - 1];

    size_t lastSection = 0;
    Shape outputShape = inputShape;

    for (size_t s = 0; s < splitSections.size(); s++) {
        outputShape[adjustedDim] = splitSections[s];
        outputs[s] = Tensor(outputShape);

        auto outputArr = outputs[s].span();
        size_t outputArrIdx = 0;
        const size_t nextSection = lastSection + splitSections[s];

        for (size_t i = 0; i < outerDim; i++) {
            for (size_t k = lastSection; k < nextSection; k++) {
                size_t inDimOffset =
                    outerDimStride * i + inputStrides[adjustedDim] * k;
                const auto localInputArr = inputArr.subspan(inDimOffset);
                for (size_t j = 0; j < innerDim; j++) {
                    outputArr[outputArrIdx++] = localInputArr[j];
                }
            }
        }

        lastSection = nextSection;
    }

    return outputs;
}

Tensor Concat::forward(const std::vector<Tensor> &inputs, int64_t dim) {
    assert(inputs.empty() == false);

    const Shape firstInputShape = inputs[0].getShape();
    const size_t inputRank = firstInputShape.dim();

    if (dim < 0) {
        dim += static_cast<int64_t>(inputRank);
    }
    assert(dim >= 0 && dim < static_cast<int64_t>(inputRank));
    const size_t adjustedDim = dim;

    Shape shapeToCompare = firstInputShape;
    size_t outputDimSize = 0;

    for (const auto &input : inputs) {
        const auto inputShape = input.getShape();
        assert(inputShape.dim() == firstInputShape.dim());

        // Check that inputShapes are equal except the `dim` dimmension.
        shapeToCompare[adjustedDim] = inputShape[adjustedDim];
        assert(shapeToCompare == inputShape);

        outputDimSize += inputShape[adjustedDim];
    }

    Shape outputShape = firstInputShape;
    outputShape[adjustedDim] = outputDimSize;

    Tensor output(outputShape);
    auto outputArr = output.span();

    const auto outputStrides = output.getStrides();
    const size_t outputOuterDimStride =
        adjustedDim == 0 ? 0 : outputStrides[adjustedDim - 1];
    size_t lastOutputDim = 0;

    const size_t outerDim = std::accumulate(
        firstInputShape.cbegin(),
        firstInputShape.cbegin() + static_cast<int64_t>(adjustedDim), 1,
        std::multiplies<>());
    const size_t innerDim = std::accumulate(
        firstInputShape.cbegin() + static_cast<int64_t>(adjustedDim) + 1,
        firstInputShape.cend(), 1, std::multiplies<>());

    for (const auto &input : inputs) {
        const auto inputArr = input.span();
        const auto inputStrides = input.getStrides();
        const auto inputShape = input.getShape();

        const size_t inputOuterDimStride =
            adjustedDim == 0 ? 0 : inputStrides[adjustedDim - 1];

        const size_t localPlaneSize = innerDim * inputShape[adjustedDim];

        for (size_t i = 0; i < outerDim; i++) {
            const size_t inputOffset = inputOuterDimStride * i;
            const size_t outputOffset =
                outputOuterDimStride * i +
                outputStrides[adjustedDim] * lastOutputDim;
            const auto localInputArr = inputArr.subspan(inputOffset);
            const auto localOutputArr = outputArr.subspan(outputOffset);
            memcpy(localOutputArr.data(), localInputArr.data(),
                   sizeof(float) * localPlaneSize);
        }
        lastOutputDim += inputShape[adjustedDim];
    }

    return output;
}

Tensor Embedding::forward(const Tensor &input, const Tensor &weight) {
    auto inputShape = input.getShape();
    const auto weightShape = weight.getShape();

    assert(inputShape.dim() <= 2);
    assert(weightShape.dim() == 2);

    const bool inputHasBatch = inputShape.dim() == 2;
    const size_t embeddingDim = weightShape[1];

    size_t batch = 1;
    size_t seqLen = inputShape[0];

    if (inputHasBatch) {
        batch = inputShape[0];
        seqLen = inputShape[1];
    }

    nt::Shape outputShape{batch, seqLen, embeddingDim};
    nt::Tensor output(outputShape);

    const auto inputArr = input.span();
    const auto weightArr = weight.span();
    auto outputArr = output.span();

    size_t inputArrIdx = 0;

    const auto weightStrides = weight.getStrides();
    const auto outputStrides = output.getStrides();

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seqLen; s++) {
            float token = inputArr[inputArrIdx++];
            assert(token >= 0 && token < weightShape[0]);
            const size_t weightOffset =
                static_cast<int64_t>(token) * weightStrides[0];
            const size_t outputOffset =
                b * outputStrides[0] + s * outputStrides[1];
            std::memcpy(outputArr.subspan(outputOffset).data(),
                        weightArr.subspan(weightOffset).data(),
                        sizeof(float) * embeddingDim);
        }
    }

    if (!inputHasBatch) {
        outputShape.erase(outputShape.begin());
        output.reshape(outputShape);
    }

    return output;
}

Tensor DiscreteDistribution::forward(const Tensor &input,
                                     std::optional<std::mt19937> gen) {
    if (input.getShape().dim() != 1) {
        throw std::runtime_error(
            "Discrete distribuition requires inputShape == 1.");
    }

    if (!gen.has_value()) {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    std::discrete_distribution<> discreteDistribution(input.begin(),
                                                      input.end());
    return Tensor(static_cast<float>(discreteDistribution(gen.value())));
}

} // namespace nt::functional::cpu
