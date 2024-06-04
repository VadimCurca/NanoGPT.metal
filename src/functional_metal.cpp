#include "functional_metal.h"
#include "Metal/MTLCommandBuffer.hpp"
#include "metal_device_resources.h"
#include "utils.h"
#include <sys/types.h>

namespace nt::functional::metal {

using T = float;

MetalBuffer Add::encode(const MetalBuffer &input, const MetalBuffer &other,
                        MTL::CommandBuffer *commandBuffer) {
    assert(commandBuffer != nullptr);

    MetalDeviceResources &mdr = MetalDeviceResources::getInstance();

    const auto inputShape = input.getShape();
    const auto otherShape = other.getShape();

    const size_t inputNumel = inputShape.numel();
    const size_t otherNumel = otherShape.numel();

    if (inputShape != otherShape && inputNumel != 1 && otherNumel != 1) {
        throw std::runtime_error(
            "Elementwise binary operation does not support shape broadcast, "
            "expected 2 identical shapes or 1 arbitrary shape and a constant.");
    }

    const auto outputShape = broadcastShapes(inputShape, otherShape);
    const size_t outputNumel = outputShape.numel();
    MetalBuffer output(outputShape);
    output.attachCommandBuffer(commandBuffer);

    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();
    MTL::ComputePipelineState *computePipelineState = nullptr;
    if (inputNumel == otherNumel) {
        computePipelineState =
            mdr.getComputePipelineState(getAddVecToVecKernelName());
    } else if (inputNumel == 1) {
        computePipelineState =
            mdr.getComputePipelineState(getAddScltoVecKernelName());
    } else {
        computePipelineState =
            mdr.getComputePipelineState(getAddVecToSclKernelName());
    }

    computeEncoder->setComputePipelineState(computePipelineState);
    computeEncoder->setBuffer(input.raw(), 0, 0);
    computeEncoder->setBuffer(other.raw(), 0, 1);
    computeEncoder->setBuffer(output.raw(), 0, 2);

    const auto float4Numel =
        static_cast<size_t>(std::ceil(static_cast<float>(outputNumel) / 4));

    const MTL::Size gridSize = MTL::Size(float4Numel, 1, 1);
    const NS::UInteger threadGroupNumel = std::min(
        computePipelineState->maxTotalThreadsPerThreadgroup(), float4Numel);
    const MTL::Size threadGroupSize = MTL::Size(threadGroupNumel, 1, 1);

    computeEncoder->dispatchThreads(gridSize, threadGroupSize);
    computeEncoder->endEncoding();

    return output;
}

Tensor Add::forward(const Tensor &input, const Tensor &other) {
    nt::MetalDeviceResources &mdr = nt::MetalDeviceResources::getInstance();

    nt::MetalBuffer inputBuffer(input);
    nt::MetalBuffer otherBuffer(other);

    MTL::CommandBuffer *commandBuffer = mdr.commandBuffer();
    nt::MetalBuffer outputBuffer =
        encode(inputBuffer, otherBuffer, commandBuffer);

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    assert(commandBuffer->status() ==
           MTL::CommandBufferStatus::CommandBufferStatusCompleted);

    return outputBuffer.toTensor();
}

MetalBuffer Multiply::encode(const MetalBuffer &inputBuffer,
                             const MetalBuffer &otherBuffer,
                             MTL::CommandBuffer *commandBuffer) {
    MetalDeviceResources &mdr = MetalDeviceResources::getInstance();

    const auto inputShape = inputBuffer.getShape();
    const auto otherShape = otherBuffer.getShape();

    const size_t inputNumel = inputShape.numel();
    const size_t otherNumel = otherShape.numel();

    if (inputShape != otherShape && inputNumel != 1 && otherNumel != 1) {
        throw std::runtime_error(
            "Elementwise binary operation does not support shape broadcast, "
            "expected 2 identical shapes or 1 arbitrary shape and a constant.");
    }

    const auto outputShape = broadcastShapes(inputShape, otherShape);
    const size_t outputNumel = outputShape.numel();
    MetalBuffer outputBuffer(outputShape);
    outputBuffer.attachCommandBuffer(commandBuffer);

    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();

    MTL::ComputePipelineState *computePipelineState = nullptr;

    if (inputNumel == otherNumel) {
        computePipelineState =
            mdr.getComputePipelineState(getMultiplyVecToVecKernelName());
    } else if (inputNumel == 1) {
        computePipelineState =
            mdr.getComputePipelineState(getMultiplySclToVecKernelName());
    } else {
        computePipelineState =
            mdr.getComputePipelineState(getMultiplyVecToSclKernelName());
    }

    computeEncoder->setComputePipelineState(computePipelineState);
    computeEncoder->setBuffer(inputBuffer.raw(), 0, 0);
    computeEncoder->setBuffer(otherBuffer.raw(), 0, 1);
    computeEncoder->setBuffer(outputBuffer.raw(), 0, 2);

    const auto float4Numel =
        static_cast<size_t>(std::ceil(static_cast<float>(outputNumel) / 4));

    const MTL::Size gridSize = MTL::Size(float4Numel, 1, 1);
    const NS::UInteger threadGroupNumel = std::min(
        computePipelineState->maxTotalThreadsPerThreadgroup(), float4Numel);
    const MTL::Size threadGroupSize = MTL::Size(threadGroupNumel, 1, 1);

    computeEncoder->dispatchThreads(gridSize, threadGroupSize);
    computeEncoder->endEncoding();

    return outputBuffer;
}

Tensor Multiply::forward(const Tensor &input, const Tensor &other) {
    nt::MetalDeviceResources &mdr = nt::MetalDeviceResources::getInstance();
    nt::MetalBuffer inputBuffer(input);
    nt::MetalBuffer otherBuffer(other);

    MTL::CommandBuffer *commandBuffer = mdr.commandBuffer();
    nt::MetalBuffer outputBuffer =
        encode(inputBuffer, otherBuffer, commandBuffer);

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    assert(commandBuffer->status() ==
           MTL::CommandBufferStatus::CommandBufferStatusCompleted);

    return outputBuffer.toTensor();
}

MetalBuffer Linear::encode(const MetalBuffer &input, const MetalBuffer &weight,
                           MTL::CommandBuffer *commandBuffer,
                           const std::optional<MetalBuffer> &bias,
                           const std::optional<float> &triangleFill) {
    assert(commandBuffer != nullptr);
    MetalDeviceResources &mdr = MetalDeviceResources::getInstance();

    const auto inputShape = input.getShape();
    const auto weightShape = weight.getShape();
    Shape biasShape{};

    assert(inputShape.dim() >= 1);
    assert(weightShape.dim() == 2);

    const size_t outFeatures = weightShape[0];
    const size_t inFeatures = weightShape[1];
    const bool hasBias = bias.has_value();

    assert(inputShape[inputShape.dim() - 1] == inFeatures);
    if (hasBias) {
        biasShape = bias->getShape();
        assert(biasShape == Shape{outFeatures});
    }

    auto outputShape = inputShape;
    outputShape[outputShape.dim() - 1] = outFeatures;
    MetalBuffer output(outputShape);
    output.attachCommandBuffer(commandBuffer);

    size_t batchDim = std::accumulate(
        inputShape.cbegin(),
        inputShape.cbegin() + static_cast<int>(inputShape.dim()) - 1, 1,
        std::multiplies<>());

    MTL::Buffer *inputBuffer = input.raw();
    MTL::Buffer *weightBuffer = weight.raw();
    MTL::Buffer *biasBuffer = hasBias ? bias->raw() : nullptr;
    MTL::Buffer *outputBuffer = output.raw();

    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();

    MTL::ComputePipelineState *computePipelineState =
        triangleFill.has_value()
            ? mdr.getComputePipelineState(getKernelNameTriangleFill())
            : mdr.getComputePipelineState(getKernelName());

    computeEncoder->setComputePipelineState(computePipelineState);

    int argIdx = 0;
    computeEncoder->setBuffer(inputBuffer, 0, argIdx++);
    computeEncoder->setBuffer(weightBuffer, 0, argIdx++);
    if (hasBias) {
        computeEncoder->setBuffer(biasBuffer, 0, argIdx);
    }
    argIdx++;
    computeEncoder->setBuffer(outputBuffer, 0, argIdx++);

    const uint inFeaturesUint = static_cast<uint>(inFeatures);
    const uint outFeaturesUint = static_cast<uint>(outFeatures);
    const uint hasBiasUint = static_cast<uint>(hasBias);

    computeEncoder->setBytes(&inFeaturesUint, sizeof(inFeatures), argIdx++);
    computeEncoder->setBytes(&outFeaturesUint, sizeof(outFeatures), argIdx++);
    computeEncoder->setBytes(&hasBiasUint, sizeof(hasBias), argIdx++);

    if (triangleFill.has_value()) {
        const float triangleFillValue = triangleFill.value();
        computeEncoder->setBytes(&triangleFillValue, sizeof(triangleFillValue),
                                 argIdx++);
    }

    const MTL::Size gridSize = MTL::Size(outFeatures, batchDim, 1);
    NS::UInteger w =
        std::min(outFeatures, computePipelineState->threadExecutionWidth());
    NS::UInteger h = std::min(
        batchDim, computePipelineState->maxTotalThreadsPerThreadgroup() / w);
    MTL::Size threadsPerThreadgroup = MTL::Size(w, h, 1);

    computeEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
    computeEncoder->endEncoding();

    return output;
}

Tensor Linear::forward(const Tensor &input, const Tensor &weight,
                       const std::optional<Tensor> &bias,
                       const std::optional<float> &triangleFill) {
    nt::MetalDeviceResources &mdr = nt::MetalDeviceResources::getInstance();
    nt::MetalBuffer inputBuffer(input);
    nt::MetalBuffer weightBuffer(weight);
    std::optional<nt::MetalBuffer> biasBuffer(bias);

    MTL::CommandBuffer *commandBuffer = mdr.commandBuffer();
    nt::MetalBuffer outputBuffer = encode(
        inputBuffer, weightBuffer, commandBuffer, biasBuffer, triangleFill);

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    assert(commandBuffer->status() ==
           MTL::CommandBufferStatus::CommandBufferStatusCompleted);

    return outputBuffer.toTensor();
}

MetalBuffer Relu::encode(const MetalBuffer &input,
                         MTL::CommandBuffer *commandBuffer) {
    assert(commandBuffer != nullptr);
    const auto inputShape = input.getShape();
    const auto numel = inputShape.numel();

    nt::MetalBuffer output(inputShape);
    output.attachCommandBuffer(commandBuffer);

    MTL::Buffer *inputBuffer = input.raw();
    MTL::Buffer *outputBuffer = output.raw();

    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();

    MetalDeviceResources &mdr = MetalDeviceResources::getInstance();
    MTL::ComputePipelineState *computePipelineState =
        mdr.getComputePipelineState(getKernelName());

    computeEncoder->setComputePipelineState(computePipelineState);

    int argIdx = 0;
    computeEncoder->setBuffer(inputBuffer, 0, argIdx++);
    computeEncoder->setBuffer(outputBuffer, 0, argIdx++);

    const auto numelFloat4 =
        static_cast<size_t>(std::ceil(static_cast<float>(numel) / 4));
    const MTL::Size gridSize = MTL::Size(numelFloat4, 1, 1);
    const NS::UInteger w = std::min(
        computePipelineState->maxTotalThreadsPerThreadgroup(), numelFloat4);
    MTL::Size threadsPerThreadgroup = MTL::Size(w, 1, 1);

    computeEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
    computeEncoder->endEncoding();

    return output;
}
Tensor Relu::forward(const Tensor &input) {
    nt::MetalDeviceResources &mdr = nt::MetalDeviceResources::getInstance();
    nt::MetalBuffer inputBuffer(input);

    MTL::CommandBuffer *commandBuffer = mdr.commandBuffer();
    nt::MetalBuffer outputBuffer = encode(inputBuffer, commandBuffer);

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    assert(commandBuffer->status() ==
           MTL::CommandBufferStatus::CommandBufferStatusCompleted);

    return outputBuffer.toTensor();
}

MetalBuffer Softmax::encode(const MetalBuffer &input, int64_t dim,
                            MTL::CommandBuffer *commandBuffer) {
    assert(commandBuffer != nullptr);
    nt::Shape inputShape = input.getShape();
    nt::Shape inputStrides = nt::Shape::fromVector(input.getStrides());

    if (dim < 0) {
        dim += static_cast<int64_t>(inputShape.dim());
    }

    if (dim < 0 || dim >= static_cast<int64_t>(inputShape.dim())) {
        throw std::runtime_error("Softmax dim is out of expected range "
                                 "[-inputShape.dim(), inputShape.dim() - 1].");
    }

    MetalBuffer output(inputShape);
    output.attachCommandBuffer(commandBuffer);

    auto adjustedDim = static_cast<size_t>(dim);
    size_t outerDim = 1;
    size_t innerDim = 1;

    // Remove static_casts here as well.
    for (size_t i = 0; i < adjustedDim; i++) {
        outerDim *= inputShape[i];
    }
    for (size_t i = adjustedDim + 1; i < inputShape.dim(); i++) {
        innerDim *= inputShape[i];
    }

    const uint outerDimStride =
        adjustedDim == 0 ? 0 : static_cast<uint>(inputStrides[adjustedDim - 1]);
    const uint dimStride = static_cast<uint>(inputStrides[adjustedDim]);
    const uint dimSize = static_cast<uint>(inputShape[adjustedDim]);
    const uint innerDimStride = (adjustedDim == inputShape.dim() - 1) ? 0 : 1;

    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();

    MetalDeviceResources &mdr = MetalDeviceResources::getInstance();
    MTL::ComputePipelineState *computePipelineState =
        mdr.getComputePipelineState(getKernelName());

    computeEncoder->setComputePipelineState(computePipelineState);

    int argIdx = 0;
    computeEncoder->setBuffer(input.raw(), 0, argIdx++);
    computeEncoder->setBuffer(output.raw(), 0, argIdx++);

    computeEncoder->setBytes(&innerDimStride, sizeof(uint), argIdx++);
    computeEncoder->setBytes(&dimStride, sizeof(uint), argIdx++);
    computeEncoder->setBytes(&dimSize, sizeof(uint), argIdx++);
    computeEncoder->setBytes(&outerDimStride, sizeof(uint), argIdx++);

    const MTL::Size gridSize = MTL::Size(innerDim, outerDim, 1);
    NS::UInteger w =
        std::min(innerDim, computePipelineState->threadExecutionWidth());
    NS::UInteger h = std::min(
        outerDim, computePipelineState->maxTotalThreadsPerThreadgroup() / w);
    MTL::Size threadsPerThreadgroup = MTL::Size(w, h, 1);

    computeEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
    computeEncoder->endEncoding();

    return output;
}

Tensor Softmax::forward(const Tensor &input, int64_t dim) {
    nt::MetalDeviceResources &mdr = nt::MetalDeviceResources::getInstance();
    nt::MetalBuffer inputBuffer(input);

    MTL::CommandBuffer *commandBuffer = mdr.commandBuffer();
    nt::MetalBuffer outputBuffer = encode(inputBuffer, dim, commandBuffer);

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    assert(commandBuffer->status() ==
           MTL::CommandBufferStatus::CommandBufferStatusCompleted);

    return outputBuffer.toTensor();
}

MetalBuffer MatMul::encode(const MetalBuffer &inputBuffer,
                           const MetalBuffer &otherBuffer,
                           MTL::CommandBuffer *commandBuffer) {
    auto inputShape = inputBuffer.getShape();
    auto otherShape = otherBuffer.getShape();

    assert(inputShape.dim() == 2);
    assert(otherShape.dim() == 2);

    const size_t m = inputShape[0];
    const size_t n = inputShape[1];
    const size_t p = otherShape[1];
    assert(n == otherShape[0]);

    Shape outputShape{m, p};
    MetalBuffer outputBuffer(outputShape);
    outputBuffer.attachCommandBuffer(commandBuffer);

    MetalDeviceResources &mdr = MetalDeviceResources::getInstance();

    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();

    MTL::ComputePipelineState *computePipelineState =
        mdr.getComputePipelineState(getKernelName());

    computeEncoder->setComputePipelineState(computePipelineState);

    int argIdx = 0;
    computeEncoder->setBuffer(inputBuffer.raw(), 0, argIdx++);
    computeEncoder->setBuffer(otherBuffer.raw(), 0, argIdx++);
    computeEncoder->setBuffer(outputBuffer.raw(), 0, argIdx++);

    computeEncoder->setBytes(&m, sizeof(size_t), argIdx++);
    computeEncoder->setBytes(&n, sizeof(size_t), argIdx++);
    computeEncoder->setBytes(&p, sizeof(size_t), argIdx++);

    const MTL::Size gridSize = MTL::Size(p, m, 1);
    NS::UInteger w = std::min(p, computePipelineState->threadExecutionWidth());
    NS::UInteger h =
        std::min(m, computePipelineState->maxTotalThreadsPerThreadgroup() / w);
    MTL::Size threadsPerThreadgroup = MTL::Size(w, h, 1);

    computeEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
    computeEncoder->endEncoding();

    return outputBuffer;
}

Tensor MatMul::forward(const Tensor &input, const Tensor &other) {
    nt::MetalDeviceResources &mdr = nt::MetalDeviceResources::getInstance();

    nt::MetalBuffer inputBuffer(input);
    nt::MetalBuffer otherBuffer(other);

    MTL::CommandBuffer *commandBuffer = mdr.commandBuffer();
    nt::MetalBuffer outputBuffer =
        encode(inputBuffer, otherBuffer, commandBuffer);

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    assert(commandBuffer->status() ==
           MTL::CommandBufferStatus::CommandBufferStatusCompleted);

    return outputBuffer.toTensor();
}

MetalBuffer Concat::encode(const std::vector<MetalBuffer> &inputs, int64_t dim,
                           MTL::CommandBuffer *commandBuffer) {
    assert(commandBuffer != nullptr);
    assert(inputs.empty() == false);

    const Shape firstInputShape = inputs[0].getShape();
    size_t inputRank = firstInputShape.dim();

    if (dim < 0) {
        dim += static_cast<int64_t>(inputRank);
    }
    assert(dim >= 0 && dim < static_cast<int64_t>(inputRank));
    size_t adjustedDim = dim;

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

    MetalBuffer output(outputShape);
    output.attachCommandBuffer(commandBuffer);

    const auto outputStrides = output.getStrides();
    const size_t outputOuterDimStride =
        adjustedDim == 0 ? 0 : outputStrides[adjustedDim - 1];
    size_t lastOutputDim = 0;

    size_t outerDim = std::accumulate(firstInputShape.cbegin(),
                                      firstInputShape.cbegin() +
                                          static_cast<int>(adjustedDim),
                                      1, std::multiplies<>());
    size_t innerDim = std::accumulate(
        firstInputShape.cbegin() + static_cast<int>(adjustedDim) + 1,
        firstInputShape.cend(), 1, std::multiplies<>());

    MTL::BlitCommandEncoder *blitEncoder = commandBuffer->blitCommandEncoder();

    for (const auto &input : inputs) {

        const auto inputStrides = input.getStrides();
        const auto inputShape = input.getShape();

        const size_t inputOuterDimStride =
            adjustedDim == 0 ? 0 : inputStrides[adjustedDim - 1];

        const size_t localPlaneSize = innerDim * inputShape[adjustedDim];

        for (size_t i = 0; i < outerDim; i++) {
            const uint inputByteOffset =
                static_cast<uint>(inputOuterDimStride * i * sizeof(T));
            const uint outputByteOffset =
                static_cast<uint>((outputOuterDimStride * i +
                                   outputStrides[adjustedDim] * lastOutputDim) *
                                  sizeof(T));
            const uint byteSize = static_cast<uint>(sizeof(T) * localPlaneSize);

            assert(inputByteOffset % 4 == 0);
            assert(outputByteOffset % 4 == 0);
            assert(byteSize % 4 == 0);

            blitEncoder->copyFromBuffer(input.raw(), inputByteOffset,
                                        output.raw(), outputByteOffset,
                                        byteSize);
        }
        lastOutputDim += inputShape[adjustedDim];
    }

    blitEncoder->endEncoding();

    return output;
}

Tensor Concat::forward(const std::vector<Tensor> &inputs, int64_t dim) {
    nt::MetalDeviceResources &mdr = nt::MetalDeviceResources::getInstance();

    std::vector<nt::MetalBuffer> inputBuffers(inputs.size());
    for (int idx = 0; const auto &input : inputs) {
        inputBuffers[idx++] = nt::MetalBuffer(input);
    }

    MTL::CommandBuffer *commandBuffer = mdr.commandBuffer();
    nt::MetalBuffer outputBuffer = encode(inputBuffers, dim, commandBuffer);

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    assert(commandBuffer->status() ==
           MTL::CommandBufferStatus::CommandBufferStatusCompleted);

    return outputBuffer.toTensor();
}

MetalBuffer LayerNormalization::encode(const MetalBuffer &input,
                                       const int64_t normalizedShape,
                                       const MetalBuffer &weight,
                                       const MetalBuffer &bias,
                                       MTL::CommandBuffer *commandBuffer,
                                       float eps) {

    assert(commandBuffer != nullptr);

    const auto inputShape = input.getShape();
    const auto inputStride = input.getStrides();
    assert(inputShape.dim() >= 2);

    if (normalizedShape !=
        static_cast<int64_t>(inputShape[inputShape.dim() - 1])) {
        throw std::runtime_error(
            "Layer Normalization expected normalizedShape to be a scalar, "
            "equal to inputShape[inputShape.dim() - 1]");
    }

    MetalBuffer output(inputShape);
    output.attachCommandBuffer(commandBuffer);

    const size_t numBatch = std::accumulate(
        inputShape.cbegin(), inputShape.cend() - 1, 1, std::multiplies<>());
    const size_t lastDimSize = inputShape[inputShape.dim() - 1];
    const size_t batchStride = lastDimSize;

    assert(weight.getShape() == nt::Shape{lastDimSize});
    assert(bias.getShape() == nt::Shape{lastDimSize});

    mean = MetalBuffer(nt::Shape{numBatch});
    var = MetalBuffer(nt::Shape{numBatch});

    MetalDeviceResources &mdr = MetalDeviceResources::getInstance();
    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();
    MTL::ComputePipelineState *computePipelineState =
        mdr.getComputePipelineState(getMeanVarKernelName());
    computeEncoder->setComputePipelineState(computePipelineState);

    int argIdx = 0;
    computeEncoder->setBuffer(input.raw(), 0, argIdx++);
    computeEncoder->setBuffer(mean.raw(), 0, argIdx++);
    computeEncoder->setBuffer(var.raw(), 0, argIdx++);

    const uint lastDimSizeUint = static_cast<uint>(lastDimSize);
    const uint batchStrideUint = static_cast<uint>(batchStride);
    computeEncoder->setBytes(&lastDimSizeUint, sizeof(uint), argIdx++);
    computeEncoder->setBytes(&batchStrideUint, sizeof(uint), argIdx++);

    MTL::Size gridSize = MTL::Size(numBatch, 1, 1);
    NS::UInteger w = std::min(
        computePipelineState->maxTotalThreadsPerThreadgroup(), numBatch);
    MTL::Size threadsPerThreadgroup = MTL::Size(w, 1, 1);

    computeEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
    computeEncoder->endEncoding();

    computeEncoder = commandBuffer->computeCommandEncoder();
    computePipelineState =
        mdr.getComputePipelineState(getLayerNormKernelName());
    computeEncoder->setComputePipelineState(computePipelineState);

    argIdx = 0;
    computeEncoder->setBuffer(input.raw(), 0, argIdx++);
    computeEncoder->setBuffer(weight.raw(), 0, argIdx++);
    computeEncoder->setBuffer(bias.raw(), 0, argIdx++);
    computeEncoder->setBuffer(mean.raw(), 0, argIdx++);
    computeEncoder->setBuffer(var.raw(), 0, argIdx++);
    computeEncoder->setBuffer(output.raw(), 0, argIdx++);

    computeEncoder->setBytes(&eps, sizeof(float), argIdx++);
    computeEncoder->setBytes(&batchStrideUint, sizeof(uint), argIdx++);

    gridSize = MTL::Size(lastDimSize, numBatch, 1);
    w = std::min(lastDimSize, computePipelineState->threadExecutionWidth());
    NS::UInteger h = std::min(
        numBatch, computePipelineState->maxTotalThreadsPerThreadgroup() / w);
    threadsPerThreadgroup = MTL::Size(w, h, 1);

    computeEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
    computeEncoder->endEncoding();

    return output;
}

Tensor LayerNormalization::forward(const Tensor &input,
                                   const int64_t normalizedShape,
                                   const Tensor &weight, const Tensor &bias,
                                   float eps) {
    nt::MetalDeviceResources &mdr = nt::MetalDeviceResources::getInstance();
    nt::MetalBuffer inputBuffer(input);
    nt::MetalBuffer weightBuffer(weight);
    nt::MetalBuffer biasBuffer(bias);

    LayerNormalization layerNormOp{};
    MTL::CommandBuffer *commandBuffer = mdr.commandBuffer();
    nt::MetalBuffer outputBuffer =
        layerNormOp.encode(inputBuffer, normalizedShape, weightBuffer,
                           biasBuffer, commandBuffer, eps);

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    assert(commandBuffer->status() ==
           MTL::CommandBufferStatus::CommandBufferStatusCompleted);

    return outputBuffer.toTensor();
}

} // namespace nt::functional::metal
