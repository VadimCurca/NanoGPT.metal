#include <metal_stdlib>

using namespace metal;

kernel void addVecToVec(device const float4* inA,
                        device const float4* inB,
                        device float4* result,
                        uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] + inB[index];
}

kernel void addVecToScl(device const float4* inA,
                        device const float* inB,
                        device float4* result,
                        uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] + *inB;
}

kernel void addSclToVec(device const float* inA,
                        device const float4* inB,
                        device float4* result,
                        uint index [[thread_position_in_grid]]) {
    result[index] = *inA + inB[index];
}

kernel void multiplyVecToVec(device const float4* inA,
                             device const float4* inB,
                             device float4* result,
                             uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] * inB[index];
}

kernel void multiplyVecToScl(device const float4* inA,
                             device const float* inB,
                             device float4* result,
                             uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] * *inB;
}

kernel void multiplySclToVec(device const float* inA,
                             device const float4* inB,
                             device float4* result,
                             uint index [[thread_position_in_grid]]) {
    result[index] = *inA * inB[index];
}

kernel void relu(device const float4* inputArr,
                 device float4* outputArr,
                 uint index [[thread_position_in_grid]]) {
    outputArr[index] = max(0.f, inputArr[index]);
}

kernel void softmax(device const float* inputArr,
                    device float* outputArr,
                    constant uint& innerDimStride,
                    constant uint& dimStride,
                    constant uint& dimSize,
                    constant uint& outerDimStride,
                    uint3 index [[thread_position_in_grid]]) {
    const uint i = index[1];
    const uint j = index[0];
    
    uint dimOffset = outerDimStride * i + innerDimStride * j;
    
    float max = inputArr[dimOffset];
    for (uint k = 0, localOffset = dimOffset; k < dimSize; k++) {
        max = metal::max(max, inputArr[localOffset]);
        localOffset += dimStride;
    }
    
    float sum = 0;
    for (uint k = 0, localOffset = dimOffset; k < dimSize; k++) {
        outputArr[localOffset] = metal::exp(inputArr[localOffset] - max);
        sum += outputArr[localOffset];
        localOffset += dimStride;
    }
    
    for (uint k = 0, localOffset = dimOffset; k < dimSize; k++) {
        outputArr[localOffset] /= sum;
        localOffset += dimStride;
    }
}

kernel void meanVar(device const float* inputArr,
                    device float* meanArr,
                    device float* varArr,
                    constant uint& lastDimSize,
                    constant uint& batchStride,
                    uint index [[thread_position_in_grid]]) {
    const uint localOffset = index * batchStride;
    device const float *localInputArr = inputArr + localOffset;
    float sum = 0;
    
    for (uint c = 0; c < lastDimSize; c++) {
        sum += localInputArr[c];
    }
    meanArr[index] = sum / lastDimSize;
    
    sum = 0;
    for (uint c = 0; c < lastDimSize; c++) {
        sum += metal::pow(localInputArr[c] - meanArr[index], 2);
    }
    varArr[index] = sum / lastDimSize;
}

kernel void layerNorm(device const float* inputArr,
                      device const float* weightArr,
                      device const float* biasArr,
                      device float* meanArr,
                      device float* varArr,
                      device float* outputArr,
                      constant float& eps,
                      constant uint& batchStride,
                      uint3 index [[thread_position_in_grid]]) {
    const uint c = index[0];
    const uint b = index[1];
    
    const uint offset = b * batchStride + c;
    
    const float x = (inputArr[offset] - meanArr[b]) / metal::sqrt(varArr[b] + eps);
    outputArr[offset] = x * weightArr[c] + biasArr[c];
}

kernel void linear(device const float* inputArr [[buffer(0)]],
                   device const float* weightArr [[buffer(1)]],
                   device const float* biasArr [[buffer(2)]],
                   device float* outputArr [[buffer(3)]],
                   constant uint& inFeatures [[buffer(4)]],
                   constant uint& outFeatures [[buffer(5)]],
                   constant uint& hasBias [[buffer(6)]],
                   uint3 index [[thread_position_in_grid]]) {
    const uint i = index[1];
    const uint j = index[0];
    
    const device float *localInputArr = inputArr + i * inFeatures;
    const device float *localWeightArr = weightArr + j * inFeatures;
    device float *localOutputArr = outputArr + i * outFeatures;

    const device float4 *localInputArrVec = reinterpret_cast<const device float4*>(localInputArr);
    const device float4 *localWeightArrVec = reinterpret_cast<const device float4*>(localWeightArr);
    const uint numVec = inFeatures / 4;
    float4 sum = 0;
    
    for (uint k = 0; k < numVec; k++) {
        sum += localInputArrVec[k] * localWeightArrVec[k];
    }
    for (uint k = numVec * 4; k < inFeatures; k++) {
        sum[0] += localInputArr[k] * localWeightArr[k];
    }
    
    sum[0] = sum[0] + sum[1] + sum[2] + sum[3];
    
    if (hasBias) {
        sum[0] += biasArr[j];
    }
    
    localOutputArr[j] = sum[0];
}

kernel void linearTriangleFill(device const float* inputArr [[buffer(0)]],
                               device const float* weightArr [[buffer(1)]],
                               device const float* biasArr [[buffer(2)]],
                               device float* outputArr [[buffer(3)]],
                               constant uint& inFeatures [[buffer(4)]],
                               constant uint& outFeatures [[buffer(5)]],
                               constant uint& hasBias [[buffer(6)]],
                               constant float& triangleFillValue [[buffer(7)]],
                               uint3 index [[thread_position_in_grid]]) {
    const uint i = index[1];
    const uint j = index[0];
    
    const device float *localInputArr = inputArr + i * inFeatures;
    const device float *localWeightArr = weightArr + j * inFeatures;
    device float *localOutputArr = outputArr + i * outFeatures;
    
    if (j > i) {
        localOutputArr[j] = triangleFillValue;
        return;
    }
    
    const uint numVec = inFeatures / 4;
    const uint inFeaturesVecLoop = numVec * 4;
    
    float4 sum = 0;
    for (uint k = 0; k < inFeaturesVecLoop; k += 4) {
        const float4 vec1(localInputArr[k + 0], localInputArr[k + 1], localInputArr[k + 2], localInputArr[k + 3]);
        const float4 vec2(localWeightArr[k + 0], localWeightArr[k + 1], localWeightArr[k + 2], localWeightArr[k + 3]);
        sum += vec1 * vec2;
    }
    
    for (uint k = inFeaturesVecLoop; k < inFeatures; k++) {
        sum[0] += localInputArr[k] * localWeightArr[k];
    }
    
    sum[0] = sum[0] + sum[1] + sum[2] + sum[3];
    
    if (hasBias) {
        sum[0] += biasArr[j];
    }
    
    localOutputArr[j] = sum[0];
}

kernel void matmul(device const float* inputArr [[buffer(0)]],
                   device const float* otherArr [[buffer(1)]],
                   device float* outputArr [[buffer(2)]],
                   constant int& m [[buffer(3)]],
                   constant int& n [[buffer(4)]],
                   constant int& p [[buffer(5)]],
                   uint3 index [[thread_position_in_grid]]) {
    const uint i = index[1];
    const uint j = index[0];

    const device float *localInputArr = inputArr + i * n;
    const device float *localOtherArr = otherArr + j;
    device float *localOutputArr = outputArr + i * p;
    float sum = 0;

    for (size_t k = 0; k < n; k++) {
        sum += localInputArr[k] * localOtherArr[k * p];
    }

    localOutputArr[j] = sum;
}
