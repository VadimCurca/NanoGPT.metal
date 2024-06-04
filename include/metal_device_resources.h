#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <map>

namespace nt {

class MetalDeviceResources {
  public:
    static MetalDeviceResources &getInstance();
    MTL::Buffer *newBufferWithBytesNoCopy(const void *pointer,
                                          size_t size) const {
        return _device->newBuffer(pointer, size, MTL::ResourceStorageModeShared,
                                  nil);
    }
    MTL::Buffer *newBufferWithBytes(const void *pointer, size_t size) const {
        return _device->newBuffer(pointer, size,
                                  MTL::ResourceStorageModeShared);
    }
    [[nodiscard]] MTL::Buffer *newBuffer(size_t size) const {
        return _device->newBuffer(size, MTL::ResourceStorageModeShared);
    }
    [[nodiscard]] MTL::CommandBuffer *commandBuffer() const {
        return _commandQueue->commandBuffer();
    }
    MTL::ComputePipelineState *
    getComputePipelineState(const std::string &name) {
        if (functionLibrary.find(name) == functionLibrary.end()) {
            registerFunction(name);
        }
        return functionLibrary.at(name);
    }

    MetalDeviceResources(const MetalDeviceResources &) = delete;
    MetalDeviceResources(MetalDeviceResources &&) = delete;
    MetalDeviceResources &operator=(const MetalDeviceResources &) = delete;
    MetalDeviceResources &operator=(MetalDeviceResources &&) = delete;

  private:
    MetalDeviceResources() : _pool(NS::AutoreleasePool::alloc()->init()) {
        const std::string path = METAL_DEFAULT_LIBRARY;
        const NS::String *defaultLibraryPath = NS::String::string(
            path.c_str(), NS::StringEncoding::UTF8StringEncoding);
        assert(defaultLibraryPath != nullptr);

        NS::Error *error = nullptr;
        _device = MTL::CreateSystemDefaultDevice();
        _library = _device->newLibrary(defaultLibraryPath, &error);
        _commandQueue = _device->newCommandQueue();
    };

    ~MetalDeviceResources() {
        for (auto &[name, computePipelineState] : functionLibrary) {
            computePipelineState->release();
        }
        functionLibrary.clear();

        _commandQueue->release();
        _library->release();
        _device->release();
        _pool->release();
    };

    void registerFunction(const std::string &name);

    NS::AutoreleasePool *_pool;
    MTL::Device *_device;
    MTL::Library *_library;
    MTL::CommandQueue *_commandQueue;
    std::map<std::string, MTL::ComputePipelineState *> functionLibrary;
};

} // namespace nt
