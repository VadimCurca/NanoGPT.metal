#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION

#include "metal_device_resources.h"
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

namespace nt {

MetalDeviceResources &MetalDeviceResources::getInstance() {
    static MetalDeviceResources instance;
    return instance;
}

void MetalDeviceResources::registerFunction(const std::string &name) {
    if (functionLibrary.find(name) != functionLibrary.end()) {
        return;
    }

    auto *const function_name =
        NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
    NS::Error *error = nullptr;
    MTL::Function *function = _library->newFunction(function_name);
    assert(function != nil);

    MTL::ComputePipelineState *computePipelineState =
        _device->newComputePipelineState(function, &error);
    function->release();

    functionLibrary[name] = computePipelineState;
}

} // namespace nt
