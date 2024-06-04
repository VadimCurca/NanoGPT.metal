#include "wb_loader.h"
#include "tensor.h"
#include <filesystem>
#include <stdexcept>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/torch.h>

namespace nt {

void WbLoader::loadFromJitModulePath(const std::string &filePath) {
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error("The file does not exist.");
    }
    torch::jit::script::Module module = torch::jit::load(filePath);
    const auto moduleNamedParameters = module.named_parameters();
    const auto moduleNamedAttributes = module.named_attributes();

    for (const auto &[name, tensor] : moduleNamedParameters) {
        const auto tensorShape = tensor.sizes();
        const nt::Shape ntShape(tensorShape.begin(), tensorShape.end());
        const float *rawData = static_cast<float *>(tensor.data_ptr());
        _dict[name] = nt::Tensor(ntShape, rawData);
    }

    for (const auto &[name, value] : moduleNamedAttributes) {
        torch::Tensor tensor;
        if (!value.isTensor()) {
            continue;
        }
        tensor = value.toTensor();
        const auto tensorShape = tensor.sizes();
        const nt::Shape ntShape(tensorShape.begin(), tensorShape.end());
        const float *rawData = static_cast<float *>(tensor.data_ptr());
        _dict[name] = nt::Tensor(ntShape, rawData);
    }
}

Tensor WbLoader::getTensor(const std::string &key) const {
    return _dict.at(key);
}

} // namespace nt
