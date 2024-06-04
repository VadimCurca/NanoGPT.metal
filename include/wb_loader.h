#pragma once

#include "tensor.h"
#include <iostream>
#include <map>
#include <string>

namespace nt {

class WbLoader {
  public:
    // Loads a network saved in Torch format and populate _dict.
    // Throws a runtime_error if it fails.
    void loadFromJitModulePath(const std::string &fileName);

    // Get the tensor associated if the key.
    // Throws an out_of_range exception if the key was not found.
    [[nodiscard]] Tensor getTensor(const std::string &key) const;

    void printKeys() const {
        std::cout << "Keys: {\n";
        for (const auto &[key, value] : _dict) {
            std::cout << key << '\n';
        }
        std::cout << "}\n";
    }

  private:
    std::map<std::string, Tensor> _dict;
};

} // namespace nt
