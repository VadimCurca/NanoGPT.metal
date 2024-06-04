An implementation of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) in C++, optimized for Apple GPU using the Metal API. It targets only the inference and uses weights exported from Pytorch.

Demo:

https://github.com/VadimCurca/NanoGPT.metal/assets/80581374/18d83de7-e5d6-4da8-8c6b-bc9afc7a5ff3

Build and run commands:
```
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DMETAL_CPP_DIR=/path/to/metal-cpp .. -G Xcode && cmake --build . -j
./Release/nanoGPT "Prompt"
```
