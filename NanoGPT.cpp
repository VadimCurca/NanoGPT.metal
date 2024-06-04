#include "Metal/MTLCommandBuffer.hpp"
#include "clock.h"
#include "functional.h"
#include "functional_metal.h"
#include "metal_device_resources.h"
#include "shape.h"
#include "wb_loader.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <ostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

class Tokenizer {
  public:
    // Read the tokens from a file path
    explicit Tokenizer(const std::string &filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            throw std::runtime_error("File could not be opened.");
        }
        // Use iterator to not skip the new line char if it exists.
        std::istreambuf_iterator<char> it{file};
        std::istreambuf_iterator<char> end;
        _vocab = std::set<char>(it, end);
        file.close();

        for (int idx = 0; char c : _vocab) {
            _charToInt[c] = idx;
            _intToChar[idx] = c;
            idx++;
        }
    }

    [[nodiscard]] int32_t stoi(char c) const {
        try {
            return _charToInt.at(c);
        } catch (const std::out_of_range &) {
            const std::string message = "The token \"" + std::string(1, c) +
                                        "\" is not in the vocabulary\n";
            throw std::runtime_error(message);
        }
    }
    [[nodiscard]] char itos(int32_t x) const { return _intToChar.at(x); }
    [[nodiscard]] size_t getVocabSize() const { return _vocab.size(); }

    [[nodiscard]] std::vector<int32_t> encode(const std::string &input) const {
        std::vector<int32_t> output(input.size());

        for (size_t i = 0; i < input.size(); i++) {
            output[i] = stoi(input[i]);
        }

        return output;
    }

    [[nodiscard]] std::string decode(const std::vector<int32_t> &input) const {
        const size_t numel = input.size();
        std::string output;
        output.reserve(numel);

        for (size_t i = 0; i < numel; i++) {
            output.push_back(itos(input[i]));
        }
        return output;
    }

    [[nodiscard]] std::string getVocab() const {
        return {_vocab.begin(), _vocab.end()};
    }

  private:
    std::set<char> _vocab;
    std::map<char, int32_t> _charToInt;
    std::map<int32_t, char> _intToChar;
};

class Embedding {
  public:
    Embedding(size_t numEmbeddings, size_t embedingDim)
        : _numEmbeddings(numEmbeddings), _embeddingDim(embedingDim) {}

    void loadWeights(const nt::WbLoader &wbLoader, const std::string &name) {
        _embeddingTable = wbLoader.getTensor(name + ".weight");
        assert(_embeddingTable.getShape() ==
               nt::Shape({_numEmbeddings, _embeddingDim}));
    }

    nt::Tensor forward(const nt::Tensor &input) {
        assert(_embeddingTable.getShape() ==
               nt::Shape({_numEmbeddings, _embeddingDim}));
        return nt::functional::cpu::Embedding::forward(input, _embeddingTable);
    }

  private:
    size_t _numEmbeddings;
    size_t _embeddingDim;
    nt::Tensor _embeddingTable;
};

class Linear {
  public:
    explicit Linear(bool bias = true) : _hasBias(bias) {}

    void loadWeights(const nt::WbLoader &wbLoader, const std::string &name) {
        _weightBuffer = nt::MetalBuffer(wbLoader.getTensor(name + ".weight"));
        if (_hasBias) {
            _biasBuffer = nt::MetalBuffer(wbLoader.getTensor(name + ".bias"));
        }
    }

    nt::MetalBuffer encode(const nt::MetalBuffer &inputBuffer,
                           MTL::CommandBuffer *commandBuffer) {
        return nt::functional::metal::Linear::encode(
            inputBuffer, _weightBuffer, commandBuffer, _biasBuffer);
    }

  private:
    bool _hasBias;

    nt::MetalBuffer _weightBuffer;
    std::optional<nt::MetalBuffer> _biasBuffer;
};

class Head {
  public:
    explicit Head(int64_t headSize)
        : _headSize(headSize), _key(Linear(false)), _query(Linear(false)),
          _value(Linear(false)) {

        _mulConstBuffer = nt::MetalBuffer(
            nt::Tensor(1.F / std::sqrt(static_cast<float>(_headSize))));
    }

    void loadWeights(const nt::WbLoader &wbLoader, const std::string &name) {
        _key.loadWeights(wbLoader, name + ".key");
        _query.loadWeights(wbLoader, name + ".query");
        _value.loadWeights(wbLoader, name + ".value");
    }

    nt::MetalBuffer encode(const nt::MetalBuffer &inputBuffer,
                           MTL::CommandBuffer *commandBuffer) {
        const auto inputShape = inputBuffer.getShape();

        assert(inputShape.dim() == 2);

        _k = _key.encode(inputBuffer, commandBuffer);
        _q = _query.encode(inputBuffer, commandBuffer);
        _v = _value.encode(inputBuffer, commandBuffer);

        _k = nt::functional::metal::Linear::encode(
            _q, _k, commandBuffer, {}, -std::numeric_limits<float>::infinity());
        _wei = nt::functional::metal::Multiply::encode(_k, _mulConstBuffer,
                                                       commandBuffer);
        _wei = nt::functional::metal::Softmax::encode(_wei, -1, commandBuffer);

        return nt::functional::metal::MatMul::encode(_wei, _v, commandBuffer);
    }

  private:
    int64_t _headSize;
    Linear _key, _query, _value;

    nt::MetalBuffer _mulConstBuffer;
    nt::MetalBuffer _k, _q, _v, _wei;
};

class MultiHeadAttention {
  public:
    MultiHeadAttention(int64_t numHeads, int64_t headSize)
        : _numHeads(numHeads), _headSize(headSize) {

        _heads.reserve(_numHeads);
        for (int64_t i = 0; i < _numHeads; i++) {
            _heads.emplace_back(_headSize);
        }
    }

    void loadWeights(const nt::WbLoader &wbLoader, const std::string &name) {
        for (size_t idx = 0; auto &head : _heads) {
            head.loadWeights(wbLoader,
                             name + ".heads." + std::to_string(idx++));
        }
        _proj.loadWeights(wbLoader, name + ".proj");
    }

    nt::MetalBuffer encode(const nt::MetalBuffer &inputBuffer,
                           MTL::CommandBuffer *&commandBuffer) {
        _xBuffers = std::vector<nt::MetalBuffer>(_heads.size());

        for (int idx = 0; idx < _numHeads; idx++) {
            _xBuffers[idx] = _heads[idx].encode(inputBuffer, commandBuffer);
        }

        _x =
            nt::functional::metal::Concat::encode(_xBuffers, -1, commandBuffer);
        return _proj.encode(_x, commandBuffer);
    }

  private:
    int64_t _numHeads;
    int64_t _headSize;
    std::vector<Head> _heads;
    Linear _proj;
    std::vector<nt::MetalBuffer> _xBuffers;
    nt::MetalBuffer _x;
};

class FeedForward {
  public:
    void loadWeights(const nt::WbLoader &wbLoader, const std::string &name) {
        _proj1.loadWeights(wbLoader, name + ".net.0");
        _proj2.loadWeights(wbLoader, name + ".net.2");
    }

    nt::MetalBuffer encode(const nt::MetalBuffer &inputBuffer,
                           MTL::CommandBuffer *commandBuffer) {
        _x = _proj1.encode(inputBuffer, commandBuffer);
        _x = nt::functional::metal::Relu::encode(_x, commandBuffer);
        return _proj2.encode(_x, commandBuffer);
    }

  private:
    Linear _proj1;
    Linear _proj2;
    nt::MetalBuffer _x;
};

class LayerNorm {
  public:
    void loadWeights(const nt::WbLoader &wbLoader, const std::string &name) {
        _weight = nt::MetalBuffer(wbLoader.getTensor(name + ".weight"));
        _bias = nt::MetalBuffer(wbLoader.getTensor(name + ".bias"));
    }

    nt::MetalBuffer encode(const nt::MetalBuffer &input,
                           MTL::CommandBuffer *commandBuffer) {
        const auto inputShape = input.getShape();
        const auto normalizedShape =
            static_cast<int64_t>(inputShape[inputShape.dim() - 1]);

        return _layerNormOp.encode(input, normalizedShape, _weight, _bias,
                                   commandBuffer);
    }

  private:
    nt::MetalBuffer _weight;
    nt::MetalBuffer _bias;
    nt::functional::metal::LayerNormalization _layerNormOp{};
};

class Block {
  public:
    Block(int64_t nEmbed, int64_t numHeads)
        : _numHeads(numHeads),
          _sa(MultiHeadAttention(_numHeads, nEmbed / _numHeads)) {}

    void loadWeights(const nt::WbLoader &wbLoader, const std::string &name) {
        _ln1.loadWeights(wbLoader, name + ".ln1");
        _sa.loadWeights(wbLoader, name + ".sa");
        _ln2.loadWeights(wbLoader, name + ".ln2");
        _ffwd.loadWeights(wbLoader, name + ".ffwd");
    }

    nt::MetalBuffer encode(const nt::MetalBuffer &inputBuffer,
                           MTL::CommandBuffer *&commandBuffer) {
        _x1 = _ln1.encode(inputBuffer, commandBuffer);
        _x1 = _sa.encode(_x1, commandBuffer);
        _x1 =
            nt::functional::metal::Add::encode(inputBuffer, _x1, commandBuffer);

        _x2 = _ln2.encode(_x1, commandBuffer);
        _x2 = _ffwd.encode(_x2, commandBuffer);
        return nt::functional::metal::Add::encode(_x1, _x2, commandBuffer);
    }

  private:
    int64_t _numHeads;
    MultiHeadAttention _sa;
    FeedForward _ffwd;
    LayerNorm _ln1;
    LayerNorm _ln2;
    nt::MetalBuffer _x1, _x2;
};

class GPTLanguageModel {
  public:
    GPTLanguageModel(size_t vocabSize, size_t nEmbed, size_t blockSize,
                     size_t nLayer, size_t nHead)
        : _vocabSize(vocabSize), _nEmbed(nEmbed), _blockSize(blockSize),
          _nHead(nHead), _tokenEmbeddingTable(Embedding(vocabSize, nEmbed)),
          _positionEmbeddingTable(Embedding(blockSize, nEmbed)) {

        _blocks.reserve(nLayer);
        for (size_t i = 0; i < nLayer; i++) {
            _blocks.emplace_back(_nEmbed, _nHead);
        }
    }

    void loadWeights(const nt::WbLoader &wbLoader) {
        _tokenEmbeddingTable.loadWeights(wbLoader, "token_embedding_table");
        _positionEmbeddingTable.loadWeights(wbLoader,
                                            "position_embedding_table");

        for (size_t idx = 0; auto &block : _blocks) {
            block.loadWeights(wbLoader, "blocks." + std::to_string(idx++));
        }

        _lnF.loadWeights(wbLoader, "ln_f");
        _lmHead.loadWeights(wbLoader, "lm_head");
    }

    nt::Tensor forward(const nt::Tensor &idx) {
        const auto idxShape = idx.getShape();
        assert(idxShape.dim() == 1);
        const size_t t = idxShape[0];

        const auto tokenEmb = _tokenEmbeddingTable.forward(idx); // (T,C)
        const auto posEmb = _positionEmbeddingTable.forward(
            nt::Tensor::arrange(t).reshape(nt::Shape{t})); // (T, C)

        const auto &mdr = nt::MetalDeviceResources::getInstance();
        MTL::CommandBuffer *commandBuffer = mdr.commandBuffer();

        nt::MetalBuffer tokenEmbBuffer(tokenEmb);
        nt::MetalBuffer posEmbBuffer(posEmb);

        nt::MetalBuffer x = nt::functional::metal::Add::encode(
            tokenEmbBuffer, posEmbBuffer, commandBuffer);
        commandBuffer->commit();

        for (Block &block : _blocks) {
            commandBuffer = mdr.commandBuffer();
            x = block.encode(x, commandBuffer);
            commandBuffer->commit();
        }

        commandBuffer = mdr.commandBuffer();

        x = _lnF.encode(x, commandBuffer);
        x = _lmHead.encode(x, commandBuffer);

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        assert(commandBuffer->status() ==
               MTL::CommandBufferStatus::CommandBufferStatusCompleted);

        return x.toTensor();
    }

    nt::Tensor generate(const std::string &prompt, size_t maxNewTokens,
                        const Tokenizer &tokenizer) {
        assert(idx.getShape().dim() == 1);
        std::cout << prompt;
        nt::Tensor idx(tokenizer.encode(prompt));

        for (size_t i = 0; i < maxNewTokens; i++) {
            const auto contextShape = idx.getShape();

            const auto idxSize = std::min(contextShape[0], _blockSize);
            const auto idxCond = nt::functional::cpu::Split::forward(
                idx,
                std::vector<int64_t>{
                    static_cast<int64_t>(contextShape[0] - idxSize),
                    static_cast<int64_t>(idxSize)},
                0)[1];                      // (T)
            auto logits = forward(idxCond); // (T, C)
            logits = nt::functional::cpu::Split::forward(
                logits,
                std::vector<int64_t>{
                    static_cast<int64_t>(logits.getShape()[0] - 1), 1},
                0)[1];
            logits.reshape(nt::Shape({_vocabSize}));
            const auto probs =
                nt::functional::cpu::Softmax::forward(logits, -1); // (C)
            const auto idxNext =
                nt::functional::cpu::DiscreteDistribution::forward(probs);
            idx = nt::functional::cpu::Concat::forward({idx, idxNext},
                                                       -1); // (T+1)
            std::cout << tokenizer.itos(static_cast<int>(idxNext.span()[0]))
                      << std::flush;
        }
        std::cout << '\n';

        return idx;
    }

  private:
    size_t _vocabSize;
    size_t _nEmbed;
    size_t _blockSize;
    size_t _nHead;

    Embedding _tokenEmbeddingTable;
    Embedding _positionEmbeddingTable;
    std::vector<Block> _blocks;
    LayerNorm _lnF;
    Linear _lmHead;
};

int main(int argc, char *argv[]) {
    std::filesystem::path jitModelPath;
    std::filesystem::path vocabPath;
    std::span argvSpan(argv, argc);

    switch (argc) {
    case 2: {
        std::filesystem::path dirPath = __FILE__;
        dirPath.remove_filename();

        jitModelPath = dirPath / "gpt2_trained_shakespeare.pt";
        vocabPath = dirPath / "vocab.txt";
        break;
    }
    case 4: {
        jitModelPath = argvSpan[2];
        vocabPath = argvSpan[3];
        break;
    }
    default: {
        std::cout << "Usage ./nanoGPT \"prompt\" or ./nanoGPT \"prompt\" "
                     "path/to/jibModel.pt path/to/vocab.txt\n";
        return 1;
    }
    }
    const std::string prompt = argvSpan[1];

    try {
        Tokenizer tokenizer(vocabPath);
        nt::WbLoader dict;
        dict.loadFromJitModulePath(jitModelPath);

        const size_t vocabSize = tokenizer.getVocabSize();
        const size_t nEmbed = 384;
        const size_t blockSize = 256;
        const size_t nLayer = 6;
        const size_t nHead = 6;
        GPTLanguageModel model(vocabSize, nEmbed, blockSize, nLayer, nHead);
        model.loadWeights(dict);

        nt::Clock perfClockGenerate;
        perfClockGenerate.start();
        const int numTokensToGenerate = 250;
        auto out = model.generate(prompt, numTokensToGenerate, tokenizer);
        perfClockGenerate.stop();
        auto outText = tokenizer.decode(out.toVector<int32_t>());

        std::cout << "\nPerformance clock: "
                  << perfClockGenerate.getDuration<std::chrono::milliseconds>()
                  << " ms" << '\n';
    } catch (const std::exception &e) {
        std::cout << "An unhandled exception has accurred with the following "
                     "message: "
                  << e.what() << "\n";
    }
}
