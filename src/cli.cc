#include <iostream>
#include <string>
#include <cstring>

#include "piece_spec.h"
#include "naive_counter.h"
#include "naive_tokenizer.h"
#include "sentencepiece_counter.h"
#include "sentencepiece_tokenizer.h"
#include "bytepiece_counter.h"
#include "bytepiece_tokenizer.h"
#include "new_normalizer.h"

namespace piece {

void PrintUsage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " train [options]\n"
              << "  " << prog << " encode --model <file>\n"
              << "  " << prog << " decode --model <file>\n"
              << "\nTrain options:\n"
              << "  --method <naive|simple|sentencepiece|bytepiece>  (default: bytepiece)\n"
              << "  --input <file>         Input corpus file\n"
              << "  --model <prefix>       Model output prefix (default: tokenizer)\n"
              << "  --vocab-size <int>     Vocabulary size (default: 8000)\n"
              << "  --normalize <name>     Normalizer: identity|NMT_NFKC (default: identity)\n"
              << "\nEncode/Decode read from stdin, write to stdout.\n"
              << "Encode outputs one token per line (piece TAB id).\n"
              << "Decode reads token ids (space-separated) and outputs text.\n";
}

void RunTrain(const std::string& method, const std::string& input,
              const std::string& model_prefix, int vocab_size,
              const std::string& normalizer_name) {
    CounterSpec counter_spec;
    counter_spec.add_input(input);
    counter_spec.set_model_prefix(model_prefix);
    counter_spec.set_method(method);

    NormalizerSpec normalizer_spec;
    normalizer_spec.SetName(normalizer_name);

    // Adjust vocab_size for byte tokens and control tokens
    int size = vocab_size;
    if (method == "bytepiece" || method == "sentencepiece") {
        size = vocab_size + 256 + 3;  // +256 byte tokens +3 control tokens
    } else if (method == "naive" || method == "simple") {
        size = vocab_size + 3;
    }
    counter_spec.set_vocab_size(size);

    std::cerr << "Training: method=" << method << " input=" << input
              << " vocab_size=" << vocab_size << " model=" << model_prefix << "\n";

    if (method == "naive") {
        NaiveCounter counter(counter_spec, normalizer_spec);
        counter.Count();
        counter.Save();
    } else if (method == "simple") {
        SimpleCounter counter(counter_spec, normalizer_spec);
        counter.Count();
        counter.Save();
    } else if (method == "sentencepiece") {
        SentencePieceCounter counter(counter_spec, normalizer_spec);
        counter.Count();
        counter.Save();
    } else if (method == "bytepiece") {
        BytePieceCounter counter(counter_spec, normalizer_spec);
        counter.Count();
        counter.Save();
    } else {
        std::cerr << "Unknown method: " << method << "\n";
        return;
    }

    std::cerr << "Model saved to " << model_prefix << ".model\n";
}

void RunEncode(const std::string& model_file) {
    Model model;
    if (!model.Load(model_file)) {
        std::cerr << "Error: cannot load model: " << model_file << "\n";
        return;
    }

    const std::string& method = model.GetCounterSpec().method();
    const auto& normalizer_spec = model.GetNormalizerSpec();
    Normalizer normalizer(normalizer_spec);

    std::string line;
    if (method == "naive") {
        NaiveTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Encode(normalizer.Normalize(line));
            for (const auto& t : tokens) {
                std::cout << t.first << "\t" << t.second << "\n";
            }
            std::cout << "\n";
        }
    } else if (method == "simple") {
        SimpleTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Encode(normalizer.Normalize(line));
            for (const auto& t : tokens) {
                std::cout << t.first << "\t" << t.second << "\n";
            }
            std::cout << "\n";
        }
    } else if (method == "sentencepiece") {
        SentencePieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Encode(line);
            for (const auto& t : tokens) {
                std::cout << t.first << "\t" << t.second << "\n";
            }
            std::cout << "\n";
        }
    } else if (method == "bytepiece") {
        BytePieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            auto tokens = tokenizer.Encode(line);
            for (const auto& t : tokens) {
                std::cout << t.first << "\t" << t.second << "\n";
            }
            std::cout << "\n";
        }
    } else {
        std::cerr << "Unknown method in model: " << method << "\n";
    }
}

void RunDecode(const std::string& model_file) {
    Model model;
    if (!model.Load(model_file)) {
        std::cerr << "Error: cannot load model: " << model_file << "\n";
        return;
    }

    const std::string& method = model.GetCounterSpec().method();

    std::string line;
    if (method == "bytepiece") {
        BytePieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            // Parse space-separated ids
            std::vector<std::pair<std::string, int>> tokens;
            const auto& pieces = model.GetPieces();
            std::istringstream iss(line);
            int id;
            while (iss >> id) {
                if (id >= 0 && id < static_cast<int>(pieces.size())) {
                    tokens.emplace_back(pieces[id].GetPiece(), id);
                }
            }
            std::cout << tokenizer.Decode(tokens) << "\n";
        }
    } else if (method == "sentencepiece") {
        SentencePieceTokenizer tokenizer(model);
        while (std::getline(std::cin, line)) {
            std::vector<std::pair<std::string, int>> tokens;
            const auto& pieces = model.GetPieces();
            std::istringstream iss(line);
            int id;
            while (iss >> id) {
                if (id >= 0 && id < static_cast<int>(pieces.size())) {
                    tokens.emplace_back(pieces[id].GetPiece(), id);
                }
            }
            std::cout << tokenizer.Decode(tokens) << "\n";
        }
    } else {
        // naive/simple: reconstruct from piece strings
        const auto& pieces = model.GetPieces();
        while (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            int id;
            std::string result;
            while (iss >> id) {
                if (id >= 0 && id < static_cast<int>(pieces.size())) {
                    result += pieces[id].GetPiece();
                }
            }
            std::cout << result << "\n";
        }
    }
}

} // namespace piece

int main(int argc, char* argv[]) {
    if (argc < 2) {
        piece::PrintUsage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "train") {
        std::string method = "bytepiece";
        std::string input;
        std::string model_prefix = "tokenizer";
        int vocab_size = 8000;
        std::string normalizer = "identity";

        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
                method = argv[++i];
            } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                input = argv[++i];
            } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_prefix = argv[++i];
            } else if (std::strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) {
                vocab_size = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--normalize") == 0 && i + 1 < argc) {
                normalizer = argv[++i];
            } else {
                std::cerr << "Unknown option: " << argv[i] << "\n";
                piece::PrintUsage(argv[0]);
                return 1;
            }
        }

        if (input.empty()) {
            std::cerr << "Error: --input is required for train\n";
            return 1;
        }

        piece::RunTrain(method, input, model_prefix, vocab_size, normalizer);

    } else if (command == "encode" || command == "decode") {
        std::string model_file;

        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_file = argv[++i];
            }
        }

        if (model_file.empty()) {
            std::cerr << "Error: --model is required\n";
            return 1;
        }

        if (command == "encode") {
            piece::RunEncode(model_file);
        } else {
            piece::RunDecode(model_file);
        }

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        piece::PrintUsage(argv[0]);
        return 1;
    }

    return 0;
}
