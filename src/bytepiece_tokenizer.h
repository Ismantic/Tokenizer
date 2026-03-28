#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "darts.h"
#include "normalizer.h"
#include "piece_spec.h"
#include "ustr.h"

namespace piece {

using float_t = double;

inline int SizeUTF8(uint8_t c) {
    if ((c & 0b10000000) == 0) return 1;
    if ((c & 0b11100000) == 0b11000000) return 2;
    if ((c & 0b11110000) == 0b11100000) return 3;
    if ((c & 0b11111000) == 0b11110000) return 4;
    return 1;
}

class BytePieceTokenizer {
public:
    using EncodeResult = std::vector<std::pair<std::string, int>>;
    using StrToInt = std::unordered_map<std::string_view, int>;

    explicit BytePieceTokenizer(
        const std::unordered_map<std::string, float_t>& dict);
    explicit BytePieceTokenizer(const Model& model);
    ~BytePieceTokenizer();

    EncodeResult Encode(std::string_view text) const;
    std::string Decode(const std::vector<int>& ids) const;
    std::string Decode(const EncodeResult& encoded) const;
    std::vector<std::string> Tokenize(const std::string& sentence) const;

private:
    struct Match {
        int e;
        int n;
        float_t w;
        Match(int e, int n, float_t w) : e(e), n(n), w(w) {}
    };

    int PieceID(std::string_view piece) const;
    std::vector<Match> GetMatches(const std::string& sentence) const;
    void InitFromModel();
    void InitFromDict(const std::unordered_map<std::string, float_t>& dict);

    const Model* model_ = nullptr;
    const std::unique_ptr<Normalizer> normalizer_;
    StrToInt pieces_;
    StrToInt reserve_;
    int unk_id_ = 0;
    new_darts::DoubleArray<int> trie_;
    std::unordered_map<int, float_t> value_map_;
};

}  // namespace piece
