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
#include "new_normalizer.h"
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

inline bool IsValidUTF8(const std::string& str) {
    const unsigned char* bytes =
        reinterpret_cast<const unsigned char*>(str.c_str());
    const size_t len = str.length();

    for (size_t i = 0; i < len; i++) {
        if (bytes[i] <= 0x7F) {
            continue;
        }
        if ((bytes[i] & 0xE0) == 0xC0) {
            if (i + 1 >= len || (bytes[i + 1] & 0xC0) != 0x80) return false;
            if ((bytes[i] & 0x1E) == 0) return false;
            i += 1;
            continue;
        }
        if ((bytes[i] & 0xF0) == 0xE0) {
            if (i + 2 >= len || (bytes[i + 1] & 0xC0) != 0x80 ||
                (bytes[i + 2] & 0xC0) != 0x80) {
                return false;
            }
            if ((bytes[i] == 0xE0 && (bytes[i + 1] & 0x20) == 0) ||
                (bytes[i] == 0xED && (bytes[i + 1] & 0x20) == 0x20)) {
                return false;
            }
            i += 2;
            continue;
        }
        if ((bytes[i] & 0xF8) == 0xF0) {
            if (i + 3 >= len || (bytes[i + 1] & 0xC0) != 0x80 ||
                (bytes[i + 2] & 0xC0) != 0x80 ||
                (bytes[i + 3] & 0xC0) != 0x80) {
                return false;
            }
            if ((bytes[i] == 0xF0 && (bytes[i + 1] & 0x30) == 0) ||
                bytes[i] > 0xF4 ||
                (bytes[i] == 0xF4 && bytes[i + 1] > 0x8F)) {
                return false;
            }
            i += 3;
            continue;
        }
        return false;
    }
    return true;
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
