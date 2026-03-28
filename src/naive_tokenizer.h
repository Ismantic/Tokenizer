#pragma once

#include "naive_counter.h"

namespace piece {

class NaiveTokenizer {
public:
    using EncodeResult = std::vector<std::pair<std::string, int>>;
    using StrToInt = std::unordered_map<std::string_view, int>;

    NaiveTokenizer();
    explicit NaiveTokenizer(const Model& model_proto);
    ~NaiveTokenizer();

    void InitFromModel();
    bool IsInitialized() const;
    float GetScore(int id) const;
    int PieceID(std::string_view piece) const;
    EncodeResult Encode(std::string_view text) const;
    std::string Detokenize(const std::vector<int>& ids) const;

private:
    void ApplyMerges(std::vector<int>& ids) const;
    void DecodeToken(int id, std::string& text) const;
    void RebuildMerges();

    const Model* model_proto_;
    std::vector<Merge> merges_;
    std::vector<std::vector<uint8_t>> vocab_;
    StrToInt pieces_;
    StrToInt reserved_;
    int unk_id_;
    bool initialized_;

    static constexpr int INITAL_VOCAB_SIZE = 256;
    static constexpr size_t MAX_TEXT_SIZE = 1024 * 1024;
};

}  // namespace piece
