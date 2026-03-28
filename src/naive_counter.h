#pragma once 

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "misc.h"
#include "normalizer.h"
#include "piece_spec.h"
#include "sentence.h"
#include "ustr.h"

namespace piece {
constexpr size_t INITAL_VOCAB_SIZE = 256;
constexpr size_t MAX_TEXT_SIZE = 1024;

struct IntPair {
    int first;
    int second;

    bool operator==(const IntPair& other) const {
        return first == other.first && second == other.second;
    }
};

struct Merge {
    IntPair pair;
    int idx;
};

struct NaiveModel {
    std::vector<Merge> merges;
    std::vector<std::vector<uint8_t>> vocab;
};

namespace naive {
    inline size_t GetPairIndexInCounts(const std::vector<size_t>& pair_counts,
                                      size_t pair_counts_size,
                                      const IntPair& pair) {
        for (size_t i = 0; i < pair_counts_size; ++i) {
            if (pair_counts[i * 3] == static_cast<size_t>(pair.first) && 
                pair_counts[i * 3 + 1] == static_cast<size_t>(pair.second)) {
                return i;
            }
        }
        return pair_counts_size;
    }

    inline void TokenCounts(const std::vector<int>& ids, 
                           std::vector<size_t>& pair_counts,
                           size_t& pair_counts_size) {
        pair_counts_size = 0;
        for (size_t i = 0; i < ids.size()-1; ++i) {
            IntPair pair{ids[i], ids[i+1]};
            size_t index = GetPairIndexInCounts(pair_counts, pair_counts_size, pair);

            if (index == pair_counts_size) {
                pair_counts[pair_counts_size*3] = pair.first;
                pair_counts[pair_counts_size*3 + 1] = pair.second;
                pair_counts[pair_counts_size*3 + 2] = 1;
                pair_counts_size++;
            } else {
                pair_counts[index*3 + 2]++;
            }
        }
    }

    inline void MergePair(std::vector<int>& ids, const IntPair& pair, int idx) {
        std::vector<int> new_ids;
        new_ids.reserve(ids.size());

        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == pair.first && i < ids.size() -1 && 
                ids[i+1] == pair.second) {
                new_ids.push_back(idx);
                ++i;
            } else {
                new_ids.push_back(ids[i]);
            }
        }

        ids = std::move(new_ids);
    }

    inline size_t GetPairIndex(const std::vector<Merge>& merges, const IntPair& pair) {
        auto it = std::find_if(merges.begin(), merges.end(),
            [&pair](const Merge& m) {return m.pair == pair;});
        
        return it != merges.end() ?
            std::distance(merges.begin(), it) :
            merges.size();
    }
} // namespace naive


class NaiveCounter {
public:
    NaiveCounter(const CounterSpec& counter_spec,
                 const NormalizerSpec& normalizer_spec);
    ~NaiveCounter();

    bool Count();
    bool Save() const;
    bool Serialize(Model* model) const;

private:
    using Sentence = std::pair<std::string, int64_t>;
    using Sentences = std::vector<Sentence>;

    void InitializeVocab();
    void DecodeToken(int id, std::string& text) const;
    bool InitMetaPieces();
    bool LoadSentences();
    void SplitSentencesByWhitespace();

    std::map<int, std::pair<std::string, Model::Piece::Type>> meta_pieces_;
    std::vector<std::pair<std::string, float>> pieces_;
    Sentences sentences_;
    CounterSpec counter_spec_;
    NormalizerSpec normalizer_spec_;

    std::vector<Merge> merges_;
    std::vector<std::vector<uint8_t>> vocab_;

    static constexpr int INITAL_VOCAB_SIZE = 256;
    static constexpr size_t MAX_TEXT_SIZE = 1024 * 1024;
};

} // namespace piece
