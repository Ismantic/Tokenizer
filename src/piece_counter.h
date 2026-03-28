#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "misc.h"
#include "normalizer.h"
#include "piece_spec.h"
#include "piece_tokenizer.h"
#include "sentence.h"
#include "ustr.h"

namespace piece {

class PieceCounter {
public:
  PieceCounter(const CounterSpec& counter_spec,
               const NormalizerSpec& normalizer_spec);
  ~PieceCounter();

  bool Count();
  bool Save() const;
  bool Serialize(Model* model) const;

private:
  using Sentence = std::pair<std::string, int64_t>;
  using Sentences = std::vector<Sentence>;

  bool InitMetaPieces();
  bool LoadSentences();
  void SplitSentencesByWhitespace();
  static IndexedList<int> BuildIndexedList(const std::string& text);
  static Multiset<std::pair<int, int>> InitPairsStats(
      const std::vector<std::string>& texts);
  static void Merge(const std::pair<int, int>& pair,
                    int new_id,
                    IndexedList<int>& indexed_list,
                    Multiset<std::pair<int, int>>* stats);

  std::map<int, std::pair<std::string, Model::Piece::Type>> meta_pieces_;
  std::vector<std::pair<std::vector<std::string>, float>> pieces_;
  Sentences sentences_;
  CounterSpec counter_spec_;
  NormalizerSpec normalizer_spec_;
  std::vector<std::pair<std::pair<int, int>, int>> merge_tree_;
  std::unordered_map<int, std::string> vocab_;
  int vocab_size_;
};

}  // namespace piece
