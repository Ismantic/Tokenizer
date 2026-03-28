#include "naive_tokenizer.h"

#include <limits>
#include <stdexcept>

namespace piece {

NaiveTokenizer::NaiveTokenizer() : initialized_(false), model_proto_(nullptr) {}

NaiveTokenizer::NaiveTokenizer(const Model& model_proto)
    : model_proto_(&model_proto), initialized_(true) {
  InitFromModel();
}

NaiveTokenizer::~NaiveTokenizer() = default;

void NaiveTokenizer::InitFromModel() {
  if (!model_proto_) return;

  merges_.clear();
  vocab_.clear();
  pieces_.clear();
  reserved_.clear();

  const auto& counter_spec = model_proto_->GetCounterSpec();
  unk_id_ = counter_spec.unk_id();

  for (size_t i = 0; i < model_proto_->PiecesSize(); ++i) {
    const auto& p = model_proto_->GetPieces(i);
    const std::string& piece = p.GetPiece();
    pieces_[piece] = i;

    if (i < INITAL_VOCAB_SIZE) {
      if (piece.size() == 1) {
        vocab_.push_back({static_cast<uint8_t>(piece[0])});
      } else {
        vocab_.push_back({static_cast<uint8_t>(i)});
      }
    }

    if (p.GetType() != Model::Piece::NORMAL) {
      reserved_[piece] = i;
    }
  }

  RebuildMerges();
  initialized_ = true;
}

bool NaiveTokenizer::IsInitialized() const { return initialized_; }

float NaiveTokenizer::GetScore(int id) const {
  if (!model_proto_ || id < 0 || id >= static_cast<int>(model_proto_->PiecesSize())) {
    return 0.0f;
  }
  return model_proto_->GetPieces(id).GetScore();
}

int NaiveTokenizer::PieceID(std::string_view piece) const {
  const auto it = pieces_.find(piece);
  if (it != pieces_.end()) {
    return it->second;
  }

  const auto it2 = reserved_.find(piece);
  if (it2 != reserved_.end()) {
    return it2->second;
  }

  return unk_id_;
}

NaiveTokenizer::EncodeResult NaiveTokenizer::Encode(std::string_view text) const {
  if (!initialized_) {
    throw std::runtime_error("Tokenizer not initialized with a model");
  }

  std::string text_str(text);
  std::vector<int> ids;
  ids.assign(text_str.begin(), text_str.end());
  ApplyMerges(ids);

  EncodeResult result;
  for (int id : ids) {
    std::string piece;
    DecodeToken(id, piece);
    result.emplace_back(piece, id);
  }
  return result;
}

std::vector<std::string> NaiveTokenizer::Tokenize(std::string_view text) const {
  std::vector<std::string> tokens;
  auto encoded = Encode(text);
  tokens.reserve(encoded.size());
  for (const auto& [piece, id] : encoded) {
    tokens.push_back(piece);
  }
  return tokens;
}

std::string NaiveTokenizer::Decode(const std::vector<int>& ids) const {
  if (!initialized_) {
    throw std::runtime_error("Tokenizer not initialized with a model");
  }

  std::string result;
  for (int id : ids) {
    DecodeToken(id, result);
  }
  return result;
}

std::string NaiveTokenizer::Decode(const EncodeResult& encoded) const {
  std::vector<int> ids;
  ids.reserve(encoded.size());
  for (const auto& [piece, id] : encoded) {
    ids.push_back(id);
  }
  return Decode(ids);
}

void NaiveTokenizer::ApplyMerges(std::vector<int>& ids) const {
  while (ids.size() >= 2) {
    std::vector<size_t> pair_counts(MAX_TEXT_SIZE * 3);
    size_t pair_counts_size;
    naive::TokenCounts(ids, pair_counts, pair_counts_size);

    IntPair next{-1, -1};
    size_t next_idx = 0;
    size_t min_merge_idx = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < pair_counts_size * 3; i += 3) {
      IntPair pair{static_cast<int>(pair_counts[i]),
                   static_cast<int>(pair_counts[i + 1])};
      size_t idx = naive::GetPairIndex(merges_, pair);
      if (idx < merges_.size() && idx < min_merge_idx) {
        min_merge_idx = idx;
        next = pair;
        next_idx = idx;
      }
    }

    if (next.first == -1) break;
    naive::MergePair(ids, next, merges_[next_idx].idx);
  }
}

void NaiveTokenizer::DecodeToken(int id, std::string& text) const {
  if (id < INITAL_VOCAB_SIZE) {
    text.push_back(static_cast<char>(id));
    return;
  }

  for (const auto& merge : merges_) {
    if (merge.idx == id) {
      DecodeToken(merge.pair.first, text);
      DecodeToken(merge.pair.second, text);
      return;
    }
  }

  const auto& counter_spec = model_proto_->GetCounterSpec();
  if (id == counter_spec.unk_id()) {
    text += counter_spec.unk_piece();
  } else if (id == counter_spec.bos_id()) {
    text += counter_spec.bos_piece();
  } else if (id == counter_spec.eos_id()) {
    text += counter_spec.eos_piece();
  } else if (id == counter_spec.pad_id()) {
    text += counter_spec.pad_piece();
  }
}

void NaiveTokenizer::RebuildMerges() {
  std::unordered_map<std::string, int> piece_to_id;
  for (size_t i = 0; i < model_proto_->PiecesSize(); ++i) {
    piece_to_id[model_proto_->GetPieces(i).GetPiece()] = i;
  }

  for (const auto& [piece, id] : piece_to_id) {
    if (id >= INITAL_VOCAB_SIZE && piece.length() > 1) {
      for (size_t split = 1; split < piece.length(); ++split) {
        std::string prefix = piece.substr(0, split);
        std::string suffix = piece.substr(split);

        auto prefix_it = piece_to_id.find(prefix);
        auto suffix_it = piece_to_id.find(suffix);
        if (prefix_it != piece_to_id.end() && suffix_it != piece_to_id.end()) {
          IntPair pair{prefix_it->second, suffix_it->second};
          merges_.push_back({pair, id});

          if (id >= static_cast<int>(vocab_.size())) {
            while (vocab_.size() < static_cast<size_t>(id)) {
              vocab_.push_back({});
            }
            vocab_.push_back({});
          }
          break;
        }
      }
    }
  }

  std::sort(merges_.begin(), merges_.end(),
            [](const Merge& a, const Merge& b) { return a.idx < b.idx; });
}

}  // namespace piece
