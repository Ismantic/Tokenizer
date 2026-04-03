#include "piece_tokenizer.h"

namespace piece {

PieceTokenizer::PieceTokenizer(const Model& model) : model_(&model) {
  const auto& counter_spec = model_->GetCounterSpec();
  unk_id_ = counter_spec.unk_id();

  for (size_t i = 0; i < model_->PiecesSize(); ++i) {
    const auto& piece = model_->GetPieces(i);
    pieces_[piece.GetPiece()] = i;

    const std::string& u = piece.u();
    const std::string& v = piece.v();
    if (!u.empty() && !v.empty()) {
      int u_id = PieceID(u);
      int v_id = PieceID(v);
      if (u_id >= 0 && v_id >= 0) {
        merge_rules_.emplace_back(std::pair<int, int>(u_id, v_id),
                                  static_cast<int>(i));
      }
    }
  }

  LOG(INFO) << "Building merge rules from model with "
            << model_->PiecesSize() << " pieces, "
            << merge_rules_.size() << " merge rules";
}

PieceTokenizer::~PieceTokenizer() = default;

PieceTokenizer::EncodeResult PieceTokenizer::Encode(std::string_view text) const {
  std::string normalized_text = NormalizeText(text);
  auto token_list = BuildInitialTokenList(normalized_text);
  ApplyMergeRules(token_list);
  return TokenIdsToResult(token_list);
}

std::vector<std::string> PieceTokenizer::Tokenize(std::string_view text) const {
  std::vector<std::string> tokens;
  auto encoded = Encode(text);
  tokens.reserve(encoded.size());
  for (const auto& [piece, id] : encoded) {
    tokens.push_back(piece);
  }
  return tokens;
}

std::string PieceTokenizer::Decode(const std::vector<int>& ids) const {
  std::string result;
  for (int id : ids) {
    if (id < 0 || id >= static_cast<int>(model_->PiecesSize())) {
      if (unk_id_ >= 0) {
        result += model_->GetPieces(unk_id_).GetPiece();
      }
      continue;
    }

    const auto& piece = model_->GetPieces(id);
    if (piece.GetType() == Model::Piece::BYTE) {
      auto byte_value = ustr::PieceToByte(piece.GetPiece());
      if (byte_value >= 0) {
        result.push_back(static_cast<char>(byte_value));
      } else {
        result += piece.GetPiece();
      }
    } else if (piece.GetType() != Model::Piece::UNKNOWN &&
               piece.GetType() != Model::Piece::CONTROL) {
      result += piece.GetPiece();
    }
  }

  if (model_->HasNormalizerSpec()) {
    Normalizer normalizer(model_->GetNormalizerSpec());
    return normalizer.ReplaceSpace(result);
  }

  return result;
}

std::string PieceTokenizer::Decode(const EncodeResult& rs) const {
  std::vector<int> ids;
  ids.reserve(rs.size());
  for (const auto& [piece, id] : rs) {
    ids.push_back(id);
  }
  return Decode(ids);
}

int PieceTokenizer::PieceID(std::string_view piece) const {
  const auto it = pieces_.find(piece);
  return it != pieces_.end() ? it->second : unk_id_;
}

std::string PieceTokenizer::NormalizeText(std::string_view text) const {
  if (model_->HasNormalizerSpec()) {
    Normalizer normalizer(model_->GetNormalizerSpec());
    return normalizer.Normalize(std::string(text));
  }
  return std::string(text);
}

IndexedList<int> PieceTokenizer::BuildInitialTokenList(
    const std::string& text) const {
  std::vector<int> initial_token_ids;
  initial_token_ids.reserve(text.size());

  for (unsigned char c : text) {
    std::string byte_str(1, c);
    auto it = pieces_.find(byte_str);
    initial_token_ids.push_back(it != pieces_.end() ? it->second : unk_id_);
  }

  return IndexedList<int>(initial_token_ids.begin(), initial_token_ids.end());
}

void PieceTokenizer::ApplyMergeRules(IndexedList<int>& token_list) const {
  for (const auto& rule : merge_rules_) {
    const auto& pair = rule.first;
    int merged_id = rule.second;
    auto& nodes = token_list.GetIndex(pair);
    if (!nodes.empty()) {
      Merge(pair, merged_id, token_list);
    }
  }
}

void PieceTokenizer::Merge(const std::pair<int, int>& pair,
                           int new_id,
                           IndexedList<int>& indexed_list) const {
  auto& nodes = indexed_list.GetIndex(pair);
  for (auto* node : nodes) {
    if (node->value != pair.first ||
        node->next == nullptr ||
        node->next->value != pair.second) {
      continue;
    }

    indexed_list.RemoveIndex(node);
    indexed_list.RemoveIndex(node->next);

    auto* to_remove = node->next;
    node->value = new_id;
    node->next = to_remove->next;
    if (to_remove->next) to_remove->next->prev = node;
    delete to_remove;

    indexed_list.UpdateIndex(node);
  }
}

PieceTokenizer::EncodeResult PieceTokenizer::TokenIdsToResult(
    const IndexedList<int>& token_list) const {
  EncodeResult result;

  for (auto it = token_list.begin(); it != token_list.end(); ++it) {
    auto* node = *it;
    if (!node) continue;

    int token_id = node->value;
    if (token_id >= 0 && token_id < static_cast<int>(model_->PiecesSize())) {
      const std::string& piece = model_->GetPieces(token_id).GetPiece();
      result.emplace_back(piece, token_id);
    } else if (unk_id_ >= 0) {
      const std::string& unk_piece = model_->GetPieces(unk_id_).GetPiece();
      result.emplace_back(unk_piece, unk_id_);
    }
  }

  return result;
}

}  // namespace piece
