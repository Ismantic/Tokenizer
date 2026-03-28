#include "naive_counter.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace piece {

NaiveCounter::NaiveCounter(const CounterSpec& counter_spec,
                           const NormalizerSpec& normalizer_spec)
    : counter_spec_(counter_spec),
      normalizer_spec_(normalizer_spec) {
  InitMetaPieces();
}

NaiveCounter::~NaiveCounter() = default;

bool NaiveCounter::Count() {
  if (!LoadSentences()) {
    LOG(ERROR) << "Failed to load sentences.";
    return false;
  }

  SplitSentencesByWhitespace();

  std::string combined_text;
  for (const auto& sentence : sentences_) {
    combined_text += sentence.first;
    combined_text += " ";
  }

  InitializeVocab();

  const int num_merges =
      counter_spec_.vocab_size() - INITAL_VOCAB_SIZE - meta_pieces_.size();
  if (num_merges <= 0) {
    LOG(ERROR) << "Vocabulary size too small for BPE training.";
    return false;
  }

  LOG(INFO) << "Starting BPE training with " << num_merges << " merges";

  std::vector<int> ids(combined_text.begin(), combined_text.end());
  merges_.reserve(num_merges);

  int valid_utf8_merges = 0;
  const int max_iterations = num_merges * 3;
  for (int i = 0;
       i < max_iterations && ids.size() >= 2 && valid_utf8_merges < num_merges;
       ++i) {
    std::vector<size_t> pair_counts(MAX_TEXT_SIZE * 3);
    size_t pair_counts_size;
    naive::TokenCounts(ids, pair_counts, pair_counts_size);

    size_t max_count = 0;
    IntPair next{0, 0};

    for (size_t j = 0; j < pair_counts_size * 3; j += 3) {
      IntPair pair{static_cast<int>(pair_counts[j]),
                   static_cast<int>(pair_counts[j + 1])};
      size_t count = pair_counts[j + 2];
      if (count > max_count) {
        max_count = count;
        next = pair;
      }
    }

    if (max_count <= 1) break;

    std::string piece;
    const int first_id = next.first;
    const int second_id = next.second;

    std::string first_piece;
    if (first_id < INITAL_VOCAB_SIZE) {
      first_piece.push_back(static_cast<char>(first_id));
    } else {
      for (const auto& merge : merges_) {
        if (merge.idx == first_id) {
          std::string temp;
          DecodeToken(merge.pair.first, temp);
          DecodeToken(merge.pair.second, temp);
          first_piece = temp;
          break;
        }
      }
    }

    std::string second_piece;
    if (second_id < INITAL_VOCAB_SIZE) {
      second_piece.push_back(static_cast<char>(second_id));
    } else {
      for (const auto& merge : merges_) {
        if (merge.idx == second_id) {
          std::string temp;
          DecodeToken(merge.pair.first, temp);
          DecodeToken(merge.pair.second, temp);
          second_piece = temp;
          break;
        }
      }
    }

    piece = first_piece + second_piece;

    const bool is_valid_utf8 = ustr::IsStructurallyValid(piece);

    if (!is_valid_utf8) {
      int idx = INITAL_VOCAB_SIZE + merges_.size();
      naive::MergePair(ids, next, idx);
      merges_.push_back({next, idx});
      vocab_.push_back(
          {static_cast<uint8_t>(next.first), static_cast<uint8_t>(next.second)});
      continue;
    }

    valid_utf8_merges++;
    int idx = INITAL_VOCAB_SIZE + merges_.size();
    naive::MergePair(ids, next, idx);
    merges_.push_back({next, idx});
    vocab_.push_back(
        {static_cast<uint8_t>(next.first), static_cast<uint8_t>(next.second)});
    pieces_.emplace_back(piece, -static_cast<float>(pieces_.size()));

    LOG(INFO) << "Merge " << valid_utf8_merges << "/" << num_merges
              << " with count " << max_count
              << " piece: " << piece
              << " (valid UTF-8)";
  }

  LOG(INFO) << "BPE training completed with " << valid_utf8_merges
            << " valid UTF-8 merges";
  return true;
}

bool NaiveCounter::Save() const {
  const std::string filename = counter_spec_.model_prefix() + ".model";
  LOG(INFO) << "Saving model: " << filename;
  Model model;
  if (!Serialize(&model)) return false;
  auto output = NewWritableFile(filename);
  output->Write(model.AsStr());
  return true;
}

bool NaiveCounter::Serialize(Model* model) const {
  model->Clear();

  for (int id = 0; id < counter_spec_.vocab_size(); ++id) {
    const auto it = meta_pieces_.find(id);
    if (it != meta_pieces_.end()) {
      auto* p = model->InsertPieces();
      p->SetPiece(it->second.first);
      p->SetType(it->second.second);
      p->SetScore(0.0);
    } else if (id < INITAL_VOCAB_SIZE) {
      std::string byte_piece(1, static_cast<char>(id));
      auto* p = model->InsertPieces();
      p->SetPiece(byte_piece);
      p->SetType(Model::Piece::BYTE);
      p->SetScore(0.0);
    } else if (id - INITAL_VOCAB_SIZE < static_cast<int>(pieces_.size())) {
      const auto& piece = pieces_[id - INITAL_VOCAB_SIZE];
      auto* p = model->InsertPieces();
      p->SetPiece(piece.first);
      p->SetScore(piece.second);
    }
  }

  model->SetCounterSpec(counter_spec_);
  model->SetNormalizerSpec(normalizer_spec_);
  return true;
}

void NaiveCounter::InitializeVocab() {
  vocab_.resize(INITAL_VOCAB_SIZE);
  for (int i = 0; i < INITAL_VOCAB_SIZE; ++i) {
    vocab_[i] = std::vector<uint8_t>{static_cast<unsigned char>(i)};
  }
  merges_.clear();
  pieces_.clear();
}

void NaiveCounter::DecodeToken(int id, std::string& text) const {
  if (id < INITAL_VOCAB_SIZE) {
    text.push_back(static_cast<char>(id));
    return;
  }

  const auto& merge = merges_[id - INITAL_VOCAB_SIZE];
  DecodeToken(merge.pair.first, text);
  DecodeToken(merge.pair.second, text);
}

bool NaiveCounter::InitMetaPieces() {
  if (counter_spec_.unk_id() >= 0) {
    meta_pieces_[counter_spec_.unk_id()] = std::make_pair(
        counter_spec_.unk_piece(), Model::Piece::UNKNOWN);
  }
  if (counter_spec_.bos_id() >= 0) {
    meta_pieces_[counter_spec_.bos_id()] = std::make_pair(
        counter_spec_.bos_piece(), Model::Piece::CONTROL);
  }
  if (counter_spec_.eos_id() >= 0) {
    meta_pieces_[counter_spec_.eos_id()] = std::make_pair(
        counter_spec_.eos_piece(), Model::Piece::CONTROL);
  }
  if (counter_spec_.pad_id() >= 0) {
    meta_pieces_[counter_spec_.pad_id()] = std::make_pair(
        counter_spec_.pad_piece(), Model::Piece::CONTROL);
  }
  return true;
}

bool NaiveCounter::LoadSentences() {
  LOG(INFO) << "Loading sentences ...";

  auto iter_ = std::make_unique<MultiFileSentenceIterator>(
      std::vector<std::string>(counter_spec_.input().begin(),
                               counter_spec_.input().end()));
  SentenceIterator* sentence_iterator_ = iter_.get();

  for (; !sentence_iterator_->done(); sentence_iterator_->Next()) {
    std::string sentence = sentence_iterator_->value();
    if (sentence.empty()) {
      continue;
    }
    sentences_.emplace_back(std::make_pair(sentence, 1));
  }

  LOG(INFO) << "Normalizing sentences ...";
  const Normalizer normalizer(normalizer_spec_);
  for (size_t i = 0; i < sentences_.size(); ++i) {
    auto* s = &sentences_[i].first;
    *s = normalizer.Normalize(*s);
  }

  LOG(INFO) << "Done! preprocessed " << sentences_.size() << " sentences.";
  return true;
}

void NaiveCounter::SplitSentencesByWhitespace() {
  LOG(INFO) << "Tokenizing input sentences with whitespace: "
            << sentences_.size();
  const std::string_view space = normalizer_spec_.GetSpace();
  std::unordered_map<std::string, int64_t> tokens;
  for (const auto& s : sentences_) {
    for (const auto& w : ustr::SplitText(s.first, space)) {
      tokens[std::string(w)] += s.second;
    }
  }
  sentences_ = misc::Sorted(tokens);
  LOG(INFO) << "Done! " << sentences_.size();
}

}  // namespace piece
