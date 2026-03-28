#include "piece_counter.h"

#include <iostream>

namespace piece {

PieceCounter::PieceCounter(const CounterSpec& counter_spec,
                           const NormalizerSpec& normalizer_spec)
    : counter_spec_(counter_spec),
      normalizer_spec_(normalizer_spec),
      vocab_size_(counter_spec.vocab_size()) {
  InitMetaPieces();
}

PieceCounter::~PieceCounter() = default;

bool PieceCounter::Count() {
  if (!LoadSentences()) {
    LOG(ERROR) << "Failed to load sentences.";
    return false;
  }

  SplitSentencesByWhitespace();

  std::vector<std::string> texts;
  for (const auto& sentence : sentences_) {
    texts.push_back(sentence.first);
  }

  merge_tree_.clear();
  auto stats = InitPairsStats(texts);
  std::vector<IndexedList<int>> indexed_lists;
  indexed_lists.reserve(texts.size());
  for (const auto& text : texts) {
    indexed_lists.push_back(BuildIndexedList(text));
  }

  const int num_merges = vocab_size_ - meta_pieces_.size();
  LOG(INFO) << "Starting BBPE training with " << num_merges << " merges";

  int cnt = 0;
  for (int i = 0; i < 256; i++) {
    std::string t(1, i);
    vocab_[i] = t;
    std::vector<std::string> vec = {t, "", ""};
    pieces_.emplace_back(std::make_pair(vec, 1));
    cnt += 1;
  }

  while (cnt < num_merges && stats) {
    auto top = stats.Top();
    const int n = stats.GetCount(top);
    const int new_id = vocab_.size();
    merge_tree_.emplace_back(top, new_id);
    vocab_[new_id] = vocab_[top.first] + vocab_[top.second];

    std::string p = vocab_[new_id];
    std::string u = vocab_[top.first];
    std::string v = vocab_[top.second];
    std::vector<std::string> vec = {p, u, v};
    pieces_.emplace_back(std::make_pair(vec, n));

    for (auto& list : indexed_lists) {
      Merge(top, new_id, list, &stats);
    }

    if (cnt % 10 == 0) {
      LOG(INFO) << "Merge " << cnt + 1 << "/" << num_merges
                << ": (" << top.first << "," << top.second
                << ") -> " << new_id << " (" << vocab_[new_id]
                << ") had " << n << " occurrences";
    }

    cnt += 1;
  }
  LOG(INFO) << "Done! " << cnt << " merges";

  return true;
}

bool PieceCounter::Save() const {
  const std::string filename = counter_spec_.model_prefix() + ".model";
  LOG(INFO) << "Saving model: " << filename;
  Model model;
  if (!Serialize(&model)) return false;
  auto output = NewWritableFile(filename);
  output->Write(model.AsStr());
  return true;
}

bool PieceCounter::Serialize(Model* model) const {
  model->Clear();

  size_t fid = 0;
  for (int id = 0; id < counter_spec_.vocab_size(); ++id) {
    const auto it = meta_pieces_.find(id);
    if (it != meta_pieces_.end()) {
      auto* p = model->InsertPieces();
      p->SetPiece(it->second.first);
      p->SetType(it->second.second);
      p->SetScore(0.0);
    } else if (fid < pieces_.size()) {
      const auto x = pieces_[fid++];
      const auto& w = x.first;
      const auto s = x.second;
      auto* piece = model->InsertPieces();
      piece->SetPiece(w[0], w[1], w[2]);
      piece->SetScore(s);
    } else {
      LOG(ERROR) << "Invalid piece id: " << id;
      return false;
    }
  }

  model->SetCounterSpec(counter_spec_);
  model->SetNormalizerSpec(normalizer_spec_);
  return true;
}

bool PieceCounter::InitMetaPieces() {
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

  const int byte_id = meta_pieces_.size();
  if (byte_id + 256 > counter_spec_.vocab_size()) {
    LOG(ERROR) << "Vocabulary size too small for byte_fallback. Need at least "
               << (byte_id + 256) << " slots.";
    return false;
  }

  return true;
}

bool PieceCounter::LoadSentences() {
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

void PieceCounter::SplitSentencesByWhitespace() {
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

IndexedList<int> PieceCounter::BuildIndexedList(const std::string& text) {
  std::vector<int> bytes;
  for (unsigned char c : text) {
    bytes.push_back(static_cast<int>(c));
  }
  return IndexedList<int>(bytes.begin(), bytes.end());
}

Multiset<std::pair<int, int>> PieceCounter::InitPairsStats(
    const std::vector<std::string>& texts) {
  Multiset<std::pair<int, int>> stats;
  std::vector<int> bytes;
  for (const auto& text : texts) {
    bytes.clear();
    for (uint8_t c : text) {
      bytes.push_back(static_cast<int>(c));
    }
    for (size_t i = 0; i + 1 < bytes.size(); i++) {
      stats.Insert({bytes[i], bytes[i + 1]});
    }
  }
  return stats;
}

void PieceCounter::Merge(const std::pair<int, int>& pair,
                         int new_id,
                         IndexedList<int>& indexed_list,
                         Multiset<std::pair<int, int>>* stats) {
  auto& nodes = indexed_list.GetIndex(pair);
  for (auto* node : nodes) {
    if (node->value != pair.first ||
        node->next == nullptr ||
        node->next->value != pair.second) {
      continue;
    }

    indexed_list.RemoveIndex(node);
    indexed_list.RemoveIndex(node->next);

    if (stats != nullptr) {
      stats->Remove(pair);
      if (node->next->next != nullptr) {
        stats->Remove({node->next->value, node->next->next->value});
        stats->Insert({new_id, node->next->next->value});
      }
      if (node->prev != nullptr) {
        stats->Remove({node->prev->value, pair.first});
        stats->Insert({node->prev->value, new_id});
      }
    }

    auto* remove = node->next;
    node->next->Delete();
    delete remove;
    node->value = new_id;
    indexed_list.UpdateIndex(node);
  }
}

}  // namespace piece
