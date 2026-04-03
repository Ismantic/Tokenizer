#include "naive_counter.h"

namespace piece {

NaiveCounter::NaiveCounter(const CounterSpec& counter_spec)
    : counter_spec_(counter_spec) {}

NaiveCounter::~NaiveCounter() = default;

bool NaiveCounter::Count() {
  LOG(INFO) << "Loading sentences ...";

  auto iter = std::make_unique<MultiFileSentenceIterator>(
      std::vector<std::string>(counter_spec_.input().begin(),
                               counter_spec_.input().end()));

  std::string combined_text;
  for (; !iter->done(); iter->Next()) {
    const std::string& line = iter->value();
    if (line.empty()) continue;
    combined_text += line;
    combined_text += "\n";
  }

  LOG(INFO) << "Loaded " << combined_text.size() << " bytes";

  merges_.clear();
  pieces_.clear();

  const int num_merges = counter_spec_.vocab_size() - INITIAL_VOCAB_SIZE;
  if (num_merges <= 0) {
    LOG(ERROR) << "Vocabulary size too small for BPE training.";
    return false;
  }

  LOG(INFO) << "Starting BPE training with " << num_merges << " merges";

  std::vector<int> ids;
  ids.reserve(combined_text.size());
  for (unsigned char c : combined_text) {
    ids.push_back(c);
  }
  merges_.reserve(num_merges);

  for (int i = 0; i < num_merges && ids.size() >= 2; ++i) {
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

    std::string first_piece, second_piece;
    DecodeToken(next.first, first_piece);
    DecodeToken(next.second, second_piece);
    std::string piece = first_piece + second_piece;

    int idx = INITIAL_VOCAB_SIZE + merges_.size();
    naive::MergePair(ids, next, idx);
    merges_.push_back({next, idx});
    pieces_.push_back({piece, first_piece, second_piece});

    LOG(INFO) << "Merge " << (i + 1) << "/" << num_merges
              << " with count " << max_count
              << " piece: " << Escape(piece);
  }

  LOG(INFO) << "BPE training completed with " << pieces_.size() << " merges";
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

  for (int id = 0; id < INITIAL_VOCAB_SIZE; ++id) {
    auto* p = model->InsertPieces();
    p->SetPiece(std::string(1, static_cast<char>(id)));
    p->SetType(Model::Piece::BYTE);
    p->SetScore(0.0);
  }

  for (size_t i = 0; i < pieces_.size(); ++i) {
    auto* p = model->InsertPieces();
    p->SetPiece(pieces_[i][0], pieces_[i][1], pieces_[i][2]);
    p->SetScore(-static_cast<float>(i));
  }

  model->SetCounterSpec(counter_spec_);
  return true;
}

void NaiveCounter::DecodeToken(int id, std::string& text) const {
  if (id < INITIAL_VOCAB_SIZE) {
    text.push_back(static_cast<char>(id));
    return;
  }

  const auto& merge = merges_[id - INITIAL_VOCAB_SIZE];
  DecodeToken(merge.pair.first, text);
  DecodeToken(merge.pair.second, text);
}

}  // namespace piece
