#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "normalizer.h"
#include "piece_spec.h"
#include "ustr.h"

namespace piece {

// Pre-tokenizer: Normalize + SplitText.
// Standalone interface for splitting text into tokens without a trained model.
class Tokenizer {
public:
  explicit Tokenizer(const NormalizerSpec& spec);
  ~Tokenizer();

  std::vector<std::string> Tokenize(std::string_view text) const;

private:
  Normalizer normalizer_;
  std::string space_;
  int cut_;
};

}  // namespace piece
