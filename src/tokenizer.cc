#include "tokenizer.h"

namespace piece {

Tokenizer::Tokenizer(const NormalizerSpec& spec)
    : normalizer_(spec),
      space_(spec.GetSpace()),
      cut_(spec.GetCut()) {}

Tokenizer::~Tokenizer() = default;

std::vector<std::string> Tokenizer::Tokenize(std::string_view text) const {
  std::string normalized = normalizer_.Normalize(text);
  std::vector<std::string> result;
  for (const auto& piece : ustr::SplitText(normalized, space_, cut_))
    result.emplace_back(piece);
  return result;
}

}  // namespace piece
