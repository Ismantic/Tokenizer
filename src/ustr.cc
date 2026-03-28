#include "ustr.h"

namespace ustr {

// mblen stores the number of bytes consumed after decoding.
uint32_t DecodeUTF8(const char *begin, const char *end, size_t *mblen) {
    const size_t len = end - begin;
    if (static_cast<unsigned char>(begin[0]) < 0x80) {
        *mblen = 1;
        return static_cast<unsigned char>(begin[0]);
    } else if (len >= 2 && (begin[0] & 0xE0) == 0xC0) {
        const uint32_t cp = (((begin[0] & 0x1F) << 6) | ((begin[1] & 0x3F)));
        if (IsTrailByte(begin[1]) && cp >= 0x0080 && IsValidCodepoint(cp)) {
            *mblen = 2;
            return cp;
        }
    } else if (len >= 3 && (begin[0] & 0xF0) == 0xE0) {
        const uint32_t cp = (((begin[0] & 0x0F) << 12) | ((begin[1] & 0x3F) << 6) |
                       ((begin[2] & 0x3F)));
        if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) && cp >= 0x0800 &&
                IsValidCodepoint(cp)) {
            *mblen = 3;
            return cp;
        }
    } else if (len >= 4 && (begin[0] & 0xf8) == 0xF0) {
        const uint32_t cp = (((begin[0] & 0x07) << 18) | ((begin[1] & 0x3F) << 12) |
                       ((begin[2] & 0x3F) << 6) | ((begin[3] & 0x3F)));
        if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) &&
                IsTrailByte(begin[3]) && cp >= 0x10000 && IsValidCodepoint(cp)) {
            *mblen = 4;
            return cp;
        }
    }

    // Invalid UTF-8
    *mblen = 1;
    return UnicodeError;
}

bool IsStructurallyValid(std::string_view str) {
    const char *begin = str.data();
    const char *end = str.data() + str.size();
    size_t mblen = 0;
    while (begin < end) {
        const uint32_t c = DecodeUTF8(begin, end, &mblen);
        if (c == UnicodeError && mblen != 3) return false;
        if (!IsValidCodepoint(c)) return false;
        begin += mblen;
    }
    return true;
}

size_t EncodeUTF8(uint32_t c, char *output) {
  if (c <= 0x7F) {
    *output = static_cast<char>(c);
    return 1;
  }

  if (c <= 0x7FF) {
    output[1] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[0] = 0xC0 | c;
    return 2;
  }

  // if `c` is out-of-range, convert it to REPLACEMENT CHARACTER (U+FFFD).
  // This treatment is the same as the original runetochar.
  if (c > 0x10FFFF) c = UnicodeError;

  if (c <= 0xFFFF) {
    output[2] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[1] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[0] = 0xE0 | c;
    return 3;
  }

  output[3] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[2] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[1] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[0] = 0xF0 | c;

  return 4;
}

UnicodeText UTF8ToUnicodeText(std::string_view utf8) {
    UnicodeText uc;
    const char *begin = utf8.data();
    const char *end = utf8.data() + utf8.size();
    while (begin < end) {
        size_t mblen;
        const uint32_t c = DecodeUTF8(begin, end, &mblen);
        uc.push_back(c);
        begin += mblen;
    }
    return uc;
}

std::string UnicodeTextToUTF8(const UnicodeText &utext) {
    char buf[8];
    std::string result;
    for (const uint32_t c : utext) {
        const size_t mblen = EncodeUTF8(c, buf);
        result.append(buf, mblen);
    }
    return result;
}

bool IsDigitToken(std::string_view text) {
    if (text.size() == 1 && text[0] >= '0' && text[0] <= '9') return true;

    if (text.size() == 3 &&
        static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBC &&
        static_cast<unsigned char>(text[2]) >= 0x90 &&
        static_cast<unsigned char>(text[2]) <= 0x99) {
        return true;
    }

    return false;
}

bool IsPunctuationToken(std::string_view text) {
    if (text.size() == 1) {
        char c = text[0];
        if (c == ',' || c == '.' || c == '!' || c == '?' || c == ';' ||
            c == ':' || c == '"' || c == '\'' || c == '(' || c == ')' ||
            c == '[' || c == ']' || c == '{' || c == '}' || c == '-' ||
            c == '_' || c == '+' || c == '=' || c == '/' || c == '\\' ||
            c == '<' || c == '>' || c == '~' || c == '@' || c == '#' ||
            c == '$' || c == '%' || c == '^' || c == '&' || c == '*') {
            return true;
        }
    }

    if (text.size() == 3) {
        unsigned char b1 = static_cast<unsigned char>(text[0]);
        unsigned char b2 = static_cast<unsigned char>(text[1]);
        unsigned char b3 = static_cast<unsigned char>(text[2]);

        if (b1 == 0xE3 && b2 == 0x80 && (b3 >= 0x80 && b3 <= 0xBF)) return true;
        if (b1 == 0xEF && b2 == 0xBC && (b3 >= 0x81 && b3 <= 0xBF)) return true;
        if (b1 == 0xEF && b2 == 0xBD && (b3 >= 0x80 && b3 <= 0x9F)) return true;
        if (b1 == 0xE3 && b2 == 0x80 && (b3 >= 0x89 && b3 <= 0x9F)) return true;

        if (b1 == 0xE3 && b2 == 0x80 && b3 == 0x82) return true;
        if (b1 == 0xEF && b2 == 0xBC && b3 == 0x8C) return true;
        if (b1 == 0xEF && b2 == 0xBC && b3 == 0x9A) return true;
        if (b1 == 0xEF && b2 == 0xBC && b3 == 0x9B) return true;
        if (b1 == 0xEF && b2 == 0xBC && b3 == 0x9C) return true;
        if (b1 == 0xEF && b2 == 0xBC && b3 == 0x9E) return true;
        if (b1 == 0xEF && b2 == 0xBC && b3 == 0x81) return true;
        if (b1 == 0xEF && b2 == 0xBC && b3 == 0x9F) return true;
        if (b1 == 0xE3 && b2 == 0x80 && b3 == 0x94) return true;
        if (b1 == 0xE3 && b2 == 0x80 && b3 == 0x95) return true;
        if (b1 == 0xE3 && b2 == 0x80 && b3 == 0x96) return true;
    }

    return false;
}

std::vector<std::string_view> SplitText(std::string_view text, const std::string_view space) {
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    std::vector<std::string_view> result;

    if (begin >= end) return result;

    result.emplace_back(begin, 0);
    
    while (begin < end) {
        const int mblen = std::min<int>(ustr::OneUTF8Size(begin), end-begin);
        std::string_view current_char(begin, mblen);
        
        if (current_char == space) {
            result.emplace_back(begin, mblen);
            begin += mblen;
            continue;
        }
        
        if (IsSeparatorToken(current_char)) {
            if (result.back().size() == 0) {
                result.back() = std::string_view(begin, mblen);
            } else {
                result.emplace_back(begin, mblen);
            }
            
            result.emplace_back(begin + mblen, 0);
            begin += mblen;
            continue;
        } 

        result.back() = std::string_view(
            result.back().data(),
            result.back().size() + mblen
        );
        
        begin += mblen;
    }
    
    result.erase(
        std::remove_if(result.begin(), result.end(), 
            [](const std::string_view& s) { return s.empty(); }),
        result.end()
    );
    
    return result;
}

std::vector<std::string_view> SplitWords(std::string_view text) {
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const char* word_start = nullptr;
    std::vector<std::string_view> result;

    auto flush = [&]() {
        if (word_start && begin > word_start) {
            result.emplace_back(word_start, begin - word_start);
        }
        word_start = nullptr;
    };

    while (begin < end) {
        unsigned char c = static_cast<unsigned char>(*begin);
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            flush();
            ++begin;
            continue;
        }

        int mblen = static_cast<int>(UTF8CharLen(c));
        if (begin + mblen > end) {
            mblen = static_cast<int>(end - begin);
        }

        std::string_view current(begin, mblen);
        if (IsSeparatorToken(current)) {
            flush();
            result.push_back(current);
            begin += mblen;
            continue;
        }

        if (!word_start) {
            word_start = begin;
        }
        begin += mblen;
    }

    flush();
    return result;
}

} // namespace
