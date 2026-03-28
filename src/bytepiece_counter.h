#pragma once
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <memory>
#include <algorithm>
#include <stdint.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include "bytepiece_tokenizer.h"
#include "piece_spec.h"
#include "common.h"
#include "ustr.h"
#include "misc.h"
#include "sentence.h"
#include "normalizer.h"

namespace piece {
class BytePieceCounter {
public:
    BytePieceCounter(const CounterSpec& counter_spec,
                   const NormalizerSpec& normalizer_spec);
    ~BytePieceCounter();
    
    bool Count();
    bool Save() const;
    bool Serialize(Model* model_proto) const;
    
private:
    using Str2Int = std::unordered_map<std::string, int>;

    size_t GetWorkerCount(size_t work_items) const;

    template <typename Map>
    static void MergeCounts(Map* dest, const Map& src) {
        for (const auto& [key, value] : src) {
            (*dest)[key] += value;
        }
    }

    template <typename Map, typename CountFn>
    Map ParallelBatchCount(size_t total, size_t batch_size, CountFn&& count_fn) const {
        Map merged;
        if (total == 0) {
            return merged;
        }

        const size_t workers = GetWorkerCount((total + batch_size - 1) / batch_size);
        if (workers <= 1) {
            return count_fn(0, total);
        }

        std::atomic<size_t> next_begin{0};
        std::mutex merge_mu;
        std::vector<std::thread> threads;
        threads.reserve(workers);

        for (size_t worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&, worker]() {
                while (true) {
                    const size_t begin = next_begin.fetch_add(batch_size);
                    if (begin >= total) {
                        break;
                    }
                    const size_t end = std::min(total, begin + batch_size);
                    Map batch_counts = count_fn(begin, end);
                    std::lock_guard<std::mutex> lock(merge_mu);
                    MergeCounts(&merged, batch_counts);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        return merged;
    }
    
    bool InitMetaPieces();
    std::unique_ptr<MultiFileSentenceIterator> MakeIterator() const;
    static bool IsSeparator(const char* p, int len);

    template<typename Fn>
    static void ForEachWord(const std::string& line, Fn&& fn) {
        const char* p = line.c_str();
        const char* end = p + line.size();
        const char* word_start = nullptr;

        auto flush = [&]() {
            if (word_start && p > word_start) {
                fn(std::string_view(word_start, p - word_start));
            }
            word_start = nullptr;
        };

        while (p < end) {
            unsigned char c = static_cast<unsigned char>(*p);

            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                flush();
                ++p;
                continue;
            }

            int mblen = SizeUTF8(c);
            if (p + mblen > end) mblen = static_cast<int>(end - p);

            if (IsSeparator(p, mblen)) {
                flush();
                fn(std::string_view(p, mblen));
                p += mblen;
                continue;
            }

            if (!word_start) word_start = p;
            p += mblen;
        }
        flush();
    }

    bool StreamCountRaw();
    void PruneRaw();
    std::vector<std::string> Tokenize(const std::string& text) const;
    Str2Int StreamCountPieces();
    Str2Int SplitPieces(const Str2Int& keep, const Str2Int& drop);
    Str2Int PrunePieces(Str2Int& pieces);
    void InitT();
    
    std::map<int, std::pair<std::string, Model::Piece::Type>> meta_pieces_;
    std::vector<std::pair<std::string, float>> pieces_;
    CounterSpec counter_spec_;
    NormalizerSpec normalizer_spec_;
    
    size_t max_piece_count_;
    size_t max_piece_size_;
    size_t min_count_;
    std::vector<std::vector<float_t>> T_;
    std::vector<std::unordered_map<std::string, float_t>> N_;
    const float_t INF = std::numeric_limits<float_t>::infinity();
};

} // namespace piece
