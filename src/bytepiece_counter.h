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
#include <future>
#include <atomic>
#include <mutex>
#include "bytepiece_tokenizer.h"
#include "piece_spec.h"
#include "common.h"
#include "ustr.h"
#include "misc.h"
#include "sentence.h"
#include "new_normalizer.h"

namespace piece {
class BytePieceCounter {
public:
    BytePieceCounter(const CounterSpec& counter_spec,
                   const NormalizerSpec& normalizer_spec)
        : counter_spec_(counter_spec),
          normalizer_spec_(normalizer_spec),
          max_piece_count_(6),
          max_piece_size_(18),
          min_count_(2) {
        InitMetaPieces();
        InitT();
        N_.resize(max_piece_count_ + 1);
    }
    
    virtual ~BytePieceCounter() {}
    
    bool Count() {
        if (!StreamCountRaw()) {
            LOG(ERROR) << "Failed to count raw substrings.";
            return false;
        }
        PruneRaw();

        auto pieces_count = StreamCountPieces();
        auto pruned_pieces = PrunePieces(pieces_count);

        std::vector<std::pair<std::string, int>> sorted_pieces(pruned_pieces.begin(), pruned_pieces.end());
        std::sort(sorted_pieces.begin(), sorted_pieces.end(), 
                  [](const auto& a, const auto& b) {
                      return a.second > b.second;  // 降序排序（大的在前）
                  });

        pieces_.clear();

        for (const auto& [piece, count] : sorted_pieces) {
            pieces_.emplace_back(piece, count);
            if (pieces_.size() % 100 == 0) {
                LOG(INFO) << "Added piece: " << piece << " count: " << count
                          << " total: " << pieces_.size();
            }
        }
        
        LOG(INFO) << "BytePiece training completed with " << pieces_.size() << " pieces";
        
        return true;
    }
    
    bool Save() const {
        std::string filename = counter_spec_.model_prefix()+".model";
        LOG(INFO) << "Saving model: " << filename;
        Model model;
        if (!Serialize(&model)) return false;
        auto output = NewWritableFile(filename);
        output->Write(model.AsStr());
        return true;
    }
    
    bool Serialize(Model* model) const {
        model->Clear();

        // Total pieces = meta_pieces + trained pieces
        size_t total = meta_pieces_.size() + pieces_.size();
        size_t fid = 0;
        for (size_t id = 0; id < total; ++id) {
            const auto it = meta_pieces_.find(id);
            if (it != meta_pieces_.end()) {
                auto *p = model->InsertPieces();
                p->SetPiece(it->second.first);
                p->SetType(it->second.second);
                p->SetScore(0.0);
            } else if (fid < pieces_.size()) {
                const auto &w = pieces_[fid++];
                auto *p = model->InsertPieces();
                p->SetPiece(w.first);
                p->SetScore(w.second);
            }
        }

        // Update vocab_size in counter_spec to match actual count
        CounterSpec spec = counter_spec_;
        spec.set_vocab_size(total);
        model->SetCounterSpec(spec);
        model->SetNormalizerSpec(normalizer_spec_);

        return true;
    }
    
private:
    using Str2Int = std::unordered_map<std::string, int>;
    using Str2Float = std::unordered_map<std::string, float_t>;

    size_t GetWorkerCount(size_t work_items) const {
        if (work_items <= 1) {
            return 1;
        }
        const unsigned int hw = std::thread::hardware_concurrency();
        const size_t max_workers = hw == 0 ? 4 : static_cast<size_t>(hw);
        return std::max<size_t>(1, std::min(work_items, max_workers));
    }

    template <typename Fn>
    void ParallelFor(size_t total, Fn&& fn) const {
        const size_t workers = GetWorkerCount(total);
        if (workers <= 1) {
            fn(0, total, 0);
            return;
        }

        std::vector<std::thread> threads;
        threads.reserve(workers);
        const size_t chunk = (total + workers - 1) / workers;
        for (size_t worker = 0; worker < workers; ++worker) {
            const size_t begin = worker * chunk;
            const size_t end = std::min(total, begin + chunk);
            if (begin >= end) {
                break;
            }
            threads.emplace_back([&, begin, end, worker]() {
                fn(begin, end, worker);
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

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
    
    bool InitMetaPieces() {
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

        if (meta_pieces_.size() + 256 > counter_spec_.vocab_size()) {
            LOG(ERROR) << "Vocab size is too small for byte pieces. "
                       << "Need at least " << meta_pieces_.size() + 256 << " slots." << std::endl;
            return false;
        }

        int byte_id = meta_pieces_.size();
        for (int i = 0; i < 256; ++i) {
            std::string byte_piece = ustr::ByteToPiece(i);
            meta_pieces_[byte_id++] = std::make_pair(byte_piece, Model::Piece::BYTE);
        }

        return true;
    }
    
    std::unique_ptr<MultiFileSentenceIterator> MakeIterator() const {
        return std::make_unique<MultiFileSentenceIterator>(
            std::vector<std::string>(counter_spec_.input().begin(),
                                     counter_spec_.input().end()));
    }

    static bool IsSeparator(const char* p, int len) {
        if (len == 1) {
            unsigned char c = static_cast<unsigned char>(p[0]);
            if (c >= '0' && c <= '9') return true;
            if (c == ',' || c == '.' || c == '!' || c == '?' || c == ';' ||
                c == ':' || c == '"' || c == '\'' || c == '(' || c == ')' ||
                c == '[' || c == ']' || c == '{' || c == '}' || c == '-' ||
                c == '_' || c == '+' || c == '=' || c == '/' || c == '\\' ||
                c == '<' || c == '>' || c == '~' || c == '@' || c == '#' ||
                c == '$' || c == '%' || c == '^' || c == '&' || c == '*')
                return true;
        }
        if (len == 3) {
            unsigned char b1 = static_cast<unsigned char>(p[0]);
            unsigned char b2 = static_cast<unsigned char>(p[1]);
            unsigned char b3 = static_cast<unsigned char>(p[2]);
            if (b1 == 0xEF && b2 == 0xBC && b3 >= 0x90 && b3 <= 0x99) return true;
            if (b1 == 0xE3 && b2 == 0x80 && b3 >= 0x80 && b3 <= 0xBF) return true;
            if (b1 == 0xEF && b2 == 0xBC && b3 >= 0x81 && b3 <= 0xBF) return true;
            if (b1 == 0xEF && b2 == 0xBD && b3 >= 0x80 && b3 <= 0x9F) return true;
        }
        return false;
    }

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

    bool StreamCountRaw() {
        const size_t workers = std::min<size_t>(
            std::thread::hardware_concurrency(), 8);
        LOG(INFO) << "Pass 1: counting substrings with " << workers << " workers...";

        using NMaps = std::vector<std::unordered_map<std::string, float_t>>;

        N_.clear();
        N_.resize(max_piece_count_ + 1);

        auto iter = MakeIterator();
        size_t line_count = 0;
        constexpr size_t kBatchLines = 1000000;

        // Fill a batch of words from the iterator (called from reader thread only)
        auto fill_batch = [&](std::vector<std::string>& batch) -> size_t {
            batch.clear();
            size_t lines = 0;
            for (; !iter->done() && lines < kBatchLines; iter->Next(), ++lines) {
                const std::string& line = iter->value();
                if (line.empty()) continue;
                ForEachWord(line, [&batch](std::string_view word) {
                    batch.emplace_back(word);
                });
            }
            return lines;
        };

        // Process a batch: parallel count into per-thread maps, merge into N_
        auto process_batch = [&](std::vector<std::string>& batch) {
            if (batch.empty()) return;
            std::vector<NMaps> per_thread(workers);
            for (auto& nm : per_thread) nm.resize(max_piece_count_ + 1);

            std::vector<std::thread> threads;
            const size_t chunk = (batch.size() + workers - 1) / workers;
            for (size_t w = 0; w < workers; ++w) {
                const size_t begin = w * chunk;
                const size_t end = std::min(batch.size(), begin + chunk);
                if (begin >= end) break;
                threads.emplace_back([&, begin, end, w]() {
                    auto& local_N = per_thread[w];
                    for (size_t idx = begin; idx < end; ++idx) {
                        const auto& text = batch[idx];
                        for (size_t i = 0; i < text.length(); ++i) {
                            local_N[0][""] += 1;
                            std::string piece;
                            const size_t max_len = std::min(
                                max_piece_count_, text.length() - i);
                            for (size_t j = 1; j <= max_len; ++j) {
                                piece.push_back(text[i + j - 1]);
                                local_N[j][piece] += 1;
                            }
                        }
                    }
                });
            }
            for (auto& t : threads) t.join();

            for (auto& local_N : per_thread) {
                for (size_t i = 0; i <= max_piece_count_; ++i) {
                    for (auto& [k, v] : local_N[i]) {
                        N_[i][k] += v;
                    }
                }
            }
        };

        // Double buffering: read next batch while processing current batch
        std::vector<std::string> work_batch, read_batch;
        line_count += fill_batch(work_batch);

        while (!work_batch.empty()) {
            // Async read next batch
            auto read_future = std::async(std::launch::async,
                [&]() { return fill_batch(read_batch); });

            // Process current batch on main thread (8 workers)
            process_batch(work_batch);

            // Wait for next batch to be ready, swap buffers
            line_count += read_future.get();
            LOG(INFO) << "Pass 1: " << line_count << " lines";
            std::swap(work_batch, read_batch);
        }

        size_t cnt = 0;
        for (size_t i = 0; i < N_.size(); ++i) cnt += N_[i].size();
        LOG(INFO) << "Pass 1 done: " << cnt << " entries";
        return true;
    }
    
    void PruneRaw() {
        LOG(INFO) << "Pruning raw counts...";
        
        // 确保所有字节值都在N_[1]中，与Python版本一致
        for (int i = 0; i < 256; ++i) {
            std::string byte_str(1, static_cast<char>(i));
            if (N_[1].find(byte_str) == N_[1].end()) {
                N_[1][byte_str] = 1;
                N_[0][""] += 1;  // 增加空字符串的计数
            }
        }
        
        // 从最长的n-gram开始向下迭代
        for (int i = N_.size() - 1; i >= 0; --i) {
            std::unordered_map<std::string, float_t> pruned;
            
            // 只保留长度正好为i且频率超过阈值的n-gram，并取对数
            for (const auto& [k, v] : N_[i]) {
                if (k.length() == i && v >= (i > 1 ? min_count_ : 0)) {
                    pruned[k] = std::log(v);
                }
            }
            
            // 对于长度i+1的n-gram，减去其前i个字符的对数频率
            if (i < N_.size() - 1) {
                std::unordered_map<std::string, float_t> next_pruned;
                
                for (const auto& [k, v] : N_[i + 1]) {
                    std::string prefix = k.substr(0, i);
                        
                    auto it = pruned[prefix];
                    // 减去前缀的对数频率
                    next_pruned[k] = v - it;
                }
                
                N_[i + 1] = std::move(next_pruned);
            }
            
            // 用修剪后的结果替换原始统计
            N_[i] = std::move(pruned);
        }
        
        int cnt = 0;
        for (size_t i = 0; i < N_.size(); ++i) {
            cnt += N_[i].size();
        }
        LOG(INFO) << "Done pruning raw counts " << cnt;
    }

    std::vector<std::string> Tokenize(const std::string& text) const {
        const int num = text.length();
        if (num == 0) return {};
        
        // 创建并初始化节点评分矩阵
        std::vector<std::vector<float_t>> nodes(num, std::vector<float_t>(max_piece_count_, -INF));
        
        // 预处理UTF-8字符结构
        std::vector<int> utf8_position(num, 0); // 0=首字节或ASCII, 1=第二字节, 2=第三字节, 3=第四字节
        
        int i = 0;
        while (i < num) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            int char_length = 1; // 默认为1个字节（ASCII字符）
            
            if (c < 128) {
                // ASCII字符，长度为1
                char_length = 1;
            } else if (c >= 192 && c < 224) {
                // 2字节UTF-8字符
                char_length = 2;
            } else if (c >= 224 && c < 240) {
                // 3字节UTF-8字符
                char_length = 3;
            } else if (c >= 240 && c < 248) {
                // 4字节UTF-8字符
                char_length = 4;
            } else if (c >= 128 && c < 192) {
                // 这是一个UTF-8后续字节，但没有前导首字节
                // 我们将其视为单个字节继续处理
                char_length = 1;
            }
            
            // 处理这个UTF-8字符的所有字节
            for (int j = 0; j < char_length && i + j < num; ++j) {
                utf8_position[i + j] = j;
            }
            
            // 移动到下一个字符
            i += char_length;
        }
        
        // 填充节点评分矩阵
        for (int j = 0; j < max_piece_count_; ++j) {
            for (int i = j; i < num; ++i) {
                // 对于j=0，确保位置i是UTF-8首字节
                if (j == 0 && utf8_position[i] > 0) {
                    continue;  // 跳过UTF-8后续字节
                }
                
                std::string piece = text.substr(i - j, j + 1);
                if (j + 1 < N_.size()) {
                    auto it = N_[j + 1].find(piece);
                    if (it != N_[j + 1].end()) {
                        nodes[i][j] = it->second;
                    }
                }
            }
        }
        
        // 创建路由矩阵
        std::vector<std::vector<int>> routes(num - 1, std::vector<int>(max_piece_count_, 0));
        
        // 执行动态规划算法，寻找最优路径
        for (i = 1; i < num; ++i) {
            for (int curr_j = 0; curr_j < max_piece_count_; ++curr_j) {
                // 如果当前状态对于此位置无效，跳过
                if (curr_j < utf8_position[i]) continue;
                
                int best_prev_j = -1; // 初始化为-1，表示尚未找到有效的前一个状态
                float_t best_score = -INF;
                
                for (int prev_j = 0; prev_j < max_piece_count_; ++prev_j) {
                    // 如果前一个状态对于前一个位置无效，跳过
                    if (prev_j < utf8_position[i-1]) continue;
                    
                    // 检查转移是否允许
                    if (T_[prev_j][curr_j] == -INF) continue;
                    
                   bool skip_ngram_check = (prev_j == max_piece_count_ - 1 && curr_j == max_piece_count_ - 1);
            
                   if (!skip_ngram_check) {
                       // 检查n-gram的起始位置
                       int ngram_start = i - curr_j;
                       if (ngram_start > 0 && utf8_position[ngram_start] > 0) {
                           // n-gram不能从UTF-8字符的中间开始
                           continue;
                       }
                   }
                   
                    
                    float_t score = nodes[i - 1][prev_j] + T_[prev_j][curr_j] + nodes[i][curr_j];
                    
                    if (score > best_score) {
                        best_score = score;
                        best_prev_j = prev_j;
                    }
                }
                
                if (best_prev_j != -1) {
                    routes[i - 1][curr_j] = best_prev_j;
                    nodes[i][curr_j] = best_score;
                } else {
                    // 没有找到有效的前一个状态，跳过这个状态
                    nodes[i][curr_j] = -INF;
                }
            }
        }
        
        // 找到最后一个位置的最佳状态
        int best_last_state = 0;
        float_t best_score = -INF;
        for (int j = 0; j < max_piece_count_; ++j) {
            if (j >= utf8_position[num - 1] && nodes[num - 1][j] > best_score) {
                best_score = nodes[num - 1][j];
                best_last_state = j;
            }
        }
        
        // 回溯构建最优路径
        std::vector<int> opt_route(num);
        int curr_pos = num - 1;
        int curr_state = best_last_state;
        
        while (curr_pos >= 0) {
            opt_route[curr_pos] = curr_state;
            
            if (curr_pos > 0) {
                curr_state = routes[curr_pos-1][curr_state];
                curr_pos--;
            } else {
                // Error
                break;
            }
        }
        
        // 找到所有状态为0的位置作为切分点
        std::vector<int> split_points;
        split_points.push_back(0); // 总是从文本开始处切分
        
        for (i = 1; i < opt_route.size(); ++i) {
            if (opt_route[i] == 0 && utf8_position[i] == 0) {
                // 确保只在UTF-8首字节处切分
                split_points.push_back(i);
            }
        }
        
        split_points.push_back(num); // 添加文本结束位置
        
        // 根据切分点提取标记
        std::vector<std::string> tokens;
        for (size_t i = 0; i < split_points.size() - 1; ++i) {
            tokens.push_back(text.substr(split_points[i], split_points[i + 1] - split_points[i]));
        }
        
        // 输出opt_route和分词结果（用于调试）
        //for (int i = 0; i < opt_route.size(); ++i) {
        //    std::cout << opt_route[i] << " ";
        //}
        //std::cout << "\n";
        
        //std::cout << text << "=>";
        //for (auto t : tokens) {
        //    std::cout << t << " ";
        //}
        //std::cout << "\n";
        
        return tokens;
    } 

    Str2Int StreamCountPieces() {
        LOG(INFO) << "Pass 2: counting pieces...";
        Str2Int total_pieces;

        auto iter = MakeIterator();
        size_t line_count = 0;

        std::vector<std::string> buffer;
        const size_t flush_size = 50000;

        auto flush = [&]() {
            if (buffer.empty()) return;
            Str2Int batch = ParallelBatchCount<Str2Int>(
                buffer.size(), 2048,
                [&](size_t begin, size_t end) {
                    Str2Int counts;
                    for (size_t i = begin; i < end; ++i) {
                        for (const auto& piece : Tokenize(buffer[i])) {
                            if (!piece.empty()) counts[piece] += 1;
                        }
                    }
                    return counts;
                });
            MergeCounts(&total_pieces, batch);
            buffer.clear();
        };

        for (; !iter->done(); iter->Next()) {
            const std::string& line = iter->value();
            if (line.empty()) continue;

            ForEachWord(line, [&](std::string_view word) {
                buffer.emplace_back(word);
            });

            if (buffer.size() >= flush_size) {
                flush();
            }

            if (++line_count % 1000000 == 0) {
                LOG(INFO) << "Pass 2: " << line_count << " lines, "
                          << total_pieces.size() << " pieces";
            }
        }
        flush();

        LOG(INFO) << "Pass 2 done: " << total_pieces.size() << " unique pieces";
        return total_pieces;
    }


    Str2Int SplitPieces(const Str2Int& keep, const Str2Int& drop) {
        std::unordered_map<std::string, float_t> dict;
        for (const auto& p : keep) {
            dict.emplace(p.first, static_cast<float_t>(p.second));
        }
        
        BytePieceTokenizer tokenizer(dict);

        std::vector<std::pair<std::string, int>> drop_entries(drop.begin(), drop.end());
        const size_t batch_size = 1024;
        Str2Int counter = ParallelBatchCount<Str2Int>(
            drop_entries.size(), batch_size,
            [&](size_t begin, size_t end) {
                Str2Int batch_counts;
                for (size_t idx = begin; idx < end; ++idx) {
                    const auto& [str, cnt] = drop_entries[idx];
                    for (const auto& token : tokenizer.Tokenize(str)) {
                        batch_counts[token] += cnt;
                    }
                }
                return batch_counts;
            });
        return counter;
    }

   
    Str2Int PrunePieces(Str2Int& pieces) {
        LOG(INFO) << "Pruning pieces...";
       
        Str2Int keep, drop;
        
        for (const auto& [str, cnt] : pieces) {
            if (str.length() == 1 ||
                (str.length() <= max_piece_size_ && cnt >= min_count_)) {
                keep[str] = cnt;
            } else {
                drop[str] = cnt;
            }
        }
        
        auto new_counter = SplitPieces(keep, drop);
        for (const auto& [str, cnt] : new_counter) {
            keep[str] += cnt;
        }
        
        while (true) {
            size_t n = keep.size();
            auto entire_keep_as_drop = keep;
            
            keep = SplitPieces(keep, entire_keep_as_drop);
            
            if (keep.size() == n) {
                break;
            }
        }
        
        int vocab_size = counter_spec_.vocab_size();
        if (keep.size() <= vocab_size - meta_pieces_.size()) {
            LOG(INFO) << "Final pieces count: " << keep.size();
            return keep;
        }
        
        std::vector<std::pair<std::string, int>> pieces_vec(
            keep.begin(), keep.end());
        
        std::sort(pieces_vec.begin(), pieces_vec.end(),
               [](const auto& a, const auto& b) {
                   bool a_num = a.first.length() > 1;
                   bool b_num = b.first.length() > 1;
                   if (a_num != b_num) return a_num < b_num;
                   if (a.second != b.second) return a.second > b.second;
                   if (a.first.length() != b.first.length())
                       return a.first.length() > b.first.length();
                   return a.first < b.first;
               });
        
        Str2Int new_pieces;
        size_t limit = vocab_size - meta_pieces_.size();
        for (size_t i = 0; i < limit && i < pieces_vec.size(); ++i) {
            new_pieces[pieces_vec[i].first] = pieces_vec[i].second;
        }
        
        LOG(INFO) << "Final pieces count after pruning: " << new_pieces.size();
        return new_pieces;
    }
    
    void InitTX() {
        int num_ = max_piece_count_;
        T_.resize(num_, std::vector<float_t>(num_, -INF));
        for (int i = 0; i < num_; ++i) {
            T_[i][0] = 0;
            T_[i][std::min(i + 1, num_ - 1)] = 0;
        }
    }
    void InitT() {
        int num_ = max_piece_count_;
        T_.resize(num_, std::vector<float_t>(num_, -INF));
        
        // 修改：只允许特定的状态转移
        for (int i = 0; i < num_; ++i) {
            // 从任何状态可以转移到状态0
            T_[i][0] = 0;
            
            // 从状态i只能转移到状态i+1
            if (i + 1 < num_) {
                T_[i][i + 1] = 0;
            }
            
            // 最高状态可以自环
            if (i == num_ - 1) {
                T_[i][i] = 0;
            }
        }
    }
    
    std::map<int, std::pair<std::string, Model::Piece::Type>> meta_pieces_;
    std::vector<std::pair<std::string, float>> byte_pieces_;
    std::vector<std::pair<std::string, float>> pieces_;
    CounterSpec counter_spec_;
    NormalizerSpec normalizer_spec_;
    
    size_t max_piece_count_;
    size_t max_piece_size_; // TODO
    size_t min_count_;
    std::vector<std::vector<float_t>> T_;
    std::vector<std::unordered_map<std::string, float_t>> N_;
    const float_t INF = std::numeric_limits<float_t>::infinity();
};

} // namespace piece
