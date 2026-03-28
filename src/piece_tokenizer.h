#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common.h"
#include "misc.h"
#include "normalizer.h"
#include "piece_spec.h"
#include "sentence.h"
#include "ustr.h"

namespace piece {

template<typename T>
class Node {
public:
  int count;
  T value;
  int pos;

  Node(int count, const T& value, int pos)
    : count(count), value(value), pos(pos) {}

  int GetCount() const { return count; }

  bool operator<(const Node& o) const {
    return this->GetCount() < o.GetCount();
  }
};

template<typename T>
class Multiset {
public:
  Multiset() = default;
  ~Multiset() {
    for (auto node : vec_) {
      delete node;
    }
  }

  void Insert(const T& item, int count = 1) {
    to_insert_[item] += count;
  }

  void Remove(const T& item, int count = 1) {
    to_remove_[item] += count;
  }

  int GetCount(const T& item) {
    _Commit();
    auto it = map_.find(item);
    return it == map_.end() ? 0 : it->second->count;
  }

  T Top() {
    _Commit();
    return vec_.empty() ? T() : vec_[0]->value;
  }

  std::vector<std::pair<T, int>> TopK(int k) {
    _Commit();
    std::vector<std::pair<T, int>> res;
    if (vec_.empty()) return res;

    using HeapItem = std::tuple<int, T, int, int>;
    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>>
        heap;

    heap.push(std::make_tuple(-vec_[0]->GetCount(),
                              vec_[0]->value,
                              vec_[0]->count,
                              vec_[0]->pos));

    for (int i = 0; i < k && !heap.empty(); ++i) {
      auto [_key, val, count, pos] = heap.top();
      heap.pop();
      res.emplace_back(val, count);

      for (int child_pos : {pos * 2 + 1, pos * 2 + 2}) {
        if (child_pos < static_cast<int>(vec_.size())) {
          heap.push(std::make_tuple(-vec_[child_pos]->GetCount(),
                                    vec_[child_pos]->value,
                                    vec_[child_pos]->count,
                                    child_pos));
        }
      }
    }
    return res;
  }

  explicit operator bool() const {
    const_cast<Multiset*>(this)->_Commit();
    return !vec_.empty();
  }

private:
  void _Insert(const T& item, int count = 1) {
    auto it = map_.find(item);
    if (it == map_.end()) {
      Node<T>* node = new Node<T>(0, item, vec_.size());
      map_[item] = node;
      vec_.push_back(node);
      it = map_.find(item);
    }
    it->second->count += count;
    _ItemIncrease(it->second->pos);
  }

  void _Remove(const T& item, int count = 1) {
    auto it = map_.find(item);
    if (it != map_.end()) {
      it->second->count -= count;
      _ItemDecrease(it->second->pos);
    }
  }

  void _Commit() {
    for (const auto& pair : to_insert_) {
      _Insert(pair.first, pair.second);
    }
    for (const auto& pair : to_remove_) {
      _Remove(pair.first, pair.second);
    }
    to_insert_.clear();
    to_remove_.clear();
  }

  void _ItemIncrease(int pos) {
    Node<T>* node = vec_[pos];
    while (pos > 0) {
      int uppos = (pos - 1) >> 1;
      Node<T>* up = vec_[uppos];
      if (*up < *node) {
        vec_[pos] = up;
        up->pos = pos;
        pos = uppos;
        continue;
      }
      break;
    }
    vec_[pos] = node;
    node->pos = pos;
  }

  void _ItemDecrease(int pos) {
    int endpos = vec_.size();
    Node<T>* node = vec_[pos];
    int downpos = 2 * pos + 1;
    while (downpos < endpos) {
      int rightpos = downpos + 1;
      if (rightpos < endpos && !(*vec_[rightpos] < *vec_[downpos])) {
        downpos = rightpos;
      }
      Node<T>* downnode = vec_[downpos];
      if (*node < *downnode) {
        vec_[pos] = downnode;
        downnode->pos = pos;
        pos = downpos;
        downpos = 2 * pos + 1;
      } else {
        break;
      }
    }
    vec_[pos] = node;
    node->pos = pos;
  }

  struct Hash {
    template<typename U>
    size_t operator()(const U& x) const {
      return std::hash<U>()(x);
    }

    size_t operator()(const std::pair<int, int>& p) const {
      auto h1 = std::hash<int>{}(p.first);
      auto h2 = std::hash<int>{}(p.second);
      return h1 ^ (h2 << 1);
    }
  };

  std::vector<Node<T>*> vec_;
  std::unordered_map<T, Node<T>*, Hash> map_;
  std::unordered_map<T, int, Hash> to_insert_;
  std::unordered_map<T, int, Hash> to_remove_;
};

template<typename T>
class IndexedList {
public:
  class Node {
  public:
    T value;
    Node* prev;
    Node* next;

    Node(const T& value, Node* prev, Node* next)
        : value(value), prev(prev), next(next) {}
    ~Node() {}

    void Delete() {
      if (prev) prev->next = next;
      if (next) next->prev = prev;
      next = prev = nullptr;
    }
  };

  struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
      auto s1 = std::hash<T1>{}(p.first);
      auto s2 = std::hash<T2>{}(p.second);
      return s1 ^ s2;
    }
  };

  Node* start_;
  std::unordered_map<std::pair<T, T>, std::vector<Node*>, PairHash> index_;

public:
  template<typename Iterator>
  IndexedList(Iterator begin, Iterator end) {
    if (begin == end) {
      start_ = nullptr;
      return;
    }

    auto it = begin;
    T a = *it;
    start_ = new Node(a, nullptr, nullptr);
    Node* prev_node = start_;

    ++it;
    while (it != end) {
      T b = *it;
      Node* node = new Node(b, prev_node, nullptr);
      prev_node->next = node;
      InsertToIndex(std::make_pair(a, b), prev_node);
      a = b;
      prev_node = node;
      ++it;
    }
  }

  ~IndexedList() {
    Node* current = start_;
    while (current) {
      Node* next = current->next;
      delete current;
      current = next;
    }
  }

  IndexedList(const IndexedList& other) : start_(nullptr) {
    if (!other.start_) {
      return;
    }

    start_ = new Node(other.start_->value, nullptr, nullptr);
    Node* curr = start_;
    Node* other_curr = other.start_->next;

    while (other_curr) {
      curr->next = new Node(other_curr->value, curr, nullptr);
      curr = curr->next;
      other_curr = other_curr->next;
    }

    curr = start_;
    while (curr->next) {
      InsertToIndex(std::make_pair(curr->value, curr->next->value), curr);
      curr = curr->next;
    }
  }

  const std::unordered_map<std::pair<T, T>, std::vector<Node*>, PairHash>&
  GetIndexMap() const {
    return index_;
  }

  IndexedList(IndexedList&& other) noexcept
      : start_(other.start_), index_(std::move(other.index_)) {
    other.start_ = nullptr;
  }

  IndexedList& operator=(const IndexedList& other) {
    if (this != &other) {
      IndexedList temp(other);
      std::swap(start_, temp.start_);
      std::swap(index_, temp.index_);
    }
    return *this;
  }

  IndexedList& operator=(IndexedList&& other) noexcept {
    if (this != &other) {
      Node* current = start_;
      while (current) {
        Node* next = current->next;
        delete current;
        current = next;
      }

      start_ = other.start_;
      index_ = std::move(other.index_);
      other.start_ = nullptr;
    }
    return *this;
  }

  std::vector<Node*>& GetIndex(const std::pair<T, T>& pair) {
    return index_[pair];
  }

  const std::vector<Node*>& GetIndex(const std::pair<T, T>& pair) const {
    return index_.at(pair);
  }

  void InsertToIndex(const std::pair<T, T>& pair, Node* node) {
    index_[pair].push_back(node);
  }

  void RemoveIndex(Node* node) {
    if (node->prev) {
      auto pair = std::make_pair(node->prev->value, node->value);
      auto it = index_.find(pair);
      if (it != index_.end()) {
        auto& nodes = it->second;
        nodes.erase(std::remove(nodes.begin(), nodes.end(), node->prev),
                    nodes.end());
        if (nodes.empty()) {
          index_.erase(it);
        }
      }
    }

    if (node->next) {
      auto pair = std::make_pair(node->value, node->next->value);
      auto it = index_.find(pair);
      if (it != index_.end()) {
        auto& nodes = it->second;
        nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
        if (nodes.empty()) {
          index_.erase(it);
        }
      }
    }
  }

  void UpdateIndex(Node* node) {
    if (node->prev) {
      InsertToIndex(std::make_pair(node->prev->value, node->value), node->prev);
    }
    if (node->next) {
      InsertToIndex(std::make_pair(node->value, node->next->value), node);
    }
  }

  class Iterator {
  private:
    Node* current;

  public:
    explicit Iterator(Node* node) : current(node) {}

    Iterator& operator++() {
      if (current) current = current->next;
      return *this;
    }

    Node* operator*() const { return current; }

    bool operator!=(const Iterator& o) const { return current != o.current; }
  };

  Iterator begin() { return Iterator(start_); }
  Iterator end() { return Iterator(nullptr); }
  Iterator begin() const { return Iterator(start_); }
  Iterator end() const { return Iterator(nullptr); }
};

class PieceTokenizer {
public:
  using EncodeResult = std::vector<std::pair<std::string, int>>;
  using StrToInt = std::unordered_map<std::string_view, int>;

  explicit PieceTokenizer(const Model& model);
  ~PieceTokenizer();

  EncodeResult Encode(std::string_view text) const;
  std::vector<std::string> Tokenize(std::string_view text) const;
  std::string Decode(const std::vector<int>& ids) const;
  std::string Decode(const EncodeResult& rs) const;
  int PieceID(std::string_view piece) const;

private:
  struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
      auto h1 = std::hash<int>{}(p.first);
      auto h2 = std::hash<int>{}(p.second);
      return h1 ^ (h2 << 1);
    }
  };

  using MergeRule = std::pair<std::pair<int, int>, int>;
  using RuleInfo = std::pair<size_t, int>;

  std::string NormalizeText(std::string_view text) const;
  IndexedList<int> BuildInitialTokenList(const std::string& text) const;
  void ApplyMergeRules(IndexedList<int>& token_list) const;
  void Merge(const std::pair<int, int>& pair,
             int new_id,
             IndexedList<int>& indexed_list) const;
  EncodeResult TokenIdsToResult(const IndexedList<int>& token_list) const;
  void InitFromModel();
  void BuildMergeRules();

  const Model* model_;
  std::vector<MergeRule> merge_rules_;
  std::unordered_map<std::pair<int, int>, RuleInfo, PairHash> pair_to_rule_;
  std::unordered_map<int, std::string> vocab_;
  StrToInt pieces_;
  StrToInt reserve_;
  int unk_id_;
};

}  // namespace piece
