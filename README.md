# Tokenizer

BPE (Byte Pair Encoding) 分词器的 C++ 实现，支持多种训练算法和推理策略。

## 特性

- **多种训练算法**
  - NaiveCounter — 基础字节级 BPE
  - SimpleCounter — 基于索引链表的高效 BPE
  - SentencePieceCounter — 基于 Symbol 缓存的高级 BPE，支持大词表
  - BytePieceCounter — 字节 + 分片混合方案

- **多种推理方式**
  - NaiveTokenizer — 顺序合并规则
  - SimpleTokenizer — 基于合并规则映射的贪心扫描
  - SentencePieceTokenizer — 带 Unicode 归一化的编解码
  - BytePieceTokenizer — 基于 Trie 的最长匹配 + 字节回退

- **Unicode 支持**
  - NFKC 归一化（预编译 Trie）
  - UTF-8 编解码与校验
  - 字节回退机制处理未知字符

- **模型序列化**
  - 文本格式模型文件，支持保存和加载
  - 特殊 token 处理（`<unk>`, `<s>`, `</s>`, `<pad>`）

详细的算法原理见 [docs/](docs/) 目录：
- [BPE.md](docs/BPE.md) — BPE 算法原理
- [BytePieceCounter.md](docs/BytePieceCounter.md) — BytePiece 训练算法
- [BytePieceTokenizer.md](docs/BytePieceTokenizer.md) — BytePiece 推理算法
- [Label.md](docs/Label.md) — 标注规范

## 构建

```bash
# 仅构建 C++ CLI
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 同时构建 Python 模块
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON
cmake --build build

# 或通过 pip 安装 Python 模块
pip install .
```

## CLI 使用

### 训练

```bash
isma-tokenizer train --method sentencepiece --input corpus.txt --vocab-size 8000
```

支持的训练方法：`naive`、`simple`、`sentencepiece`、`bytepiece`（默认）

可选参数：
- `--model <prefix>` — 模型输出前缀（默认：tokenizer）
- `--normalize <name>` — 归一化：identity | NMT_NFKC（默认：identity）

### 编码

```bash
echo "你好世界" | isma-tokenizer encode --model tokenizer.model
```

输出每行一个 token（piece + id），模型文件中记录了训练方法，自动选择对应的 tokenizer。

### 解码

```bash
echo "231 192 163 897 411 591" | isma-tokenizer decode --model tokenizer.model
```

## Python 接口

```python
import isma_tokenizer

tok = isma_tokenizer.Tokenizer()
tok.load("tokenizer.model")

tok.encode("你好世界")          # → [('你', 897), ('好', 411), ...]
tok.encode_as_ids("你好世界")   # → [897, 411, ...]
tok.encode_as_pieces("你好世界") # → ['你', '好', ...]
tok.decode([897, 411, 591])     # → '你好世界'

tok.piece_to_id("好")           # → 897
tok.id_to_piece(897)            # → '好'
tok.vocab_size()                # → 1259
tok.method                      # → 'sentencepiece'
```

## 预训练模型

`data/tokenizer.model` — 基于中文维基百科语料训练的 BytePiece 模型，80K 词表。

```
$ echo "中华人民共和国成立于1949年" | isma-tokenizer encode --model data/tokenizer.model
中华    520
人民共和国      449
成立于  1157
1       261
9       266
4       274
9       266
年      269
```

## 项目结构

```
src/
  cli.cc                - CLI 程序（train/encode/decode）
  piece_spec.h          - 核心数据结构（Model, CounterSpec, NormalizerSpec）
  naive_counter.h/cc    - 基础 BPE 训练与推理
  naive_tokenizer.h/cc  - 链表优化的 BPE 训练与推理
  sentencepiece_counter.h/cc - Symbol 缓存 BPE 训练
  sentencepiece_tokenizer.h/cc - 高级编解码推理
  bytepiece_counter.h/cc - BytePiece 训练
  bytepiece_tokenizer.h/cc - BytePiece 推理（Trie 最长匹配）
  new_normalizer.h/cc   - NFKC Unicode 归一化
  ustr.h/cc             - UTF-8 编解码
  sentence.h/cc         - 文件 I/O
  darts.h / trie.h      - Double-Array Trie
  common.h/cc           - 日志
  misc.h/cc             - 工具函数
python/
  isma_tokenizer.cc     - pybind11 Python 绑定
data/
  tokenizer.model       - 预训练 BytePiece 模型（80K 词表，中文维基百科）
docs/
  BPE.md                - BPE 算法原理
  BytePieceCounter.md   - BytePiece 训练详解
  BytePieceTokenizer.md - BytePiece 推理详解
  Label.md              - 标注规范
```

## License

MIT
