# PieceTokenizer

BPE (Byte Pair Encoding) 分词器的 C++ 实现，支持多种训练算法和推理策略。

## 特性

- NFKC Unicode 归一化，UTF-8 字节回退
- 文本格式模型文件，可读可编辑
- Python 绑定（pybind11）
- 内置中英文维基百科语料下载与预处理流程

## 构建

需要 CMake 3.14+，C++17。

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Python 模块（需要 pybind11）：

```bash
pip install .
```

## 快速开始

### 准备数据

```bash
cd data
make          # 下载中英文维基百科，处理并分句
```

默认使用 hf-mirror.com 镜像加速下载，直连 HuggingFace 可用 `make download HF_ENDPOINT=`。

产出文件：`cn_sentences.txt`（中文）、`en_sentences.txt`（英文）。

### 训练模型

```bash
cd scripts
make bytepiece                    # 训练单个方法
make                              # 训练所有方法
make bytepiece VOCAB_SIZE=16000   # 自定义词表大小
```

模型输出到 `scripts/output/{method}.model`。

### 分词 / 编码 / 解码

```bash
echo "你好世界" | ./build/piece-tokenizer tokenize --model output/bytepiece.model
echo "你好世界" | ./build/piece-tokenizer encode --model output/bytepiece.model
echo "231 192 163 897" | ./build/piece-tokenizer decode --model output/bytepiece.model
```

### Python 接口

```python
import piece_tokenizer

tok = piece_tokenizer.Tokenizer()
tok.load("output/bytepiece.model")

tok.encode("你好世界")            # → [('你', 897), ('好', 411), ...]
tok.encode_as_ids("你好世界")     # → [897, 411, ...]
tok.encode_as_pieces("你好世界")  # → ['你', '好', ...]
tok.decode([897, 411, 591])       # → '你好世界'

tok.vocab_size()                  # → 8259
tok.method                        # → 'bytepiece'
```

## 训练方法

| 方法 | 训练 | 推理 | 说明 |
|------|------|------|------|
| `piece` | PieceCounter | PieceTokenizer | 类似 NanoChat 的 RustBPE 实现 |
| `sentencepiece` | SentencePieceCounter | SentencePieceTokenizer | Google SentencePiece BPE 实现 |
| `bytepiece` | BytePieceCounter | BytePieceTokenizer | 科学空间 BytePiece 实现 |


## License

MIT
