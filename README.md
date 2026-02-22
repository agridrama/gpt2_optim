# gpt2_optim

KV cacheの正しさ検証・推論速度比較・プロファイルを再現するための作業ディレクトリです。
`karpathy/llm.c`の`llmc/`実装に依存します。

## 目的
- **KV cacheの正しさ検証**: 基準実装（naive）とKV cache版のロジット/トークンを比較。
- **推論速度比較**: naiveとKV cacheでのtokens/secを測定。
- **プロファイル**: KV cacheのdecode stepを`nsys/ncu`で計測。

## ディレクトリ構成
- `gpt2_optim/src/` 実装
- `gpt2_optim/src/inference_optimize/` 推論共通/naive/KV cacheの実装
- `gpt2_optim/bin/` ビルド成果物
- `gpt2_optim/notebooks/exp_colab.ipynb` 参考Notebook（Colab用）

## 前提
- `nvcc`が利用可能
- `llm.c`リポジトリが手元にあり、`llmc/`が存在
- チェックポイントとトークナイザ（例: `gpt2_124M.bin`, `gpt2_tokenizer.bin`）

`llm.c`の取得例:
```bash
git clone https://github.com/karpathy/llm.c.git
cd llm.c
./dev/download_starter_pack.sh
```

## ビルド
`LLM_C_ROOT`で`llm.c`のルートを指定します（既定: `../llm.c`）。
`GPU_COMPUTE_CAPABILITY`は必須です。

```bash
cd gpt2_optim
make all GPU_COMPUTE_CAPABILITY=75 PRECISION=BF16 LLM_C_ROOT=../llm.c
```

- `PRECISION`: `FP32` / `FP16` / `BF16`
- `GPU_COMPUTE_CAPABILITY`: 例 `75` (T4) / `80` (A100) / `90` (H100)

## 実行例
### 推論速度比較（naive vs KV cache）
```bash
./bin/inference_gpt2optimcu \
  -e /path/to/gpt2_124M_bf16.bin \
  -tk /path/to/gpt2_tokenizer.bin \
  -g 64 -b 4 -m 0
```

### KV cacheの正しさ検証
```bash
./bin/validate_kvcache_optimization -g 128 -b 2
```

### プロファイル（nsys）
```bash
nsys profile -t cuda,nvtx \
  --capture-range=nvtx --nvtx-capture='MEASURE@*' --capture-range-end=stop-shutdown \
  -o prof_kvcache \
  ./bin/profile_kvcache_optimization
```

## メモ
- `BF16`やサンプリングを使うと、トークンの完全一致が崩れることがあります。
- Notebookは参考用で、`gpt2_optim/`の構成に合わせてパスの更新が必要な場合があります。
