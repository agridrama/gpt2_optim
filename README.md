# gpt2_optim
Optimized GPT‑2 inference by introducing KV cache and CUDA kernel tuning, achieving up to 32x speedup on T4.
---

KV cacheの正しさ検証・推論速度比較・プロファイルを行いました。
`karpathy/llm.c`の`llmc/`実装に依存しおり, コミット`1e2ace651495b74ae22d45d1723443fd00ecd3a`で動作確認しています。

2025年秋学期 Applied GPU ProgrammingのFinal Projectとして、GPT-2の推論最適化を目的としています。

## 目的
- **KV cacheの正しさ検証**: 基準実装（naive）とKV cache版のロジット/トークンを比較。
- **推論速度比較**: naiveとKV cacheでのtokens/secを測定。
- **プロファイル**: KV cacheのdecode stepを`nsys/ncu`で計測。

## 成果
**naiveとKV cacheのtokens/sec比較**
| Batch Size | Naive (tokens/sec) | KV Cache (tokens/sec) | Speedup |
| ---------- | ------------------ | --------------------- | ------- |
| 1          | 22-24              | 426                   | 19x     |
| 4          | 22-23              | 336                   | 15x     |
| 16         | 24                 | 780                  | 32x     |
(生成トークン数: 512, GPU: T4, Precision: BF16, Google Colabで実行)

T4はNaiveの性能が低くなりがちですが、KV cacheの最適化により大幅な性能向上が見られた。

**KV cacheの正しさ検証**
同一token列を入力したとき, logitsの差が数値誤差の範囲に収まっており, top-3のtoken候補が一致することを確認できます。token idが重複しているのは, バッチを跨いでtop-3を抽出しているためです。

| Step | Max Abs Diff | RMSE     | Base Top-3 (token:logit)                      | KV Cache Top-3 (token:logit)                  |
| ---- | ------------ | -------- | --------------------------------------------- | --------------------------------------------- |
| 0    | 0.000000     | 0.000000 | 11: -28.3750    13: -28.5000    11: -28.7500  | 11: -28.3750    13: -28.5000    11: -28.7500  |
| 1    | 0.500000     | 0.267544 | 290: -78.5000   262: -79.0000   543: -79.0000 | 290: -78.5000   262: -79.0000   475: -79.0000 |
| 2    | 2.000000     | 0.811436 | 257: -67.5000   262: -67.5000   407: -68.0000 | 257: -68.0000   262: -68.0000   407: -68.5000 |
| 3    | 0.500000     | 0.294186 | 717: -70.5000   366: -71.0000   649: -71.0000 | 717: -70.5000   366: -71.0000   649: -71.0000 |
## ディレクトリ構成
- `gpt2_optim/src/` 実装
- `gpt2_optim/src/inference_optimize/` 推論共通/naive/KV cacheの実装
- `gpt2_optim/bin/` ビルド成果物
- `gpt2_optim/notebooks/exp_colab.ipynb` 参考Notebook（Colab用）

## Colab Notebook Demo
Colab上での実行手順は `notebooks/exp_colab.ipynb` にまとまっています。  
主な流れは以下です:
1. `llm.c` をクローンして `download_starter_pack.sh` を実行
2. `gpt2_optim` をGitHubからクローン
3. `make all` でビルド
4. 推論速度比較、KV cache検証、`nsys`プロファイルを順に実行

NotebookはColab用に `nsys` のインストールも含んでいます。


## Local実行での前提
詳細はNotebookを参考にお願いしますが、以下が前提条件です:

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
`GPU_COMPUTE_CAPABILITY` (GPU世代の指定) は必須です。

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

### KV cache以外の最適化を無効化したビルド
`fused_residual_forward5` を無効化した推論バイナリを別名で生成します。
```bash
make all GPU_COMPUTE_CAPABILITY=75 PRECISION=BF16 LLM_C_ROOT=../llm.c
./bin/inference_gpt2optimcu_kvonly \
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
- `BF16`やサンプリングを使うと、トークンの完全一致が崩れます。
