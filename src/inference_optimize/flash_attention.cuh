#pragma once

#include "common_inference.cuh"

void gpt2_forward_flash_attention(GPT2 *model, const int* inputs, size_t B, size_t T);

void inference_flash_attention(
    GPT2* model,
    Tokenizer* tokenizer,
    int B,
    int* gen_tokens,
    int genT,
    int context_len,
    const std::vector<float>& coins,
    InferenceResult* result,
    InferenceScratch* scratch,
    cudaStream_t stream,
    bool use_argmax,
    bool validation_mode);