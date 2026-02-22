#pragma once

#include "common_inference.cuh"

void inference_naive(
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
