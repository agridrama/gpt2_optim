#pragma once

#include "common_inference.cuh"

typedef struct {
    floatX* x;         // (B, C)
    floatX* ln1;       // (B, C)
    float* mean;
    float* rstd;
    floatX* qkv_packed; // (B, 3*C)
    floatX* qkvr;       // (3*C)
    floatX* att_y;      // (B, C)
    floatX* proj;       // (B, C)
    floatX* ln2;        // (B, C)
    floatX* mlp_pre;    // (B, 4*C)
    floatX* mlp_h;      // (B, 4*C)
    floatX* mlp;        // (B, C)
    int scratch_maxT;
    float* scores_f32;  // (B * NH * maxT)
    float* probs_f32;   // (B * NH * maxT)
} DecodeState;

void gpt2_kvcache_init(GPT2* m, int B, int maxT);
void gpt2_kvcache_free(GPT2* m);
void decode_state_init(DecodeState* st, GPT2* model, int B);
void decode_state_free(DecodeState* st);
void attention_decode1_kvcache_split(
    floatX* att_y,
    const floatX* q,
    GPT2* model,
    DecodeState* st,
    int l,
    int B,
    int pos_T,
    cudaStream_t stream);
void gpt2_decode_step(GPT2* model, DecodeState* st,
                      const int* token_ids, int B, int pos_t,
                      floatX* logits_out_dev, cudaStream_t stream);

void inference_kvcache(
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