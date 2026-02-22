#pragma once

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <assert.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>

// ----------- CPU utilities -----------
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
// defines: create_dir_if_not_exists, find_max_step, ends_with_bin
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: manual_seed, normal_ (same as torch.manual_seed and torch.normal)
#include "llmc/rand.h"
// defines: sample_softmax, random_f32
#include "llmc/sampler.h"
// ----------- GPU utilities -----------
// defines:
// WARP_SIZE, MAX_1024_THREADS_BLOCKS, CEIL_DIV, cudaCheck, PRECISION_MODE
// NVTX_RANGE_FN
#include "llmc/cuda_common.h"
// defines:
// Packed128, f128, x128
// warpReduceSum, warpReduceMax, blockReduce, copy_and_cast_kernel, cudaMallocConditionallyManaged
#include "llmc/cuda_utils.cuh"
// defines: CUBLAS_LOWP, cublasCheck, cublaslt_workspace_size, cublaslt_workspace
// defines: cublas_compute, cublaslt_handle, cublas_handle
#include "llmc/cublas_common.h"
// ----------- Layer implementations in CUDA -----------
// defines: encoder_forward, encoder_backward
#include "llmc/encoder.cuh"
// defines: layernorm_forward, residual_forward, fused_residual_forward5, layernorm_backward
#include "llmc/layernorm.cuh"
// defines: matmul_cublaslt, matmul_forward, matmul_backward, gelu_forward, gelu_backward_inplace
#include "llmc/matmul.cuh"
#ifdef ENABLE_CUDNN
// defines: create_cudnn, destroy_cudnn, attention_forward_cudnn, attention_backward_cudnn
#include "llmc/cudnn_att.h"
#else
// defines: attention_forward, attention_backward
#include "llmc/attention.cuh"
#endif
// ----------- Multi-GPU support -----------
// defines: ncclFloatX, ncclCheck, MultiGpuConfig, ShardInfo
// defines: printf0, multi_gpu_config
// defines: multi_gpu_config_init, multi_gpu_config_free
// defines: set_zero_configs, multi_gpu_cpu_float_sum, multi_gpu_barrier
// defines: multi_gpu_get_shard_offset, multi_gpu_async_reduce_gradient
#include "llmc/zero.cuh"

extern char filename_buffer[512];
extern cudaDeviceProp deviceProp;
extern cudaStream_t main_stream;
inline constexpr size_t IO_BUF_SIZE = 32 * 1024 * 1024;

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

constexpr int NUM_PARAMETER_TENSORS = 16;
typedef struct {
    floatX* wte; // (V, C)
    floatX* wpe; // (maxT, C)
    floatX* ln1w; // (L, C)
    floatX* ln1b; // (L, C)
    floatX* qkvw; // (L, 3*C, C)
    floatX* qkvb; // (L, 3*C)
    floatX* attprojw; // (L, C, C)
    floatX* attprojb; // (L, C)
    floatX* ln2w; // (L, C)
    floatX* ln2b; // (L, C)
    floatX* fcw; // (L, 4*C, C)
    floatX* fcb; // (L, 4*C)
    floatX* fcprojw; // (L, C, 4*C)
    floatX* fcprojb; // (L, C)
    floatX* lnfw; // (C)
    floatX* lnfb; // (C)
} ParameterTensors;
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, GPT2Config config);
void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_elements, size_t *param_sizeof);

constexpr int NUM_ACTIVATION_TENSORS = 21;
typedef struct {
    floatX* encoded; // (B, T, C)
    floatX* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    floatX* atty; // (L, B, T, C)
#if ENABLE_CUDNN
    float* att;  // (L, B, NH, T)
#else
    floatX* att; // (L, B, NH, T, T)
#endif
    floatX* residual2; // (L, B, T, C)
    floatX* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    floatX* fch; // (L, B, T, 4*C)
    floatX* fch_gelu; // (L, B, T, 4*C)
    floatX* residual3; // (L, B, T, C)
    floatX* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* losses; // (B, T)
    floatX* qkvr; // (L, B, T, 3*C)
    floatX* output;
    floatX* scratch_bt4c;   // (B, T, 4*C)
    floatX* scratch_btc;    // (B, T, C)
} ActivationTensors;

struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};

#define TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)}

void fill_in_activation_sizes(const ActivationTensors* data, TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS], size_t B, size_t T, GPT2Config config, int recompute);
void* malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]);

inline __host__ __device__ size_t kv_index(int l, int b, int h, int t, int hs,
                                           int B, int NH, int maxT, int HS);

typedef struct {
    GPT2Config config;
    ParameterTensors params;
    size_t param_elements[NUM_PARAMETER_TENSORS];
    size_t param_sizeof[NUM_PARAMETER_TENSORS];
    void* params_memory;
    size_t num_parameters;
    size_t num_parameters_bytes;
    ActivationTensors acts;
    TensorSpec acts_specs[NUM_ACTIVATION_TENSORS];
    void* acts_memory;
    int batch_size;
    int seq_len;
    int* inputs;
    bool inference_only;
    int gelu_fusion;
    int recompute;
    floatX* k_cache;
    floatX* v_cache;
    int kv_maxT;
    int kv_B;
    int kv_curT;
} GPT2;

void gpt2_init_common(GPT2 *model);
void gpt2_allocate_weights(GPT2 *model);
void gpt2_allocate_state(GPT2 *model, int B, int T);
void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path, bool weight_init);
void gpt2_forward(GPT2 *model, const int* inputs, size_t B, size_t T);

void gpt2_free(GPT2 *model);

void common_start(bool override_enable_tf32, bool print_device_info);
void common_free(GPT2 model);

struct TopK5 {
    int idx[5];
    float val[5];
};

struct InferenceResult {
    int genT = 0;
    int B = 0;
    int context_len = 0;
    std::vector<int> tokens;
    std::vector<float> coins;
    std::vector<std::vector<TopK5>> top5;
    double total_ms = 0.0;
    double forward_ms = 0.0;
};

void inference_result_init(InferenceResult* result, int genT, int context_len, int B);
void inference_result_free(InferenceResult* result);

struct InferenceScratch {
    int* d_token_ids = nullptr;
    floatX* cpu_logits_raw = nullptr;
    float*  cpu_logits_f32 = nullptr;
};

void inference_scratch_init(InferenceScratch* scratch, GPT2* model, int B);
void inference_scratch_free(InferenceScratch* scratch);

void top5_from_logits(const float* x, int V, TopK5* out);
void print_top5(const TopK5& t);
int argmax(const float* x, int n);
int sampling_next_token(const float* probs, int V, float coin, bool use_argmax);
