#include "inference_optimize/kvcache_optimize.cuh"

void gpt2_kvcache_init(GPT2* m, int B, int maxT) {
    int L  = m->config.num_layers;
    int NH = m->config.num_heads;
    int C  = m->config.channels;
    int HS = C / NH;

    m->kv_maxT = maxT;
    m->kv_B = B;
    m->kv_curT = 0;

    size_t elems = (size_t)L * B * NH * maxT * HS;
    cudaCheck(cudaMalloc(&m->k_cache, elems * sizeof(floatX)));
    cudaCheck(cudaMalloc(&m->v_cache, elems * sizeof(floatX)));
}

void gpt2_kvcache_free(GPT2* m) {
    if (m->k_cache) cudaCheck(cudaFree(m->k_cache));
    if (m->v_cache) cudaCheck(cudaFree(m->v_cache));
    m->k_cache = m->v_cache = NULL;
}

__device__ __forceinline__ float fx2f(float x) { return x; }
__device__ __forceinline__ float fx2f(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float f2fx(float x) { return x; }
__device__ __forceinline__ __nv_bfloat16 f2fx_bf16(float x) { return __float2bfloat16_rn(x); }

template<typename TX>
__device__ __forceinline__ TX f2fxT(float x) {
    if constexpr (std::is_same<TX, float>::value) return x;
    else return f2fx_bf16(x);
}

void decode_state_init(DecodeState* st, GPT2* model, int B) {
    int C  = model->config.channels;
    int NH = model->config.num_heads;

    cudaCheck(cudaMalloc(&st->x,         (size_t)B * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&st->ln1,       (size_t)B * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&st->ln2,       (size_t)B * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&st->qkv_packed,(size_t)B * (3*C) * sizeof(floatX)));
    cudaCheck(cudaMalloc(&st->qkvr,      (size_t)3 * B * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&st->att_y,     (size_t)B * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&st->proj,      (size_t)B * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&st->mlp_h,     (size_t)B * (4*C) * sizeof(floatX)));
    cudaCheck(cudaMalloc(&st->mlp,       (size_t)B * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&st->mlp_pre,   (size_t)B * (4*C) * sizeof(floatX)));
    cudaCheck(cudaMalloc(&st->mean,      (size_t)B * sizeof(float)));
    cudaCheck(cudaMalloc(&st->rstd,      (size_t)B * sizeof(float)));

    int maxT = model->kv_maxT;
    st->scratch_maxT = maxT;
    cudaCheck(cudaMalloc(&st->scores_f32, (size_t)B * NH * maxT * sizeof(float)));
    cudaCheck(cudaMalloc(&st->probs_f32, (size_t)B * NH * maxT * sizeof(float)));
}

void decode_state_free(DecodeState* st) {
    cudaFreeCheck(&st->x);
    cudaFreeCheck(&st->ln1);
    cudaFreeCheck(&st->ln2);
    cudaFreeCheck(&st->qkv_packed);
    cudaFreeCheck(&st->qkvr);
    cudaFreeCheck(&st->att_y);
    cudaFreeCheck(&st->proj);
    cudaFreeCheck(&st->mlp_h);
    cudaFreeCheck(&st->mlp);
    cudaFreeCheck(&st->mlp_pre);
    cudaFreeCheck(&st->mean);
    cudaFreeCheck(&st->rstd);
    cudaFreeCheck(&st->scores_f32);
    cudaFreeCheck(&st->probs_f32);
}

__global__ void kvcache_write_step_kernel(
    floatX* __restrict__ k_cache,
    floatX* __restrict__ v_cache,
    const floatX* __restrict__ k_step,
    const floatX* __restrict__ v_step,
    int l, int B, int NH, int HS, int maxT, int t_pos)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * NH * HS;
    if (idx >= total) return;

    int hs = (int)(idx % HS);
    idx /= HS;
    int h  = (int)(idx % NH);
    int b  = (int)(idx / NH);

    size_t src = (((size_t)b * NH + h) * HS + hs);
    size_t dst = kv_index(l, b, h, t_pos, hs, B, NH, maxT, HS);

    k_cache[dst] = k_step[src];
    v_cache[dst] = v_step[src];
}

__global__ void attn_decode_scores_kernel_f32(
    float* __restrict__ scores,
    const floatX* __restrict__ q_bh,
    const floatX* __restrict__ k_cache,
    int l, int B, int NH, int HS, int maxT, int t_len)
{
    int row = blockIdx.x;
    int b = row / NH;
    int h = row % NH;
    if (b >= B) return;

    const floatX* q = q_bh + ((size_t)row * HS);
    for (int i = threadIdx.x; i < t_len; i += blockDim.x) {
        float acc = 0.f;
        #pragma unroll
        for (int hs = 0; hs < HS; ++hs) {
            size_t kidx = kv_index(l, b, h, i, hs, B, NH, maxT, HS);
            acc += fx2f(q[hs]) * fx2f(k_cache[kidx]);
        }
        acc *= rsqrtf((float)HS);
        scores[(size_t)row * maxT + i] = acc;
    }
}

__global__ void softmax_decode_row_f32_kernel(
    float* __restrict__ probs,
    const float* __restrict__ scores,
    float inv_temperature,
    int N,
    int maxT,
    int t_len)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    int row = blockIdx.x * num_warps + warp;
    if (row >= N) return;

    const float* x = scores + (size_t)row * maxT;
    float* y = probs + (size_t)row * maxT;

    const float flt_max = 340282346638528859811704183484516925440.0f;
    float maxval = -flt_max;
    float sumval = 0.f;

    int t4 = (t_len / 4) * 4;
    for (int i = lane * 4; i < t4; i += 32 * 4) {
        float r0 = x[i+0], r1 = x[i+1], r2 = x[i+2], r3 = x[i+3];
        float old_max = maxval;
        maxval = fmaxf(maxval, r0);
        maxval = fmaxf(maxval, r1);
        maxval = fmaxf(maxval, r2);
        maxval = fmaxf(maxval, r3);
        sumval *= expf(inv_temperature * (old_max - maxval));
        sumval += expf(inv_temperature * (r0 - maxval));
        sumval += expf(inv_temperature * (r1 - maxval));
        sumval += expf(inv_temperature * (r2 - maxval));
        sumval += expf(inv_temperature * (r3 - maxval));
    }
    for (int i = t4 + lane; i < t_len; i += 32) {
        float r = x[i];
        float old_max = maxval;
        maxval = fmaxf(maxval, r);
        sumval *= expf(inv_temperature * (old_max - maxval));
        sumval += expf(inv_temperature * (r - maxval));
    }

    float gmax = warpReduceMax(maxval);
    sumval *= expf(inv_temperature * (maxval - gmax));
    float sum = warpReduceSum(sumval);
    float norm = 1.f / sum;

    for (int i = lane; i < t_len; i += 32) {
        float ev = expf(inv_temperature * (x[i] - gmax));
        y[i] = ev * norm;
    }
}

__global__ void attn_decode_weighted_sum_kernel(
    floatX* __restrict__ out_bc,
    const float* __restrict__ probs,
    const floatX* __restrict__ v_cache,
    int l, int B, int NH, int HS, int maxT, int t_len)
{
    int row = blockIdx.x;
    int b = row / NH;
    int h = row % NH;
    if (b >= B) return;

    const float* p = probs + (size_t)row * maxT;
    for (int hs = threadIdx.x; hs < HS; hs += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < t_len; ++i) {
            size_t vidx = kv_index(l, b, h, i, hs, B, NH, maxT, HS);
            acc += p[i] * fx2f(v_cache[vidx]);
        }
        out_bc[(size_t)b * (NH * HS) + h * HS + hs] = f2fxT<floatX>(acc);
    }
}

void attention_decode1_kvcache_split(
    floatX* out_bc,
    const floatX* q_bh,
    GPT2* model, DecodeState* st,
    int l, int B, int t_len, cudaStream_t stream)
{
    int NH = model->config.num_heads;
    int C  = model->config.channels;
    int HS = C / NH;
    int maxT = model->kv_maxT;
    int N = B * NH;

    attn_decode_scores_kernel_f32<<<N, 256, 0, stream>>>(
        st->scores_f32, q_bh, model->k_cache, l, B, NH, HS, maxT, t_len);

    int block = 256;
    int num_warps = block / 32;
    int grid = (N + num_warps - 1) / num_warps;
    softmax_decode_row_f32_kernel<<<grid, block, 0, stream>>>(
        st->probs_f32, st->scores_f32, 1.0f, N, maxT, t_len);

    attn_decode_weighted_sum_kernel<<<N, 256, 0, stream>>>(
        out_bc, st->probs_f32, model->v_cache, l, B, NH, HS, maxT, t_len);

    cudaCheck(cudaGetLastError());
}

__global__ void embed_token_pos_kernel(
    floatX* __restrict__ x,
    const floatX* __restrict__ wte,
    const floatX* __restrict__ wpe,
    const int* token_ids,
    int B, int pos, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B*C) return;
    int b = i / C;
    int c = i % C;
    int tok = token_ids[b];
    float a = fx2f(wte[(size_t)tok * C + c]);
    float p = fx2f(wpe[(size_t)pos * C + c]);
    x[i] = f2fxT<floatX>(a + p);
}

__global__ void add_inplace_kernel(floatX* __restrict__ x, const floatX* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = fx2f(x[i]);
    float b = fx2f(y[i]);
    x[i] = f2fxT<floatX>(a + b);
}

__global__ void gelu_forward_scalar_kernel(floatX* out, const floatX* in, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)in[i];
        float y = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[i] = (floatX)y;
    }
}

inline void gelu_forward_any(floatX* out, const floatX* in, int N, cudaStream_t stream) {
    int block = 256;
    int grid  = (N + block - 1) / block;
    gelu_forward_scalar_kernel<<<grid, block, 0, stream>>>(out, in, N);
    cudaCheck(cudaGetLastError());
}

void gpt2_decode_step(GPT2* model, DecodeState* st,
                      const int* d_token_ids, int B, int pos_t,
                      floatX* logits_out_dev, cudaStream_t stream)
{
    NVTX_RANGE_FN();
    int L  = model->config.num_layers;
    int NH = model->config.num_heads;
    int C  = model->config.channels;
    int HS = C / NH;
    int Vp = model->config.padded_vocab_size;

    {
        int block = 256;
        int grid  = (B*C + block - 1) / block;
        embed_token_pos_kernel<<<grid, block, 0, stream>>>(
            st->x, model->params.wte, model->params.wpe, d_token_ids, B, pos_t, C);
    }

    for (int l = 0; l < L; l++) {
        floatX* l_qkvw     = model->params.qkvw     + (size_t)l * 3*C * C;
        floatX* l_qkvb     = model->params.qkvb     + (size_t)l * 3*C;
        floatX* l_attprojw = model->params.attprojw + (size_t)l * C * C;
        floatX* l_attprojb = model->params.attprojb + (size_t)l * C;
        floatX* l_ln1w     = model->params.ln1w     + (size_t)l * C;
        floatX* l_ln1b     = model->params.ln1b     + (size_t)l * C;
        floatX* l_ln2w     = model->params.ln2w     + (size_t)l * C;
        floatX* l_ln2b     = model->params.ln2b     + (size_t)l * C;
        floatX* l_fcw      = model->params.fcw      + (size_t)l * 4*C * C;
        floatX* l_fcb      = model->params.fcb      + (size_t)l * 4*C;
        floatX* l_fcprojw  = model->params.fcprojw  + (size_t)l * C * 4*C;
        floatX* l_fcprojb  = model->params.fcprojb  + (size_t)l * C;

        layernorm_forward(st->ln1, st->mean, st->rstd,
                          st->x, l_ln1w, l_ln1b,
                          B, 1, C, stream);

        matmul_forward_cublaslt(st->qkv_packed, st->ln1, l_qkvw, l_qkvb,
                                B, 1, C, 3*C, stream);

        floatX* q = st->qkvr + 0 * (size_t)B * C;
        floatX* k = st->qkvr + 1 * (size_t)B * C;
        floatX* v = st->qkvr + 2 * (size_t)B * C;

        {
            int total = B * NH * HS;
            int block = 256;
            int grid  = (total + block - 1) / block;
            permute_kernel<<<grid, block, 0, stream>>>(q, k, v, st->qkv_packed, B, 1, NH, HS);
        }

        {
            int block = 256;
            size_t total = (size_t)B * NH * HS;
            int grid = (int)((total + block - 1) / block);
            kvcache_write_step_kernel<<<grid, block, 0, stream>>>(
                model->k_cache, model->v_cache,
                k, v, l, B, NH, HS, model->kv_maxT, pos_t);
        }

        attention_decode1_kvcache_split(st->att_y, q, model, st, l, B, pos_t + 1, stream);

        matmul_forward_cublaslt(st->proj, st->att_y, l_attprojw, l_attprojb,
                                B, 1, C, C, stream);

        {
            int block = 256;
            int grid  = (B*C + block - 1) / block;
            add_inplace_kernel<<<grid, block, 0, stream>>>(st->x, st->proj, B*C);
        }

        layernorm_forward(st->ln2, st->mean, st->rstd,
                          st->x, l_ln2w, l_ln2b,
                          B, 1, C, stream);

        matmul_forward_cublaslt(st->mlp_pre, st->ln2, l_fcw, l_fcb,
                        B, 1, C, 4*C, stream,
                        NULL, 0);
        int N = B * (4*C);
        gelu_forward_any(st->mlp_h, st->mlp_pre, N, stream);

        matmul_forward_cublaslt(st->mlp, st->mlp_h, l_fcprojw, l_fcprojb,
                                B, 1, 4*C, C, stream);

        {
            int block = 256;
            int grid  = (B*C + block - 1) / block;
            add_inplace_kernel<<<grid, block, 0, stream>>>(st->x, st->mlp, B*C);
        }
    }

    layernorm_forward(st->ln1, st->mean, st->rstd,
                      st->x, model->params.lnfw, model->params.lnfb,
                      B, 1, C, stream);

    matmul_forward_cublaslt(logits_out_dev, st->ln1, model->params.wte, nullptr,
                            B, 1, C, Vp, stream);
    cudaCheck(cudaStreamSynchronize(stream));
}

// KV cache implementation backend
void inference_kvcache(
    GPT2* model,
    Tokenizer* tokenizer,
    int B,
    int* gen_tokens, // size [B * context_len], stride context_len
    int genT,
    int context_len,
    const std::vector<float>& coins, // size genT-1, instead of seeding rng state. Shared across batch
    InferenceResult* result,
    InferenceScratch* scratch,
    cudaStream_t stream, // stream to use for inference
    bool use_argmax = false, // if true, use argmax instead of sampling
    bool validation_mode = false // if true, print generated tokens and save top5 logits
) {
    if (genT != result->genT || context_len != result->context_len || B != result->B) {
        fprintf(stderr, "inference_kvcache: result size mismatch\n");
        exit(EXIT_FAILURE);
    }
    const int V  = model->config.vocab_size;
    const int Vp = model->config.padded_vocab_size;

    timeval total_start, total_end;
    timeval forward_start, forward_end;
    double total_forward_msec = 0.0;
    double total_msec = 0.0;

    if (validation_mode) {
        printf("generating with KV cache:\n---\n");
        if (tokenizer->init_ok) printf("use tokenizer\n");
        else printf("tokenizer init failed, outputting token ids\n");
        fflush(stdout);
    }

    floatX* d_logits_out; // temporary logits output device buffer
    cudaCheck(cudaMalloc(&d_logits_out, (size_t)(B * Vp) * sizeof(floatX)));

    gettimeofday(&total_start, 0);
    gpt2_kvcache_init(model, B, context_len); 
    DecodeState decode_st;
    decode_state_init(&decode_st, model, B);
    // copy first token ids to device
    int* h_token_ids = nullptr;
    cudaCheck(cudaMallocHost((void**)&h_token_ids, (size_t)B * sizeof(int)));
    for (int b = 0; b < B; b++) {
        h_token_ids[b] = gen_tokens[b*context_len];
    }
    cudaCheck(cudaMemcpyAsync(
        scratch->d_token_ids, h_token_ids, (size_t)B * sizeof(int), cudaMemcpyHostToDevice, stream
    ));
    
    // no prefill, since we only use one init token and decode step-by-step
    for (int t = 1; t < genT; t++) {
        // cuda synchronize is called in the end of gpt2_decode_step,
        // so timing here is accurate
        gettimeofday(&forward_start, 0);
        gpt2_decode_step(model, &decode_st, scratch->d_token_ids, B, t - 1,
                         d_logits_out, stream);
        gettimeofday(&forward_end, 0);
        double forward_msec = (1000000.0*(forward_end.tv_sec-forward_start.tv_sec) + forward_end.tv_usec-forward_start.tv_usec) / 1000.0;
        total_forward_msec += forward_msec;

        cudaCheck(cudaMemcpyAsync(
            scratch->cpu_logits_raw, d_logits_out, (size_t)(B * Vp) * sizeof(floatX), cudaMemcpyDeviceToHost, stream
        ));
        cudaCheck(cudaStreamSynchronize(stream));

        // post-process logits and sample next tokens
        // batch loop
        for (int b = 0; b < B; b++) { // be careful with sizes here
            floatX* logits_raw_b = scratch->cpu_logits_raw + (size_t)b * Vp;
            float* logits_f32_b  = scratch->cpu_logits_f32 + (size_t)b * V;
            for (int j = 0; j < V; j++) {
                logits_f32_b[j] = (float)logits_raw_b[j];
            }
   
            if (validation_mode) {
                top5_from_logits(scratch->cpu_logits_f32 + b * V, V, &result->top5[b][t - 1]);
                printf("b=%d t=%d top5: ", b, t - 1);
                print_top5(result->top5[b][t - 1]);
                printf("\n");
            }

            int next_token = sampling_next_token(
                scratch->cpu_logits_f32 + b * V, V, coins[t - 1], use_argmax);
            gen_tokens[b * context_len + t] = next_token;

            if (tokenizer->init_ok && validation_mode) {
                safe_printf(tokenizer_decode(tokenizer, next_token));
                fflush(stdout);
            }
            // write back next token ids to device
            h_token_ids[b] = gen_tokens[b * context_len + t];
        }
        cudaCheck(cudaMemcpyAsync(
            scratch->d_token_ids, h_token_ids, (size_t)B * sizeof(int), cudaMemcpyHostToDevice, stream
        ));
    }
    decode_state_free(&decode_st);
    cudaFreeCheck(&d_logits_out);
    cudaCheck(cudaFreeHost(h_token_ids));
    gpt2_kvcache_free(model);
    gettimeofday(&total_end, 0);
    total_msec = (1000000.0*(total_end.tv_sec-total_start.tv_sec) + total_end.tv_usec-total_start.tv_usec) / 1000.0;

    result->total_ms = total_msec;
    result->forward_ms = total_forward_msec;

    // keep result tokens in sync (minimal fix)
    std::copy(gen_tokens, gen_tokens + B*context_len, result->tokens.begin());
    result->coins = coins;
}
