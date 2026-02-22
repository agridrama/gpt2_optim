#include "inference_optimize/baseline.cuh"

__device__ float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}

__device__ void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
}

__global__ void __launch_bounds__(16*16, 2) matmul_forward_kernel4(float* out,
                                                                   const float* inp, const float* weight, const float* bias,
                                                                   int C, int OC, cudaStream_t stream = main_stream) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // each thread handles 8x8 elements; each block 128 by 128 elements.
    int oc = 8*(blockIdx.y * blockDim.y + threadIdx.y);

    // buffers to cache chunks of the input matrices
    __shared__ float lhs_s[128][32];
    __shared__ float rhs_s[128][32];

    // adjust our pointers for the current block
    inp += 128 * blockIdx.x * C;
    weight += 128 * blockIdx.y * C;
    out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

    float vals[8][8] = {};
    if(bias != NULL) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j += 4) {
                float4 b = ld_vec(bias + oc + j);
                vals[i][j+0] = b.x;
                vals[i][j+1] = b.y;
                vals[i][j+2] = b.z;
                vals[i][j+3] = b.w;
            }
        }
    }

    int si_start = 4*(16 * threadIdx.y + threadIdx.x);
    for (int so = 0; so < C; so += 32) {
        __syncthreads();
        int xmod8 = threadIdx.x % 8;
        int xby8 = threadIdx.x / 8;
        int xo = 4 * xmod8;
        for(int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }
        __syncthreads();

        for (int si = si_start; si < si_start + 32; si += 4) {
            float4 rhs[8];
            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
            }

            for (int ii = 0; ii < 8; ++ii) {
                float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            st_vec(out + (8*threadIdx.x+i) * OC + 8*threadIdx.y + j, result);
        }
    }
}

void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    int sqrt_block_size = 16;

    dim3 gridDim(CEIL_DIV(B * T, 8*sqrt_block_size), CEIL_DIV(OC, 8*sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    cudaCheck(cudaGetLastError());
}

// naive implementation backend
void inference_naive(
    GPT2* model,
    Tokenizer* tokenizer,
    int B,
    int* gen_tokens, // size [B * context_len]
    int genT,
    int context_len,
    const std::vector<float>& coins, // size genT-1, instead of seeding rng state
    InferenceResult* result,
    InferenceScratch* scratch,
    cudaStream_t stream, // naive always uses main_stream
    bool use_argmax = false, // if true, use argmax instead of sampling
    bool validation_mode = false // if true, print generated tokens and save top5 logits 
) {
    if (stream != main_stream) {
        fprintf(stderr, "inference_naive: stream must be main_stream\n");
        exit(EXIT_FAILURE);
    }

    if (genT != result->genT || context_len != result->context_len || B != result->B) {
        fprintf(stderr, "inference_naive: result size mismatch\n");
        exit(EXIT_FAILURE);
    }

    const int V  = model->config.vocab_size;
    const int Vp = model->config.padded_vocab_size;

    timeval total_start, total_end;
    timeval forward_start, forward_end;
    double total_forward_msec = 0.0;
    double total_msec = 0.0;

    if (validation_mode) {
        printf("generating:\n---\n");
        if (tokenizer->init_ok) printf("use tokenizer\n");
        else printf("tokenizer init failed, outputting token ids\n");
        fflush(stdout);
    }

    int current_T = -1; // current forward_T in packed_gen_tokens
    int* packed_gen_tokens; // max_size = B * context_len
    cudaCheck(cudaMallocHost((void**)&packed_gen_tokens, (size_t)B * context_len * sizeof(int)));

    gettimeofday(&total_start, 0);
    for (int t = 1; t < genT; t++) {
        int chunk = context_len < 256 ? context_len : 256;
        size_t forward_T = (size_t)CEIL_DIV(t, chunk) * chunk;
        // need to reshape gen_tokens to [B, forward_T] with packing
        if (current_T != forward_T) {
            current_T = forward_T;
            // repack
            for (int b = 0; b < B; b++) {
                int* src = &gen_tokens[b * context_len];
                int* dst = &packed_gen_tokens[b * forward_T];
                // copy existing tokens
                memcpy(dst, src, (size_t)(forward_T < context_len ? forward_T : context_len) * sizeof(int));
            }
        } // make sure packed_gen_tokens is ready for t=1

        // cuda synchronize is called in the end of gpt2_forward,
        // so timing here is accurate
        gettimeofday(&forward_start, 0);
        // inside gpt2_forward, main_stream is used for kernels
        gpt2_forward(model, packed_gen_tokens, B, forward_T);
        gettimeofday(&forward_end, 0); 

        double forward_msec = (1000000.0*(forward_end.tv_sec-forward_start.tv_sec) + forward_end.tv_usec-forward_start.tv_usec) / 1000.0;
        total_forward_msec += forward_msec;

        for (int b = 0; b < B; b++) {
            // this memcpy is not optimal, but keep it simple for now
            floatX* logits = model->acts.output + (size_t)(b * forward_T + (t - 1)) * (size_t)Vp;
            floatX* logits_raw_b = &scratch->cpu_logits_raw[b * Vp];
            cudaCheck(cudaMemcpyAsync(
                logits_raw_b, logits,(size_t)Vp * sizeof(floatX),cudaMemcpyDeviceToHost, main_stream
            ));
            // Note: naive implementation outputs i-th token logits at logits[b, i-1], so we use (t-1) here
        }
        cudaCheck(cudaStreamSynchronize(main_stream));

        // post-process logits to get next token
        // batch loop
        for (int b = 0; b < B; b++) {
            // output shape: [B, forward_T, Vp]
            // get logits for time t-1
            float* logits_f32_b  = &scratch->cpu_logits_f32[b * V];
            for(int j = 0; j < V; j++) {
                logits_f32_b[j] = (float)scratch->cpu_logits_raw[b * Vp + j];
            }

            if (validation_mode) {
                top5_from_logits(logits_f32_b, V, &result->top5[b][t - 1]);
                printf("t=%d b=%d top5: ", t - 1, b);
                print_top5(result->top5[b][t - 1]);
                printf("\n");
            }

            int next_token = sampling_next_token(
                logits_f32_b, V, coins[t - 1], use_argmax);
            gen_tokens[b * context_len + t] = next_token;
            // write back to packed_gen_tokens
            packed_gen_tokens[b * current_T + t] = next_token;

            if (tokenizer->init_ok && validation_mode) {
                safe_printf(tokenizer_decode(tokenizer, next_token));
                fflush(stdout);
            }
        }
    }
    cudaCheck(cudaFreeHost(packed_gen_tokens));

    gettimeofday(&total_end, 0);
    total_msec = (1000000.0*(total_end.tv_sec-total_start.tv_sec) + total_end.tv_usec-total_start.tv_usec) / 1000.0;

    result->total_ms = total_msec;
    result->forward_ms = total_forward_msec;

    // keep result tokens in sync (minimal fix)
    std::copy(gen_tokens, gen_tokens + B*context_len, result->tokens.begin());
    result->coins = coins;
}
