#include "inference_optimize/flash_attention.cuh"

// almost indentical to common_inference gpt2_forward, but uses flash attention
// also gelu_fusion in matmul_forward_cublaslt is supported
void gpt2_forward_flash_attention(GPT2 *model, const int* inputs, size_t B, size_t T) {
    NVTX_RANGE_FN();
    // we must be careful and use size_t instead of int, otherwise
    // we could overflow int. E.g. l * B * NH * T * T overflows int at B 16.

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    // validate B,T are not larger than the values used at initialisation
    // (smaller B,T are okay for inference only)
    if (B > model->batch_size || T > model->seq_len) {
        printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
        exit(EXIT_FAILURE);
    }

    // copy inputs to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    // validate inputs, all indices must be in the range [0, V)
    // we can do this while the copies are already underway
    tokenCheck(inputs, B*T, V);

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, main_stream); // encoding goes into residual[0]

    // first layernorm isn't fused
    layernorm_forward((model->recompute < 2) ? acts.ln1 : acts.lnf, acts.ln1_mean, acts.ln1_rstd, acts.encoded, params.ln1w, params.ln1b, B, T, C, main_stream);

    // this is the main transformer block loop
    // L = 12 for gpt2-small
    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);

        // for layer 0, residual input is the encoded input
        // for other layers, residual input is from previous layer's residual3
        floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_qkvb = params.qkvb + l * 3*C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_attprojb = params.attprojb + l * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcb = params.fcb + l * 4*C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        floatX* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch = acts.fch + l * B * T * 4*C;
        // reuse the same activation buffer at each layer, as we'll re-compute the gelu during backward
        // very useful because we dramatically reduce VRAM usage, and may be able to fit larger batch size
        floatX* l_fch_gelu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C : acts.fch_gelu;
        floatX* l_residual3 = acts.residual3 + l * B * T * C;
        floatX* scratch = (floatX*)acts.output; // used for non-cudnn attention, fcproj, attproj, etc.

        // now do the forward pass
        float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
        matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward_cudnn(l_atty, (float*)l_att, l_qkvr, B, T, NH, C, main_stream);
        

        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);
        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd, residual, scratch, l_ln2w, l_ln2b, B*T, C, main_stream);

        // MLP (FFN): C->4C->C with GELU
        matmul_forward_cublaslt(l_fch_gelu, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, main_stream, l_fch, 2); // gelu fusion
        matmul_forward_cublaslt(scratch, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, main_stream); // the most time consuming matmul
        // OK, fusion across blocks.
        if(l+1 != L) {
            floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + (l + 1) * B * T * C : acts.lnf;
            float* l_ln1_mean = acts.ln1_mean + (l + 1) * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + (l + 1) * B * T;
            const floatX* l_ln1w = params.ln1w + (l + 1) * C;
            const floatX* l_ln1b = params.ln1b + (l + 1) * C;
            fused_residual_forward5(l_residual3, l_ln1, l_ln1_mean, l_ln1_rstd, l_residual2, scratch, l_ln1w, l_ln1b,
                                    B * T, C, main_stream);
        } else {
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd, l_residual2, scratch,
                                    params.lnfw, params.lnfb,
                                    B * T, C, main_stream);
        }
    }

    matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);
    // cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaStreamSynchronize(main_stream));
}

// identical to common_inference inference_naive, but uses flash attention gpt2_forward
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
    bool validation_mode
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
        gpt2_forward_flash_attention(model, packed_gen_tokens, B, forward_T);
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