/*
- This file is for 1) validating correctness of optimized inference code.
- To check correctness, we assume baseline implementation is correct, and compare the logits from optimized implementation to that of baseline.


*/

#define TESTING
#include "inference_gpt2_optimize.cu"


int main(int argc, char *argv[]) {
    /* ----------------
     argument parsing
    ---------------- */
    const char* load_filename = "gpt2_124M.bin";
    const char* tokenizer_path = "gpt2_tokenizer.bin";
    int B = 4;
    int genT = 64;
    int context_len = -1;
    int override_enable_tf32 = 1;

    for (int i = 1; i < argc; i += 2) {
        // if (i + 1 >= argc || argv[i][0] != '-') { error_usage(); }
        std::string flag(argv[i]);
        if (flag == "-e") { load_filename = argv[i+1]; }
        else if (flag == "-tk") { tokenizer_path = argv[i+1]; }
        else if (flag == "-g") { genT = atoi(argv[i+1]); }
        else if (flag == "-b") { B = atoi(argv[i+1]); }
        // else { error_usage(); }
    }

    // disable multi-gpu for inference
    char empty_str[] = "";
    multi_gpu_config = multi_gpu_config_init(1, 0, 1, empty_str, empty_str, "mpi");
    common_start(override_enable_tf32, true);

    /* ----------------
     model loading
    ---------------- */
    GPT2 model;
    gpt2_init_common(&model);
    model.inference_only = true;
    gpt2_build_from_checkpoint(&model, load_filename);

    printf("=== KV Cache Validation ===\n");
    printf("[Run Config]\n");
    printf("  checkpoint: %s\n", load_filename);
    printf("  tokenizer:  %s\n", tokenizer_path);
    printf("  genT:       %d\n", genT);
    printf("  batch size: %d\n", B);
    printf("  precision:  %s\n", PRECISION_MODE == PRECISION_FP32 ? "FP32" : (PRECISION_MODE == PRECISION_FP16 ? "FP16" : "BF16"));
    printf("===========================\n\n");
    fflush(stdout);

    context_len = model.config.max_seq_len;
    if (genT > context_len) {
        printf0("Warning: genT=%d > context_len=%d, clipping to context length\n", genT, context_len);
        genT = context_len;
    }


    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, tokenizer_path);

    gpt2_allocate_state(&model, B, context_len);
    
    int* gen_tokens = (int*)mallocCheck(B * context_len * sizeof(int));
    int* ref_tokens = (int*)mallocCheck(B * context_len * sizeof(int));
    int eot_token = tokenizer.eot_token;
    
    for (int i = 0; i < B * context_len; ++i) {
        if (i % context_len == 0) {
            gen_tokens[i] = 43*(i / context_len); // some random token, 43*B < vocab_size
        } else {
            gen_tokens[i] = eot_token;
        }
    } // fill gen_tokens with eot for base implementation,


    /* ---------------
     validate correctness and measure performance
    --------------- */
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;

    // 1) baseline logits (single forward for all steps, causal)
    float* baseline_logits = (float*)mallocCheck(size_t(B * V) * sizeof(float));
    floatX* baseline_logits_raw = (floatX*)mallocCheck(size_t(B * Vp) * sizeof(floatX)); // for mixed precision
    size_t forward_T = context_len; // avoid index related issues by running full context length

    gpt2_forward(&model, gen_tokens, B, forward_T);

    floatX* d_logits = model.acts.output; // (B, forward_T, Vp)

    // build reference tokens by greedy (argmax) decoding from baseline logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < context_len; t++) {
            ref_tokens[b * context_len + t] = gen_tokens[b * context_len + t];
        }
    }
    for (int t = 0; t < genT; t++) {
        // copy baseline logits for step t
        for (int b = 0; b < B; b++) {
            cudaCheck(cudaMemcpy(
                &baseline_logits_raw[b * Vp],
                &d_logits[b * forward_T * Vp + t * Vp],
                (size_t)(Vp) * sizeof(floatX),
                cudaMemcpyDeviceToHost
            ));
        }
        // argmax per batch
        for (int b = 0; b < B; b++) {
            int best_i = 0;
            float best_v = float(baseline_logits_raw[b * Vp + 0]);
            for (int i = 1; i < V; i++) {
                float v = float(baseline_logits_raw[b * Vp + i]);
                if (v > best_v) {
                    best_v = v;
                    best_i = i;
                }
            }
            // next token goes to position t+1 if within context
            if (t + 1 < context_len) {
                ref_tokens[b * context_len + (t + 1)] = best_i;
            }
        }
    }

    // Recompute baseline logits using ref_tokens so both paths share the same context
    gpt2_forward(&model, ref_tokens, B, forward_T);
    d_logits = model.acts.output; // (B, forward_T, Vp)

    // 2) run optimized implementation (teacher forcing with ref_tokens)
    float* optimized_logits;
    cudaCheck(cudaMallocHost(&optimized_logits, (size_t)(B * V) * sizeof(float)));
    floatX* d_optimized_logits;
    cudaCheck(cudaMalloc(&d_optimized_logits, (size_t)(B * Vp) * sizeof(floatX)));

    // KV cache optimization
    {
        gpt2_kvcache_init(&model, B, context_len); 
        DecodeState decode_st;
        decode_state_init(&decode_st, &model, B);

        floatX* h_optimized_logits;
        cudaCheck(cudaMallocHost(&h_optimized_logits, (size_t)(B * Vp) * sizeof(floatX)));
    
        int* d_token_ids;
        int* h_token_ids;
        cudaCheck(cudaMalloc(&d_token_ids, (size_t)B * sizeof(int)));
        cudaCheck(cudaMallocHost((void **)&h_token_ids, (size_t)B * sizeof(int)));
        // copy initial tokens
        for (int b = 0; b < B; b++) {
            h_token_ids[b] = gen_tokens[b * context_len];
        }
        cudaCheck(cudaMemcpy(d_token_ids, h_token_ids, (size_t)B * sizeof(int), cudaMemcpyHostToDevice));

        printf("Step | max_abs_diff | rmse      | base_top3(token:val)                 | opt_top3(token:val)\n");
        printf("-----+--------------+-----------+-------------------------------------+-------------------------------------\n");

        for (int t = 0; t < genT; t++) {
            // copy baseline logits for step t
            for (int b = 0; b < B; b++) {
                cudaCheck(cudaMemcpy(
                    &baseline_logits_raw[b * Vp],
                    &d_logits[b * forward_T * Vp + t * Vp],
                    (size_t)(Vp) * sizeof(floatX),
                    cudaMemcpyDeviceToHost
                ));
            }

            // copy to baseline_logits
            for (int b = 0; b < B; b++) {
                for (int i = 0; i < V; i++) {
                    baseline_logits[b * V + i] = float(baseline_logits_raw[b * Vp + i]);
                }
            }

            // set token ids for step t
            for (int b = 0; b < B; b++) {
                h_token_ids[b] = ref_tokens[b * context_len + t];
            }
            cudaCheck(cudaMemcpy(d_token_ids, h_token_ids, (size_t)B * sizeof(int), cudaMemcpyHostToDevice));

            gpt2_decode_step(&model, &decode_st, d_token_ids, B, t, d_optimized_logits, main_stream);
        
            cudaCheck(cudaMemcpy(h_optimized_logits, d_optimized_logits, (size_t)(B * Vp) * sizeof(floatX), cudaMemcpyDeviceToHost));
            // copy to optimized_logits
            for (int b = 0; b < B; b++) {
                for (int i = 0; i < V; i++) {
                    optimized_logits[b * V + i] = float(h_optimized_logits[b * Vp + i]);
                }
            }

            // 3) compare logits
            float max_abs_diff = 0.0f;
            double sum_rmse = 0.0;
            int base_top_idx[3] = {0, 0, 0};
            float base_top_val[3] = {-1e30f, -1e30f, -1e30f};
            int opt_top_idx[3] = {0, 0, 0};
            float opt_top_val[3] = {-1e30f, -1e30f, -1e30f};
            for (int b = 0; b < B; b++) {
                for (int i = 0; i < V; i++) {
                    float base_v = baseline_logits[b * V + i];
                    float opt_v = optimized_logits[b * V + i];
                    float diff = fabsf(opt_v - base_v);
                    if (diff > max_abs_diff) {
                        max_abs_diff = diff;
                    }
                    sum_rmse += (double)diff * diff;
                    // track top-3 per batch aggregated (keep overall top-3 across all batches)
                    if (base_v > base_top_val[0]) {
                        base_top_val[2] = base_top_val[1]; base_top_idx[2] = base_top_idx[1];
                        base_top_val[1] = base_top_val[0]; base_top_idx[1] = base_top_idx[0];
                        base_top_val[0] = base_v;          base_top_idx[0] = i;
                    } else if (base_v > base_top_val[1]) {
                        base_top_val[2] = base_top_val[1]; base_top_idx[2] = base_top_idx[1];
                        base_top_val[1] = base_v;          base_top_idx[1] = i;
                    } else if (base_v > base_top_val[2]) {
                        base_top_val[2] = base_v;          base_top_idx[2] = i;
                    }

                    if (opt_v > opt_top_val[0]) {
                        opt_top_val[2] = opt_top_val[1]; opt_top_idx[2] = opt_top_idx[1];
                        opt_top_val[1] = opt_top_val[0]; opt_top_idx[1] = opt_top_idx[0];
                        opt_top_val[0] = opt_v;          opt_top_idx[0] = i;
                    } else if (opt_v > opt_top_val[1]) {
                        opt_top_val[2] = opt_top_val[1]; opt_top_idx[2] = opt_top_idx[1];
                        opt_top_val[1] = opt_v;          opt_top_idx[1] = i;
                    } else if (opt_v > opt_top_val[2]) {
                        opt_top_val[2] = opt_v;          opt_top_idx[2] = i;
                    }
                }
            }
            double rmse = sqrt(sum_rmse / (B * V));

            printf("%4d | %12.6f | %9.6f | %5d:%9.4f %5d:%9.4f %5d:%9.4f | %5d:%9.4f %5d:%9.4f %5d:%9.4f\n",
                   t, max_abs_diff, (float)rmse,
                   base_top_idx[0], base_top_val[0], base_top_idx[1], base_top_val[1], base_top_idx[2], base_top_val[2],
                   opt_top_idx[0], opt_top_val[0], opt_top_idx[1], opt_top_val[1], opt_top_idx[2], opt_top_val[2]);
        }

        cudaFreeHost(h_optimized_logits);
        decode_state_free(&decode_st);
        gpt2_kvcache_free(&model);
        cudaFree(d_token_ids);
        cudaFreeHost(h_token_ids);
    }

    // cleanup
    cudaFreeHost(optimized_logits);
    cudaFree(d_optimized_logits);
    free(baseline_logits);
    cudaFreeHost(baseline_logits_raw);

    tokenizer_free(&tokenizer);
    free(gen_tokens);
    free(ref_tokens);
    gpt2_free(&model);
    multi_gpu_config_free(&multi_gpu_config);
    common_free(model);
    return 0;
}
