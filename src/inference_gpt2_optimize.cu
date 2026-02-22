#include "inference_optimize/common_inference.cuh"
#include "inference_optimize/baseline.cuh"
#include "inference_optimize/kvcache_optimize.cuh"
#ifdef ENABLE_CUDNN
#include "inference_optimize/flash_attention.cuh"
#endif

// Compile everything as a single TU to avoid duplicated llmc kernel definitions.
#include "inference_optimize/common_inference.cu"
#include "inference_optimize/baseline.cu"
#include "inference_optimize/kvcache_optimize.cu"
#ifdef ENABLE_CUDNN
#include "inference_optimize/flash_attention.cu"
#endif

#ifndef TESTING
// ----------------------------------------------------------------------------
// CLI, poor man's argparse
// (all single letters have been claimed now)

void error_usage() {
    fprintf(stderr, "Usage:   ./inference_gpt2cu [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -e  <string>   model checkpoint (.bin) to load (default = gpt2_124M_bf16.bin)\n");
    fprintf(stderr, "  -g  <int>      number of tokens to generate (default = 64)\n");
    fprintf(stderr, "  -s  <uint64>   RNG seed for sampling (default = 1337)\n");
    fprintf(stderr, "  -tk <string>   tokenizer file path (default = gpt2_tokenizer.bin)\n");
    fprintf(stderr, "  -f  <int>      enable_tf32 override (default = 1)\n");
    fprintf(stderr, "  -b  <int>      batch size (default = 4)\n");
    fprintf(stderr, "  -m  <int>      sampling method (default = 0: random, else: argmax)\n");               
    fprintf(stderr, "  -v  <int>      validation mode (default = 0, set to 1 to enable)\n");
    fprintf(stderr, "  -q  <int>      quiet output (default = 0, set to 1 for tokens/sec only)\n");
    exit(EXIT_FAILURE);
}
// ----------------------------------------------------------------------------
// main inference loop
int main(int argc, char *argv[]) {

    /* ----------------
     argument parsing
    ---------------- */
    const char* load_filename = "gpt2_124M.bin";
    const char* tokenizer_path = "gpt2_tokenizer.bin";
    int B = 4;
    int genT = 64;
    int context_len = -1;
    unsigned long long seed = 1337;
    int override_enable_tf32 = 1;
    bool use_argmax = true;
    bool validation_mode = false;
    bool quiet_output = false;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-') { error_usage(); }
        std::string flag(argv[i]);
        if (flag == "-e") { load_filename = argv[i+1]; }
        else if (flag == "-g") { genT = atoi(argv[i+1]); }
        else if (flag == "-s") { seed = strtoull(argv[i+1], NULL, 10); }
        else if (flag == "-tk") { tokenizer_path = argv[i+1]; }
        else if (flag == "-f") { override_enable_tf32 = atoi(argv[i+1]); }
        else if (flag == "-b") { B = atoi(argv[i+1]); }
        else if (flag == "-m") { use_argmax = atoi(argv[i+1]) != 0; }
        else if (flag == "-v") { validation_mode = atoi(argv[i+1]) != 0; }
        else if (flag == "-q") { quiet_output = atoi(argv[i+1]) != 0; }
        else { error_usage(); }
    }

    if (genT <= 0) {
        fprintf(stderr, "genT must be positive.\n");
        return EXIT_FAILURE;
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
    if (ends_with_bin(load_filename)) {
        gpt2_build_from_checkpoint(&model, load_filename);
    } else {
        fprintf(stderr, "Only .bin checkpoint files are supported for inference.\n");
        return EXIT_FAILURE;
    }
    context_len = model.config.max_seq_len;

    if (genT > context_len) {
        printf0("Warning: genT=%d > context_len=%d, clipping to context length\n", genT, context_len);
        genT = context_len;
    }

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, tokenizer_path);

    gpt2_allocate_state(&model, B, context_len);

    int* gen_tokens = (int*)mallocCheck(B * context_len * sizeof(int));
    int eot_token = tokenizer.eot_token;
    for (int i = 0; i < B * context_len; ++i) {
        if (i % context_len == 0) {
            gen_tokens[i] = 43*(i / context_len); 
        } else {
            gen_tokens[i] = eot_token;
        }
    } // fill gen_tokens with eot for base implementation

    unsigned long long sample_rng_state = seed;
    std::vector<float> coins(genT - 1);
    for (int i = 0; i < genT - 1; i++) {
        coins[i] = random_f32(&sample_rng_state);
    }

    if (!quiet_output) {
        printf0("=== GPT-2 Inference (gpt2_optim) ===\n");
        printf0("[Run Config]\n");
        printf0("  checkpoint: %s\n", load_filename);
        printf0("  tokenizer:  %s\n", tokenizer_path);
        printf0("  genT:       %d\n", genT);
        printf0("  batch size: %d\n", B);
        printf0("  sampling:   %s\n", use_argmax ? "argmax" : "random");
        printf0("  validation: %s\n", validation_mode ? "on" : "off");
        printf0("===================================\n\n");
    }
    
    /* ----------------
     run inference with base implementation
    ---------------- */
    if (!quiet_output) {
        printf0("=== Section: Naive Inference (Baseline) ===\n");
    }
    InferenceResult base_result;
    inference_result_init(&base_result, genT, context_len, B);
    InferenceScratch scratch;
    inference_scratch_init(&scratch, &model, B);
    inference_naive(&model, &tokenizer, B, gen_tokens, genT, context_len,
                    coins, &base_result, &scratch,
                    /*stream=*/main_stream,
                    /*use_argmax=*/use_argmax,
                    validation_mode);
    double base_tps = (double)(B * genT) / (base_result.total_ms / 1000.0);
    if (quiet_output) {
        printf0("base_tokens_per_sec: %.2f\n", base_tps);
    } else {
        printf0("\nBase implementation: total time = %.2f ms, forward time = %.2f ms\n",
                base_result.total_ms, base_result.forward_ms);
        printf0("Token per second: %.2f\n", base_tps);
        printf0("Generated tokens:\n");
        for(int b = 0; b < B; b++) {
            printf0("Batch %d:\n", b);
            for (int t = 0; t < genT; t++) {
                safe_printf(tokenizer_decode(&tokenizer, base_result.tokens[b * context_len + t]));
            }
            printf0("\n===\n");
        }
    }
    
    /* ----------------
     run inference with KV cache implementation
    ---------------- */
    if (!quiet_output) {
        printf0("\n=== Section: KV Cache Inference ===\n");
    }
    InferenceResult kv_result;
    inference_result_init(&kv_result, genT, context_len, B);
    inference_scratch_init(&scratch, &model, B);
    inference_kvcache(&model, &tokenizer, B, gen_tokens, genT, context_len,
                      coins, &kv_result, &scratch,
                      main_stream,
                        /*use_argmax=*/use_argmax,
                        validation_mode);
    double kv_tps = (double)(B * genT) / (kv_result.total_ms / 1000.0);
    if (quiet_output) {
        printf0("kvcache_tokens_per_sec: %.2f\n", kv_tps);
    } else {
        printf0("\nKV Cache implementation: total time = %.2f ms, forward time = %.2f ms\n",
                kv_result.total_ms, kv_result.forward_ms);
        printf0("Token per second: %.2f\n", kv_tps);
        printf0("Generated tokens:\n");
        for(int b = 0; b < B; b++) {
            printf0("Batch %d:\n", b);
            for (int t = 0; t < genT; t++) {
                safe_printf(tokenizer_decode(&tokenizer, kv_result.tokens[b * context_len + t]));
            }
            printf0("\n===\n");
        }
    }

    #ifdef ENABLE_CUDNN
    /* ----------------
     run inference with flash attention (cuDNN) implementation
    ---------------- */
    InferenceResult flash_result;
    inference_result_init(&flash_result, genT, context_len, B);
    inference_scratch_init(&scratch, &model, B);
    inference_flash_attention(&model, &tokenizer, B, gen_tokens, genT, context_len,
                      coins, &flash_result, &scratch,
                      main_stream,
                        /*use_argmax=*/use_argmax,
                        validation_mode);
    printf0("\nFlash Attention (cuDNN) implementation: total time = %.2f ms, forward time = %.2f ms\n",
            flash_result.total_ms, flash_result.forward_ms);
    printf0("Token per second: %.2f\n",
            (double)(B * genT) / (flash_result.total_ms / 1000.0));
    printf0("Generated tokens:\n");
    for(int b = 0; b < B; b++) {
        printf0("Batch %d:\n", b);
        for (int t = 0; t < genT; t++) {
            safe_printf(tokenizer_decode(&tokenizer, flash_result.tokens[b * context_len + t]));
        }
        printf0("\n===\n");
    }
    #endif

    /* ---------------
        compare results
    --------------- */
    if (use_argmax) {
        printf("\nUsing argmax sampling, outputs should match almost identically.\n");
    } else {
        printf("\nUsing random sampling, outputs will differ due to different numerical behavior.\n");
    }
    printf("\n\nComparing base and KV cache implementations...\n");
    int mismatch = 0;
    for (int b = 0; b < B; b++) {
        printf("\n Batch %d:\n", b);
        for (int t = 0; t < genT; t++) {
            if (base_result.tokens[b * context_len + t] != kv_result.tokens[b * context_len + t]) {
                printf("Mismatch at t=%d: base=%d kv=%d\n", t, base_result.tokens[b * context_len + t], kv_result.tokens[b * context_len + t]);
                mismatch++;
                if (mismatch >= 20) break;
            }
            #ifdef ENABLE_CUDNN
            if (base_result.tokens[b * context_len + t] != flash_result.tokens[b * context_len + t]) {
                printf("Mismatch at t=%d: base=%d flash=%d\n", t, base_result.tokens[b * context_len + t], flash_result.tokens[b * context_len + t]);
                mismatch++;
                if (mismatch >= 20) break;
            }
            #endif
        }
    }
    printf("Token mismatches: %d (showing up to 20)\n", mismatch);
    
    /* ---------------
        cleanup
    --------------- */
    // results and scratch
    inference_result_free(&base_result);
    inference_result_free(&kv_result);
    inference_scratch_free(&scratch);
    
    // model and tokenizer
    tokenizer_free(&tokenizer);
    free(gen_tokens);
    gpt2_free(&model);
    multi_gpu_config_free(&multi_gpu_config);
    common_free(model);
    return 0;
}
#endif
