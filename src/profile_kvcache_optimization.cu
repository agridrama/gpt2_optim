/*
- This file is for profiling gpt2_decode_step.
- It runs only one forward pass with KV cache optimization.
- ncu or nvprof can be used to profile this program.
- For simplicity, we avoid some sanity checks and command-line argument parsing.

- build example:
    nvcc --threads=0 -t=0 --use_fast_math -std=c++17 -O3 --generate-code \
    arch=compute_75,code=[compute_75,sm_75] \ 
    -DENABLE_FP32 profile_kvcache_optimization.cu \
    -lcublas -lcublasLt -lnvidia-ml -lnvToolsExt -o profile_kvcache_optimization

- Usage example:
    NSYS_NVTX_PROFILER_REGISTER_ONLY=0 && nsys profile -t cuda,nvtx --capture-range=nvtx --nvtx-capture=MEASURE --capture-range-end=stop-shutdown -o prof ./profile_kvcache_optimization
- Using NVTX ranges, the region to be profiled is marked with "MEASURE".
*/

#define TESTING
#include "inference_gpt2_optimize.cu"

int main(int argc, char *argv[]) {
    /* ----------------
     argument parsing
    ---------------- */
    const char* load_filename = "gpt2_124M.bin";
    const char* tokenizer_path = "gpt2_tokenizer.bin";

    int num_runs = 10; // number of runs for performance measurement
    int num_warmup = 3; // number of warm-up runs

    int B = 4;
    int genT = 64;
    int context_len = -1;
    int override_enable_tf32 = 1;
        for (int i = 1; i < argc; i += 2) {
        // if (i + 1 >= argc || argv[i][0] != '-') { error_usage(); }
        std::string flag(argv[i]);
        if (flag == "-g") { genT = atoi(argv[i+1]); }
        else if (flag == "-b") { B = atoi(argv[i+1]); }
        // else { error_usage(); }
    }

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
    
    context_len = model.config.max_seq_len; // 1024

    if (genT > context_len) {
        printf0("Warning: genT=%d > context_len=%d, clipping to context length\n", genT, context_len);
        genT = context_len;
    }


    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, tokenizer_path);

    gpt2_allocate_state(&model, B, context_len);

    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;

    
    // KV cache optimization
    {
        // KV cache init
        gpt2_kvcache_init(&model, B, context_len); 
        DecodeState decode_st;
        decode_state_init(&decode_st, &model, B);
        int* d_token_ids;
        int* h_token_ids;
        cudaCheck(cudaMalloc(&d_token_ids, (size_t)B * sizeof(int)));
        cudaCheck(cudaMallocHost((void **)&h_token_ids, (size_t)B * sizeof(int)));
        // copy initial tokens
        for (int b = 0; b < B; b++) {
            h_token_ids[b] = (b*100) % V;
        }
        cudaCheck(cudaMemcpy(d_token_ids, h_token_ids, (size_t)B * sizeof(int), cudaMemcpyHostToDevice));
        
        floatX* d_optimized_logits;
        cudaCheck(cudaMalloc(&d_optimized_logits, (size_t)(B * Vp) * sizeof(floatX)));

        // run warm-up and profiled runs
        for (int run = 0; run < num_warmup; run++) {
            gpt2_decode_step(&model, &decode_st, d_token_ids, B, 0, d_optimized_logits, main_stream);
        }
        cudaDeviceSynchronize();

        nvtxRangePush("MEASURE");
        for (int run = 0; run < num_runs; run++) {
            gpt2_decode_step(&model, &decode_st, d_token_ids, B, run, d_optimized_logits, main_stream);

            // random token ids for next step, this is fast and does not affect the profiling
            for (int b = 0; b < B; b++) {
                h_token_ids[b] = (b * 43 + run * 12) % model.config.vocab_size;
            }
            cudaCheck(cudaMemcpy(d_token_ids, h_token_ids, (size_t)B * sizeof(int), cudaMemcpyHostToDevice));
        }
        cudaDeviceSynchronize();
        nvtxRangePop();
    
        decode_state_free(&decode_st);
        gpt2_kvcache_free(&model);
        cudaFree(d_token_ids);
        cudaFreeHost(h_token_ids);
        cudaFree(d_optimized_logits);
    }

    // cleanup

    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    multi_gpu_config_free(&multi_gpu_config);
    common_free(model);
    return 0;
}
