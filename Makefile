# GPT-2 inference optimization experiments (KV cache, speed compare, profiling)

NVCC ?= $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
  $(error nvcc not found in PATH)
endif

ROOT := $(abspath .)
SRC_DIR := $(ROOT)/src
BIN_DIR := $(ROOT)/bin

# Path to karpathy/llm.c repo root (must contain llmc/)
LLM_C_ROOT ?= $(abspath ../llm.c)

FORCE_NVCC_O ?= 3
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O)
NVCC_INCLUDES = -I$(SRC_DIR) -I$(LLM_C_ROOT)
NVCC_LDFLAGS = -lcublas -lcublasLt -lnvidia-ml -lnvToolsExt
NVCC_LDLIBS =

ifeq ($(GPU_COMPUTE_CAPABILITY),)
  $(error GPU_COMPUTE_CAPABILITY is required, e.g. GPU_COMPUTE_CAPABILITY=75)
endif
NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]

PRECISION ?= BF16
VALID_PRECISIONS := FP32 FP16 BF16
ifeq ($(filter $(PRECISION),$(VALID_PRECISIONS)),)
  $(error Invalid PRECISION $(PRECISION), choose from $(VALID_PRECISIONS))
endif
ifeq ($(PRECISION), BF16)
  PFLAGS = -DENABLE_BF16
else ifeq ($(PRECISION), FP16)
  PFLAGS = -DENABLE_FP16
else
  PFLAGS = -DENABLE_FP32
endif


INFERENCE_SRC := $(SRC_DIR)/inference_gpt2_optimize.cu
VALIDATE_SRC := $(SRC_DIR)/validate_kvcache_optimization.cu
PROFILE_SRC := $(SRC_DIR)/profile_kvcache_optimization.cu

INFERENCE_BIN := $(BIN_DIR)/inference_gpt2optimcu
INFERENCE_BIN_KVONLY := $(BIN_DIR)/inference_gpt2optimcu_kvonly
VALIDATE_BIN := $(BIN_DIR)/validate_kvcache_optimization
PROFILE_BIN := $(BIN_DIR)/profile_kvcache_optimization

INFERENCE_DEPS := $(INFERENCE_SRC) $(wildcard $(SRC_DIR)/inference_optimize/*.cu) $(wildcard $(SRC_DIR)/inference_optimize/*.cuh)

.PHONY: all clean

all: $(INFERENCE_BIN) $(INFERENCE_BIN_KVONLY) $(VALIDATE_BIN) $(PROFILE_BIN)

$(INFERENCE_BIN): $(INFERENCE_DEPS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $(INFERENCE_SRC) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

$(INFERENCE_BIN_KVONLY): $(INFERENCE_DEPS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -DDISABLE_OTHER_OPT $(NVCC_INCLUDES) $(INFERENCE_SRC) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

$(VALIDATE_BIN): $(VALIDATE_SRC) $(INFERENCE_DEPS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $(VALIDATE_SRC) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

$(PROFILE_BIN): $(PROFILE_SRC) $(INFERENCE_DEPS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $(PROFILE_SRC) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

clean:
	rm -f $(INFERENCE_BIN) $(INFERENCE_BIN_KVONLY) $(VALIDATE_BIN) $(PROFILE_BIN)
