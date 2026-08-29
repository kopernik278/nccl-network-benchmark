# Build the Phase 7B communication/compute overlap benchmark.
#   make                 # measured binary
#   make USE_NVTX=1      # profiled binary (never used for reported numbers)
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
SM        ?=
BUILD     ?= build
CXXFLAGS  = -O3 -std=c++17 -lineinfo
INCLUDES  = -I$(CUDA_HOME)/include
LDFLAGS   = -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lnccl
ifneq ($(NCCL_HOME),)
  INCLUDES += -I$(NCCL_HOME)/include
  LDFLAGS  += -L$(NCCL_HOME)/lib -Wl,-rpath,$(NCCL_HOME)/lib
endif
ifneq ($(SM),)
  GENCODE = -gencode=arch=compute_$(SM),code=sm_$(SM)
endif
ifeq ($(USE_NVTX),1)
  CXXFLAGS += -DUSE_NVTX=1
endif
all: $(BUILD)/overlap_bench
$(BUILD)/overlap_bench: overlap_bench.cu | $(BUILD)
	$(NVCC) $(CXXFLAGS) $(GENCODE) $(INCLUDES) $< -o $@ $(LDFLAGS)
$(BUILD):
	mkdir -p $(BUILD)
clean:
	rm -rf $(BUILD)
.PHONY: all clean
