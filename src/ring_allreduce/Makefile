# Build the custom Ring AllReduce benchmark.
#
#   make                       # uses /usr/local/cuda and the system NCCL
#   make NCCL_HOME=/path/nccl  # link a specific NCCL (e.g. a pip-installed one)
#   make SM=89                 # target one architecture explicitly
#
# NCCL is linked as a correctness oracle and performance reference only; the
# custom ring never calls it.

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
SM        ?=
BUILD     ?= build

CXXFLAGS  = -O3 -std=c++17 -lineinfo
INCLUDES  = -I$(CUDA_HOME)/include
LDFLAGS   = -L$(CUDA_HOME)/lib64 -lcudart -lnccl

ifneq ($(NCCL_HOME),)
  INCLUDES += -I$(NCCL_HOME)/include
  LDFLAGS  += -L$(NCCL_HOME)/lib -Wl,-rpath,$(NCCL_HOME)/lib
endif

ifneq ($(SM),)
  GENCODE = -gencode=arch=compute_$(SM),code=sm_$(SM)
endif

all: $(BUILD)/ring_allreduce

$(BUILD)/ring_allreduce: ring_allreduce.cu | $(BUILD)
	$(NVCC) $(CXXFLAGS) $(GENCODE) $(INCLUDES) $< -o $@ $(LDFLAGS)

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all clean
