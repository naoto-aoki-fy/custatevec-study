CUDA_ROOT ?= /usr/local/cuda
CUQUANTUM_ROOT ?= /opt/nvidia/cuquantum
CUQUANTUM_LIB_DIR ?= $(CUQUANTUM_ROOT)/lib
NVCC ?= $(CUDA_ROOT)/bin/nvcc
CPPFLAGS ?=
NVCCFLAGS ?= -O3 -std=c++17
LDFLAGS ?=
LDLIBS ?= -lcustatevec -lcudart

.PHONY: all clean
.DELETE_ON_ERROR:

all: hadamard

hadamard: hadamard.cu Makefile
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -I"$(CUQUANTUM_ROOT)/include" $< -o $@ $(LDFLAGS) -L"$(CUQUANTUM_LIB_DIR)" -Xlinker -rpath -Xlinker "$(CUQUANTUM_LIB_DIR)" $(LDLIBS)

clean:
	rm -f hadamard
