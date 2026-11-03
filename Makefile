.DEFAULT_GOAL := all
CONFIG_MK ?= config.mk

# clean and help are usable even before a site configuration is supplied.
ifneq ($(strip $(filter-out clean help,$(MAKECMDGOALS))$(if $(MAKECMDGOALS),,all)),)
ifeq ($(wildcard $(CONFIG_MK)),)
$(error Missing CONFIG_MK='$(CONFIG_MK)'. Supply an existing nvccoptions config.mk; see README.md)
endif
include $(CONFIG_MK)
$(foreach setting,CFLAGS_VENDOR LDFLAGS_VENDOR GENCODE_FLAGS,$(if $(filter undefined,$(origin $(setting))),$(error $(setting) is not defined in $(CONFIG_MK))))
endif

NVCC ?= nvcc --forward-unknown-to-host-compiler
PROJECT_FLAGS ?= -O3 -std=c++17
EXTRA_CPPFLAGS ?=
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS ?=
CUSTATEVEC_ROOT ?= /usr/local/cuquantum
CUSTATEVEC_INCLUDEDIR ?= $(CUSTATEVEC_ROOT)/include
CUSTATEVEC_LIBDIR ?= $(CUSTATEVEC_ROOT)/lib
CUSTATEVEC_CPPFLAGS ?= -I$(CUSTATEVEC_INCLUDEDIR)
CUSTATEVEC_LDFLAGS ?= -L$(CUSTATEVEC_LIBDIR) -Xlinker -rpath -Xlinker $(CUSTATEVEC_LIBDIR) -lcustatevec
SOURCES := src/main.cu src/runtime.cu src/benchmark.cu
OBJECTS := $(SOURCES:src/%.cu=build/%.o)

.PHONY: all clean help dependencies
all: custatevec_bench

dependencies:
	@command -v $(firstword $(NVCC)) >/dev/null 2>&1 || { echo 'Missing nvcc: load the CUDA compiler environment or set NVCC.' >&2; exit 1; }
	@test -r '$(CUSTATEVEC_INCLUDEDIR)/custatevec.h' || { echo 'Missing custatevec.h: set CUSTATEVEC_ROOT or CUSTATEVEC_INCLUDEDIR.' >&2; exit 1; }

build/%.o: src/%.cu include/bench.hpp $(CONFIG_MK) Makefile | dependencies
	@mkdir -p build
	$(NVCC) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) $(PROJECT_FLAGS) -Iinclude $(CUSTATEVEC_CPPFLAGS) $(EXTRA_CPPFLAGS) $(EXTRA_NVCCFLAGS) -MMD -MP -c $< -o $@

custatevec_bench: $(OBJECTS)
	$(NVCC) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) $^ $(LDFLAGS_VENDOR) $(CUSTATEVEC_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

clean:
	rm -rf build custatevec_bench

help:
	@echo 'make CONFIG_MK=/path/to/existing/config.mk CUSTATEVEC_ROOT=/path/to/cuquantum'
	@echo 'See README.md for compiler, MPI, library, and additional flag overrides.'

-include $(OBJECTS:.o=.d)
