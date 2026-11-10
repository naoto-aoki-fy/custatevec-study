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

.PHONY: all clean help dependencies library test-qcs test-qcs-gates test-qcs-measurement
all: custatevec_bench

dependencies:
	@command -v $(firstword $(NVCC)) >/dev/null 2>&1 || { echo 'Missing nvcc: load the CUDA compiler environment or set NVCC.' >&2; exit 1; }
	@test -r '$(CUSTATEVEC_INCLUDEDIR)/custatevec.h' || { echo 'Missing custatevec.h: set CUSTATEVEC_ROOT or CUSTATEVEC_INCLUDEDIR.' >&2; exit 1; }

build/%.o: src/%.cu include/bench.hpp include/device_assignment.hpp include/transfer_policy.hpp $(CONFIG_MK) Makefile | dependencies
	@mkdir -p build
	$(NVCC) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) $(PROJECT_FLAGS) -Iinclude $(CUSTATEVEC_CPPFLAGS) $(EXTRA_CPPFLAGS) $(EXTRA_NVCCFLAGS) -MMD -MP -c $< -o $@

custatevec_bench: $(OBJECTS)
	$(NVCC) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) $^ $(LDFLAGS_VENDOR) $(CUSTATEVEC_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

# Separate PIC object keeps benchmark compile and link flags unchanged.
library: libqcs.so

build/qcs.o: src/qcs.cu src/qcs_internal.hpp src/qcs_gate.hpp src/qcs_gates.cuh src/qcs_measurement.hpp src/qcs_measurement.cuh include/transfer_policy.hpp include/qcs.h include/device_assignment.hpp $(CONFIG_MK) Makefile | dependencies
	@mkdir -p build
	$(NVCC) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) $(PROJECT_FLAGS) -Iinclude $(CUSTATEVEC_CPPFLAGS) $(EXTRA_CPPFLAGS) $(EXTRA_NVCCFLAGS) -Xcompiler -fPIC -MMD -MP -c $< -o $@

libqcs.so: build/qcs.o
	$(NVCC) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -shared $^ $(LDFLAGS_VENDOR) $(CUSTATEVEC_LDFLAGS) -lcurand $(EXTRA_LDFLAGS) -o $@

build/qcs_c_caller.o: tests/qcs_c_caller.c include/qcs.h
	@mkdir -p build
	$(CC) -std=c11 -Iinclude -c $< -o $@

build/qcs_foundation_test: tests/qcs_foundation_test.cu src/qcs_internal.hpp include/device_assignment.hpp build/qcs_c_caller.o libqcs.so
	$(NVCC) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) $(PROJECT_FLAGS) -Iinclude $(CUSTATEVEC_CPPFLAGS) $(EXTRA_CPPFLAGS) $(EXTRA_NVCCFLAGS) $< build/qcs_c_caller.o -L. -lqcs $(LDFLAGS_VENDOR) $(CUSTATEVEC_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

test-qcs: build/qcs_foundation_test
	LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH ./build/qcs_foundation_test

build/qcs_gate_test: tests/qcs_gate_test.cu tests/qcs_gate_catalog.hpp tests/qcs_gate_reference.hpp src/qcs_gate.hpp src/qcs_internal.hpp libqcs.so
	$(NVCC) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) $(PROJECT_FLAGS) -Iinclude $(CUSTATEVEC_CPPFLAGS) $(EXTRA_CPPFLAGS) $(EXTRA_NVCCFLAGS) $< -L. -lqcs $(LDFLAGS_VENDOR) $(CUSTATEVEC_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

test-qcs-gates: build/qcs_gate_test
	LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH ./build/qcs_gate_test

build/qcs_measurement_c.o: tests/qcs_measurement_c.c include/qcs.h
	@mkdir -p build
	$(CC) -std=c11 -Iinclude -c $< -o $@

build/qcs_measurement_test: tests/qcs_measurement_test.cu src/qcs_internal.hpp build/qcs_measurement_c.o libqcs.so
	$(NVCC) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) $(PROJECT_FLAGS) -Iinclude $(CUSTATEVEC_CPPFLAGS) $(EXTRA_CPPFLAGS) $(EXTRA_NVCCFLAGS) $< build/qcs_measurement_c.o -L. -lqcs $(LDFLAGS_VENDOR) $(CUSTATEVEC_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

test-qcs-measurement: build/qcs_measurement_test
	LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH ./build/qcs_measurement_test

clean:
	rm -rf build custatevec_bench libqcs.so bin/qcs

help:
	@echo 'make CONFIG_MK=/path/to/existing/config.mk CUSTATEVEC_ROOT=/path/to/cuquantum'
	@echo 'See README.md for compiler, MPI, library, and additional flag overrides.'

-include $(OBJECTS:.o=.d)

-include build/qcs.d

.PHONY: test-qcs-observation
build/qcs_observation_test: tests/qcs_observation_test.cu src/qcs_internal.hpp libqcs.so
	$(NVCC) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) $(PROJECT_FLAGS) -Iinclude $(CUSTATEVEC_CPPFLAGS) $(EXTRA_CPPFLAGS) $(EXTRA_NVCCFLAGS) $< -L. -lqcs $(LDFLAGS_VENDOR) $(CUSTATEVEC_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

test-qcs-observation: build/qcs_observation_test
	LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH ./build/qcs_observation_test

build/qcs.o: src/qcs_observation.hpp

# Host-only CLI sources; vendor nvcc flags stay on the nvcc driver.
# HOST_CFLAGS accepts ordinary host compiler flags, not nvcc -Xcompiler flags.
HOST_CFLAGS ?= -O2 -std=c11

build/qcs_args.o: src/qcs_args.c src/qcs_args.h Makefile
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(HOST_CFLAGS) -Isrc -MMD -MP -c $< -o $@

build/qcs_main.o: src/qcs_main.cpp src/qcs_args.h include/qcs.h $(CONFIG_MK) Makefile
	@mkdir -p build
	$(NVCC) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) $(PROJECT_FLAGS) $(EXTRA_CPPFLAGS) $(EXTRA_NVCCFLAGS) -Iinclude -Isrc -MMD -MP -c $< -o $@

bin/qcs: build/qcs_main.o build/qcs_args.o libqcs.so
	@mkdir -p bin
	$(NVCC) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) build/qcs_main.o build/qcs_args.o -L. -lqcs $(LDFLAGS_VENDOR) $(CUSTATEVEC_LDFLAGS) -ldl -Xlinker -rpath -Xlinker '$$ORIGIN/..' $(EXTRA_LDFLAGS) -o $@

-include build/qcs_main.d build/qcs_args.d
