.DELETE_ON_ERROR:
.PHONY: all clean

V = 0
AT_0 := @
AT_1 :=
AT = $(AT_$(V))

ifeq ($(V), 1)
    PRINTF := @\#
else
    PRINTF := @printf
endif

all: pipeline
clean:
	$(AT)rm pipeline

pipeline: pipeline.cpp
	$(AT)g++ -fopenmp -mcmodel=large -fPIC -march=sandybridge -g -O3 -o $@ $< -L$(MKL_LIB) -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread

