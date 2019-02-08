.DELETE_ON_ERROR:
.PHONY: all test test-% time time-% clean

V = 0
AT_0 := @
AT_1 :=
AT = $(AT_$(V))
OUTDIR?=/var/scratch/mverstra/pipeline

ifeq ($(V), 1)
    PRINTF := @\#
else
    PRINTF := @printf
endif

WARNINGS := -Wall -Wextra -Wpedantic -Wconversion -Wno-unknown-pragmas -std=c++17

ifeq ($(CXX), clang++)
    WARNINGS+=-Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic \
              -Wno-disabled-macro-expansion -Wno-global-constructors
    CFLAGS+=-isystem/cm/shared/apps/intel/composer_xe/2015.5.223/mkl/include/ \
            -stdlib=libc++
    LDFLAGS+=-stdlib=libc++
endif

VARIANTS := pipeline-delay-bandpass pipeline-bandpass pipeline-delay \
    pipeline-none
.PRECIOUS: $(VARIANTS:%=$(OUTDIR)/%.test)

if-contains = $(if $(findstring $(2),$(1)), $(3))

all: $(VARIANTS)

test: $(VARIANTS:pipeline-%=test-%)

test-%: $(OUTDIR)/pipeline-%.test
	diff -q $(OUTDIR)/$*.reference $<

time: $(VARIANTS:pipeline-%=time-%)

time-%: pipeline-%
	$(AT)prun -np 1 time -f "$*: real = %e, user = %U, sys = %S" ./$< time

clean:
	$(AT)rm $(VARIANTS)

CFLAGS+=$(WARNINGS)

pipeline-%.o: CFLAGS+=$(call if-contains,$*,delay,-DDELAY_COMPENSATION) \
    $(call if-contains,$*,bandpass,-DBANDPASS_CORRECTION)

pipeline-%.o: pipeline.cpp
	$(PRINTF) " CC $<\n"
	$(AT)$(CXX) $(CFLAGS) -fopenmp -g -O3 -c -o $@ $<

pipeline-%: pipeline-%.o
	$(PRINTF) " LD $@\n"
	$(AT)$(CXX) $(LDFLAGS) -fopenmp -o $@ $< -L$(MKL_LIB) -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread

$(OUTDIR)/pipeline-%.test: pipeline-%
	$(PRINTF) " TEST $<\n"
	$(AT)prun -np 1 ./$< >$@
