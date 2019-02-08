.DELETE_ON_ERROR:
.PHONY: all test test-% time time-% clean

V = 0
AT_0 := @
AT_1 :=
AT = $(AT_$(V))
OUTDIR?=/var/scratch/mverstra/pipeline
BUILDDIR?=./.build

ifeq ($(V), 1)
    PRINTF := @\#
else
    PRINTF := @printf
endif

CFLAGS := -MMD -MP -g -O3 -std=c++17
WARNINGS := -Wall -Wextra -Wpedantic -Wconversion -Wno-unknown-pragmas

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

-include $(patsubst %.cpp, .build/%.d, $(wildcard *.cpp))

$(BUILDDIR)/:
	$(AT)mkdir -p $@

$(BUILDDIR)/correlator-%.o: CFLAGS+= \
    $(call if-contains,$*,delay,-DDELAY_COMPENSATION) \
    $(call if-contains,$*,bandpass,-DBANDPASS_CORRECTION)

$(BUILDDIR)/correlator-%.o: correlator.cpp | $(BUILDDIR)/
	$(PRINTF) " CC $<\n"
	$(AT)$(CXX) $(CFLAGS) -fopenmp -g -O3 -c -o $@ $<

$(BUILDDIR)/pipeline.o: pipeline.cpp | $(BUILDDIR)/
	$(PRINTF) " CC $<\n"
	$(AT)$(CXX) $(CFLAGS) -fopenmp -g -O3 -c -o $@ $<

pipeline-%: $(BUILDDIR)/pipeline.o $(BUILDDIR)/correlator-%.o
	$(PRINTF) " LD $@\n"
	$(AT)$(CXX) $(LDFLAGS) -fopenmp -o $@ $^ -L$(MKL_LIB) -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread

$(OUTDIR)/pipeline-%.test: pipeline-%
	$(PRINTF) " TEST $<\n"
	$(AT)prun -np 1 ./$< >$@
