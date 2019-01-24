// (C) 2013,2014,2015 John Romein/ASTRON
// Copyright 2018-2019 Netherlands eScience Center and ASTRON
// Licensed under the Apache License, version 2.0. See LICENSE for details.

#if defined __AVX512F__
#define _mm512_storenrngo_ps _mm512_stream_ps
#endif

#if !defined NR_INPUTS
#define NR_INPUTS			(2*576)
#endif

#if !defined NR_CHANNELS
#define NR_CHANNELS			64
#endif

#if !defined NR_SAMPLES_PER_CHANNEL
#define NR_SAMPLES_PER_CHANNEL		3072
#endif

#define SUBBAND_BANDWIDTH		195312.5f

#if defined __AVX512F__ || defined __MIC__
#undef USE_FUSED_FILTER
#else
#endif

#define NR_STREAMS			1

#if defined __AVX512F__ || defined __MIC__
#define VECTOR_SIZE			16
#else
#define VECTOR_SIZE			8
#endif

#define NR_TAPS				16
#define NR_BASELINES			(NR_INPUTS * (NR_INPUTS + 1) / 2)

#if defined __AVX512F__ && !defined USE_FUSED_FILTER
#define NR_SAMPLES_PER_MINOR_LOOP	NR_SAMPLES_PER_CHANNEL
#else
#define NR_SAMPLES_PER_MINOR_LOOP	64
#endif

#define REAL				0
#define IMAG				1
#define COMPLEX				2

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))
#define NR_SAMPLES			((uint64_t) NR_INPUTS * NR_SAMPLES_PER_CHANNEL * NR_CHANNELS)

#include <cassert>
#include <cstdio>
#include <cstring>
#include <complex>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <fcntl.h>
#include <immintrin.h>
#include <mkl.h>
#include <omp.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined __AVX512F__
static inline __m512 load_8_bit_samples(const signed char *ptr)
{
  return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(* (const __m128i *) ptr));
}
#elif defined __MIC__
static inline __m512 load_8_bit_samples(const signed char *ptr)
{
  return _mm512_extload_ps(ptr, _MM_UPCONV_PS_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
}
#endif

std::ostream& cout = std::cout;

std::ostream& cerr = std::cerr;

std::ostream& clog = std::clog;


static inline double rdtsc()
{
  unsigned low, high;

  __asm__ __volatile__ ("rdtsc" : "=a" (low), "=d" (high));
  return static_cast<double>(((unsigned long long) high << 32) | low);
}


#if defined MEASURE_POWER
#include "../../LikwidPowerSensor/libPowerSensor.h"
#else

class PowerSensor
{
  public:
    typedef double State;

    State read() { return omp_get_wtime(); }
    void mark(const State &, const char *) { }
    static double Joules(const State &, const State &) { return 0; }
    static double seconds(const State &firstState, const State &secondState) { return secondState - firstState; }
    static double Watt(const State &, const State &) { return 0; }
};

#endif


typedef int8_t InputDataType[NR_INPUTS][COMPLEX][NR_SAMPLES_PER_CHANNEL + NR_TAPS - 1][NR_CHANNELS] __attribute__((aligned(16)));
typedef float FilteredDataType[ALIGN(NR_INPUTS, VECTOR_SIZE)][NR_SAMPLES_PER_CHANNEL][COMPLEX][NR_CHANNELS] __attribute__((aligned(64)));
typedef float FilterWeightsType[NR_TAPS][NR_CHANNELS] __attribute__((aligned(64)));
typedef float BandPassCorrectionWeights[NR_CHANNELS] __attribute__((aligned(64)));
typedef double DelaysType[NR_INPUTS];
typedef float CorrectedDataType[NR_CHANNELS][ALIGN(NR_INPUTS, VECTOR_SIZE) / VECTOR_SIZE][NR_SAMPLES_PER_CHANNEL][COMPLEX][VECTOR_SIZE] __attribute__((aligned(64)));
typedef float VisibilitiesType[NR_CHANNELS][COMPLEX][NR_BASELINES];


static InputDataType inputData[NR_STREAMS];
#if defined CORRECTNESS_TEST
static FilteredDataType filteredData;
#endif
static FilterWeightsType filterWeights;
static BandPassCorrectionWeights bandPassCorrectionWeights;
static DelaysType delaysAtBegin[NR_STREAMS], delaysAfterEnd[NR_STREAMS];
static CorrectedDataType correctedData;
static VisibilitiesType visibilities[NR_STREAMS + 1]; // this is really too much, but avoids a potential segfault on as (masked!!!) vpackstorehps
static uint64_t totalNrOperations;

static PowerSensor powerSensor;


#if defined __AVX__ && !defined __MIC__

std::ostream &operator << (std::ostream &str, __m256 v)
{
  str << '[';

  for (int i = 0; i < 8; i ++)
    str << ((float *) &v)[i] << (i == 7 ? ']' : ',');

  return str;
}

#endif


////// FIR filter

#if defined CORRECTNESS_TEST
static void setInputTestPattern(InputDataType inputData)
{
  signed char count = 64;

  for (unsigned input = 0; input < NR_INPUTS; input ++)
    for (unsigned ri = 0; ri < COMPLEX; ri ++)
      for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL + NR_TAPS - 1; time ++)
	for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
	  inputData[input][ri][time][channel] = count ++;

  if (NR_INPUTS > 9 && NR_SAMPLES_PER_CHANNEL > 99 && NR_CHANNELS > 12) {
    inputData[9][REAL][98 + NR_TAPS - 1][12] = 4;
    inputData[9][REAL][99 + NR_TAPS - 1][12] = 5;
  }
}


static void setFilterWeightsTestPattern(FilterWeightsType filterWeights)
{
  memset(filterWeights, 0, sizeof(FilterWeightsType));

  if (NR_TAPS > 4 && NR_CHANNELS > 12) {
    filterWeights[15][12] = 2;
    filterWeights[14][12] = 3;
  }
}


static void filter(FilteredDataType filteredData, const InputDataType inputData, const FilterWeightsType filterWeights, unsigned)
{
#pragma omp parallel
  {
#if defined __AVX512F__ || defined __MIC__
#pragma noprefetch
#pragma omp for collapse(3) schedule(dynamic)
    for (int input = 0; input < NR_INPUTS; input ++) {
      for (int real_imag = 0; real_imag < COMPLEX; real_imag ++) {
       for (int time_split = 0; time_split < NR_SAMPLES_PER_CHANNEL; time_split += 256) {
	for (int channelBase = 0; channelBase < NR_CHANNELS; channelBase += 16) {
#if 0
	  for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	    //_mm_prefetch((const char *) &inputData[input][real_imag][time + (NR_TAPS - 1) + 8][channelBase], _MM_HINT_T1);
	    //_mm_prefetch((const char *) &inputData[input][real_imag][time + (NR_TAPS - 1) + 1][channelBase], _MM_HINT_T0);

	    __m512 sum = _mm512_setzero_ps();

	    for (int tap = 0; tap < NR_TAPS; tap ++)
	      sum = _mm512_fmadd_ps(* (__m512 *) &filterWeights[tap][channelBase] , load_8_bit_samples(&inputData[input][real_imag][time + tap][channelBase]), sum);

	    _mm512_storenrngo_ps(&filteredData[input][time][real_imag][channelBase], sum);
	  }
#else
	  __m512 weight_0 = * (__m512 *) &filterWeights[ 0][channelBase];
	  __m512 weight_1 = * (__m512 *) &filterWeights[ 1][channelBase];
	  __m512 weight_2 = * (__m512 *) &filterWeights[ 2][channelBase];
	  __m512 weight_3 = * (__m512 *) &filterWeights[ 3][channelBase];
	  __m512 weight_4 = * (__m512 *) &filterWeights[ 4][channelBase];
	  __m512 weight_5 = * (__m512 *) &filterWeights[ 5][channelBase];
	  __m512 weight_6 = * (__m512 *) &filterWeights[ 6][channelBase];
	  __m512 weight_7 = * (__m512 *) &filterWeights[ 7][channelBase];
	  __m512 weight_8 = * (__m512 *) &filterWeights[ 8][channelBase];
	  __m512 weight_9 = * (__m512 *) &filterWeights[ 9][channelBase];
	  __m512 weight_A = * (__m512 *) &filterWeights[10][channelBase];
	  __m512 weight_B = * (__m512 *) &filterWeights[11][channelBase];
	  __m512 weight_C = * (__m512 *) &filterWeights[12][channelBase];
	  __m512 weight_D = * (__m512 *) &filterWeights[13][channelBase];
	  __m512 weight_E = * (__m512 *) &filterWeights[14][channelBase];
	  __m512 weight_F = * (__m512 *) &filterWeights[15][channelBase];

	  __m512 samples_0;
	  __m512 samples_1 = load_8_bit_samples(&inputData[input][real_imag][ time_split + 0][channelBase]);
	  __m512 samples_2 = load_8_bit_samples(&inputData[input][real_imag][ time_split + 1][channelBase]);
	  __m512 samples_3 = load_8_bit_samples(&inputData[input][real_imag][ time_split + 2][channelBase]);
	  __m512 samples_4 = load_8_bit_samples(&inputData[input][real_imag][ time_split + 3][channelBase]);
	  __m512 samples_5 = load_8_bit_samples(&inputData[input][real_imag][ time_split + 4][channelBase]);
	  __m512 samples_6 = load_8_bit_samples(&inputData[input][real_imag][ time_split + 5][channelBase]);
	  __m512 samples_7 = load_8_bit_samples(&inputData[input][real_imag][ time_split + 6][channelBase]);
	  __m512 samples_8 = load_8_bit_samples(&inputData[input][real_imag][ time_split + 7][channelBase]);
	  __m512 samples_9 = load_8_bit_samples(&inputData[input][real_imag][ time_split + 8][channelBase]);
	  __m512 samples_A = load_8_bit_samples(&inputData[input][real_imag][ time_split + 9][channelBase]);
	  __m512 samples_B = load_8_bit_samples(&inputData[input][real_imag][ time_split +10][channelBase]);
	  __m512 samples_C = load_8_bit_samples(&inputData[input][real_imag][ time_split +11][channelBase]);
	  __m512 samples_D = load_8_bit_samples(&inputData[input][real_imag][ time_split +12][channelBase]);
	  __m512 samples_E = load_8_bit_samples(&inputData[input][real_imag][ time_split +13][channelBase]);
	  __m512 samples_F = load_8_bit_samples(&inputData[input][real_imag][ time_split +14][channelBase]);

	  __m512 sum;

	  const int8_t *__restrict inputPtr = &inputData[input][real_imag][NR_TAPS - 1][channelBase];
	  float *__restrict outputPtr = &filteredData[input][0][real_imag][channelBase];

	  for (int time = time_split; time < time_split + 256; time += 16) {
#if defined __MIC__
	    for (int t = 0; t < NR_TAPS; t ++) {
	      _mm_prefetch((const char *) inputPtr + NR_CHANNELS * (time + t + 16), _MM_HINT_T2);
	      _mm_prefetch((const char *) inputPtr + NR_CHANNELS * (time + t + 0), _MM_HINT_T0);
	    }
#endif

	    //samples_0 = load_8_bit_samples(&inputData[input][real_imag][time + (NR_TAPS - 1) +  0][channelBase]);
	    samples_0 = load_8_bit_samples(inputPtr + (time +  0) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_1, weight_0);
	    sum = _mm512_fmadd_ps(samples_2, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_F, sum);
	    //_mm512_storenrngo_ps(&filteredData[input][time +  0][real_imag][channelBase], sum);
	    _mm512_storenrngo_ps(outputPtr + (time +  0) * COMPLEX * NR_CHANNELS, sum);

	    samples_1 = load_8_bit_samples(inputPtr + (time +  1) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_2, weight_0);
	    sum = _mm512_fmadd_ps(samples_3, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time +  1) * COMPLEX * NR_CHANNELS, sum);

	    samples_2 = load_8_bit_samples(inputPtr + (time +  2) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_3, weight_0);
	    sum = _mm512_fmadd_ps(samples_4, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time +  2) * COMPLEX * NR_CHANNELS, sum);

	    samples_3 = load_8_bit_samples(inputPtr + (time +  3) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_4, weight_0);
	    sum = _mm512_fmadd_ps(samples_5, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time +  3) * COMPLEX * NR_CHANNELS, sum);

	    samples_4 = load_8_bit_samples(inputPtr + (time +  4) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_5, weight_0);
	    sum = _mm512_fmadd_ps(samples_6, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time +  4) * COMPLEX * NR_CHANNELS, sum);

	    samples_5 = load_8_bit_samples(inputPtr + (time +  5) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_6, weight_0);
	    sum = _mm512_fmadd_ps(samples_7, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time +  5) * COMPLEX * NR_CHANNELS, sum);

	    samples_6 = load_8_bit_samples(inputPtr + (time +  6) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_7, weight_0);
	    sum = _mm512_fmadd_ps(samples_8, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time +  6) * COMPLEX * NR_CHANNELS, sum);

	    samples_7 = load_8_bit_samples(inputPtr + (time +  7) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_8, weight_0);
	    sum = _mm512_fmadd_ps(samples_9, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time +  7) * COMPLEX * NR_CHANNELS, sum);

	    samples_8 = load_8_bit_samples(inputPtr + (time +  8) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_9, weight_0);
	    sum = _mm512_fmadd_ps(samples_A, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time +  8) * COMPLEX * NR_CHANNELS, sum);

	    samples_9 = load_8_bit_samples(inputPtr + (time +  9) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_A, weight_0);
	    sum = _mm512_fmadd_ps(samples_B, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time +  9) * COMPLEX * NR_CHANNELS, sum);

	    samples_A = load_8_bit_samples(inputPtr + (time + 10) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_B, weight_0);
	    sum = _mm512_fmadd_ps(samples_C, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time + 10) * COMPLEX * NR_CHANNELS, sum);

	    samples_B = load_8_bit_samples(inputPtr + (time + 11) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_C, weight_0);
	    sum = _mm512_fmadd_ps(samples_D, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time + 11) * COMPLEX * NR_CHANNELS, sum);

	    samples_C = load_8_bit_samples(inputPtr + (time + 12) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_D, weight_0);
	    sum = _mm512_fmadd_ps(samples_E, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time + 12) * COMPLEX * NR_CHANNELS, sum);

	    samples_D = load_8_bit_samples(inputPtr + (time + 13) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_E, weight_0);
	    sum = _mm512_fmadd_ps(samples_F, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_0, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time + 13) * COMPLEX * NR_CHANNELS, sum);

	    samples_E = load_8_bit_samples(inputPtr + (time + 14) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_F, weight_0);
	    sum = _mm512_fmadd_ps(samples_0, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_1, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time + 14) * COMPLEX * NR_CHANNELS, sum);

	    samples_F = load_8_bit_samples(inputPtr + (time + 15) * NR_CHANNELS);
	    sum = _mm512_mul_ps  (samples_0, weight_0);
	    sum = _mm512_fmadd_ps(samples_1, weight_1, sum);
	    sum = _mm512_fmadd_ps(samples_2, weight_2, sum);
	    sum = _mm512_fmadd_ps(samples_3, weight_3, sum);
	    sum = _mm512_fmadd_ps(samples_4, weight_4, sum);
	    sum = _mm512_fmadd_ps(samples_5, weight_5, sum);
	    sum = _mm512_fmadd_ps(samples_6, weight_6, sum);
	    sum = _mm512_fmadd_ps(samples_7, weight_7, sum);
	    sum = _mm512_fmadd_ps(samples_8, weight_8, sum);
	    sum = _mm512_fmadd_ps(samples_9, weight_9, sum);
	    sum = _mm512_fmadd_ps(samples_A, weight_A, sum);
	    sum = _mm512_fmadd_ps(samples_B, weight_B, sum);
	    sum = _mm512_fmadd_ps(samples_C, weight_C, sum);
	    sum = _mm512_fmadd_ps(samples_D, weight_D, sum);
	    sum = _mm512_fmadd_ps(samples_E, weight_E, sum);
	    sum = _mm512_fmadd_ps(samples_F, weight_F, sum);
	    _mm512_storenrngo_ps(outputPtr + (time + 15) * COMPLEX * NR_CHANNELS, sum);
	  }
#endif
	}
       }
      }
    }
#elif 1 && defined __AVX__
#pragma omp for schedule(dynamic)
    for (unsigned input = 0; input < NR_INPUTS; input ++) {
      float history[COMPLEX][NR_TAPS][NR_CHANNELS] __attribute__((aligned(32)));

      for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++) {
	for (unsigned time = 0; time < NR_TAPS - 1; time ++) {
	  for (unsigned channelBase = 0; channelBase < NR_CHANNELS; channelBase += VECTOR_SIZE) {
	    __m128 eight_signed_chars = _mm_loadl_pi(eight_signed_chars, (__m64 *) &inputData[input][real_imag][time][channelBase]);
	    __m128i eight_signed_chars_i = _mm_castps_si128(eight_signed_chars);
	    __m128i four_high_bytes = _mm_srli_si128(eight_signed_chars_i, 4);
	    __m128i four_low_dwords = _mm_cvtepi8_epi32(eight_signed_chars_i);
	    __m128i four_high_dwords = _mm_cvtepi8_epi32(four_high_bytes);
	    __m256i eight_dwords = _mm256_castsi128_si256(four_low_dwords);
	    eight_dwords = _mm256_insertf128_si256(eight_dwords, four_high_dwords, 1);
	    __m256 eight_floats = _mm256_cvtepi32_ps(eight_dwords);
	    _mm256_store_ps(&history[real_imag][time][channelBase], eight_floats);
	  }
	}
      }

      for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++) {
	  for (unsigned channelBase = 0; channelBase < NR_CHANNELS; channelBase += VECTOR_SIZE) {
	    __m128 eight_signed_chars = _mm_loadl_pi(eight_signed_chars, (__m64 *) &inputData[input][real_imag][time + NR_TAPS - 1][channelBase]);
	    __m128i eight_signed_chars_i = _mm_castps_si128(eight_signed_chars);
	    __m128i four_high_bytes = _mm_srli_si128(eight_signed_chars_i, 4);
	    __m128i four_low_dwords = _mm_cvtepi8_epi32(eight_signed_chars_i);
	    __m128i four_high_dwords = _mm_cvtepi8_epi32(four_high_bytes);
	    __m256i eight_dwords = _mm256_castsi128_si256(four_low_dwords);
	    eight_dwords = _mm256_insertf128_si256(eight_dwords, four_high_dwords, 1);
	    __m256 eight_floats = _mm256_cvtepi32_ps(eight_dwords);
	    _mm256_store_ps(&history[real_imag][(time - 1) % NR_TAPS][channelBase], eight_floats);

	    __m256 sum = _mm256_setzero_ps();

	    for (unsigned tap = 0; tap < NR_TAPS; tap ++) {
	      __m256 weights = _mm256_load_ps(&filterWeights[tap][channelBase]);
	      __m256 samples = _mm256_load_ps(&history[real_imag][(time + tap) % NR_TAPS][channelBase]);
#if defined __AVX2__
	      sum = _mm256_fmadd_ps(weights, samples, sum);
#else
	      sum = _mm256_add_ps(_mm256_mul_ps(weights, samples), sum);
#endif
	    }

	    _mm256_stream_ps(&filteredData[input][time][real_imag][channelBase], sum);
	  }
	}
      }
    }

    _mm_mfence();
#elif 1
#pragma omp for schedule(dynamic)
    for (unsigned input = 0; input < NR_INPUTS; input ++) {
      float history[COMPLEX][NR_TAPS][NR_CHANNELS] __attribute__((aligned(sizeof(float[VECTOR_SIZE]))));

      for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++)
	for (unsigned time = 0; time < NR_TAPS - 1; time ++)
	  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
	    history[real_imag][time][channel] = inputData[input][real_imag][time][channel];

      for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++) {
#pragma simd
	  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
	    history[real_imag][(time - 1) % NR_TAPS][channel] = inputData[input][real_imag][time + NR_TAPS - 1][channel];

	    float sum = 0;

	    for (int tap = 0; tap < NR_TAPS; tap ++)
	      sum += filterWeights[tap][channel] * history[real_imag][(time + tap) % NR_TAPS][channel];

	    filteredData[input][time][real_imag][channel] = sum;
	  }
	}
      }
    }
#else
#pragma omp for collapse(2) schedule(dynamic)
    for (int real_imag = 0; real_imag < COMPLEX; real_imag ++) {
      for (int input = 0; input < NR_INPUTS; input ++) {
	for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	  for (int channel = 0; channel < NR_CHANNELS; channel ++) {
	    float sum = 0;

	    for (int tap = 0; tap < NR_TAPS; tap ++)
	      sum += filterWeights[tap][channel] * inputData[input][real_imag][time + tap][channel];

	    filteredData[input][time][real_imag][channel] = sum;
	  }
	}
      }
    }
#endif
  }
}


static void copyInputData(int stream)
{
  //int8_t *inputDataPtr = &inputData[0][0][0][0];

#if !defined CORRECTNESS_TEST
  double start_time = omp_get_wtime();
#endif

#pragma omp target update to(inputData[stream])

#if defined DELAY_COMPENSATION
#pragma omp target update to(delaysAtBegin[stream], delaysAfterEnd[stream])
#endif

#if !defined CORRECTNESS_TEST
  double copy_time = omp_get_wtime() - start_time;

#pragma omp critical (cout)
  cout << "input data: time = " << copy_time << "s (total), " << "BW = " << sizeof(InputDataType) / copy_time / 1e9 << " GB/s" << std::endl;
#endif
}


static void FIR_filter(int stream, unsigned iteration)
{
#pragma omp target map(to:iteration)
  filter(filteredData, inputData[stream], filterWeights, iteration);
}


static void checkFIR_FilterTestPattern(const FilteredDataType filteredData)
{
  for (unsigned input = 0; input < NR_INPUTS; input ++)
    for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
      for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
	if (filteredData[input][time][REAL][channel] != 0 || filteredData[input][time][IMAG][channel] != 0)
	  cout << "input = " << input << ", time = " << time << ", channel = " << channel << ", sample = (" << filteredData[input][time][REAL][channel] << ',' << filteredData[input][time][IMAG][channel] << ')' << std::endl;
}


static void testFIR_Filter()
{
#if 0
  SmartPtr<InputDataType, SmartPtrFree<InputDataType> > inputData;

  if ((posix_memalign((void **) &inputData, 64, sizeof(InputDataType))) != 0)
    throw std::bad_alloc();
#endif

  setInputTestPattern(inputData[0]);
  setFilterWeightsTestPattern(filterWeights);
#pragma omp target update to(filterWeights)

  copyInputData(0);
  FIR_filter(0, 0);

#pragma omp target update from(filteredData)
  checkFIR_FilterTestPattern(filteredData);
}
#endif


////// FFT

DFTI_DESCRIPTOR_HANDLE handle;


static void fftInit()
{
  MKL_LONG error;

  error = DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 1, NR_CHANNELS);
    
  if (error != DFTI_NO_ERROR) {
    cerr << "DftiCreateDescriptor failed" << std::endl;
    exit(1);
  };

  error = DftiSetValue(handle, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
  
  if (error != DFTI_NO_ERROR) {
    cerr << "DftiSetValue failed" << std::endl;
    exit(1);
  };

  error = DftiSetValue(handle, DFTI_NUMBER_OF_TRANSFORMS, NR_SAMPLES_PER_MINOR_LOOP);

  if (error != DFTI_NO_ERROR) {
    cerr << "DftiSetValue failed" << std::endl;
    exit(1);
  }

  error = DftiSetValue(handle, DFTI_INPUT_DISTANCE, COMPLEX * NR_CHANNELS);

  if (error != DFTI_NO_ERROR) {
    cerr << "DftiSetValue failed" << std::endl;
    exit(1);
  }

  error = DftiSetValue(handle, DFTI_OUTPUT_DISTANCE, COMPLEX * NR_CHANNELS);

  if (error != DFTI_NO_ERROR) {
    cerr << "DftiSetValue failed" << std::endl;
    exit(1);
  }

  error = DftiCommitDescriptor(handle);
  
  if (error != DFTI_NO_ERROR) {
    cerr << "DftiCommitDescriptor failed" << std::endl;
    exit(1);
  }
}


static void fftDestroy()
{
  MKL_LONG error;

  error = DftiFreeDescriptor(&handle);
  
  if (error != DFTI_NO_ERROR) {
    cerr << "DftiFreeDescriptor failed" << std::endl;
    exit(1);
  }
}


#if !defined USE_FUSED_FILTER
static void FFT(FilteredDataType filteredData, unsigned)
{
#pragma omp parallel
  {
#pragma omp for collapse(2) schedule(dynamic)
    for (int input = 0; input < NR_INPUTS; input ++)
//      for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
//    DftiComputeForward(handle, filteredData[input][time][REAL], filteredData[input][time][IMAG]);
      for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time += NR_SAMPLES_PER_MINOR_LOOP)
	DftiComputeForward(handle, filteredData[input][time][REAL], filteredData[input][time][IMAG]);
  }
}
#endif


////// transpose

#if defined CORRECTNESS_TEST
static void transpose(
  CorrectedDataType correctedData,
  const FilteredDataType filteredData,
#if defined BANDPASS_CORRECTION
  const BandPassCorrectionWeights bandPassCorrectionWeights,
#endif
  unsigned
)
{
#pragma omp parallel
  {
#if defined __AVX512F__ || defined __MIC__
#if 0
    int	  stride  = NR_SAMPLES_PER_CHANNEL * NR_CHANNELS;
    __m512i indices = _mm512_set_epi32(15 * stride, 14 * stride, 13 * stride, 12 * stride, 11 * stride, 10 * stride, 9 * stride, 8 * stride, 7 * stride, 6 * stride, 5 * stride, 4 * stride, 3 * stride, 2 * stride, 1 * stride, 0 * stride);

#pragma omp for collapse(2) schedule(dynamic)
    for (int inputMajor = 0; inputMajor < NR_INPUTS / VECTOR_SIZE; inputMajor ++) {
      for (int channel = 0; channel < NR_CHANNELS; channel ++) {
	for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	    //_mm_prefetch((const char *) &filteredData[inputMajor * VECTOR_SIZE + channelMinor][time + 8][REAL][channelMajor], _MM_HINT_T1);
	    //_mm_prefetch((const char *) &filteredData[inputMajor * VECTOR_SIZE + channelMinor][time + 8][IMAG][channelMajor], _MM_HINT_T1);
	    //_mm_prefetch((const char *) &filteredData[inputMajor * VECTOR_SIZE + channelMinor][time + 1][REAL][channelMajor], _MM_HINT_T0);
	    //_mm_prefetch((const char *) &filteredData[inputMajor * VECTOR_SIZE + channelMinor][time + 1][IMAG][channelMajor], _MM_HINT_T0);

	    _mm512_prefetch_i32gather_ps(indices, &filteredData[inputMajor * VECTOR_SIZE][time + 8][REAL][channel], sizeof(float), _MM_HINT_T1);
	    _mm512_prefetch_i32gather_ps(indices, &filteredData[inputMajor * VECTOR_SIZE][time + 8][IMAG][channel], sizeof(float), _MM_HINT_T1);
	    _mm512_prefetch_i32gather_ps(indices, &filteredData[inputMajor * VECTOR_SIZE][time + 1][REAL][channel], sizeof(float), _MM_HINT_NTA);
	    _mm512_prefetch_i32gather_ps(indices, &filteredData[inputMajor * VECTOR_SIZE][time + 1][IMAG][channel], sizeof(float), _MM_HINT_NTA);

	    __m512 v_real = _mm512_i32gather_ps(indices, &filteredData[inputMajor * VECTOR_SIZE][time][REAL][channel], sizeof(float));
	    __m512 v_imag = _mm512_i32gather_ps(indices, &filteredData[inputMajor * VECTOR_SIZE][time][IMAG][channel], sizeof(float));

	    _mm512_storenrngo_ps(&correctedData[channel][inputMajor][time][REAL][0], v_real);
	    _mm512_storenrngo_ps(&correctedData[channel][inputMajor][time][IMAG][0], v_imag);
	  }
      }
    }
#else
    const uint64_t inputStride  = COMPLEX * NR_SAMPLES_PER_CHANNEL * NR_CHANNELS;
    const uint64_t outputStride = NR_SAMPLES_PER_CHANNEL * COMPLEX * ALIGN(NR_INPUTS, VECTOR_SIZE);

#pragma noprefetch
#pragma omp for collapse(2) schedule(dynamic)
    for (int inputMajor = 0; inputMajor < ALIGN(NR_INPUTS, VECTOR_SIZE) / VECTOR_SIZE; inputMajor ++) {
      for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	for (int channelMajor = 0; channelMajor < NR_CHANNELS; channelMajor += 16) {
	  _mm_prefetch((const char *) &bandPassCorrectionWeights[channelMajor], _MM_HINT_T1);
	}

	for (int real_imag = 0; real_imag < COMPLEX; real_imag ++) {
	  for (int channelMajor = 0; channelMajor < NR_CHANNELS; channelMajor += 16) {
#pragma unroll(16)
	    for (int i = 0; i < 16; i ++)
	      _mm_prefetch((const char *) (&filteredData[inputMajor * VECTOR_SIZE + i][time][real_imag][channelMajor]), _MM_HINT_T1);

	  }
	}

	for (int channelMajor = 0; channelMajor < NR_CHANNELS; channelMajor += 16) {

#if defined BANDPASS_CORRECTION
	  _mm_prefetch((const char *) &bandPassCorrectionWeights[channelMajor], _MM_HINT_NTA);
	  __m512 weights = _mm512_load_ps(&bandPassCorrectionWeights[channelMajor]);
#endif

	  for (int real_imag = 0; real_imag < COMPLEX; real_imag ++) {
	    const float *__restrict__ src = &filteredData[inputMajor * VECTOR_SIZE][time][real_imag][channelMajor];

#pragma unroll(16)
	    for (int i = 0; i < 16; i ++)
	      _mm_prefetch((const char *) (src + i * inputStride), _MM_HINT_NTA);

	    __m512 a0a1a2a3a4a5a6a7a8a9aAaBaCaDaEaF = _mm512_load_ps(src +  0 * inputStride);
	    __m512 b0b1b2b3b4b5b6b7b8b9bAbBbCbDbEbF = _mm512_load_ps(src +  1 * inputStride);

#if defined BANDPASS_CORRECTION
	    a0a1a2a3a4a5a6a7a8a9aAaBaCaDaEaF = _mm512_mul_ps(a0a1a2a3a4a5a6a7a8a9aAaBaCaDaEaF, weights);
	    b0b1b2b3b4b5b6b7b8b9bAbBbCbDbEbF = _mm512_mul_ps(b0b1b2b3b4b5b6b7b8b9bAbBbCbDbEbF, weights);
#endif

	    __m512 a0b0a2b2a4b4a6b6a8b8aAbAaCbCaEbE = _mm512_mask_swizzle_ps(a0a1a2a3a4a5a6a7a8a9aAaBaCaDaEaF, 0xAAAA, b0b1b2b3b4b5b6b7b8b9bAbBbCbDbEbF, _MM_SWIZ_REG_CDAB);
	    __m512 a1b1a3b3a5b5a7b7a9b9aBbBaDbDaFbF = _mm512_mask_swizzle_ps(b0b1b2b3b4b5b6b7b8b9bAbBbCbDbEbF, 0x5555, a0a1a2a3a4a5a6a7a8a9aAaBaCaDaEaF, _MM_SWIZ_REG_CDAB);

	    __m512 c0c1c2c3c4c5c6c7c8c9cAcBcCcDcEcF = _mm512_load_ps(src +  2 * inputStride);
	    __m512 d0d1d2d3d4d5d6d7d8d9dAdBdCdDdEdF = _mm512_load_ps(src +  3 * inputStride);

#if defined BANDPASS_CORRECTION
	    c0c1c2c3c4c5c6c7c8c9cAcBcCcDcEcF = _mm512_mul_ps(c0c1c2c3c4c5c6c7c8c9cAcBcCcDcEcF, weights);
	    d0d1d2d3d4d5d6d7d8d9dAdBdCdDdEdF = _mm512_mul_ps(d0d1d2d3d4d5d6d7d8d9dAdBdCdDdEdF, weights);
#endif

	    __m512 c0d0c2d2c4d4c6d6c8d8cAdAcCdCcEdE = _mm512_mask_swizzle_ps(c0c1c2c3c4c5c6c7c8c9cAcBcCcDcEcF, 0xAAAA, d0d1d2d3d4d5d6d7d8d9dAdBdCdDdEdF, _MM_SWIZ_REG_CDAB);
	    __m512 c1d1c3d3c5d5c7d7c9d9cBdBcDdDcFdF = _mm512_mask_swizzle_ps(d0d1d2d3d4d5d6d7d8d9dAdBdCdDdEdF, 0x5555, c0c1c2c3c4c5c6c7c8c9cAcBcCcDcEcF, _MM_SWIZ_REG_CDAB);

	    __m512 a0b0c0d0a4b4c4d4a8b8c8d8aCbCcCdC = _mm512_mask_swizzle_ps(a0b0a2b2a4b4a6b6a8b8aAbAaCbCaEbE, 0xCCCC, c0d0c2d2c4d4c6d6c8d8cAdAcCdCcEdE, _MM_SWIZ_REG_BADC);
	    __m512 a1b1c1d1a5b5c5d5a9b9c9d9aDbDcDdD = _mm512_mask_swizzle_ps(a1b1a3b3a5b5a7b7a9b9aBbBaDbDaFbF, 0xCCCC, c1d1c3d3c5d5c7d7c9d9cBdBcDdDcFdF, _MM_SWIZ_REG_BADC);
	    __m512 a2b2c2d2a6b6c6d6aAbAcAdAaEbEcEdE = _mm512_mask_swizzle_ps(c0d0c2d2c4d4c6d6c8d8cAdAcCdCcEdE, 0x3333, a0b0a2b2a4b4a6b6a8b8aAbAaCbCaEbE, _MM_SWIZ_REG_BADC);
	    __m512 a3b3c3d3a7b7c7d7aBbBcBdBaFbFcFdF = _mm512_mask_swizzle_ps(c1d1c3d3c5d5c7d7c9d9cBdBcDdDcFdF, 0x3333, a1b1a3b3a5b5a7b7a9b9aBbBaDbDaFbF, _MM_SWIZ_REG_BADC);

	    __m512 e0e1e2e3e4e5e6e7e8e9eAeBeCeDeEeF = _mm512_load_ps(src +  4 * inputStride);
	    __m512 f0f1f2f3f4f5f6f7f8f9fAfBfCfDfEfF = _mm512_load_ps(src +  5 * inputStride);

#if defined BANDPASS_CORRECTION
	    e0e1e2e3e4e5e6e7e8e9eAeBeCeDeEeF = _mm512_mul_ps(e0e1e2e3e4e5e6e7e8e9eAeBeCeDeEeF, weights);
	    f0f1f2f3f4f5f6f7f8f9fAfBfCfDfEfF = _mm512_mul_ps(f0f1f2f3f4f5f6f7f8f9fAfBfCfDfEfF, weights);
#endif

	    __m512 e0f0e2f2e4f4e6f6e8f8eAfAeCfCeEfE = _mm512_mask_swizzle_ps(e0e1e2e3e4e5e6e7e8e9eAeBeCeDeEeF, 0xAAAA, f0f1f2f3f4f5f6f7f8f9fAfBfCfDfEfF, _MM_SWIZ_REG_CDAB);
	    __m512 e1f1e3f3e5f5e7f7e9f9eBfBeDfDeFfF = _mm512_mask_swizzle_ps(f0f1f2f3f4f5f6f7f8f9fAfBfCfDfEfF, 0x5555, e0e1e2e3e4e5e6e7e8e9eAeBeCeDeEeF, _MM_SWIZ_REG_CDAB);

	    __m512 g0g1g2g3g4g5g6g7g8g9gAgBgCgDgEgF = _mm512_load_ps(src +  6 * inputStride);
	    __m512 h0h1h2h3h4h5h6h7h8h9hAhBhChDhEhF = _mm512_load_ps(src +  7 * inputStride);

#if defined BANDPASS_CORRECTION
	    g0g1g2g3g4g5g6g7g8g9gAgBgCgDgEgF = _mm512_mul_ps(g0g1g2g3g4g5g6g7g8g9gAgBgCgDgEgF, weights);
	    h0h1h2h3h4h5h6h7h8h9hAhBhChDhEhF = _mm512_mul_ps(h0h1h2h3h4h5h6h7h8h9hAhBhChDhEhF, weights);
#endif

	    __m512 g0h0g2h2g4h4g6h6g8h8gAhAgChCgEhE = _mm512_mask_swizzle_ps(g0g1g2g3g4g5g6g7g8g9gAgBgCgDgEgF, 0xAAAA, h0h1h2h3h4h5h6h7h8h9hAhBhChDhEhF, _MM_SWIZ_REG_CDAB);
	    __m512 g1h1g3h3g5h5g7h7g9h9gBhBgDhDgFhF = _mm512_mask_swizzle_ps(h0h1h2h3h4h5h6h7h8h9hAhBhChDhEhF, 0x5555, g0g1g2g3g4g5g6g7g8g9gAgBgCgDgEgF, _MM_SWIZ_REG_CDAB);

	    __m512 e0f0g0h0e4f4g4h4e8f8g8h8eCfCgChC = _mm512_mask_swizzle_ps(e0f0e2f2e4f4e6f6e8f8eAfAeCfCeEfE, 0xCCCC, g0h0g2h2g4h4g6h6g8h8gAhAgChCgEhE, _MM_SWIZ_REG_BADC);
	    __m512 e1f1g1h1e5f5g5h5e9f9g9h9eDfDgDhD = _mm512_mask_swizzle_ps(e1f1e3f3e5f5e7f7e9f9eBfBeDfDeFfF, 0xCCCC, g1h1g3h3g5h5g7h7g9h9gBhBgDhDgFhF, _MM_SWIZ_REG_BADC);
	    __m512 e2f2g2h2e6f6g6h6eAfAgAhAeEfEgEhE = _mm512_mask_swizzle_ps(g0h0g2h2g4h4g6h6g8h8gAhAgChCgEhE, 0x3333, e0f0e2f2e4f4e6f6e8f8eAfAeCfCeEfE, _MM_SWIZ_REG_BADC);
	    __m512 e3f3g3h3e7f7g7h7eBfBgBhBeFfFgFhF = _mm512_mask_swizzle_ps(g1h1g3h3g5h5g7h7g9h9gBhBgDhDgFhF, 0x3333, e1f1e3f3e5f5e7f7e9f9eBfBeDfDeFfF, _MM_SWIZ_REG_BADC);

	    __m512 a0b0c0d0e0f0g0h0a8b8c8d8e8f8g8h8 = _mm512_mask_permute4f128_ps(a0b0c0d0a4b4c4d4a8b8c8d8aCbCcCdC, 0xF0F0, e0f0g0h0e4f4g4h4e8f8g8h8eCfCgChC, _MM_PERM_CDAB);
	    __m512 a1b1c1d1e1f1g1h1a9b9c9d9e9f9g9h9 = _mm512_mask_permute4f128_ps(a1b1c1d1a5b5c5d5a9b9c9d9aDbDcDdD, 0xF0F0, e1f1g1h1e5f5g5h5e9f9g9h9eDfDgDhD, _MM_PERM_CDAB);
	    __m512 a2b2c2d2e2f2g2h2aAbAcAdAeAfAgAhA = _mm512_mask_permute4f128_ps(a2b2c2d2a6b6c6d6aAbAcAdAaEbEcEdE, 0xF0F0, e2f2g2h2e6f6g6h6eAfAgAhAeEfEgEhE, _MM_PERM_CDAB);
	    __m512 a3b3c3d3e3f3g3h3aBbBcBdBeBfBgBhB = _mm512_mask_permute4f128_ps(a3b3c3d3a7b7c7d7aBbBcBdBaFbFcFdF, 0xF0F0, e3f3g3h3e7f7g7h7eBfBgBhBeFfFgFhF, _MM_PERM_CDAB);
	    __m512 a4b4c4d4e4f4g4h4aCbCcCdCeCfCgChC = _mm512_mask_permute4f128_ps(e0f0g0h0e4f4g4h4e8f8g8h8eCfCgChC, 0x0F0F, a0b0c0d0a4b4c4d4a8b8c8d8aCbCcCdC, _MM_PERM_CDAB);
	    __m512 a5b5c5d5e5f5g5h5aDbDcDdDeDfDgDhD = _mm512_mask_permute4f128_ps(e1f1g1h1e5f5g5h5e9f9g9h9eDfDgDhD, 0x0F0F, a1b1c1d1a5b5c5d5a9b9c9d9aDbDcDdD, _MM_PERM_CDAB);
	    __m512 a6b6c6d6e6f6g6h6aEbEcEdEeEfEgEhE = _mm512_mask_permute4f128_ps(e2f2g2h2e6f6g6h6eAfAgAhAeEfEgEhE, 0x0F0F, a2b2c2d2a6b6c6d6aAbAcAdAaEbEcEdE, _MM_PERM_CDAB);
	    __m512 a7b7c7d7e7f7g7h7aFbFcFdFeFfFgFhF = _mm512_mask_permute4f128_ps(e3f3g3h3e7f7g7h7eBfBgBhBeFfFgFhF, 0x0F0F, a3b3c3d3a7b7c7d7aBbBcBdBaFbFcFdF, _MM_PERM_CDAB);

	    __m512 i0i1i2i3i4i5i6i7i8i9iAiBiCiDiEiF = _mm512_load_ps(src +  8 * inputStride);
	    __m512 j0j1j2j3j4j5j6j7j8j9jAjBjCjDjEjF = _mm512_load_ps(src +  9 * inputStride);

#if defined BANDPASS_CORRECTION
	    i0i1i2i3i4i5i6i7i8i9iAiBiCiDiEiF = _mm512_mul_ps(i0i1i2i3i4i5i6i7i8i9iAiBiCiDiEiF, weights);
	    j0j1j2j3j4j5j6j7j8j9jAjBjCjDjEjF = _mm512_mul_ps(j0j1j2j3j4j5j6j7j8j9jAjBjCjDjEjF, weights);
#endif

	    __m512 i0j0i2j2i4j4i6j6i8j8iAjAiCjCiEjE = _mm512_mask_swizzle_ps(i0i1i2i3i4i5i6i7i8i9iAiBiCiDiEiF, 0xAAAA, j0j1j2j3j4j5j6j7j8j9jAjBjCjDjEjF, _MM_SWIZ_REG_CDAB);
	    __m512 i1j1i3j3i5j5i7j7i9j9iBjBiDjDiFjF = _mm512_mask_swizzle_ps(j0j1j2j3j4j5j6j7j8j9jAjBjCjDjEjF, 0x5555, i0i1i2i3i4i5i6i7i8i9iAiBiCiDiEiF, _MM_SWIZ_REG_CDAB);

	    __m512 k0k1k2k3k4k5k6k7k8k9kAkBkCkDkEkF = _mm512_load_ps(src + 10 * inputStride);
	    __m512 l0l1l2l3l4l5l6l7l8l9lAlBlClDlElF = _mm512_load_ps(src + 11 * inputStride);

#if defined BANDPASS_CORRECTION
	    k0k1k2k3k4k5k6k7k8k9kAkBkCkDkEkF = _mm512_mul_ps(k0k1k2k3k4k5k6k7k8k9kAkBkCkDkEkF, weights);
	    l0l1l2l3l4l5l6l7l8l9lAlBlClDlElF = _mm512_mul_ps(l0l1l2l3l4l5l6l7l8l9lAlBlClDlElF, weights);
#endif

	    __m512 k0l0k2l2k4l4k6l6k8l8kAlAkClCkElE = _mm512_mask_swizzle_ps(k0k1k2k3k4k5k6k7k8k9kAkBkCkDkEkF, 0xAAAA, l0l1l2l3l4l5l6l7l8l9lAlBlClDlElF, _MM_SWIZ_REG_CDAB);
	    __m512 k1l1k3l3k5l5k7l7k9l9kBlBkDlDkFlF = _mm512_mask_swizzle_ps(l0l1l2l3l4l5l6l7l8l9lAlBlClDlElF, 0x5555, k0k1k2k3k4k5k6k7k8k9kAkBkCkDkEkF, _MM_SWIZ_REG_CDAB);

	    __m512 i0j0k0l0i4j4k4l4i8j8k8l8iCjCkClC = _mm512_mask_swizzle_ps(i0j0i2j2i4j4i6j6i8j8iAjAiCjCiEjE, 0xCCCC, k0l0k2l2k4l4k6l6k8l8kAlAkClCkElE, _MM_SWIZ_REG_BADC);
	    __m512 i1j1k1l1i5j5k5l5i9j9k9l9iDjDkDlD = _mm512_mask_swizzle_ps(i1j1i3j3i5j5i7j7i9j9iBjBiDjDiFjF, 0xCCCC, k1l1k3l3k5l5k7l7k9l9kBlBkDlDkFlF, _MM_SWIZ_REG_BADC);
	    __m512 i2j2k2l2i6j6k6l6iAjAkAlAiEjEkElE = _mm512_mask_swizzle_ps(k0l0k2l2k4l4k6l6k8l8kAlAkClCkElE, 0x3333, i0j0i2j2i4j4i6j6i8j8iAjAiCjCiEjE, _MM_SWIZ_REG_BADC);
	    __m512 i3j3k3l3i7j7k7l7iBjBkBlBiFjFkFlF = _mm512_mask_swizzle_ps(k1l1k3l3k5l5k7l7k9l9kBlBkDlDkFlF, 0x3333, i1j1i3j3i5j5i7j7i9j9iBjBiDjDiFjF, _MM_SWIZ_REG_BADC);

	    __m512 m0m1m2m3m4m5m6m7m8m9mAmBmCmDmEmF = _mm512_load_ps(src + 12 * inputStride);
	    __m512 n0n1n2n3n4n5n6n7n8n9nAnBnCnDnEnF = _mm512_load_ps(src + 13 * inputStride);

#if defined BANDPASS_CORRECTION
	    m0m1m2m3m4m5m6m7m8m9mAmBmCmDmEmF = _mm512_mul_ps(m0m1m2m3m4m5m6m7m8m9mAmBmCmDmEmF, weights);
	    n0n1n2n3n4n5n6n7n8n9nAnBnCnDnEnF = _mm512_mul_ps(n0n1n2n3n4n5n6n7n8n9nAnBnCnDnEnF, weights);
#endif

	    __m512 m0n0m2n2m4n4m6n6m8n8mAnAmCnCmEnE = _mm512_mask_swizzle_ps(m0m1m2m3m4m5m6m7m8m9mAmBmCmDmEmF, 0xAAAA, n0n1n2n3n4n5n6n7n8n9nAnBnCnDnEnF, _MM_SWIZ_REG_CDAB);
	    __m512 m1n1m3n3m5n5m7n7m9n9mBnBmDnDmFnF = _mm512_mask_swizzle_ps(n0n1n2n3n4n5n6n7n8n9nAnBnCnDnEnF, 0x5555, m0m1m2m3m4m5m6m7m8m9mAmBmCmDmEmF, _MM_SWIZ_REG_CDAB);

	    __m512 o0o1o2o3o4o5o6o7o8o9oAoBoCoDoEoF = _mm512_load_ps(src + 14 * inputStride);
	    __m512 p0p1p2p3p4p5p6p7p8p9pApBpCpDpEpF = _mm512_load_ps(src + 15 * inputStride);

#if defined BANDPASS_CORRECTION
	    o0o1o2o3o4o5o6o7o8o9oAoBoCoDoEoF = _mm512_mul_ps(o0o1o2o3o4o5o6o7o8o9oAoBoCoDoEoF, weights);
	    p0p1p2p3p4p5p6p7p8p9pApBpCpDpEpF = _mm512_mul_ps(p0p1p2p3p4p5p6p7p8p9pApBpCpDpEpF, weights);
#endif

	    __m512 o0p0o2p2o4p4o6p6o8p8oApAoCpCoEpE = _mm512_mask_swizzle_ps(o0o1o2o3o4o5o6o7o8o9oAoBoCoDoEoF, 0xAAAA, p0p1p2p3p4p5p6p7p8p9pApBpCpDpEpF, _MM_SWIZ_REG_CDAB);
	    __m512 o1p1o3p3o5p5o7p7o9p9oBpBoDpDoFpF = _mm512_mask_swizzle_ps(p0p1p2p3p4p5p6p7p8p9pApBpCpDpEpF, 0x5555, o0o1o2o3o4o5o6o7o8o9oAoBoCoDoEoF, _MM_SWIZ_REG_CDAB);

	    __m512 m0n0o0p0m4n4o4p4m8n8o8p8mCnCoCpC = _mm512_mask_swizzle_ps(m0n0m2n2m4n4m6n6m8n8mAnAmCnCmEnE, 0xCCCC, o0p0o2p2o4p4o6p6o8p8oApAoCpCoEpE, _MM_SWIZ_REG_BADC);
	    __m512 m1n1o1p1m5n5o5p5m9n9o9p9mDnDoDpD = _mm512_mask_swizzle_ps(m1n1m3n3m5n5m7n7m9n9mBnBmDnDmFnF, 0xCCCC, o1p1o3p3o5p5o7p7o9p9oBpBoDpDoFpF, _MM_SWIZ_REG_BADC);
	    __m512 m2n2o2p2m6n6o6p6mAnAoApAmEnEoEpE = _mm512_mask_swizzle_ps(o0p0o2p2o4p4o6p6o8p8oApAoCpCoEpE, 0x3333, m0n0m2n2m4n4m6n6m8n8mAnAmCnCmEnE, _MM_SWIZ_REG_BADC);
	    __m512 m3n3o3p3m7n7o7p7mBnBoBpBmFnFoFpF = _mm512_mask_swizzle_ps(o1p1o3p3o5p5o7p7o9p9oBpBoDpDoFpF, 0x3333, m1n1m3n3m5n5m7n7m9n9mBnBmDnDmFnF, _MM_SWIZ_REG_BADC);

	    __m512 i0j0k0l0m0n0o0p0i8j8k8l8m8n8o8p8 = _mm512_mask_permute4f128_ps(i0j0k0l0i4j4k4l4i8j8k8l8iCjCkClC, 0xF0F0, m0n0o0p0m4n4o4p4m8n8o8p8mCnCoCpC, _MM_PERM_CDAB);
	    __m512 i1j1k1l1m1n1o1p1i9j9k9l9m9n9o9p9 = _mm512_mask_permute4f128_ps(i1j1k1l1i5j5k5l5i9j9k9l9iDjDkDlD, 0xF0F0, m1n1o1p1m5n5o5p5m9n9o9p9mDnDoDpD, _MM_PERM_CDAB);
	    __m512 i2j2k2l2m2n2o2p2iAjAkAlAmAnAoApA = _mm512_mask_permute4f128_ps(i2j2k2l2i6j6k6l6iAjAkAlAiEjEkElE, 0xF0F0, m2n2o2p2m6n6o6p6mAnAoApAmEnEoEpE, _MM_PERM_CDAB);
	    __m512 i3j3k3l3m3n3o3p3iBjBkBlBmBnBoBpB = _mm512_mask_permute4f128_ps(i3j3k3l3i7j7k7l7iBjBkBlBiFjFkFlF, 0xF0F0, m3n3o3p3m7n7o7p7mBnBoBpBmFnFoFpF, _MM_PERM_CDAB);
	    __m512 i4j4k4l4m4n4o4p4iCjCkClCmCnCoCpC = _mm512_mask_permute4f128_ps(m0n0o0p0m4n4o4p4m8n8o8p8mCnCoCpC, 0x0F0F, i0j0k0l0i4j4k4l4i8j8k8l8iCjCkClC, _MM_PERM_CDAB);
	    __m512 i5j5k5l5m5n5o5p5iDjDkDlDmDnDoDpD = _mm512_mask_permute4f128_ps(m1n1o1p1m5n5o5p5m9n9o9p9mDnDoDpD, 0x0F0F, i1j1k1l1i5j5k5l5i9j9k9l9iDjDkDlD, _MM_PERM_CDAB);
	    __m512 i6j6k6l6m6n6o6p6iEjEkElEmEnEoEpE = _mm512_mask_permute4f128_ps(m2n2o2p2m6n6o6p6mAnAoApAmEnEoEpE, 0x0F0F, i2j2k2l2i6j6k6l6iAjAkAlAiEjEkElE, _MM_PERM_CDAB);
	    __m512 i7j7k7l7m7n7o7p7iFjFkFlFmFnFoFpF = _mm512_mask_permute4f128_ps(m3n3o3p3m7n7o7p7mBnBoBpBmFnFoFpF, 0x0F0F, i3j3k3l3i7j7k7l7iBjBkBlBiFjFkFlF, _MM_PERM_CDAB);

	    __m512 a0b0c0d0e0f0g0h0i0j0k0l0m0n0o0p0 = _mm512_mask_permute4f128_ps(a0b0c0d0e0f0g0h0a8b8c8d8e8f8g8h8, 0xFF00, i0j0k0l0m0n0o0p0i8j8k8l8m8n8o8p8, _MM_PERM_BADC);
	    __m512 a1b1c1d1e1f1g1h1i1j1k1l1m1n1o1p1 = _mm512_mask_permute4f128_ps(a1b1c1d1e1f1g1h1a9b9c9d9e9f9g9h9, 0xFF00, i1j1k1l1m1n1o1p1i9j9k9l9m9n9o9p9, _MM_PERM_BADC);
	    __m512 a2b2c2d2e2f2g2h2i2j2k2l2m2n2o2p2 = _mm512_mask_permute4f128_ps(a2b2c2d2e2f2g2h2aAbAcAdAeAfAgAhA, 0xFF00, i2j2k2l2m2n2o2p2iAjAkAlAmAnAoApA, _MM_PERM_BADC);
	    __m512 a3b3c3d3e3f3g3h3i3j3k3l3m3n3o3p3 = _mm512_mask_permute4f128_ps(a3b3c3d3e3f3g3h3aBbBcBdBeBfBgBhB, 0xFF00, i3j3k3l3m3n3o3p3iBjBkBlBmBnBoBpB, _MM_PERM_BADC);
	    __m512 a4b4c4d4e4f4g4h4i4j4k4l4m4n4o4p4 = _mm512_mask_permute4f128_ps(a4b4c4d4e4f4g4h4aCbCcCdCeCfCgChC, 0xFF00, i4j4k4l4m4n4o4p4iCjCkClCmCnCoCpC, _MM_PERM_BADC);
	    __m512 a5b5c5d5e5f5g5h5i5j5k5l5m5n5o5p5 = _mm512_mask_permute4f128_ps(a5b5c5d5e5f5g5h5aDbDcDdDeDfDgDhD, 0xFF00, i5j5k5l5m5n5o5p5iDjDkDlDmDnDoDpD, _MM_PERM_BADC);
	    __m512 a6b6c6d6e6f6g6h6i6j6k6l6m6n6o6p6 = _mm512_mask_permute4f128_ps(a6b6c6d6e6f6g6h6aEbEcEdEeEfEgEhE, 0xFF00, i6j6k6l6m6n6o6p6iEjEkElEmEnEoEpE, _MM_PERM_BADC);
	    __m512 a7b7c7d7e7f7g7h7i7j7k7l7m7n7o7p7 = _mm512_mask_permute4f128_ps(a7b7c7d7e7f7g7h7aFbFcFdFeFfFgFhF, 0xFF00, i7j7k7l7m7n7o7p7iFjFkFlFmFnFoFpF, _MM_PERM_BADC);
	    __m512 a8b8c8d8e8f8g8h8i8j8k8l8m8n8o8p8 = _mm512_mask_permute4f128_ps(i0j0k0l0m0n0o0p0i8j8k8l8m8n8o8p8, 0x00FF, a0b0c0d0e0f0g0h0a8b8c8d8e8f8g8h8, _MM_PERM_BADC);
	    __m512 a9b9c9d9e9f9g9h9i9j9k9l9m9n9o9p9 = _mm512_mask_permute4f128_ps(i1j1k1l1m1n1o1p1i9j9k9l9m9n9o9p9, 0x00FF, a1b1c1d1e1f1g1h1a9b9c9d9e9f9g9h9, _MM_PERM_BADC);
	    __m512 aAbAcAdAeAfAgAhAiAjAkAlAmAnAoApA = _mm512_mask_permute4f128_ps(i2j2k2l2m2n2o2p2iAjAkAlAmAnAoApA, 0x00FF, a2b2c2d2e2f2g2h2aAbAcAdAeAfAgAhA, _MM_PERM_BADC);
	    __m512 aBbBcBdBeBfBgBhBiBjBkBlBmBnBoBpB = _mm512_mask_permute4f128_ps(i3j3k3l3m3n3o3p3iBjBkBlBmBnBoBpB, 0x00FF, a3b3c3d3e3f3g3h3aBbBcBdBeBfBgBhB, _MM_PERM_BADC);
	    __m512 aCbCcCdCeCfCgChCiCjCkClCmCnCoCpC = _mm512_mask_permute4f128_ps(i4j4k4l4m4n4o4p4iCjCkClCmCnCoCpC, 0x00FF, a4b4c4d4e4f4g4h4aCbCcCdCeCfCgChC, _MM_PERM_BADC);
	    __m512 aDbDcDdDeDfDgDhDiDjDkDlDmDnDoDpD = _mm512_mask_permute4f128_ps(i5j5k5l5m5n5o5p5iDjDkDlDmDnDoDpD, 0x00FF, a5b5c5d5e5f5g5h5aDbDcDdDeDfDgDhD, _MM_PERM_BADC);
	    __m512 aEbEcEdEeEfEgEhEiEjEkElEmEnEoEpE = _mm512_mask_permute4f128_ps(i6j6k6l6m6n6o6p6iEjEkElEmEnEoEpE, 0x00FF, a6b6c6d6e6f6g6h6aEbEcEdEeEfEgEhE, _MM_PERM_BADC);
	    __m512 aFbFcFdFeFfFgFhFiFjFkFlFmFnFoFpF = _mm512_mask_permute4f128_ps(i7j7k7l7m7n7o7p7iFjFkFlFmFnFoFpF, 0x00FF, a7b7c7d7e7f7g7h7aFbFcFdFeFfFgFhF, _MM_PERM_BADC);

	    float *__restrict__ dst = &correctedData[channelMajor][inputMajor][time][real_imag][0];
	    _mm512_storenrngo_ps(dst +  0 * outputStride, a0b0c0d0e0f0g0h0i0j0k0l0m0n0o0p0);
	    _mm512_storenrngo_ps(dst +  1 * outputStride, a1b1c1d1e1f1g1h1i1j1k1l1m1n1o1p1);
	    _mm512_storenrngo_ps(dst +  2 * outputStride, a2b2c2d2e2f2g2h2i2j2k2l2m2n2o2p2);
	    _mm512_storenrngo_ps(dst +  3 * outputStride, a3b3c3d3e3f3g3h3i3j3k3l3m3n3o3p3);
	    _mm512_storenrngo_ps(dst +  4 * outputStride, a4b4c4d4e4f4g4h4i4j4k4l4m4n4o4p4);
	    _mm512_storenrngo_ps(dst +  5 * outputStride, a5b5c5d5e5f5g5h5i5j5k5l5m5n5o5p5);
	    _mm512_storenrngo_ps(dst +  6 * outputStride, a6b6c6d6e6f6g6h6i6j6k6l6m6n6o6p6);
	    _mm512_storenrngo_ps(dst +  7 * outputStride, a7b7c7d7e7f7g7h7i7j7k7l7m7n7o7p7);
	    _mm512_storenrngo_ps(dst +  8 * outputStride, a8b8c8d8e8f8g8h8i8j8k8l8m8n8o8p8);
	    _mm512_storenrngo_ps(dst +  9 * outputStride, a9b9c9d9e9f9g9h9i9j9k9l9m9n9o9p9);
	    _mm512_storenrngo_ps(dst + 10 * outputStride, aAbAcAdAeAfAgAhAiAjAkAlAmAnAoApA);
	    _mm512_storenrngo_ps(dst + 11 * outputStride, aBbBcBdBeBfBgBhBiBjBkBlBmBnBoBpB);
	    _mm512_storenrngo_ps(dst + 12 * outputStride, aCbCcCdCeCfCgChCiCjCkClCmCnCoCpC);
	    _mm512_storenrngo_ps(dst + 13 * outputStride, aDbDcDdDeDfDgDhDiDjDkDlDmDnDoDpD);
	    _mm512_storenrngo_ps(dst + 14 * outputStride, aEbEcEdEeEfEgEhEiEjEkElEmEnEoEpE);
	    _mm512_storenrngo_ps(dst + 15 * outputStride, aFbFcFdFeFfFgFhFiFjFkFlFmFnFoFpF);
	  }
	}
      }
    }
#endif
#else
#pragma omp for schedule(dynamic)
    for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
      for (unsigned inputMajor = 0; inputMajor < ALIGN(NR_INPUTS, VECTOR_SIZE); inputMajor += VECTOR_SIZE) {
	for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
	  for (unsigned realImag = 0; realImag < COMPLEX; realImag ++) {
	    for (unsigned inputMinor = 0; inputMinor < VECTOR_SIZE; inputMinor ++) {
	      unsigned input = inputMajor + inputMinor;

	      if (NR_INPUTS % VECTOR_SIZE == 0 || input < NR_INPUTS) {
#if defined BANDPASS_CORRECTION
		correctedData[channel][inputMajor / VECTOR_SIZE][time][realImag][inputMinor] = bandPassCorrectionWeights[channel] * filteredData[input][time][realImag][channel];
#else
		correctedData[channel][inputMajor / VECTOR_SIZE][time][realImag][inputMinor] = filteredData[input][time][realImag][channel];
#endif
	      }
	    }
	  }
	}
      }
    }
#endif
  }
}
#endif


#if defined DELAY_COMPENSATION

#if defined CORRECTNESS_TEST
static void applyDelays(CorrectedDataType correctedData, const DelaysType delaysAtBegin, const DelaysType delaysAfterEnd, float subbandFrequency, unsigned)
{
#pragma omp parallel
  {
#pragma omp for collapse(2)
    for (unsigned inputMajor = 0; inputMajor < ALIGN(NR_INPUTS, VECTOR_SIZE) / VECTOR_SIZE; inputMajor ++) {
      for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
	float v_rf[VECTOR_SIZE] __attribute__((aligned(sizeof (float[VECTOR_SIZE]))));
	float v_if[VECTOR_SIZE] __attribute__((aligned(sizeof (float[VECTOR_SIZE]))));
	float dv_rf[VECTOR_SIZE] __attribute__((aligned(sizeof (float[VECTOR_SIZE]))));
	float dv_if[VECTOR_SIZE] __attribute__((aligned(sizeof (float[VECTOR_SIZE]))));

	for (unsigned inputMinor = 0; inputMinor < VECTOR_SIZE; inputMinor ++) {
	  unsigned input = inputMajor * VECTOR_SIZE + inputMinor;

	  if (NR_INPUTS % VECTOR_SIZE == 0 || input < NR_INPUTS) {
	    double phiBegin = -2.0 * 3.141592653589793 * delaysAtBegin[input];
	    double phiEnd   = -2.0 * 3.141592653589793 * delaysAfterEnd[input];
	    double deltaPhi = (phiEnd - phiBegin) / NR_SAMPLES_PER_CHANNEL;
	    double channelFrequency = subbandFrequency - .5 * SUBBAND_BANDWIDTH + channel * (SUBBAND_BANDWIDTH / NR_CHANNELS);
	    float myPhiBegin = static_cast<float>((phiBegin /* + startTime * deltaPhi */) * channelFrequency /* + phaseOffsets[stationPol + major] */);
	    float myPhiDelta	= static_cast<float>(deltaPhi * channelFrequency);
	    sincosf(myPhiBegin, &v_if[inputMinor], &v_rf[inputMinor]);
	    sincosf(myPhiDelta, &dv_if[inputMinor], &dv_rf[inputMinor]);
	  }
	}

#if defined __AVX512F__ || defined __MIC__
	__m512 v_r = _mm512_load_ps(v_rf);
	__m512 v_i = _mm512_load_ps(v_if);
	__m512 dv_r = _mm512_load_ps(dv_rf);
	__m512 dv_i = _mm512_load_ps(dv_if);

	for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	  __m512 sample_r = _mm512_load_ps(&correctedData[channel][inputMajor][time][REAL][0]);
	  __m512 sample_i = _mm512_load_ps(&correctedData[channel][inputMajor][time][IMAG][0]);

	  _mm_prefetch((const char *) (&correctedData[channel][inputMajor][time + 8][REAL][0]), _MM_HINT_T2);
	  _mm_prefetch((const char *) (&correctedData[channel][inputMajor][time + 8][IMAG][0]), _MM_HINT_T2);

	  _mm_prefetch((const char *) (&correctedData[channel][inputMajor][time + 1][REAL][0]), _MM_HINT_NTA);
	  _mm_prefetch((const char *) (&correctedData[channel][inputMajor][time + 1][IMAG][0]), _MM_HINT_NTA);

	  __m512 tmp = _mm512_mul_ps(sample_r, v_i);
	  sample_r = _mm512_fmsub_ps(sample_r, v_r, _mm512_mul_ps(sample_i, v_i));
	  sample_i = _mm512_fmadd_ps(sample_i, v_r, tmp);

	  _mm512_storenrngo_ps(&correctedData[channel][inputMajor][time][REAL][0], sample_r);
	  _mm512_storenrngo_ps(&correctedData[channel][inputMajor][time][IMAG][0], sample_i);

	  tmp = _mm512_mul_ps(v_r, dv_i);
	  v_r = _mm512_fmsub_ps(v_r, dv_r, _mm512_mul_ps(v_i, dv_i));
	  v_i = _mm512_fmadd_ps(v_i, dv_r, tmp);
	}
#else
	__m256 v_r = _mm256_load_ps(v_rf);
	__m256 v_i = _mm256_load_ps(v_if);
	__m256 dv_r = _mm256_load_ps(dv_rf);
	__m256 dv_i = _mm256_load_ps(dv_if);

	for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	  __m256 sample_r = _mm256_load_ps(&correctedData[channel][inputMajor][time][REAL][0]);
	  __m256 sample_i = _mm256_load_ps(&correctedData[channel][inputMajor][time][IMAG][0]);

#if 0
	  _mm_prefetch((const char *) (&correctedData[channel][inputMajor][time + 8][REAL][0]), _MM_HINT_T2);
	  _mm_prefetch((const char *) (&correctedData[channel][inputMajor][time + 8][IMAG][0]), _MM_HINT_T2);

	  _mm_prefetch((const char *) (&correctedData[channel][inputMajor][time + 1][REAL][0]), _MM_HINT_NTA);
	  _mm_prefetch((const char *) (&correctedData[channel][inputMajor][time + 1][IMAG][0]), _MM_HINT_NTA);
#endif

	  __m256 tmp = _mm256_mul_ps(sample_r, v_i);
	  sample_r = _mm256_sub_ps(_mm256_mul_ps(sample_r, v_r), _mm256_mul_ps(sample_i, v_i));
	  sample_i = _mm256_add_ps(_mm256_mul_ps(sample_i, v_r), tmp);

	  tmp = _mm256_mul_ps(v_r, dv_i);
	  v_r = _mm256_sub_ps(_mm256_mul_ps(v_r, dv_r), _mm256_mul_ps(v_i, dv_i));
	  v_i = _mm256_add_ps(_mm256_mul_ps(v_i, dv_r), tmp);

	  _mm256_stream_ps(&correctedData[channel][inputMajor][time][REAL][0], sample_r);
	  _mm256_stream_ps(&correctedData[channel][inputMajor][time][IMAG][0], sample_i);
	}
#endif
      }
    }
  }
}
#endif
#endif


#if defined BANDPASS_CORRECTION
static void setBandPassTestPattern(BandPassCorrectionWeights bandPassCorrectionWeights)
{
  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
    bandPassCorrectionWeights[channel] = 1;

  if (NR_CHANNELS > 5)
    bandPassCorrectionWeights[5] = 2;
}
#endif


#if defined DELAY_COMPENSATION
static void setDelaysTestPattern(DelaysType delaysAtBegin, DelaysType delaysAfterEnd)
{
  memset(delaysAtBegin, 0, sizeof(DelaysType));
  memset(delaysAfterEnd, 0, sizeof(DelaysType));

  if (NR_INPUTS > 22)
    delaysAfterEnd[22] = 1e-6;
}
#endif


#if defined CORRECTNESS_TEST
static void setTransposeTestPattern(FilteredDataType filteredData)
{
  memset(filteredData, 0, sizeof filteredData);

  if (NR_INPUTS > 22 && NR_SAMPLES_PER_CHANNEL > 99 && NR_CHANNELS > 5) {
    filteredData[22][99][REAL][5] = 2;
    filteredData[22][99][IMAG][5] = 3;
  }
}


static void checkTransposeTestPattern(const CorrectedDataType correctedData)
{
  for (int channel = 0; channel < NR_CHANNELS; channel ++)
    for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
      for (int input = 0; input < NR_INPUTS; input ++)
	if (correctedData[channel][input / VECTOR_SIZE][time][REAL][input % VECTOR_SIZE] != 0 || correctedData[channel][input / VECTOR_SIZE][time][IMAG][input % VECTOR_SIZE] != 0)
	  cout << "channel = " << channel << ", time = " << time << ", input = " << input << ", value = (" << correctedData[channel][input / VECTOR_SIZE][time][REAL][input % VECTOR_SIZE] << ',' << correctedData[channel][input / VECTOR_SIZE][time][IMAG][input % VECTOR_SIZE] << ')' << std::endl;
}


static void testTranspose()
{
  setTransposeTestPattern(filteredData);

#if defined BANDPASS_CORRECTION
  setBandPassTestPattern(bandPassCorrectionWeights);
#pragma omp target update to(filteredData, bandPassCorrectionWeights)
#pragma omp target
  transpose(correctedData, filteredData, bandPassCorrectionWeights, 0);
#else
#pragma omp target update to(filteredData)
#pragma omp target
  transpose(correctedData, filteredData, 0);
#endif

#if defined DELAY_COMPENSATION
  setDelaysTestPattern(delaysAtBegin[0], delaysAfterEnd[0]);
#pragma omp target update to(delaysAtBegin[0], delaysAfterEnd[0])
#pragma omp target
  applyDelays(correctedData, delaysAtBegin[0], delaysAfterEnd[0], 60e6, 0);
#endif

#pragma omp target update from(correctedData)
  checkTransposeTestPattern(correctedData);
}
#endif


//////

template <typename T> static inline void cmul(T &c_r, T &c_i, T a_r, T a_i, T b_r, T b_i)
{
  c_r = a_r * b_r - a_i * b_i;
  c_i = a_r * b_i + a_i * b_r;
}


static void fused_FIRfilterInit(
  const InputDataType inputData,
  float history[COMPLEX][NR_TAPS][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/,
  unsigned input,
  unsigned,
  double &FIRfilterTime
)
{
  FIRfilterTime -= rdtsc();
  // fill FIR filter history

  for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++)
    for (unsigned time = 0; time < NR_TAPS - 1; time ++)
#pragma simd
#pragma vector aligned
      for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
	history[real_imag][time][channel] = inputData[input][real_imag][time][channel];

  FIRfilterTime += rdtsc();
}


static void fused_FIRfilter(
  const InputDataType inputData,
  float history[COMPLEX][NR_TAPS][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/,
  float filteredData[NR_SAMPLES_PER_MINOR_LOOP][COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/,
  unsigned input,
  unsigned majorTime,
  unsigned,
  double &FIRfilterTime
)
{
  FIRfilterTime -= rdtsc();

  for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++) {
    for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++) {
#pragma simd
//#pragma vector aligned // why does specifying this yields wrong results ???
      for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
	history[real_imag][(minorTime - 1) % NR_TAPS][channel] = inputData[input][real_imag][majorTime + minorTime + NR_TAPS - 1][channel];

	float sum = 0;

	for (unsigned tap = 0; tap < NR_TAPS; tap ++)
	  sum += filterWeights[tap][channel] * history[real_imag][(minorTime + tap) % NR_TAPS][channel];

	filteredData[minorTime][real_imag][channel] = sum;
      }
    }
  }

  FIRfilterTime += rdtsc();
}


static void fused_FFT(float filteredData[NR_SAMPLES_PER_MINOR_LOOP][COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/, unsigned, double &FFTtime)
{
  FFTtime -= rdtsc();

//  for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++)
//  DftiComputeForward(handle, filteredData[minorTime][REAL], filteredData[minorTime][IMAG]);
// Do batch FFT instead of for-loop
    DftiComputeForward(handle, filteredData[0][REAL], filteredData[0][IMAG]);

  FFTtime += rdtsc();
}


static void fused_TransposeInit(
#if defined DELAY_COMPENSATION
  float v[COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/,
  float dv[COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/,
#endif
#if defined BANDPASS_CORRECTION
  const BandPassCorrectionWeights bandPassCorrectionWeights,
#endif
#if defined DELAY_COMPENSATION
  const DelaysType delaysAtBegin,
  const DelaysType delaysAfterEnd,
  float subbandFrequency,
#endif
  unsigned input,
  unsigned,
  double &trsTime
)
{
#if defined DELAY_COMPENSATION
  trsTime -= rdtsc();

  // prepare delay compensation: compute complex weights
  double phiBegin = -2.0 * 3.141592653589793 * delaysAtBegin[input];
  double phiEnd   = -2.0 * 3.141592653589793 * delaysAfterEnd[input];
  double deltaPhi = (phiEnd - phiBegin) / NR_SAMPLES_PER_CHANNEL;

#pragma simd
#pragma vector aligned
  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
    double channelFrequency = subbandFrequency - .5 * SUBBAND_BANDWIDTH + channel * (SUBBAND_BANDWIDTH / NR_CHANNELS);
    float myPhiBegin = static_cast<float>((phiBegin /* + startTime * deltaPhi */) * channelFrequency /* + phaseOffsets[stationPol + major] */);
    float myPhiDelta = static_cast<float>(deltaPhi * channelFrequency);
    sincosf(myPhiBegin, &v[IMAG][channel], &v[REAL][channel]);
    sincosf(myPhiDelta, &dv[IMAG][channel], &dv[REAL][channel]);

#if defined BANDPASS_CORRECTION
    v[REAL][channel] *= bandPassCorrectionWeights[channel];
    v[IMAG][channel] *= bandPassCorrectionWeights[channel];
#endif
  }

  trsTime += rdtsc();
#endif
}


static void fused_Transpose(
  CorrectedDataType correctedData,
  float filteredData[NR_SAMPLES_PER_MINOR_LOOP][COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/,
#if defined DELAY_COMPENSATION
  float v[COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/,
  float dv[COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/,
#endif
  unsigned input,
  unsigned majorTime,
  unsigned,
  double &trsTime
)
{
  trsTime -= rdtsc();

#if defined BANDPASS_CORRECTION && !defined DELAY_COMPENSATION
  // BandPass correction, if not doing delay compensation

  for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++) {
#pragma simd
#pragma vector aligned
    for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
      for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++) {
	filteredData[minorTime][real_imag][channel] *= bandPassCorrectionWeights[channel];
      }
    }
  }
#endif

  // Delay compensation & transpose

#pragma simd
#pragma vector aligned
  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
    for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++) {
      float sample_r = filteredData[minorTime][REAL][channel];
      float sample_i = filteredData[minorTime][IMAG][channel];

#if defined DELAY_COMPENSATION
      cmul(sample_r, sample_i, sample_r, sample_i, v[REAL][channel], v[IMAG][channel]);
      cmul(v[REAL][channel], v[IMAG][channel], v[REAL][channel], v[IMAG][channel], dv[REAL][channel], dv[IMAG][channel]);
#endif

      correctedData[channel][input / VECTOR_SIZE][majorTime + minorTime][REAL][input % VECTOR_SIZE] = sample_r;
      correctedData[channel][input / VECTOR_SIZE][majorTime + minorTime][IMAG][input % VECTOR_SIZE] = sample_i;
    }
  }

  trsTime += rdtsc();
}


static void fused(
  CorrectedDataType correctedData,
  const InputDataType inputData,
  const FilterWeightsType,
#if defined BANDPASS_CORRECTION
  const BandPassCorrectionWeights bandPassCorrectionWeights,
#endif
#if defined DELAY_COMPENSATION
  const DelaysType delaysAtBegin,
  const DelaysType delaysAfterEnd,
  float subbandFrequency,
#endif
  unsigned iteration,
  double &FIRfilterTimeRef, double &FFTtimeRef, double &trsTimeRef)
{
  double FIRfilterTime = 0, FFTtime = 0, trsTime = 0;

#pragma omp parallel reduction(+: FIRfilterTime, FFTtime, trsTime)
  {
#pragma omp for schedule(dynamic)
    for (unsigned input = 0; input < NR_INPUTS; input ++) {
      float history[COMPLEX][NR_TAPS][NR_CHANNELS] __attribute__((aligned(sizeof(float[VECTOR_SIZE]))));
      float filteredData[NR_SAMPLES_PER_MINOR_LOOP][COMPLEX][NR_CHANNELS] __attribute__((aligned(sizeof(float[VECTOR_SIZE]))));

#if defined DELAY_COMPENSATION
      float v[COMPLEX][NR_CHANNELS] __attribute__((aligned(sizeof(float[VECTOR_SIZE]))));
      float dv[COMPLEX][NR_CHANNELS] __attribute__((aligned(sizeof(float[VECTOR_SIZE]))));
#endif

      fused_TransposeInit(
#if defined DELAY_COMPENSATION
              v, dv,
#endif
#if defined BANDPASS_CORRECTION
              bandPassCorrectionWeights,
#endif
#if defined DELAY_COMPENSATION
              delaysAtBegin, delaysAfterEnd, subbandFrequency,
#endif
              input, iteration, trsTime);
      fused_FIRfilterInit(inputData, history, input, iteration, FIRfilterTime);

      for (unsigned majorTime = 0; majorTime < NR_SAMPLES_PER_CHANNEL; majorTime += NR_SAMPLES_PER_MINOR_LOOP) {
	fused_FIRfilter(inputData, history, filteredData, input, majorTime, iteration, FIRfilterTime);
	fused_FFT(filteredData, iteration, FFTtime);
	fused_Transpose(correctedData, filteredData,
#if defined DELAY_COMPENSATION
                v, dv,
#endif
                input, majorTime, iteration, trsTime);
      }
    }
  }

  if (iteration > 0)
    FIRfilterTimeRef = FIRfilterTime, FFTtimeRef = FFTtime, trsTimeRef = trsTime;
}


#if defined CORRECTNESS_TEST
static void setFusedTestPattern(InputDataType inputData, FilterWeightsType filterWeights, BandPassCorrectionWeights bandPassCorrectionWeights, DelaysType delaysAtBegin, DelaysType delaysAfterEnd)
{
  memset(inputData, 0, sizeof(InputDataType));
  memset(filterWeights, 0, sizeof(FilterWeightsType));

  memset(delaysAtBegin, 0, sizeof(DelaysType));
  memset(delaysAfterEnd, 0, sizeof(DelaysType));

  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
    bandPassCorrectionWeights[channel] = 1;

  if (NR_TAPS > 11 && NR_CHANNELS > 12)
    filterWeights[15][12] = 2;

  if (NR_INPUTS > 6 && NR_SAMPLES_PER_CHANNEL > 27 && NR_CHANNELS > 12) {
    inputData[6][REAL][27 + NR_TAPS - 1][12] = 2;
    inputData[6][IMAG][27 + NR_TAPS - 1][12] = 3;
  }
}


static void checkFusedTestPattern(const CorrectedDataType correctedData)
{
  for (unsigned input = 0; input < NR_INPUTS; input ++)
    for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
      for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
	if (correctedData[channel][input / VECTOR_SIZE][time][REAL][input % VECTOR_SIZE] != 0 || correctedData[channel][input / VECTOR_SIZE][time][IMAG][input % VECTOR_SIZE] != 0)
	  cout << "input = " << input << ", time = " << time << ", channel = " << channel << ": (" << correctedData[channel][input / VECTOR_SIZE][time][REAL][input % VECTOR_SIZE] << ", " << correctedData[channel][input / VECTOR_SIZE][time][IMAG][input % VECTOR_SIZE] << ')' << std::endl;
}


static void testFused()
{
  setFusedTestPattern(inputData[0], filterWeights, bandPassCorrectionWeights, delaysAtBegin[0], delaysAfterEnd[0]);
  double FIRfilterTime, FFTtime, trsTime;

#pragma omp target update to(inputData, filterWeights, bandPassCorrectionWeights, delaysAtBegin, delaysAfterEnd)
#pragma omp target
  fused(
    correctedData, inputData[0], filterWeights,
#if defined BANDPASS_CORRECTION
    bandPassCorrectionWeights,
#endif
#if defined DELAY_COMPENSATION
    delaysAtBegin[0], delaysAfterEnd[0], 60e6,
#endif
    0, FIRfilterTime, FFTtime, trsTime
  );
#pragma omp target update from(correctedData)

  checkFusedTestPattern(correctedData);
}
#endif



////// correlator

#if defined __MIC__ || defined __AVX512F__

static inline void correlate_column(__m512 &sum_real, __m512 &sum_imag, const float *sample_X_real_ptr, const float *sample_X_imag_ptr, __m512 samples_Y_real, __m512 samples_Y_imag)
{
#if defined __MIC__
  __m512 sample_X_real = _mm512_extload_ps(sample_X_real_ptr, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  __m512 sample_X_imag = _mm512_extload_ps(sample_X_imag_ptr, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
#elif defined __AVX512F__
  __m512 sample_X_real = _mm512_set1_ps(*sample_X_real_ptr);
  __m512 sample_X_imag = _mm512_set1_ps(*sample_X_imag_ptr);
#endif

  sum_real = _mm512_fmadd_ps(samples_Y_real, sample_X_real, sum_real);
  sum_imag = _mm512_fmadd_ps(samples_Y_imag, sample_X_real, sum_imag);
  sum_real = _mm512_fmadd_ps(samples_Y_imag, sample_X_imag, sum_real);
  sum_imag = _mm512_fnmadd_ps(samples_Y_real, sample_X_imag, sum_imag);
}

#endif


#if defined __AVX__ && !defined __MIC__

static inline void correlate_column(__m256 &sum_real, __m256 &sum_imag, const float *sample_X_real_ptr, const float *sample_X_imag_ptr, __m256 samples_Y_real, __m256 samples_Y_imag)
{
  __m256 sample_X_real = _mm256_broadcast_ss(sample_X_real_ptr);
  __m256 sample_X_imag = _mm256_broadcast_ss(sample_X_imag_ptr);

#if defined __AVX2__
  sum_real = _mm256_fmadd_ps(samples_Y_real, sample_X_real, sum_real);
  sum_real = _mm256_fmadd_ps(samples_Y_imag, sample_X_imag, sum_real);
  sum_imag = _mm256_fmadd_ps(samples_Y_imag, sample_X_real, sum_imag);
  sum_imag = _mm256_fnmadd_ps(samples_Y_real, sample_X_imag, sum_imag);
#else
  sum_real = _mm256_add_ps(_mm256_mul_ps(samples_Y_real, sample_X_real), sum_real);
  sum_imag = _mm256_add_ps(_mm256_mul_ps(samples_Y_imag, sample_X_real), sum_imag);
  sum_real = _mm256_add_ps(_mm256_mul_ps(samples_Y_imag, sample_X_imag), sum_real);
  sum_imag = _mm256_sub_ps(_mm256_mul_ps(samples_Y_real, sample_X_imag), sum_imag);
#endif
}

#endif


#if defined __MIC__ || defined __AVX512F__

static inline void store_unaligned(float *ptr, __m512 value)
{
#if defined __MIC__
  _mm512_packstorelo_ps(ptr     , value);
  _mm512_packstorehi_ps(ptr + 16, value);
#elif defined __AVX512F__
  _mm512_storeu_ps(ptr, value);
#endif
}


static inline void store_unaligned(float *ptr, __mmask16 mask, __m512 value)
{
#if defined __MIC__
  _mm512_mask_packstorelo_ps(ptr     , mask, value);
  _mm512_mask_packstorehi_ps(ptr + 16, mask, value);
#elif defined __AVX512F__
  _mm512_mask_storeu_ps(ptr, mask, value);
#endif
}


static inline void write_visibilities(VisibilitiesType visibilities, int channel, int blockX, int blockY, int offset, __m512 sum_real, __m512 sum_imag)
{
#if NR_INPUTS % VECTOR_SIZE != 0
  if (VECTOR_SIZE * blockX + offset < NR_INPUTS)
#endif
  {
    int baseline = ((VECTOR_SIZE * blockX + offset) * ((VECTOR_SIZE * blockX + offset) + 1) / 2) + VECTOR_SIZE * blockY;

    if (blockX == blockY) {
      __mmask16 mask = (2 << offset) - 1;
      store_unaligned(&visibilities[channel][REAL][baseline], mask, sum_real);
      store_unaligned(&visibilities[channel][IMAG][baseline], mask, sum_imag);
    } else {
      store_unaligned(&visibilities[channel][REAL][baseline], sum_real);
      store_unaligned(&visibilities[channel][IMAG][baseline], sum_imag);
    }
  }
}

#endif


#if defined __AVX__ && !defined __MIC__

static inline void write_visibilities(VisibilitiesType visibilities, int channel, int blockX, int blockY, int offset, __m256 sum_real, __m256 sum_imag)
{
#if NR_INPUTS % VECTOR_SIZE != 0
  if (VECTOR_SIZE * blockX + offset < NR_INPUTS)
#endif
  {
    int baseline = ((VECTOR_SIZE * blockX + offset) * ((VECTOR_SIZE * blockX + offset) + 1) / 2) + VECTOR_SIZE * blockY;

    if (blockX == blockY) {
      static const __m256i masks[8] = {
	_mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0, -1),
	_mm256_set_epi32( 0,  0,  0,  0,  0,  0, -1, -1),
	_mm256_set_epi32( 0,  0,  0,  0,  0, -1, -1, -1),
	_mm256_set_epi32( 0,  0,  0,  0, -1, -1, -1, -1),
	_mm256_set_epi32( 0,  0,  0, -1, -1, -1, -1, -1),
	_mm256_set_epi32( 0,  0, -1, -1, -1, -1, -1, -1),
	_mm256_set_epi32( 0, -1, -1, -1, -1, -1, -1, -1),
	_mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1),
      };

      __m256i mask = masks[offset];
      _mm256_maskstore_ps(&visibilities[channel][REAL][baseline], mask, sum_real);
      _mm256_maskstore_ps(&visibilities[channel][IMAG][baseline], mask, sum_imag);
    } else {
      _mm256_storeu_ps(&visibilities[channel][REAL][baseline], sum_real);
      _mm256_storeu_ps(&visibilities[channel][IMAG][baseline], sum_imag);
    }
  }
//for (int i = 0; i < 8; i ++)
  //std::cout << visibilities[5][0][i] << ' ';
//std::cout << std::endl;
}

#endif


static void correlate(VisibilitiesType visibilities, const CorrectedDataType correctedData, unsigned)
{
#pragma omp parallel
  {
#if 1 && defined __AVX512F__
    // correlate blocks of 32x32 inputs
#define NR_32X32_BLOCKS ((ALIGN(NR_INPUTS, 32) / 32) * (ALIGN(NR_INPUTS, 32) / 32 + 1) / 2)

    // collapsing three loops on channel, blockX, *and* blockY does not work
#pragma omp for collapse(2) schedule(dynamic)
    for (int channel = 0; channel < NR_CHANNELS; channel ++) {
      for (int block = 0; block < NR_32X32_BLOCKS; block ++) {
	int x = (sqrtf(8 * block + 1) - .99999f) / 2;
	int y = block - x * (x + 1) / 2;

	__m512 sum_A_real, sum_A_imag;
	__m512 sum_B_real, sum_B_imag;
	__m512 sum_C_real, sum_C_imag;
	__m512 sum_D_real, sum_D_imag;
	__m512 sum_E_real, sum_E_imag;
	__m512 sum_F_real, sum_F_imag;
	__m512 sum_G_real, sum_G_imag;
	__m512 sum_H_real, sum_H_imag;
	__m512 sum_I_real, sum_I_imag;
	__m512 sum_J_real, sum_J_imag;
	__m512 sum_K_real, sum_K_imag;
	__m512 sum_L_real, sum_L_imag;
	__m512 sum_M_real, sum_M_imag;
	__m512 sum_N_real, sum_N_imag;
	__m512 sum_O_real, sum_O_imag;
	__m512 sum_P_real, sum_P_imag;

	__m512 tmp[2][2][16][2] __attribute__((aligned(64)));

	for (int timeMajor = 0; timeMajor < NR_SAMPLES_PER_CHANNEL; timeMajor += 64) {
	  for (int yMinor = 0; yMinor < 2; yMinor ++) {
	    for (int xMinor = 0; xMinor < 2; xMinor ++) {
	      int blockX = 2 * x + xMinor, blockY = 2 * y + yMinor;

	      if (blockY <= blockX) {
		if (timeMajor == 0) {
		  sum_A_real = _mm512_setzero_ps(), sum_A_imag = _mm512_setzero_ps();
		  sum_B_real = _mm512_setzero_ps(), sum_B_imag = _mm512_setzero_ps();
		  sum_C_real = _mm512_setzero_ps(), sum_C_imag = _mm512_setzero_ps();
		  sum_D_real = _mm512_setzero_ps(), sum_D_imag = _mm512_setzero_ps();
		  sum_E_real = _mm512_setzero_ps(), sum_E_imag = _mm512_setzero_ps();
		  sum_F_real = _mm512_setzero_ps(), sum_F_imag = _mm512_setzero_ps();
		  sum_G_real = _mm512_setzero_ps(), sum_G_imag = _mm512_setzero_ps();
		  sum_H_real = _mm512_setzero_ps(), sum_H_imag = _mm512_setzero_ps();
		  sum_I_real = _mm512_setzero_ps(), sum_I_imag = _mm512_setzero_ps();
		  sum_J_real = _mm512_setzero_ps(), sum_J_imag = _mm512_setzero_ps();
		  sum_K_real = _mm512_setzero_ps(), sum_K_imag = _mm512_setzero_ps();
		  sum_L_real = _mm512_setzero_ps(), sum_L_imag = _mm512_setzero_ps();
		  sum_M_real = _mm512_setzero_ps(), sum_M_imag = _mm512_setzero_ps();
		  sum_N_real = _mm512_setzero_ps(), sum_N_imag = _mm512_setzero_ps();
		  sum_O_real = _mm512_setzero_ps(), sum_O_imag = _mm512_setzero_ps();
		  sum_P_real = _mm512_setzero_ps(), sum_P_imag = _mm512_setzero_ps();
		} else {
		  sum_A_real = tmp[yMinor][xMinor][ 0][0], sum_A_imag = tmp[yMinor][xMinor][ 0][1];
		  sum_B_real = tmp[yMinor][xMinor][ 1][0], sum_B_imag = tmp[yMinor][xMinor][ 1][1];
		  sum_C_real = tmp[yMinor][xMinor][ 2][0], sum_C_imag = tmp[yMinor][xMinor][ 2][1];
		  sum_D_real = tmp[yMinor][xMinor][ 3][0], sum_D_imag = tmp[yMinor][xMinor][ 3][1];
		  sum_E_real = tmp[yMinor][xMinor][ 4][0], sum_E_imag = tmp[yMinor][xMinor][ 4][1];
		  sum_F_real = tmp[yMinor][xMinor][ 5][0], sum_F_imag = tmp[yMinor][xMinor][ 5][1];
		  sum_G_real = tmp[yMinor][xMinor][ 6][0], sum_G_imag = tmp[yMinor][xMinor][ 6][1];
		  sum_H_real = tmp[yMinor][xMinor][ 7][0], sum_H_imag = tmp[yMinor][xMinor][ 7][1];
		  sum_I_real = tmp[yMinor][xMinor][ 8][0], sum_I_imag = tmp[yMinor][xMinor][ 8][1];
		  sum_J_real = tmp[yMinor][xMinor][ 9][0], sum_J_imag = tmp[yMinor][xMinor][ 9][1];
		  sum_K_real = tmp[yMinor][xMinor][10][0], sum_K_imag = tmp[yMinor][xMinor][10][1];
		  sum_L_real = tmp[yMinor][xMinor][11][0], sum_L_imag = tmp[yMinor][xMinor][11][1];
		  sum_M_real = tmp[yMinor][xMinor][12][0], sum_M_imag = tmp[yMinor][xMinor][12][1];
		  sum_N_real = tmp[yMinor][xMinor][13][0], sum_N_imag = tmp[yMinor][xMinor][13][1];
		  sum_O_real = tmp[yMinor][xMinor][14][0], sum_O_imag = tmp[yMinor][xMinor][14][1];
		  sum_P_real = tmp[yMinor][xMinor][15][0], sum_P_imag = tmp[yMinor][xMinor][15][1];
		}

		for (int timeMinor = 0; timeMinor < 64; timeMinor ++) {
		  int time = timeMajor + timeMinor;

		  __m512 samples_Y_real = _mm512_load_ps((__m512 *) &correctedData[channel][blockY][time][REAL][0]);
		  __m512 samples_Y_imag = _mm512_load_ps((__m512 *) &correctedData[channel][blockY][time][IMAG][0]);

		  correlate_column(sum_A_real, sum_A_imag, &correctedData[channel][blockX][time][REAL][ 0], &correctedData[channel][blockX][time][IMAG][ 0], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_B_real, sum_B_imag, &correctedData[channel][blockX][time][REAL][ 1], &correctedData[channel][blockX][time][IMAG][ 1], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_C_real, sum_C_imag, &correctedData[channel][blockX][time][REAL][ 2], &correctedData[channel][blockX][time][IMAG][ 2], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_D_real, sum_D_imag, &correctedData[channel][blockX][time][REAL][ 3], &correctedData[channel][blockX][time][IMAG][ 3], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_E_real, sum_E_imag, &correctedData[channel][blockX][time][REAL][ 4], &correctedData[channel][blockX][time][IMAG][ 4], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_F_real, sum_F_imag, &correctedData[channel][blockX][time][REAL][ 5], &correctedData[channel][blockX][time][IMAG][ 5], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_G_real, sum_G_imag, &correctedData[channel][blockX][time][REAL][ 6], &correctedData[channel][blockX][time][IMAG][ 6], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_H_real, sum_H_imag, &correctedData[channel][blockX][time][REAL][ 7], &correctedData[channel][blockX][time][IMAG][ 7], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_I_real, sum_I_imag, &correctedData[channel][blockX][time][REAL][ 8], &correctedData[channel][blockX][time][IMAG][ 8], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_J_real, sum_J_imag, &correctedData[channel][blockX][time][REAL][ 9], &correctedData[channel][blockX][time][IMAG][ 9], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_K_real, sum_K_imag, &correctedData[channel][blockX][time][REAL][10], &correctedData[channel][blockX][time][IMAG][10], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_L_real, sum_L_imag, &correctedData[channel][blockX][time][REAL][11], &correctedData[channel][blockX][time][IMAG][11], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_M_real, sum_M_imag, &correctedData[channel][blockX][time][REAL][12], &correctedData[channel][blockX][time][IMAG][12], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_N_real, sum_N_imag, &correctedData[channel][blockX][time][REAL][13], &correctedData[channel][blockX][time][IMAG][13], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_O_real, sum_O_imag, &correctedData[channel][blockX][time][REAL][14], &correctedData[channel][blockX][time][IMAG][14], samples_Y_real, samples_Y_imag);
		  correlate_column(sum_P_real, sum_P_imag, &correctedData[channel][blockX][time][REAL][15], &correctedData[channel][blockX][time][IMAG][15], samples_Y_real, samples_Y_imag);
		}

		if (timeMajor + 64 == NR_SAMPLES_PER_CHANNEL) {
		  write_visibilities(visibilities, channel, blockX, blockY,  0, sum_A_real, sum_A_imag);
		  write_visibilities(visibilities, channel, blockX, blockY,  1, sum_B_real, sum_B_imag);
		  write_visibilities(visibilities, channel, blockX, blockY,  2, sum_C_real, sum_C_imag);
		  write_visibilities(visibilities, channel, blockX, blockY,  3, sum_D_real, sum_D_imag);
		  write_visibilities(visibilities, channel, blockX, blockY,  4, sum_E_real, sum_E_imag);
		  write_visibilities(visibilities, channel, blockX, blockY,  5, sum_F_real, sum_F_imag);
		  write_visibilities(visibilities, channel, blockX, blockY,  6, sum_G_real, sum_G_imag);
		  write_visibilities(visibilities, channel, blockX, blockY,  7, sum_H_real, sum_H_imag);
		  write_visibilities(visibilities, channel, blockX, blockY,  8, sum_I_real, sum_I_imag);
		  write_visibilities(visibilities, channel, blockX, blockY,  9, sum_J_real, sum_J_imag);
		  write_visibilities(visibilities, channel, blockX, blockY, 10, sum_K_real, sum_K_imag);
		  write_visibilities(visibilities, channel, blockX, blockY, 11, sum_L_real, sum_L_imag);
		  write_visibilities(visibilities, channel, blockX, blockY, 12, sum_M_real, sum_M_imag);
		  write_visibilities(visibilities, channel, blockX, blockY, 13, sum_N_real, sum_N_imag);
		  write_visibilities(visibilities, channel, blockX, blockY, 14, sum_O_real, sum_O_imag);
		  write_visibilities(visibilities, channel, blockX, blockY, 15, sum_P_real, sum_P_imag);
		} else {
		  tmp[yMinor][xMinor][ 0][0] = sum_A_real, tmp[yMinor][xMinor][ 0][1] = sum_A_imag;
		  tmp[yMinor][xMinor][ 1][0] = sum_B_real, tmp[yMinor][xMinor][ 1][1] = sum_B_imag;
		  tmp[yMinor][xMinor][ 2][0] = sum_C_real, tmp[yMinor][xMinor][ 2][1] = sum_C_imag;
		  tmp[yMinor][xMinor][ 3][0] = sum_D_real, tmp[yMinor][xMinor][ 3][1] = sum_D_imag;
		  tmp[yMinor][xMinor][ 4][0] = sum_E_real, tmp[yMinor][xMinor][ 4][1] = sum_E_imag;
		  tmp[yMinor][xMinor][ 5][0] = sum_F_real, tmp[yMinor][xMinor][ 5][1] = sum_F_imag;
		  tmp[yMinor][xMinor][ 6][0] = sum_G_real, tmp[yMinor][xMinor][ 6][1] = sum_G_imag;
		  tmp[yMinor][xMinor][ 7][0] = sum_H_real, tmp[yMinor][xMinor][ 7][1] = sum_H_imag;
		  tmp[yMinor][xMinor][ 8][0] = sum_I_real, tmp[yMinor][xMinor][ 8][1] = sum_I_imag;
		  tmp[yMinor][xMinor][ 9][0] = sum_J_real, tmp[yMinor][xMinor][ 9][1] = sum_J_imag;
		  tmp[yMinor][xMinor][10][0] = sum_K_real, tmp[yMinor][xMinor][10][1] = sum_K_imag;
		  tmp[yMinor][xMinor][11][0] = sum_L_real, tmp[yMinor][xMinor][11][1] = sum_L_imag;
		  tmp[yMinor][xMinor][12][0] = sum_M_real, tmp[yMinor][xMinor][12][1] = sum_M_imag;
		  tmp[yMinor][xMinor][13][0] = sum_N_real, tmp[yMinor][xMinor][13][1] = sum_N_imag;
		  tmp[yMinor][xMinor][14][0] = sum_O_real, tmp[yMinor][xMinor][14][1] = sum_O_imag;
		  tmp[yMinor][xMinor][15][0] = sum_P_real, tmp[yMinor][xMinor][15][1] = sum_P_imag;
		}
	      }
	    }
	  }
	}
      }
    }
#elif 1 && (defined __MIC__ || defined __AVX512F__)
      // correlate blocks of 16x16 inputs
#define NR_16X16_BLOCKS ((ALIGN(NR_INPUTS, 16) / 16) * (ALIGN(NR_INPUTS, 16) / 16 + 1) / 2)

      // collapsing three loops on channel, blockX, *and* blockY does not work
#pragma omp for collapse(2) schedule(dynamic)
      for (int channel = 0; channel < NR_CHANNELS; channel ++) {
	for (int block = 0; block < NR_16X16_BLOCKS; block ++) {
	  int blockX = (sqrtf(8 * block + 1) - .99999f) / 2;
	  int blockY = block - blockX * (blockX + 1) / 2;

#if defined __MIC__
	for (int time = 0; time < (NR_SAMPLES_PER_CHANNEL < 8 ? NR_SAMPLES_PER_CHANNEL : 8); time ++) {
	  _mm_prefetch((const char *) &correctedData[channel][blockX][time][REAL][0], _MM_HINT_T0);
	  _mm_prefetch((const char *) &correctedData[channel][blockX][time][IMAG][0], _MM_HINT_T0);
	  _mm_prefetch((const char *) &correctedData[channel][blockY][time][REAL][0], _MM_HINT_T0);
	  _mm_prefetch((const char *) &correctedData[channel][blockY][time][IMAG][0], _MM_HINT_T0);
	}
#endif

	__m512 sum_A_real = _mm512_setzero_ps(), sum_A_imag = _mm512_setzero_ps();
	__m512 sum_B_real = _mm512_setzero_ps(), sum_B_imag = _mm512_setzero_ps();
	__m512 sum_C_real = _mm512_setzero_ps(), sum_C_imag = _mm512_setzero_ps();
	__m512 sum_D_real = _mm512_setzero_ps(), sum_D_imag = _mm512_setzero_ps();
	__m512 sum_E_real = _mm512_setzero_ps(), sum_E_imag = _mm512_setzero_ps();
	__m512 sum_F_real = _mm512_setzero_ps(), sum_F_imag = _mm512_setzero_ps();
	__m512 sum_G_real = _mm512_setzero_ps(), sum_G_imag = _mm512_setzero_ps();
	__m512 sum_H_real = _mm512_setzero_ps(), sum_H_imag = _mm512_setzero_ps();
	__m512 sum_I_real = _mm512_setzero_ps(), sum_I_imag = _mm512_setzero_ps();
	__m512 sum_J_real = _mm512_setzero_ps(), sum_J_imag = _mm512_setzero_ps();
	__m512 sum_K_real = _mm512_setzero_ps(), sum_K_imag = _mm512_setzero_ps();
	__m512 sum_L_real = _mm512_setzero_ps(), sum_L_imag = _mm512_setzero_ps();
	__m512 sum_M_real = _mm512_setzero_ps(), sum_M_imag = _mm512_setzero_ps();
	__m512 sum_N_real = _mm512_setzero_ps(), sum_N_imag = _mm512_setzero_ps();
	__m512 sum_O_real = _mm512_setzero_ps(), sum_O_imag = _mm512_setzero_ps();
	__m512 sum_P_real = _mm512_setzero_ps(), sum_P_imag = _mm512_setzero_ps();

#pragma noprefetch
	for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	  __m512 samples_Y_real = _mm512_load_ps((__m512 *) &correctedData[channel][blockY][time][REAL][0]);
	  __m512 samples_Y_imag = _mm512_load_ps((__m512 *) &correctedData[channel][blockY][time][IMAG][0]);

#if defined __MIC__
	  _mm_prefetch((const char *) &correctedData[channel][blockX][time + 8][REAL][0], _MM_HINT_T1);
	  _mm_prefetch((const char *) &correctedData[channel][blockY][time + 8][REAL][0], _MM_HINT_T1);
	  _mm_prefetch((const char *) &correctedData[channel][blockX][time + 8][IMAG][0], _MM_HINT_T1);
	  _mm_prefetch((const char *) &correctedData[channel][blockY][time + 8][IMAG][0], _MM_HINT_T1);
	  _mm_prefetch((const char *) &correctedData[channel][blockX][time + 1][REAL][0], _MM_HINT_T0);
	  _mm_prefetch((const char *) &correctedData[channel][blockY][time + 1][REAL][0], _MM_HINT_T0);
	  _mm_prefetch((const char *) &correctedData[channel][blockX][time + 1][IMAG][0], _MM_HINT_T0);
	  _mm_prefetch((const char *) &correctedData[channel][blockY][time + 1][IMAG][0], _MM_HINT_T0);
#endif

	  correlate_column(sum_A_real, sum_A_imag, &correctedData[channel][blockX][time][REAL][ 0], &correctedData[channel][blockX][time][IMAG][ 0], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_B_real, sum_B_imag, &correctedData[channel][blockX][time][REAL][ 1], &correctedData[channel][blockX][time][IMAG][ 1], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_C_real, sum_C_imag, &correctedData[channel][blockX][time][REAL][ 2], &correctedData[channel][blockX][time][IMAG][ 2], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_D_real, sum_D_imag, &correctedData[channel][blockX][time][REAL][ 3], &correctedData[channel][blockX][time][IMAG][ 3], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_E_real, sum_E_imag, &correctedData[channel][blockX][time][REAL][ 4], &correctedData[channel][blockX][time][IMAG][ 4], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_F_real, sum_F_imag, &correctedData[channel][blockX][time][REAL][ 5], &correctedData[channel][blockX][time][IMAG][ 5], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_G_real, sum_G_imag, &correctedData[channel][blockX][time][REAL][ 6], &correctedData[channel][blockX][time][IMAG][ 6], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_H_real, sum_H_imag, &correctedData[channel][blockX][time][REAL][ 7], &correctedData[channel][blockX][time][IMAG][ 7], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_I_real, sum_I_imag, &correctedData[channel][blockX][time][REAL][ 8], &correctedData[channel][blockX][time][IMAG][ 8], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_J_real, sum_J_imag, &correctedData[channel][blockX][time][REAL][ 9], &correctedData[channel][blockX][time][IMAG][ 9], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_K_real, sum_K_imag, &correctedData[channel][blockX][time][REAL][10], &correctedData[channel][blockX][time][IMAG][10], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_L_real, sum_L_imag, &correctedData[channel][blockX][time][REAL][11], &correctedData[channel][blockX][time][IMAG][11], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_M_real, sum_M_imag, &correctedData[channel][blockX][time][REAL][12], &correctedData[channel][blockX][time][IMAG][12], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_N_real, sum_N_imag, &correctedData[channel][blockX][time][REAL][13], &correctedData[channel][blockX][time][IMAG][13], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_O_real, sum_O_imag, &correctedData[channel][blockX][time][REAL][14], &correctedData[channel][blockX][time][IMAG][14], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_P_real, sum_P_imag, &correctedData[channel][blockX][time][REAL][15], &correctedData[channel][blockX][time][IMAG][15], samples_Y_real, samples_Y_imag);
	}

	write_visibilities(visibilities, channel, blockX, blockY,  0, sum_A_real, sum_A_imag);
	write_visibilities(visibilities, channel, blockX, blockY,  1, sum_B_real, sum_B_imag);
	write_visibilities(visibilities, channel, blockX, blockY,  2, sum_C_real, sum_C_imag);
	write_visibilities(visibilities, channel, blockX, blockY,  3, sum_D_real, sum_D_imag);
	write_visibilities(visibilities, channel, blockX, blockY,  4, sum_E_real, sum_E_imag);
	write_visibilities(visibilities, channel, blockX, blockY,  5, sum_F_real, sum_F_imag);
	write_visibilities(visibilities, channel, blockX, blockY,  6, sum_G_real, sum_G_imag);
	write_visibilities(visibilities, channel, blockX, blockY,  7, sum_H_real, sum_H_imag);
	write_visibilities(visibilities, channel, blockX, blockY,  8, sum_I_real, sum_I_imag);
	write_visibilities(visibilities, channel, blockX, blockY,  9, sum_J_real, sum_J_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 10, sum_K_real, sum_K_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 11, sum_L_real, sum_L_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 12, sum_M_real, sum_M_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 13, sum_N_real, sum_N_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 14, sum_O_real, sum_O_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 15, sum_P_real, sum_P_imag);
      }
    }
#elif 1
    // correlate blocks of 8x8 inputs
#define NR_8X8_BLOCKS ((ALIGN(NR_INPUTS, 8) / 8) * (ALIGN(NR_INPUTS, 8) / 8 + 1) / 2)

    // collapsing three loops on channel, blockX, *and* blockY does not work
#pragma omp for collapse(2) schedule(dynamic)
    for (int channel = 0; channel < NR_CHANNELS; channel ++) {
      for (int block = 0; block < NR_8X8_BLOCKS; block ++) {
	int blockX = static_cast<int>(sqrtf(8 * static_cast<float>(block) + 1) - .99999f) / 2;
	int blockY = block - blockX * (blockX + 1) / 2;

#if 0
#pragma unroll
	for (int time = 0; time < (NR_SAMPLES_PER_CHANNEL < 3 ? NR_SAMPLES_PER_CHANNEL : 3); time ++) {
	  _mm_prefetch((const char *) &correctedData[channel][blockX][time][REAL][0], _MM_HINT_NTA);
	  _mm_prefetch((const char *) &correctedData[channel][blockX][time][IMAG][0], _MM_HINT_NTA);
	  _mm_prefetch((const char *) &correctedData[channel][blockY][time][REAL][0], _MM_HINT_NTA);
	  _mm_prefetch((const char *) &correctedData[channel][blockY][time][IMAG][0], _MM_HINT_NTA);
	}
#endif

	__m256 sum_A_real = _mm256_setzero_ps(), sum_A_imag = _mm256_setzero_ps();
	__m256 sum_B_real = _mm256_setzero_ps(), sum_B_imag = _mm256_setzero_ps();
	__m256 sum_C_real = _mm256_setzero_ps(), sum_C_imag = _mm256_setzero_ps();
	__m256 sum_D_real = _mm256_setzero_ps(), sum_D_imag = _mm256_setzero_ps();
	__m256 sum_E_real = _mm256_setzero_ps(), sum_E_imag = _mm256_setzero_ps();
	__m256 sum_F_real = _mm256_setzero_ps(), sum_F_imag = _mm256_setzero_ps();
	__m256 sum_G_real = _mm256_setzero_ps(), sum_G_imag = _mm256_setzero_ps();
	__m256 sum_H_real = _mm256_setzero_ps(), sum_H_imag = _mm256_setzero_ps();

#pragma noprefetch
	for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	  __m256 samples_Y_real = _mm256_load_ps(&correctedData[channel][blockY][time][REAL][0]);
	  __m256 samples_Y_imag = _mm256_load_ps(&correctedData[channel][blockY][time][IMAG][0]);

	  _mm_prefetch((const char *) &correctedData[channel][blockX][time + 3][REAL][0], _MM_HINT_T0);
	  _mm_prefetch((const char *) &correctedData[channel][blockY][time + 3][REAL][0], _MM_HINT_T0); 
	  correlate_column(sum_A_real, sum_A_imag, &correctedData[channel][blockX][time][REAL][0], &correctedData[channel][blockX][time][IMAG][0], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_B_real, sum_B_imag, &correctedData[channel][blockX][time][REAL][1], &correctedData[channel][blockX][time][IMAG][1], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_C_real, sum_C_imag, &correctedData[channel][blockX][time][REAL][2], &correctedData[channel][blockX][time][IMAG][2], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_D_real, sum_D_imag, &correctedData[channel][blockX][time][REAL][3], &correctedData[channel][blockX][time][IMAG][3], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_E_real, sum_E_imag, &correctedData[channel][blockX][time][REAL][4], &correctedData[channel][blockX][time][IMAG][4], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_F_real, sum_F_imag, &correctedData[channel][blockX][time][REAL][5], &correctedData[channel][blockX][time][IMAG][5], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_G_real, sum_G_imag, &correctedData[channel][blockX][time][REAL][6], &correctedData[channel][blockX][time][IMAG][6], samples_Y_real, samples_Y_imag);
	  correlate_column(sum_H_real, sum_H_imag, &correctedData[channel][blockX][time][REAL][7], &correctedData[channel][blockX][time][IMAG][7], samples_Y_real, samples_Y_imag);
	}

	write_visibilities(visibilities, channel, blockX, blockY, 0, sum_A_real, sum_A_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 1, sum_B_real, sum_B_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 2, sum_C_real, sum_C_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 3, sum_D_real, sum_D_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 4, sum_E_real, sum_E_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 5, sum_F_real, sum_F_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 6, sum_G_real, sum_G_imag);
	write_visibilities(visibilities, channel, blockX, blockY, 7, sum_H_real, sum_H_imag);
      }
    }
#else
#pragma omp for collapse(2) schedule(dynamic,16)
    for (int channel = 0; channel < NR_CHANNELS; channel ++) {
      for (int statY = 0; statY < NR_INPUTS; statY ++) {
#pragma omp simd
	for (int statX = 0; statX <= statY; statX ++) {
	  float sum_real = 0, sum_imag = 0;

	  for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	    float sample_X_real = correctedData[channel][statX / VECTOR_SIZE][time][REAL][statX % VECTOR_SIZE];
	    float sample_X_imag = correctedData[channel][statX / VECTOR_SIZE][time][IMAG][statX % VECTOR_SIZE];
	    float sample_Y_real = correctedData[channel][statY / VECTOR_SIZE][time][REAL][statY % VECTOR_SIZE];
	    float sample_Y_imag = correctedData[channel][statY / VECTOR_SIZE][time][IMAG][statY % VECTOR_SIZE];

	    sum_real += sample_X_real * sample_Y_real;
	    sum_imag += sample_X_imag * sample_Y_real;
	    sum_real += sample_X_imag * sample_Y_imag;
	    sum_imag -= sample_X_real * sample_Y_imag;
	  }

	  int baseline = statX * (statX + 1) / 2 + statY;
	  visibilities[channel][REAL][baseline] = sum_real;
	  visibilities[channel][IMAG][baseline] = sum_imag;
	}
      }
    }
#endif
  }
}


#if defined CORRECTNESS_TEST
static void setCorrelatorTestPattern(CorrectedDataType correctedData)
{
  memset(correctedData, 0, sizeof correctedData);

#if NR_CHANNELS > 5 && NR_SAMPLES_PER_CHANNEL > 99 && NR_INPUTS > 19
  correctedData[5][ 0 / VECTOR_SIZE][99][REAL][ 0 % VECTOR_SIZE] = 3;
  correctedData[5][ 0 / VECTOR_SIZE][99][IMAG][ 0 % VECTOR_SIZE] = 4;
  correctedData[5][18 / VECTOR_SIZE][99][REAL][18 % VECTOR_SIZE] = 5;
  correctedData[5][18 / VECTOR_SIZE][99][IMAG][18 % VECTOR_SIZE] = 6;
#endif
}


static void checkCorrelatorTestPattern(const VisibilitiesType visibilities)
{
  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
    for (unsigned baseline = 0; baseline < NR_BASELINES; baseline ++)
      if (visibilities[channel][REAL][baseline] != 0 || visibilities[channel][IMAG][baseline] != 0)
	cout << "channel = " << channel << ", baseline = " << baseline << ", visibility = (" << visibilities[channel][REAL][baseline] << ',' << visibilities[channel][IMAG][baseline] << ')' << std::endl;
}


static void testCorrelator()
{
  setCorrelatorTestPattern(correctedData);

#pragma omp target update to(correctedData)
#pragma omp target
  correlate(visibilities[0], correctedData, 0);
#pragma omp target update from(visibilities[0])

  checkCorrelatorTestPattern(visibilities[0]);
}
#endif


static void report(const char *msg, uint64_t nrOperations, uint64_t nrBytes, const PowerSensor::State &startState, const PowerSensor::State &stopState, double weight = 1)
{
#if defined CORRECTNESS_TEST
    (void) msg;
    (void) nrOperations;
    (void) nrBytes;
    (void) startState;
    (void) stopState;
    (void) weight;
#else
  powerSensor.mark(startState, msg);

  double runTime = PowerSensor::seconds(startState, stopState) * weight;
#if defined MEASURE_POWER
  double energy  = PowerSensor::Joules(startState, stopState) * weight;
#endif

  cout << msg << ": " << runTime << " s, "
	    << nrOperations * 1e-12 / runTime << " TFLOPS, "
	    << nrBytes * 1e-9 / runTime << " GB/s"
#if defined MEASURE_POWER
	    ", " << energy / runTime << " W"
	    ", " << nrOperations * 1e-9 / energy << " GFLOPS/W"
#endif
	    << std::endl;
#endif
}


#if 0
#pragma omp declare target
void pipeline(
  FilteredDataType filteredData,
  const InputDataType inputData,
  const FilterWeightsType filterWeights,
//#if defined BANDPASS_CORRECTION
  const BandPassCorrectionWeights bandPassCorrectionWeights,
//#endif
  CorrectedDataType correctedData,
  const DelaysType delaysAtBegin,
  const DelaysType delaysAfterEnd,
  VisibilitiesType visibilities,
  double subbandFrequency,
  double runTimes[6],
  double powers[6]
)
{
#if !defined USE_FUSED_FILTER
  filter(filteredData, inputData, filterWeights, runTimes[0], powers[0]);
  FFT(filteredData, runTimes[1], powers[1]);
#if defined BANDPASS_CORRECTION
  transpose(correctedData, filteredData, bandPassCorrectionWeights, runTimes[2], powers[2]);
#else
  transpose(correctedData, filteredData, runTimes[2], powers[2]);
#endif
#if defined DELAY_COMPENSATION
  applyDelays(correctedData, delaysAtBegin, delaysAfterEnd, subbandFrequency, runTimes[3], powers[3]);
#endif
#else
  fused(correctedData, inputData, filterWeights,
#if defined BANDPASS_CORRECTION
  bandPassCorrectionWeights,
#endif
#if defined DELAY_COMPENSATION
  delaysAtBegin, delaysAfterEnd, subbandFrequency,
#endif
  runTimes[5], powers[5]);
#endif

  correlate(visibilities, correctedData, runTimes[4], powers[4]);
}
#endif


static void pipeline(int stream, float subbandFrequency, unsigned iteration)
{
  PowerSensor::State powerStates[8];
#if defined USE_FUSED_FILTER
  double FIRfilterTime, FFTtime, trsTime;
#endif

  //powerStates[0] = powerSensor.read();

#pragma omp critical (XeonPhi)
  {
    powerStates[1] = powerSensor.read();

#if 0
#pragma omp target map(to:subbandFrequency)
    pipeline(filteredData, inputData[stream], filterWeights, bandPassCorrectionWeights, correctedData, delaysAtBegin[stream], delaysAfterEnd[stream], visibilities[stream], subbandFrequency);
#else
#if !defined USE_FUSED_FILTER
#pragma omp target map(to:iteration)
    filter(filteredData, inputData[stream], filterWeights, iteration);

    powerStates[2] = powerSensor.read();

#pragma omp target
    FFT(filteredData, iteration);

    powerStates[3] = powerSensor.read();

#if defined BANDPASS_CORRECTION
#pragma omp target map(to:iteration)
    transpose(correctedData, filteredData, bandPassCorrectionWeights, iteration);
#else
#pragma omp target map(to:iteration)
    transpose(correctedData, filteredData, iteration);
#endif

    powerStates[4] = powerSensor.read();

#if defined DELAY_COMPENSATION
#pragma omp target map(to:subbandFrequency, iteration)
    applyDelays(correctedData, delaysAtBegin[stream], delaysAfterEnd[stream], subbandFrequency, iteration);
#endif

#else
#pragma omp target map(to:subbandFrequency, iteration) map(from:FIRfilterTime, FFTtime, trsTime)
    fused(correctedData, inputData[stream], filterWeights,
#if defined BANDPASS_CORRECTION
    bandPassCorrectionWeights,
#endif
#if defined DELAY_COMPENSATION
    delaysAtBegin[stream], delaysAfterEnd[stream], subbandFrequency,
#endif
    iteration, FIRfilterTime, FFTtime, trsTime);
#endif

    powerStates[5] = powerSensor.read();

#pragma omp target map(to:iteration)
    correlate(visibilities[stream], correctedData, iteration);
#endif

    powerStates[6] = powerSensor.read();
  }

  if (iteration > 0) // do not count first iteration
#pragma omp critical (cout)
  {
    uint64_t nrFIRfilterOperations = NR_SAMPLES * COMPLEX * NR_TAPS * 2;
    uint64_t nrFFToperations       = static_cast<uint64_t>(NR_SAMPLES * 5 * log2(NR_CHANNELS));

#if defined DELAY_COMPENSATION
    uint64_t nrDelayAndBandPassOperations = NR_SAMPLES * 2 * 6;
#elif defined BANDPASS_CORRECTION
    uint64_t nrDelayAndBandPassOperations = NR_SAMPLES * COMPLEX;
#else
    uint64_t nrDelayAndBandPassOperations = 0;
#endif

    uint64_t nrCorrelatorOperations = (uint64_t) NR_INPUTS * NR_INPUTS / 2 * 8ULL * NR_CHANNELS * NR_SAMPLES_PER_CHANNEL;
    uint64_t nrFusedOperations = nrFIRfilterOperations + nrFFToperations + nrDelayAndBandPassOperations;
#if defined USE_FUSED_FILTER
    double fusedTime = FIRfilterTime + FFTtime + trsTime;
#endif

    totalNrOperations += nrFusedOperations + nrCorrelatorOperations; // is already atomic

#if !defined USE_FUSED_FILTER
    report("FIR", nrFIRfilterOperations, sizeof(InputDataType) + sizeof(FilteredDataType), powerStates[1], powerStates[2]);
    report("FFT", nrFFToperations, 2 * sizeof(FilteredDataType), powerStates[2], powerStates[3]);
#if defined BANDPASS_CORRECTION
    report("trs", nrDelayAndBandPassOperations, sizeof(FilteredDataType) + sizeof(CorrectedDataType), powerStates[3], powerStates[4]);
#else
    report("trs", 0, sizeof(FilteredDataType) + sizeof(CorrectedDataType), powerStates[3], powerStates[4]);
#endif
    report("del", NR_SAMPLES * 2 * 6, 2 * sizeof(CorrectedDataType), powerStates[4], powerStates[5]);
#else

    report("FIR", nrFIRfilterOperations, sizeof(InputDataType), powerStates[1], powerStates[5], (double) FIRfilterTime / fusedTime);
    report("FFT", nrFFToperations, 0, powerStates[1], powerStates[5], (double) FFTtime / fusedTime);
    report("trs", nrDelayAndBandPassOperations, sizeof(FilteredDataType), powerStates[1], powerStates[5], (double) trsTime / fusedTime);
    report("fused", nrFusedOperations, sizeof(InputDataType) + sizeof(CorrectedDataType), powerStates[1], powerStates[5]);
#endif

    report("cor", nrCorrelatorOperations, sizeof(CorrectedDataType) + sizeof(VisibilitiesType), powerStates[5], powerStates[6]);
  }
}


int main(int, char **)
{
  assert(NR_CHANNELS % 16 == 0);
  assert(NR_SAMPLES_PER_CHANNEL % NR_SAMPLES_PER_MINOR_LOOP == 0);

  PowerSensor::State startState;
#if !defined CORRECTNESS_TEST
  PowerSensor::State stopState;
#endif

#pragma omp target
  fftInit();

#if defined CORRECTNESS_TEST
  testFused();
  testFIR_Filter();
  testTranspose();
  testCorrelator();
#endif
  omp_set_nested(1);

#if defined BANDPASS_CORRECTION
  setBandPassTestPattern(bandPassCorrectionWeights);
#pragma omp target update to(bandPassCorrectionWeights)
#endif

#pragma omp target update to(filterWeights)

#pragma omp parallel num_threads(NR_STREAMS)
  {
    int stream = omp_get_thread_num();

    setInputTestPattern(inputData[stream]);
#if defined DELAY_COMPENSATION
    setDelaysTestPattern(delaysAtBegin[stream], delaysAfterEnd[stream]);
#endif

    setFilterWeightsTestPattern(filterWeights);

    for (unsigned i = 0; i < 100 && (i < 2 || PowerSensor::seconds(startState, powerSensor.read()) < 20); i ++) {
      if (i == 1)
      {
#pragma omp barrier
#pragma omp single
	startState = powerSensor.read();
      }

      pipeline(stream, 60e6, i);
    }
    cout << std::endl;
    checkCorrelatorTestPattern(visibilities[stream]);

  }

#if !defined CORRECTNESS_TEST
  stopState = powerSensor.read();

  cout << "total: " << PowerSensor::seconds(startState, stopState) << " s"
	       ", " << totalNrOperations / PowerSensor::seconds(startState, stopState) * 1e-12 << " TFLOPS"
#if defined MEASURE_POWER
	       ", " << PowerSensor::Watt(startState, stopState) << " W"
	       ", " << totalNrOperations / PowerSensor::Joules(startState, stopState) * 1e-9 << " GFLOPS/W"
#endif
	       << std::endl;
#endif

#pragma omp target
  fftDestroy();
  return 0;
}
