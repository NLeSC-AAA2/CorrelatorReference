// (C) 2013,2014,2015 John Romein/ASTRON
// Copyright 2018-2019 Netherlands eScience Center and ASTRON
// Licensed under the Apache License, version 2.0. See LICENSE for details.
#include <complex>
#include <iostream>
#include <stdexcept>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>

#include <fcntl.h>
#include <immintrin.h>
#include <mkl.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


constexpr int NR_INPUTS = 2 * 576;
constexpr int VECTOR_SIZE = 8;

constexpr int NR_CHANNELS = 64;
constexpr int NR_SAMPLES_PER_CHANNEL = 3072;
constexpr uint64_t NR_SAMPLES = NR_INPUTS * NR_SAMPLES_PER_CHANNEL * NR_CHANNELS;
#if defined DELAY_COMPENSATION
constexpr float SUBBAND_BANDWIDTH = 195312.5f;
#endif
constexpr int NR_TAPS = 16;
constexpr int NR_BASELINES = NR_INPUTS * (NR_INPUTS + 1) / 2;
constexpr int NR_SAMPLES_PER_MINOR_LOOP = 64;
constexpr int REAL = 0;
constexpr int IMAG = 1;
constexpr int COMPLEX = 2;

constexpr int ALIGN(int N, int A)
{ return ((N+A-1)/A)*A; }

std::ostream& cout = std::cout;

std::ostream& cerr = std::cerr;

std::ostream& clog = std::clog;


static inline double rdtsc()
{
  unsigned low, high;

  __asm__ __volatile__ ("rdtsc" : "=a" (low), "=d" (high));
  return static_cast<double>(((unsigned long long) high << 32) | low);
}


typedef int8_t InputDataType[NR_INPUTS][COMPLEX][NR_SAMPLES_PER_CHANNEL + NR_TAPS - 1][NR_CHANNELS] __attribute__((aligned(16)));
typedef float FilteredDataType[ALIGN(NR_INPUTS, VECTOR_SIZE)][NR_SAMPLES_PER_CHANNEL][COMPLEX][NR_CHANNELS] __attribute__((aligned(64)));
typedef float FilterWeightsType[NR_TAPS][NR_CHANNELS] __attribute__((aligned(64)));
typedef float BandPassCorrectionWeights[NR_CHANNELS] __attribute__((aligned(64)));
typedef double DelaysType[NR_INPUTS];
typedef float CorrectedDataType[NR_CHANNELS][ALIGN(NR_INPUTS, VECTOR_SIZE) / VECTOR_SIZE][NR_SAMPLES_PER_CHANNEL][COMPLEX][VECTOR_SIZE] __attribute__((aligned(64)));
typedef float VisibilitiesType[NR_CHANNELS][COMPLEX][NR_BASELINES];


static InputDataType inputData;
#if defined CORRECTNESS_TEST
static FilteredDataType filteredData;
#endif
static FilterWeightsType filterWeights;
static BandPassCorrectionWeights bandPassCorrectionWeights;
static DelaysType delaysAtBegin, delaysAfterEnd;
static CorrectedDataType correctedData;
static VisibilitiesType visibilities; // this is really too much, but avoids a potential segfault on as (masked!!!) vpackstorehps
static uint64_t totalNrOperations;


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
#if 1
#pragma omp for schedule(dynamic)
    for (unsigned input = 0; input < NR_INPUTS; input ++) {
      float history[COMPLEX][NR_TAPS][NR_CHANNELS] __attribute__((aligned(sizeof(float[VECTOR_SIZE]))));

      for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++)
	for (unsigned time = 0; time < NR_TAPS - 1; time ++)
	  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
	    history[real_imag][time][channel] = inputData[input][real_imag][time][channel];

      for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++) {
	  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
	    history[real_imag][(time - 1) % NR_TAPS][channel] = inputData[input][real_imag][time + NR_TAPS - 1][channel];

	    float sum = 0;

	    for (unsigned tap = 0; tap < NR_TAPS; tap ++)
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


static void copyInputData()
{
  //int8_t *inputDataPtr = &inputData[0][0][0][0];

#if !defined CORRECTNESS_TEST
  double start_time = omp_get_wtime();
#endif

#pragma omp target update to(inputData)

#if defined DELAY_COMPENSATION
#pragma omp target update to(delaysAtBegin, delaysAfterEnd)
#endif

#if !defined CORRECTNESS_TEST
  double copy_time = omp_get_wtime() - start_time;

#pragma omp critical (cout)
  cout << "input data: time = " << copy_time << "s (total), " << "BW = " << sizeof(InputDataType) / copy_time / 1e9 << " GB/s" << std::endl;
#endif
}


static void FIR_filter(unsigned iteration)
{
#pragma omp target map(to:iteration)
  filter(filteredData, inputData, filterWeights, iteration);
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
  setInputTestPattern(inputData);
  setFilterWeightsTestPattern(filterWeights);
#pragma omp target update to(filterWeights)

  copyInputData();
  FIR_filter(0);

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

	for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
            for (int i = 0; i < 8; i++) {
                float sample_r = correctedData[channel][inputMajor][time][REAL][i];
                float sample_i = correctedData[channel][inputMajor][time][IMAG][i];

                float tmp = sample_r * v_if[i];
                sample_r = (sample_r * v_rf[i]) - (sample_i * v_if[i]);
                sample_i = (sample_i * v_rf[i]) + tmp;

                tmp = v_rf[i] * dv_if[i];
                v_rf[i] = (v_rf[i] * dv_rf[i]) - (v_if[i] * dv_if[i]);
                v_if[i] = (v_if[i] * dv_rf[i]) + tmp;

                correctedData[channel][inputMajor][time][REAL][i] = sample_r;
                correctedData[channel][inputMajor][time][IMAG][i] = sample_i;
            }
	}
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
  setDelaysTestPattern(delaysAtBegin, delaysAfterEnd);
#pragma omp target update to(delaysAtBegin, delaysAfterEnd)
#pragma omp target
  applyDelays(correctedData, delaysAtBegin, delaysAfterEnd, 60e6, 0);
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
  setFusedTestPattern(inputData, filterWeights, bandPassCorrectionWeights, delaysAtBegin, delaysAfterEnd);
  double FIRfilterTime, FFTtime, trsTime;

#pragma omp target update to(inputData, filterWeights, bandPassCorrectionWeights, delaysAtBegin, delaysAfterEnd)
#pragma omp target
  fused(
    correctedData, inputData, filterWeights,
#if defined BANDPASS_CORRECTION
    bandPassCorrectionWeights,
#endif
#if defined DELAY_COMPENSATION
    delaysAtBegin, delaysAfterEnd, 60e6,
#endif
    0, FIRfilterTime, FFTtime, trsTime
  );
#pragma omp target update from(correctedData)

  checkFusedTestPattern(correctedData);
}
#endif



////// correlator

static void correlate(VisibilitiesType visibilities, const CorrectedDataType correctedData, unsigned)
{
#pragma omp parallel
  {
#pragma omp for collapse(2) schedule(dynamic,16)
    for (int channel = 0; channel < NR_CHANNELS; channel ++) {
      for (int statX = 0; statX < NR_INPUTS; statX ++) {
	for (int statY = 0; statY <= statX; statY ++) {
	  float sum_real = 0, sum_imag = 0;

	  for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
	    float sample_X_real = correctedData[channel][statX / VECTOR_SIZE][time][REAL][statX % VECTOR_SIZE];
	    float sample_X_imag = correctedData[channel][statX / VECTOR_SIZE][time][IMAG][statX % VECTOR_SIZE];
	    float sample_Y_real = correctedData[channel][statY / VECTOR_SIZE][time][REAL][statY % VECTOR_SIZE];
	    float sample_Y_imag = correctedData[channel][statY / VECTOR_SIZE][time][IMAG][statY % VECTOR_SIZE];

	    sum_real += sample_Y_real * sample_X_real;
	    sum_imag += sample_Y_imag * sample_X_real;
	    sum_real += sample_Y_imag * sample_X_imag;
	    sum_imag = (sample_Y_real * sample_X_imag) - sum_imag;
	  }

	  int baseline = statX * (statX + 1) / 2 + statY;
	  visibilities[channel][REAL][baseline] = sum_real;
	  visibilities[channel][IMAG][baseline] = sum_imag;
	}
      }
    }
  }
}


#if defined CORRECTNESS_TEST
static void setCorrelatorTestPattern(CorrectedDataType correctedData)
{
  memset(correctedData, 0, sizeof correctedData);

  if constexpr (NR_CHANNELS > 5 && NR_SAMPLES_PER_CHANNEL > 99 && NR_INPUTS > 19) {
    correctedData[5][ 0 / VECTOR_SIZE][99][REAL][ 0 % VECTOR_SIZE] = 3;
    correctedData[5][ 0 / VECTOR_SIZE][99][IMAG][ 0 % VECTOR_SIZE] = 4;
    correctedData[5][18 / VECTOR_SIZE][99][REAL][18 % VECTOR_SIZE] = 5;
    correctedData[5][18 / VECTOR_SIZE][99][IMAG][18 % VECTOR_SIZE] = 6;
  }
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
  correlate(visibilities, correctedData, 0);
#pragma omp target update from(visibilities)

  checkCorrelatorTestPattern(visibilities);
}
#endif


static void report(const char *msg, uint64_t nrOperations, uint64_t nrBytes, const double &startState, const double &stopState, double weight = 1)
{
#if defined CORRECTNESS_TEST
    (void) msg;
    (void) nrOperations;
    (void) nrBytes;
    (void) startState;
    (void) stopState;
    (void) weight;
#else
  double runTime = (stopState - startState) * weight;

  cout << msg << ": " << runTime << " s, "
	    << nrOperations * 1e-12 / runTime << " TFLOPS, "
	    << nrBytes * 1e-9 / runTime << " GB/s"
	    << std::endl;
#endif
}


static void pipeline(float subbandFrequency, unsigned iteration)
{
  double powerStates[8];
#if defined USE_FUSED_FILTER
  double FIRfilterTime, FFTtime, trsTime;
#endif

#pragma omp critical (XeonPhi)
  {
    powerStates[1] = omp_get_wtime();

#if !defined USE_FUSED_FILTER
#pragma omp target map(to:iteration)
    filter(filteredData, inputData, filterWeights, iteration);

    powerStates[2] = omp_get_wtime();

#pragma omp target
    FFT(filteredData, iteration);

    powerStates[3] = omp_get_wtime();

#if defined BANDPASS_CORRECTION
#pragma omp target map(to:iteration)
    transpose(correctedData, filteredData, bandPassCorrectionWeights, iteration);
#else
#pragma omp target map(to:iteration)
    transpose(correctedData, filteredData, iteration);
#endif

    powerStates[4] = omp_get_wtime();

#if defined DELAY_COMPENSATION
#pragma omp target map(to:subbandFrequency, iteration)
    applyDelays(correctedData, delaysAtBegin, delaysAfterEnd, subbandFrequency, iteration);
#endif

#else
#pragma omp target map(to:subbandFrequency, iteration) map(from:FIRfilterTime, FFTtime, trsTime)
    fused(correctedData, inputData, filterWeights,
#if defined BANDPASS_CORRECTION
    bandPassCorrectionWeights,
#endif
#if defined DELAY_COMPENSATION
    delaysAtBegin, delaysAfterEnd, subbandFrequency,
#endif
    iteration, FIRfilterTime, FFTtime, trsTime);
#endif

    powerStates[5] = omp_get_wtime();

#pragma omp target map(to:iteration)
    correlate(visibilities, correctedData, iteration);

    powerStates[6] = omp_get_wtime();
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

  double startState;
#if !defined CORRECTNESS_TEST
  double stopState;
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

  {
    setInputTestPattern(inputData);
#if defined DELAY_COMPENSATION
    setDelaysTestPattern(delaysAtBegin, delaysAfterEnd);
#endif

    setFilterWeightsTestPattern(filterWeights);

    for (unsigned i = 0; i < 100 && (i < 2 || (omp_get_wtime() - startState) < 20); i ++) {
      if (i == 1)
      {
#pragma omp barrier
#pragma omp single
	startState = omp_get_wtime();
      }

      pipeline(60e6, i);
    }
    cout << std::endl;
    checkCorrelatorTestPattern(visibilities);

  }

#if !defined CORRECTNESS_TEST
  stopState = omp_get_wtime();

  cout << "total: " << stopState - startState << " s"
	       ", " << totalNrOperations / (stopState - startState) * 1e-12 << " TFLOPS"
	       << std::endl;
#endif

#pragma omp target
  fftDestroy();
  return 0;
}
