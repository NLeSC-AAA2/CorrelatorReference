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

#ifdef USE_FUSED_FILTER
#undef USE_FUSED_FILTER
#define USE_FUSED_FILTER true
#else
#define USE_FUSED_FILTER false
#endif
#ifdef DELAY_COMPENSATION
#undef DELAY_COMPENSATION
#define DELAY_COMPENSATION true
#else
#define DELAY_COMPENSATION false
#endif
#ifdef BANDPASS_CORRECTION
#undef BANDPASS_CORRECTION
#define BANDPASS_CORRECTION true
#else
#define BANDPASS_CORRECTION false
#endif

constexpr int NR_INPUTS = 2 * 576;
constexpr int VECTOR_SIZE = 8;

constexpr int NR_CHANNELS = 64;
constexpr int NR_SAMPLES_PER_CHANNEL = 3072;
constexpr uint64_t NR_SAMPLES = NR_INPUTS * NR_SAMPLES_PER_CHANNEL * NR_CHANNELS;
constexpr float SUBBAND_BANDWIDTH = 195312.5f;
constexpr int NR_TAPS = 16;
constexpr int NR_BASELINES = NR_INPUTS * (NR_INPUTS + 1) / 2;
constexpr int NR_SAMPLES_PER_MINOR_LOOP = 64;
constexpr int REAL = 0;
constexpr int IMAG = 1;
constexpr int COMPLEX = 2;

using std::cout, std::cerr;

constexpr int ALIGN(int N, int A)
{ return ((N+A-1)/A)*A; }

static inline double
rdtsc()
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


static bool correctness_test = true;
static bool use_fused_filter = USE_FUSED_FILTER;
static bool delay_compensation = DELAY_COMPENSATION;
static bool bandpass_correction = BANDPASS_CORRECTION;

static InputDataType inputData;
static FilteredDataType filteredData;
static FilterWeightsType filterWeights;
static BandPassCorrectionWeights bandPassCorrectionWeights;
static DelaysType delaysAtBegin, delaysAfterEnd;
static CorrectedDataType correctedData;
static VisibilitiesType visibilities;
static uint64_t totalNrOperations;


////// FIR filter

static void
setInputTestPattern(InputDataType inputData)
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


static void
setFilterWeightsTestPattern(FilterWeightsType filterWeights)
{
  memset(filterWeights, 0, sizeof(FilterWeightsType));

  if (NR_TAPS > 4 && NR_CHANNELS > 12) {
    filterWeights[15][12] = 2;
    filterWeights[14][12] = 3;
  }
}


static void
filter
( FilteredDataType filteredData
, const InputDataType inputData
, const FilterWeightsType filterWeights
, unsigned
)
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


static void
copyInputData()
{
    if (!correctness_test) {
        double start_time = omp_get_wtime();
        double copy_time = omp_get_wtime() - start_time;

#pragma omp critical (cout)
        cout << "input data: time = " << copy_time << "s (total), " << "BW = "
             << sizeof(InputDataType) / copy_time / 1e9 << " GB/s" << std::endl;
    }
}


static void
FIR_filter(unsigned iteration)
{
    filter(filteredData, inputData, filterWeights, iteration);
}


static void
checkFIR_FilterTestPattern(const FilteredDataType filteredData)
{
    for (unsigned input = 0; input < NR_INPUTS; input ++)
        for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
                if (filteredData[input][time][REAL][channel] != 0 || filteredData[input][time][IMAG][channel] != 0) {
                    cout << "input = " << input << ", time = " << time
                         << ", channel = " << channel << ", sample = ("
                         << filteredData[input][time][REAL][channel] << ','
                         << filteredData[input][time][IMAG][channel] << ')'
                         << std::endl;
                }
}


static void
testFIR_Filter()
{
    setInputTestPattern(inputData);
    setFilterWeightsTestPattern(filterWeights);

    copyInputData();
    FIR_filter(0);

    checkFIR_FilterTestPattern(filteredData);
}


////// FFT

DFTI_DESCRIPTOR_HANDLE handle;


static void
fftInit()
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


static void
fftDestroy()
{
    MKL_LONG error;

    error = DftiFreeDescriptor(&handle);

    if (error != DFTI_NO_ERROR) {
        cerr << "DftiFreeDescriptor failed" << std::endl;
        exit(1);
    }
}


static void
FFT(FilteredDataType filteredData, unsigned)
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


////// transpose

static void
transpose
( CorrectedDataType correctedData
, const FilteredDataType filteredData
, const BandPassCorrectionWeights bandPassCorrectionWeights
, unsigned
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
                                if (bandpass_correction) {
                                    correctedData[channel][inputMajor / VECTOR_SIZE][time][realImag][inputMinor] = bandPassCorrectionWeights[channel] * filteredData[input][time][realImag][channel];
                                } else {
                                    correctedData[channel][inputMajor / VECTOR_SIZE][time][realImag][inputMinor] = filteredData[input][time][realImag][channel];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


static void
applyDelays
( CorrectedDataType correctedData
, const DelaysType delaysAtBegin
, const DelaysType delaysAfterEnd
, float subbandFrequency
, unsigned
)
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
                        float myPhiDelta = static_cast<float>(deltaPhi * channelFrequency);
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


static void
setBandPassTestPattern(BandPassCorrectionWeights bandPassCorrectionWeights)
{
    for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
        bandPassCorrectionWeights[channel] = 1;

    if (NR_CHANNELS > 5)
        bandPassCorrectionWeights[5] = 2;
}


static void
setDelaysTestPattern(DelaysType delaysAtBegin, DelaysType delaysAfterEnd)
{
    memset(delaysAtBegin, 0, sizeof(DelaysType));
    memset(delaysAfterEnd, 0, sizeof(DelaysType));

    if (NR_INPUTS > 22)
        delaysAfterEnd[22] = 1e-6;
}


static void
setTransposeTestPattern(FilteredDataType filteredData)
{
    if (NR_INPUTS > 22 && NR_SAMPLES_PER_CHANNEL > 99 && NR_CHANNELS > 5) {
        filteredData[22][99][REAL][5] = 2;
        filteredData[22][99][IMAG][5] = 3;
    }
}


static void
checkTransposeTestPattern(const CorrectedDataType correctedData)
{
    for (int channel = 0; channel < NR_CHANNELS; channel ++)
        for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
            for (int input = 0; input < NR_INPUTS; input ++)
                if (correctedData[channel][input / VECTOR_SIZE][time][REAL][input % VECTOR_SIZE] != 0 || correctedData[channel][input / VECTOR_SIZE][time][IMAG][input % VECTOR_SIZE] != 0) {
                    cout << "channel = " << channel << ", time = " << time
                         << ", input = " << input << ", value = ("
                         << correctedData[channel][input / VECTOR_SIZE][time][REAL][input % VECTOR_SIZE]
                         << ',' << correctedData[channel][input / VECTOR_SIZE][time][IMAG][input % VECTOR_SIZE]
                         << ')' << std::endl;
                }
}


static void
testTranspose()
{
    setTransposeTestPattern(filteredData);

    if (bandpass_correction) {
        setBandPassTestPattern(bandPassCorrectionWeights);
    }

    transpose(correctedData, filteredData, bandPassCorrectionWeights, 0);

    if (delay_compensation) {
        setDelaysTestPattern(delaysAtBegin, delaysAfterEnd);
        applyDelays(correctedData, delaysAtBegin, delaysAfterEnd, 60e6, 0);
    }

    checkTransposeTestPattern(correctedData);
}


//////

template <typename T>
static inline void
cmul(T &c_r, T &c_i, T a_r, T a_i, T b_r, T b_i)
{
    c_r = a_r * b_r - a_i * b_i;
    c_i = a_r * b_i + a_i * b_r;
}


static void
fused_FIRfilterInit
( const InputDataType inputData
, float history[COMPLEX][NR_TAPS][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/
, unsigned input
, unsigned
, double &FIRfilterTime
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


static void
fused_FIRfilter
( const InputDataType inputData
, float history[COMPLEX][NR_TAPS][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/
, float filteredData[NR_SAMPLES_PER_MINOR_LOOP][COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/
, unsigned input
, unsigned majorTime
, unsigned
, double &FIRfilterTime
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


static void
fused_FFT
( float filteredData[NR_SAMPLES_PER_MINOR_LOOP][COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/
, unsigned
, double &FFTtime
)
{
    FFTtime -= rdtsc();

    //  for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++)
    //  DftiComputeForward(handle, filteredData[minorTime][REAL], filteredData[minorTime][IMAG]);
    // Do batch FFT instead of for-loop
    DftiComputeForward(handle, filteredData[0][REAL], filteredData[0][IMAG]);

    FFTtime += rdtsc();
}


static void
fused_TransposeInit
( float v[COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/
, float dv[COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/
, const BandPassCorrectionWeights bandPassCorrectionWeights
, const DelaysType delaysAtBegin
, const DelaysType delaysAfterEnd
, float subbandFrequency
, unsigned input
, unsigned
, double &trsTime
)
{
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

        if (bandpass_correction) {
            v[REAL][channel] *= bandPassCorrectionWeights[channel];
            v[IMAG][channel] *= bandPassCorrectionWeights[channel];
        }
    }

    trsTime += rdtsc();
}


static void
fused_Transpose
( CorrectedDataType correctedData
, float filteredData[NR_SAMPLES_PER_MINOR_LOOP][COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/
, float v[COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/
, float dv[COMPLEX][NR_CHANNELS] /*__attribute__((aligned(sizeof(float[VECTOR_SIZE]))))*/
, unsigned input
, unsigned majorTime
, unsigned
, double &trsTime
)
{
    trsTime -= rdtsc();

    if (bandpass_correction && !delay_compensation) {
        // BandPass correction, if not doing delay compensation

        for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++) {
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
                for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++) {
                    filteredData[minorTime][real_imag][channel] *= bandPassCorrectionWeights[channel];
                }
            }
        }
    }

    // Delay compensation & transpose

    for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
        for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++) {
            float sample_r = filteredData[minorTime][REAL][channel];
            float sample_i = filteredData[minorTime][IMAG][channel];

            if (delay_compensation) {
                cmul(sample_r, sample_i, sample_r, sample_i, v[REAL][channel], v[IMAG][channel]);
                cmul(v[REAL][channel], v[IMAG][channel], v[REAL][channel], v[IMAG][channel], dv[REAL][channel], dv[IMAG][channel]);
            }

            correctedData[channel][input / VECTOR_SIZE][majorTime + minorTime][REAL][input % VECTOR_SIZE] = sample_r;
            correctedData[channel][input / VECTOR_SIZE][majorTime + minorTime][IMAG][input % VECTOR_SIZE] = sample_i;
        }
    }

    trsTime += rdtsc();
}


static void
fused
( CorrectedDataType correctedData
, const InputDataType inputData
, const FilterWeightsType
, const BandPassCorrectionWeights bandPassCorrectionWeights
, const DelaysType delaysAtBegin
, const DelaysType delaysAfterEnd
, float subbandFrequency
, unsigned iteration
, double &FIRfilterTimeRef
, double &FFTtimeRef
, double &trsTimeRef
)
{
    double FIRfilterTime = 0, FFTtime = 0, trsTime = 0;

#pragma omp parallel reduction(+: FIRfilterTime, FFTtime, trsTime)
    {
#pragma omp for schedule(dynamic)
        for (unsigned input = 0; input < NR_INPUTS; input ++) {
            float history[COMPLEX][NR_TAPS][NR_CHANNELS] __attribute__((aligned(sizeof(float[VECTOR_SIZE]))));
            float filteredData[NR_SAMPLES_PER_MINOR_LOOP][COMPLEX][NR_CHANNELS] __attribute__((aligned(sizeof(float[VECTOR_SIZE]))));

            float v[COMPLEX][NR_CHANNELS] __attribute__((aligned(sizeof(float[VECTOR_SIZE]))));
            float dv[COMPLEX][NR_CHANNELS] __attribute__((aligned(sizeof(float[VECTOR_SIZE]))));
            if (delay_compensation) {
                fused_TransposeInit(v, dv,
                        bandPassCorrectionWeights,
                        delaysAtBegin, delaysAfterEnd, subbandFrequency, input, iteration, trsTime);
            }

            fused_FIRfilterInit(inputData, history, input, iteration, FIRfilterTime);

            for (unsigned majorTime = 0; majorTime < NR_SAMPLES_PER_CHANNEL; majorTime += NR_SAMPLES_PER_MINOR_LOOP) {
                fused_FIRfilter(inputData, history, filteredData, input, majorTime, iteration, FIRfilterTime);
                fused_FFT(filteredData, iteration, FFTtime);
                fused_Transpose(correctedData, filteredData,
                        v, dv,
                        input, majorTime, iteration, trsTime);
            }
        }
    }

    if (iteration > 0) {
        FIRfilterTimeRef = FIRfilterTime;
        FFTtimeRef = FFTtime;
        trsTimeRef = trsTime;
    }
}


static void
setFusedTestPattern
( InputDataType inputData
, FilterWeightsType filterWeights
, BandPassCorrectionWeights bandPassCorrectionWeights
, DelaysType delaysAtBegin
, DelaysType delaysAfterEnd
)
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


static void
checkFusedTestPattern(const CorrectedDataType correctedData)
{
    for (unsigned input = 0; input < NR_INPUTS; input ++)
        for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
                if (correctedData[channel][input / VECTOR_SIZE][time][REAL][input % VECTOR_SIZE] != 0 || correctedData[channel][input / VECTOR_SIZE][time][IMAG][input % VECTOR_SIZE] != 0) {
                    cout << "input = " << input << ", time = " << time
                         << ", channel = " << channel << ": ("
                         << correctedData[channel][input / VECTOR_SIZE][time][REAL][input % VECTOR_SIZE]
                         << ", " << correctedData[channel][input / VECTOR_SIZE][time][IMAG][input % VECTOR_SIZE]
                         << ')' << std::endl;
                }
}


static void
testFused()
{
    setFusedTestPattern(inputData, filterWeights, bandPassCorrectionWeights, delaysAtBegin, delaysAfterEnd);
    double FIRfilterTime, FFTtime, trsTime;

    fused(
            correctedData, inputData, filterWeights,
            bandPassCorrectionWeights,
            delaysAtBegin, delaysAfterEnd, 60e6,
            0, FIRfilterTime, FFTtime, trsTime
         );

    checkFusedTestPattern(correctedData);
}



////// correlator

static void
correlate
( VisibilitiesType visibilities
, const CorrectedDataType correctedData
, unsigned
)
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


static void
setCorrelatorTestPattern(CorrectedDataType correctedData)
{
    if constexpr (NR_CHANNELS > 5 && NR_SAMPLES_PER_CHANNEL > 99 && NR_INPUTS > 19) {
        correctedData[5][ 0 / VECTOR_SIZE][99][REAL][ 0 % VECTOR_SIZE] = 3;
        correctedData[5][ 0 / VECTOR_SIZE][99][IMAG][ 0 % VECTOR_SIZE] = 4;
        correctedData[5][18 / VECTOR_SIZE][99][REAL][18 % VECTOR_SIZE] = 5;
        correctedData[5][18 / VECTOR_SIZE][99][IMAG][18 % VECTOR_SIZE] = 6;
    }
}


static void
checkCorrelatorTestPattern(const VisibilitiesType visibilities)
{
    for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
        for (unsigned baseline = 0; baseline < NR_BASELINES; baseline ++)
            if (visibilities[channel][REAL][baseline] != 0 || visibilities[channel][IMAG][baseline] != 0) {
                cout << "channel = " << channel << ", baseline = " << baseline
                     << ", visibility = (" << visibilities[channel][REAL][baseline]
                     << ',' << visibilities[channel][IMAG][baseline] << ')'
                     << std::endl;
            }
}


static void
testCorrelator()
{
    setCorrelatorTestPattern(correctedData);

    correlate(visibilities, correctedData, 0);

    checkCorrelatorTestPattern(visibilities);
}


static void
report
( const char *msg
, uint64_t nrOperations
, uint64_t nrBytes
, const double &startState
, const double &stopState
, double weight = 1
)
{
    if (!correctness_test) {
        double runTime = (stopState - startState) * weight;

        cout << msg << ": " << runTime << " s, "
             << static_cast<double>(nrOperations) * 1e-12 / runTime
             << " TFLOPS, " << static_cast<double>(nrBytes) * 1e-9 / runTime
             << " GB/s" << std::endl;
    }
}


static void
pipeline(float subbandFrequency, unsigned iteration)
{
    double powerStates[8];
    double fusedTime, FIRfilterTime, FFTtime, trsTime;
    fusedTime = FIRfilterTime = FFTtime = trsTime = 0;

#pragma omp critical (XeonPhi)
    {
        powerStates[1] = omp_get_wtime();

        if (!use_fused_filter) {
            filter(filteredData, inputData, filterWeights, iteration);

            powerStates[2] = omp_get_wtime();

            FFT(filteredData, iteration);

            powerStates[3] = omp_get_wtime();

            transpose(correctedData, filteredData, bandPassCorrectionWeights, iteration);

            powerStates[4] = omp_get_wtime();

            if (delay_compensation) {
                applyDelays(correctedData, delaysAtBegin, delaysAfterEnd, subbandFrequency, iteration);
            }
        } else {
            fused(correctedData, inputData, filterWeights,
                    bandPassCorrectionWeights,
                    delaysAtBegin, delaysAfterEnd, subbandFrequency,
                    iteration, FIRfilterTime, FFTtime, trsTime);
        }

        powerStates[5] = omp_get_wtime();

        correlate(visibilities, correctedData, iteration);

        powerStates[6] = omp_get_wtime();
    }

    if (iteration > 0) // do not count first iteration
#pragma omp critical (cout)
    {
        uint64_t nrFIRfilterOperations = NR_SAMPLES * COMPLEX * NR_TAPS * 2;
        uint64_t nrFFToperations       = static_cast<uint64_t>(NR_SAMPLES * 5 * log2(NR_CHANNELS));

        uint64_t nrDelayAndBandPassOperations = 0;
        if (delay_compensation) {
            nrDelayAndBandPassOperations = NR_SAMPLES * 2 * 6;
        } else if (bandpass_correction) {
            nrDelayAndBandPassOperations = NR_SAMPLES * COMPLEX;
        }

        uint64_t nrCorrelatorOperations = (uint64_t) NR_INPUTS * NR_INPUTS / 2 * 8ULL * NR_CHANNELS * NR_SAMPLES_PER_CHANNEL;
        uint64_t nrFusedOperations = nrFIRfilterOperations + nrFFToperations + nrDelayAndBandPassOperations;

        totalNrOperations += nrFusedOperations + nrCorrelatorOperations; // is already atomic

        if (!use_fused_filter) {
            report("FIR", nrFIRfilterOperations, sizeof(InputDataType) + sizeof(FilteredDataType), powerStates[1], powerStates[2]);
            report("FFT", nrFFToperations, 2 * sizeof(FilteredDataType), powerStates[2], powerStates[3]);
            if (bandpass_correction) {
                report("trs", nrDelayAndBandPassOperations, sizeof(FilteredDataType) + sizeof(CorrectedDataType), powerStates[3], powerStates[4]);
            } else {
                report("trs", 0, sizeof(FilteredDataType) + sizeof(CorrectedDataType), powerStates[3], powerStates[4]);
            }
            report("del", NR_SAMPLES * 2 * 6, 2 * sizeof(CorrectedDataType), powerStates[4], powerStates[5]);
        } else {
            report("FIR", nrFIRfilterOperations, sizeof(InputDataType), powerStates[1], powerStates[5], (double) FIRfilterTime / fusedTime);
            report("FFT", nrFFToperations, 0, powerStates[1], powerStates[5], (double) FFTtime / fusedTime);
            report("trs", nrDelayAndBandPassOperations, sizeof(FilteredDataType), powerStates[1], powerStates[5], (double) trsTime / fusedTime);
            report("fused", nrFusedOperations, sizeof(InputDataType) + sizeof(CorrectedDataType), powerStates[1], powerStates[5]);
        }

        report("cor", nrCorrelatorOperations, sizeof(CorrectedDataType) + sizeof(VisibilitiesType), powerStates[5], powerStates[6]);
    }
}


int main(int, char **)
{
    static_assert(NR_CHANNELS % 16 == 0);
    static_assert(NR_SAMPLES_PER_CHANNEL % NR_SAMPLES_PER_MINOR_LOOP == 0);

    double startState;
    double stopState;

    fftInit();

    if (correctness_test) {
        testFused();
        testFIR_Filter();
        testTranspose();
        testCorrelator();
    }

    omp_set_nested(1);

    setBandPassTestPattern(bandPassCorrectionWeights);

    {
        setInputTestPattern(inputData);
        setDelaysTestPattern(delaysAtBegin, delaysAfterEnd);

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
    }
    cout << std::endl;
    checkCorrelatorTestPattern(visibilities);

    if (!correctness_test) {
        stopState = omp_get_wtime();

        cout << "total: " << stopState - startState << " s" << ", "
             << static_cast<double>(totalNrOperations) / (stopState - startState) * 1e-12
             << " TFLOPS" << std::endl;
    }

    fftDestroy();
    return 0;
}
