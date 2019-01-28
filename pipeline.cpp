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

#include <boost/multi_array.hpp>

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

constexpr int NR_CHANNELS = 64;
constexpr int NR_SAMPLES_PER_CHANNEL = 3072;
constexpr double SUBBAND_BANDWIDTH = 195312.5;
constexpr int NR_TAPS = 16;
constexpr int NR_BASELINES = NR_INPUTS * (NR_INPUTS + 1) / 2;
constexpr int NR_SAMPLES_PER_MINOR_LOOP = 64;
constexpr int REAL = 0;
constexpr int IMAG = 1;
constexpr int COMPLEX = 2;

using std::cout, std::cerr;

typedef boost::multi_array<int8_t, 4> InputDataType;
static const auto InputDataDims = boost::extents[NR_INPUTS][COMPLEX][NR_SAMPLES_PER_CHANNEL + NR_TAPS - 1][NR_CHANNELS];
typedef boost::multi_array<std::complex<float>, 3> FilteredDataType;
static const auto FilteredDataDims = boost::extents[NR_INPUTS][NR_SAMPLES_PER_CHANNEL][NR_CHANNELS];
typedef boost::multi_array<float, 2> FilterWeightsType;
static const auto FilterWeightsDims = boost::extents[NR_TAPS][NR_CHANNELS];
typedef boost::multi_array<float, 1> BandPassCorrectionWeights;
static const auto BandPassCorrectionWeightsDims = boost::extents[NR_CHANNELS];
typedef boost::multi_array<double, 1> DelaysType;
static const auto DelaysDims = boost::extents[NR_INPUTS];
typedef boost::multi_array<std::complex<float>, 3> CorrectedDataType;
static const auto CorrectedDataDims = boost::extents[NR_CHANNELS][NR_INPUTS][NR_SAMPLES_PER_CHANNEL];
typedef boost::multi_array<float, 3> VisibilitiesType;
static const auto VisibilitiesDims = boost::extents[NR_CHANNELS][COMPLEX][NR_BASELINES];

typedef boost::multi_array<float, 3> HistoryType;
static const auto HistoryDims = boost::extents[COMPLEX][NR_TAPS][NR_CHANNELS];
typedef boost::multi_array<std::complex<float>, 2> FusedFilterType;
static const auto FusedFilterDims = boost::extents[NR_SAMPLES_PER_MINOR_LOOP][NR_CHANNELS];
typedef boost::multi_array<float, 2> ComplexChannelType;
static const auto ComplexChannelDims = boost::extents[COMPLEX][NR_CHANNELS];

static bool use_fused_filter = USE_FUSED_FILTER;
static bool delay_compensation = DELAY_COMPENSATION;
static bool bandpass_correction = BANDPASS_CORRECTION;

////// FIR filter

static InputDataType
inputTestPattern(bool isFused = false)
{
    InputDataType result(InputDataDims);

    if (isFused) {
        std::fill_n(result.data(), result.num_elements(), 0);
        if (NR_INPUTS > 6 && NR_SAMPLES_PER_CHANNEL > 27 && NR_CHANNELS > 12) {
            result[6][REAL][27 + NR_TAPS - 1][12] = 2;
            result[6][IMAG][27 + NR_TAPS - 1][12] = 3;
        }
    } else {
        signed char count = 64;
        for (unsigned input = 0; input < NR_INPUTS; input ++)
            for (unsigned ri = 0; ri < COMPLEX; ri ++)
                for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL + NR_TAPS - 1; time ++)
                    for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
                        result[input][ri][time][channel] = count ++;

        if (NR_INPUTS > 9 && NR_SAMPLES_PER_CHANNEL > 99 && NR_CHANNELS > 12) {
            result[9][REAL][98 + NR_TAPS - 1][12] = 4;
            result[9][REAL][99 + NR_TAPS - 1][12] = 5;
        }
    }

    return result;
}

static FilterWeightsType
filterWeightsTestPattern(bool isFused = false)
{
    FilterWeightsType result(FilterWeightsDims);
    std::fill_n(result.data(), result.num_elements(), 0.0f);

    if (isFused && NR_TAPS > 11 && NR_CHANNELS > 12) {
        result[15][12] = 2;
    } else if (NR_TAPS > 4 && NR_CHANNELS > 12) {
        result[15][12] = 2;
        result[14][12] = 3;
    }

    return result;
}


static FilteredDataType
FIR_filter
( const InputDataType& inputData
, const FilterWeightsType& filterWeights
)
{
    FilteredDataType filteredData(FilteredDataDims);
#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (unsigned input = 0; input < NR_INPUTS; input ++) {
            HistoryType history(HistoryDims);

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

                        if (real_imag == REAL) {
                            filteredData[input][time][channel] = {sum, filteredData[input][time][channel].imag()};
                        } else {
                            filteredData[input][time][channel] = {filteredData[input][time][channel].real(), sum};
                        }
                    }
                }
            }
        }
    }

    return filteredData;
}


static void
checkFIR_FilterTestPattern(const FilteredDataType& filteredData)
{
    for (unsigned input = 0; input < NR_INPUTS; input ++)
        for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
                if (filteredData[input][time][channel].real() != 0.0f || filteredData[input][time][channel].imag() != 0.0f) {
                    cout << "input = " << input << ", time = " << time
                         << ", channel = " << channel << ", sample = ("
                         << filteredData[input][time][channel].real() << ','
                         << filteredData[input][time][channel].imag() << ')'
                         << std::endl;
                }
}


static FilteredDataType
testFIR_Filter()
{
    const auto& filteredData = FIR_filter(inputTestPattern(), filterWeightsTestPattern());

    checkFIR_FilterTestPattern(filteredData);
    return filteredData;
}


////// FFT

static DFTI_DESCRIPTOR_HANDLE handle;


static void
fftInit()
{
    MKL_LONG error;

    error = DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 1, NR_CHANNELS);

    if (error != DFTI_NO_ERROR) {
        cerr << "DftiCreateDescriptor failed" << std::endl;
        exit(1);
    };

    error = DftiSetValue(handle, DFTI_NUMBER_OF_TRANSFORMS, NR_SAMPLES_PER_MINOR_LOOP);

    if (error != DFTI_NO_ERROR) {
        cerr << "DftiSetValue failed" << std::endl;
        exit(1);
    }

    error = DftiSetValue(handle, DFTI_INPUT_DISTANCE, NR_CHANNELS);

    if (error != DFTI_NO_ERROR) {
        cerr << "DftiSetValue failed" << std::endl;
        exit(1);
    }

    error = DftiSetValue(handle, DFTI_OUTPUT_DISTANCE, NR_CHANNELS);

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
FFT(FilteredDataType& filteredData)
{
#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(dynamic)
        for (int input = 0; input < NR_INPUTS; input ++)
            for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time += NR_SAMPLES_PER_MINOR_LOOP)
                DftiComputeForward(handle, filteredData[input][time].origin());
    }
}


////// transpose

static CorrectedDataType
transpose
( const FilteredDataType& filteredData
, const BandPassCorrectionWeights& bandPassCorrectionWeights
)
{
    CorrectedDataType correctedData(CorrectedDataDims);
#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
            for (unsigned input = 0; input < NR_INPUTS; input ++) {
                for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
                    float real = filteredData[input][time][channel].real();
                    float imag = filteredData[input][time][channel].imag();

                    if (bandpass_correction) {
                        real *= bandPassCorrectionWeights[channel];
                        imag *= bandPassCorrectionWeights[channel];

                    }

                    correctedData[channel][input][time] = {real, imag};
                }
            }
        }
    }

    return correctedData;
}


static void
applyDelays
( CorrectedDataType& correctedData
, const DelaysType& delaysAtBegin
, const DelaysType& delaysAfterEnd
, double subbandFrequency
)
{
#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (unsigned input = 0; input < NR_INPUTS; input ++) {
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
                float v_rf;
                float v_if;
                float dv_rf;
                float dv_if;

                double phiBegin = -2.0 * 3.141592653589793 * delaysAtBegin[input];
                double phiEnd   = -2.0 * 3.141592653589793 * delaysAfterEnd[input];
                double deltaPhi = (phiEnd - phiBegin) / NR_SAMPLES_PER_CHANNEL;
                double channelFrequency = subbandFrequency - .5 * SUBBAND_BANDWIDTH + channel * (SUBBAND_BANDWIDTH / NR_CHANNELS);
                float myPhiBegin = static_cast<float>((phiBegin /* + startTime * deltaPhi */) * channelFrequency /* + phaseOffsets[stationPol + major] */);
                float myPhiDelta = static_cast<float>(deltaPhi * channelFrequency);
                sincosf(myPhiBegin, &v_if, &v_rf);
                sincosf(myPhiDelta, &dv_if, &dv_rf);

                for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
                    float sample_r = correctedData[channel][input][time].real();
                    float sample_i = correctedData[channel][input][time].imag();

                    float tmp = sample_r * v_if;
                    sample_r = (sample_r * v_rf) - (sample_i * v_if);
                    sample_i = (sample_i * v_rf) + tmp;

                    tmp = v_rf * dv_if;
                    v_rf = (v_rf * dv_rf) - (v_if * dv_if);
                    v_if = (v_if * dv_rf) + tmp;

                    correctedData[channel][input][time] = {sample_r, sample_i};
                }
            }
        }
    }
}


static BandPassCorrectionWeights
bandPassTestPattern(bool isFused = false)
{
    BandPassCorrectionWeights result(BandPassCorrectionWeightsDims);
    std::fill_n(result.data(), result.num_elements(), 1);

    if (!isFused && NR_CHANNELS > 5)
        result[5] = 2;

    return result;
}


static DelaysType
delaysTestPattern(bool isBegin, bool isFused = false)
{
    DelaysType result(DelaysDims);
    std::fill_n(result.data(), result.num_elements(), 0);

    if (!isBegin && !isFused && NR_INPUTS > 22)
        result[22] = 1e-6;

    return result;
}


static void
checkTransposeTestPattern(const CorrectedDataType& correctedData)
{
    for (int channel = 0; channel < NR_CHANNELS; channel ++)
        for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
            for (int input = 0; input < NR_INPUTS; input ++)
                if (correctedData[channel][input][time].real() != 0.0f || correctedData[channel][input][time].imag() != 0.0f) {
                    cout << "channel = " << channel << ", time = " << time
                         << ", input = " << input << ", value = ("
                         << correctedData[channel][input][time].real()
                         << ',' << correctedData[channel][input][time].imag()
                         << ')' << std::endl;
                }
}


static CorrectedDataType
testTranspose(FilteredDataType& filteredData)
{
    if (NR_INPUTS > 22 && NR_SAMPLES_PER_CHANNEL > 99 && NR_CHANNELS > 5) {
        filteredData[22][99][5] = {2, 3};
    }

    BandPassCorrectionWeights bandPassCorrectionWeights(BandPassCorrectionWeightsDims);
    if (bandpass_correction) {
        bandPassCorrectionWeights = bandPassTestPattern();
    }

    auto correctedData = transpose(filteredData, bandPassCorrectionWeights);

    if (delay_compensation) {
        applyDelays(correctedData, delaysTestPattern(true), delaysTestPattern(false), 60e6);
    }

    checkTransposeTestPattern(correctedData);
    return correctedData;
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
( const InputDataType& inputData
, HistoryType& history
, unsigned input
)
{
    // fill FIR filter history

    for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++)
        for (unsigned time = 0; time < NR_TAPS - 1; time ++)
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
                history[real_imag][time][channel] = inputData[input][real_imag][time][channel];
}


static void
fused_FIRfilter
( const InputDataType& inputData
, const FilterWeightsType& filterWeights
, HistoryType& history
, FusedFilterType& filteredData
, unsigned input
, unsigned majorTime
)
{
    for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++) {
        for (unsigned real_imag = 0; real_imag < COMPLEX; real_imag ++) {
            //#pragma vector aligned // why does specifying this yields wrong results ???
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
                history[real_imag][(minorTime - 1) % NR_TAPS][channel] = inputData[input][real_imag][majorTime + minorTime + NR_TAPS - 1][channel];

                float sum = 0;

                for (unsigned tap = 0; tap < NR_TAPS; tap ++)
                    sum += filterWeights[tap][channel] * history[real_imag][(minorTime + tap) % NR_TAPS][channel];

                if (real_imag == IMAG) {
                    filteredData[minorTime][channel] = {filteredData[minorTime][channel].real(), sum};
                } else {
                    filteredData[minorTime][channel] = {sum, filteredData[minorTime][channel].imag()};
                }
            }
        }
    }
}


static void
fused_FFT(FusedFilterType& filteredData)
{
    // Do batch FFT instead of for-loop
    DftiComputeForward(handle, filteredData[0].origin());
}


static void
fused_TransposeInit
( ComplexChannelType& v
, ComplexChannelType& dv
, const BandPassCorrectionWeights& bandPassCorrectionWeights
, const DelaysType& delaysAtBegin
, const DelaysType& delaysAfterEnd
, double subbandFrequency
, unsigned input
)
{
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
}


static void
fused_Transpose
( CorrectedDataType& correctedData
, const BandPassCorrectionWeights& bandPassCorrectionWeights
, FusedFilterType& filteredData
, ComplexChannelType& v
, ComplexChannelType& dv
, unsigned input
, unsigned majorTime
)
{
    if (bandpass_correction && !delay_compensation) {
        // BandPass correction, if not doing delay compensation

        for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++) {
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
                float real = filteredData[minorTime][channel].real() * bandPassCorrectionWeights[channel];
                float imag = filteredData[minorTime][channel].imag() * bandPassCorrectionWeights[channel];
                filteredData[minorTime][channel] = {real, imag};
            }
        }
    }

    // Delay compensation & transpose

    for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
        for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++) {
            float sample_r = filteredData[minorTime][channel].real();
            float sample_i = filteredData[minorTime][channel].imag();

            if (delay_compensation) {
                cmul(sample_r, sample_i, sample_r, sample_i, v[REAL][channel], v[IMAG][channel]);
                cmul(v[REAL][channel], v[IMAG][channel], v[REAL][channel], v[IMAG][channel], dv[REAL][channel], dv[IMAG][channel]);
            }

            correctedData[channel][input][majorTime + minorTime] = {sample_r, sample_i};
        }
    }
}


static CorrectedDataType
fused
( const InputDataType& inputData
, const FilterWeightsType& filterWeights
, const BandPassCorrectionWeights& bandPassCorrectionWeights
, const DelaysType& delaysAtBegin
, const DelaysType& delaysAfterEnd
, double subbandFrequency
)
{
    CorrectedDataType correctedData(CorrectedDataDims);
#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (unsigned input = 0; input < NR_INPUTS; input ++) {
            HistoryType history(HistoryDims);
            FusedFilterType filteredData(FusedFilterDims);

            ComplexChannelType v(ComplexChannelDims);
            ComplexChannelType dv(ComplexChannelDims);

            if (delay_compensation) {
                fused_TransposeInit(v, dv,
                        bandPassCorrectionWeights,
                        delaysAtBegin, delaysAfterEnd, subbandFrequency, input);
            }

            fused_FIRfilterInit(inputData, history, input);

            for (unsigned majorTime = 0; majorTime < NR_SAMPLES_PER_CHANNEL; majorTime += NR_SAMPLES_PER_MINOR_LOOP) {
                fused_FIRfilter(inputData, filterWeights, history, filteredData, input, majorTime);
                fused_FFT(filteredData);
                fused_Transpose(correctedData, bandPassCorrectionWeights, filteredData,
                        v, dv,
                        input, majorTime);
            }
        }
    }

    return correctedData;
}


static void
checkFusedTestPattern(const CorrectedDataType& correctedData)
{
    for (unsigned input = 0; input < NR_INPUTS; input ++)
        for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
                if (correctedData[channel][input][time].real() != 0.0f || correctedData[channel][input][time].imag() != 0.0f) {
                    cout << "input = " << input << ", time = " << time
                         << ", channel = " << channel << ": ("
                         << correctedData[channel][input][time].real()
                         << ", " << correctedData[channel][input][time].imag()
                         << ')' << std::endl;
                }
}


static void
testFused()
{
    auto correctedData = fused(
            inputTestPattern(true), filterWeightsTestPattern(true),
            bandPassTestPattern(true),
            delaysTestPattern(true, true), delaysTestPattern(false, true), 60e6);

    checkFusedTestPattern(correctedData);
}



////// correlator

static VisibilitiesType
correlate(const CorrectedDataType& correctedData)
{
    VisibilitiesType visibilities(VisibilitiesDims);
#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(dynamic,16)
        for (int channel = 0; channel < NR_CHANNELS; channel ++) {
            for (int statX = 0; statX < NR_INPUTS; statX ++) {
                for (int statY = 0; statY <= statX; statY ++) {
                    float sum_real = 0, sum_imag = 0;

                    for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
                        float sample_X_real = correctedData[channel][statX][time].real();
                        float sample_X_imag = correctedData[channel][statX][time].imag();
                        float sample_Y_real = correctedData[channel][statY][time].real();
                        float sample_Y_imag = correctedData[channel][statY][time].imag();

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

    return visibilities;
}


static void
checkCorrelatorTestPattern(const VisibilitiesType& visibilities)
{
    for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
        for (unsigned baseline = 0; baseline < NR_BASELINES; baseline ++)
            if (visibilities[channel][REAL][baseline] != 0.0f || visibilities[channel][IMAG][baseline] != 0.0f) {
                cout << "channel = " << channel << ", baseline = " << baseline
                     << ", visibility = (" << visibilities[channel][REAL][baseline]
                     << ',' << visibilities[channel][IMAG][baseline] << ')'
                     << std::endl;
            }
}


static void
testCorrelator(CorrectedDataType& correctedData)
{
    if constexpr (NR_CHANNELS > 5 && NR_SAMPLES_PER_CHANNEL > 99 && NR_INPUTS > 19) {
        correctedData[5][ 0][99] = {3,4};
        correctedData[5][18][99] = {5,6};
    }

    auto visibilities = correlate(correctedData);

    checkCorrelatorTestPattern(visibilities);
}


static VisibilitiesType
pipeline(double subbandFrequency)
{
    CorrectedDataType correctedData(CorrectedDataDims);

    const auto& bandPassCorrectionWeights = bandPassTestPattern();
    const auto& delaysAtBegin = delaysTestPattern(true);
    const auto& delaysAfterEnd = delaysTestPattern(false);
    const auto& inputData = inputTestPattern();
    const auto& filterWeights = filterWeightsTestPattern();
    VisibilitiesType result(VisibilitiesDims);

#pragma omp critical (XeonPhi)
    {
        if (!use_fused_filter) {
            auto filteredData = FIR_filter(inputData, filterWeights);
            FFT(filteredData);
            correctedData = transpose(filteredData, bandPassCorrectionWeights);

            if (delay_compensation) {
                applyDelays(correctedData, delaysAtBegin, delaysAfterEnd, subbandFrequency);
            }
        } else {
            correctedData = fused(inputData, filterWeights,
                    bandPassCorrectionWeights,
                    delaysAtBegin, delaysAfterEnd, subbandFrequency);
        }

        result = correlate(correctedData);
    }

    return result;
}


int main(int, char **)
{
    static_assert(NR_CHANNELS % 16 == 0);
    static_assert(NR_SAMPLES_PER_CHANNEL % NR_SAMPLES_PER_MINOR_LOOP == 0);

    fftInit();

    {
        testFused();
        auto filteredData = testFIR_Filter();
        auto correctedData = testTranspose(filteredData);
        testCorrelator(correctedData);
    }

    omp_set_nested(1);

    auto visibilities = pipeline(60e6);
    cout << std::endl;
    checkCorrelatorTestPattern(visibilities);

    fftDestroy();
    return 0;
}
