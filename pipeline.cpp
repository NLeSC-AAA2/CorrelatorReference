// (C) 2013,2014,2015 John Romein/ASTRON
// Copyright 2018-2019 Netherlands eScience Center and ASTRON
// Licensed under the Apache License, version 2.0. See LICENSE for details.
#include <iostream>
#include <omp.h>

#include "correlator.hpp"

using std::cout, std::cerr;
using namespace std::complex_literals;
using namespace correlator;

static bool output_check = true;

static FilteredDataType
FIR_filter(const InputDataType&, const FilterWeightsType&);

static CorrectedDataType
transpose(const FilteredDataType&, const BandPassCorrectionWeights&);

static void
applyDelays(CorrectedDataType&, const DelaysType&, const DelaysType&, double);

static CorrectedDataType
fused_pipeline
( const InputDataType&
, const FilterWeightsType&
, const BandPassCorrectionWeights&
, const DelaysType&
, const DelaysType&
, double
);

static VisibilitiesType
correlate(const CorrectedDataType&);


////// Initialise data

static InputDataType
inputTestPattern(bool isFused = false)
{
    InputDataType result(InputDataDims);

    if (isFused) {
        std::fill_n(result.data(), result.num_elements(), 0);
        if (NR_INPUTS > 6 && NR_SAMPLES_PER_CHANNEL > 27 && NR_CHANNELS > 12) {
            result[6][27 + NR_TAPS - 1][12] = {2, 3};
        }
    } else {
        const int totalTime = NR_SAMPLES_PER_CHANNEL + NR_TAPS - 1;
        signed char count = 64;
        for (unsigned input = 0; input < NR_INPUTS; input ++) {
            for (unsigned time = 0; time < totalTime; time ++) {
                for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
                    result[input][time][channel] = std::complex<float>(count, static_cast<signed char>(totalTime * NR_CHANNELS + count));
                    count++;
                }
            }

            count += totalTime * NR_CHANNELS;
        }

        if (NR_INPUTS > 9 && NR_SAMPLES_PER_CHANNEL > 99 && NR_CHANNELS > 12) {
            result[9][98 + NR_TAPS - 1][12] = {4, result[9][98 + NR_TAPS - 1][12].imag()};
            result[9][99 + NR_TAPS - 1][12] = {5, result[9][99 + NR_TAPS - 1][12].imag()};
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


static BandPassCorrectionWeights
bandPassTestPattern(bool isFused = false)
{
    BandPassCorrectionWeights result(BandPassCorrectionWeightsDims);
    std::fill_n(result.data(), result.num_elements(), 1);

    if (bandpass_correction && !isFused && NR_CHANNELS > 5)
        result[5] = 2;

    return result;
}


static DelaysType
delaysTestPattern(bool isBegin, bool isFused = false)
{
    DelaysType result(DelaysDims);
    std::fill_n(result.data(), result.num_elements(), 0);

    if (delay_compensation && !isBegin && !isFused && NR_INPUTS > 22)
        result[22] = 1e-6;

    return result;
}


////// Check results

static void
checkFIR_FilterTestPattern(const FilteredDataType& filteredData)
{
    for (unsigned input = 0; input < NR_INPUTS; input ++)
        for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
                if (filteredData[input][time][channel] != (0.0f + 0.0if)) {
                    cout << "input = " << input << ", time = " << time
                         << ", channel = " << channel << ", sample = "
                         << filteredData[input][time][channel]
                         << std::endl;
                }
}


static void
checkTransposeTestPattern(const CorrectedDataType& correctedData)
{
    for (int channel = 0; channel < NR_CHANNELS; channel ++)
        for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
            for (int input = 0; input < NR_INPUTS; input ++)
                if (correctedData[channel][input][time] != (0.0f + 0.0if)) {
                    cout << "channel = " << channel << ", time = " << time
                         << ", input = " << input << ", value = "
                         << correctedData[channel][input][time]
                         << std::endl;
                }
}


static void
checkFusedTestPattern(const CorrectedDataType& correctedData)
{
    for (unsigned input = 0; input < NR_INPUTS; input ++)
        for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++)
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
                if (correctedData[channel][input][time] != (0.0f + 0.0if)) {
                    cout << "input = " << input << ", time = " << time
                         << ", channel = " << channel << ": ("
                         << correctedData[channel][input][time].real()
                         << ", " << correctedData[channel][input][time].imag()
                         << ')' << std::endl;
                }
}


static void
checkCorrelatorTestPattern(const VisibilitiesType& visibilities)
{
    for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
        for (unsigned baseline = 0; baseline < NR_BASELINES; baseline ++)
            if (visibilities[channel][baseline] != (0.0f + 0.0if)) {
                cout << "channel = " << channel << ", baseline = " << baseline
                     << ", visibility = " << visibilities[channel][baseline]
                     << std::endl;
            }
}


////// Test runs

static FilteredDataType
testFIR_Filter()
{
    const auto& filteredData = FIR_filter(inputTestPattern(), filterWeightsTestPattern());

    if (output_check) checkFIR_FilterTestPattern(filteredData);
    return filteredData;
}


static CorrectedDataType
testTranspose(FilteredDataType& filteredData)
{
    if (NR_INPUTS > 22 && NR_SAMPLES_PER_CHANNEL > 99 && NR_CHANNELS > 5) {
        filteredData[22][99][5] = {2, 3};
    }

    auto bandPassCorrectionWeights = bandPassTestPattern();

    auto correctedData = transpose(filteredData, bandPassCorrectionWeights);

    applyDelays(correctedData, delaysTestPattern(true), delaysTestPattern(false), 60e6);

    if (output_check) checkTransposeTestPattern(correctedData);
    return correctedData;
}


static void
testFused()
{
    auto correctedData = fused_pipeline(
            inputTestPattern(true), filterWeightsTestPattern(true),
            bandPassTestPattern(true),
            delaysTestPattern(true, true), delaysTestPattern(false, true), 60e6);

    if (output_check) checkFusedTestPattern(correctedData);
}


static void
testCorrelator(CorrectedDataType& correctedData)
{
    if constexpr (NR_CHANNELS > 5 && NR_SAMPLES_PER_CHANNEL > 99 && NR_INPUTS > 19) {
        correctedData[5][ 0][99] = {3,4};
        correctedData[5][18][99] = {5,6};
    }

    auto visibilities = correlate(correctedData);

    if (output_check) checkCorrelatorTestPattern(visibilities);
}


////// non-fused

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
            for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
                for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
                    filteredData[input][time][channel] = {0, 0};

                    for (unsigned tap = 0; tap < NR_TAPS; tap ++) {
                        filteredData[input][time][channel] +=
                            filterWeights[tap][channel] * inputData[input][time + tap][channel];
                    }
                }
            }
        }
    }

    return filteredData;
}


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
                    correctedData[channel][input][time] = filteredData[input][time][channel];

                    correctedData[channel][input][time] *= bandPassCorrectionWeights[channel];
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
                double phiBegin = -2.0 * 3.141592653589793 * delaysAtBegin[input];
                double phiEnd   = -2.0 * 3.141592653589793 * delaysAfterEnd[input];
                double deltaPhi = (phiEnd - phiBegin) / NR_SAMPLES_PER_CHANNEL;
                double channelFrequency = subbandFrequency - .5 * SUBBAND_BANDWIDTH + channel * (SUBBAND_BANDWIDTH / NR_CHANNELS);
                float myPhiBegin = static_cast<float>((phiBegin /* + startTime * deltaPhi */) * channelFrequency /* + phaseOffsets[stationPol + major] */);
                float myPhiDelta = static_cast<float>(deltaPhi * channelFrequency);
                std::complex<float> v = std::polar(1.0f, myPhiBegin);
                std::complex<float> dv = std::polar(1.0f, myPhiDelta);

                for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
                    correctedData[channel][input][time] *= v;
                    v *= dv;
                }
            }
        }
    }
}


static CorrectedDataType
nonfused_pipeline
( const InputDataType& inputData
, const FilterWeightsType& filterWeights
, const BandPassCorrectionWeights& bandPassCorrectionWeights
, const DelaysType& delaysAtBegin
, const DelaysType& delaysAfterEnd
, double subbandFrequency
)
{
    auto filteredData = FIR_filter(inputData, filterWeights);
    FFT(filteredData);
    auto result = transpose(filteredData, bandPassCorrectionWeights);

    applyDelays(result, delaysAtBegin, delaysAfterEnd, subbandFrequency);

    return result;
}


////// fused

static void
fused_FIRfilter
( const InputDataType& inputData
, const FilterWeightsType& filterWeights
, FusedFilterType& filteredData
, unsigned input
, unsigned majorTime
)
{
    for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++) {
        for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
            std::complex<float> sum = {0, 0};

            for (unsigned tap = 0; tap < NR_TAPS; tap ++) {
                sum += filterWeights[tap][channel] * inputData[input][majorTime + minorTime + tap][channel];
            }

            filteredData[minorTime][channel] = sum;
        }
    }
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
        v[channel] = std::polar(1.0f, myPhiBegin);
        dv[channel] = std::polar(1.0f, myPhiDelta);

        v[channel] *= bandPassCorrectionWeights[channel];
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
    if (!delay_compensation) {
        // BandPass correction, if not doing delay compensation

        for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++) {
            for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
                filteredData[minorTime][channel] *= bandPassCorrectionWeights[channel];
            }
        }
    }

    // Delay compensation & transpose

    for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
        for (unsigned minorTime = 0; minorTime < NR_SAMPLES_PER_MINOR_LOOP; minorTime ++) {
            correctedData[channel][input][majorTime + minorTime] = filteredData[minorTime][channel];

            if (delay_compensation) {
                correctedData[channel][input][majorTime + minorTime] *= v[channel];
                v[channel] *= dv[channel];
            }
        }
    }
}


static CorrectedDataType
fused_pipeline
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
            FusedFilterType filteredData(FusedFilterDims);

            ComplexChannelType v(ComplexChannelDims);
            ComplexChannelType dv(ComplexChannelDims);

            fused_TransposeInit(v, dv,
                    bandPassCorrectionWeights,
                    delaysAtBegin, delaysAfterEnd, subbandFrequency, input);

            for (unsigned majorTime = 0; majorTime < NR_SAMPLES_PER_CHANNEL; majorTime += NR_SAMPLES_PER_MINOR_LOOP) {
                fused_FIRfilter(inputData, filterWeights, filteredData, input, majorTime);
                fused::FFT(filteredData);
                fused_Transpose(correctedData, bandPassCorrectionWeights, filteredData,
                        v, dv,
                        input, majorTime);
            }
        }
    }

    return correctedData;
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
                    visibilities[channel][baseline] = {sum_real, sum_imag};
                }
            }
        }
    }

    return visibilities;
}


int main(int argc, char **)
{
    if (argc > 1) output_check = false;
    static_assert(NR_CHANNELS % 16 == 0);
    static_assert(NR_SAMPLES_PER_CHANNEL % NR_SAMPLES_PER_MINOR_LOOP == 0);

    {
        testFused();
        auto filteredData = testFIR_Filter();
        auto correctedData = testTranspose(filteredData);
        testCorrelator(correctedData);
    }

    omp_set_nested(1);

    {
        const auto& bandPassCorrectionWeights = bandPassTestPattern();
        const auto& delaysAtBegin = delaysTestPattern(true);
        const auto& delaysAfterEnd = delaysTestPattern(false);
        const auto& inputData = inputTestPattern();
        const auto& filterWeights = filterWeightsTestPattern();

        //non-fused version
        const auto& correctedData = nonfused_pipeline(inputData, filterWeights,
                    bandPassCorrectionWeights,
                    delaysAtBegin, delaysAfterEnd,
                    60e6);


        //fused version
        const auto& fusedCorrectedData = fused_pipeline(inputData,
                    filterWeights, bandPassCorrectionWeights, delaysAtBegin,
                    delaysAfterEnd, 60e6);

        if (correctedData != fusedCorrectedData) {
            cout << "Error!" << std::endl;
        }

        const auto& visibilities = correlate(correctedData);
        if (output_check) {
            cout << std::endl;
            checkCorrelatorTestPattern(visibilities);
        }
    }

    return 0;
}
