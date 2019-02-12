// Copyright 2019 Netherlands eScience Center and ASTRON
// Licensed under the Apache License, version 2.0. See LICENSE for details.
#include <mkl.h>
#include <omp.h>

#include "correlator.hpp"

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

namespace correlator
{

const bool delay_compensation = DELAY_COMPENSATION;
const bool bandpass_correction = BANDPASS_CORRECTION;

////// FFT
namespace
{

class FFTHandle {
    DFTI_DESCRIPTOR_HANDLE handle;

  public:
    FFTHandle(size_t N, size_t n_transforms)
    {
        MKL_LONG error;
        const char* msg;

        error = DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 1, N);
        if (error != DFTI_NO_ERROR) {
            msg = "DftiCreateDescriptor failed";
            goto MKL_ERROR;
        }

        error = DftiSetValue(handle, DFTI_NUMBER_OF_TRANSFORMS, n_transforms);
        if (error != DFTI_NO_ERROR) {
            msg = "DftiSetValue failed";
            goto MKL_ERROR;
        }

        error = DftiSetValue(handle, DFTI_INPUT_DISTANCE, N);
        if (error != DFTI_NO_ERROR) {
            msg = "DftiSetValue failed";
            goto MKL_ERROR;
        }

        error = DftiSetValue(handle, DFTI_OUTPUT_DISTANCE, N);
        if (error != DFTI_NO_ERROR) {
            msg = "DftiSetValue failed";
            goto MKL_ERROR;
        }

        error = DftiCommitDescriptor(handle);
        if (error != DFTI_NO_ERROR) {
            msg = "DftiCommitDescriptor failed";
            goto MKL_ERROR;
        }

        return;

      MKL_ERROR:
        error = DftiFreeDescriptor(&handle);
        if (error != DFTI_NO_ERROR) {
            std::cerr << "DftiFreeDescriptor failed" << std::endl;
        }
        throw std::runtime_error(msg);
    }

    ~FFTHandle()
    {
        MKL_LONG error;

        error = DftiFreeDescriptor(&handle);
        if (error != DFTI_NO_ERROR) {
            std::cerr << "DftiFreeDescriptor failed" << std::endl;
        }
    }

    void
    operator()(std::complex<float>* data) const
    {
        MKL_LONG error;
        error = DftiComputeForward(handle, data);
        if (error != DFTI_NO_ERROR) {
            throw std::runtime_error("DftiCreateDescriptor failed");
        }
    }
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
const FFTHandle fft(NR_CHANNELS, NR_SAMPLES_PER_MINOR_LOOP);
#pragma clang diagnostic pop
} // anonymous namespace

void
FFT(FilteredDataType& filteredData)
{
    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(dynamic)
        for (int input = 0; input < NR_INPUTS; input ++)
            for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time += NR_SAMPLES_PER_MINOR_LOOP)
                fft(filteredData[input][time].origin());
    }
}

FilteredDataType
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
                            filterWeights[channel][tap] * inputData[input][channel][time + tap];
                    }
                }
            }
        }
    }

    return filteredData;
}


CorrectedDataType
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


void
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
                float myPhiBegin = static_cast<float>(phiBegin * channelFrequency);
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

CorrectedDataType
pipeline
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

VisibilitiesType
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
} // namespace correlator

namespace correlator::fused
{
void
FFT(boost::multi_array_ref<std::complex<float>, 2>::reference filteredData)
{ fft(filteredData.origin()); }

void
FIR_filter
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
                sum += filterWeights[channel][tap] * inputData[input][channel][majorTime + minorTime + tap];
            }

            filteredData[minorTime][channel] = sum;
        }
    }
}


void
transposeInit
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
        float myPhiBegin = static_cast<float>(phiBegin * channelFrequency);
        float myPhiDelta = static_cast<float>(deltaPhi * channelFrequency);
        v[channel] = std::polar(1.0f, myPhiBegin);
        dv[channel] = std::polar(1.0f, myPhiDelta);

        v[channel] *= bandPassCorrectionWeights[channel];
    }
}


void
transpose
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

CorrectedDataType
pipeline
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

            transposeInit(v, dv, bandPassCorrectionWeights,
                    delaysAtBegin, delaysAfterEnd, subbandFrequency, input);

            for (unsigned majorTime = 0; majorTime < NR_SAMPLES_PER_CHANNEL; majorTime += NR_SAMPLES_PER_MINOR_LOOP) {
                FIR_filter(inputData, filterWeights, filteredData, input, majorTime);
                FFT(filteredData[0]);
                transpose(correctedData, bandPassCorrectionWeights, filteredData,
                        v, dv, input, majorTime);
            }
        }
    }

    return correctedData;
}
} // namespace correlator::fused
