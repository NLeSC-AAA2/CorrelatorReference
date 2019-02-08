// Copyright 2018-2019 Netherlands eScience Center and ASTRON
// Licensed under the Apache License, version 2.0. See LICENSE for details.
#include <complex>

#include <boost/multi_array.hpp>

namespace correlator
{
    constexpr int NR_INPUTS = 2 * 576;
    constexpr int NR_CHANNELS = 64;
    constexpr int NR_SAMPLES_PER_CHANNEL = 3072;
    constexpr double SUBBAND_BANDWIDTH = 195312.5;
    constexpr int NR_TAPS = 16;
    constexpr int NR_BASELINES = NR_INPUTS * (NR_INPUTS + 1) / 2;
    constexpr int NR_SAMPLES_PER_MINOR_LOOP = 64;

    typedef boost::multi_array<std::complex<float>, 3> InputDataType;
    const auto InputDataDims = boost::extents[NR_INPUTS][NR_SAMPLES_PER_CHANNEL + NR_TAPS - 1][NR_CHANNELS];
    typedef boost::multi_array<std::complex<float>, 3> FilteredDataType;
    const auto FilteredDataDims = boost::extents[NR_INPUTS][NR_SAMPLES_PER_CHANNEL][NR_CHANNELS];
    typedef boost::multi_array<float, 2> FilterWeightsType;
    const auto FilterWeightsDims = boost::extents[NR_TAPS][NR_CHANNELS];
    typedef boost::multi_array<float, 1> BandPassCorrectionWeights;
    const auto BandPassCorrectionWeightsDims = boost::extents[NR_CHANNELS];
    typedef boost::multi_array<double, 1> DelaysType;
    const auto DelaysDims = boost::extents[NR_INPUTS];
    typedef boost::multi_array<std::complex<float>, 3> CorrectedDataType;
    const auto CorrectedDataDims = boost::extents[NR_CHANNELS][NR_INPUTS][NR_SAMPLES_PER_CHANNEL];
    typedef boost::multi_array<std::complex<float>, 2> VisibilitiesType;
    const auto VisibilitiesDims = boost::extents[NR_CHANNELS][NR_BASELINES];

    typedef boost::multi_array<std::complex<float>, 2> FusedFilterType;
    const auto FusedFilterDims = boost::extents[NR_SAMPLES_PER_MINOR_LOOP][NR_CHANNELS];
    typedef boost::multi_array<std::complex<float>, 1> ComplexChannelType;
    const auto ComplexChannelDims = boost::extents[NR_CHANNELS];

    extern const bool delay_compensation;
    extern const bool bandpass_correction;

    void FFT(FilteredDataType&);

    namespace fused
    {
        void FFT(FusedFilterType&);
    }
}
