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
    namespace {
        class FFTHandle {
            DFTI_DESCRIPTOR_HANDLE handle;

          public:
            FFTHandle(size_t N, size_t num_transforms)
            {
                MKL_LONG error;
                const char* msg;

                error = DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 1, N);
                if (error != DFTI_NO_ERROR) {
                    msg = "DftiCreateDescriptor failed";
                    goto MKL_ERROR;
                }

                error = DftiSetValue(handle, DFTI_NUMBER_OF_TRANSFORMS, num_transforms);
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

            void operator()(std::complex<float>* data) const
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
    }

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

    namespace fused
    {
        void
        FFT(FusedFilterType& filteredData)
        { fft(filteredData[0].origin()); }
    }
}
