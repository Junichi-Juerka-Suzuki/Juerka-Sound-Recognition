#include <complex>
#include <vector>
#define MKL_Complex16 std::complex<double>
#include "mkl.h"

#include "Fft.h"

namespace Juerka::SoundRecognition
{
    using std::complex;
    using std::vector;

    void do_dft
    (
        vector<double>& in,
        vector<complex<double>>& out
    )
    {
        DFTI_DESCRIPTOR_HANDLE descriptor;
        MKL_LONG status;

        status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_REAL, 1, in.size()); //Specify size and precision
        status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE); //Out of place FFT
        status = DftiCommitDescriptor(descriptor); //Finalize the descriptor
        status = DftiComputeForward(descriptor, in.data(), out.data()); //Compute the Forward FFT
        status = DftiFreeDescriptor(&descriptor); //Free the descriptor

        (void)status;
    }
}

