#pragma once

#include <complex>
#include <vector>

namespace Juerka::SoundRecognition
{
    using std::complex;
    using std::vector;

    void do_dft
    (
        vector<double>& in,
        vector<complex<double>>& out
    );
}
