/*****************************************************************************

Class halving the sample rate (2x-downsampler) with the help of a polyphase
IIR low-pass filter. It is used by ResamplerFlt, but can be used alone.
The number of coefficients is hardwired to 7 in this implementation. Check
the Artur Krukowski's webpage (http://www.cmsa.wmin.ac.uk/~artur/Poly.html)
to know more about the filter coefficients to submit.

*Tab=3***********************************************************************/

#if ! defined (rspl_Downsampler2_HEADER_INCLUDED)
#define rspl_Downsampler2_HEADER_INCLUDED

#if defined (_MSC_VER)
 #pragma once
 #pragma warning (4 : 4250) // "Inherits via dominance."
#endif

/*\\\ INCLUDE FILES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
#include "rspl.hpp"
#include <cassert>

namespace rspl
{

class Downsampler2Flt
{
/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
public:
    enum { NBR_COEFS = 7 };

    /*
    ==============================================================================
    Name: ctor
    Throws: Nothing
    ==============================================================================
    */
    inline Downsampler2Flt();

    /*
    ==============================================================================
    Name: dtor
    Throws: Nothing
    ==============================================================================
    */
    inline ~Downsampler2Flt() = default;

    /*
    ==============================================================================
    Name: set_coefs
    Description:
      Set the filter coefficients.
    Input parameters:
      - coef_ptr: pointer on the array containing the coefficients.
    Throws: Nothing
    ==============================================================================
    */
    inline void set_coefs (const double coef_ptr [NBR_COEFS]);

    /*
    ==============================================================================
    Name: clear_buffers
    Description:
      Clear the state buffer of the filter, as if input sample were 0 for an
      infinite time.
    Throws: Nothing.
    ==============================================================================
    */
    inline void clear_buffers ();

    /*
    ==============================================================================
    Name: downsample_block
    Description:
      Downsample by 2 a block of sample data. Output rate is half the input.
      set_coefs() must have been called at least once before processing anything.
      Warning: the volume is doubled! The right formula includes a 0.5 gain,
      averaging both paths instead of just summing them. We removed this gain to
      ensure maximum performance. You can compensate it later in the processing
      path.
    Input parameters:
      - src_ptr: pointer on input sample data.
      - nbr_spl: Number of output samples to generate. Number of provided samples
                 has to be twice this value.
    Output parameters:
      - dest_ptr: pointer on downsampled sample data to generate. Can be the
                  same address as src_ptr, because the algorithm can process
                  in-place.
    Throws: Nothing
    ==============================================================================
    */
    inline void downsample_block (float dest_ptr [], const float src_ptr [], long nbr_spl);

    /*
    ==============================================================================
    Name: phase_block
    Description:
      This function is used to adjust the phase of a signal which rate is not
      change, in order to match phase of downsampled signals. Basically works
      by inserting 0 between samples and downsampling by a factor two.
      Unlike downsample_block(), the gain does not need to be fixed.
    Input parameters:
      - src_ptr: Pointer on the data whose phase has to be shifted.
      - nbr_spl: Number of samples to process. > 0.
    Output parameters:
      - dest_ptr: Pointer on data to generate. Can be the same address as
                  src_ptr, because the algorithm can process in-place.
    Throws: Nothing
    ==============================================================================
    */
    inline void phase_block (float dest_ptr [], const float src_ptr [], long nbr_spl);

/*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
private:
    // Magic code for checking if coefs have been set or not in process_block()
    enum { CHK_COEFS_NOT_SET = 12345 };

    float _coef_arr [NBR_COEFS];
    float _x_arr    [2];
    float _y_arr    [NBR_COEFS];

    /*
    ==============================================================================
    Name: process_sample
    Description:
      Filter and downsample a pair of samples. Gain is implicitly boosted by a
      factor two to save one multiplication.
    Input parameters:
      - path_0: The *second* sample
      - path_1: The *first* sample
    Returns: Downsampled sample
    Throws: Nothing
    ==============================================================================
    */
    inline rspl_FORCEINLINE float process_sample (float path_0, float path_1);

/*\\\ FORBIDDEN MEMBER FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
private:
    Downsampler2Flt (const Downsampler2Flt &)            = delete;
    Downsampler2Flt & operator = (const Downsampler2Flt &) = delete;
    bool operator == (const Downsampler2Flt &) const       = delete;
    bool operator != (const Downsampler2Flt &) const       = delete;
};

//------------------------------------------------------------------------------
// Definitions
//------------------------------------------------------------------------------

inline Downsampler2Flt::Downsampler2Flt ()
: _coef_arr ()
, _x_arr    ()
, _y_arr    ()
{
    _coef_arr [0] = static_cast <float> (CHK_COEFS_NOT_SET);
    clear_buffers ();
}

inline void Downsampler2Flt::set_coefs (const double coef_ptr [NBR_COEFS])
{
    assert (coef_ptr != nullptr);
    for (int mem = 0; mem < NBR_COEFS; ++mem)
    {
        const float coef = static_cast <float> (coef_ptr [mem]);
        assert (coef > 0.0f);
        assert (coef < 1.0f);
        _coef_arr [mem] = coef;
    }
}

inline void Downsampler2Flt::clear_buffers ()
{
    _x_arr [0] = 0.0f;
    _x_arr [1] = 0.0f;
    for (int mem = 0; mem < NBR_COEFS; ++mem)
    {
        _y_arr [mem] = 0.0f;
    }
}

inline void Downsampler2Flt::downsample_block (float dest_ptr [], const float src_ptr [], long nbr_spl)
{
    assert (_coef_arr [0] != static_cast <float> (CHK_COEFS_NOT_SET));
    assert (dest_ptr != nullptr);
    assert (src_ptr  != nullptr);
    assert (nbr_spl > 0);

    long pos = 0;
    do
    {
        const float path_0 = src_ptr [pos * 2 + 1];
        const float path_1 = src_ptr [pos * 2];
        dest_ptr [pos]     = process_sample (path_0, path_1);
        ++ pos;
    }
    while (pos < nbr_spl);
}

inline void Downsampler2Flt::phase_block (float dest_ptr [], const float src_ptr [], long nbr_spl)
{
    assert (_coef_arr [0] != static_cast <float> (CHK_COEFS_NOT_SET));
    assert (dest_ptr != nullptr);
    assert (src_ptr  != nullptr);
    assert (nbr_spl   > 0);

    long pos = 0;
    do
    {
        float path_1 = src_ptr [pos];
        dest_ptr [pos] = process_sample (0.0f, path_1);
        ++ pos;
    }
    while (pos < nbr_spl);

    // Kills denormals on path 0, if any. Theoretically we just need to do it
    // on results of multiplications with coefficients < 0.5.
    _y_arr [0] += ANTI_DENORMAL_FLT;
    _y_arr [2] += ANTI_DENORMAL_FLT;
    _y_arr [4] += ANTI_DENORMAL_FLT;
    _y_arr [6] += ANTI_DENORMAL_FLT;
    _y_arr [0] -= ANTI_DENORMAL_FLT;
    _y_arr [2] -= ANTI_DENORMAL_FLT;
    _y_arr [4] -= ANTI_DENORMAL_FLT;
    _y_arr [6] -= ANTI_DENORMAL_FLT;
}

inline rspl_FORCEINLINE float Downsampler2Flt::process_sample (float path_0, float path_1)
{
    float tmp_0 = _x_arr [0];
    float tmp_1 = _x_arr [1];
    _x_arr [0]  = path_0;
    _x_arr [1]  = path_1;

    path_0 = (path_0 - _y_arr [0]) * _coef_arr [0] + tmp_0;
    path_1 = (path_1 - _y_arr [1]) * _coef_arr [1] + tmp_1;
    tmp_0   = _y_arr [0];
    tmp_1   = _y_arr [1];
    _y_arr [0] = path_0;
    _y_arr [1] = path_1;

    path_0 = (path_0 - _y_arr [2]) * _coef_arr [2] + tmp_0;
    path_1 = (path_1 - _y_arr [3]) * _coef_arr [3] + tmp_1;
    tmp_0   = _y_arr [2];
    tmp_1   = _y_arr [3];
    _y_arr [2] = path_0;
    _y_arr [3] = path_1;

    path_0 = (path_0 - _y_arr [4]) * _coef_arr [4] + tmp_0;
    path_1 = (path_1 - _y_arr [5]) * _coef_arr [5] + tmp_1;
    tmp_0   = _y_arr [4];
    _y_arr [4] = path_0;
    _y_arr [5] = path_1;

    path_0 = (path_0 - _y_arr [6]) * _coef_arr [6] + tmp_0;
    _y_arr [6] = path_0;

    assert (NBR_COEFS == 7);
    return (path_0 + path_1);
}

} // namespace rspl

#endif // rspl_Downsampler2_HEADER_INCLUDED

/*\\\ EOF \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
