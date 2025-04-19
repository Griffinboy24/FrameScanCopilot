/*****************************************************************************
Merge of:
  - InterpFltPhase.h & InterpFltPhase.hpp
  - InterpFlt.h & InterpFlt.hpp
This header contains both the phase?optimized FIR subfilter (InterpFltPhase)
and the primary FIR interpolator (InterpFlt), fully inlined and header?only.

*Tab=3***********************************************************************/
#ifndef rspl_InterpFlt_HEADER_INCLUDED
#define rspl_InterpFlt_HEADER_INCLUDED

#if defined (_MSC_VER)
#pragma once
#pragma warning (4 : 4250) // "Inherits via dominance."
#endif

/*\\\ INCLUDE FILES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
#include "rspl.hpp"
#include <cassert>

namespace rspl
{

    /*****************************************************************************
    This class implements one phase of the scaled FIR interpolator. It stores
    per?phase impulse and delta tables, and provides a convolve() method to
    compute a single?phase output given a fractional position.
    *Tab=3***********************************************************************/
#if ! defined (rspl_InterpFltPhase_HEADER_INCLUDED)
#define rspl_InterpFltPhase_HEADER_INCLUDED

    template <int SC>
    class InterpFltPhase
    {
        /*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
    public:
        enum { SCALE = SC };
        enum { FIR_LEN = 12 * SCALE };

        /*
        ==============================================================================
        Name: ctor
        Throws: Nothing
        ==============================================================================
        */
        InterpFltPhase();

        /*
        ==============================================================================
        Name: convolve
        Description:
          Perform one phase of FIR convolution with interpolation.
        Input parameters:
          - data_ptr: Pointer to input samples (must have FIR_LEN taps available).
          - q:        Fractional coefficient (0 ? q < 1).
        Returns: Filtered value for this phase.
        Throws: assert if not initialized.
        ==============================================================================
        */
        rspl_FORCEINLINE float convolve(const float data_ptr[], float q) const;

        float _dif[FIR_LEN]; // Index inverted (Gd [FIR_LEN-1] first).
        float _imp[FIR_LEN]; // Index inverted.

        /*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
    private:
        enum { CHK_IMPULSE_NOT_SET = 12345 };

        /*\\\ FORBIDDEN MEMBER FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
    private:
        InterpFltPhase(const InterpFltPhase&) = delete;
        InterpFltPhase& operator = (const InterpFltPhase&) = delete;
        bool operator == (const InterpFltPhase&) const = delete;
        bool operator != (const InterpFltPhase&) const = delete;
    };

#endif // rspl_InterpFltPhase_HEADER_INCLUDED

    //------------------------------------------------------------------------------
    // InterpFltPhase definitions
    //------------------------------------------------------------------------------
#if ! defined (rspl_InterpFltPhase_CODEHEADER_INCLUDED)
#define rspl_InterpFltPhase_CODEHEADER_INCLUDED

    template <int SC>
    InterpFltPhase<SC>::InterpFltPhase()
    {
        _imp[0] = CHK_IMPULSE_NOT_SET;
    }

    template <int SC>
    rspl_FORCEINLINE float InterpFltPhase<SC>::convolve(const float data_ptr[], float /*q*/) const
    {
        assert(false);
        return (0.0f);
    }

    template <>
    rspl_FORCEINLINE float InterpFltPhase<1>::convolve(const float data_ptr[], float q) const
    {
        assert(_imp[0] != CHK_IMPULSE_NOT_SET);
        // This way of reordering the convolution operations seems to give the best
        // performances. Actually it may be highly compiler- and architecture-
        // dependent.
        float c_0 = (_imp[0] + _dif[0] * q) * data_ptr[0];
        float c_1 = (_imp[1] + _dif[1] * q) * data_ptr[1];
        c_0 += (_imp[2] + _dif[2] * q) * data_ptr[2];
        c_1 += (_imp[3] + _dif[3] * q) * data_ptr[3];
        c_0 += (_imp[4] + _dif[4] * q) * data_ptr[4];
        c_1 += (_imp[5] + _dif[5] * q) * data_ptr[5];
        c_0 += (_imp[6] + _dif[6] * q) * data_ptr[6];
        c_1 += (_imp[7] + _dif[7] * q) * data_ptr[7];
        c_0 += (_imp[8] + _dif[8] * q) * data_ptr[8];
        c_1 += (_imp[9] + _dif[9] * q) * data_ptr[9];
        c_0 += (_imp[10] + _dif[10] * q) * data_ptr[10];
        c_1 += (_imp[11] + _dif[11] * q) * data_ptr[11];
        assert(FIR_LEN == 12);
        return (c_0 + c_1);
    }

    template <>
    rspl_FORCEINLINE float InterpFltPhase<2>::convolve(const float data_ptr[], float q) const
    {
        assert(_imp[0] != CHK_IMPULSE_NOT_SET);
        // This way of reordering the convolution operations seems to give the best
        // performances. Actually it may be highly compiler- and architecture-
        // dependent.
        float c_0 = (_imp[0] + _dif[0] * q) * data_ptr[0];
        float c_1 = (_imp[1] + _dif[1] * q) * data_ptr[1];
        c_0 += (_imp[2] + _dif[2] * q) * data_ptr[2];
        c_1 += (_imp[3] + _dif[3] * q) * data_ptr[3];
        c_0 += (_imp[4] + _dif[4] * q) * data_ptr[4];
        c_1 += (_imp[5] + _dif[5] * q) * data_ptr[5];
        c_0 += (_imp[6] + _dif[6] * q) * data_ptr[6];
        c_1 += (_imp[7] + _dif[7] * q) * data_ptr[7];
        c_0 += (_imp[8] + _dif[8] * q) * data_ptr[8];
        c_1 += (_imp[9] + _dif[9] * q) * data_ptr[9];
        c_0 += (_imp[10] + _dif[10] * q) * data_ptr[10];
        c_1 += (_imp[11] + _dif[11] * q) * data_ptr[11];
        c_0 += (_imp[12] + _dif[12] * q) * data_ptr[12];
        c_1 += (_imp[13] + _dif[13] * q) * data_ptr[13];
        c_0 += (_imp[14] + _dif[14] * q) * data_ptr[14];
        c_1 += (_imp[15] + _dif[15] * q) * data_ptr[15];
        c_0 += (_imp[16] + _dif[16] * q) * data_ptr[16];
        c_1 += (_imp[17] + _dif[17] * q) * data_ptr[17];
        c_0 += (_imp[18] + _dif[18] * q) * data_ptr[18];
        c_1 += (_imp[19] + _dif[19] * q) * data_ptr[19];
        c_0 += (_imp[20] + _dif[20] * q) * data_ptr[20];
        c_1 += (_imp[21] + _dif[21] * q) * data_ptr[21];
        c_0 += (_imp[22] + _dif[22] * q) * data_ptr[22];
        c_1 += (_imp[23] + _dif[23] * q) * data_ptr[23];
        assert(FIR_LEN == 24);
        return (c_0 + c_1);
    }

#endif // rspl_InterpFltPhase_CODEHEADER_INCLUDED


    /*****************************************************************************
    FIR interpolator. This class is stateless and therefore can be used in "random
    access" on the source sample.
    Template parameters:
     - SC: Scale of the FIR interpolator. Its length is 64 * 12 * SC.
    *Tab=3***********************************************************************/
#if ! defined (rspl_InterpFlt_HEADER_DECLARED)
#define rspl_InterpFlt_HEADER_DECLARED

    template <int SC = 1>
    class InterpFlt
    {
        /*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
    public:
        typedef InterpFltPhase<SC> Phase;
        enum { SCALE = Phase::SCALE };
        enum { FIR_LEN = Phase::FIR_LEN };
        enum { NBR_PHASES_L2 = 6 };
        enum { NBR_PHASES = 1 << NBR_PHASES_L2 };
        enum { IMPULSE_LEN = FIR_LEN * NBR_PHASES };

        /*
        ==============================================================================
        Name: ctor
        Throws: Nothing
        ==============================================================================
        */
        InterpFlt();

        /*
        ==============================================================================
        Name: dtor
        Throws: Nothing
        ==============================================================================
        */
        virtual ~InterpFlt() = default;

        /*
        ==============================================================================
        Name: set_impulse
        Description:
          Set the FIR impulse table (length = FIR_LEN * NBR_PHASES).
          Must be called before interpolate().
        Input parameters:
          - imp_ptr: pointer to first coefficient (centered at IMPULSE_LEN/2).
        Throws: assert if imp_ptr == nullptr.
        ==============================================================================
        */
        void set_impulse(const double imp_ptr[IMPULSE_LEN]);

        /*
        ==============================================================================
        Name: interpolate
        Description:
          Perform fractional interpolation on input data.
        Input parameters:
          - data_ptr: pointer to input samples (must cover FIR_LEN taps around pos).
          - frac_pos: 32?bit fixed?point fractional sample index.
        Returns: Interpolated sample.
        Throws: assert if data_ptr == nullptr.
        ==============================================================================
        */
        rspl_FORCEINLINE float interpolate(const float data_ptr[], UInt32 frac_pos) const;

        /*\\\ PROTECTED \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
    protected:
        /*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
    private:
        Phase _phase_arr[NBR_PHASES];

        /*\\\ FORBIDDEN MEMBER FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
    private:
        InterpFlt(const InterpFlt&) = delete;
        InterpFlt& operator = (const InterpFlt&) = delete;
        bool operator == (const InterpFlt&) const = delete;
        bool operator != (const InterpFlt&) const = delete;
    };

#endif // rspl_InterpFlt_HEADER_DECLARED

    //------------------------------------------------------------------------------
    // InterpFlt definitions
    //------------------------------------------------------------------------------
#if ! defined (rspl_InterpFlt_CODEHEADER_INCLUDED)
#define rspl_InterpFlt_CODEHEADER_INCLUDED

    template <int SC>
    InterpFlt<SC>::InterpFlt()
        : _phase_arr()
    {
        // Nothing
    }

    template <int SC>
    void InterpFlt<SC>::set_impulse(const double imp_ptr[IMPULSE_LEN])
    {
        assert(imp_ptr != nullptr);
        double next_coef_dbl = 0.0;
        for (int fir_pos = FIR_LEN - 1; fir_pos >= 0; --fir_pos)
        {
            for (int phase_cnt = NBR_PHASES - 1; phase_cnt >= 0; --phase_cnt)
            {
                const int imp_pos = fir_pos * NBR_PHASES + phase_cnt;
                const double coef_dbl = imp_ptr[imp_pos];
                const float coef = static_cast<float> (coef_dbl);
                const float dif = static_cast<float> (next_coef_dbl - coef_dbl);
                const int table_pos = FIR_LEN - 1 - fir_pos;
                Phase& phase = _phase_arr[phase_cnt];
                phase._imp[table_pos] = coef;
                phase._dif[table_pos] = dif;
                next_coef_dbl = coef_dbl;
            }
        }
    }

    template <int SC>
    rspl_FORCEINLINE float InterpFlt<SC>::interpolate(const float data_ptr[], UInt32 frac_pos) const
    {
        assert(data_ptr != nullptr);
        // q is made of the lower bits of the fractional position, scaled in the
        // range [0 ; 1[.
        const float q_scl = 1.0f / (65536.0f * 65536.0f);
        const float q = static_cast<float> (frac_pos << NBR_PHASES_L2) * q_scl;
        // Compute phase index (the high-order bits)
        const int phase_index = frac_pos >> (32 - NBR_PHASES_L2);
        const Phase& phase = _phase_arr[phase_index];
        // center the FIR window
        const int offset = -FIR_LEN / 2 + 1;
        return phase.convolve(data_ptr + offset, q);
    }

#endif // rspl_InterpFlt_CODEHEADER_INCLUDED

} // namespace rspl

#endif // rspl_InterpFlt_HEADER_INCLUDED

/*\\\ EOF \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
