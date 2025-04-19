// rspl.hpp — Consolidated RSPL Library Header
#ifndef RSPL_HPP
#define RSPL_HPP



namespace rspl
{

// ————————————————————————————————————————————————————————————————————————
// Fixed‑width integer typedefs
// ————————————————————————————————————————————————————————————————————————
#if defined(_MSC_VER)

    typedef __int16            Int16;
    typedef __int32            Int32;
    typedef __int64            Int64;
    typedef unsigned __int32   UInt32;

#elif (defined(__MWERKS__) || defined(__GNUC__) || defined(__BEOS__))

    #if   SHRT_MAX  == 0x7FFF
        typedef short int     Int16;
    #else
        #error No signed 16‑bit integer type defined for this compiler!
    #endif

    #if   INT_MAX   == 0x7FFFFFFF
        typedef int          Int32;
    #else
        #error No signed 32‑bit integer type defined for this compiler!
    #endif

    typedef long long        Int64;

    #if   UINT_MAX  == 0xFFFFFFFFUL
        typedef unsigned int UInt32;
    #else
        #error No unsigned 32‑bit integer type defined for this compiler!
    #endif

#else

    #error No suitable integer types defined for this compiler!

#endif



// ————————————————————————————————————————————————————————————————————————
// Constants & FORCEINLINE macro
// ————————————————————————————————————————————————————————————————————————
const double PI   = 3.1415926535897932384626433832795;
const double LN2  = 0.69314718055994530941723212145818;
const float  ANTI_DENORMAL_FLT = 1e-20f;

#if defined(_MSC_VER)
    #define rspl_FORCEINLINE __forceinline
#else
    #define rspl_FORCEINLINE inline
#endif



// ————————————————————————————————————————————————————————————————————————
// 32.32 fixed‐point representation
// ————————————————————————————————————————————————————————————————————————
union Fixed3232
{
    Int64 _all;

    struct
    {
    #if defined(__POWERPC__)  // Big endian
        Int32  _msw;
        UInt32 _lsw;
    #else                    // Little endian
        UInt32 _lsw;
        Int32  _msw;
    #endif
    } _part;
};



// ————————————————————————————————————————————————————————————————————————
// Function templates & inline functions
// ————————————————————————————————————————————————————————————————————————
template <typename T>
inline T min(T a, T b)
{
    return (a < b) ? a : b;
}

template <typename T>
inline T max(T a, T b)
{
    return (b < a) ? a : b;
}

inline int round_int(double x)
{
    using namespace std;
    return static_cast<int>(floor(x + 0.5));
}

inline long round_long(double x)
{
    using namespace std;
    return static_cast<long>(floor(x + 0.5));
}

template <typename T>
T shift_bidi(T x, int s)
{
    if (s > 0)
    {
        x <<= s;
    }
    else if (s < 0)
    {
        assert(x >= 0);  // >> has undefined behavior for x < 0
        x >>= -s;
    }
    return x;
}

}  // namespace rspl



// ============================================================================
// System headers required by this consolidated header:
//
//   #include <climits>  // SHRT_MAX, INT_MAX, UINT_MAX
//   #include <cassert>  // assert()
//   #include <cmath>    // floor()
//
// Original internal includes removed. Please add the above as needed.
// ============================================================================

#endif // RSPL_HPP
