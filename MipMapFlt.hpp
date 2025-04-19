/*****************************************************************************

This is the sample data container. It parses original data, makes MIP-maps
with the help of the provided filter and store them into memory.
How to use this class:
 1. Build an instance of it.
 2. Call init_sample()
 3. Call fill_sample() as many times it is needed to complete the sample data.
 4. You can now use the other functions.

*Tab=3***********************************************************************/

#pragma once

#include "rspl.hpp"
#include <vector>
#include <cassert>

namespace rspl
{

class MipMapFlt
{
/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
public:
    /*
    ==============================================================================
    Name: ctor
    Throws: Nothing
    ==============================================================================
    */
    inline MipMapFlt();

    /*
    ==============================================================================
    Name: dtor
    Throws: Nothing
    ==============================================================================
    */
    inline ~MipMapFlt() = default;

    /*
    ==============================================================================
    Name: init_sample
    Description:
      Provides object with sample information.
      Sets the FIR impulse for the half‑band filter. Its length must be odd and
      the impulse should be centered. It is supposed to be symmetric.
      This function must be called before anything else. If sample length is not
      0, fill_sample() must be called before using the sample.
    Input parameters:
      - len: Length of the sample, in samples. >= 0
      - add_len_pre: Required data before each integer sample position. >= 0.
      - add_len_post: Same as add_len_pre, but after sample. >= 0.
      - nbr_tables: Number of desired mip‑map levels. 1 = just provided data. > 0.
      - imp_ptr: Pointer on impulse data.
      - nbr_taps: Number of taps. > 0 and odd.
    Returns: true if more data are needed to fill the sample.
    Throws: std::vector related exceptions
    ==============================================================================
    */
    inline bool init_sample (long len, long add_len_pre, long add_len_post, int nbr_tables, const double imp_ptr[], int nbr_taps);

    /*
    ==============================================================================
    Name: fill_sample
    Description:
      Provide sample data. Must be called after init_sample() and before using
      data. Multiple calls allowed to fill in blocks. Total length must match.
    Input parameters:
      - data_ptr: Pointer on sample data.
      - nbr_spl: Number of samples to load.
    Returns: true if more data are needed.
    Throws: Nothing.
    ==============================================================================
    */
    inline bool fill_sample (const float data_ptr[], long nbr_spl);

    /*
    ==============================================================================
    Name: clear_sample
    Description:
      Remove the loaded sample and free the allocated memory. Can be called at
      any time.
    Throws: std::vector related exceptions
    ==============================================================================
    */
    inline void clear_sample ();

    /*
    ==============================================================================
    Name: is_ready
    Description:
      Indicates whether the object is ready to use (sample fully loaded) or not.
    Returns: true if object is ready, false otherwise.
    Throws: Nothing
    ==============================================================================
    */
    inline bool is_ready () const;

    /*
    ==============================================================================
    Name: get_sample_len
    Returns: The length of the original sample.
    Throws: assert if not ready.
    ==============================================================================
    */
    inline long get_sample_len () const;

    /*
    ==============================================================================
    Name: get_lev_len
    Input:
      - level: mip‑map level index.
    Returns: Length of level‑th table.
    Throws: assert if out of range.
    ==============================================================================
    */
    inline long get_lev_len (int level) const;

    /*
    ==============================================================================
    Name: get_nbr_tables
    Returns: Number of mip‑map levels.
    Throws: assert if not ready.
    ==============================================================================
    */
    inline const int get_nbr_tables () const;

    /*
    ==============================================================================
    Name: use_table
    Input:
      - table: table index.
    Returns: Pointer to the table’s float data.
    Throws: assert if not ready or index invalid.
    ==============================================================================
    */
    inline const float * use_table (int table) const;

/*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
private:
    class TableData
    {
    public:
        typedef std::vector<float> SplData;
        SplData _data;
        float *  _data_ptr;
    };

    typedef TableData::SplData SplData;
    typedef std::vector<TableData> TableArr;

    void resize_and_clear_tables ();
    bool check_sample_and_build_mip_map ();
    void build_mip_map_level (int level);
    float filter_sample (const TableData::SplData &table, long pos) const;

    TableArr _table_arr;
    SplData  _filter;         // First stored coef is the "center".
    long     _len;            // >= 0; < 0 = uninitialized
    long     _add_len_pre;    // >= 0
    long     _add_len_post;   // >= 0
    long     _filled_len;     // >= 0
    int      _nbr_tables;     // > 0

    MipMapFlt (const MipMapFlt &)            = delete;
    MipMapFlt & operator = (const MipMapFlt &) = delete;
    bool operator == (const MipMapFlt &) const  = delete;
    bool operator != (const MipMapFlt &) const  = delete;
};

//------------------------------------------------------------------------------
// Definitions
//------------------------------------------------------------------------------

inline MipMapFlt::MipMapFlt()
: _table_arr()
, _filter()
, _len(-1)
, _add_len_pre(0)
, _add_len_post(0)
, _filled_len(0)
, _nbr_tables(0)
{
    // Nothing
}

inline bool MipMapFlt::init_sample (long len, long add_len_pre, long add_len_post, int nbr_tables, const double imp_ptr[], int nbr_taps)
{
    assert (len >= 0);
    assert (add_len_pre >= 0);
    assert (add_len_post >= 0);
    assert (nbr_tables > 0);
    assert (imp_ptr != nullptr);
    assert ((nbr_taps & 1) == 1);

    const int half_fir_len = (nbr_taps - 1) / 2;
    _filter.resize(half_fir_len + 1);
    for (int pos = 0; pos <= half_fir_len; ++pos)
    {
        _filter[pos] = static_cast<float>(imp_ptr[half_fir_len + pos]);
    }

    const long filter_sup = static_cast<long>(half_fir_len * 2);
    _len           = len;
    _add_len_pre   = std::max(add_len_pre, filter_sup);
    _add_len_post  = std::max(add_len_post, filter_sup);
    _filled_len    = 0;
    _nbr_tables    = nbr_tables;

    resize_and_clear_tables();
    return (check_sample_and_build_mip_map());
}

inline bool MipMapFlt::fill_sample (const float data_ptr[], long nbr_spl)
{
    assert (_len >= 0);
    assert (_nbr_tables > 0);
    assert (_table_arr.size() > 0);
    assert (data_ptr != nullptr);
    assert (nbr_spl > 0);
    assert (nbr_spl <= _len - _filled_len);

    TableData::SplData &sample = _table_arr[0]._data;
    const long offset = _add_len_pre + _filled_len;
    const long work_len = std::min(nbr_spl, _len - _filled_len);

    for (long pos = 0; pos < work_len; ++pos)
    {
        sample[offset + pos] = data_ptr[pos];
    }
    _filled_len += work_len;
    return (check_sample_and_build_mip_map());
}

inline void MipMapFlt::clear_sample ()
{
    _len        = -1;
    _add_len_pre   = 0;
    _add_len_post  = 0;
    _filled_len = 0;
    _nbr_tables = 0;
    TableArr().swap(_table_arr);
    SplData().swap(_filter);
}

inline bool MipMapFlt::is_ready () const
{
    bool ready = (_len >= 0) && (_nbr_tables > 0) && (_filled_len == _len);
    return ready;
}

inline long MipMapFlt::get_sample_len () const
{
    assert (is_ready());
    return _len;
}

inline const int MipMapFlt::get_nbr_tables () const
{
    assert (is_ready());
    return _nbr_tables;
}

inline long MipMapFlt::get_lev_len (int level) const
{
    assert (_len >= 0);
    assert (level >= 0 && level < _nbr_tables);
    const long scale = 1L << level;
    return (_len + scale - 1) >> level;
}

inline const float * MipMapFlt::use_table (int table) const
{
    assert (is_ready());
    assert (table >= 0 && table < _nbr_tables);
    return _table_arr[table]._data_ptr;
}

inline void MipMapFlt::resize_and_clear_tables ()
{
    _table_arr.resize(_nbr_tables);
    for (int i = 0; i < _nbr_tables; ++i)
    {
        const long lev_len = get_lev_len(i);
        const long tbl_len = _add_len_pre + lev_len + _add_len_post;
        TableData &tbl = _table_arr[i];
        SplData(tbl_len, 0.0f).swap(tbl._data);
        tbl._data_ptr = &tbl._data[_add_len_pre];
    }
}

inline bool MipMapFlt::check_sample_and_build_mip_map ()
{
    if (_filled_len == _len)
    {
        for (int lvl = 1; lvl < _nbr_tables; ++lvl)
        {
            build_mip_map_level(lvl);
        }
        SplData().swap(_filter);
    }
    return (_filled_len < _len);
}

inline void MipMapFlt::build_mip_map_level (int level)
{
    assert (level > 0 && level < _nbr_tables);
    SplData &ref = _table_arr[level - 1]._data;
    SplData &dst = _table_arr[level]._data;

    const long half = _filter.size() - 1;
    const long quarter = half / 2;
    const long end_pos = get_lev_len(level) + quarter;

    for (long pos = -quarter; pos < end_pos; ++pos)
    {
        const long ref_pos = _add_len_pre + pos * 2;
        float val = filter_sample(ref, ref_pos);
        const long dst_pos = _add_len_pre + pos;
        dst[dst_pos] = val;
    }
}

inline float MipMapFlt::filter_sample (const TableData::SplData &tbl, long pos) const
{
    const long half = _filter.size() - 1;
    assert (pos - half >= 0 && pos + half < static_cast<long>(tbl.size()));

    float sum = tbl[pos] * _filter[0];
    for (long i = 1; i <= half; ++i)
    {
        float s2 = tbl[pos - i] + tbl[pos + i];
        sum += s2 * _filter[i];
    }
    return sum;
}

} // namespace rspl
