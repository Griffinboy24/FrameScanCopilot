/*****************************************************************************

This is a private class used by ResamplerFlt to pass voice data to InterpPack
for the rendering.

*Tab=3***********************************************************************/



#if ! defined (rspl_BaseVoiceState_HEADER_INCLUDED)
#define	rspl_BaseVoiceState_HEADER_INCLUDED

#if defined (_MSC_VER)
	#pragma once
	#pragma warning (4 : 4250) // "Inherits via dominance."
#endif



/*\\\ INCLUDE FILES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

#include	"rspl.hpp"



namespace rspl
{



class BaseVoiceState
{

/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

public:

	enum {			NBR_BITS_PER_OCT	= 16	};
	enum {			FADE_LEN				= 64	};

						BaseVoiceState ();
	BaseVoiceState &
						operator = (const BaseVoiceState &other);

	void				compute_step (long pitch);

	Fixed3232		_pos;			// Position in the current MIP-map level
	Fixed3232		_step;		// Step in the current MIP-map level
	const float *	_table_ptr;
	long				_table_len;
	int				_table;
	bool				_ovrspl_flag;



/*\\\ PROTECTED \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

protected:



/*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

private:



/*\\\ FORBIDDEN MEMBER FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

private:

						BaseVoiceState (const BaseVoiceState &other);
	bool				operator == (const BaseVoiceState &other);
	bool				operator != (const BaseVoiceState &other);

};	// class BaseVoiceState



}	// namespace rspl



//#include	"rspl/BaseVoiceState.hpp"



#endif	// rspl_BaseVoiceState_HEADER_INCLUDED



/*\\\ EOF \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
