// Copyright AStarship <https://astarship.net>.
#include "../CrabsTK/_Seams.h"

// SubsecondDb seams continue CrabsTK's counter. CrabsTK ends at SEAM_N == 51
// (see CrabsTK/_Seams.h); pin that as a literal base so the new macros do not
// self-reference the SEAM_N we redefine below (a macro alias to SEAM_N would
// recurse once SEAM_N is redefined to point back at it).
static_assert(SEAM_N == 51, "CrabsTK SEAM_N moved; update SUBSECOND_DB_SEAM_BASE");
#define SUBSECOND_DB_SEAM_BASE 51
#undef SEAM_N

#define SUBSECOND_DB_FOO SUBSECOND_DB_SEAM_BASE + 1
#define SUBSECOND_DB_BAR SUBSECOND_DB_SEAM_BASE + 2
#define SEAM_N SUBSECOND_DB_BAR
