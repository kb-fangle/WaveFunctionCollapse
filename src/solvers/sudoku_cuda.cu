#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"

#if !defined(WFC_CUDA)
#error "WDC_CUDA should be defined..."
#endif

bool
solve_cuda(wfc_blocks_ptr blocks)
{
    return false;
}
