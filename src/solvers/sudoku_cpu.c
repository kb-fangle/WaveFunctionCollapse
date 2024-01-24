#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"

#include <omp.h>

bool
solve_cpu(wfc_blocks_ptr blocks)
{
    uint64_t iteration  = 0;
    const uint64_t seed = blocks->states[0];
    struct {
        uint32_t gy, x, y, _1;
        uint64_t state;
    } row_changes[blocks->grid_side];

    forever {
        bool changed = false;
        // 1. Collapse
        // 2. Propagate
        // 3. Check Error

        iteration += 1;
        if (!changed)
            break;
    }

    return false;
}
