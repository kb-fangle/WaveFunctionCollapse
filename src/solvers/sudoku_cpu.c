#include <stdio.h>
#include <stdlib.h>
#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"

#include <omp.h>

bool
solve_cpu(wfc_blocks_ptr blocks)
{
    uint64_t iteration  = 0;

    position min_entropy_loc;
    bool changed = true;

    while (changed && (min_entropy_loc = grd_min_entropy(blocks)).x != UINT32_MAX) {
        const uint32_t gx = min_entropy_loc.gx;
        const uint32_t gy = min_entropy_loc.gy;
        const uint32_t x = min_entropy_loc.x;
        const uint32_t y = min_entropy_loc.y;

        uint64_t* collapsed = blk_at(blocks, gx, gy, x, y);
        *collapsed = entropy_collapse_state(*collapsed, gx, gy, x, y, blocks->seed, iteration);

        changed = propagate(blocks, gx, gy, x, y);

        if (!check_grid(blocks)) {
            return false;
        }

        iteration++;
    }
    
    return changed;
}
