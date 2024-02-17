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
    // const uint64_t seed = blocks->seed;
    // struct {
    //     uint32_t gy, x, y, _1;
    //     uint64_t state;
    // } row_changes[blocks->grid_side];

    entropy_location min_entropy_loc;
    bool changed = true;

    while (changed && (min_entropy_loc = grd_min_entropy(blocks)).location.x != UINT32_MAX) {
        const uint32_t gx = min_entropy_loc.location.gx;
        const uint32_t gy = min_entropy_loc.location.gy;
        const uint32_t x = min_entropy_loc.location.x;
        const uint32_t y = min_entropy_loc.location.y;

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
