#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"

#include <omp.h>

bool
solve_openmp(wfc_blocks_ptr blocks)
{

    uint64_t iteration  = 0;

    entropy_location min_entropy_loc;
    bool changed = true;

    while (changed && (min_entropy_loc = grd_min_entropy_omp(blocks)).location.x != UINT32_MAX) {
        const uint32_t gx = min_entropy_loc.location.gx;
        const uint32_t gy = min_entropy_loc.location.gy;
        const uint32_t x = min_entropy_loc.location.x;
        const uint32_t y = min_entropy_loc.location.y;

        uint64_t* collapsed = blk_at(blocks, gx, gy, x, y);
        *collapsed = entropy_collapse_state(*collapsed, gx, gy, x, y, blocks->seed, iteration);
        
        changed = propagate_omp(blocks, gx, gy, x, y);

        if (!check_grid_omp(blocks)) {
            fprintf(stderr, "The grid is in an invalid state\n");
            exit(EXIT_FAILURE);
        }

        iteration++;
    }
    
    return changed;
}
