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

        // Find minimum entropy
        entropy_location min_entropy_loc;
        min_entropy_loc.entropy = blocks->block_side * blocks->block_side;
        entropy_location entropy_loc;

        for (int gy=0; gy < blocks->grid_side; gy++){
            for (int gx=0; gx < blocks->grid_side; gx++){
                entropy_loc = blk_min_entropy(blocks,gx,gy);
                if (entropy_loc.entropy < min_entropy_loc.entropy){
                    min_entropy_loc = entropy_loc;
                }
            }
        } 

        // 2. Propagate
        // 3. Check Error

        iteration += 1;
        if (!changed)
            break;
    }

    // return false;
    return true;
}
