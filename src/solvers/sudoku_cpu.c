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
    while ((min_entropy_loc = grd_min_entropy(blocks)).location.x != UINT32_MAX) {
        uint32_t gx = min_entropy_loc.location.x / blocks->block_side;
        uint32_t gy = min_entropy_loc.location.y / blocks->block_side;
        uint32_t x = min_entropy_loc.location.x % blocks->block_side;
        uint32_t y = min_entropy_loc.location.y % blocks->block_side;

        uint64_t* collapsed = blk_at(blocks, gx, gy, x, y);
        *collapsed = entropy_collapse_state(*collapsed, gx, gy, x, y, blocks->seed, iteration);

        propagate(blocks, gx, gy, x, y);

        if (!check_grid(blocks)) {
            fprintf(stderr, "The grid is in an invalid state\n");
            exit(EXIT_FAILURE);
        }

        iteration++;
    }

    blk_print(stderr, blocks, 0, 0);

    // forever {
    //     bool changed = false;
    //     // 1. Collapse
    //
    //     // Find minimum entropy
    //     entropy_location min_entropy_loc;
    //     min_entropy_loc.entropy = blocks->block_side * blocks->block_side;
    //     entropy_location entropy_loc;
    //
    //     for (uint32_t gy=0; gy < blocks->grid_side; gy++){
    //         for (uint32_t gx=0; gx < blocks->grid_side; gx++){
    //             entropy_loc = blk_min_entropy(blocks,gx,gy);
    //             if (entropy_loc.entropy < min_entropy_loc.entropy){
    //                 min_entropy_loc = entropy_loc;
    //             }
    //         }
    //     } 
    //
    //     // 2. Propagate
    //     // 3. Check Error
    //
    //     iteration += 1;
    //     if (!changed)
    //         break;
    // }

    // return false;
    return true;
}
