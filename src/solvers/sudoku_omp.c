#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"
#include "wfc_omp.h"

#include <omp.h>


/// Entropy location with a field for the cell's address
typedef struct {
    uint64_t* cell;
    uint8_t entropy;
    position loc;
} entropy_location_alt;

/// stores in `out` the minimum entropy location between `out` and `in`
/// Because the cells are laid out blocks by blocks in memory, we can directly
/// order them using their address
void entropy_min(entropy_location_alt* out, entropy_location_alt* in) {
    if (in->entropy > 1 && (in->entropy < out->entropy || (in->entropy == out->entropy && in->cell < out->cell))) {
        *out = *in;
    }
}

bool
solve_openmp(wfc_blocks_ptr blocks)
{
    #pragma omp declare reduction(min : entropy_location_alt : entropy_min(&omp_out, &omp_in))\
                        initializer(omp_priv = { (uint64_t*)UINTPTR_MAX, UINT8_MAX, { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX } })

    uint64_t iteration  = 0;

    bool changed = true;

    bool has_min_entropy = true;
    bool valid = true;

    position_list collapsed_stack = position_list_init();
    omp_lock_t stack_lock;
    omp_init_lock(&stack_lock);

    while (changed && has_min_entropy && valid) {
        // Compute the min entropy location
        entropy_location_alt min_entropy = { (uint64_t*)UINTPTR_MAX, UINT8_MAX, { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX } };

        #pragma omp parallel default(shared)
        {
        #pragma omp for collapse(4) reduction(min: min_entropy)
        for (uint32_t gy=0; gy < blocks->grid_side; gy++){
            for (uint32_t gx=0; gx < blocks->grid_side; gx++){
                for (uint32_t y = 0; y < blocks->block_side; y++) {
                    for (uint32_t x = 0; x < blocks->block_side; x++) {
                        uint64_t* cell = blk_at(blocks, gx, gy, x, y);
                        entropy_location_alt loc = {
                                cell,
                                entropy_compute(*cell),
                                { gx, gy, x, y },
                        };
                        entropy_min(&min_entropy, &loc);
                    }
                }
            }
        }

    
        #pragma omp single
        {
            has_min_entropy = min_entropy.loc.x != UINT32_MAX;

            if (has_min_entropy) {
                // propagate
                const uint32_t gx = min_entropy.loc.gx;
                const uint32_t gy = min_entropy.loc.gy;
                const uint32_t x = min_entropy.loc.x;
                const uint32_t y = min_entropy.loc.y;

                position collapsed_pos = { gx, gy, x, y };
                position_list_push(&collapsed_stack, collapsed_pos);
                
                uint64_t* collapsed_cell = blk_at(blocks, gx, gy, x, y);
                *collapsed_cell = entropy_collapse_state(*collapsed_cell, gx, gy, x, y, blocks->seed, iteration);
                
                changed = false;

                while (!position_list_is_empty(&collapsed_stack)) {
                    collapsed_pos = position_list_pop(&collapsed_stack);
                    
                    uint64_t* collapsed_cell = blk_at(blocks, collapsed_pos.gx, collapsed_pos.gy, collapsed_pos.x, collapsed_pos.y);
                    const uint64_t collapsed = *collapsed_cell;
                    
                    bool changed_block, changed_row, changed_col;
                    #pragma omp task shared(changed_block, blocks)
                    changed_block = blk_propagate_omp(blocks, collapsed_pos.gx, collapsed_pos.gy, collapsed, &collapsed_stack, &stack_lock);
                    #pragma omp task shared(changed_row, blocks)
                    changed_row = grd_propagate_row_omp(blocks, collapsed_pos, collapsed, &collapsed_stack, &stack_lock);
                    #pragma omp task shared(changed_col, blocks)
                    changed_col = grd_propagate_column_omp(blocks, collapsed_pos, collapsed, &collapsed_stack, &stack_lock);
                    #pragma omp taskwait

                    changed = changed || changed_block || changed_row || changed_col;

                    // The propagate functions will overwrite the collapsed state, so we
                    // reset it to the right value
                    *collapsed_cell = collapsed;
                }

                // verify
                bool valid_block, valid_row, valid_col;
                #pragma omp task shared(valid_block)
                valid_block = grd_check_block_errors_omp(blocks);
                #pragma omp task shared(valid_row)
                valid_row = grd_check_row_errors_omp(blocks);
                #pragma omp task shared(valid_col)
                valid_col = grd_check_column_errors_omp(blocks);
                #pragma omp taskwait

                valid = valid_block && valid_row && valid_col;
            }
        } // end single
        } // end parallel

        iteration++;
    }

    omp_destroy_lock(&stack_lock);
    
    return changed && !has_min_entropy && valid;
}
