#define _GNU_SOURCE

#include <stdint.h>

#include "bitfield.h"
#include "position_list.h"
#include "types.h"
#include "wfc.h"
#include "wfc_omp.h"

bool
grd_check_block_errors_omp(wfc_blocks_ptr blocks)
{
    const uint32_t block_size = blocks->block_side * blocks->block_side;
    for (uint32_t gy = 0; gy < blocks->grid_side; gy++) {
        for (uint32_t gx = 0; gx < blocks->grid_side; gx++) {
            uint64_t *blk = grd_at(blocks, gx, gy);

            // mask composed of all collapsed states
            uint64_t collapsed_mask = 0;
            for (uint32_t i = 0; i < block_size; i++) {
                if (blk[i] == 0 || (blk[i] & collapsed_mask) != 0) {
                    return false;
                }

                if (entropy_compute(blk[i]) == 1) {
                    collapsed_mask |= blk[i];
                }
            }
        }
    }

    return true;
}

bool
grd_check_row_errors_omp(wfc_blocks_ptr blocks)
{
    for (uint32_t gy = 0; gy < blocks->grid_side; gy++) {
        for (uint32_t y = 0; y < blocks->block_side; y++) {
            uint64_t collapsed_mask = 0;

            for (uint32_t gx = 0; gx < blocks->grid_side; gx++) {
                uint64_t *row = blk_at(blocks, gx, gy, 0, y);

                for (uint32_t x = 0; x < blocks->block_side; x++) {
                    if (row[x] == 0 || (row[x] & collapsed_mask) != 0) {
                        return false;
                    }

                    if (entropy_compute(row[x]) == 1) {
                        collapsed_mask |= row[x];
                    }
                }
            }
        }
    }

    return true;
}

bool
grd_check_column_errors_omp(wfc_blocks_ptr blocks)
{
    for (uint32_t gx = 0; gx < blocks->grid_side; gx++) {
        for (uint32_t x = 0; x < blocks->block_side; x++) {
            uint64_t collapsed_mask = 0;

            for (uint32_t gy = 0; gy < blocks->grid_side; gy++) {
                uint64_t *col = blk_at(blocks, gx, gy, x, 0);

                for (uint32_t y = 0; y < blocks->block_side; y++) {
                    const uint64_t state = col[y * blocks->block_side];
                    if (state == 0 || (state & collapsed_mask) != 0) {
                        return false;
                    }

                    if (entropy_compute(state) == 1) {
                        collapsed_mask |= state;
                    }
                }
            }
        }
    }

    return true;
}

bool
blk_propagate_omp(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint64_t collapsed,
                  position_list *collapsed_stack, omp_lock_t *stack_lock)
{
    uint64_t *block_loc = grd_at(blocks, gx, gy);
    bool changed = false;

    for (uint32_t i = 0; i < blocks->block_side * blocks->block_side; i++) {
        const uint64_t new_state = block_loc[i] & ~collapsed;

        changed |= new_state != 0 && new_state != block_loc[i];

        if (new_state != block_loc[i] && bitfield_count(new_state) == 1) {
            position pos = { gx, gy, i % blocks->block_side, i / blocks->block_side };
            omp_set_lock(stack_lock);
            position_list_push(collapsed_stack, pos);
            omp_unset_lock(stack_lock);
        }
        block_loc[i] = new_state;
    }

    return changed;
}

bool
grd_propagate_row_omp(wfc_blocks_ptr blocks, position loc, uint64_t collapsed,
                      position_list *collapsed_stack, omp_lock_t *stack_lock)
{
    uint64_t *row = 0;
    bool changed = false;

    for (uint32_t gx = 0; gx < blocks->grid_side; gx++) {
        if (gx == loc.gx) {
            continue;
        }
        row = blk_at(blocks, gx, loc.gy, 0, loc.y);
        for (uint32_t x = 0; x < blocks->block_side; x++) {
            const uint64_t new_state = row[x] & ~collapsed;

            changed |= new_state != 0 && new_state != row[x];

            if (new_state != row[x] && bitfield_count(new_state) == 1) {
                position pos = { gx, loc.gy, x, loc.y };
                omp_set_lock(stack_lock);
                position_list_push(collapsed_stack, pos);
                omp_unset_lock(stack_lock);
            }

            row[x] = new_state;
        }
    }

    return changed;
}

bool
grd_propagate_column_omp(wfc_blocks_ptr blocks, position loc, uint64_t collapsed,
                         position_list *collapsed_stack, omp_lock_t *stack_lock)
{
    uint64_t *col = 0;
    bool changed = false;

    for (uint32_t gy = 0; gy < blocks->grid_side; gy++) {
        if (gy == loc.gy) {
            continue;
        }
        col = blk_at(blocks, loc.gx, gy, loc.x, 0);
        for (uint32_t y = 0; y < blocks->block_side; y++) {
            const uint32_t index = y * blocks->block_side;
            const uint64_t new_state = col[index] & ~collapsed;

            changed |= new_state != 0 && new_state != col[index];

            if (new_state != col[index] && bitfield_count(new_state) == 1) {
                position pos = { loc.gx, gy, loc.x, y };
                omp_set_lock(stack_lock);
                position_list_push(collapsed_stack, pos);
                omp_unset_lock(stack_lock);
            }

            col[index] = new_state;
        }
    }

    return changed;
}
