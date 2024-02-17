#include <stdint.h>
#define _GNU_SOURCE

#include "wfc.h"
#include "wfc_omp.h"
#include "bitfield.h"
#include "md5.h"
#include "types.h"
#include "position_list.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <limits.h>
#include <errno.h>
#include <string.h>
#include <strings.h>


entropy_location 
grd_min_entropy_omp(const wfc_blocks_ptr blocks)
{
    entropy_location min_entropy_loc = {
        .entropy = UINT8_MAX,
        .location = { .x = UINT32_MAX, .y = UINT32_MAX }
    };

    #pragma omp parallel for shared(min_entropy_loc) schedule(static,blocks->grid_side)
    for (uint32_t gy=0; gy < blocks->grid_side; gy++){

        #pragma omp parallel for shared(min_entropy_loc) schedule(static,blocks->grid_side)
        for (uint32_t gx=0; gx < blocks->grid_side; gx++){
            entropy_location block_entropy = blk_min_entropy(blocks,gx,gy);
            if (compare_grd_entropy_locs(&min_entropy_loc, &block_entropy)){
                min_entropy_loc = block_entropy;
            }
        }
    }

    return min_entropy_loc;
}

bool
grd_check_block_errors_omp(wfc_blocks_ptr blocks) {
    const uint32_t block_size = blocks->block_side * blocks->block_side;
    bool err_blk = true;
    #pragma omp parallel for
    for (uint32_t gy = 0; gy < blocks->grid_side; gy++) {
        
        #pragma omp parallel for
        for (uint32_t gx = 0; gx < blocks->grid_side; gx++) {
            uint64_t* blk = grd_at(blocks, gx, gy);

            // mask composed of all collapsed states
            uint64_t collapsed_mask = 0;
            for (uint32_t i = 0; i < block_size; i++) {
                if (blk[i] == 0 || (blk[i] & collapsed_mask) != 0) {
                    err_blk = false;
                    // return false;
                }

                if (entropy_compute(blk[i]) == 1) {
                    collapsed_mask |= blk[i];
                }
            }
        }
    }

    return err_blk;
}

// Check for duplicate values in all rows of the grid

bool
grd_check_row_errors_omp(wfc_blocks_ptr blocks) {
    bool err_row = true;
    #pragma omp parallel for
    for (uint32_t gy = 0; gy < blocks->grid_side; gy++) {
        
        #pragma omp parallel for
        for (uint32_t y = 0; y < blocks->block_side; y++) {
            uint64_t collapsed_mask = 0;

            for (uint32_t gx = 0; gx < blocks->grid_side; gx++) {
                uint64_t* row = blk_at(blocks, gx, gy, 0, y);

                for (uint32_t x = 0; x < blocks->block_side; x++) {
                    if (row[x] == 0 || (row[x] & collapsed_mask) != 0) {
                        // return false;
                        err_row = false;
                    }

                    if (entropy_compute(row[x]) == 1) {
                        collapsed_mask |= row[x];
                    }
                }
            }
        }
    }

    return err_row;
}

// Check for duplicate values in all columns of the grid

bool
grd_check_column_errors_omp(wfc_blocks_ptr blocks) {
    
    bool err_col = true;
    #pragma omp parallel for
    for (uint32_t gx = 0; gx < blocks->grid_side; gx++) {
        
        #pragma omp parallel for
        for (uint32_t x = 0; x < blocks->block_side; x++) {
            uint64_t collapsed_mask = 0;
        
            for (uint32_t gy = 0; gy < blocks->grid_side; gy++) {
                uint64_t* col = blk_at(blocks, gx, gy, x, 0);

                for (uint32_t y = 0; y < blocks->block_side; y++) {
                    const uint64_t state = col[y * blocks->block_side];
                    if (state == 0 || (state & collapsed_mask) != 0) {
                        // return false;
                        err_col = false;
                    }

                    if (entropy_compute(state) == 1) {
                        collapsed_mask |= state;
                    }
                }
            }
        }
    }

    return err_col;
}

bool
check_grid_omp(const wfc_blocks_ptr blocks)
{
    bool err_blk = true;
    bool err_row = true;
    bool err_col = true;
    #pragma omp parallel sections
    {
        #pragma omp section
        err_blk = grd_check_block_errors_omp(blocks);
        #pragma omp section
        err_row = grd_check_row_errors_omp(blocks);
        #pragma omp section
        err_col = grd_check_column_errors_omp(blocks);
    }
    return err_blk && err_row && err_col;
}

bool
blk_propagate_omp(wfc_blocks_ptr blocks,
              uint32_t gx, uint32_t gy,
              uint64_t collapsed, position_list* collapsed_stack)
{
    uint64_t* block_loc = grd_at(blocks,gx,gy);
    bool changed = false;

    #pragma omp parallel for shared(collapsed_stack)
    for (uint32_t i = 0; i < blocks->block_side * blocks->block_side; i++) {
        const uint64_t new_state = block_loc[i] & ~collapsed;

        changed |= new_state != 0 && new_state != block_loc[i];

        if (new_state != block_loc[i] && bitfield_count(new_state) == 1) {
            position pos = { gx, gy, i % blocks->block_side, i / blocks->block_side };
            position_list_push(collapsed_stack, pos);
        }
        block_loc[i] = new_state;
    }

    return changed;
}

bool
grd_propagate_row_omp(wfc_blocks_ptr blocks, uint32_t gy, uint32_t y,
                  uint64_t collapsed, position_list* collapsed_stack)
{
    uint64_t* row = 0;
    bool changed = false;

    #pragma omp parallel for 
    for (uint32_t gx=0; gx < blocks->grid_side;gx++){
        row = blk_at(blocks,gx,gy,0,y);
        #pragma omp parallel for shared(collapsed_stack)
        for (uint32_t x=0; x < blocks->block_side;x++){
            const uint64_t new_state = row[x] & ~collapsed;
            
            changed |= new_state != 0 && new_state != row[x];

            if (new_state != row[x] && bitfield_count(new_state) == 1) {
                position pos = { gx, gy, x, y };
                position_list_push(collapsed_stack, pos);
            }

            row[x] = new_state;
        }
    }

    return changed;
}

bool
grd_propagate_column_omp(wfc_blocks_ptr blocks, uint32_t gx,
                     uint32_t x, uint64_t collapsed, position_list* collapsed_stack)
{
    uint64_t* col = 0;
    bool changed = false;

    #pragma omp parallel for
    for (uint32_t gy=0; gy < blocks->grid_side;gy++){
        col = blk_at(blocks,gx,gy,x,0);
        #pragma omp parallel for shared(collapsed_stack)
        for (uint32_t y=0; y < blocks->block_side;y++){
            const uint32_t index = y * blocks->block_side;
            const uint64_t new_state = col[index] & ~collapsed;

            changed |= new_state != 0 && new_state != col[index];

            if (new_state != col[index] && bitfield_count(new_state) == 1) {
                position pos = { gx, gy, x, y };
                position_list_push(collapsed_stack, pos);
            }

            col[index] = new_state;
        }
    }

    return changed;
}

bool
propagate_omp(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y) {
    position_list collapsed_stack = position_list_init();

    position collapsed_pos = { gx, gy, x, y };
    position_list_push(&collapsed_stack, collapsed_pos);

    bool changed = false;

    while (!position_list_is_empty(&collapsed_stack)) {
        const position pos = position_list_pop(&collapsed_stack);
        
        uint64_t* collapsed_cell = blk_at(blocks, pos.gx, pos.gy, pos.x, pos.y);
        const uint64_t collapsed = *collapsed_cell;

        #pragma omp parallel sections shared(changed)
        {
            #pragma omp section
            changed |= blk_propagate_omp(blocks, pos.gx, pos.gy, collapsed, &collapsed_stack);
            #pragma omp section
            changed |= grd_propagate_row_omp(blocks, pos.gy, pos.y, collapsed, &collapsed_stack);
            #pragma omp section
            changed |= grd_propagate_column_omp(blocks, pos.gx, pos.x, collapsed, &collapsed_stack);
        }

        // The propagate functions will overwrite the collapsed state, so we
        // reset it to the right value
        *collapsed_cell = collapsed;
    }

    return changed;
}
