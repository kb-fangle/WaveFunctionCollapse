#define _GNU_SOURCE

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitfield.h"
#include "md5.h"
#include "position_list.h"
#include "types.h"
#include "wfc.h"

uint64_t
entropy_collapse_state(uint64_t state, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                       uint64_t seed, uint64_t iteration)
{
    uint8_t digest[16] = { 0 };
    struct {
        uint32_t gx, gy, x, y;
        uint64_t seed, iteration;
    } random_state = {
        .gx = gx,
        .gy = gy,
        .x = x,
        .y = y,
        .seed = seed,
        .iteration = iteration,
    };

    md5((uint8_t *)&random_state, sizeof(random_state), digest);

    uint8_t entropy = bitfield_count(state);
    uint8_t collapse_index = (uint8_t)(*((uint64_t *)digest) % entropy + 1);
    uint32_t real_index = bitfield_get_nth_set_bit(state, collapse_index);

    state = bitfield_set(0, (uint8_t)real_index - 1);

    return state;
}

void
wfc_clone_into(wfc_blocks_ptr *const restrict ret_ptr, uint64_t seed,
               const wfc_blocks_ptr blocks)
{
    const uint64_t grid_size = blocks->grid_side;
    const uint64_t block_size = blocks->block_side;
    wfc_blocks_ptr ret = *ret_ptr;

    const uint64_t size =
        (grid_size * grid_size * block_size * block_size * sizeof(uint64_t))
        + sizeof(wfc_blocks);

    if (NULL == ret) {
        if (NULL == (ret = malloc(size))) {
            fprintf(stderr, "failed to clone blocks structure\n");
            exit(EXIT_FAILURE);
        }
    } else if (grid_size != ret->grid_side || block_size != ret->block_side) {
        fprintf(stderr, "size mismatch!\n");
        exit(EXIT_FAILURE);
    }

    memcpy(ret, blocks, size);
    ret->seed = seed;
    *ret_ptr = ret;
}

void
blk_print(FILE *const file, const wfc_blocks_ptr block, uint32_t gx, uint32_t gy)
{
    uint64_t *block_ptr = grd_at(block, gx, gy);
    for (uint8_t y = 0; y < block->block_side; y++) {
        for (uint8_t x = 0; x < block->block_side; x++) {
            const uint64_t state = block_ptr[y * block->block_side + x];
            switch (entropy_compute(state)) {
            case 0:
                fprintf(file, " . ");
                break;
            case 1:
                fprintf(file, "%2d ", bitfield_get_nth_set_bit(state, 1));
                break;
            default:
                fprintf(file, " ? ");
                break;
            }
        }
        fprintf(file, "\n");
    }
}

void
grd_print(FILE *const file, const wfc_blocks_ptr blocks)
{
    for (uint8_t gy = 0; gy < blocks->grid_side; gy++) {
        for (uint8_t y = 0; y < blocks->block_side; y++) {
            for (uint8_t gx = 0; gx < blocks->grid_side; gx++) {
                uint64_t *row = blk_at(blocks, gx, gy, 0, y);
                for (uint8_t x = 0; x < blocks->block_side; x++) {
                    const uint64_t state = row[x];
                    const uint8_t entropy = entropy_compute(state);
                    switch (entropy) {
                    case 0:
                        fprintf(file, " . ");
                        break;
                    case 1:
                        fprintf(file, "%2d ", bitfield_get_nth_set_bit(state, 1));
                        break;
                    default:
                        fprintf(file, " ? ");
                        break;
                    }
                }
                fprintf(file, gx < blocks->grid_side - 1 ? " | " : "\n");
            }
        }
        if (gy < blocks->grid_side - 1) {
            for (uint8_t gx = 0; gx < blocks->grid_side; gx++) {
                for (uint8_t i = 0; i < 3 * blocks->block_side; i++) {
                    putc('-', file);
                }
                fprintf(file, gx < blocks->grid_side - 1 ? " + " : "\n");
            }
        }
    }
}

position
grd_min_entropy(const wfc_blocks_ptr blocks)
{
    position none = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };

    uint32_t sudoku_size = (uint32_t)blocks->block_side * blocks->block_side
                           * (uint32_t)blocks->grid_side * blocks->grid_side;

    uint8_t min_entropy = UINT8_MAX;
    uint32_t nb_min_entropy = 0;

    for (uint32_t i = 0; i < sudoku_size; i++) {
        blocks->entropies[i] = entropy_compute(blocks->states[i]);
    }

    for (uint32_t i = 0; i < sudoku_size; i++) {
        if (blocks->entropies[i] > 1 && blocks->entropies[i] < min_entropy) {
            min_entropy = blocks->entropies[i];
            nb_min_entropy = 0;
        }

        nb_min_entropy += blocks->entropies[i] == min_entropy;
    }

    if (min_entropy == UINT8_MAX) {
        return none;
    }

    uint32_t digest[4];
    struct {
        uint64_t seed;
        uint16_t a, b, c, d;
    } random_state = {
        blocks->seed,
        blocks->entropies[nb_min_entropy - 1],
        blocks->entropies[0],
        blocks->entropies[nb_min_entropy << 1],
        blocks->entropies[nb_min_entropy << 2],
    };

    md5((uint8_t *)&random_state, sizeof(random_state), (uint8_t *)digest);
    uint32_t id = digest[1] % nb_min_entropy;
    for (uint32_t i = 0; i < sudoku_size; i++) {
        if (blocks->entropies[i] == min_entropy) {
            if (id == 0) {
                return position_at(blocks, i);
            } else {
                id--;
            }
        }
    }

    return none;
}

// Check for duplicate values in all blocks of the grid
bool
grd_check_block_errors(wfc_blocks_ptr blocks)
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

// Check for duplicate values in all rows of the grid
bool
grd_check_row_errors(wfc_blocks_ptr blocks)
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

// Check for duplicate values in all columns of the grid
bool
grd_check_column_errors(wfc_blocks_ptr blocks)
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
check_grid(const wfc_blocks_ptr blocks)
{
    return grd_check_block_errors(blocks) && grd_check_row_errors(blocks)
           && grd_check_column_errors(blocks);
}

bool
blk_propagate(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint64_t collapsed,
              position_list *collapsed_stack)
{
    uint64_t *block_loc = grd_at(blocks, gx, gy);
    bool changed = false;

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
grd_propagate_row(wfc_blocks_ptr blocks, uint32_t gy, uint32_t y, uint64_t collapsed,
                  position_list *collapsed_stack)
{
    uint64_t *row = 0;
    bool changed = false;

    for (uint32_t gx = 0; gx < blocks->grid_side; gx++) {
        row = blk_at(blocks, gx, gy, 0, y);
        for (uint32_t x = 0; x < blocks->block_side; x++) {
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
grd_propagate_column(wfc_blocks_ptr blocks, uint32_t gx, uint32_t x, uint64_t collapsed,
                     position_list *collapsed_stack)
{
    uint64_t *col = 0;
    bool changed = false;

    for (uint32_t gy = 0; gy < blocks->grid_side; gy++) {
        col = blk_at(blocks, gx, gy, x, 0);
        for (uint32_t y = 0; y < blocks->block_side; y++) {
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
propagate(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y)
{
    position_list collapsed_stack = position_list_init();

    position collapsed_pos = { gx, gy, x, y };
    position_list_push(&collapsed_stack, collapsed_pos);

    bool changed = false;

    while (!position_list_is_empty(&collapsed_stack)) {
        const position pos = position_list_pop(&collapsed_stack);

        uint64_t *collapsed_cell = blk_at(blocks, pos.gx, pos.gy, pos.x, pos.y);
        const uint64_t collapsed = *collapsed_cell;

        changed |= blk_propagate(blocks, pos.gx, pos.gy, collapsed, &collapsed_stack);
        changed |=
            grd_propagate_row(blocks, pos.gy, pos.y, collapsed, &collapsed_stack);
        changed |=
            grd_propagate_column(blocks, pos.gx, pos.x, collapsed, &collapsed_stack);

        // The propagate functions will overwrite the collapsed state, so we
        // reset it to the right value
        *collapsed_cell = collapsed;
    }

    return changed;
}
