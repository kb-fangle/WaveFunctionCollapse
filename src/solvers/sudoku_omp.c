#define _GNU_SOURCE

#include <omp.h>

#include "md5.h"
#include "wfc.h"
#include "wfc_omp.h"

bool
solve_openmp(wfc_blocks_ptr blocks)
{
    uint64_t iteration = 0;

    bool changed = true;

    bool has_min_entropy = true;
    bool valid = true;

    position_list collapsed_stack = position_list_init();
    omp_lock_t stack_lock;
    omp_init_lock(&stack_lock);

    const uint32_t sudoku_size = (uint32_t)blocks->block_side * blocks->block_side
                                 * (uint32_t)blocks->grid_side * blocks->grid_side;

    while (changed && has_min_entropy && valid) {
#pragma omp parallel default(shared) firstprivate(sudoku_size)
        {
            // Compute the min entropy location
#pragma omp for
            for (uint32_t i = 0; i < sudoku_size; i++) {
                blocks->entropies[i] = entropy_compute(blocks->states[i]);
            }

#pragma omp single
            {
                uint8_t min_entropy = UINT8_MAX;
                uint32_t nb_min_entropy = 0;

                for (uint32_t i = 0; i < sudoku_size; i++) {
                    if (blocks->entropies[i] > 1 && blocks->entropies[i] < min_entropy)
                    {
                        min_entropy = blocks->entropies[i];
                        nb_min_entropy = 0;
                    }

                    nb_min_entropy += blocks->entropies[i] == min_entropy;
                }

                has_min_entropy = min_entropy != UINT8_MAX;

                if (has_min_entropy) {
                    // choose the collapsed cell at random
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

                    md5((uint8_t *)&random_state, sizeof(random_state),
                        (uint8_t *)digest);
                    uint32_t id = digest[1] % nb_min_entropy;

                    position collapsed_pos;
                    for (uint32_t i = 0; i < sudoku_size; i++) {
                        if (blocks->entropies[i] == min_entropy) {
                            if (id == 0) {
                                collapsed_pos = position_at(blocks, i);
                                break;
                            } else {
                                id--;
                            }
                        }
                    }

                    const uint32_t gx = collapsed_pos.gx;
                    const uint32_t gy = collapsed_pos.gy;
                    const uint32_t x = collapsed_pos.x;
                    const uint32_t y = collapsed_pos.y;

                    position_list_push(&collapsed_stack, collapsed_pos);

                    uint64_t *collapsed_cell = blk_at(blocks, gx, gy, x, y);
                    *collapsed_cell = entropy_collapse_state(
                        *collapsed_cell, gx, gy, x, y, blocks->seed, iteration);

                    // propagate
                    changed = false;

                    while (!position_list_is_empty(&collapsed_stack)) {
                        collapsed_pos = position_list_pop(&collapsed_stack);

                        collapsed_cell =
                            blk_at(blocks, collapsed_pos.gx, collapsed_pos.gy,
                                   collapsed_pos.x, collapsed_pos.y);
                        const uint64_t collapsed = *collapsed_cell;

                        bool changed_block, changed_row, changed_col;

#pragma omp task shared(changed_block, blocks)
                        changed_block = blk_propagate_omp(
                            blocks, collapsed_pos.gx, collapsed_pos.gy, collapsed,
                            &collapsed_stack, &stack_lock);
#pragma omp task shared(changed_row, blocks)
                        changed_row =
                            grd_propagate_row_omp(blocks, collapsed_pos, collapsed,
                                                  &collapsed_stack, &stack_lock);
#pragma omp task shared(changed_col, blocks)
                        changed_col =
                            grd_propagate_column_omp(blocks, collapsed_pos, collapsed,
                                                     &collapsed_stack, &stack_lock);
#pragma omp taskwait

                        changed =
                            changed || changed_block || changed_row || changed_col;

                        // The propagate functions will overwrite the collapsed state,
                        // so we reset it to the right value
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
        }     // end parallel

        iteration++;
    }

    omp_destroy_lock(&stack_lock);

    return changed && !has_min_entropy && valid;
}
