#pragma once

#include <omp.h>
#include <stdbool.h>
#include <stdint.h>

#include "position_list.h"
#include "types.h"

///
/// @group Propagtion functions
///
/// @param blocks The grid
/// @param collapsed The state to collapse
/// @param position_list A list containing the position of the states resolved after
/// collapsing.
///
/// @return true if at least one of the states changed, false otherwise.
///
/// @{

/// Propagate a collapsed state in a block.
bool blk_propagate_omp(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                       uint64_t collapsed, position_list *collapsed_stack,
                       omp_lock_t *stack_lock);
/// Propagate a collapsed state in a column.
bool grd_propagate_column_omp(wfc_blocks_ptr blocks, position loc, uint64_t collapsed,
                              position_list *collapsed_stack, omp_lock_t *stack_lock);
/// Propagate a collapsed state in a row.
bool grd_propagate_row_omp(wfc_blocks_ptr blocks, position loc, uint64_t collapsed,
                           position_list *collapsed_stack, omp_lock_t *stack_lock);
/// Propagate a collapsed state in the grid.

/// }@

// Check functions

///
/// @brief Checks the grid for errors (duplicate values in a block, row or column)
///
/// @param blocks The grid
///
/// @return true if the grid is valid, false otherwise
///
bool grd_check_block_errors_omp(wfc_blocks_ptr blocks);
bool grd_check_row_errors_omp(wfc_blocks_ptr blocks);
bool grd_check_column_errors_omp(wfc_blocks_ptr blocks);
