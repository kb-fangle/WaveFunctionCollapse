#pragma once

#include <stdbool.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "position_list.h"

// Entropy functions
entropy_location grd_min_entropy_omp(const wfc_blocks_ptr blocks);
///
/// @group Propagtion functions
///
/// @param blocks The grid
/// @param collapsed The state to collapse
/// @param position_list A list containing the position of the states resolved after collapsing.
///
/// @return true if at least one of the states changed, false otherwise.
///
/// @{

/// Propagate a collapsed state in a block.
bool blk_propagate_omp(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint64_t collapsed, position_list* collapsed_stack);
/// Propagate a collapsed state in a column.
bool grd_propagate_column_omp(wfc_blocks_ptr blocks, uint32_t gx, uint32_t x, uint64_t collapsed, position_list* collapsed_stack);
/// Propagate a collapsed state in a row.
bool grd_propagate_row_omp(wfc_blocks_ptr blocks, uint32_t gy, uint32_t y, uint64_t collapsed, position_list* collapsed_stack);
/// Propagate a collapsed state in the grid.
bool propagate_omp(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y);

/// }@


// Check functions

///
/// @brief Checks the grid for errors (duplicate values in a block, row or column)
///
/// @param blocks The grid
///
/// @return true if the grid is valid, false otherwise
///
bool check_grid_omp(const wfc_blocks_ptr blocks);
bool grd_check_block_errors_omp(wfc_blocks_ptr blocks);
bool grd_check_row_errors_omp(wfc_blocks_ptr blocks);
bool grd_check_column_errors_omp(wfc_blocks_ptr blocks);


