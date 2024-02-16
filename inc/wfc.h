#pragma once

#include "types.h"

#include "position_list.h"
#include <stdbool.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define forever for (;;)

/// Parses the arguments, prints the help message if needed and abort on error.
wfc_args wfc_parse_args(int argc, char **argv);

/// Get the next seed to try. If there are no more seeds to try, it will exit the process.
bool try_next_seed(seeds_list *restrict *const, uint64_t *restrict);

/// Count the total number of seeds.
uint64_t count_seeds(const seeds_list *restrict const);

/// Load the positions from a file. You must free the thing yourself. On error
/// kill the program.
wfc_blocks_ptr wfc_load(uint64_t, const char *);

/// Clone the blocks structure. You must free the return yourself.
void wfc_clone_into(wfc_blocks_ptr *const restrict, uint64_t, const wfc_blocks_ptr);

/// Save the grid to a folder by creating a new file or overwrite it, on error kills the program.
void wfc_save_into(const wfc_blocks_ptr, const char data[], const char folder[]);

static inline uint64_t
wfc_control_states_count(uint64_t grid_size, uint64_t block_size)
{
    return 0;
}

///
/// @brief Computes the address of the begining of a block
///
/// @param blocks The grid
/// @param gx The block's x position (col)
/// @param gy The block's y position (row)
///
static inline uint64_t *
grd_at(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy)
{
    uint32_t block_size = blocks->block_side * blocks->block_side;
    return &blocks->states[gy * blocks->grid_side * block_size + gx * block_size];
}


///
/// @brief Computes the address of a cell
///
/// @param blocks The grid
/// @param gx, gy The grid coordinates of the containing block
/// @param x, y The block coordinates of the cell
///
static inline uint64_t *
blk_at(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y)
{
    
    return grd_at(blocks,gx,gy) + y * blocks->block_side + x;
}


// Printing functions
void blk_print(FILE *const, const wfc_blocks_ptr block, uint32_t gx, uint32_t gy);
void grd_print(FILE *const, const wfc_blocks_ptr block);


// Entropy functions
entropy_location blk_min_entropy(const wfc_blocks_ptr block, uint32_t gx, uint32_t gy);
entropy_location grd_min_entropy(const wfc_blocks_ptr blocks);
entropy_location grd_min_entropy_omp(const wfc_blocks_ptr blocks);
uint8_t entropy_compute(uint64_t);
uint64_t entropy_collapse_state(uint64_t state, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y, uint64_t seed, uint64_t iteration);


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
bool blk_propagate(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint64_t collapsed, position_list* collapsed_stack);
bool blk_propagate_omp(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint64_t collapsed, position_list* collapsed_stack);
/// Propagate a collapsed state in a column.
bool grd_propagate_column(wfc_blocks_ptr blocks, uint32_t gx, uint32_t x, uint64_t collapsed, position_list* collapsed_stack);
bool grd_propagate_column_omp(wfc_blocks_ptr blocks, uint32_t gx, uint32_t x, uint64_t collapsed, position_list* collapsed_stack);
/// Propagate a collapsed state in a row.
bool grd_propagate_row(wfc_blocks_ptr blocks, uint32_t gy, uint32_t y, uint64_t collapsed, position_list* collapsed_stack);
bool grd_propagate_row_omp(wfc_blocks_ptr blocks, uint32_t gy, uint32_t y, uint64_t collapsed, position_list* collapsed_stack);
/// Propagate a collapsed state in the grid.
bool propagate(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y);
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
bool check_grid(const wfc_blocks_ptr blocks);
bool check_grid_omp(const wfc_blocks_ptr blocks);
// bool grd_check_error_in_column(wfc_blocks_ptr, uint32_t);

// Solvers
bool solve_cpu(wfc_blocks_ptr);
bool solve_openmp(wfc_blocks_ptr);
bool solve_target(wfc_blocks_ptr);
#if defined(WFC_CUDA)
bool solve_cuda(wfc_blocks_ptr);
#endif

static const wfc_solver solvers[] = {
    { "cpu", solve_cpu },
    { "omp", solve_openmp },
    { "target", solve_target },
#if defined(WFC_CUDA)
    { "cuda", solve_cuda },
#endif
};
