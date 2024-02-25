#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "bitfield.h"
#include "position_list.h"
#include "types.h"

///
/// @brief Computes the address of the begining of a block
///
/// @param blocks The grid
/// @param gx The block's x position (col)
/// @param gy The block's y position (row)
///
static inline uint64_t *
grd_at(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy) {
    uint32_t block_size = blocks->block_side * blocks->block_side;
    return &blocks->states[gy * blocks->grid_side * block_size
                           + gx * block_size];
}

///
/// @brief Computes the address of a cell
///
/// @param blocks The grid
/// @param gx, gy The grid coordinates of the containing block
/// @param x, y The block coordinates of the cell
///
static inline uint64_t *
blk_at(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x,
       uint32_t y) {

    return grd_at(blocks, gx, gy) + y * blocks->block_side + x;
}

///
/// @brief Computes the position of a cell given its index
///
/// @param blocks The grid
/// @param i the cell's 1D index
/// @param The 2D position (gx, gy, x, y) of the cell
///
static inline position
position_at(wfc_blocks_ptr blocks, uint32_t index) {
    position pos;
    uint32_t block_size = blocks->block_side * blocks->block_side;
    pos.gx = (index / block_size) % blocks->grid_side;
    pos.gy = index / (blocks->grid_side * block_size);
    index -= pos.gy * blocks->grid_side * block_size + pos.gx * block_size;
    pos.x = index % blocks->block_side;
    pos.y = index / blocks->block_side;
    return pos;
}

// Printing functions
/// Print a block to `out`
void blk_print(FILE *const out, const wfc_blocks_ptr block, uint32_t gx,
               uint32_t gy);
/// Print the grid to `out`
void grd_print(FILE *const out, const wfc_blocks_ptr block);

// Entropy functions

/// @brief Find the cell with minimum entropy in the grid
/// If two or more cells have minimum entropy, teh function choses one at
/// random.
position grd_min_entropy(const wfc_blocks_ptr blocks);

/// @brief Computes the entropy of a state
static inline uint8_t
entropy_compute(uint64_t x) {
    return bitfield_count(x);
}

/// @brief Collapses a state
uint64_t entropy_collapse_state(uint64_t state, uint32_t gx, uint32_t gy,
                                uint32_t x, uint32_t y, uint64_t seed,
                                uint64_t iteration);
///
/// @group Propagtion functions
///
/// @param blocks The grid
/// @param collapsed The state to collapse
/// @param position_list A list containing the position of the states resolved
/// after collapsing.
///
/// @return true if at least one of the states changed, false otherwise.
///
/// @{

/// Propagate a collapsed state in a block.
bool blk_propagate(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                   uint64_t collapsed, position_list *collapsed_stack);
/// Propagate a collapsed state in a column.
bool grd_propagate_column(wfc_blocks_ptr blocks, uint32_t gx, uint32_t x,
                          uint64_t collapsed, position_list *collapsed_stack);
/// Propagate a collapsed state in a row.
bool grd_propagate_row(wfc_blocks_ptr blocks, uint32_t gy, uint32_t y,
                       uint64_t collapsed, position_list *collapsed_stack);
/// Propagate a collapsed state in the grid.
bool propagate(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x,
               uint32_t y);

/// }@

// Check functions

///
/// @brief Checks the grid for errors (duplicate values in a block, row or
/// column)
///
/// @param blocks The grid
///
/// @return true if the grid is valid, false otherwise
///
bool check_grid(const wfc_blocks_ptr blocks);

// Solvers
bool solve_cpu(wfc_blocks_ptr);
bool solve_openmp(wfc_blocks_ptr);

static const wfc_solver solvers[] = {
    { "cpu", CPU, solve_cpu },
    { "omp", OMP, solve_openmp },
    { "omp_par", OMP_PAR, solve_cpu },
#ifdef WFC_CUDA
    { "cuda", CUDA, solve_cpu },
#endif
};
