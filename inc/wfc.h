#pragma once

#include "types.h"

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

static inline uint64_t *
grd_at(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy)
{
    return &blocks->states[gy*blocks->grid_side + gx];
}

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
uint8_t entropy_compute(uint64_t);
uint64_t entropy_collapse_state(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t, uint64_t);

// Propagation functions
void blk_propagate(wfc_blocks_ptr, uint32_t, uint32_t, uint64_t);
void grd_propagate_column(wfc_blocks_ptr, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t);
void grd_propagate_row(wfc_blocks_ptr, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t);

// Check functions
bool grd_check_error_in_column(wfc_blocks_ptr, uint32_t);

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
