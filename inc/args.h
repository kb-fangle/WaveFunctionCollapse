#pragma once

#ifdef __cplusplus
#define restrict
extern "C" {
#endif

#include "types.h"

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

#ifdef __cplusplus
}
#undef restrict
#endif
