#pragma once

#ifdef __cplusplus
#define restrict
extern "C" {
#endif

#include <inttypes.h>
#include <stdbool.h>

/// Opaque type to store the seeds to try for the solving process. You may push to it and pop from
/// it. You may not try to index it manually or free this structure, it will be automatically freed
/// when no more items are present inside it.
typedef struct seeds_list seeds_list;

typedef struct {
    uint32_t x, y;
} vec2;

typedef struct {
    uint32_t gx;
    uint32_t gy;
    uint32_t x;
    uint32_t y;
} position;


typedef struct {
    position location;
    uint8_t entropy;

    uint8_t _1;
    uint16_t _2;
} entropy_location;

typedef struct {
    uint8_t block_side;
    uint8_t grid_side;

    uint8_t _1;
    uint8_t _2;
    uint32_t _3;

    uint64_t seed;

    uint64_t states[];
} wfc_blocks;

typedef wfc_blocks *wfc_blocks_ptr;

typedef enum {
    CPU, OMP, OMP2, TARGET, CUDA
} solver_kind;

typedef struct {
    const char *const data_file;
    const char *const output_folder;
    seeds_list *restrict seeds;
    const uint32_t parallel;
    const solver_kind kind;
    bool (*const solver)(wfc_blocks_ptr);
} wfc_args;

typedef struct {
    const char *const name;
    solver_kind kind;
    bool (*function)(wfc_blocks_ptr);
} wfc_solver;

#ifdef __cplusplus
}
#undef restrict
#endif
