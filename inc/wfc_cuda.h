#pragma once

#include "types.h"
#include <cstdio>
#include <stdint.h>
#include "helper_cuda.h"


inline __device__ dim3 global_grid_dim;
inline __device__ dim3 global_block_dim;

inline __device__ dim3 block_propagation_grid_dim;
inline __device__ dim3 block_propagation_block_dim;

inline __device__ dim3 row_propagation_grid_dim;
inline __device__ dim3 row_propagation_block_dim;

inline __device__ dim3 col_propagation_grid_dim;
inline __device__ dim3 col_propagation_block_dim;


inline __global__ void init(uint8_t grid_side, uint8_t block_side) {
    global_grid_dim = dim3(grid_side, grid_side);
    global_block_dim = dim3(block_side, block_side);

    block_propagation_grid_dim = dim3(1, 1);
    block_propagation_block_dim = dim3(block_side, block_side);

    row_propagation_grid_dim = dim3(grid_side, 1);
    row_propagation_block_dim = dim3(block_side, 1);

    col_propagation_grid_dim = dim3(1, grid_side);
    col_propagation_block_dim = dim3(1, block_side);
}

struct wfc_cuda_blocks {
    uint8_t block_side;
    uint8_t grid_side;

    uint32_t grid_size;
    uint32_t block_size;
    uint32_t sudoku_size;

    uint8_t _1;
    uint8_t _2;
    uint32_t _3;

    uint64_t seed;

    uint64_t* d_states = nullptr;
    uint32_t* d_min_entropy = nullptr;
    bool* d_changed = nullptr;
    bool* d_collapsed = nullptr;
    uint64_t* d_block_collapsed_mask = nullptr;
    uint64_t* d_row_collapsed_mask = nullptr;
    uint64_t* d_column_collapsed_mask = nullptr;

    cudaStream_t streams[3];

    wfc_cuda_blocks(uint8_t grid_side, uint8_t block_side) : grid_side(grid_side), block_side(block_side), grid_size(grid_side * grid_side), block_size(block_side * block_side), sudoku_size(grid_size * block_size) {
        cudaMalloc((void**)&d_states, sudoku_size * sizeof(uint64_t));
        cudaMalloc((void**)&d_min_entropy, 2 * grid_size * sizeof(*d_min_entropy));
        cudaMalloc((void**)&d_changed, sudoku_size * sizeof(*d_changed));
        cudaMalloc((void**)&d_collapsed, sudoku_size * sizeof(*d_collapsed));
        cudaMalloc((void**)&d_block_collapsed_mask, grid_size * sizeof(*d_block_collapsed_mask));
        cudaMalloc((void**)&d_row_collapsed_mask, grid_size * sizeof(*d_row_collapsed_mask));
        cudaMalloc((void**)&d_column_collapsed_mask, grid_size * sizeof(*d_column_collapsed_mask));

        for (auto& stream: streams) {
            cudaStreamCreate(&stream);
        }

        init<<<1, 1>>>(grid_side, block_side);
    }

    // ~wfc_cuda_blocks() {
    // }

    void clean() {
        cudaFree(d_states);
        cudaFree(d_min_entropy);
        cudaFree(d_changed);
        cudaFree(d_collapsed);
        cudaFree(d_block_collapsed_mask);
        cudaFree(d_row_collapsed_mask);
        cudaFree(d_column_collapsed_mask);

        for (auto stream: streams) {
            cudaStreamDestroy(stream);
        }
    }


    __host__ __device__ uint32_t cell_index(uint32_t grid_x, uint32_t grid_y, uint32_t x, uint32_t y) {
        return grid_y * grid_side * block_size + grid_x * block_size + y * block_side + x;
    }

    __host__ __device__ position position_at(uint32_t index) {
        position pos;
        pos.gx = (index / block_size) % grid_side;
        pos.gy = index / (grid_side * block_size);
        index -= pos.gy * grid_side * block_size + pos.gx * block_size;
        pos.x = index % block_side;
        pos.y = index / block_side;
        return pos;
    }
};
