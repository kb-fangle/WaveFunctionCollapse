#pragma once

#include <cstdint>
#include <cstdio>

#include "helper_cuda.h"

struct wfc_cuda_blocks {
    uint8_t block_side;
    uint8_t grid_side;

    uint8_t _1;
    uint8_t _2;
    uint32_t _3;

    uint32_t grid_size;
    uint32_t block_size;
    uint32_t sudoku_size;

    uint64_t seed;

    uint64_t *d_states = nullptr;
    uint64_t *h_states = nullptr;
    uint64_t *d_states_init = nullptr;
    uint8_t *d_entropies = nullptr;
    uint8_t *h_entropies = nullptr;
    bool *d_changed = nullptr;
    bool *h_changed = nullptr;
    bool *d_collapsed = nullptr;
    bool *h_collapsed = nullptr;
    uint64_t *d_block_collapsed_mask = nullptr;
    uint64_t *d_row_collapsed_mask = nullptr;
    uint64_t *d_column_collapsed_mask = nullptr;

    cudaStream_t streams[4];
    cudaStream_t propagate_streams[3];

    wfc_cuda_blocks(uint8_t grid_side, uint8_t block_side)
        : grid_side(grid_side), block_side(block_side),
          grid_size(grid_side * grid_side), block_size(block_side * block_side),
          sudoku_size(grid_size * block_size)
    {
        cudaMalloc((void **)&d_states, sudoku_size * sizeof(*d_states));
        cudaMalloc((void **)&d_states_init, sudoku_size * sizeof(*d_states_init));
        cudaMalloc((void **)&d_entropies, sudoku_size * sizeof(*d_entropies));
        cudaMalloc((void **)&d_changed, sudoku_size * sizeof(*d_changed));
        cudaMalloc((void **)&d_collapsed, sudoku_size * sizeof(*d_collapsed));
        cudaMalloc((void **)&d_block_collapsed_mask,
                   grid_size * sizeof(*d_block_collapsed_mask));
        cudaMalloc((void **)&d_row_collapsed_mask,
                   grid_side * block_side * sizeof(*d_row_collapsed_mask));
        cudaMalloc((void **)&d_column_collapsed_mask,
                   grid_side * block_side * sizeof(*d_column_collapsed_mask));

        cudaMallocHost((void **)&h_states, sudoku_size * sizeof(*h_states));
        cudaMallocHost((void **)&h_entropies, sudoku_size * sizeof(*h_entropies));
        cudaMallocHost((void **)&h_changed, sudoku_size * sizeof(*h_changed));
        cudaMallocHost((void **)&h_collapsed, sudoku_size * sizeof(*h_collapsed));

        for (auto &stream : streams) {
            cudaStreamCreate(&stream);
        }

        for (auto &stream : propagate_streams) {
            cudaStreamCreate(&stream);
        }
    }

    ~wfc_cuda_blocks()
    {
        cudaFree(d_states);
        cudaFree(d_entropies);
        cudaFree(d_states_init);
        cudaFree(d_changed);
        cudaFree(d_collapsed);
        cudaFree(d_block_collapsed_mask);
        cudaFree(d_row_collapsed_mask);
        cudaFree(d_column_collapsed_mask);

        cudaFreeHost(h_states);
        cudaFreeHost(h_entropies);
        cudaFreeHost(h_changed);
        cudaFreeHost(h_collapsed);

        for (auto stream : streams) {
            cudaStreamDestroy(stream);
        }

        for (auto stream : propagate_streams) {
            cudaStreamDestroy(stream);
        }
    }

    __host__ __device__ uint32_t
    cell_index(uint32_t grid_x, uint32_t grid_y, uint32_t x, uint32_t y)
    {
        return grid_y * grid_side * block_size + grid_x * block_size + y * block_side
               + x;
    }

    __host__ __device__ position
    position_at(uint32_t index)
    {
        position pos;
        pos.gx = (index / block_size) % grid_side;
        pos.gy = index / (grid_side * block_size);
        index -= pos.gy * grid_side * block_size + pos.gx * block_size;
        pos.x = index % block_side;
        pos.y = index / block_side;
        return pos;
    }
};
