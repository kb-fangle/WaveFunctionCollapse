#include <cstdio>
#include <cuda.h>
#include <limits>

#include "bitfield.h"
#include "helper_cuda.h"
#include "md5.h"
#include "types.h"
#include "wfc_cuda.h"
#include <stdio.h>
#include <string.h>


__device__ uint8_t entropy_compute(uint64_t state) {
    return __popcll(state);
}

__device__ uint64_t entropy_collapse_state(uint64_t state, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y, uint64_t seed, uint64_t iteration) {
    uint8_t digest[16] = {0};
    struct {
        uint32_t gx, gy, x, y;
        uint64_t seed, iteration;
    } random_state = {gx, gy, x, y, seed, iteration};

    md5((uint8_t*)&random_state, sizeof(random_state), digest);

    uint8_t entropy = entropy_compute(state);
    uint8_t collapse_index = (uint8_t)(*((uint64_t*)digest) % entropy + 1);
    uint32_t real_index = bitfield_get_nth_set_bit(state, collapse_index);

    state = bitfield_set(0, (uint8_t)real_index - 1);

    return state;
}

__host__ __device__ bool entropy_is_less(uint8_t entropy, uint8_t other_entropy, uint32_t index, uint32_t other_index) {
    return entropy < other_entropy || (entropy == other_entropy && index < other_index);
}

__global__ void compute_min_entropy(uint64_t* blocks, uint32_t* out) {
    extern __shared__ uint32_t shared_memory[];

    uint32_t block_index = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t grid_index = blockIdx.y * gridDim.x + blockIdx.x;
    const uint32_t block_size = blockDim.x * blockDim.y;
    uint32_t global_index = grid_index * block_size + block_index;

    uint32_t* indices = shared_memory;
    uint32_t* entropies = shared_memory + block_size;

    indices[block_index] = global_index;
    entropies[block_index] = entropy_compute(blocks[global_index]);
    if (entropies[block_index] == 1) {
        entropies[block_index] = UINT32_MAX;
    }

    __syncthreads();

    uint32_t prev_offset = block_size;
    uint32_t offset = block_size / 2;
    while (offset > 0) {
        if (block_index < offset
            && entropy_is_less(entropies[block_index + offset], entropies[block_index], indices[block_index + offset], indices[block_index]))
        {
            __syncthreads();
            indices[block_index] = indices[block_index + offset];
            entropies[block_index] = entropies[block_index + offset];
        }

        if (block_index == 0 && offset * 2 < prev_offset) {
            if (entropy_is_less(entropies[prev_offset - 1], entropies[0], indices[prev_offset - 1], indices[0])) {
                indices[0] = indices[prev_offset - 1];
                entropies[0] = entropies[prev_offset - 1];
            }
        }
    
        prev_offset = offset;
        offset /= 2;
        __syncthreads();
    }

    if (block_index == 0) {
        out[grid_index] = entropies[0];
        out[grid_index + gridDim.x * gridDim.x] = indices[0];
    }
}

template<typename T, typename U>
__global__ void set_value(T* arr, U value, size_t size) {
    uint32_t block_index = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t grid_index = blockIdx.y * gridDim.x + blockIdx.x;
    const uint32_t block_size = blockDim.x * blockDim.y;
    uint32_t global_index = grid_index * block_size + block_index;

    if (global_index < size) {
        arr[global_index] = value;
    }
}

__global__ void propagate_block(uint64_t* block, uint64_t collapsed, uint32_t collapsed_x, uint32_t collapsed_y, bool* changed_map, bool* collapsed_map) {
    int id = threadIdx.y * blockDim.x + threadIdx.x;
    if (threadIdx.x == collapsed_x && threadIdx.y == collapsed_y) {
        block[id] = collapsed;
    } else {
        uint64_t new_state = block[id] & ~collapsed;
        changed_map[id] = new_state != 0 && new_state != block[id];
        collapsed_map[id] = new_state != block[id] && entropy_compute(new_state) == 1;
        block[id] = new_state;
    }
}

__global__ void propagate_row(uint64_t* blocks, uint64_t collapsed, uint32_t collapsed_grid_x, bool* changed_map, bool* collapsed_map) {
    if (blockIdx.x != collapsed_grid_x) {
        uint32_t id = blockIdx.x * blockDim.x * blockDim.x + threadIdx.x;

        uint64_t new_state = blocks[id] & ~collapsed;
        changed_map[id] = new_state != 0 && new_state != blocks[id];
        collapsed_map[id] = new_state != blocks[id] && entropy_compute(new_state) == 1;
        blocks[id] = new_state;
    }
}

__global__ void propagate_column(uint64_t* blocks, uint64_t collapsed, uint32_t collapsed_grid_y, bool* changed_map, bool* collapsed_map) {
    if (blockIdx.y != collapsed_grid_y) {
        uint32_t id = blockIdx.y * blockDim.y * blockDim.y * blockDim.y + threadIdx.y * blockDim.y;

        uint64_t new_state = blocks[id] & ~collapsed;
        changed_map[id] = new_state != 0 && new_state != blocks[id];
        collapsed_map[id] = new_state != blocks[id] && entropy_compute(new_state) == 1;
        blocks[id] = new_state;
    }
}

__global__ void check_errors(uint64_t* blocks, uint64_t* block_mask, uint64_t* row_mask, uint64_t* column_mask, bool* error) {
    uint32_t block_index = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t grid_index = blockIdx.y * gridDim.x + blockIdx.x;
    const uint32_t block_size = blockDim.x * blockDim.y;
    uint32_t global_index = grid_index * block_size + block_index;
    
    error[global_index] = blocks[global_index] == 0 || (entropy_compute(blocks[global_index]) != 1 && (
                        (blocks[global_index] & block_mask[blockIdx.y * gridDim.x + blockIdx.x])
                        || (blocks[global_index] & row_mask[blockIdx.y * blockDim.y + threadIdx.y])
                        || (blocks[global_index] & column_mask[blockIdx.x * blockDim.x + threadIdx.x])));
}

__device__ bool propagate_gpu(wfc_cuda_blocks blocks, position collapsed_location, uint64_t collapsed) {
    memset(blocks.d_changed, 0, blocks.sudoku_size * sizeof(*blocks.d_changed));
    memset(blocks.d_collapsed, 0, blocks.sudoku_size * sizeof(*blocks.d_collapsed));


    position pos = collapsed_location;
    uint32_t i = blocks.cell_index(pos.gx, pos.gy, pos.x, pos.y);
    // cudaStream_t streams[3];
    // cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
    // cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
    // cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);

    do {
        pos = blocks.position_at(i);
        uint32_t block_index = blocks.cell_index(pos.gx, pos.gy, 0, 0);
        uint32_t row_index = blocks.cell_index(0, pos.gy, 0, pos.y);
        uint32_t column_index = blocks.cell_index(pos.gx, 0, pos.x, 0);

        blocks.d_block_collapsed_mask[pos.gy * blocks.grid_side + pos.gx] |= collapsed;
        blocks.d_row_collapsed_mask[pos.gy * blocks.block_side + pos.y] |= collapsed;
        blocks.d_column_collapsed_mask[pos.gx * blocks.block_side + pos.x] |= collapsed;

        propagate_block<<<block_propagation_grid_dim, block_propagation_block_dim, 0>>>(
          &blocks.d_states[block_index], collapsed, pos.x, pos.y, &blocks.d_changed[block_index], &blocks.d_collapsed[block_index]);
        propagate_row<<<row_propagation_grid_dim, row_propagation_block_dim, 0>>>(
          &blocks.d_states[row_index], collapsed, pos.gx, &blocks.d_changed[row_index], &blocks.d_collapsed[row_index]);
        propagate_column<<<col_propagation_grid_dim, col_propagation_block_dim, 0>>>(
          &blocks.d_states[column_index], collapsed, pos.gy, &blocks.d_changed[column_index], &blocks.d_collapsed[column_index]);
        cudaDeviceSynchronize();

        // cudaDeviceSynchronize();
        // cudaStreamSynchronize(streams[0]);
        // cudaStreamSynchronize(streams[1]);
        // cudaStreamSynchronize(streams[2]);

        for (i = 0; i < blocks.sudoku_size; i++) {
            if (blocks.d_collapsed[i]) {
                blocks.d_collapsed[i] = 0;
                collapsed = blocks.d_states[i];
                break;
            }
        }

    } while (i < blocks.sudoku_size);

    for (uint32_t i = 0; i < blocks.sudoku_size; i++) {
        if (blocks.d_changed) {
            return true;
        }
    }

    // cudaFree(streams[0]);
    // cudaFree(streams[1]);
    // cudaFree(streams[2]);

    return false;
}

__global__ void solve_gpu(wfc_cuda_blocks blocks) {
    bool changed = true;
    bool valid = true;
    bool has_min_entropy = true;
    const size_t min_entropy_shared_mem_size = (sizeof(uint32_t) + sizeof(uint32_t)) * blocks.block_side * blocks.block_side;
    bool ret = true;
    uint64_t iteration = 0;
    
    memset(blocks.d_block_collapsed_mask, 0, blocks.grid_size * sizeof(*blocks.d_block_collapsed_mask));
    memset(blocks.d_row_collapsed_mask, 0, blocks.grid_size * sizeof(*blocks.d_row_collapsed_mask));
    memset(blocks.d_column_collapsed_mask, 0, blocks.grid_size * sizeof(*blocks.d_column_collapsed_mask));

    while (changed && valid) {
        compute_min_entropy<<<global_grid_dim, global_block_dim, min_entropy_shared_mem_size>>>(blocks.d_states, blocks.d_min_entropy);
        cudaDeviceSynchronize();

        uint32_t min_entropy = UINT32_MAX;
        uint32_t min_index = UINT32_MAX;
        for (int i = 0; i < blocks.grid_size; i++) {
            if (entropy_is_less(blocks.d_min_entropy[i], min_entropy, blocks.d_min_entropy[i + blocks.grid_size], min_index)) {
                min_entropy = blocks.d_min_entropy[i];
                min_index = blocks.d_min_entropy[i + blocks.grid_size];
            }
        }

        if (min_entropy == UINT32_MAX) {
            has_min_entropy = false;
            break;
        }

        position min_entropy_loc = blocks.position_at(min_index);
        uint64_t collapsed_state = entropy_collapse_state(blocks.d_states[min_index], min_entropy_loc.gx, min_entropy_loc.gy, min_entropy_loc.x, min_entropy_loc.y, blocks.seed, iteration);

        changed = propagate_gpu(blocks, min_entropy_loc, collapsed_state);

        check_errors<<<global_grid_dim, global_block_dim>>>(blocks.d_states, blocks.d_block_collapsed_mask, blocks.d_row_collapsed_mask, blocks.d_column_collapsed_mask, blocks.d_changed);
        cudaDeviceSynchronize();
        
        for (int i = 0; i < blocks.sudoku_size; i++) {
            if (blocks.d_changed[i]) {
                valid = false;
                // break;
            }
        }

        iteration++;
    }

    blocks.d_changed[0] = changed && valid && !has_min_entropy;
}

bool solve_cuda(wfc_cuda_blocks* blocks) {
    bool ret;

    solve_gpu<<<1, 1>>>(*blocks);
    cudaMemcpy(&ret, blocks->d_changed, sizeof(ret), cudaMemcpyDeviceToHost);

    return ret;
}
