
#include <cuda.h>
#include "wfc.h"
#include "bitfield.h"
#include "helper_cuda.h"

#if !defined(WFC_CUDA)
#error "WDC_CUDA should be defined..."
#endif

/// Entropy location with a field for the cell's address and its state
typedef struct {
    uint64_t* cell;
    uint64_t state;
    uint8_t entropy;
    position loc;
} entropy_location_alt;

/// stores in `out` the minimum entropy location between `out` and `in`
/// Because the cells are laid out blocks by blocks in memory, we can directly
/// order them using their address
void entropy_min(entropy_location_alt* out, entropy_location_alt* in) {
    if (in->entropy > 1 && (in->entropy < out->entropy || (in->entropy == out->entropy && in->cell < out->cell))) {
        *out = *in;
    }
}

static inline wfc_blocks *
safe_cudamalloc(uint64_t blkcnt)
{
    uint64_t size   = sizeof(wfc_blocks) + sizeof(uint64_t) * blkcnt;
    wfc_blocks *ret = (wfc_blocks *)malloc(size);
    checkCudaErrors(cudaMalloc((void**)&ret, size));
    // if (ret != NULL) {
    //     return ret;
    // } else {
    //     fprintf(stderr, "failed to malloc %zu bytes\n", size);
    //     exit(EXIT_FAILURE);
    // }
    return ret;
}

static inline uint64_t
grd_at_idx(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy)
{
    uint32_t block_size = blocks->block_side * blocks->block_side;
    return gy * blocks->grid_side * block_size + gx * block_size;
}

static inline uint64_t
blk_at_idx(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y)
{
    
    return grd_at(blocks,gx,gy) + y * blocks->block_side + x;
}

__global__ void cuda_entropy(wfc_blocks *blocks, entropy_location_alt min_entropy){
    const uint32_t gx = blockIdx.x;
    const uint32_t gy = blockIdx.y;
    const uint32_t x = threadIdx.x;
    const uint32_t y = threadIdx.y;
    uint64_t* cell = blk_at(blocks, gx, gy, x, y);
    entropy_location_alt loc = {cell,
                                *cell,
                                entropy_compute(*cell),
                                { gx, gy, x, y },
                                };

    entropy_min(&min_entropy, &loc);
}



__global__ void cuda_blk_propagate(wfc_blocks_ptr blocks,
              position loc, position_list* collapsed_stack)
{
    // Collapsed state in shared memory
    __shared__ uint64_t collapsed;
    if (threadIdx.x==0 && threadIdx.y==0) {
        uint64_t* collapsed_cell = blk_at(blocks,loc.gx,loc.gy,loc.x,loc.y);
        collapsed = *collapsed_cell;
    }
    __syncthreads();  
    
    uint64_t* cell = blk_at(blocks,loc.gx,loc.gy,threadIdx.x,threadIdx.y);
    const uint64_t new_state = cell & ~collapsed;
    
    // Checking entropy of the cell with new state
    if (new_state != cell && bitfield_count(new_state) == 1) {
            position pos = { loc.gx, loc.gy, threadIdx.x, threadIdx.y };
            position_list_push(collapsed_stack, pos);  // besoin mutex ??
        }

    cell = new_state;
}

__global__ void cuda_row_propagate(wfc_blocks_ptr blocks,
              position loc, position_list* collapsed_stack)
{
    // Collapsed state in shared memory
    __shared__ uint64_t collapsed;
    if (threadIdx.x==0 && threadIdx.y==0) {
        uint64_t* collapsed_cell = blk_at(blocks,loc.gx,loc.gy,loc.x,loc.y);
        collapsed = *collapsed_cell;
    }
    __syncthreads();  
    
    // One block is useless
    if (blockIdx.x != loc.gx) {

        uint64_t* cell = blk_at(blocks,blockIdx.x,loc.gy,threadIdx.x,loc.y);
        const uint64_t new_state = cell & ~collapsed;
        
        // Checking entropy of the cell with new state
        if (new_state != cell && bitfield_count(new_state) == 1) {
                position pos = { blockIdx.x, loc.gy, threadIdx.x, loc.y };
                position_list_push(collapsed_stack, pos);  // besoin mutex ??
            }

        cell = new_state;

    }
}

__global__ void cuda_col_propagate(wfc_blocks_ptr blocks,
              position loc, position_list* collapsed_stack)
{
    // Collapsed state in shared memory
    __shared__ uint64_t collapsed;
    if (threadIdx.x==0 && threadIdx.y==0) {
        uint64_t* collapsed_cell = blk_at(blocks,loc.gx,loc.gy,loc.x,loc.y);
        collapsed = *collapsed_cell;
    }
    __syncthreads();  
    
    // One block is useless
    if (blockIdx.y != loc.gy) {

        uint64_t* cell = blk_at(blocks,loc.gx,blockIdx.y,loc.x,threadIdx.y);
        const uint64_t new_state = cell & ~collapsed;
        
        // Checking entropy of the cell with new state
        if (new_state != cell && bitfield_count(new_state) == 1) {
                position pos = { loc.gx, blockIdx.y, loc.x, threadIdx.y };
                position_list_push(collapsed_stack, pos);  // besoin mutex ??
            }

        cell = new_state;

    }
}

__device__ bool d_valid = true;

__global__ void cuda_blk_check(wfc_blocks_ptr blocks) {
    const uint32_t gx = blockIdx.x;
    const uint32_t gy = blockIdx.y;
    const uint32_t x = threadIdx.x;
    const uint32_t y = threadIdx.y;

    // One mask per block
    __shared__ uint64_t mask;

    // Init mask
    if (x==0 && y==0) {mask = 0;}
    __syncthreads();

    uint64_t* cell = blk_at(blocks,gx,gy,x,y);
    if (cell == 0 || (cell & mask) != 0) {
        d_valid = false;
    }

    if (entropy_compute(cell) == 1) {
        mask |= cell;    // besoin mutex ??
    }
}
 
__global__ cuda_row_check(wfc_blocks_ptr blocks) {
    const uint32_t gx = threadIdx.x / blockDim.x;
    const uint32_t gy = blockIdx.y / gridDim.y;
    const uint32_t x = threadIdx.x % blockDim.x;
    const uint32_t y = blockIdx.y % gridDim.y;

    // One mask per block
    __shared__ uint64_t mask;

    // Init mask
    if (mask == NULL) {
        mask = 0    // besoin mutex ??
    }

    uint64_t* cell = blk_at(blocks,gx,gy,x,y);
    if (cell == 0 || (cell & mask) != 0) {
        d_valid = false;
    }

    if (entropy_compute(cell) == 1) {
        mask |= cell;    // besoin mutex ??
    }
}

__global__ cuda_col_check(wfc_blocks_ptr blocks) {
    const uint32_t gx = blockIdx.x / gridDim.x;
    const uint32_t gy = threadIdx.y / blockDim.y;
    const uint32_t x = blockIdx.x % gridDim.x;
    const uint32_t y = threadIdx.y % blockDim.y;

    // One mask per block
    __shared__ uint64_t mask;

    // Init mask
    if (mask == NULL) {
        mask = 0    // besoin mutex ??
    }

    uint64_t* cell = blk_at(blocks,gx,gy,x,y);
    if (cell == 0 || (cell & mask) != 0) {
        d_valid = false;
    }

    if (entropy_compute(cell) == 1) {
        mask |= cell;    // besoin mutex ??
    }
}



bool
solve_cuda(wfc_blocks_ptr blocks)
{
    
    uint64_t iteration  = 0;

    bool changed = true;
    bool has_min_entropy = true;
    bool valid = true;


    // Streams init
    cudaStream_t stream[3];
    cudaStreamCreate (&stream[0]);
    cudaStreamCreate (&stream[1]);
    cudaStreamCreate (&stream[2]);

    // Allocate once for all sudoku on device
    wfc_blocks *d_blocks = NULL;
    int block_size = blocks->block_side * blocks->block_side;
    int blkcnt = block_size * block_size;
    d_blocks = safe_cudamalloc(blkcnt);

    // Allocate once for all min_entropy on device
    entropy_location_alt min_entropy = malloc(sizeof(entropy_location_alt));
    entropy_location_alt d_min_entropy = NULL;
    checkCudaErrors(cudaMalloc((void**)&d_min_entropy, sizeof(entropy_location_alt)));
    

    // Allocate once for all collapsed_stack on device
    position_list collapsed_stack = position_list_init();  // pas bon, besoin nouvelle fonction list cuda
    position_list d_collapsed_stack = NULL;
    checkCudaErrors(cudaMalloc((void**)&d_collapsed_stack, sizeof(position_list))); 

    // Transfert data (H2D)
    uint64_t size   = sizeof(wfc_blocks) + sizeof(uint64_t) * blkcnt;
    checkCudaErrors(cudaMemcpy(d_blocks, blocks, size, cudaMemcpyHostToDevice));
    bool d_valid = NULL;
    // checkCudaErrors(cudaMemcpy(d_valid, valid, sizeof(bool), cudaMemcpyHostToDevice));

    // Kernel configuration for full sudoku (2D)
    dim3 dimBlock(blocks->block_side,blocks->block_side,1);
    dim3 dimGrid(blocks->grid_side,blocks->grid_side,1);

    // Kernel configuration for full sudoku with one full row (bloc 1D, grid 1D)
    dim3 dimBlock_full_row(block_size,1,1);
    dim3 dimGrid_full_row(1,block_size,1);

    // Kernel configuration for full sudoku with one full col (bloc 1D, grid 1D)
    dim3 dimBlock_full_col(1,block_size,1);
    dim3 dimGrid_full_col(block_size,1,1);

    // Kernel configuration for one block (bloc 2D)
    dim3 dimBlock_blk(blocks->block_side,blocks->block_side,1);
    dim3 dimGrid_blk(1,1,1);

    // Kernel configuration for one row (bloc 1D, grid 1D)
    dim3 dimBlock_row(blocks->block_side,1,1);
    dim3 dimGrid_row(blocks->block_side,1,1);

    // Kernel configuration for one col (bloc 1D, grid 1D)
    dim3 dimBlock_col(1,blocks->block_side,1);
    dim3 dimGrid_col(1,blocks->block_side,1);

    
    

    while (changed && has_min_entropy && valid) {
        /// Entropy
        
        // Init and transfert to device min_entropy (H2D)
        min_entropy = { (uint64_t*)UINTPTR_MAX, UINT64_MAX, UINT8_MAX, { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX } };
        checkCudaErrors(cudaMemcpy(d_min_entropy, min_entropy, sizeof(entropy_location_alt), cudaMemcpyHostToDevice));

        // Min entropy kernel launch
        cuda_entropy<<<dimGrid,dimBlock>>>(d_blocks,d_min_entropy);
        checkCudaErrors(cudaGetLastError());
	    checkCudaErrors(cudaDeviceSynchronize());

        // Retrieve the min entropy (D2H)
        checkCudaErrors(cudaMemcpy(min_entropy, d_min_entropy, sizeof(entropy_location_alt), cudaMemcpyDeviceToHost));

        has_min_entropy = min_entropy.loc.x != UINT32_MAX;

        
        if (has_min_entropy) {
            const uint32_t gx = min_entropy.loc.gx;
            const uint32_t gy = min_entropy.loc.gy;
            const uint32_t x = min_entropy.loc.x;
            const uint32_t y = min_entropy.loc.y;

            position collapsed_pos = { gx, gy, x, y };
                position_list_push(&collapsed_stack, collapsed_pos);

            /// Collapse   
            uint64_t collapsed_idx = blk_at_idx(blocks, gx, gy, x, y);
            min_entropy->state = entropy_collapse_state(min_entropy->state, gx, gy, x, y, blocks->seed, iteration);
            
            // Transfer collapsed_cell to sudoku in device memory (H2D)
            checkCudaErrors(cudaMemcpy(d_blocks->state[collapsed_idx], min_entropy->state, sizeof(uint64_t), cudaMemcpyHostToDevice));


                /// Propagate
                while (!position_list_is_empty(&collapsed_stack)) {
                    collapsed_pos = position_list_pop(&collapsed_stack);

                    // Transfert collapsed_stack to device (H2D)
                    checkCudaErrors(cudaMemcpy(d_collapsed_stack, collapsed_stack, sizeof(position_list), cudaMemcpyHostToDevice));

                    // Block propagate kernel launch
                    cuda_blk_propagate<<<dimGrid_blk,dimBlock_blk>>>(d_blocks,
                                                                    collapsed_pos,
                                                                    d_collapsed_stack);
                    checkCudaErrors(cudaGetLastError());
                    checkCudaErrors(cudaDeviceSynchronize());  // à retirer après debug
                    
                    // Row propagate kernel launch
                    cuda_row_propagate<<<dimGrid_row,dimBlock_row>>>(d_blocks,
                                                                    collapsed_pos,
                                                                    d_collapsed_stack);
                    checkCudaErrors(cudaGetLastError());
                    checkCudaErrors(cudaDeviceSynchronize());  // à retirer après debug

                    // Col propagate kernel launch
                    cuda_col_propagate<<<dimGrid_col,dimBlock_col>>>(d_blocks,
                                                                    collapsed_pos,
                                                                    d_collapsed_stack);
                    checkCudaErrors(cudaGetLastError());
                    checkCudaErrors(cudaDeviceSynchronize());

                    // Retrieve the collapsed_stack (D2H)
                    checkCudaErrors(cudaMemcpy(collapsed_stack,d_collapsed_stack, sizeof(position_list), cudaMemcpyDeviceToHost));
                }

            /// Verify

            // Block check kernel launch
            cuda_blk_check<<<dimGrid_blk,dimBlock_blk,0,stream[0]>>>(d_blocks);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());  // à retirer après debug

            // Row check kernel launch
            cuda_row_check<<<dimGrid_full_row,dimBlock_full_row,0,stream[1]>>>(d_blocks);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());  // à retirer après debug

            // Col check kernel launch
            cuda_col_check<<<dimGrid_full_col,dimBlock_full_col,0,stream[2]>>>(d_blocks);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());  

            // Retrieve the verification result (D2H)
            checkCudaErrors(cudaMemcpy(valid,d_valid, sizeof(bool), cudaMemcpyDeviceToHost));
        }
        
        iteration++;
    }

    // Retrieve the sudoku (D2H)
    checkCudaErrors(cudaMemcpy(blocks,d_blocks, size, cudaMemcpyDeviceToHost));

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);

    free(min_entropy);
    cudaFree(d_blocks);
    cudaFree(d_min_entropy);
    cudaFree(d_collapsed_stack);

    return changed && !has_min_entropy && valid;
}
