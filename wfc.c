#define _GNU_SOURCE

#include "wfc.h"
#include "bitfield.h"
// #include "utils.h"
#include "md5.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <limits.h>
#include <errno.h>
#include <string.h>
#include <strings.h>

uint64_t
entropy_collapse_state(uint64_t state,
                       uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                       uint64_t seed,
                       uint64_t iteration)
{
    uint8_t digest[16]     = { 0 };
    uint64_t random_number = 0;
    struct {
        uint32_t gx, gy, x, y;
        uint64_t seed, iteration;
    } random_state = {
        .gx        = gx,
        .gy        = gy,
        .x         = x,
        .y         = y,
        .seed      = seed,
        .iteration = iteration,
    };

    md5((uint8_t *)&random_state, sizeof(random_state), digest);

    return 0;
}

uint8_t
entropy_compute(uint64_t state)
{
    // int count = 0;
    // // Compte le nombre de 1 dans le mask
    // while(state){
    //     count += state & 1;
    //     state >>=1;
    // }
    
    
    return __buitlin_popcount(state);
}

void
wfc_clone_into(wfc_blocks_ptr *const restrict ret_ptr, uint64_t seed, const wfc_blocks_ptr blocks)
{
    const uint64_t grid_size  = blocks->grid_side;
    const uint64_t block_size = blocks->block_side;
    wfc_blocks_ptr ret        = *ret_ptr;

    const uint64_t size = (wfc_control_states_count(grid_size, block_size) * sizeof(uint64_t)) +
                          (grid_size * grid_size * block_size * block_size * sizeof(uint64_t)) +
                          sizeof(wfc_blocks);

    if (NULL == ret) {
        if (NULL == (ret = malloc(size))) {
            fprintf(stderr, "failed to clone blocks structure\n");
            exit(EXIT_FAILURE);
        }
    } else if (grid_size != ret->grid_side || block_size != ret->block_side) {
        fprintf(stderr, "size mismatch!\n");
        exit(EXIT_FAILURE);
    }

    memcpy(ret, blocks, size);
    ret->states[0] = seed;
    *ret_ptr       = ret;
}

entropy_location
blk_min_entropy(const wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy)
{
    // Initialisation entropy max
    // entropy_location *blk_entropy = (entropy_location *)malloc(sizeof(entropy_location));
    entropy_location blk_entropy;
    uint8_t entropy = blocks->block_side * blocks->block_side;
    vec2 the_location   = { 0 };
    uint8_t min_entropy = entropy;

    // On parcours chaque case
    uint32_t* block_loc = grd_at(blocks,gx,gy);
    for (int i=0; i < blocks->block_side*blocks->block_side;i++){
        entropy = entropy_compute(block_loc[i]);
        if (entropy < min_entropy && entropy > 0){
            min_entropy = entropy;
            the_location.x = i % blocks->block_side;
            the_location.y = i / blocks->block_side;
            }
    }

    blk_entropy.location = the_location;
    blk_entropy.entropy = min_entropy;
    
    return blk_entropy;
}

static inline uint64_t
blk_filter_mask_for_column(wfc_blocks_ptr blocks,
                           uint32_t gy, uint32_t y,
                           uint64_t collapsed)
{
    return 0;
}

static inline uint64_t
blk_filter_mask_for_row(wfc_blocks_ptr blocks,
                        uint32_t gx, uint32_t x,
                        uint64_t collapsed)
{
    return 0;
}

static inline uint64_t
blk_filter_mask_for_block(wfc_blocks_ptr blocks,
                          uint32_t gy, uint32_t gx,
                          uint64_t collapsed)
{
    return 0;
}

bool
grd_check_error_in_column(wfc_blocks_ptr blocks, uint32_t gx)
{
    return 0;
}

void
blk_propagate(wfc_blocks_ptr blocks,
              uint32_t gx, uint32_t gy,
              uint64_t collapsed)
{
    
    uint32_t* block_loc = grd_at(blocks,gx,gy);
    for (int i=0; i < blocks->block_side*blocks->block_side;i++){
        block_loc[i] = bitfield_unset(block_loc[i],collapsed);
    }
}

void
grd_propagate_row(wfc_blocks_ptr blocks,
                  uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                  uint64_t collapsed)
{
    uint32_t* row = 0;

    
    for (int i=0; i < blocks->grid_side;i++){
        row = blk_at(blocks,i,gy,0,y);
        for (int j=0; j < blocks->block_side;j++){
            row[j] = bitfield_unset(row[j],collapsed);
        }
    }
}

void
grd_propagate_column(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                     uint32_t x, uint32_t y, uint64_t collapsed)
{
    uint32_t* col = 0;

    
    for (int i=0; i < blocks->grid_side;i++){
        col = blk_at(blocks,gx,i,x,0);
        for (int j=0; j < blocks->block_side;j++){
            col[j*blocks->block_side] = bitfield_unset(col[j*blocks->block_side],collapsed);
        }
    }
}
