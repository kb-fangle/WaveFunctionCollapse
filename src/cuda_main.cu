#include "types.h"
#include "args.h"
#include "wfc_cuda.h"
#include <cstring>
#include <stdio.h>
#include <omp.h>

extern bool solve_cuda(wfc_cuda_blocks* blocks);
extern bool solve_cuda2(wfc_cuda_blocks& blocks);

extern "C" void cuda_main_loop(wfc_args args, wfc_blocks_ptr init) {
    bool quit = false;
    uint64_t iterations      = 0;
    wfc_blocks_ptr blocks = NULL;

    wfc_clone_into(&blocks, 0, init);

    wfc_cuda_blocks cu_blocks(init->grid_side, init->block_side);
    checkCudaErrors(cudaMemcpy(cu_blocks.d_states_init, init->states, cu_blocks.sudoku_size * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const uint64_t max_iterations = count_seeds(args.seeds);
    const double start            = omp_get_wtime();

    while (!quit) {
        uint64_t next_seed       = 0;
        const bool has_next_seed = try_next_seed(&args.seeds, &next_seed);

        if (!has_next_seed) {
            fprintf(stderr, "\nno more seed to try\n");
            break;
        }
        
        cu_blocks.seed = next_seed;
        // cudaMemcpy(cu_blocks.d_states, init->states, cu_blocks.sudoku_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
        // const bool solved = solve_cuda(&cu_blocks);
        const bool solved = solve_cuda2(cu_blocks);
        iterations += 1;

        if (solved && args.output_folder != NULL) {
            fprintf(stdout, "\nsuccess with seed %lu\n", next_seed);
	    cudaMemcpy(blocks->states, cu_blocks.d_states, cu_blocks.sudoku_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            blocks->seed = cu_blocks.seed;
            wfc_save_into(blocks, args.data_file, args.output_folder);
            break;
        }

        else if (solved) {
            fputs("\nsuccess with result:\n", stdout);
            break;
        }

	fprintf(stdout, "\r%.2f%% -> %.2fs",
		((double)iterations / (double)(max_iterations)) * 100.0,
		omp_get_wtime() - start);
    }

    cu_blocks.clean();

    free(blocks);
}
