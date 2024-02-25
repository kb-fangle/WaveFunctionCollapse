#define _GNU_SOURCE

#include <omp.h>
#include <stdio.h>

#include "args.h"
#include "types.h"
#include "wfc.h"

void
cpu_main_loop(wfc_args args, wfc_blocks_ptr init)
{
    uint64_t iterations = 0;
    wfc_blocks_ptr blocks = NULL;

    uint8_t *entropies = (uint8_t *)malloc((size_t)init->block_side * init->block_side
                                           * (size_t)init->grid_side * init->grid_side
                                           * sizeof(uint8_t));

    const uint64_t max_iterations = count_seeds(args.seeds);
    const double start = omp_get_wtime();

    while (true) {
        uint64_t next_seed = 0;
        const bool has_next_seed = try_next_seed(&args.seeds, &next_seed);

        if (!has_next_seed) {
            fprintf(stderr, "\nno more seed to try\n");
            break;
        }

        wfc_clone_into(&blocks, next_seed, init);
        blocks->entropies = entropies;
        const bool solved = args.solver(blocks);
        iterations += 1;

        if (solved && args.output_folder != NULL) {
            fprintf(stdout, "\nsuccess with seed %lu\n", blocks->seed);
            wfc_save_into(blocks, args.data_file, args.output_folder);
            break;
        } else if (solved) {
            fputs("\nsuccess with result:\n", stdout);
            break;
        }

        fprintf(stdout, "\r%.2f%% -> %.2fs",
                ((double)iterations / (double)(max_iterations)) * 100.0,
                omp_get_wtime() - start);
    }

    free(entropies);
}

void
omp_main_loop(wfc_args args, wfc_blocks_ptr init)
{
    bool quit = false;
    uint64_t iterations = 0;

    const uint64_t max_iterations = count_seeds(args.seeds);
    const double start = omp_get_wtime();

#pragma omp parallel shared(quit, iterations)
    {
        uint8_t *entropies = (uint8_t *)malloc(
            (size_t)init->block_side * init->block_side * (size_t)init->grid_side
            * init->grid_side * sizeof(uint8_t));
        uint64_t next_seed;
        wfc_blocks_ptr blocks = NULL;
        bool has_next_seed = false;
        while (!quit) {
#pragma omp critical
            {
                has_next_seed = try_next_seed(&args.seeds, &next_seed);
            }

            if (!has_next_seed) {
#pragma omp single
                fprintf(stderr, "\nno more seed to try\n");
                break;
            }

            wfc_clone_into(&blocks, next_seed, init);
            blocks->entropies = entropies;

            bool solved = solve_cpu(blocks);
            iterations++;

            if (solved) {
                quit = true;
                fprintf(stdout, "\nsuccess with seed %lu\n", blocks->seed);
                if (args.output_folder != NULL) {
                    wfc_save_into(blocks, args.data_file, args.output_folder);
                }
            }

            if (!quit) {
                fprintf(stdout, "\r%.2f%% -> %.2fs",
                        ((double)iterations / (double)max_iterations) * 100.0,
                        omp_get_wtime() - start);
            }
        }

        free(entropies);
    }
}

#ifdef WFC_CUDA
extern void cuda_main_loop(wfc_args args, wfc_blocks_ptr init);
#endif

int
main(int argc, char **argv)
{
    omp_set_dynamic(false);

    wfc_args args = wfc_parse_args(argc, argv);
    const wfc_blocks_ptr init = wfc_load(0, args.data_file);

    switch (args.kind) {
    case OMP:
        omp_set_num_threads((int)args.parallel);
        cpu_main_loop(args, init);
        break;
    case CPU:
        cpu_main_loop(args, init);
        break;
    case OMP_PAR:
        omp_set_num_threads((int)args.parallel);
        omp_main_loop(args, init);
        break;
    case CUDA:
#ifdef WFC_CUDA
        cuda_main_loop(args, init);
#endif
    default:
        break;
    }

    return 0;
}
