#include "types.h"
#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"
#include "wfc_omp.h"

#include <stdio.h>
#include <omp.h>
#include <pthread.h>

void cpu_main_loop(wfc_args args, wfc_blocks_ptr init) {
    bool quit         = false;
    uint64_t iterations      = 0;
    wfc_blocks_ptr blocks    = NULL;

    const uint64_t max_iterations = count_seeds(args.seeds);
    const double start            = omp_get_wtime();

    while (!quit) {
        uint64_t next_seed       = 0;
        const bool has_next_seed = try_next_seed(&args.seeds, &next_seed);

        if (!has_next_seed) {
            quit = true;
            fprintf(stderr, "no more seed to try\n");
            break;
        }

        wfc_clone_into(&blocks, next_seed, init);
        const bool solved = args.solver(blocks);
        iterations += 1;

        if (solved && args.output_folder != NULL) {
            quit = true;
            fprintf(stdout, "\nsuccess with seed %lu\n", blocks->seed);
            wfc_save_into(blocks, args.data_file, args.output_folder);
        }

        else if (solved) {
            quit = true;
            fputs("\nsuccess with result:\n", stdout);
            abort();
        }

        else if (!quit) {
            fprintf(stdout, "\r%.2f%% -> %.2fs",
                    ((double)iterations / (double)(max_iterations)) * 100.0,
                    omp_get_wtime() - start);
        }
    }
}

void
omp_main_loop(wfc_args args, wfc_blocks_ptr init) {
    bool quit = false;
    uint64_t iterations = 0;

    const uint64_t max_iterations = count_seeds(args.seeds);
    const double start = omp_get_wtime();

    #pragma omp parallel shared(quit,iterations)
    {
        uint64_t next_seed;
        wfc_blocks_ptr blocks = NULL;
        bool has_next_seed = false;
        while (!quit) {
            #pragma omp critical
            {
                has_next_seed = try_next_seed(&args.seeds, &next_seed);
                // iteration = iteration_counter++; 
            }

            if (!has_next_seed) {
            #pragma omp single
                fprintf(stderr, "no more seed to try\n");
                break;
            }

            wfc_clone_into(&blocks, next_seed, init);

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
    }
}

int
main(int argc, char **argv)
{
    omp_set_dynamic(false);

    wfc_args args             = wfc_parse_args(argc, argv);
    const wfc_blocks_ptr init = wfc_load(0, args.data_file);

    switch (args.kind) {
        case OMP:
            omp_set_num_threads((int)args.parallel);
        case CPU:
            cpu_main_loop(args, init);
            break;
        case OMP2:
            omp_set_num_threads((int)args.parallel);
            omp_main_loop(args, init);
            break;
        default:
            break;
    }

    return 0;
}
