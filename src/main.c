#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"
#include "wfc_omp.h"

#include <float.h>
#include <stdatomic.h>
#include <stdio.h>
#include <omp.h>
#include <pthread.h>

int
main(int argc, char **argv)
{
    omp_set_dynamic(false);

    wfc_args args             = wfc_parse_args(argc, argv);
    const wfc_blocks_ptr init = wfc_load(0, args.data_file);

    // bool quit                = false;
    _Atomic int quit         = 0;
    _Atomic uint64_t iterations      = 0;
    wfc_blocks_ptr blocks    = NULL;
    pthread_mutex_t seed_mtx = PTHREAD_MUTEX_INITIALIZER;

    // bool *volatile const quit_ptr           = &quit;
    // _Atomic int *volatile quit_ptr           = &quit;
    // uint64_t *volatile const iterations_ptr = &iterations;

    const uint64_t max_iterations = count_seeds(args.seeds);
    const double start            = omp_get_wtime();

    while (!atomic_load(&quit)) {
        pthread_mutex_lock(&seed_mtx);
        uint64_t next_seed       = 0;
        const bool has_next_seed = try_next_seed(&args.seeds, &next_seed);
        pthread_mutex_unlock(&seed_mtx);

        if (!has_next_seed) {
            atomic_fetch_or_explicit(&quit, true, memory_order_seq_cst);
            fprintf(stderr, "no more seed to try\n");
            break;
        }

        wfc_clone_into(&blocks, next_seed, init);
        const bool solved = args.solver(blocks);
        atomic_fetch_add_explicit(&iterations, 1, memory_order_seq_cst);

        if (solved && args.output_folder != NULL) {
            atomic_fetch_or_explicit(&quit, true, memory_order_seq_cst);
            fprintf(stdout, "\nsuccess with seed %lu\n", blocks->seed);
            wfc_save_into(blocks, args.data_file, args.output_folder);
            
        }

        else if (solved) {
            atomic_fetch_or_explicit(&quit, true, memory_order_seq_cst);
            fputs("\nsuccess with result:\n", stdout);
            abort();
        }

        else if (!atomic_load(&quit)) {
            fprintf(stdout, "\r%.2f%% -> %.2fs",
                    ((double)(atomic_load(&iterations)) / (double)(max_iterations)) * 100.0,
                    omp_get_wtime() - start);
        }
    }

    return 0;
}
