#define _GNU_SOURCE

#include "wfc.h"

#include <string.h>
#include <strings.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>

typedef struct {
    enum {
        seed_item_single = 1,
        seed_item_tuple  = 2,
    } type;

    union {
        /// Only when `.type == seed_item_single`
        uint32_t single;

        /// Only when `.type == seed_item_tuple`
        struct {
            uint32_t from;
            uint32_t to;
        };
    } content;
} seed_item;

typedef struct seeds_list {
    uint64_t count;
    uint64_t size;
    seed_item items[];
} seeds_list;

uint64_t
count_seeds(const seeds_list *restrict const seeds)
{
    uint64_t total = 0;
    for (uint64_t i = 0; i < seeds->count; i += 1) {
        switch (seeds->items[i].type) {
        case seed_item_single: total += 1; break;
        case seed_item_tuple: {
            total += (seeds->items[i].content.to - seeds->items[i].content.from + 1);
            break;
        }
        }
    }
    return total;
}

static inline seeds_list *
seeds_list_push_item(seeds_list *restrict list, seed_item item)
{
    // First call, need to allocate the thing.
    if (NULL == list) {
        static const uint64_t DEFAULT_SIZE = 10;
        list                               = malloc(sizeof(seeds_list) + DEFAULT_SIZE * sizeof(seed_item));
        if (NULL == list) {
            fprintf(stderr, "failed to allocate seeds list\n");
            exit(EXIT_FAILURE);
        }
        list->size     = DEFAULT_SIZE;
        list->count    = 1;
        list->items[0] = item;
    }

    // Already exists, but need to reallocate
    else if (list->size <= list->count) {
        static const uint64_t GROWTH_FACTOR = 2;
        const uint64_t new_list_size        = list->size * GROWTH_FACTOR;
        list                                = realloc(list, sizeof(seeds_list) + new_list_size * sizeof(seed_item));
        if (NULL == list) {
            fprintf(stderr, "failed to realloc the seeds list\n");
            exit(EXIT_FAILURE);
        }
        list->size               = new_list_size;
        list->items[list->count] = item;
        list->count += 1;
        abort();
    }

    // Already exists, don't need to reallocate
    else {
        list->items[list->count] = item;
        list->count += 1;
    }

    return list;
}

static inline void
list_pop(seeds_list *restrict *const list_ptr)
{
    seeds_list *restrict list = *list_ptr;
    if (NULL == list) {
        return;
    } else if (list->size >= 2) {
        memmove(&list->items[0], &list->items[1], (list->count - 1) * sizeof(seed_item));
        list->size -= 1;
    } else {
        free(list);
        *list_ptr = NULL;
    }
}

static inline uint64_t
seeds_list_pop(seeds_list *restrict *const list_ptr)
{
    if (NULL == list_ptr) {
        return UINT64_MAX;
    }
    seeds_list *restrict list = *list_ptr;

    // Nothing to see
    if (NULL == list) {
        return UINT64_MAX;
    } else if (0 == list->count) {
        free(list);
        *list_ptr = NULL;
        return UINT64_MAX;
    }

    // Handle tuples
    else if (list->items[0].type == seed_item_tuple) {
        const uint64_t ret = list->items[0].content.from;
        list->items[0].content.from += 1;
        if (list->items[0].content.from >= list->items[0].content.to) {
            list_pop(list_ptr);
        }
        return ret;
    }

    // Handle single items
    else if (list->items[0].type == seed_item_single) {
        const uint64_t ret = list->items[0].content.single;
        list_pop(list_ptr);
        return ret;
    }

    // Oupsy doupsy
    else {
        abort();
    }
}

bool
try_next_seed(seeds_list *restrict *const seeds, uint64_t *restrict return_seed)
{
    *return_seed = seeds_list_pop(seeds);
    return UINT64_MAX != *return_seed;
}

_Noreturn static inline void
print_help(const char *exec)
{
    fprintf(stdout, "usage: %s [-h] [-o folder/] [-l solver] [-p count] [-s seeds...] <path/to/file.data>\n", exec);
    puts("  -h          print this help message.");
    puts("  -o          output folder to save solutions. adds the seed to the data file name.");
    puts("  -p count    number of seeds that can be processed in parallel");
    puts("  -s seeds    seeds to use. can an integer or a range: `from-to`.");

    fprintf(stdout, "  -l solver   solver to use. possible values are:");
    for (unsigned long i = 0; i < sizeof(solvers) / sizeof(wfc_solver); i += 1) {
        (i == 0) ? fprintf(stdout, " [%s]", solvers[i].name)
                 : fprintf(stdout, ", %s", solvers[i].name);
    }
    fputc('\n', stdout);

    exit(EXIT_SUCCESS);
}

static inline uint32_t
to_u32(const char *arg, char **end)
{
    const long value = strtol(arg, end, 10);
    if (value < 0) {
        fprintf(stderr, "negative seeds are not possible: %ld\n", value);
        exit(EXIT_FAILURE);
    }
    return (uint32_t)value;
}

static inline bool
str_ends_with(const char str[], const char suffix[])
{
    size_t lenstr    = strlen(str);
    size_t lensuffix = strlen(suffix);
    if (lensuffix > lenstr)
        return 0;
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

wfc_args
wfc_parse_args(int argc, char **argv)
{
    int opt;
    seeds_list *restrict seeds     = NULL;
    const char *output             = NULL;
    uint64_t parallel              = 1;
    bool (*solver)(wfc_blocks_ptr) = NULL;
    char *end                      = NULL;

    while ((opt = getopt(argc, argv, "hs:o:l:p:")) != -1) {
        switch (opt) {
        case 's': {
            const uint32_t from = to_u32(optarg, &end);
            if ('\0' == *end) {
                seeds = seeds_list_push_item(seeds, (seed_item){ .type    = seed_item_single,
                                                                 .content = { .single = from } });
            } else if ('-' == *end) {
                const uint32_t to = to_u32(end + 1, &end);
                if (*end != '\0') {
                    fprintf(stderr, "failed to get the upper value of the range\n");
                    exit(EXIT_FAILURE);
                } else if (from >= to) {
                    fprintf(stderr, "invalid range: %u >= %u\n", from, to);
                    exit(EXIT_FAILURE);
                }
                seeds = seeds_list_push_item(
                    seeds, (seed_item){ .type    = seed_item_tuple,
                                        .content = { .from = from, .to = to } });
            } else {
                fprintf(stderr, "invalid range delimiter: '%c'\n", *end);
                exit(EXIT_FAILURE);
            }
            break;
        }

        case 'l': {
            for (uint64_t i = 0; i < sizeof(solvers) / sizeof(wfc_solver); i += 1) {
                if (0 == strcasecmp(optarg, solvers[i].name)) {
                    solver = solvers[i].function;
                }
            }
            if (NULL == solver) {
                fprintf(stderr, "unknown solver `%s`\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        }

        case 'p': {
            if ((parallel = to_u32(optarg, &end)) <= 0) {
                fputs("you must at least process one seed at a time...", stderr);
                exit(EXIT_FAILURE);
            } else if ('\0' != *end) {
                fprintf(stderr, "invalid integer: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        }

        case 'o': {
            struct stat sb;
            if (stat(optarg, &sb) == 0 && S_ISDIR(sb.st_mode)) {
                output = optarg;
                break;
            } else {
                fprintf(stderr, "folder `%s` doesn't exist\n", optarg);
                exit(EXIT_FAILURE);
            }
        }

        case 'h':
        default: print_help(argv[0]);
        }
    }

    if (optind >= argc) {
        print_help(argv[0]);
    } else if (!str_ends_with(argv[optind], ".data")) {
        fprintf(stderr, "expected the suffix `.data` for the data file: %s\n", argv[optind]);
        exit(EXIT_FAILURE);
    }

    return (wfc_args){
        .data_file     = argv[optind],
        .seeds         = seeds,
        .output_folder = output,
        .parallel      = parallel,
        .solver        = (NULL == solver) ? solvers[0].function : solver,
    };
}
