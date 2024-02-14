#pragma once

#include "types.h"
#include <stdbool.h>
#include <stdlib.h>

typedef struct position_list_node {
    position value;
    struct position_list_node* next;
} position_list_node;

typedef struct position_list {
    position_list_node* first;
    position_list_node* last;
} position_list;

static inline position_list position_list_init() {
    position_list list = { NULL, NULL };
    return list;
}

static inline bool position_list_is_empty(const position_list* list) {
    return list->first == NULL;
}

static inline void position_list_push(position_list* list, position v) {
    position_list_node* node = malloc(sizeof(position_list_node));
    node->value = v;
    node->next = NULL;

    if (position_list_is_empty(list)) {
        list->first = node;
    } else {
        list->last->next = node;
    }

    list->last = node;
}

static inline position position_list_pop(position_list* list) {
    if (position_list_is_empty(list)) {
        abort();
    }

    position_list_node* front = list->first;

    list->first = front->next;
    if (list->first == NULL) {
        list->last = NULL;
    }

    position v = front->value;
    free(front);

    return v;
}
