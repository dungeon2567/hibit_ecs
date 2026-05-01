#pragma once

#include <stdio.h>
#include <stdint.h>
#include "ecs.h"

/* Reuses g_passed / g_failed / EXPECT from test_ecs.h.
   test_ecs.h must be included before this file. */

static void test_list_empty(void) {
    ecs_list_t l = {0};
    EXPECT(l.h == NULL,                       "zero-init list has NULL header");
    EXPECT(ecs_list_size(l) == 0u,            "empty list size == 0");
    EXPECT(ecs_list_capacity(l) == 0u,        "empty list capacity == 0");
    ecs_list_destroy(&l);
    EXPECT(l.h == NULL,                       "destroy on empty leaves NULL");
}

static void test_list_push_int(void) {
    ecs_list_t l = {0};

    int v0 = 42, v1 = -7, v2 = 1000;
    int* p0 = (int*)ecs_list_push(&l, sizeof(int), &v0);
    int* p1 = (int*)ecs_list_push(&l, sizeof(int), &v1);
    int* p2 = (int*)ecs_list_push(&l, sizeof(int), &v2);
    EXPECT(p0 && p1 && p2,                              "push returns non-NULL slots");
    EXPECT(ecs_list_size(l) == 3u * sizeof(int),        "size in bytes after 3 pushes");
    EXPECT(ecs_list_capacity(l) >= 3u * sizeof(int),    "capacity covers size");

    EXPECT(*(int*)ecs_list_at(l, sizeof(int), 0) == 42,    "at(0) == 42");
    EXPECT(*(int*)ecs_list_at(l, sizeof(int), 1) == -7,    "at(1) == -7");
    EXPECT(*(int*)ecs_list_at(l, sizeof(int), 2) == 1000,  "at(2) == 1000");

    ecs_list_destroy(&l);
    EXPECT(l.h == NULL, "destroy clears header");
}

static void test_list_push_null_value(void) {
    /* push with value==NULL returns slot pointer without copying. */
    ecs_list_t l = {0};
    int* slot = (int*)ecs_list_push(&l, sizeof(int), NULL);
    EXPECT(slot != NULL,                             "push(NULL) returns slot ptr");
    *slot = 0xC0FFEE;
    EXPECT(*(int*)ecs_list_at(l, sizeof(int), 0) == 0xC0FFEE,
                                                     "in-place write through returned slot");
    EXPECT(ecs_list_size(l) == sizeof(int),          "size advances even on NULL push");
    ecs_list_destroy(&l);
}

static void test_list_growth(void) {
    /* Base capacity 8 bytes, grows 1.5x. With 8-byte elements:
       cap=8 holds 1 elem; second push triggers grow to >=16. */
    ecs_list_t l = {0};
    uint64_t a = 0x1111111111111111ULL;
    uint64_t b = 0x2222222222222222ULL;
    uint64_t c = 0x3333333333333333ULL;

    ecs_list_push(&l, sizeof(uint64_t), &a);
    EXPECT(ecs_list_capacity(l) >= sizeof(uint64_t),  "cap >= 8 after first push");
    uint32_t cap1 = ecs_list_capacity(l);

    ecs_list_push(&l, sizeof(uint64_t), &b);
    EXPECT(ecs_list_capacity(l) > cap1,               "capacity grows on overflow");

    ecs_list_push(&l, sizeof(uint64_t), &c);
    EXPECT(*(uint64_t*)ecs_list_at(l, sizeof(uint64_t), 0) == a, "growth preserves [0]");
    EXPECT(*(uint64_t*)ecs_list_at(l, sizeof(uint64_t), 1) == b, "growth preserves [1]");
    EXPECT(*(uint64_t*)ecs_list_at(l, sizeof(uint64_t), 2) == c, "growth preserves [2]");

    ecs_list_destroy(&l);
}

static void test_list_reserve(void) {
    ecs_list_t l = {0};
    ecs_list_reserve(&l, 64);
    EXPECT(ecs_list_capacity(l) >= 64u,  "reserve sets capacity >= request");
    EXPECT(ecs_list_size(l) == 0u,       "reserve does not change size");

    uint32_t cap_after = ecs_list_capacity(l);
    ecs_list_reserve(&l, 32);
    EXPECT(ecs_list_capacity(l) == cap_after, "reserve below current is no-op");

    /* Header alignment: data must be 8-byte aligned (FAM at offset 8). */
    EXPECT(((uintptr_t)l.h->data & 7u) == 0u, "data pointer 8-byte aligned");

    ecs_list_destroy(&l);
}

static void test_list_pop(void) {
    ecs_list_t l = {0};
    int v0 = 1, v1 = 2, v2 = 3;
    ecs_list_push(&l, sizeof(int), &v0);
    ecs_list_push(&l, sizeof(int), &v1);
    ecs_list_push(&l, sizeof(int), &v2);

    uint32_t cap = ecs_list_capacity(l);
    ecs_list_pop(l, sizeof(int));
    EXPECT(ecs_list_size(l) == 2u * sizeof(int),  "pop shrinks size by elem_size");
    EXPECT(ecs_list_capacity(l) == cap,           "pop does not shrink capacity");
    EXPECT(*(int*)ecs_list_at(l, sizeof(int), 1) == 2, "remaining tail intact");

    ecs_list_pop(l, sizeof(int));
    ecs_list_pop(l, sizeof(int));
    EXPECT(ecs_list_size(l) == 0u,                "pop down to empty");

    ecs_list_destroy(&l);
}

static void test_list_swap_remove(void) {
    /* swap_remove(i): copies last over slot i, shrinks. */
    ecs_list_t l = {0};
    int v[5] = { 10, 20, 30, 40, 50 };
    for (int i = 0; i < 5; ++i) ecs_list_push(&l, sizeof(int), &v[i]);

    /* Remove index 1 (value 20). Last is 50; expect [10, 50, 30, 40]. */
    ecs_list_swap_remove(l, sizeof(int), 1);
    EXPECT(ecs_list_size(l) == 4u * sizeof(int),       "swap_remove shrinks size");
    EXPECT(*(int*)ecs_list_at(l, sizeof(int), 0) == 10, "swap_remove keeps [0]");
    EXPECT(*(int*)ecs_list_at(l, sizeof(int), 1) == 50, "swap_remove moves last to slot");
    EXPECT(*(int*)ecs_list_at(l, sizeof(int), 2) == 30, "swap_remove keeps [2]");
    EXPECT(*(int*)ecs_list_at(l, sizeof(int), 3) == 40, "swap_remove keeps [3]");

    /* Remove last index (off == last): no memcpy path. */
    ecs_list_swap_remove(l, sizeof(int), 3);
    EXPECT(ecs_list_size(l) == 3u * sizeof(int),       "swap_remove of last shrinks");
    EXPECT(*(int*)ecs_list_at(l, sizeof(int), 2) == 30, "tail untouched after remove-last");

    ecs_list_destroy(&l);
}

static void test_list_clear(void) {
    ecs_list_t l = {0};
    int v = 7;
    ecs_list_push(&l, sizeof(int), &v);
    ecs_list_push(&l, sizeof(int), &v);
    uint32_t cap = ecs_list_capacity(l);

    ecs_list_clear(l);
    EXPECT(ecs_list_size(l) == 0u,        "clear sets size to 0");
    EXPECT(ecs_list_capacity(l) == cap,   "clear preserves capacity");

    ecs_list_clear((ecs_list_t){0});      /* clear on empty is a no-op */

    /* Reuse after clear. */
    int w = 99;
    ecs_list_push(&l, sizeof(int), &w);
    EXPECT(*(int*)ecs_list_at(l, sizeof(int), 0) == 99, "push after clear writes to slot 0");

    ecs_list_destroy(&l);
}

static void test_list_struct_payload(void) {
    typedef struct { float x, y, z; uint32_t id; } Item;
    ecs_list_t l = {0};

    Item a = { 1.0f, 2.0f, 3.0f, 100u };
    Item b = { 4.0f, 5.0f, 6.0f, 200u };
    Item c = { 7.0f, 8.0f, 9.0f, 300u };

    ecs_list_push(&l, sizeof(Item), &a);
    ecs_list_push(&l, sizeof(Item), &b);
    ecs_list_push(&l, sizeof(Item), &c);

    Item* got = (Item*)ecs_list_at(l, sizeof(Item), 1);
    EXPECT(got->id == 200u && got->y == 5.0f, "struct payload round-trips");

    ecs_list_swap_remove(l, sizeof(Item), 0);
    Item* moved = (Item*)ecs_list_at(l, sizeof(Item), 0);
    EXPECT(moved->id == 300u, "swap_remove on struct payload");

    ecs_list_destroy(&l);
}

static int test_list_all(void) {
    int before = g_failed;
    printf("=== ecs_list_t tests ===\n\n");
    test_list_empty();
    test_list_push_int();
    test_list_push_null_value();
    test_list_growth();
    test_list_reserve();
    test_list_pop();
    test_list_swap_remove();
    test_list_clear();
    test_list_struct_payload();
    int failed = g_failed - before;
    printf("\nlist: %d failed\n", failed);
    return failed ? 1 : 0;
}
