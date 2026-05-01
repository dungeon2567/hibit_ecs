#pragma once

#include "ecs.h"
#include "test_ecs.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Heap-owning component lifecycle tests. Verifies ecs_tree_set / iterator_set /
   ecs_tree_remove / ecs_tree_rollback / ecs_tree_destroy correctly deep-copy
   on write and free buffers on destroy/discard, with no leaks or double-frees.

   Component owns a heap int buffer. copy_fn deep-clones it; destroy_fn frees
   it. Global counters track every allocation and free across both the test
   driver (caller-local payloads) and the engine (tree-internal slots) — final
   alive count must be zero at end of each test. */

typedef struct {
    int*     buf;
    uint32_t cap;
} owned_t;

static int g_owned_alloc_total_;
static int g_owned_free_total_;

static void owned_reset_counters_(void) {
    g_owned_alloc_total_ = 0;
    g_owned_free_total_  = 0;
}

static int owned_alive_(void) {
    return g_owned_alloc_total_ - g_owned_free_total_;
}

/* Caller-side helpers — populate / free a local owned_t, increment counters. */
static void owned_make_local_(owned_t* o, uint32_t cap, int seed) {
    o->buf = (int*)malloc((size_t)cap * sizeof(int));
    o->cap = cap;
    for (uint32_t i = 0; i < cap; ++i) o->buf[i] = seed + (int)i;
    g_owned_alloc_total_++;
}

static void owned_free_local_(owned_t* o) {
    if (o->buf) { free(o->buf); o->buf = NULL; o->cap = 0; g_owned_free_total_++; }
}

/* Engine hooks. */
static void owned_copy_fn_(void* dst, const void* src, size_t data_size) {
    (void)data_size;
    owned_t*       d = (owned_t*)dst;
    const owned_t* s = (const owned_t*)src;
    d->cap = s->cap;
    if (s->cap == 0) { d->buf = NULL; return; }
    d->buf = (int*)malloc((size_t)s->cap * sizeof(int));
    memcpy(d->buf, s->buf, (size_t)s->cap * sizeof(int));
    g_owned_alloc_total_++;
}

static void owned_destroy_fn_(void* element, size_t data_size) {
    (void)data_size;
    owned_t* o = (owned_t*)element;
    if (o->buf) {
        free(o->buf);
        o->buf = NULL;
        o->cap = 0;
        g_owned_free_total_++;
    }
}

static void owned_destroy_batch_fn_(void* l1_data, size_t block_size, uint64_t mask) {
    while (mask) {
        int k = ecs_ctz64(mask); mask &= mask - 1;
        owned_destroy_fn_((char*)l1_data + (size_t)k * block_size, block_size);
    }
}

static ecs_tree_t* make_owned_tree_(void) {
    ecs_tree_t* t = (ecs_tree_t*)ecs_xcalloc(1, sizeof(ecs_tree_t));
    ecs_tree_init(t, sizeof(owned_t));
    t->copy_element                = owned_copy_fn_;
    t->destroy_component_fn        = owned_destroy_fn_;
    t->destroy_component_batch_fn  = owned_destroy_batch_fn_;
    return t;
}

/* ---- 1. CONFIRMED-mode set on empty slot ---- */
static void test_owned_confirmed_set_empty(void) {
    owned_reset_counters_();
    ecs_tree_t* t = make_owned_tree_();

    owned_t a; owned_make_local_(&a, 4, 100);
    ecs_tree_set(t, 0, &a);                              /* deep-copy into confirmed */
    EXPECT(owned_alive_() == 2,  "after set: caller A + tree A_copy alive");
    owned_free_local_(&a);
    EXPECT(owned_alive_() == 1,  "after caller free: only tree A_copy alive");

    free_tree(t);                                        /* batch destroy frees A_copy */
    EXPECT(owned_alive_() == 0,  "tree destroy frees confirmed slot");
}

/* ---- 2. CONFIRMED-mode overwrite — old slot destroyed before deep-copy ---- */
static void test_owned_confirmed_overwrite(void) {
    owned_reset_counters_();
    ecs_tree_t* t = make_owned_tree_();

    owned_t a; owned_make_local_(&a, 4, 100);
    ecs_tree_set(t, 0, &a);
    owned_free_local_(&a);                               /* tree A_copy alive */

    owned_t b; owned_make_local_(&b, 8, 200);
    ecs_tree_set(t, 0, &b);                              /* must destroy A_copy then deep-copy B */
    EXPECT(owned_alive_() == 2,  "tree B_copy + caller B alive (A_copy destroyed)");
    owned_free_local_(&b);
    EXPECT(owned_alive_() == 1,  "only tree B_copy alive after caller free");

    free_tree(t);
    EXPECT(owned_alive_() == 0,  "tree destroy frees overwritten slot");
}

/* ---- 3. CONFIRMED-mode remove fires destroy hook ---- */
static void test_owned_confirmed_remove(void) {
    owned_reset_counters_();
    ecs_tree_t* t = make_owned_tree_();

    owned_t a; owned_make_local_(&a, 4, 100);
    ecs_tree_set(t, 0, &a);
    owned_free_local_(&a);                               /* tree A_copy alive */

    int rc = ecs_tree_remove(t, 0);
    EXPECT(rc == 1,              "remove returned 1");
    EXPECT(owned_alive_() == 0,  "confirmed-remove destroyed slot heap");

    free_tree(t);
    EXPECT(owned_alive_() == 0,  "no leaks");
}

/* ---- 4. PREDICT-mode set on empty + rollback frees predicted clone ---- */
static void test_owned_predict_set_empty_rollback(void) {
    owned_reset_counters_();
    ecs_tree_t* t = make_owned_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    owned_t a; owned_make_local_(&a, 4, 100);
    ecs_tree_set(t, 0, &a);
    owned_free_local_(&a);
    EXPECT(owned_alive_() == 1,  "tree A_copy alive in predicted slot");

    ecs_tree_rollback(t);                                /* discard predicted-only add */
    EXPECT(owned_alive_() == 0,  "rollback freed predicted-only clone");

    free_tree(t);
}

/* ---- 5. PREDICT-mode set on confirmed-alive + rollback ----
   Confirmed buffer must survive; predicted buffer must be freed. */
static void test_owned_predict_set_alive_rollback(void) {
    owned_reset_counters_();
    ecs_tree_t* t = make_owned_tree_();

    owned_t a; owned_make_local_(&a, 4, 100);
    ecs_tree_set(t, 0, &a);                              /* CONFIRMED: A_copy at conf slot */
    owned_free_local_(&a);

    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    owned_t b; owned_make_local_(&b, 8, 200);
    ecs_tree_set(t, 0, &b);                              /* PREDICT: B_copy at pred slot */
    owned_free_local_(&b);
    EXPECT(owned_alive_() == 2,  "A_copy (conf) + B_copy (pred) alive");

    ecs_tree_rollback(t);
    EXPECT(owned_alive_() == 1,  "rollback freed B_copy, A_copy intact");

    free_tree(t);
    EXPECT(owned_alive_() == 0,  "tree destroy freed A_copy");
}

/* ---- 6. PREDICT-mode set then set again on same slot ----
   First clone must be destroyed before second deep-copy. No leak, no double-free. */
static void test_owned_predict_set_set(void) {
    owned_reset_counters_();
    ecs_tree_t* t = make_owned_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    owned_t a; owned_make_local_(&a, 4, 100);
    ecs_tree_set(t, 0, &a);
    owned_free_local_(&a);
    EXPECT(owned_alive_() == 1,  "after first predict-set");

    owned_t b; owned_make_local_(&b, 8, 200);
    ecs_tree_set(t, 0, &b);                              /* destroys A_copy, clones B */
    owned_free_local_(&b);
    EXPECT(owned_alive_() == 1,  "after second predict-set: only B_copy alive");

    ecs_tree_rollback(t);
    EXPECT(owned_alive_() == 0,  "rollback freed B_copy");

    free_tree(t);
}

/* ---- 7. PREDICT-mode set then predict-remove + rollback ----
   Predicted clone destroyed inline at remove. Rollback must NOT double-free. */
static void test_owned_predict_set_remove(void) {
    owned_reset_counters_();
    ecs_tree_t* t = make_owned_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    owned_t a; owned_make_local_(&a, 4, 100);
    ecs_tree_set(t, 0, &a);
    owned_free_local_(&a);
    EXPECT(owned_alive_() == 1,  "after predict-set: A_copy alive");

    int rc = ecs_tree_remove(t, 0);
    EXPECT(rc == 1,              "predict-remove returned 1");
    EXPECT(owned_alive_() == 0,  "predict-remove inline-destroyed A_copy");

    ecs_tree_rollback(t);
    EXPECT(owned_alive_() == 0,  "rollback did not double-free");

    free_tree(t);
}

/* ---- 8. PREDICT-mode set, predict-remove, predict-set again ----
   Both clones must be freed exactly once. */
static void test_owned_predict_set_remove_set(void) {
    owned_reset_counters_();
    ecs_tree_t* t = make_owned_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    owned_t a; owned_make_local_(&a, 4, 100);
    ecs_tree_set(t, 0, &a);
    owned_free_local_(&a);

    ecs_tree_remove(t, 0);                               /* destroys A_copy inline */
    EXPECT(owned_alive_() == 0,  "remove freed A_copy");

    owned_t b; owned_make_local_(&b, 8, 200);
    ecs_tree_set(t, 0, &b);                              /* fresh predict-set, no destroy of stale */
    owned_free_local_(&b);
    EXPECT(owned_alive_() == 1,  "B_copy alive in predicted slot");

    ecs_tree_rollback(t);
    EXPECT(owned_alive_() == 0,  "rollback freed B_copy, no leftover");

    free_tree(t);
}

/* ---- 9. PREDICT-mode predict-remove on confirmed-alive (no prior predict-set) + rollback ----
   No inline destroy (predicted slot bytes were stale POD), no rollback destroy.
   Slot resurrects with confirmed buffer intact. */
static void test_owned_predict_remove_confirmed_rollback(void) {
    owned_reset_counters_();
    ecs_tree_t* t = make_owned_tree_();

    owned_t a; owned_make_local_(&a, 4, 100);
    ecs_tree_set(t, 0, &a);                              /* CONFIRMED: A_copy at conf slot */
    owned_free_local_(&a);
    EXPECT(owned_alive_() == 1,  "A_copy alive after seed");

    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    int rc = ecs_tree_remove(t, 0);
    EXPECT(rc == 1,              "predict-remove returned 1");
    EXPECT(owned_alive_() == 1,  "predict-remove did NOT touch confirmed buffer");

    ecs_tree_rollback(t);
    EXPECT(owned_alive_() == 1,  "rollback restored slot, A_copy intact");
    EXPECT(l1_of(t, 0, 0)->predicted_mask_any == 1ULL,
                                 "slot resurrected after rollback");

    free_tree(t);
    EXPECT(owned_alive_() == 0,  "tree destroy freed A_copy");
}

/* ---- 10. iterator_set deep-copies in CONFIRMED mode ----
   Smoke test that iterator-driven writes go through the owned path. */
static void test_owned_iterator_set_confirmed(void) {
    owned_reset_counters_();
    ecs_tree_t* t = make_owned_tree_();

    owned_t seed; owned_make_local_(&seed, 4, 100);
    ecs_tree_set(t, 0, &seed);
    owned_free_local_(&seed);

    /* Build a single-tree query touching the seeded slot. */
    ecs_compiled_query_t query = {0};
    query.tree_count         = 1;
    query.trees[0]           = t;
    query.clause_count       = 1;
    query.clauses[0].include = 1u;

    ecs_iterator_t it = {0};
    ecs_iterator_init(&it, &query);
    it.write_mask = 1u;
    int hits = 0;
    while (ecs_iterator_next(&it)) {
        owned_t newval; owned_make_local_(&newval, 8, 200);
        ecs_iterator_set(&it, 0, &newval);               /* must destroy old + deep-copy */
        owned_free_local_(&newval);
        hits++;
    }
    EXPECT(hits == 1,            "iterator visited 1 slot");
    EXPECT(owned_alive_() == 1,  "after iterator_set: only new tree clone alive");

    free_tree(t);
    EXPECT(owned_alive_() == 0,  "tree destroy freed slot");
}

/* ---- 11. tree_destroy fallback path: only single-fn destroy hook wired ---- */
static void test_owned_destroy_singlefn_fallback(void) {
    owned_reset_counters_();
    ecs_tree_t* t = (ecs_tree_t*)ecs_xcalloc(1, sizeof(ecs_tree_t));
    ecs_tree_init(t, sizeof(owned_t));
    t->copy_element                = owned_copy_fn_;
    t->destroy_component_fn        = owned_destroy_fn_;
    /* deliberately NO destroy_component_batch_fn — exercise per-slot fallback */

    owned_t a; owned_make_local_(&a, 4, 100); ecs_tree_set(t, 0, &a); owned_free_local_(&a);
    owned_t b; owned_make_local_(&b, 4, 200); ecs_tree_set(t, 1, &b); owned_free_local_(&b);
    EXPECT(owned_alive_() == 2,  "two confirmed slots alive");

    ecs_tree_destroy(t);
    ecs_free(t);
    EXPECT(owned_alive_() == 0,  "fallback per-slot destroy freed both slots");
}

static int test_owned_all(void) {
    setvbuf(stdout, NULL, _IOFBF, 65536);
    printf("=== Owned-component lifecycle tests ===\n\n");

    int before = g_failed;
    test_owned_confirmed_set_empty();
    test_owned_confirmed_overwrite();
    test_owned_confirmed_remove();
    test_owned_predict_set_empty_rollback();
    test_owned_predict_set_alive_rollback();
    test_owned_predict_set_set();
    test_owned_predict_set_remove();
    test_owned_predict_set_remove_set();
    test_owned_predict_remove_confirmed_rollback();
    test_owned_iterator_set_confirmed();
    test_owned_destroy_singlefn_fallback();

    int local_failed = g_failed - before;
    printf("\nowned: %d failed\n", local_failed);
    fflush(stdout);
    return local_failed ? 1 : 0;
}
