#pragma once

#include "ecs.h"
#include "test_ecs.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* LIST-component lifecycle tests. Verifies ecs_tree_set / ecs_tree_remove /
   ecs_tree_rollback / ecs_tree_destroy correctly deep-copy ecs_list_t slots
   and free their headers on destroy/discard, with no leaks or double-frees.

   Engine semantics under test:
     - ecs_tree_set  on LIST tree → frees old slot's h then ecs_list_assign clones src.
     - ecs_tree_remove           → ecs_free(slot->h) + clear presence (CONFIRMED).
     - ecs_tree_remove           → ecs_free(predicted slot->h) iff (predicted_mask
                                    & dirty) bit set (PREDICT mode owned-clone rule).
     - ecs_tree_rollback         → frees every (predicted_mask & dirty) victim then
                                    resets predicted to confirmed.
     - ecs_tree_destroy          → frees every live confirmed slot's h, then frees
                                    L1/L2/root nodes.

   Verification = read-back content + mask state at each step. Heap leak
   detection relies on mimalloc's process-exit reporting (ECS_DEBUG_LEAKS or
   MIMALLOC_VERBOSE) plus the structural assertions below — every test that
   sets a slot must end in a state where remove/rollback/destroy has been
   called for every live slot. */

typedef struct {
    int seed;
    uint32_t cap;
} list_recipe_t;

/* Build an ecs_list_t holding `cap` ints filled (seed + i). Caller owns the
   returned list (must ecs_list_destroy after passing into a tree — engine
   deep-copies on assign). */
static ecs_list_t make_int_list_(int seed, uint32_t cap) {
    ecs_list_t l = {0};
    for (uint32_t i = 0; i < cap; i++) {
        int v = seed + (int)i;
        ecs_list_push(&l, sizeof(int), &v);
    }
    return l;
}

static int list_int_at_(const ecs_list_t* l, uint32_t i) {
    return *(const int*)ecs_list_at(*l, sizeof(int), i);
}

static int lists_equal_(const ecs_list_t* a, const ecs_list_t* b) {
    uint32_t na = ecs_list_size(*a) / (uint32_t)sizeof(int);
    uint32_t nb = ecs_list_size(*b) / (uint32_t)sizeof(int);
    if (na != nb) return 0;
    for (uint32_t i = 0; i < na; i++)
        if (list_int_at_(a, i) != list_int_at_(b, i)) return 0;
    return 1;
}

static ecs_tree_t* make_list_tree_(void) {
    ecs_tree_t* t = (ecs_tree_t*)ecs_xcalloc(1, sizeof(ecs_tree_t));
    ecs_tree_init(t, sizeof(ecs_list_t), ECS_TREE_FLAG_LIST);
    return t;
}

/* ---- 1. CONFIRMED-mode set on empty slot, content survives ---- */
static void test_owned_confirmed_set_empty(void) {
    ecs_tree_t* t = make_list_tree_();

    ecs_list_t a = make_int_list_(100, 4);
    ecs_tree_set(t, 0, &a);

    const ecs_list_t* slot = (const ecs_list_t*)ecs_tree_get(t, 0);
    EXPECT(lists_equal_(slot, &a), "tree slot deep-clones source list");
    /* Mutating the local source must not affect the tree slot (deep-copy). */
    int extra = 999; ecs_list_push(&a, sizeof(int), &extra);
    EXPECT(ecs_list_size(*slot) != ecs_list_size(a), "tree slot independent of source");

    ecs_list_destroy(&a);
    free_tree(t);
}

/* ---- 2. CONFIRMED-mode overwrite — old slot freed before new assign ---- */
static void test_owned_confirmed_overwrite(void) {
    ecs_tree_t* t = make_list_tree_();

    ecs_list_t a = make_int_list_(100, 4);
    ecs_tree_set(t, 0, &a);
    ecs_list_destroy(&a);

    ecs_list_t b = make_int_list_(200, 8);
    ecs_tree_set(t, 0, &b);
    const ecs_list_t* slot = (const ecs_list_t*)ecs_tree_get(t, 0);
    EXPECT(lists_equal_(slot, &b), "overwritten slot reflects B");
    EXPECT(ecs_list_size(*slot) == 8u * sizeof(int), "size matches B (old A header freed)");
    ecs_list_destroy(&b);

    free_tree(t);
}

/* ---- 3. CONFIRMED-mode remove clears slot + frees header ---- */
static void test_owned_confirmed_remove(void) {
    ecs_tree_t* t = make_list_tree_();

    ecs_list_t a = make_int_list_(100, 4);
    ecs_tree_set(t, 0, &a);
    ecs_list_destroy(&a);

    int rc = ecs_tree_remove(t, 0);
    EXPECT(rc == 1, "remove returned 1");
    EXPECT(l1_of(t, 0, 0) == &ecs_default_l1 ||
           (l1_of(t, 0, 0)->predicted_mask_any & 1ULL) == 0,
           "slot 0 absent post-remove");

    free_tree(t);
}

/* ---- 4. PREDICT-mode set on empty + rollback discards predicted clone ---- */
static void test_owned_predict_set_empty_rollback(void) {
    ecs_tree_t* t = make_list_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    ecs_list_t a = make_int_list_(100, 4);
    ecs_tree_set(t, 0, &a);
    ecs_list_destroy(&a);
    EXPECT(l1_of(t, 0, 0)->predicted_mask_any == 1ULL, "slot present in predicted");
    EXPECT(l1_of(t, 0, 0)->dirty             == 1ULL, "dirty bit set");

    ecs_tree_rollback(t);
    EXPECT(l1_of(t, 0, 0) == &ecs_default_l1, "predicted-only slot reclaimed on rollback");

    free_tree(t);
}

/* ---- 5. PREDICT-mode set on confirmed-alive + rollback ----
   Confirmed slot must survive intact; predicted clone must be freed. */
static void test_owned_predict_set_alive_rollback(void) {
    ecs_tree_t* t = make_list_tree_();

    ecs_list_t a = make_int_list_(100, 4);
    ecs_tree_set(t, 0, &a);                              /* CONFIRMED: A at conf slot */
    ecs_list_destroy(&a);

    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    ecs_list_t b = make_int_list_(200, 8);
    ecs_tree_set(t, 0, &b);                              /* PREDICT: B at pred slot */
    ecs_list_destroy(&b);

    const ecs_list_t* visible_pre = (const ecs_list_t*)ecs_tree_get(t, 0);
    EXPECT(ecs_list_size(*visible_pre) == 8u * sizeof(int), "pre-rollback visible = B");

    ecs_tree_rollback(t);
    const ecs_list_t* visible_post = (const ecs_list_t*)ecs_tree_get(t, 0);
    EXPECT(ecs_list_size(*visible_post) == 4u * sizeof(int), "post-rollback visible = A");
    EXPECT(list_int_at_(visible_post, 0) == 100, "A content intact");

    free_tree(t);
}

/* ---- 6. PREDICT-mode set then set again on same slot ----
   First clone must be freed before second deep-copy. No leak, no double-free. */
static void test_owned_predict_set_set(void) {
    ecs_tree_t* t = make_list_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    ecs_list_t a = make_int_list_(100, 4);
    ecs_tree_set(t, 0, &a);
    ecs_list_destroy(&a);

    ecs_list_t b = make_int_list_(200, 8);
    ecs_tree_set(t, 0, &b);                              /* frees A clone, clones B */
    ecs_list_destroy(&b);

    const ecs_list_t* slot = (const ecs_list_t*)ecs_tree_get(t, 0);
    EXPECT(ecs_list_size(*slot) == 8u * sizeof(int), "slot reflects B");
    EXPECT(list_int_at_(slot, 0) == 200, "B content");

    ecs_tree_rollback(t);                                /* frees B clone */
    free_tree(t);
}

/* ---- 7. PREDICT-mode set then predict-remove + rollback ----
   Predicted clone freed inline at remove. Rollback must NOT double-free. */
static void test_owned_predict_set_remove(void) {
    ecs_tree_t* t = make_list_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    ecs_list_t a = make_int_list_(100, 4);
    ecs_tree_set(t, 0, &a);
    ecs_list_destroy(&a);

    int rc = ecs_tree_remove(t, 0);
    EXPECT(rc == 1, "predict-remove returned 1");

    ecs_tree_rollback(t);
    /* Predicted-only add removed → slot reclaimed. */
    EXPECT(l1_of(t, 0, 0) == &ecs_default_l1, "L1 reclaimed (predicted-only set+remove)");

    free_tree(t);
}

/* ---- 8. PREDICT-mode set, predict-remove, predict-set again ----
   Both clones must be freed exactly once. */
static void test_owned_predict_set_remove_set(void) {
    ecs_tree_t* t = make_list_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    ecs_list_t a = make_int_list_(100, 4);
    ecs_tree_set(t, 0, &a);
    ecs_list_destroy(&a);

    ecs_tree_remove(t, 0);                               /* frees A clone inline */

    ecs_list_t b = make_int_list_(200, 8);
    ecs_tree_set(t, 0, &b);                              /* fresh predict-set, no double-destroy */
    ecs_list_destroy(&b);

    const ecs_list_t* slot = (const ecs_list_t*)ecs_tree_get(t, 0);
    EXPECT(ecs_list_size(*slot) == 8u * sizeof(int), "slot now B");

    ecs_tree_rollback(t);                                /* frees B clone */
    free_tree(t);
}

/* ---- 9. PREDICT-mode predict-remove on confirmed-alive (no prior predict-set) + rollback ----
   No inline free (predicted slot bytes were stale POD), no rollback free.
   Slot resurrects with confirmed clone intact. */
static void test_owned_predict_remove_confirmed_rollback(void) {
    ecs_tree_t* t = make_list_tree_();

    ecs_list_t a = make_int_list_(100, 4);
    ecs_tree_set(t, 0, &a);                              /* CONFIRMED: A at conf slot */
    ecs_list_destroy(&a);

    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    int rc = ecs_tree_remove(t, 0);
    EXPECT(rc == 1, "predict-remove returned 1");

    ecs_tree_rollback(t);
    EXPECT(l1_of(t, 0, 0)->predicted_mask_any == 1ULL, "slot resurrected after rollback");
    const ecs_list_t* slot = (const ecs_list_t*)ecs_tree_get(t, 0);
    EXPECT(ecs_list_size(*slot) == 4u * sizeof(int) && list_int_at_(slot, 0) == 100,
           "A content intact after predict-remove + rollback");

    free_tree(t);
}

/* ---- 10. tree_destroy walks live confirmed slots and frees headers ---- */
static void test_owned_destroy_walk(void) {
    ecs_tree_t* t = make_list_tree_();

    ecs_list_t a = make_int_list_(100, 4); ecs_tree_set(t, 0, &a);    ecs_list_destroy(&a);
    ecs_list_t b = make_int_list_(200, 4); ecs_tree_set(t, 1, &b);    ecs_list_destroy(&b);
    ecs_list_t c = make_int_list_(300, 4); ecs_tree_set(t, 4096, &c); ecs_list_destroy(&c);  /* L2 != 0 */

    /* Implicit verification: free_tree → ecs_tree_destroy frees A_h, B_h, C_h.
       Leak detection via mimalloc process-exit; functional check below. */
    EXPECT(l1_of(t, 0, 0)->predicted_mask_any & 0x3, "slots 0+1 in same L1");
    EXPECT(l1_of(t, 1, 0)->predicted_mask_any & 0x1, "slot 4096 in (l3=1, l2=0)");
    free_tree(t);
}

static int test_owned_all(void) {
    setvbuf(stdout, NULL, _IOFBF, 65536);
    printf("=== LIST-component lifecycle tests ===\n\n");

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
    test_owned_destroy_walk();

    int local_failed = g_failed - before;
    printf("\nowned: %d failed\n", local_failed);
    fflush(stdout);
    return local_failed ? 1 : 0;
}
