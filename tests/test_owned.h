#pragma once

#include "ecs.h"
#include "test_ecs.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* BUFFER-component lifecycle tests. Verifies element-level ecs_tree_buffer_push/
   pop/clear + ecs_tree_remove / ecs_tree_rollback / ecs_tree_destroy correctly
   manage slot buffers across CONFIRMED + PREDICT modes with no leaks /
   double-frees.

   COW model under test:
     - CONFIRMED push: mutates confirmed buffer in place. Survives across ticks
                       (no free-on-tick).
     - PREDICT first push on confirmed-alive slot: deep-copies confirmed bytes
                       into predicted slot ONCE, then mutates predicted in place.
                       Subsequent pushes/pops in same tick: zero copies.
     - PREDICT first push on virgin slot: predicted slot allocs fresh buffer.
     - PREDICT clear: drops predicted buffer (does not seed from confirmed).
     - ecs_tree_remove (CONFIRMED) : ecs_free(slot->h) + clear presence.
     - ecs_tree_remove (PREDICT)   : ecs_free(predicted slot->h) iff
                                     (predicted_mask & dirty) bit set.
     - ecs_tree_rollback           : frees every (predicted_mask & dirty) clone,
                                     resets predicted = confirmed.
     - ecs_tree_destroy            : frees every live confirmed slot's h.

   Heap leak detection relies on mimalloc's process-exit reporting + the
   structural assertions below Ã¢â‚¬â€ every test that touches a slot ends with
   remove/rollback/destroy reaching every live buffer. */

static ecs_tree_t* make_buffer_tree_(void) {
    ecs_tree_t* t = (ecs_tree_t*)ecs_xcalloc(1, sizeof(ecs_tree_t));
    ecs_tree_init(t, sizeof(ecs_buffer_t), ECS_TREE_FLAG_BUFFER);
    return t;
}

static void push_ints_(ecs_tree_t* t, int idx, int seed, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        int v = seed + (int)i;
        ecs_tree_buffer_push(t, idx, sizeof(int), &v);
    }
}

static uint32_t slot_count_(const ecs_tree_t* t, int idx) {
    const ecs_buffer_t* slot = (const ecs_buffer_t*)ecs_tree_get(t, idx);
    return ecs_buffer_size(*slot) / (uint32_t)sizeof(int);
}

static int slot_at_(const ecs_tree_t* t, int idx, uint32_t i) {
    const ecs_buffer_t* slot = (const ecs_buffer_t*)ecs_tree_get(t, idx);
    return *(const int*)ecs_buffer_at(*slot, sizeof(int), i);
}

/* ---- 1. CONFIRMED-mode push grows confirmed buffer in place ---- */
static void test_owned_confirmed_push(void) {
    ecs_tree_t* t = make_buffer_tree_();

    push_ints_(t, 0, 100, 4);
    EXPECT(slot_count_(t, 0) == 4u, "slot has 4 elems");
    EXPECT(slot_at_(t, 0, 0) == 100 && slot_at_(t, 0, 3) == 103, "elements 100..103");

    /* CONFIRMED push survives across rollback (rollback only discards
       predicted; confirmed buffer keeps its allocation). */
    ecs_tree_rollback(t);
    EXPECT(slot_count_(t, 0) == 4u, "confirmed buffer survives rollback");

    /* Further pushes append to the same buffer (no churn). */
    push_ints_(t, 0, 200, 2);
    EXPECT(slot_count_(t, 0) == 6u, "confirmed buffer extended in place");
    EXPECT(slot_at_(t, 0, 5) == 201, "tail = 201");

    free_tree(t);
}

/* ---- 2. CONFIRMED-mode clear zeroes size, keeps capacity ---- */
static void test_owned_confirmed_clear(void) {
    ecs_tree_t* t = make_buffer_tree_();

    push_ints_(t, 0, 100, 4);
    const ecs_buffer_t* before = (const ecs_buffer_t*)ecs_tree_get(t, 0);
    uint32_t cap_before = ecs_buffer_capacity(*before);
    EXPECT(cap_before > 0u, "capacity allocated post-push");

    ecs_tree_buffer_clear(t, 0);
    EXPECT(slot_count_(t, 0) == 0u, "post-clear size 0");
    const ecs_buffer_t* after = (const ecs_buffer_t*)ecs_tree_get(t, 0);
    EXPECT(ecs_buffer_capacity(*after) == cap_before, "capacity preserved across clear");

    /* Push again Ã¢â‚¬â€ should reuse the buffer (cap unchanged for small refills). */
    push_ints_(t, 0, 999, 2);
    EXPECT(slot_count_(t, 0) == 2u, "refill after clear");

    free_tree(t);
}

/* ---- 3. CONFIRMED-mode remove clears slot + frees header ---- */
static void test_owned_confirmed_remove(void) {
    ecs_tree_t* t = make_buffer_tree_();

    push_ints_(t, 0, 100, 4);

    int rc = ecs_tree_remove(t, 0);
    EXPECT(rc == 1, "remove returned 1");
    EXPECT(l1_of(t, 0, 0) == &ecs_default_l1 ||
           (l1_of(t, 0, 0)->predicted_mask_any & 1ULL) == 0,
           "slot 0 absent post-remove");

    free_tree(t);
}

/* ---- 4. PREDICT-mode push on virgin slot + rollback discards clone ---- */
static void test_owned_predict_push_empty_rollback(void) {
    ecs_tree_t* t = make_buffer_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    push_ints_(t, 0, 100, 4);
    EXPECT(l1_of(t, 0, 0)->predicted_mask_any == 1ULL, "slot present in predicted");
    EXPECT(l1_of(t, 0, 0)->dirty             == 1ULL, "dirty bit set");
    EXPECT(slot_count_(t, 0) == 4u, "pred slot has 4 elems");

    ecs_tree_rollback(t);
    EXPECT(l1_of(t, 0, 0) == &ecs_default_l1, "predicted-only slot reclaimed on rollback");

    free_tree(t);
}

/* ---- 5. PREDICT-mode push extends confirmed (one COW) + rollback restores ----
   First predict push deep-copies confirmed Ã¢â€ â€™ predicted (4 elems), then appends.
   Rollback frees predicted clone, confirmed buffer survives intact. */
static void test_owned_predict_push_alive_rollback(void) {
    ecs_tree_t* t = make_buffer_tree_();

    push_ints_(t, 0, 100, 4);                                 /* CONFIRMED: A = [100..103] */
    ecs_tree_rollback(t);                                     /* finalize confirmed tick */

    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    push_ints_(t, 0, 200, 8);                                 /* PREDICT: COW + append B = [200..207] */

    EXPECT(slot_count_(t, 0) == 12u, "pred slot = A (4) + B (8)");
    EXPECT(slot_at_(t, 0, 0)  == 100, "head still A[0]");
    EXPECT(slot_at_(t, 0, 4)  == 200, "B starts at idx 4");
    EXPECT(slot_at_(t, 0, 11) == 207, "B tail at idx 11");

    ecs_tree_rollback(t);
    EXPECT(slot_count_(t, 0) == 4u, "post-rollback = confirmed (A only)");
    EXPECT(slot_at_(t, 0, 0) == 100 && slot_at_(t, 0, 3) == 103,
           "A content intact post-rollback");

    free_tree(t);
}

/* ---- 6. PREDICT push then push: only one COW, no leak ---- */
static void test_owned_predict_push_push(void) {
    ecs_tree_t* t = make_buffer_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    push_ints_(t, 0, 100, 4);                                 /* virgin: alloc fresh pred buffer */
    push_ints_(t, 0, 200, 8);                                 /* dirty=1: extend in place */

    EXPECT(slot_count_(t, 0) == 12u, "12 elems after two pushes");
    EXPECT(slot_at_(t, 0, 4) == 200, "second push appended");

    ecs_tree_rollback(t);                                     /* frees predicted clone */
    free_tree(t);
}

/* ---- 7. PREDICT push then predict-remove + rollback ----
   Predicted clone freed inline at remove. Rollback must NOT double-free. */
static void test_owned_predict_push_remove(void) {
    ecs_tree_t* t = make_buffer_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    push_ints_(t, 0, 100, 4);

    int rc = ecs_tree_remove(t, 0);
    EXPECT(rc == 1, "predict-remove returned 1");

    ecs_tree_rollback(t);
    EXPECT(l1_of(t, 0, 0) == &ecs_default_l1, "L1 reclaimed (predicted-only push+remove)");

    free_tree(t);
}

/* ---- 8. PREDICT push, predict-remove, predict-push again ----
   Both clones freed exactly once. */
static void test_owned_predict_push_remove_push(void) {
    ecs_tree_t* t = make_buffer_tree_();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    push_ints_(t, 0, 100, 4);                                 /* alloc clone A */
    ecs_tree_remove(t, 0);                                    /* frees clone A inline */

    push_ints_(t, 0, 200, 8);                                 /* fresh clone B */

    EXPECT(slot_count_(t, 0) == 8u, "slot now B (8 elems)");
    EXPECT(slot_at_(t, 0, 0) == 200, "B content");

    ecs_tree_rollback(t);                                     /* frees clone B */
    free_tree(t);
}

/* ---- 9. PREDICT-mode predict-remove on confirmed-alive (no prior predict-push) + rollback ----
   No inline free (predicted slot bytes were stale POD post-remove), no rollback free.
   Slot resurrects with confirmed buffer intact. */
static void test_owned_predict_remove_confirmed_rollback(void) {
    ecs_tree_t* t = make_buffer_tree_();

    push_ints_(t, 0, 100, 4);                                 /* CONFIRMED: A at conf slot */
    ecs_tree_rollback(t);

    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    int rc = ecs_tree_remove(t, 0);
    EXPECT(rc == 1, "predict-remove returned 1");

    ecs_tree_rollback(t);
    EXPECT(l1_of(t, 0, 0)->predicted_mask_any == 1ULL, "slot resurrected after rollback");
    EXPECT(slot_count_(t, 0) == 4u && slot_at_(t, 0, 0) == 100,
           "A content intact after predict-remove + rollback");

    free_tree(t);
}

/* ---- 10. PREDICT-mode clear drops predicted buffer (no COW) + rollback restores ----
   Clear in PREDICT first-touch must NOT seed from confirmed. Rollback restores
   confirmed view intact. */
static void test_owned_predict_clear_alive_rollback(void) {
    ecs_tree_t* t = make_buffer_tree_();

    push_ints_(t, 0, 100, 4);                                 /* CONFIRMED A */
    ecs_tree_rollback(t);

    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    ecs_tree_buffer_clear(t, 0);                                /* pred->h = NULL */
    EXPECT(slot_count_(t, 0) == 0u, "pred view empty post-clear");

    push_ints_(t, 0, 999, 2);                                 /* push on cleared pred */
    EXPECT(slot_count_(t, 0) == 2u, "post-clear push starts fresh");
    EXPECT(slot_at_(t, 0, 0) == 999, "fresh content (no A inheritance)");

    ecs_tree_rollback(t);
    EXPECT(slot_count_(t, 0) == 4u, "post-rollback = confirmed A");
    EXPECT(slot_at_(t, 0, 0) == 100, "A content restored");

    free_tree(t);
}

/* ---- 11. tree_destroy walks live confirmed slots and frees headers ---- */
static void test_owned_destroy_walk(void) {
    ecs_tree_t* t = make_buffer_tree_();

    push_ints_(t, 0,    100, 4);
    push_ints_(t, 1,    200, 4);
    push_ints_(t, 4096, 300, 4);                              /* L2 != 0 */

    EXPECT(l1_of(t, 0, 0)->predicted_mask_any & 0x3, "slots 0+1 in same L1");
    EXPECT(l1_of(t, 1, 0)->predicted_mask_any & 0x1, "slot 4096 in (l3=1, l2=0)");
    free_tree(t);
}

static int test_owned_all(void) {
    setvbuf(stdout, NULL, _IOFBF, 65536);
    printf("=== BUFFER-component lifecycle tests ===\n\n");

    int before = g_failed;
    RUN_TEST(test_owned_confirmed_push);
    RUN_TEST(test_owned_confirmed_clear);
    RUN_TEST(test_owned_confirmed_remove);
    RUN_TEST(test_owned_predict_push_empty_rollback);
    RUN_TEST(test_owned_predict_push_alive_rollback);
    RUN_TEST(test_owned_predict_push_push);
    RUN_TEST(test_owned_predict_push_remove);
    RUN_TEST(test_owned_predict_push_remove_push);
    RUN_TEST(test_owned_predict_remove_confirmed_rollback);
    RUN_TEST(test_owned_predict_clear_alive_rollback);
    RUN_TEST(test_owned_destroy_walk);

    int local_failed = g_failed - before;
    printf("\nowned: %d failed\n", local_failed);
    fflush(stdout);
    return local_failed ? 1 : 0;
}
