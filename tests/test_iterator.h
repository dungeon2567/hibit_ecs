#pragma once

#include "ecs.h"
#include "test_ecs.h"   /* g_passed/g_failed/EXPECT/comp_t */

/* Iterator coverage: init, next, get, set, write_mask propagation.

   Uses compile_query so the world pointer is wired into the compiled query
   (required by the changed-clause invariant when present). All trees default
   to PREDICT mode; iterator visits predicted_mask_any. Promoted entities have
   predicted == confirmed, so they're visible immediately. */

static ecs_world_t* iter_make_world(void) {
    const char* names[] = { "Pos", "Vel" };
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    ecs_tree_init(&w->trees[1], sizeof(comp_t), 0);
    w->trees[0].name = names[0];
    w->trees[1].name = names[1];
    w->mask = 0x3;
    return w;
}

static void test_iter_atom_count(void) {
    ecs_world_t* w = iter_make_world();
    set_val(&w->trees[0], ENT(0, 0, 0), 10);
    set_val(&w->trees[0], ENT(0, 0, 5), 20);
    set_val(&w->trees[0], ENT(0, 1, 3), 30);
    set_val(&w->trees[0], ENT(2, 0, 0), 40);
    ecs_world_rollback(w);

    ecs_compiled_query_t* q = ecs_compile_query(w, "Pos");
    ecs_iterator_t it = {0};
    ecs_iterator_init(&it, q);
    int count = 0;
    while (ecs_iterator_next(&it) >= 0) count++;
    EXPECT(count == 4, "iterator visits all 4 Pos entities");

    ecs_free(q);
    ecs_world_destroy(w); ecs_free(w);
}

static void test_iter_and_intersection(void) {
    ecs_world_t* w = iter_make_world();
    /* Pos at 3 slots, Vel at 2 slots — intersection = 1. */
    set_val(&w->trees[0], 0, 1);
    set_val(&w->trees[0], 1, 2);
    set_val(&w->trees[0], 2, 3);
    set_val(&w->trees[1], 1, 100);     /* overlap */
    set_val(&w->trees[1], 7, 200);
    ecs_world_rollback(w);

    ecs_compiled_query_t* q = ecs_compile_query(w, "Pos & Vel");
    ecs_iterator_t it = {0};
    ecs_iterator_init(&it, q);
    int count = 0, observed_l1 = -1, slot;
    while ((slot = ecs_iterator_next(&it)) >= 0) {
        observed_l1 = slot;
        count++;
    }
    EXPECT(count == 1, "Pos & Vel intersection = 1");
    EXPECT(observed_l1 == 1, "intersection at slot 1");

    ecs_free(q);
    ecs_world_destroy(w); ecs_free(w);
}

static void test_iter_exclude(void) {
    ecs_world_t* w = iter_make_world();
    set_val(&w->trees[0], 0, 1);
    set_val(&w->trees[0], 1, 2);
    set_val(&w->trees[0], 2, 3);
    set_val(&w->trees[1], 1, 100);     /* mask out slot 1 */
    ecs_world_rollback(w);

    ecs_compiled_query_t* q = ecs_compile_query(w, "Pos & !Vel");
    ecs_iterator_t it = {0};
    ecs_iterator_init(&it, q);
    int count = 0;
    while (ecs_iterator_next(&it) >= 0) count++;
    EXPECT(count == 2, "Pos & !Vel = 2");

    ecs_free(q);
    ecs_world_destroy(w); ecs_free(w);
}

static void test_iter_get_reads_value(void) {
    ecs_world_t* w = iter_make_world();
    set_val(&w->trees[0], 0, 0xAA);
    set_val(&w->trees[0], 5, 0xBB);
    ecs_world_rollback(w);

    ecs_compiled_query_t* q = ecs_compile_query(w, "Pos");
    ecs_iterator_t it = {0};
    ecs_iterator_init(&it, q);
    uint32_t observed = 0;
    int slot;
    while ((slot = ecs_iterator_next(&it)) >= 0) {
        const comp_t* v = (const comp_t*)ecs_iterator_get(&it, 0, slot);
        observed |= *v;
    }
    EXPECT(observed == (0xAA | 0xBB), "iterator_get sees both values");

    ecs_free(q);
    ecs_world_destroy(w); ecs_free(w);
}

static void test_iter_set_modifies_value(void) {
    ecs_world_t* w = iter_make_world();
    set_val(&w->trees[0], 3, 1);
    ecs_world_rollback(w);

    /* Switch to PREDICT to exercise the predicted-slot + dirty path. */
    ecs_world_set_mode(w, ECS_MODE_PREDICT);

    ecs_compiled_query_t* q = ecs_compile_query(w, "Pos");
    ecs_iterator_t it = {0};
    ecs_iterator_init(&it, q);
    it.write_mask = 1u;                /* tree 0 receives writes */
    int slot;
    while ((slot = ecs_iterator_next(&it)) >= 0) {
        *(comp_t*)ecs_iterator_get_mut(&it, 0, slot) = 999;
    }

    /* Predict-mode write: dirty bit set, predicted slot holds 999, get reads
       predicted via the dirty_hi shift. */
    EXPECT(get_current_val(&w->trees[0], 3) == 999, "iter_set updated value visible via tree_get");
    EXPECT((l1_of(&w->trees[0], 0, 0)->dirty & (1ULL << 3)) != 0, "iter_set set dirty bit");

    ecs_free(q);
    ecs_world_rollback(w);
    ecs_world_destroy(w); ecs_free(w);
}

static void test_iter_write_mask_propagates(void) {
    ecs_world_t* w = iter_make_world();
    /* Two entities in different L2 buckets within same L3 — exercises the
       l1->l2 propagation edge in next_slow's advance_l2 block. */
    set_val(&w->trees[0], ENT(0, 0, 0), 10);
    set_val(&w->trees[0], ENT(0, 1, 0), 20);
    ecs_world_rollback(w);

    /* Pre-condition: post-rollback, all dirty == 0. */
    EXPECT(w->trees[0].root->dirty == 0, "root dirty 0 post-rollback");

    /* PREDICT mode so iter_set raises dirty and propagation can be observed. */
    ecs_world_set_mode(w, ECS_MODE_PREDICT);

    ecs_compiled_query_t* q = ecs_compile_query(w, "Pos");
    ecs_iterator_t it = {0};
    ecs_iterator_init(&it, q);
    it.write_mask = 1u;
    int slot;
    while ((slot = ecs_iterator_next(&it)) >= 0) {
        *(comp_t*)ecs_iterator_get_mut(&it, 0, slot) = 0xC0FFEE;
    }

    /* After exhausting iteration, l2->dirty must reflect both l1 buckets that
       received writes; l3->dirty must reflect the l2. */
    ecs_l2_t* l2 = l2_of(&w->trees[0], 0);
    EXPECT((l2->dirty & 0x3) == 0x3, "l2 dirty propagated for both l1 buckets");
    EXPECT((w->trees[0].root->dirty & 0x1) != 0, "l3 dirty propagated for l2[0]");

    ecs_free(q);
    ecs_world_rollback(w);
    ecs_world_destroy(w); ecs_free(w);
}

static int test_iterator_all(void) {
    int before = g_failed;
    printf("=== iterator tests ===\n\n");
    RUN_TEST(test_iter_atom_count);
    RUN_TEST(test_iter_and_intersection);
    RUN_TEST(test_iter_exclude);
    RUN_TEST(test_iter_get_reads_value);
    RUN_TEST(test_iter_set_modifies_value);
    RUN_TEST(test_iter_write_mask_propagates);
    int failed = g_failed - before;
    printf("\niterator: %d failed\n", failed);
    return failed ? 1 : 0;
}
