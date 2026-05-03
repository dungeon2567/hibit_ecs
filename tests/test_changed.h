#pragma once

#include "ecs.h"
#include "test_ecs.h"

/* `changed` semantics:
     - per-tick delta (cleared by ecs_tree_begin_tick / ecs_world_begin_tick)
     - set unconditionally (both CONFIRMED + PREDICT modes)
     - propagated bottom-up via ecs_iterator_next_block's write_mask path
     - cleared by rollback on dirty-walked nodes (predicted-only writes only)
     - reset by deserialize (no leak across snapshot loads) */

static void test_changed_set_marks_hierarchy(void) {
    ecs_tree_t* t = make_tree();
    set_val(t, ENT(2, 5, 7), 42);
    EXPECT((t->root->changed                    & (1ULL << 2)) != 0, "l3 changed bit 2");
    EXPECT((l2_of(t, 2)->changed                & (1ULL << 5)) != 0, "l2 changed bit 5");
    EXPECT((l1_of(t, 2, 5)->changed             & (1ULL << 7)) != 0, "l1 changed bit 7");
    ecs_tree_rollback(t);
    free_tree(t);
}

static void test_changed_remove_marks(void) {
    ecs_tree_t* t = make_tree();
    set_val(t, ENT(0, 0, 3), 5);
    ecs_tree_rollback(t);
    /* Rollback clears `changed` everywhere it walked, so post-rollback tree
       is a clean slate. */
    EXPECT(t->root->changed == 0, "rollback cleared root changed");

    /* Switch to PREDICT so the remove leaves the L1 alive (CONFIRMED-mode
       remove releases the L1 inline, after which children[] points at the
       default L1 with all masks zero). */
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    ecs_tree_remove(t, ENT(0, 0, 3));
    EXPECT((l1_of(t, 0, 0)->changed & (1ULL << 3)) != 0, "remove sets l1 changed");
    EXPECT((l2_of(t, 0)->changed    & 1ULL)         != 0, "remove sets l2 changed");
    EXPECT((t->root->changed        & 1ULL)         != 0, "remove sets l3 changed");
    ecs_tree_rollback(t);
    free_tree(t);
}

static void test_changed_begin_tick_clears(void) {
    ecs_tree_t* t = make_tree();
    set_val(t, 0,    1);
    set_val(t, 64,   2);                 /* different L1 */
    set_val(t, 4096, 3);                 /* different L2 */
    EXPECT(t->root->changed != 0, "changed populated");

    ecs_tree_begin_tick(t);
    EXPECT(t->root->changed == 0, "l3 changed zero");
    /* Each l2 / l1 visited via the changed walk must also be zeroed. */
    EXPECT(l2_of(t, 0)->changed       == 0, "l2[0] changed zero");
    EXPECT(l2_of(t, 1)->changed       == 0, "l2[1] changed zero");
    EXPECT(l1_of(t, 0, 0)->changed    == 0, "l1[0,0] changed zero");
    EXPECT(l1_of(t, 0, 1)->changed    == 0, "l1[0,1] changed zero");
    EXPECT(l1_of(t, 1, 0)->changed    == 0, "l1[1,0] changed zero");
    ecs_tree_rollback(t);
    free_tree(t);
}

static void test_changed_rollback_clears_on_dirty_walk(void) {
    ecs_tree_t* t = make_tree();
    set_val(t, 0, 1);
    ecs_tree_rollback(t);
    ecs_tree_begin_tick(t);

    /* PREDICT mode write — dirty + changed both set on this node. */
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    set_val(t, 0, 99);
    EXPECT(l1_of(t, 0, 0)->changed != 0, "predict write set changed");
    EXPECT(l1_of(t, 0, 0)->dirty   != 0, "predict write set dirty");

    ecs_tree_rollback(t);
    /* rollback walks `dirty | changed` — each visited node has changed cleared. */
    EXPECT(l1_of(t, 0, 0)->changed == 0, "rollback cleared l1 changed");
    EXPECT(t->root->changed        == 0, "rollback cleared l3 changed");
    free_tree(t);
}

static void test_changed_iter_set_propagates(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    w->trees[0].name = "Pos";
    w->mask = 1;

    /* Spread two entities across different L2 buckets to exercise the l2->l3
       propagation edge in next_slow. */
    set_val(&w->trees[0], ENT(0, 0, 0), 10);
    set_val(&w->trees[0], ENT(0, 2, 0), 20);
    ecs_world_rollback(w);
    ecs_world_begin_tick(w);
    EXPECT(w->trees[0].root->changed == 0, "begin_tick cleared root changed");

    ecs_compiled_query_t* q = ecs_compile_query(w, "Pos");
    ecs_iterator_t it = {0};
    ecs_iterator_init(&it, q, 1u);
    uint64_t mask;
    while ((mask = ecs_iterator_next_block(&it))) {
        while (mask) {
            int slot = ecs_ctz64(mask);
            mask &= mask - 1;
            *(comp_t*)ecs_iterator_get_mut(&it, 0, slot) = 0xBEEF;
        }
    }

    /* l1 changed set directly by iter_set; l2 / l3 set by write_mask propagation. */
    ecs_l2_t* l2 = l2_of(&w->trees[0], 0);
    EXPECT((l1_of(&w->trees[0], 0, 0)->changed & 1ULL) != 0, "l1[0,0] changed");
    EXPECT((l1_of(&w->trees[0], 0, 2)->changed & 1ULL) != 0, "l1[0,2] changed");
    EXPECT((l2->changed                       & 0x5)  == 0x5, "l2 changed bits 0+2 propagated");
    EXPECT((w->trees[0].root->changed         & 1ULL) != 0, "l3 changed bit 0 propagated");

    ecs_free(q);
    ecs_world_rollback(w);
    ecs_world_destroy(w); ecs_free(w);
}

static void test_changed_deserialize_zeros(void) {
    /* Pre-populate dst with `changed` bits, then deserialize a different
       state on top — `changed` must be zeroed (no leak). */
    ecs_tree_t* src = make_tree();
    set_val(src, 5, 0xA);
    ecs_tree_rollback(src);

    uint8_t* buf = (uint8_t*)ecs_xcalloc(1, 1 << 16);
    ecs_serializer_t s; ecs_serializer_init(&s, buf, 1 << 16);
    ecs_tree_serialize(src, &s);
    ecs_serializer_flush_bits(&s);
    int32_t bytes = ecs_serializer_get_bytes_written(&s);

    ecs_tree_t* dst = make_tree();
    set_val(dst, 5,    0xFF);             /* changed bits populated */
    set_val(dst, 4096, 0xEE);
    EXPECT(dst->root->changed != 0, "dst pre-populated with changed bits");

    ecs_deserializer_t d;
    ecs_deserializer_init(&d, buf, ((bytes + 7) / 8) * 8);
    int rc = ecs_tree_deserialize(dst, &d);
    EXPECT(rc == 0, "deserialize ok");
    EXPECT(dst->root->changed == 0, "deserialize zeroed l3 changed");
    /* Repurposed l2 must also have changed cleared. */
    EXPECT(l2_of(dst, 0)->changed    == 0, "deserialize zeroed l2 changed");
    EXPECT(l1_of(dst, 0, 0)->changed == 0, "deserialize zeroed l1 changed");

    free_tree(src); free_tree(dst); ecs_free(buf);
}

static int test_changed_all(void) {
    int before = g_failed;
    printf("=== changed-mask tests ===\n\n");
    RUN_TEST(test_changed_set_marks_hierarchy);
    RUN_TEST(test_changed_remove_marks);
    RUN_TEST(test_changed_begin_tick_clears);
    RUN_TEST(test_changed_rollback_clears_on_dirty_walk);
    RUN_TEST(test_changed_iter_set_propagates);
    RUN_TEST(test_changed_deserialize_zeros);
    int failed = g_failed - before;
    printf("\nchanged: %d failed\n", failed);
    return failed ? 1 : 0;
}
