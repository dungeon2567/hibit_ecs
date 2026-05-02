#pragma once

#include "ecs.h"
#include "test_ecs.h"

/* End-of-pipeline destruction pass: a designated tag tree marks entities
   for removal. After all systems run, ecs_pipeline_run batch-clears the
   tagged bits from every other tree, then wipes the tag tree itself. */

static void test_destruction_basic_clear(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);   /* Pos */
    ecs_tree_init(&w->trees[1], sizeof(comp_t), 0);   /* Vel */
    ecs_tree_init(&w->trees[7], 0, 0);                /* Destroyed (tag) */
    w->mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 7);
    ecs_world_set_destroyed_tag(w, 7);

    /* Entities {3, 5, 100} have Pos+Vel; mark 3 and 100 for destruction. */
    set_val(&w->trees[0], 3,   10);
    set_val(&w->trees[0], 5,   20);
    set_val(&w->trees[0], 100, 30);
    set_val(&w->trees[1], 3,   1);
    set_val(&w->trees[1], 5,   2);
    set_val(&w->trees[1], 100, 3);
    ecs_tree_set(&w->trees[7], 3,   NULL);
    ecs_tree_set(&w->trees[7], 100, NULL);

    ecs_pipeline_t p; ecs_pipeline_init(&p);
    ecs_pipeline_run(&p, w);
    ecs_pipeline_destroy(&p);

    /* Slot 100 = L3=0, L2=1, L1=36 — single occupant of L2[0][1] L1.
       Tagged entities cleared from data trees. */
    EXPECT(w->trees[0].root->confirmed_mask_any == 1ULL,
           "Pos: only L3[0] populated (slot 5 survives)");
    EXPECT((w->trees[0].root->children[0]->children[0]->confirmed_mask_any
            & ((1ULL << 3) | (1ULL << 5))) == (1ULL << 5),
           "Pos: slot 3 cleared, slot 5 kept");
    EXPECT(w->trees[0].root->children[0]->children[1] == &ecs_default_l1,
           "Pos: empty L1 at L2[0][1] released");
    EXPECT((w->trees[1].root->children[0]->children[0]->confirmed_mask_any
            & ((1ULL << 3) | (1ULL << 5))) == (1ULL << 5),
           "Vel: slot 3 cleared, slot 5 kept");
    EXPECT(w->trees[1].root->children[0]->children[1] == &ecs_default_l1,
           "Vel: empty L1 at L2[0][1] released");

    /* Destroyed tag itself wiped. */
    EXPECT(w->trees[7].root->confirmed_mask_any == 0,
           "Destroyed tag wiped post-pipeline");
    EXPECT(w->trees[7].root->predicted_mask_any == 0,
           "Destroyed tag predicted_mask wiped");
    EXPECT(w->trees[7].root->children[0] == &ecs_default_l2,
           "Destroyed tag L2 nodes returned to pool");

    /* Survivors readable. */
    EXPECT(get_current_val(&w->trees[0], 5) == 20, "Pos[5] survives");
    EXPECT(get_current_val(&w->trees[1], 5) == 2,  "Vel[5] survives");

    EXPECT(ecs_tree_masks_valid(&w->trees[0]), "Pos masks valid");
    EXPECT(ecs_tree_masks_valid(&w->trees[1]), "Vel masks valid");
    EXPECT(ecs_tree_masks_valid(&w->trees[7]), "Destroyed masks valid");

    ecs_world_destroy(w); ecs_free(w);
}

/* Tag entities not present in some trees — pass must skip absent slots
   without breaking masks_valid invariant. */
static void test_destruction_partial_membership(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    ecs_tree_init(&w->trees[1], sizeof(comp_t), 0);
    ecs_tree_init(&w->trees[2], 0, 0);
    w->mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 2);
    ecs_world_set_destroyed_tag(w, 2);

    set_val(&w->trees[0], 5, 100);    /* Only in tree 0 */
    set_val(&w->trees[1], 9, 200);    /* Only in tree 1 */
    ecs_tree_set(&w->trees[2], 5, NULL);
    ecs_tree_set(&w->trees[2], 9, NULL);
    ecs_tree_set(&w->trees[2], 99, NULL);  /* Tagged but absent from data trees */

    ecs_pipeline_t p; ecs_pipeline_init(&p);
    ecs_pipeline_run(&p, w);
    ecs_pipeline_destroy(&p);

    EXPECT(w->trees[0].root->confirmed_mask_any == 0, "tree 0 emptied");
    EXPECT(w->trees[1].root->confirmed_mask_any == 0, "tree 1 emptied");
    EXPECT(w->trees[2].root->confirmed_mask_any == 0, "tag wiped");
    EXPECT(ecs_tree_masks_valid(&w->trees[0]), "tree 0 masks valid");
    EXPECT(ecs_tree_masks_valid(&w->trees[1]), "tree 1 masks valid");

    ecs_world_destroy(w); ecs_free(w);
}

/* No tag registered → pipeline runs nothing destructive. */
static void test_destruction_no_tag_registered(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    w->mask = 1ULL;
    set_val(&w->trees[0], 3, 42);

    ecs_pipeline_t p; ecs_pipeline_init(&p);
    ecs_pipeline_run(&p, w);
    ecs_pipeline_destroy(&p);

    EXPECT(get_current_val(&w->trees[0], 3) == 42, "untouched");
    ecs_world_destroy(w); ecs_free(w);
}

/* Empty destroyed tag (no entities marked) → other trees untouched, tag
   stays empty. */
static void test_destruction_empty_tag(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    ecs_tree_init(&w->trees[3], 0, 0);
    w->mask = (1ULL << 0) | (1ULL << 3);
    ecs_world_set_destroyed_tag(w, 3);
    set_val(&w->trees[0], 7, 99);

    ecs_pipeline_t p; ecs_pipeline_init(&p);
    ecs_pipeline_run(&p, w);
    ecs_pipeline_destroy(&p);

    EXPECT(get_current_val(&w->trees[0], 7) == 99, "data tree untouched");
    EXPECT(w->trees[3].root->confirmed_mask_any == 0, "tag still empty");

    ecs_world_destroy(w); ecs_free(w);
}

/* Many entities spread across multiple L1/L2 — exercises deep mask walks
   and node release on both levels. */
static void test_destruction_multi_l2(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    ecs_tree_init(&w->trees[1], 0, 0);
    w->mask = (1ULL << 0) | (1ULL << 1);
    ecs_world_set_destroyed_tag(w, 1);

    /* Entities at varied L3/L2/L1 paths. */
    set_val(&w->trees[0], 0,      1);    /* L3=0 L2=0 L1=0 */
    set_val(&w->trees[0], 64,     2);    /* L3=0 L2=1 L1=0 */
    set_val(&w->trees[0], 4096,   3);    /* L3=1 L2=0 L1=0 */
    set_val(&w->trees[0], 100000, 4);    /* L3=24 L2=26 L1=32 */
    /* Mark all but slot 64 for destruction. */
    ecs_tree_set(&w->trees[1], 0,      NULL);
    ecs_tree_set(&w->trees[1], 4096,   NULL);
    ecs_tree_set(&w->trees[1], 100000, NULL);

    ecs_pipeline_t p; ecs_pipeline_init(&p);
    ecs_pipeline_run(&p, w);
    ecs_pipeline_destroy(&p);

    EXPECT(get_current_val(&w->trees[0], 64) == 2, "slot 64 survives");
    EXPECT((w->trees[0].root->confirmed_mask_any & 1ULL) != 0,
           "L3[0] still populated (via slot 64)");
    EXPECT(w->trees[0].root->children[1]  == &ecs_default_l2, "L3[1] released");
    EXPECT(w->trees[0].root->children[24] == &ecs_default_l2, "L3[24] released");
    EXPECT(w->trees[1].root->confirmed_mask_any == 0, "tag wiped");
    EXPECT(ecs_tree_masks_valid(&w->trees[0]), "data masks valid");
    EXPECT(ecs_tree_masks_valid(&w->trees[1]), "tag masks valid");

    ecs_world_destroy(w); ecs_free(w);
}

/* Destruction mid-pipeline: a system that tags entities runs, then end
   of pipeline cleans them up. */
static int g_tag_system_calls;
static void tag_destroy_3_and_5(ecs_world_t* w, void* ctx) {
    (void)ctx;
    g_tag_system_calls++;
    ecs_tree_set(&w->trees[1], 3, NULL);
    ecs_tree_set(&w->trees[1], 5, NULL);
}

static void test_destruction_via_system(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    ecs_tree_init(&w->trees[1], 0, 0);
    w->mask = (1ULL << 0) | (1ULL << 1);
    ecs_world_set_destroyed_tag(w, 1);

    set_val(&w->trees[0], 3, 11);
    set_val(&w->trees[0], 5, 22);
    set_val(&w->trees[0], 7, 33);

    g_tag_system_calls = 0;
    ecs_pipeline_t p; ecs_pipeline_init(&p);
    ecs_pipeline_add(&p, tag_destroy_3_and_5, NULL);
    ecs_pipeline_run(&p, w);
    ecs_pipeline_destroy(&p);

    EXPECT(g_tag_system_calls == 1, "tag system ran once");
    EXPECT(get_current_val(&w->trees[0], 7) == 33, "untagged survives");
    EXPECT(w->trees[1].root->confirmed_mask_any == 0, "tag wiped after pipeline");
    EXPECT((w->trees[0].root->children[0]->children[0]->confirmed_mask_any
            & ((1ULL << 3) | (1ULL << 5) | (1ULL << 7))) == (1ULL << 7),
           "only slot 7 remains");

    ecs_world_destroy(w); ecs_free(w);
}

/* Disable destruction pass via tree_idx = -1. */
static void test_destruction_disable(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    ecs_tree_init(&w->trees[1], 0, 0);
    w->mask = (1ULL << 0) | (1ULL << 1);
    ecs_world_set_destroyed_tag(w, 1);
    ecs_world_set_destroyed_tag(w, -1);

    set_val(&w->trees[0], 3, 99);
    ecs_tree_set(&w->trees[1], 3, NULL);

    ecs_pipeline_t p; ecs_pipeline_init(&p);
    ecs_pipeline_run(&p, w);
    ecs_pipeline_destroy(&p);

    EXPECT(get_current_val(&w->trees[0], 3) == 99, "destruction disabled — slot kept");
    EXPECT((w->trees[1].root->children[0]->children[0]->confirmed_mask_any
            & (1ULL << 3)) != 0, "tag itself untouched");

    ecs_world_destroy(w); ecs_free(w);
}

static int test_destruction_all(void) {
    int before = g_failed;
    printf("=== destruction tests ===\n\n");
    test_destruction_basic_clear();
    test_destruction_partial_membership();
    test_destruction_no_tag_registered();
    test_destruction_empty_tag();
    test_destruction_multi_l2();
    test_destruction_via_system();
    test_destruction_disable();
    int failed = g_failed - before;
    printf("\ndestruction: %d failed\n", failed);
    return failed ? 1 : 0;
}
