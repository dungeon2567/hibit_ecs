#pragma once

#include "ecs.h"
#include "test_ecs.h"

/* Pipeline: ordered list of systems run once per tick. Single-threaded. */

typedef struct {
    int  call_count;
    int  call_order;       /* set on each call */
    int* shared_counter;   /* global ordering counter */
    int  id;
} pipe_sys_state_t;

static void pipe_record(ecs_world_t* w, void* user) {
    (void)w;
    pipe_sys_state_t* st = (pipe_sys_state_t*)user;
    st->call_count++;
    st->call_order = (*st->shared_counter)++;
}

static void test_pipeline_empty_runs_begin_tick(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    w->mask = 1;
    set_val(&w->trees[0], 0, 1);

    ecs_pipeline_t p; ecs_pipeline_init(&p);
    uint64_t tid_before = w->tick_id;
    ecs_pipeline_run(&p, w);
    EXPECT(w->tick_id == tid_before + 1, "empty pipeline still bumps tick_id");
    EXPECT(w->trees[0].tick_id_at_begin == w->tick_id, "tree tick_id snapshot synced");

    ecs_pipeline_destroy(&p);
    ecs_world_rollback(w);
    ecs_world_destroy(w); ecs_free(w);
}

static void test_pipeline_runs_single_system_once(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    w->mask = 1;

    int counter = 0;
    pipe_sys_state_t s = { 0, -1, &counter, 0 };

    ecs_pipeline_t p; ecs_pipeline_init(&p);
    ecs_pipeline_add(&p, pipe_record, &s);
    ecs_pipeline_run(&p, w);

    EXPECT(s.call_count == 1, "single system called once per pipeline_run");

    ecs_pipeline_run(&p, w);
    EXPECT(s.call_count == 2, "called again on second run");

    ecs_pipeline_destroy(&p);
    ecs_world_destroy(w); ecs_free(w);
}

static void test_pipeline_preserves_registration_order(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    w->mask = 1;

    int counter = 0;
    pipe_sys_state_t a = { 0, -1, &counter, 0 };
    pipe_sys_state_t b = { 0, -1, &counter, 1 };
    pipe_sys_state_t c = { 0, -1, &counter, 2 };

    ecs_pipeline_t p; ecs_pipeline_init(&p);
    ecs_pipeline_add(&p, pipe_record, &a);
    ecs_pipeline_add(&p, pipe_record, &b);
    ecs_pipeline_add(&p, pipe_record, &c);

    EXPECT(ecs_pipeline_count(&p) == 3, "pipeline_count == 3");

    ecs_pipeline_run(&p, w);
    EXPECT(a.call_order == 0, "a ran first");
    EXPECT(b.call_order == 1, "b ran second");
    EXPECT(c.call_order == 2, "c ran third");

    ecs_pipeline_destroy(&p);
    ecs_world_destroy(w); ecs_free(w);
}

static void test_pipeline_tick_id_synced_across_trees(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);
    ecs_tree_init(&w->trees[3], sizeof(comp_t), 0);
    ecs_tree_init(&w->trees[7], sizeof(comp_t), 0);
    w->mask = (1ULL << 0) | (1ULL << 3) | (1ULL << 7);

    ecs_pipeline_t p; ecs_pipeline_init(&p);
    ecs_pipeline_run(&p, w);

    EXPECT(w->trees[0].tick_id_at_begin == w->tick_id, "tree[0] synced");
    EXPECT(w->trees[3].tick_id_at_begin == w->tick_id, "tree[3] synced");
    EXPECT(w->trees[7].tick_id_at_begin == w->tick_id, "tree[7] synced");

    ecs_pipeline_destroy(&p);
    ecs_world_destroy(w); ecs_free(w);
}

static int test_pipeline_all(void) {
    int before = g_failed;
    printf("=== pipeline tests ===\n\n");
    RUN_TEST(test_pipeline_empty_runs_begin_tick);
    RUN_TEST(test_pipeline_runs_single_system_once);
    RUN_TEST(test_pipeline_preserves_registration_order);
    RUN_TEST(test_pipeline_tick_id_synced_across_trees);
    int failed = g_failed - before;
    printf("\npipeline: %d failed\n", failed);
    return failed ? 1 : 0;
}
