#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ecs.h"

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

static int g_passed = 0;
static int g_failed = 0;

#define EXPECT(cond, msg) do { \
    if (cond) { g_passed++; } \
    else { g_failed++; printf("FAIL [line %d]: %s\n", __LINE__, msg); } \
} while(0)

/* Run a void test_*(void) function and print "  <name>  OK" on success
   (no EXPECT failed during the call). The "test_" prefix is stripped for
   readability so each line matches the per-test status format used by
   the inline tests in main.c. */
#define RUN_TEST(fn) do { \
    int _f0 = g_failed; fn(); \
    if (g_failed == _f0) { \
        const char* _n = #fn; \
        if (strncmp(_n, "test_", 5) == 0) _n += 5; \
        printf("  %s  OK\n", _n); \
    } \
} while (0)

typedef uint32_t comp_t;

static inline ecs_tree_t* make_tree(void) {
    ecs_tree_t* t = (ecs_tree_t*)ecs_xcalloc(1, sizeof(ecs_tree_t));
    ecs_tree_init(t, sizeof(comp_t), 0);
    return t;
}

static inline void free_tree(ecs_tree_t* t) {
    ecs_tree_destroy(t);
    ecs_free(t);
}

static inline void set_val(ecs_tree_t* t, int idx, comp_t val) {
    *(comp_t*)ecs_tree_get_mut(t, idx) = val;
}

/* Reads current-frame value (predicted if dirty, else confirmed). */
static inline comp_t get_current_val(const ecs_tree_t* t, int idx) {
    return *(const comp_t*)ecs_tree_get(t, idx);
}

static ecs_l2_t* l2_of(ecs_tree_t* t, int l3_idx) {
    return t->root->children[l3_idx];
}

static ecs_l1_t* l1_of(ecs_tree_t* t, int l3_idx, int l2_idx) {
    return l2_of(t, l3_idx)->children[l2_idx];
}

#define ENT(l3, l2, l1) (((l3) << 12) | ((l2) << 6) | (l1))

static ecs_world_t* make_world(int n_trees) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    for (int i = 0; i < n_trees; i++) {
        ecs_tree_init(&w->trees[i], sizeof(comp_t), 0);
        w->mask |= (1ULL << i);
    }
    return w;
}

static void world_set_val(ecs_world_t* w, int tree_i, int idx, comp_t v) {
    *(comp_t*)ecs_tree_get_mut(&w->trees[tree_i], idx) = v;
}

static void world_remove(ecs_world_t* w, int tree_i, int idx) {
    ecs_tree_remove(&w->trees[tree_i], idx);
}

/* ---- predict-mode rollback tests ----
   Rollback discards in-flight predicted writes — predicted is always speculative,
   confirmed is unchanged. */

static void test_predict_rollback_data(void) {
    ecs_tree_t* t = make_tree();

    /* Seed confirmed = 42 via CONFIRMED write. */
    set_val(t, 0, 42);
    ecs_tree_rollback(t);
    EXPECT(ecs_tree_masks_valid(t), "masks valid post-rollback");
    EXPECT(get_current_val(t, 0) == 42, "current = 42 post-rollback");

    /* Predicted overwrite — visible mid-tick, gone after rollback. */
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    set_val(t, 0, 99);
    EXPECT(get_current_val(t, 0) == 99, "current sees predicted 99");

    ecs_tree_rollback(t);
    EXPECT(ecs_tree_masks_valid(t),       "masks valid post-rollback");
    EXPECT(get_current_val(t, 0) == 42, "current back to confirmed after rollback");

    free_tree(t);
}

static void test_predict_rollback_add(void) {
    ecs_tree_t* t = make_tree();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    set_val(t, 0, 5);
    /* Predicted-only add (no rollback yet). */
    EXPECT(l1_of(t, 0, 0)->predicted_mask_any == 1ULL, "predicted_mask has bit 0");
    EXPECT(l1_of(t, 0, 0)->confirmed_mask_any == 0,    "confirmed_mask still 0");
    EXPECT(l1_of(t, 0, 0)->dirty == 1ULL,              "slot dirty");

    ecs_tree_rollback(t);
    EXPECT(ecs_tree_masks_valid(t),     "masks valid post-rollback");
    EXPECT(t->root->confirmed_mask_any == 0,
           "rollback releases predicted-only L1 (confirmed empty)");
    EXPECT(t->root->predicted_mask_any == 0, "predicted mask cleared");

    free_tree(t);
}

static void test_predict_rollback_remove(void) {
    ecs_tree_t* t = make_tree();

    /* Seed confirmed via CONFIRMED-mode write. */
    set_val(t, 0, 77);
    ecs_tree_rollback(t);

    /* PREDICT-mode remove stages removal in predicted-mask, leaves confirmed alone. */
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    ecs_tree_remove(t, 0);
    EXPECT(l1_of(t, 0, 0)->predicted_mask_any == 0,    "predicted remove cleared bit");
    EXPECT(l1_of(t, 0, 0)->confirmed_mask_any == 1ULL, "confirmed still has entity");
    EXPECT(l1_of(t, 0, 0)->dirty              == 1ULL, "slot dirty");

    ecs_tree_rollback(t);
    EXPECT(ecs_tree_masks_valid(t),                    "masks valid post-rollback");
    EXPECT(l1_of(t, 0, 0)->predicted_mask_any == 1ULL, "remove undone — predicted has entity");
    EXPECT(get_current_val(t, 0) == 77,              "data unchanged");

    free_tree(t);
}

static void test_predict_promote_advances_confirmed(void) {
    ecs_tree_t* t = make_tree();

    set_val(t, 0, 42);
    EXPECT(ecs_tree_rollback(t) == 1, "rollback reports advance after first promote");
    EXPECT(get_current_val(t, 0) == 42, "confirmed = 42 after first promote");

    set_val(t, 0, 99);
    EXPECT(ecs_tree_rollback(t) == 1, "rollback reports advance after second promote");
    EXPECT(get_current_val(t, 0) == 99, "confirmed = 99 after second promote");

    free_tree(t);
}

static void test_predict_rollback_no_dirty_noop(void) {
    ecs_tree_t* t = make_tree();

    set_val(t, 0, 5);
    ecs_tree_rollback(t);
    uint64_t crc1 = ecs_tree_crc64(t);

    /* No predicted writes since promote - rollback is a no-op. */
    ecs_tree_rollback(t);
    EXPECT(ecs_tree_masks_valid(t), "masks valid post-noop-rollback");
    EXPECT(ecs_tree_crc64(t) == crc1, "crc unchanged after no-op rollback");

    free_tree(t);
}

static void test_predict_l2_l3_bitmaps(void) {
    ecs_tree_t* t = make_tree();

    set_val(t, ENT(0, 0, 0), 10);
    set_val(t, ENT(0, 1, 0), 20);
    set_val(t, ENT(1, 0, 0), 30);
    ecs_tree_rollback(t);

    EXPECT(t->root->confirmed_mask_any == 3ULL,         "L3 confirmed_mask = 3");
    EXPECT(l2_of(t, 0)->confirmed_mask_any == 3ULL,    "L2[0] confirmed_mask = 3");
    EXPECT(l2_of(t, 1)->confirmed_mask_any == 1ULL,    "L2[1] confirmed_mask = 1");

    free_tree(t);
}

/* ---- CRC64 tests ---- */

static void test_crc64_empty_deterministic(void) {
    ecs_tree_t* t = make_tree();
    uint64_t crc = ecs_tree_crc64(t);
    EXPECT(ecs_tree_crc64(t) == crc, "empty tree CRC is deterministic");
    free_tree(t);
}

static void test_crc64_changes_on_add(void) {
    ecs_tree_t* t = make_tree();
    uint64_t crc0 = ecs_tree_crc64(t);

    set_val(t, 0, 42);
    ecs_tree_rollback(t);
    EXPECT(ecs_tree_masks_valid(t), "masks valid after promote");
    uint64_t crc1 = ecs_tree_crc64(t);

    EXPECT(crc0 != crc1, "CRC changes after entity add+promote");
    free_tree(t);
}

static void test_crc64_changes_on_modify(void) {
    ecs_tree_t* t = make_tree();

    set_val(t, 0, 42); ecs_tree_rollback(t);
    uint64_t crc1 = ecs_tree_crc64(t);

    set_val(t, 0, 99); ecs_tree_rollback(t);
    uint64_t crc2 = ecs_tree_crc64(t);

    EXPECT(crc1 != crc2, "CRC changes after data modify+promote");
    free_tree(t);
}

static void test_crc64_position_sensitive(void) {
    ecs_tree_t* t1 = make_tree();
    ecs_tree_t* t2 = make_tree();

    set_val(t1, 0, 42); ecs_tree_rollback(t1);
    set_val(t2, 1, 42); ecs_tree_rollback(t2);

    EXPECT(ecs_tree_crc64(t1) != ecs_tree_crc64(t2),
           "CRC differs for same data at different entity slots");
    free_tree(t1);
    free_tree(t2);
}

/* CRC is view-aware: predicted writes shift it, rollback restores it. */
static void test_crc64_rollback_restores_predicted(void) {
    ecs_tree_t* t = make_tree();

    set_val(t, 0, 42);
    ecs_tree_rollback(t);
    uint64_t crc_confirmed = ecs_tree_crc64(t);

    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    set_val(t, 0, 99);
    EXPECT(ecs_tree_crc64(t) != crc_confirmed,
           "CRC tracks predicted writes (view-aware)");

    ecs_tree_rollback(t);
    EXPECT(ecs_tree_crc64(t) == crc_confirmed,
           "rollback discards predicted → CRC back to confirmed");
    EXPECT(get_current_val(t, 0) == 42, "value restored to 42");

    free_tree(t);
}

/* ---- world tests ---- */

static void test_world_crc64_empty_deterministic(void) {
    ecs_world_t* w = make_world(2);
    uint64_t crc = ecs_world_crc64(w);
    EXPECT(ecs_world_crc64(w) == crc, "empty world CRC is deterministic");
    ecs_world_destroy(w); ecs_free(w);
}

static void test_world_crc64_changes_on_promote(void) {
    ecs_world_t* w = make_world(2);
    uint64_t crc0 = ecs_world_crc64(w);

    world_set_val(w, 0, 5, 42);
    ecs_world_rollback(w);
    uint64_t crc1 = ecs_world_crc64(w);

    EXPECT(crc0 != crc1, "world CRC changes after stage+promote");
    ecs_world_destroy(w); ecs_free(w);
}

/* ---- ecs_compile_query tests ---- */

static ecs_world_t* make_named_world(const char** names, int n) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    for (int i = 0; i < n; i++) {
        ecs_tree_init(&w->trees[i], sizeof(comp_t), 0);
        w->trees[i].name = names[i];
        w->mask |= (1ULL << i);
    }
    return w;
}

static void test_compile_query_atom(void) {
    const char* names[] = {"Pos", "Vel"};
    ecs_world_t* w = make_named_world(names, 2);

    ecs_compiled_query_t* q = ecs_compile_query(w, "Pos");
    EXPECT(q != NULL, "atom query compiles");
    EXPECT(q->tree_count == 1, "atom: 1 tree");
    EXPECT(q->clause_count == 1, "atom: 1 clause");
    EXPECT(q->trees[0] == &w->trees[0], "atom: tree[0] = Pos");
    EXPECT(q->clauses[0].include == 1u, "atom: include bit 0");
    EXPECT(q->clauses[0].exclude == 0u, "atom: no exclude");
    ecs_free(q);

    ecs_world_destroy(w); ecs_free(w);
}

static void test_compile_query_and(void) {
    const char* names[] = {"Pos", "Vel"};
    ecs_world_t* w = make_named_world(names, 2);

    ecs_compiled_query_t* q = ecs_compile_query(w, "Pos & Vel");
    EXPECT(q != NULL, "and query compiles");
    EXPECT(q->tree_count == 2, "and: 2 trees");
    EXPECT(q->clause_count == 1, "and: 1 clause");
    EXPECT(q->clauses[0].include == 0x3u, "and: include bits 0|1");
    EXPECT(q->clauses[0].exclude == 0u, "and: no exclude");
    ecs_free(q);

    ecs_world_destroy(w); ecs_free(w);
}

static void test_compile_query_or(void) {
    const char* names[] = {"Pos", "Vel"};
    ecs_world_t* w = make_named_world(names, 2);

    ecs_compiled_query_t* q = ecs_compile_query(w, "Pos | Vel");
    EXPECT(q != NULL, "or query compiles");
    EXPECT(q->clause_count == 2, "or: 2 clauses");
    uint32_t a = q->clauses[0].include, b = q->clauses[1].include;
    EXPECT((a | b) == 0x3u && (a & b) == 0u, "or: clauses cover {Pos},{Vel}");
    ecs_free(q);

    ecs_world_destroy(w); ecs_free(w);
}

static void test_compile_query_not(void) {
    const char* names[] = {"Pos", "Vel"};
    ecs_world_t* w = make_named_world(names, 2);

    ecs_compiled_query_t* q = ecs_compile_query(w, "Pos & !Vel");
    EXPECT(q != NULL, "not query compiles");
    EXPECT(q->clause_count == 1, "not: 1 clause");
    EXPECT(q->clauses[0].include == 1u, "not: include = Pos");
    EXPECT(q->clauses[0].exclude == 2u, "not: exclude = Vel");
    ecs_free(q);

    ecs_world_destroy(w); ecs_free(w);
}

static void test_compile_query_unknown_name(void) {
    const char* names[] = {"Pos"};
    ecs_world_t* w = make_named_world(names, 1);

    ecs_compiled_query_t* q = ecs_compile_query(w, "Ghost");
    EXPECT(q == NULL, "unknown name returns NULL");

    ecs_world_destroy(w); ecs_free(w);
}

static void test_compile_query_bad_syntax(void) {
    const char* names[] = {"Pos"};
    ecs_world_t* w = make_named_world(names, 1);

    EXPECT(ecs_compile_query(w, "Pos &") == NULL, "trailing & returns NULL");
    EXPECT(ecs_compile_query(w, "(Pos")   == NULL, "unclosed paren returns NULL");
    EXPECT(ecs_compile_query(w, "")       == NULL, "empty expr returns NULL");

    ecs_world_destroy(w); ecs_free(w);
}

static void test_compile_query_null_args(void) {
    const char* names[] = {"Pos"};
    ecs_world_t* w = make_named_world(names, 1);

    EXPECT(ecs_compile_query(NULL, "Pos") == NULL, "NULL world returns NULL");
    EXPECT(ecs_compile_query(w, NULL)     == NULL, "NULL expr returns NULL");

    ecs_world_destroy(w); ecs_free(w);
}

static void test_world_predict_rollback(void) {
    ecs_world_t* w = make_world(3);

    /* Seed confirmed via CONFIRMED writes. */
    world_set_val(w, 0, ENT(0, 0, 1), 10);
    world_set_val(w, 2, ENT(1, 0, 0), 20);
    ecs_world_rollback(w);                              /* tick 1 */
    uint64_t crc_confirmed = ecs_world_crc64(w);

    /* Switch to PREDICT for speculative ops. */
    ecs_world_set_mode(w, ECS_MODE_PREDICT);
    world_set_val(w, 1, ENT(0, 0, 2), 30);
    world_set_val(w, 0, ENT(0, 0, 1), 11);
    world_remove(w, 2, ENT(1, 0, 0));
    EXPECT(ecs_world_crc64(w) != crc_confirmed,
           "world CRC tracks predicted writes (view-aware)");

    /* Rollback discards predicted state. */
    ecs_world_rollback(w);
    for (int k = 0; k < 3; k++)
        EXPECT(ecs_tree_masks_valid(&w->trees[k]),
               "tree masks valid post-rollback");
    EXPECT(ecs_world_crc64(w) == crc_confirmed,
           "world CRC matches confirmed after rollback");
    EXPECT(get_current_val(&w->trees[0], ENT(0, 0, 1)) == 10, "tree 0 value restored");
    EXPECT(get_current_val(&w->trees[2], ENT(1, 0, 0)) == 20, "tree 2 entity restored");

    ecs_world_destroy(w); ecs_free(w);
}

static int test_all(void) {
#ifdef _MSC_VER
    _CrtSetDbgFlag(0);
#endif
    setvbuf(stdout, NULL, _IOFBF, 65536);
    ecs_crc64_init();
    printf("=== ECS predict-mode + CRC64 tests ===\n\n");

    RUN_TEST(test_predict_rollback_data);
    RUN_TEST(test_predict_rollback_add);
    RUN_TEST(test_predict_rollback_remove);
    RUN_TEST(test_predict_promote_advances_confirmed);
    RUN_TEST(test_predict_rollback_no_dirty_noop);
    RUN_TEST(test_predict_l2_l3_bitmaps);
    RUN_TEST(test_crc64_empty_deterministic);
    RUN_TEST(test_crc64_changes_on_add);
    RUN_TEST(test_crc64_changes_on_modify);
    RUN_TEST(test_crc64_position_sensitive);
    RUN_TEST(test_crc64_rollback_restores_predicted);
    RUN_TEST(test_world_crc64_empty_deterministic);
    RUN_TEST(test_world_crc64_changes_on_promote);
    RUN_TEST(test_compile_query_atom);
    RUN_TEST(test_compile_query_and);
    RUN_TEST(test_compile_query_or);
    RUN_TEST(test_compile_query_not);
    RUN_TEST(test_compile_query_unknown_name);
    RUN_TEST(test_compile_query_bad_syntax);
    RUN_TEST(test_compile_query_null_args);
    RUN_TEST(test_world_predict_rollback);

    printf("\n%d passed, %d failed\n", g_passed, g_failed);
    fflush(stdout);
    return g_failed ? 1 : 0;
}
