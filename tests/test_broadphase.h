#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "broadphase.h"

/* Reuses g_passed / g_failed / EXPECT / RUN_TEST from test_ecs.h.
   test_ecs.h must be included before this file in the TU.

   The broadphase is now a wide-8 LBVH built once per tick from the inserted
   item buffer. Tests follow the cycle:
       broadphase_clear (or fresh init)
       broadphase_insert ... broadphase_insert
       broadphase_build
       broadphase_query_begin / broadphase_query_next */

static aabb_t bp_aabb_xyz_(int x, int y, int z, int half) {
    vec3_t c = vec3_make(fixed_from_int(x), fixed_from_int(y), fixed_from_int(z));
    vec3_t e = vec3_make(fixed_from_int(half), fixed_from_int(half), fixed_from_int(half));
    return aabb_from_center_extents(c, e);
}

static int bp_u32_cmp_(const void* a, const void* b) {
    uint32_t ax = *(const uint32_t*)a, bx = *(const uint32_t*)b;
    return (ax > bx) - (ax < bx);
}

/* ============================================================
   tests
   ============================================================ */

static void test_broadphase_init_destroy(void) {
    broadphase_t bp;
    broadphase_init(&bp, 1024);

    EXPECT(bp.item_cap == 1024,  "item_cap matches request");
    EXPECT(bp.n_items  == 0,     "no items on init");
    EXPECT(bp.n_nodes  == 0,     "no nodes on init");
    EXPECT(bp.has_tree == 0,     "tree not built on init");
    EXPECT(bp.item_ids   != NULL, "item_ids allocated");
    EXPECT(bp.item_aabbs != NULL, "item_aabbs allocated");
    EXPECT(bp.morton     != NULL, "morton allocated");
    EXPECT(bp.perm       != NULL, "perm allocated");
    EXPECT(bp.nodes      != NULL, "node arena allocated");

    broadphase_destroy(&bp);
    EXPECT(bp.item_ids == NULL, "item_ids NULL after destroy");
    EXPECT(bp.nodes    == NULL, "nodes NULL after destroy");
}

static void test_broadphase_empty_build(void) {
    /* Build with zero items -> has_tree stays 0; queries report exhausted. */
    broadphase_t bp; broadphase_init(&bp, 16);
    broadphase_build(&bp);
    EXPECT(bp.has_tree == 0, "empty build leaves has_tree=0");

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(0, 0, 0, 4));
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 0, "no hits on empty tree");

    broadphase_destroy(&bp);
}

static void test_broadphase_insert_query_single(void) {
    broadphase_t bp; broadphase_init(&bp, 16);

    broadphase_insert(&bp, 42, bp_aabb_xyz_(0, 0, 0, 1));
    broadphase_build(&bp);
    EXPECT(bp.has_tree == 1, "tree built");
    EXPECT(bp.n_nodes  == 1, "single item -> single leaf node");
    EXPECT(bp.root     == 0, "root index 0");

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(0, 0, 0, 2));
    uint32_t id = 0;
    EXPECT(broadphase_query_next(&it, &id) == 1, "single hit");
    EXPECT(id == 42,                              "id round-trips");
    EXPECT(broadphase_query_next(&it, &id) == 0, "iterator drained");

    broadphase_destroy(&bp);
}

static void test_broadphase_query_aabb_miss(void) {
    /* Disjoint AABBs: SIMD overlap mask must reject. */
    broadphase_t bp; broadphase_init(&bp, 16);
    broadphase_insert(&bp, 1, bp_aabb_xyz_(0, 0, 0, 1));   /* [-1,1] */
    broadphase_build(&bp);

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(3, 0, 0, 1)); /* [2,4] */
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 0, "AABB-disjoint rejected");

    broadphase_destroy(&bp);
}

static void test_broadphase_leaf_overflow(void) {
    /* 20 items all centred on origin force >1 leaf (8 + 8 + 4) plus an
       internal level. Verifies multi-leaf yield + internal-node descent. */
    broadphase_t bp; broadphase_init(&bp, 64);

    enum { N = 20 };
    for (uint32_t i = 0; i < N; ++i)
        broadphase_insert(&bp, 100u + i, bp_aabb_xyz_(0, 0, 0, 1));
    broadphase_build(&bp);

    /* leaves = ceil(20/8) = 3, plus 1 root internal -> 4 nodes total. */
    EXPECT(bp.n_nodes == 4, "3 leaves + 1 root packed");

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(0, 0, 0, 2));

    uint32_t found[N + 4] = {0};
    int n = 0;
    uint32_t id;
    while (broadphase_query_next(&it, &id)) {
        if (n < (int)(sizeof(found) / sizeof(found[0]))) found[n] = id;
        ++n;
    }
    EXPECT(n == N, "all 20 items returned");

    qsort(found, (size_t)n, sizeof(uint32_t), bp_u32_cmp_);
    int ordered = (n == N);
    for (int i = 0; ordered && i < n; ++i)
        if (found[i] != 100u + (uint32_t)i) ordered = 0;
    EXPECT(ordered, "ids 100..119 each returned exactly once, no dupes");

    broadphase_destroy(&bp);
}

static void test_broadphase_multi_level(void) {
    /* >64 items force two internal levels: leaves -> L1 -> root. */
    broadphase_t bp; broadphase_init(&bp, 256);

    enum { N = 100 };
    for (uint32_t i = 0; i < N; ++i) {
        int x = (int)(i % 10);
        int y = (int)((i / 10) % 10);
        broadphase_insert(&bp, 1u + i, bp_aabb_xyz_(x, y, 0, 1));
    }
    broadphase_build(&bp);
    EXPECT(bp.n_nodes >= 13 + 2, "leaves(13) + L1(2) + root or merged");

    aabb_t q = {
        .min = vec3_make(fixed_from_int(-2), fixed_from_int(-2), fixed_from_int(-2)),
        .max = vec3_make(fixed_from_int(20), fixed_from_int(20), fixed_from_int(2))
    };
    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, q);

    int seen[N + 1] = {0};
    int hits = 0;
    uint32_t id;
    while (broadphase_query_next(&it, &id)) {
        EXPECT(id >= 1 && id <= N, "id within inserted range");
        if (id >= 1 && id <= N) seen[id]++;
        ++hits;
    }
    EXPECT(hits == N, "every item visible to whole-grid query");
    int dupes = 0;
    for (int i = 1; i <= (int)N; ++i) if (seen[i] != 1) dupes = 1;
    EXPECT(dupes == 0, "no item returned twice or skipped");

    broadphase_destroy(&bp);
}

static void test_broadphase_aabb_filter(void) {
    /* SIMD lane filter: three items pass the bounds check; only the one
       whose AABB hits q must be reported. */
    broadphase_t bp; broadphase_init(&bp, 16);

    fixed_t f05 = fixed_from_parts(0, 5);
    vec3_t  e   = vec3_make(f05, f05, f05);

    broadphase_insert(&bp, 10, aabb_from_center_extents(vec3_make(0, 0, 0), e));
    broadphase_insert(&bp, 11, aabb_from_center_extents(vec3_make(fixed_from_int(3), 0, 0), e));
    broadphase_insert(&bp, 12, aabb_from_center_extents(vec3_make(0, fixed_from_int(3), 0), e));
    broadphase_build(&bp);

    aabb_t q = aabb_from_center_extents(vec3_make(fixed_from_int(3), 0, 0), e);
    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, q);

    int count = 0;
    uint32_t id;
    while (broadphase_query_next(&it, &id)) {
        EXPECT(id == 11, "only id 11 inside the tight query AABB");
        ++count;
    }
    EXPECT(count == 1, "exactly one match");

    broadphase_destroy(&bp);
}

static void test_broadphase_clear_reuse(void) {
    /* clear must drop the previous tick's items + tree so a fresh insert/
       build cycle reports only new ids. */
    broadphase_t bp; broadphase_init(&bp, 64);

    for (uint32_t i = 0; i < 30; ++i)
        broadphase_insert(&bp, 1000u + i, bp_aabb_xyz_(0, 0, 0, 1));
    broadphase_build(&bp);
    EXPECT(bp.has_tree == 1, "tree built");
    EXPECT(bp.n_items  == 30, "items recorded");

    broadphase_clear(&bp);
    EXPECT(bp.n_items  == 0, "items zeroed by clear");
    EXPECT(bp.n_nodes  == 0, "nodes zeroed by clear");
    EXPECT(bp.has_tree == 0, "tree invalidated by clear");

    broadphase_build(&bp);
    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(0, 0, 0, 2));
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 0, "no items visible after clear/build");

    broadphase_insert(&bp, 99u, bp_aabb_xyz_(0, 0, 0, 1));
    broadphase_build(&bp);
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(0, 0, 0, 2));
    int got = broadphase_query_next(&it, &id);
    EXPECT(got == 1 && id == 99u, "fresh insert reported");
    EXPECT(broadphase_query_next(&it, &id) == 0, "no stale ids leak");

    broadphase_destroy(&bp);
}

static void test_broadphase_query_far_away(void) {
    /* Query AABB far from any item. Whole tree pruned at the root. */
    broadphase_t bp; broadphase_init(&bp, 16);

    broadphase_insert(&bp, 1, bp_aabb_xyz_(2, 2, 2, 1));
    broadphase_build(&bp);

    aabb_t q = {
        .min = vec3_make(fixed_from_int(100), fixed_from_int(100), fixed_from_int(100)),
        .max = vec3_make(fixed_from_int(200), fixed_from_int(200), fixed_from_int(200))
    };
    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, q);
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 0, "far query -> no hits");

    broadphase_destroy(&bp);
}

static void test_broadphase_full_grid(void) {
    /* 27 items on a 3^3 lattice. Query covering the whole world must yield
       every id exactly once. */
    broadphase_t bp; broadphase_init(&bp, 64);

    uint32_t id_counter = 0;
    for (int z = 0; z < 3; ++z)
    for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x) {
        int cx = 4 * x + 2;
        int cy = 4 * y + 2;
        int cz = 4 * z + 2;
        broadphase_insert(&bp, id_counter++, bp_aabb_xyz_(cx, cy, cz, 1));
    }
    broadphase_build(&bp);
    EXPECT(id_counter == 27, "27 items inserted");

    aabb_t q = {
        .min = vec3_make(0, 0, 0),
        .max = vec3_make(fixed_from_int(12), fixed_from_int(12), fixed_from_int(12))
    };
    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, q);

    uint32_t seen[32] = {0};
    int n = 0;
    uint32_t id;
    while (broadphase_query_next(&it, &id)) {
        if (n < 32) seen[n] = id;
        ++n;
    }
    EXPECT(n == 27, "every id hit exactly once");

    qsort(seen, (size_t)n, sizeof(uint32_t), bp_u32_cmp_);
    int complete = (n == 27);
    for (int i = 0; complete && i < n; ++i)
        if (seen[i] != (uint32_t)i) complete = 0;
    EXPECT(complete, "ids 0..26 all returned, no dupes");

    broadphase_destroy(&bp);
}

static void test_broadphase_negative_coords(void) {
    /* Negative coords work end-to-end: world AABB derived from items, so
       the morton quantization origin is automatically item-relative. */
    broadphase_t bp; broadphase_init(&bp, 16);

    broadphase_insert(&bp, 5, bp_aabb_xyz_(-10, -10, -10, 1));
    broadphase_build(&bp);

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(-10, -10, -10, 2));
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 1, "negative-coord item reachable");
    EXPECT(id == 5, "id matches");
    EXPECT(broadphase_query_next(&it, &id) == 0, "drained");

    broadphase_destroy(&bp);
}

static void test_broadphase_query_before_build(void) {
    /* Insert without build -> has_tree=0 -> queries find nothing. Catches
       missed broadphase_build calls instead of returning stale data. */
    broadphase_t bp; broadphase_init(&bp, 16);
    broadphase_insert(&bp, 7, bp_aabb_xyz_(0, 0, 0, 1));

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(0, 0, 0, 4));
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 0, "query without build -> empty");

    broadphase_destroy(&bp);
}

/* ============================================================
   runner
   ============================================================ */

static int test_broadphase_all(void) {
    int before = g_failed;
    printf("=== broadphase tests ===\n\n");
    RUN_TEST(test_broadphase_init_destroy);
    RUN_TEST(test_broadphase_empty_build);
    RUN_TEST(test_broadphase_insert_query_single);
    RUN_TEST(test_broadphase_query_aabb_miss);
    RUN_TEST(test_broadphase_leaf_overflow);
    RUN_TEST(test_broadphase_multi_level);
    RUN_TEST(test_broadphase_aabb_filter);
    RUN_TEST(test_broadphase_clear_reuse);
    RUN_TEST(test_broadphase_query_far_away);
    RUN_TEST(test_broadphase_full_grid);
    RUN_TEST(test_broadphase_negative_coords);
    RUN_TEST(test_broadphase_query_before_build);
    int failed = g_failed - before;
    printf("\nbroadphase: %d failed\n", failed);
    return failed ? 1 : 0;
}
