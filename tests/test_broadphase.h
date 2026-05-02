#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "broadphase.h"

/* Reuses g_passed / g_failed / EXPECT / RUN_TEST from test_ecs.h.
   test_ecs.h must be included before this file in the TU.

   World layout used by most tests: origin (-32,-32,-32), size 64^3 metres.
   That gives 16 cells per axis, cell (8,8,8) covers world [0,4) on each
   axis, so anything centred in [0,4) collapses into the same cell.        */

static aabb_t bp_aabb_xyz_(int x, int y, int z, int half) {
    vec3_t c = vec3_make(fixed_from_int(x), fixed_from_int(y), fixed_from_int(z));
    vec3_t e = vec3_make(fixed_from_int(half), fixed_from_int(half), fixed_from_int(half));
    return aabb_from_center_extents(c, e);
}

static void bp_setup_world_(broadphase_t* bp) {
    vec3_t origin = vec3_make(fixed_from_int(-32), fixed_from_int(-32), fixed_from_int(-32));
    vec3_t size   = vec3_make(fixed_from_int(64),  fixed_from_int(64),  fixed_from_int(64));
    broadphase_init(bp, origin, size, 64u * 1024u);
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
    vec3_t origin = vec3_make(0, 0, 0);
    vec3_t size   = vec3_make(fixed_from_int(16), fixed_from_int(8), fixed_from_int(4));
    broadphase_init(&bp, origin, size, 32u * 1024u);

    EXPECT(bp.cells_x == 4, "cells_x = ceil(16/4)");
    EXPECT(bp.cells_y == 2, "cells_y = ceil(8/4)");
    EXPECT(bp.cells_z == 1, "cells_z = ceil(4/4)");
    EXPECT(bp.heads != NULL,        "heads allocated");
    EXPECT(bp.arena.base != NULL,   "arena allocated");
    EXPECT(bp.arena.cap == 32u * 1024u, "arena cap matches request");
    EXPECT(bp.arena.off == 0,       "arena empty on init");

    broadphase_destroy(&bp);
    EXPECT(bp.heads == NULL,        "heads NULL after destroy");
    EXPECT(bp.arena.base == NULL,   "arena base NULL after destroy");
}

static void test_broadphase_size_clamp(void) {
    /* Zero-size axis must clamp to >= 1 cell so the head index math is sound. */
    broadphase_t bp;
    vec3_t origin = vec3_make(0, 0, 0);
    vec3_t size   = vec3_make(0, fixed_from_int(4), 0);
    broadphase_init(&bp, origin, size, 4u * 1024u);
    EXPECT(bp.cells_x == 1, "cells_x clamped to 1");
    EXPECT(bp.cells_y == 1, "cells_y exactly 1 (4m / 4m)");
    EXPECT(bp.cells_z == 1, "cells_z clamped to 1");
    broadphase_destroy(&bp);
}

static void test_broadphase_insert_query_single(void) {
    broadphase_t bp; bp_setup_world_(&bp);

    broadphase_insert(&bp, 42, bp_aabb_xyz_(0, 0, 0, 1));

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(0, 0, 0, 2));
    uint32_t id = 0;
    EXPECT(broadphase_query_next(&it, &id) == 1, "single hit");
    EXPECT(id == 42,                              "id round-trips");
    EXPECT(broadphase_query_next(&it, &id) == 0, "iterator drained");

    broadphase_destroy(&bp);
}

static void test_broadphase_query_aabb_miss(void) {
    /* Centre lands in the same cell the query covers, but AABBs are disjoint
       so the SIMD overlap mask must reject it. */
    broadphase_t bp; bp_setup_world_(&bp);
    broadphase_insert(&bp, 1, bp_aabb_xyz_(0, 0, 0, 1));   /* [-1,1] */

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(3, 0, 0, 1)); /* [2,4] */
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 0, "AABB-disjoint same-cell rejected");

    broadphase_destroy(&bp);
}

static void test_broadphase_centre_outside_dropped(void) {
    broadphase_t bp;
    vec3_t origin = vec3_make(0, 0, 0);
    vec3_t size   = vec3_make(fixed_from_int(8), fixed_from_int(8), fixed_from_int(8));
    broadphase_init(&bp, origin, size, 16u * 1024u);

    /* Centre (-2,-2,-2) sits one cell below the grid -- silent drop. */
    broadphase_insert(&bp, 7, bp_aabb_xyz_(-2, -2, -2, 1));

    broadphase_iter_t it;
    /* Query covers entire grid plus the dropped item's AABB. */
    aabb_t q = {
        .min = vec3_make(fixed_from_int(-4), fixed_from_int(-4), fixed_from_int(-4)),
        .max = vec3_make(fixed_from_int(8),  fixed_from_int(8),  fixed_from_int(8))
    };
    broadphase_query_begin(&it, &bp, q);
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 0, "out-of-grid centre silently dropped");

    broadphase_destroy(&bp);
}

static void test_broadphase_node_overflow_chain(void) {
    /* 20 items at the same centre force three chained nodes (8 + 8 + 4).
       Verifies linked-list walk and per-slot ctz are both correct. */
    broadphase_t bp; bp_setup_world_(&bp);

    enum { N = 20 };
    for (uint32_t i = 0; i < N; ++i)
        broadphase_insert(&bp, 100u + i, bp_aabb_xyz_(0, 0, 0, 1));

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(0, 0, 0, 2));

    uint32_t found[N + 4] = {0};
    int n = 0;
    uint32_t id;
    while (broadphase_query_next(&it, &id)) {
        if (n < (int)(sizeof(found) / sizeof(found[0]))) found[n] = id;
        ++n;
    }
    EXPECT(n == N, "all 20 items returned across chained nodes");

    qsort(found, (size_t)n, sizeof(uint32_t), bp_u32_cmp_);
    int ordered = (n == N);
    for (int i = 0; ordered && i < n; ++i)
        if (found[i] != 100u + (uint32_t)i) ordered = 0;
    EXPECT(ordered, "ids 100..119 each returned exactly once, no dupes");

    broadphase_destroy(&bp);
}

static void test_broadphase_multi_cell(void) {
    /* Items spread across three cells along x. Query AABB spans the first
       two, so iterator must visit two cells and reject the third. */
    broadphase_t bp; bp_setup_world_(&bp);

    broadphase_insert(&bp, 1, bp_aabb_xyz_(0,  0, 0, 1));     /* cell x=8 */
    broadphase_insert(&bp, 2, bp_aabb_xyz_(4,  0, 0, 1));     /* cell x=9 */
    broadphase_insert(&bp, 3, bp_aabb_xyz_(8,  0, 0, 1));     /* cell x=10 */
    broadphase_insert(&bp, 4, bp_aabb_xyz_(0,  4, 0, 1));     /* cell y=9 */

    aabb_t q = {
        .min = vec3_make(fixed_from_int(-1), fixed_from_int(-1), fixed_from_int(-1)),
        .max = vec3_make(fixed_from_int(5),  fixed_from_int(1),  fixed_from_int(1))
    };
    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, q);

    uint32_t found[8] = {0};
    int n = 0;
    uint32_t id;
    while (broadphase_query_next(&it, &id)) {
        if (n < 8) found[n] = id;
        ++n;
    }
    qsort(found, (size_t)n, sizeof(uint32_t), bp_u32_cmp_);
    EXPECT(n == 2,                            "two cells worth of hits");
    EXPECT(n >= 2 && found[0] == 1,           "id 1 returned");
    EXPECT(n >= 2 && found[1] == 2,           "id 2 returned, 3 and 4 filtered");

    broadphase_destroy(&bp);
}

static void test_broadphase_aabb_filter_same_cell(void) {
    /* Three items live in the same cell (index 8 on each axis = world [0,4));
       the SIMD per-node compare must keep only the one whose AABB hits q. */
    broadphase_t bp; bp_setup_world_(&bp);

    fixed_t f05 = fixed_from_parts(0, 5);
    vec3_t  e   = vec3_make(f05, f05, f05);

    broadphase_insert(&bp, 10, aabb_from_center_extents(vec3_make(0, 0, 0), e));
    broadphase_insert(&bp, 11, aabb_from_center_extents(vec3_make(fixed_from_int(3), 0, 0), e));
    broadphase_insert(&bp, 12, aabb_from_center_extents(vec3_make(0, fixed_from_int(3), 0), e));

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
    /* clear must reset both the head array (so no stale links remain) and
       the arena offset (cheap reuse). The next round of inserts must report
       only the new ids -- a missed presence_mask reset would leak the old. */
    broadphase_t bp; bp_setup_world_(&bp);

    for (uint32_t i = 0; i < 30; ++i)
        broadphase_insert(&bp, 1000u + i, bp_aabb_xyz_(0, 0, 0, 1));

    EXPECT(bp.arena.off > 0, "arena bump advanced after inserts");

    broadphase_clear(&bp);
    EXPECT(bp.arena.off == 0, "clear zeroes arena offset");

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(0, 0, 0, 2));
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 0, "no items visible after clear");

    broadphase_insert(&bp, 99u, bp_aabb_xyz_(0, 0, 0, 1));
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(0, 0, 0, 2));
    int got = broadphase_query_next(&it, &id);
    EXPECT(got == 1 && id == 99u, "fresh insert reported");
    EXPECT(broadphase_query_next(&it, &id) == 0, "no stale ids leak from arena reuse");

    broadphase_destroy(&bp);
}

static void test_broadphase_query_outside_grid(void) {
    /* Query whose entire cell range is past the grid clamps to an empty
       sentinel range; the iterator must immediately report exhausted. */
    broadphase_t bp;
    vec3_t origin = vec3_make(0, 0, 0);
    vec3_t size   = vec3_make(fixed_from_int(8), fixed_from_int(8), fixed_from_int(8));
    broadphase_init(&bp, origin, size, 16u * 1024u);

    broadphase_insert(&bp, 1, bp_aabb_xyz_(2, 2, 2, 1));

    aabb_t q_above = {
        .min = vec3_make(fixed_from_int(100), fixed_from_int(100), fixed_from_int(100)),
        .max = vec3_make(fixed_from_int(200), fixed_from_int(200), fixed_from_int(200))
    };
    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, q_above);
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 0, "query above grid -> empty");

    aabb_t q_below = {
        .min = vec3_make(fixed_from_int(-200), fixed_from_int(-200), fixed_from_int(-200)),
        .max = vec3_make(fixed_from_int(-100), fixed_from_int(-100), fixed_from_int(-100))
    };
    broadphase_query_begin(&it, &bp, q_below);
    EXPECT(broadphase_query_next(&it, &id) == 0, "query below grid -> empty");

    broadphase_destroy(&bp);
}

static void test_broadphase_full_grid_query(void) {
    /* Sanity: insert one item per cell on a tiny grid, query the entire
       world, and check we get every id back. Walks all three loop axes. */
    broadphase_t bp;
    vec3_t origin = vec3_make(0, 0, 0);
    vec3_t size   = vec3_make(fixed_from_int(12), fixed_from_int(12), fixed_from_int(12)); /* 3x3x3 cells */
    broadphase_init(&bp, origin, size, 32u * 1024u);

    uint32_t id_counter = 0;
    for (int z = 0; z < 3; ++z)
    for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x) {
        int cx = 4 * x + 2;  /* cell centre in metres */
        int cy = 4 * y + 2;
        int cz = 4 * z + 2;
        broadphase_insert(&bp, id_counter++, bp_aabb_xyz_(cx, cy, cz, 1));
    }
    EXPECT(id_counter == 27, "27 items inserted (3^3 cells)");

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
    EXPECT(n == 27, "every cell hit exactly once");

    qsort(seen, (size_t)n, sizeof(uint32_t), bp_u32_cmp_);
    int complete = (n == 27);
    for (int i = 0; complete && i < n; ++i)
        if (seen[i] != (uint32_t)i) complete = 0;
    EXPECT(complete, "ids 0..26 all returned, no dupes");

    broadphase_destroy(&bp);
}

static void test_broadphase_negative_coords(void) {
    /* Centre in negative-coord cell. Verifies the cell-axis floor on
       two's-complement (uint32 subtract + arithmetic shift) correctly
       maps a negative offset into a non-negative cell index. */
    broadphase_t bp; bp_setup_world_(&bp);  /* origin -32 */

    broadphase_insert(&bp, 5, bp_aabb_xyz_(-10, -10, -10, 1));

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(-10, -10, -10, 2));
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 1, "negative-coord item reachable");
    EXPECT(id == 5, "id matches");
    EXPECT(broadphase_query_next(&it, &id) == 0, "drained");

    broadphase_destroy(&bp);
}

/* ============================================================
   runner
   ============================================================ */

static int test_broadphase_all(void) {
    int before = g_failed;
    printf("=== broadphase tests ===\n\n");
    RUN_TEST(test_broadphase_init_destroy);
    RUN_TEST(test_broadphase_size_clamp);
    RUN_TEST(test_broadphase_insert_query_single);
    RUN_TEST(test_broadphase_query_aabb_miss);
    RUN_TEST(test_broadphase_centre_outside_dropped);
    RUN_TEST(test_broadphase_node_overflow_chain);
    RUN_TEST(test_broadphase_multi_cell);
    RUN_TEST(test_broadphase_aabb_filter_same_cell);
    RUN_TEST(test_broadphase_clear_reuse);
    RUN_TEST(test_broadphase_query_outside_grid);
    RUN_TEST(test_broadphase_full_grid_query);
    RUN_TEST(test_broadphase_negative_coords);
    int failed = g_failed - before;
    printf("\nbroadphase: %d failed\n", failed);
    return failed ? 1 : 0;
}
