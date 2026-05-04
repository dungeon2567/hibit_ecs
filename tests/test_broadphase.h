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

static inline broadphase_object_t bp_obj_(uint32_t id) {
    broadphase_object_t o;
    o.transform = transform_identity();
    o.id        = id;
    return o;
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
    EXPECT(bp.item_min_x != NULL, "item_min_x allocated");
    EXPECT(bp.item_max_z != NULL, "item_max_z allocated");
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

    broadphase_insert(&bp, bp_obj_(42), bp_aabb_xyz_(0, 0, 0, 1));
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
    broadphase_insert(&bp, bp_obj_(1), bp_aabb_xyz_(0, 0, 0, 1));   /* [-1,1] */
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
        broadphase_insert(&bp, bp_obj_(100u + i), bp_aabb_xyz_(0, 0, 0, 1));
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
        broadphase_insert(&bp, bp_obj_(1u + i), bp_aabb_xyz_(x, y, 0, 1));
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

    broadphase_insert(&bp, bp_obj_(10), aabb_from_center_extents(vec3_make(0, 0, 0), e));
    broadphase_insert(&bp, bp_obj_(11), aabb_from_center_extents(vec3_make(fixed_from_int(3), 0, 0), e));
    broadphase_insert(&bp, bp_obj_(12), aabb_from_center_extents(vec3_make(0, fixed_from_int(3), 0), e));
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
        broadphase_insert(&bp, bp_obj_(1000u + i), bp_aabb_xyz_(0, 0, 0, 1));
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

    broadphase_insert(&bp, bp_obj_(99u), bp_aabb_xyz_(0, 0, 0, 1));
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

    broadphase_insert(&bp, bp_obj_(1), bp_aabb_xyz_(2, 2, 2, 1));
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
        broadphase_insert(&bp, bp_obj_(id_counter++), bp_aabb_xyz_(cx, cy, cz, 1));
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

    broadphase_insert(&bp, bp_obj_(5), bp_aabb_xyz_(-10, -10, -10, 1));
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
    broadphase_insert(&bp, bp_obj_(7), bp_aabb_xyz_(0, 0, 0, 1));

    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, bp_aabb_xyz_(0, 0, 0, 4));
    uint32_t id;
    EXPECT(broadphase_query_next(&it, &id) == 0, "query without build -> empty");

    broadphase_destroy(&bp);
}

/* Determinism golden: 16 collinear items at x=0..15 (y=z=0). Build is fully
   deterministic across platforms because:
     - fixed_t is Q16.16 integer math (no FP drift),
     - the radix sort is stable, and
     - SIMD min/max reductions are commutative.
   Tree shape: 2 leaves + 1 root. perm = identity (x ascending == morton
   ascending), so leaf 0 packs items 0..7, leaf 1 packs 8..15. Root packs
   leaves into lanes 0,1. Query br() pushes hit children low-to-high then
   pops LIFO, so leaf 1 yields first; within a leaf, lanes 0..7 yield in
   order via the ctz scan. Sequence is locked. */
static void test_broadphase_known_order(void) {
    broadphase_t bp; broadphase_init(&bp, 32);

    enum { N = 16 };
    for (uint32_t i = 0; i < N; ++i)
        broadphase_insert(&bp, bp_obj_(100u + i), bp_aabb_xyz_((int)i, 0, 0, 1));
    broadphase_build(&bp);

    EXPECT(bp.n_nodes == 3, "deterministic shape: 2 leaves + 1 root");
    EXPECT(bp.root    == 2, "deterministic root index");

    /* Query covers x=[-93,107] -- hits every item. */
    aabb_t q = bp_aabb_xyz_(7, 0, 0, 100);
    broadphase_iter_t it;
    broadphase_query_begin(&it, &bp, q);

    static const uint32_t expected[N] = {
        108u, 109u, 110u, 111u, 112u, 113u, 114u, 115u,
        100u, 101u, 102u, 103u, 104u, 105u, 106u, 107u,
    };
    uint32_t got[N] = {0};
    int n = 0;
    uint32_t id;
    while (broadphase_query_next(&it, &id)) {
        if (n < N) got[n] = id;
        ++n;
    }
    EXPECT(n == N, "all 16 items yielded");
    int match = (n == N);
    for (int i = 0; match && i < N; ++i)
        if (got[i] != expected[i]) match = 0;
    EXPECT(match, "yield order matches platform-independent golden");

    broadphase_destroy(&bp);
}

/* Cross-instance determinism: two bp instances with identical insert
   sequence produce byte-identical query streams. Catches accidental
   dependence on prior heap state, ASLR-influenced pointer ordering, or
   uninitialized scratch. */
static void test_broadphase_determinism_two_instances(void) {
    enum { N = 50 };
    broadphase_t a, b;
    broadphase_init(&a, 64);
    broadphase_init(&b, 64);

    for (uint32_t i = 0; i < N; ++i) {
        int x = (int)(i % 7) - 3;
        int y = (int)((i / 7) % 7) - 3;
        int z = (int)((i / 3) % 5) - 2;
        aabb_t aabb = bp_aabb_xyz_(x, y, z, 1);
        broadphase_insert(&a, bp_obj_(1000u + i), aabb);
        broadphase_insert(&b, bp_obj_(1000u + i), aabb);
    }
    broadphase_build(&a);
    broadphase_build(&b);

    EXPECT(a.n_nodes == b.n_nodes, "deterministic n_nodes");
    EXPECT(a.root    == b.root,    "deterministic root");

    aabb_t q = bp_aabb_xyz_(0, 0, 0, 10);
    broadphase_iter_t ia, ib;
    broadphase_query_begin(&ia, &a, q);
    broadphase_query_begin(&ib, &b, q);

    int order_match = 1, count = 0;
    for (;;) {
        uint32_t id_a = 0, id_b = 0;
        int ra = broadphase_query_next(&ia, &id_a);
        int rb = broadphase_query_next(&ib, &id_b);
        if (ra != rb) { order_match = 0; break; }
        if (!ra) break;
        if (id_a != id_b) { order_match = 0; break; }
        ++count;
    }
    EXPECT(order_match, "identical yield order across two instances");
    EXPECT(count > 0,   "query non-empty");

    broadphase_destroy(&a);
    broadphase_destroy(&b);
}

/* Rebuild determinism: clear + re-insert + rebuild on the same bp must
   produce the same query stream as the first build. Exercises arena reuse
   (stale node lanes from prior build must not leak through). */
static void test_broadphase_determinism_rebuild(void) {
    enum { N = 30 };
    broadphase_t bp; broadphase_init(&bp, 64);

    aabb_t boxes[N];
    for (uint32_t i = 0; i < N; ++i) {
        int x = (int)(i % 5);
        int y = (int)((i / 5) % 5);
        boxes[i] = bp_aabb_xyz_(x, y, 0, 1);
        broadphase_insert(&bp, bp_obj_(7u + i), boxes[i]);
    }
    broadphase_build(&bp);

    aabb_t q = bp_aabb_xyz_(2, 2, 0, 5);
    uint32_t first[N + 8] = {0};
    int n1 = 0;
    {
        broadphase_iter_t it;
        broadphase_query_begin(&it, &bp, q);
        uint32_t id;
        while (broadphase_query_next(&it, &id)) {
            if (n1 < (int)(sizeof(first) / sizeof(first[0]))) first[n1] = id;
            ++n1;
        }
    }

    broadphase_clear(&bp);
    for (uint32_t i = 0; i < N; ++i)
        broadphase_insert(&bp, bp_obj_(7u + i), boxes[i]);
    broadphase_build(&bp);

    uint32_t second[N + 8] = {0};
    int n2 = 0;
    {
        broadphase_iter_t it;
        broadphase_query_begin(&it, &bp, q);
        uint32_t id;
        while (broadphase_query_next(&it, &id)) {
            if (n2 < (int)(sizeof(second) / sizeof(second[0]))) second[n2] = id;
            ++n2;
        }
    }

    EXPECT(n1 == n2, "rebuild yields same hit count");
    int same_order = (n1 == n2);
    for (int i = 0; same_order && i < n1; ++i)
        if (first[i] != second[i]) same_order = 0;
    EXPECT(same_order, "rebuild yields identical hit order");

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
    RUN_TEST(test_broadphase_known_order);
    RUN_TEST(test_broadphase_determinism_two_instances);
    RUN_TEST(test_broadphase_determinism_rebuild);
    int failed = g_failed - before;
    printf("\nbroadphase: %d failed\n", failed);
    return failed ? 1 : 0;
}
