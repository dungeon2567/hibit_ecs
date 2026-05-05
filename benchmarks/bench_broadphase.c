#include "bench_broadphase.h"
#include "broadphase.h"
#include "ecs_math.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* World cube spans [0, BENCH_BP_WORLD] on each axis (Q16.16). 256 fixed
   units = 256.0 in user space. Item half-extent 1 fixed unit, query
   half-extent 4 -> queries return ~a few hits each on average. */
#define BENCH_BP_WORLD 256

struct bench_broadphase_ctx {
    broadphase_t bp;

    /* Pre-built insert list, replayed verbatim each build iter. */
    broadphase_object_t* objs;
    aabb_t*              boxes;
    int                  n_items;

    /* Pre-built query list. */
    aabb_t* queries;
    int     n_queries;

    /* Accumulator so the optimizer cannot drop the query loop. */
    volatile uint32_t sink;
};

/* Tiny deterministic LCG -- enough for spatial scatter without dragging
   stdlib rand() (which would share state with other benches). */
static inline uint32_t bench_bp_lcg(uint32_t* s) {
    *s = (*s) * 1664525u + 1013904223u;
    return *s;
}

static aabb_t bench_bp_item_aabb(uint32_t* rng) {
    fixed_t cx = fixed_from_int((int)(bench_bp_lcg(rng) % BENCH_BP_WORLD));
    fixed_t cy = fixed_from_int((int)(bench_bp_lcg(rng) % BENCH_BP_WORLD));
    fixed_t cz = fixed_from_int((int)(bench_bp_lcg(rng) % BENCH_BP_WORLD));
    vec3_t  c  = vec3_make(cx, cy, cz);
    vec3_t  e  = vec3_make(fixed_from_int(1), fixed_from_int(1), fixed_from_int(1));
    return aabb_from_center_extents(c, e);
}

static aabb_t bench_bp_query_aabb(uint32_t* rng) {
    fixed_t cx = fixed_from_int((int)(bench_bp_lcg(rng) % BENCH_BP_WORLD));
    fixed_t cy = fixed_from_int((int)(bench_bp_lcg(rng) % BENCH_BP_WORLD));
    fixed_t cz = fixed_from_int((int)(bench_bp_lcg(rng) % BENCH_BP_WORLD));
    vec3_t  c  = vec3_make(cx, cy, cz);
    vec3_t  e  = vec3_make(fixed_from_int(4), fixed_from_int(4), fixed_from_int(4));
    return aabb_from_center_extents(c, e);
}

static void bench_bp_insert_all(bench_broadphase_ctx* ctx) {
    for (int i = 0; i < ctx->n_items; i++) {
        broadphase_insert(&ctx->bp, ctx->objs[i], ctx->boxes[i]);
    }
}

static uint32_t bench_bp_run_queries(bench_broadphase_ctx* ctx) {
    uint32_t s = 0;
    for (int i = 0; i < ctx->n_queries; i++) {
        broadphase_iter_t it;
        broadphase_query_begin(&it, &ctx->bp, ctx->queries[i]);
        const broadphase_object_t* obj;
        while ((obj = broadphase_query_next(&it))) {
            s += obj->entity_id;
        }
    }
    return s;
}

bench_broadphase_ctx* bench_broadphase_setup(int n_items, int n_queries) {
    bench_broadphase_ctx* ctx = (bench_broadphase_ctx*)calloc(1, sizeof(*ctx));
    ctx->n_items   = n_items;
    ctx->n_queries = n_queries;

    /* item_cap must be > 8 and a multiple of 8 -- round up. */
    size_t cap = (size_t)((n_items + 7) & ~7);
    if (cap < 16) cap = 16;
    broadphase_init(&ctx->bp, cap);

    ctx->objs    = (broadphase_object_t*)malloc((size_t)n_items * sizeof(broadphase_object_t));
    ctx->boxes   = (aabb_t*)             malloc((size_t)n_items * sizeof(aabb_t));
    ctx->queries = (aabb_t*)             malloc((size_t)n_queries * sizeof(aabb_t));

    uint32_t rng = 0xC0FFEEu;
    for (int i = 0; i < n_items; i++) {
        ctx->objs[i].transform = transform_identity();
        ctx->objs[i].entity_id = (uint32_t)i;
        ctx->boxes[i]          = bench_bp_item_aabb(&rng);
    }
    for (int i = 0; i < n_queries; i++) {
        ctx->queries[i] = bench_bp_query_aabb(&rng);
    }

    /* Pre-build once so iter_query has a tree from the first iteration. */
    bench_bp_insert_all(ctx);
    broadphase_build(&ctx->bp);

    /* Cache warmup. */
    for (int k = 0; k < 5; k++) {
        ctx->sink = bench_bp_run_queries(ctx);
    }

    return ctx;
}

/* Lattice placement: items spaced on an integer grid. side = cbrt(n_items)
   rounded up; spacing 4 fixed units; jitter < 1 unit -> no overlap. World
   cube grows to fit the lattice, queries reuse the same volume. */
bench_broadphase_ctx* bench_broadphase_setup_scatter(int n_items, int n_queries) {
    bench_broadphase_ctx* ctx = (bench_broadphase_ctx*)calloc(1, sizeof(*ctx));
    ctx->n_items   = n_items;
    ctx->n_queries = n_queries;

    size_t cap = (size_t)((n_items + 7) & ~7);
    if (cap < 16) cap = 16;
    broadphase_init(&ctx->bp, cap);

    ctx->objs    = (broadphase_object_t*)malloc((size_t)n_items * sizeof(broadphase_object_t));
    ctx->boxes   = (aabb_t*)             malloc((size_t)n_items * sizeof(aabb_t));
    ctx->queries = (aabb_t*)             malloc((size_t)n_queries * sizeof(aabb_t));

    int side = 1;
    while (side * side * side < n_items) side++;
    const int spacing = 4;
    const int world   = side * spacing;

    uint32_t rng = 0xC0FFEEu;
    for (int i = 0; i < n_items; i++) {
        int ix = i % side;
        int iy = (i / side) % side;
        int iz = i / (side * side);
        /* Jitter in (-0.5, +0.5) fixed units. */
        fixed_t jx = (fixed_t)((bench_bp_lcg(&rng) & 0xFFFF) - 0x8000);
        fixed_t jy = (fixed_t)((bench_bp_lcg(&rng) & 0xFFFF) - 0x8000);
        fixed_t jz = (fixed_t)((bench_bp_lcg(&rng) & 0xFFFF) - 0x8000);
        vec3_t  c  = vec3_make(fixed_from_int(ix * spacing) + jx,
                               fixed_from_int(iy * spacing) + jy,
                               fixed_from_int(iz * spacing) + jz);
        vec3_t  e  = vec3_make(fixed_from_int(1), fixed_from_int(1), fixed_from_int(1));
        ctx->objs[i].transform = transform_identity();
        ctx->objs[i].entity_id = (uint32_t)i;
        ctx->boxes[i]          = aabb_from_center_extents(c, e);
    }
    for (int i = 0; i < n_queries; i++) {
        fixed_t cx = fixed_from_int((int)(bench_bp_lcg(&rng) % (uint32_t)world));
        fixed_t cy = fixed_from_int((int)(bench_bp_lcg(&rng) % (uint32_t)world));
        fixed_t cz = fixed_from_int((int)(bench_bp_lcg(&rng) % (uint32_t)world));
        vec3_t  c  = vec3_make(cx, cy, cz);
        vec3_t  e  = vec3_make(fixed_from_int(4), fixed_from_int(4), fixed_from_int(4));
        ctx->queries[i] = aabb_from_center_extents(c, e);
    }

    bench_bp_insert_all(ctx);
    broadphase_build(&ctx->bp);

    for (int k = 0; k < 5; k++) {
        ctx->sink = bench_bp_run_queries(ctx);
    }
    return ctx;
}

void bench_broadphase_iter_build(bench_broadphase_ctx* ctx) {
    broadphase_clear(&ctx->bp);
    bench_bp_insert_all(ctx);
    broadphase_build(&ctx->bp);
}

void bench_broadphase_iter_query(bench_broadphase_ctx* ctx) {
    ctx->sink = bench_bp_run_queries(ctx);
}

void bench_broadphase_iter_full(bench_broadphase_ctx* ctx) {
    broadphase_clear(&ctx->bp);
    bench_bp_insert_all(ctx);
    broadphase_build(&ctx->bp);
    ctx->sink = bench_bp_run_queries(ctx);
}

void bench_broadphase_teardown(bench_broadphase_ctx* ctx) {
    broadphase_destroy(&ctx->bp);
    free(ctx->objs);
    free(ctx->boxes);
    free(ctx->queries);
    free(ctx);
}
