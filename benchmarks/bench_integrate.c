#include "bench_integrate.h"
#include "ecs.h"
#include "ecs_math.h"
#include <stdlib.h>
#include <string.h>

/* 1/60 in Q16.16 ≈ 1092 */
#define BENCH_DT ((fixed_t)1092)

struct bench_integrate_ctx {
    ecs_world_t*         w;
    ecs_compiled_query_t query;
    int                  n_match;
};

static void bench_integrate_fn(ecs_iterator_t* it) {
    uint64_t mask;
    while ((mask = ecs_iterator_next_block(it))) {
        do {
            int slot = ecs_ctz64(mask);
            mask &= mask - 1;
            const fixed_4_t* vel = (const fixed_4_t* __restrict)ecs_iterator_get(it, 1, slot);
            const fixed_4_t* pos_cur = (const fixed_4_t* __restrict)ecs_iterator_get(it, 0, slot);
            fixed_4_t new_pos = fixed4_add(*pos_cur, fixed4_scale(*vel, BENCH_DT));

            if (fixed4_eq(new_pos, *pos_cur) != 0xF) {
                *(fixed_4_t*)ecs_iterator_get_mut(it, 0, slot) = new_pos;
            }
        } while (mask);
    }
}

bench_integrate_ctx* bench_integrate_setup(int n_total, int n_match) {
    bench_integrate_ctx* ctx = (bench_integrate_ctx*)calloc(1, sizeof(*ctx));
    ctx->n_match = n_match;
    ctx->w       = (ecs_world_t*)calloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&ctx->w->trees[0], sizeof(fixed_4_t), 0);
    ecs_tree_init(&ctx->w->trees[1], sizeof(fixed_4_t), 0);
    ctx->w->mask = 3;

    for (int i = 0; i < n_total; i++) {
        *(fixed_4_t*)ecs_tree_get_mut(&ctx->w->trees[0], i) =
            fixed4_vec3(fixed_from_int(i), 0, 0);
    }

    fixed_4_t vel = fixed4_vec3(fixed_from_int(60), fixed_from_int(-30), 0);

    if (n_match <= n_total / 2) {
        int step = n_total / n_match;
        for (int i = 0, got = 0; got < n_match; i += step, got++) {
            *(fixed_4_t*)ecs_tree_get_mut(&ctx->w->trees[1], i) = vel;
        }
    } else {
        int n_skip = n_total - n_match;
        int step   = n_total / n_skip;
        for (int i = 0; i < n_total; i++) {
            if (i % step == 0) continue;
            *(fixed_4_t*)ecs_tree_get_mut(&ctx->w->trees[1], i) = vel;
        }
    }

    ctx->query.world              = ctx->w;
    ctx->query.tree_count         = 2;
    ctx->query.trees[0]           = &ctx->w->trees[0];
    ctx->query.trees[1]           = &ctx->w->trees[1];
    ctx->query.clause_count       = 1;
    ctx->query.clauses[0].include = (1u << 0) | (1u << 1);

    /* Cache warmup. Confirmed mode — writes land in place, pos accumulates
       (signed wrap is fine; benchmarks build without UBSan). */
    for (int i = 0; i < 70; i++) {
        ecs_iterator_t it = {0};
        ecs_iterator_init(&it, &ctx->query, 1u);
        bench_integrate_fn(&it);
    }

    return ctx;
}

void bench_integrate_iter(bench_integrate_ctx* ctx) {
    ecs_iterator_t it = {0};
    ecs_iterator_init(&it, &ctx->query, 1u);
    bench_integrate_fn(&it);
}

void bench_integrate_teardown(bench_integrate_ctx* ctx) {
    ecs_world_destroy(ctx->w);
    free(ctx->w);
    free(ctx);
}

struct bench_random_access_ctx {
    ecs_tree_t*       tree;
    int*              indices;
    int               n_access;
    volatile uint32_t sink;
};

bench_random_access_ctx* bench_random_access_setup(int n_total, int n_access) {
    bench_random_access_ctx* ctx = (bench_random_access_ctx*)calloc(1, sizeof(*ctx));
    ctx->n_access = n_access;
    ctx->indices  = (int*)malloc((size_t)n_access * sizeof(int));

    srand(0xC0FFEEu);
    for (int i = 0; i < n_access; i++) {
        unsigned r = ((unsigned)rand() << 15) ^ (unsigned)rand();
        ctx->indices[i] = (int)(r % (unsigned)n_total);
    }

    ctx->tree = (ecs_tree_t*)calloc(1, sizeof(ecs_tree_t));
    ecs_tree_init(ctx->tree, sizeof(fixed_4_t), 0);
    for (int i = 0; i < n_total; i++) {
        *(fixed_4_t*)ecs_tree_get_mut(ctx->tree, i) =
            fixed4_vec3(fixed_from_int(i), 0, 0);
    }
    ecs_tree_rollback(ctx->tree);

    /* Cache warmup. */
    for (int k = 0; k < 5; k++) {
        uint32_t s = 0;
        for (int i = 0; i < n_access; i++) {
            const fixed_4_t* p = (const fixed_4_t*)ecs_tree_get(ctx->tree, ctx->indices[i]);
            s += (uint32_t)p->x;
        }
        ctx->sink = s;
    }

    return ctx;
}

void bench_random_access_iter(bench_random_access_ctx* ctx) {
    uint32_t s = 0;
    for (int i = 0; i < ctx->n_access; i++) {
        const fixed_4_t* p = (const fixed_4_t*)ecs_tree_get(ctx->tree, ctx->indices[i]);
        s += (uint32_t)p->x;
    }
    ctx->sink = s;
}

void bench_random_access_teardown(bench_random_access_ctx* ctx) {
    ecs_tree_destroy(ctx->tree);
    free(ctx->tree);
    free(ctx->indices);
    free(ctx);
}

/* Readonly sum: iterate intersection (pos+vel) and accumulate pos into a
   single vec3. Setup mirrors bench_integrate_setup and runs one integrate
   iter so positions hold the post-integrate state. */
struct bench_sum_pos_ctx {
    ecs_world_t*         w;
    ecs_compiled_query_t query;
    fixed_4_t               sink;
};

static void bench_sum_pos_fn(ecs_iterator_t* it, fixed_4_t* out) {
    fixed_4_t s = fixed4_zero();
    uint64_t mask;
    while ((mask = ecs_iterator_next_block(it))) {
        while (mask) {
            int slot = ecs_ctz64(mask);
            mask &= mask - 1;
            const fixed_4_t* p = (const fixed_4_t*)ecs_iterator_get(it, 0, slot);
            s = fixed4_add(s, *p);
        }
    }
    *out = s;
}

bench_sum_pos_ctx* bench_sum_pos_setup(int n_total, int n_match) {
    bench_sum_pos_ctx* ctx = (bench_sum_pos_ctx*)calloc(1, sizeof(*ctx));
    ctx->w = (ecs_world_t*)calloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&ctx->w->trees[0], sizeof(fixed_4_t), 0);
    ecs_tree_init(&ctx->w->trees[1], sizeof(fixed_4_t), 0);
    ctx->w->mask = 3;

    for (int i = 0; i < n_total; i++) {
        *(fixed_4_t*)ecs_tree_get_mut(&ctx->w->trees[0], i) =
            fixed4_vec3(fixed_from_int(i), 0, 0);
    }

    fixed_4_t vel = fixed4_vec3(fixed_from_int(60), fixed_from_int(-30), 0);

    if (n_match <= n_total / 2) {
        int step = n_total / n_match;
        for (int i = 0, got = 0; got < n_match; i += step, got++) {
            *(fixed_4_t*)ecs_tree_get_mut(&ctx->w->trees[1], i) = vel;
        }
    } else {
        int n_skip = n_total - n_match;
        int step   = n_total / n_skip;
        for (int i = 0; i < n_total; i++) {
            if (i % step == 0) continue;
            *(fixed_4_t*)ecs_tree_get_mut(&ctx->w->trees[1], i) = vel;
        }
    }

    ctx->query.world              = ctx->w;
    ctx->query.tree_count         = 2;
    ctx->query.trees[0]           = &ctx->w->trees[0];
    ctx->query.trees[1]           = &ctx->w->trees[1];
    ctx->query.clause_count       = 1;
    ctx->query.clauses[0].include = (1u << 0) | (1u << 1);

    /* Advance pos once so the sum reads post-integrate state. */
    {
        ecs_iterator_t it = {0};
        ecs_iterator_init(&it, &ctx->query, 1u);
        uint64_t mask;
        while ((mask = ecs_iterator_next_block(&it))) {
            while (mask) {
                int slot = ecs_ctz64(mask);
                mask &= mask - 1;
                const fixed_4_t* vel_p   = (const fixed_4_t*)ecs_iterator_get(&it, 1, slot);
                const fixed_4_t* pos_cur = (const fixed_4_t*)ecs_iterator_get(&it, 0, slot);
                fixed_4_t new_pos = fixed4_add(*pos_cur, fixed4_scale(*vel_p, BENCH_DT));
                *(fixed_4_t*)ecs_iterator_get_mut(&it, 0, slot) = new_pos;
            }
        }
    }

    /* Cache warmup. */
    for (int k = 0; k < 70; k++) {
        ecs_iterator_t it = {0};
        ecs_iterator_init(&it, &ctx->query, 0u);
        bench_sum_pos_fn(&it, &ctx->sink);
    }

    return ctx;
}

void bench_sum_pos_iter(bench_sum_pos_ctx* ctx) {
    ecs_iterator_t it = {0};
    ecs_iterator_init(&it, &ctx->query, 0u);
    bench_sum_pos_fn(&it, &ctx->sink);
}

void bench_sum_pos_teardown(bench_sum_pos_ctx* ctx) {
    ecs_world_destroy(ctx->w);
    free(ctx->w);
    free(ctx);
}

