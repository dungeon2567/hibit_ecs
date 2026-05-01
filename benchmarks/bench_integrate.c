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
    while (ecs_iterator_next(it)) {
        const vec3_t* vel     = (const vec3_t*)ecs_iterator_get(it, 1);
        const vec3_t* pos_cur = (const vec3_t*)ecs_iterator_get(it, 0);
        vec3_t        pos_new = vec3_add(*pos_cur, vec3_scale(*vel, BENCH_DT));
        ecs_iterator_set(it, 0, &pos_new);
    }
}

bench_integrate_ctx* bench_integrate_setup(int n_total, int n_match) {
    bench_integrate_ctx* ctx = (bench_integrate_ctx*)calloc(1, sizeof(*ctx));
    ctx->n_match = n_match;
    ctx->w       = (ecs_world_t*)calloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&ctx->w->trees[0], sizeof(vec3_t));
    ecs_tree_init(&ctx->w->trees[1], sizeof(vec3_t));
    ctx->w->mask = 3;

    for (int i = 0; i < n_total; i++) {
        vec3_t pos = vec3_make(fixed_from_int(i), 0, 0);
        ecs_tree_set(&ctx->w->trees[0], i, &pos);
    }

    vec3_t vel = vec3_make(fixed_from_int(60), fixed_from_int(-30), 0);

    if (n_match <= n_total / 2) {
        int step = n_total / n_match;
        for (int i = 0, got = 0; got < n_match; i += step, got++) {
            ecs_tree_set(&ctx->w->trees[1], i, &vel);
        }
    } else {
        int n_skip = n_total - n_match;
        int step   = n_total / n_skip;
        for (int i = 0; i < n_total; i++) {
            if (i % step == 0) continue;
            ecs_tree_set(&ctx->w->trees[1], i, &vel);
        }
    }

    ecs_world_rollback(ctx->w);

    /* Predict mode so per-iter integrate writes are discarded by rollback —
       otherwise pos accumulates and overflows int32 (UBSan trip in NEON). */
    ecs_tree_set_mode(&ctx->w->trees[0], ECS_MODE_PREDICT);
    ecs_tree_set_mode(&ctx->w->trees[1], ECS_MODE_PREDICT);

    ctx->query.tree_count         = 2;
    ctx->query.trees[0]           = &ctx->w->trees[0];
    ctx->query.trees[1]           = &ctx->w->trees[1];
    ctx->query.clause_count       = 1;
    ctx->query.clauses[0].include = (1u << 0) | (1u << 1);

    /* Cache warmup. */
    for (int i = 0; i < 70; i++) {
        ecs_iterator_t it = {0};
        ecs_iterator_init(&it, &ctx->query);
        it.write_mask = 1u;
        bench_integrate_fn(&it);
        ecs_world_rollback(ctx->w);
    }

    return ctx;
}

void bench_integrate_iter(bench_integrate_ctx* ctx) {
    ecs_iterator_t it = {0};
    ecs_iterator_init(&it, &ctx->query);
    it.write_mask = 1u;
    bench_integrate_fn(&it);
    ecs_world_rollback(ctx->w);
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
    ecs_tree_init(ctx->tree, sizeof(vec3_t));
    for (int i = 0; i < n_total; i++) {
        vec3_t p = vec3_make(fixed_from_int(i), 0, 0);
        ecs_tree_set(ctx->tree, i, &p);
    }
    ecs_tree_rollback(ctx->tree);

    /* Cache warmup. */
    for (int k = 0; k < 5; k++) {
        uint32_t s = 0;
        for (int i = 0; i < n_access; i++) {
            const vec3_t* p = (const vec3_t*)ecs_tree_get(ctx->tree, ctx->indices[i]);
            s += (uint32_t)p->x;
        }
        ctx->sink = s;
    }

    return ctx;
}

void bench_random_access_iter(bench_random_access_ctx* ctx) {
    uint32_t s = 0;
    for (int i = 0; i < ctx->n_access; i++) {
        const vec3_t* p = (const vec3_t*)ecs_tree_get(ctx->tree, ctx->indices[i]);
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
