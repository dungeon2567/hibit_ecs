#pragma once

typedef struct bench_broadphase_ctx bench_broadphase_ctx;

/* All three setups share the same item layout: `n_items` AABBs scattered on
   a deterministic lattice + LCG jitter inside a fixed world cube. Queries are
   `n_queries` small AABBs centred on deterministic points within the same
   cube. Same seed each call -> reproducible across runs. */
bench_broadphase_ctx* bench_broadphase_setup(int n_items, int n_queries);

/* Scatter layout: items placed on a deterministic 3D lattice with tiny
   sub-cell jitter so no two AABBs overlap. Spacing is 4 fixed units, item
   half-extent 1 -> guaranteed >=2-unit gap on every axis. Same iter / teardown
   API as the random-scatter setup above. */
bench_broadphase_ctx* bench_broadphase_setup_scatter(int n_items, int n_queries);

/* Build-only: clear, re-insert all items, build. Measures rebuild cost. */
void bench_broadphase_iter_build(bench_broadphase_ctx* ctx);

/* Query-only: tree built once in setup, each iter runs n_queries traversals.
   Measures pure traversal cost (build amortized to zero). */
void bench_broadphase_iter_query(bench_broadphase_ctx* ctx);

/* Full tick: clear + insert + build + n_queries. Mirrors the per-frame cost
   a physics broadphase pays on a fully dynamic scene. */
void bench_broadphase_iter_full(bench_broadphase_ctx* ctx);

void bench_broadphase_teardown(bench_broadphase_ctx* ctx);
