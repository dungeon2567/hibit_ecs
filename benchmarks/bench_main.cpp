#include <benchmark/benchmark.h>

extern "C" {
#include "bench_integrate.h"
#include "bench_broadphase.h"
void ecs_crc64_init(void);
}

static void BM_IntegrateSparse(benchmark::State& state) {
    bench_integrate_ctx* ctx = bench_integrate_setup(10000, 500);
    for (auto _ : state) {
        bench_integrate_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 500);
    bench_integrate_teardown(ctx);
}
BENCHMARK(BM_IntegrateSparse)->Unit(benchmark::kMicrosecond);

static void BM_IntegrateDense(benchmark::State& state) {
    bench_integrate_ctx* ctx = bench_integrate_setup(10000, 9500);
    for (auto _ : state) {
        bench_integrate_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 9500);
    bench_integrate_teardown(ctx);
}
BENCHMARK(BM_IntegrateDense)->Unit(benchmark::kMicrosecond);

static void BM_IntegrateDense100k(benchmark::State& state) {
    bench_integrate_ctx* ctx = bench_integrate_setup(100000, 95000);
    for (auto _ : state) {
        bench_integrate_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 95000);
    bench_integrate_teardown(ctx);
}
BENCHMARK(BM_IntegrateDense100k)->Unit(benchmark::kMicrosecond);

/* SOA baseline: same N as BM_IntegrateDense, no ECS — pure two-array vec3 loop.
   Lower bound for what the integrate kernel could ever cost. */
static void BM_IntegrateDenseSOA(benchmark::State& state) {
    bench_integrate_soa_ctx* ctx = bench_integrate_soa_setup(9500);
    for (auto _ : state) {
        bench_integrate_soa_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 9500);
    bench_integrate_soa_teardown(ctx);
}
BENCHMARK(BM_IntegrateDenseSOA)->Unit(benchmark::kMicrosecond);

static void BM_IntegrateDenseSOA100k(benchmark::State& state) {
    bench_integrate_soa_ctx* ctx = bench_integrate_soa_setup(95000);
    for (auto _ : state) {
        bench_integrate_soa_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 95000);
    bench_integrate_soa_teardown(ctx);
}
BENCHMARK(BM_IntegrateDenseSOA100k)->Unit(benchmark::kMicrosecond);

static void BM_RandomAccess10k(benchmark::State& state) {
    bench_random_access_ctx* ctx = bench_random_access_setup(100000, 10000);
    for (auto _ : state) {
        bench_random_access_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 10000);
    bench_random_access_teardown(ctx);
}
BENCHMARK(BM_RandomAccess10k)->Unit(benchmark::kMicrosecond);

static void BM_RandomAccess100k(benchmark::State& state) {
    bench_random_access_ctx* ctx = bench_random_access_setup(100000, 100000);
    for (auto _ : state) {
        bench_random_access_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 100000);
    bench_random_access_teardown(ctx);
}
BENCHMARK(BM_RandomAccess100k)->Unit(benchmark::kMicrosecond);

/* Readonly sum: positions post-integrate, accumulated to a single vec3.
   Sparse / dense mirror BM_Integrate*. SOA is the no-ECS lower bound. */
static void BM_SumPosSparse(benchmark::State& state) {
    bench_sum_pos_ctx* ctx = bench_sum_pos_setup(10000, 500);
    for (auto _ : state) {
        bench_sum_pos_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 500);
    bench_sum_pos_teardown(ctx);
}
BENCHMARK(BM_SumPosSparse)->Unit(benchmark::kMicrosecond);

static void BM_SumPosDense(benchmark::State& state) {
    bench_sum_pos_ctx* ctx = bench_sum_pos_setup(10000, 9500);
    for (auto _ : state) {
        bench_sum_pos_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 9500);
    bench_sum_pos_teardown(ctx);
}
BENCHMARK(BM_SumPosDense)->Unit(benchmark::kMicrosecond);

static void BM_SumPosDense100k(benchmark::State& state) {
    bench_sum_pos_ctx* ctx = bench_sum_pos_setup(100000, 95000);
    for (auto _ : state) {
        bench_sum_pos_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 95000);
    bench_sum_pos_teardown(ctx);
}
BENCHMARK(BM_SumPosDense100k)->Unit(benchmark::kMicrosecond);

static void BM_SumPosDenseSOA(benchmark::State& state) {
    bench_sum_pos_soa_ctx* ctx = bench_sum_pos_soa_setup(9500);
    for (auto _ : state) {
        bench_sum_pos_soa_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 9500);
    bench_sum_pos_soa_teardown(ctx);
}
BENCHMARK(BM_SumPosDenseSOA)->Unit(benchmark::kMicrosecond);

static void BM_SumPosDenseSOA100k(benchmark::State& state) {
    bench_sum_pos_soa_ctx* ctx = bench_sum_pos_soa_setup(95000);
    for (auto _ : state) {
        bench_sum_pos_soa_iter(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 95000);
    bench_sum_pos_soa_teardown(ctx);
}
BENCHMARK(BM_SumPosDenseSOA100k)->Unit(benchmark::kMicrosecond);

/* Broadphase: 10k items + 10k queries. Three slices --
     Build  rebuilds tree from a fixed insert list,
     Query  reuses a tree built once in setup,
     Full   pays clear+insert+build+queries every iter (per-frame cost). */
static void BM_BroadphaseBuild10k(benchmark::State& state) {
    bench_broadphase_ctx* ctx = bench_broadphase_setup(10000, 10000);
    for (auto _ : state) {
        bench_broadphase_iter_build(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 10000);
    bench_broadphase_teardown(ctx);
}
BENCHMARK(BM_BroadphaseBuild10k)->Unit(benchmark::kMicrosecond);

static void BM_BroadphaseQuery10k(benchmark::State& state) {
    bench_broadphase_ctx* ctx = bench_broadphase_setup(10000, 10000);
    for (auto _ : state) {
        bench_broadphase_iter_query(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 10000);
    bench_broadphase_teardown(ctx);
}
BENCHMARK(BM_BroadphaseQuery10k)->Unit(benchmark::kMicrosecond);

static void BM_BroadphaseFull10k(benchmark::State& state) {
    bench_broadphase_ctx* ctx = bench_broadphase_setup(10000, 10000);
    for (auto _ : state) {
        bench_broadphase_iter_full(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 10000);
    bench_broadphase_teardown(ctx);
}
BENCHMARK(BM_BroadphaseFull10k)->Unit(benchmark::kMicrosecond);

/* Scatter variant: 10k lattice-placed items (no overlap) + 10k queries. */
static void BM_BroadphaseBuildScatter10k(benchmark::State& state) {
    bench_broadphase_ctx* ctx = bench_broadphase_setup_scatter(10000, 10000);
    for (auto _ : state) {
        bench_broadphase_iter_build(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 10000);
    bench_broadphase_teardown(ctx);
}
BENCHMARK(BM_BroadphaseBuildScatter10k)->Unit(benchmark::kMicrosecond);

static void BM_BroadphaseQueryScatter10k(benchmark::State& state) {
    bench_broadphase_ctx* ctx = bench_broadphase_setup_scatter(10000, 10000);
    for (auto _ : state) {
        bench_broadphase_iter_query(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 10000);
    bench_broadphase_teardown(ctx);
}
BENCHMARK(BM_BroadphaseQueryScatter10k)->Unit(benchmark::kMicrosecond);

static void BM_BroadphaseFullScatter10k(benchmark::State& state) {
    bench_broadphase_ctx* ctx = bench_broadphase_setup_scatter(10000, 10000);
    for (auto _ : state) {
        bench_broadphase_iter_full(ctx);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 10000);
    bench_broadphase_teardown(ctx);
}
BENCHMARK(BM_BroadphaseFullScatter10k)->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv) {
    ecs_crc64_init();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
