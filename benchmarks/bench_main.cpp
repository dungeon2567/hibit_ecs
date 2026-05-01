#include <benchmark/benchmark.h>

extern "C" {
#include "bench_integrate.h"
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

int main(int argc, char** argv) {
    ecs_crc64_init();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
