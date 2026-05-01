#pragma once

typedef struct bench_integrate_ctx bench_integrate_ctx;

bench_integrate_ctx* bench_integrate_setup(int n_total, int n_match);
void                 bench_integrate_iter(bench_integrate_ctx* ctx);
void                 bench_integrate_teardown(bench_integrate_ctx* ctx);

typedef struct bench_integrate_soa_ctx bench_integrate_soa_ctx;

bench_integrate_soa_ctx* bench_integrate_soa_setup(int n);
void                     bench_integrate_soa_iter(bench_integrate_soa_ctx* ctx);
void                     bench_integrate_soa_teardown(bench_integrate_soa_ctx* ctx);

typedef struct bench_random_access_ctx bench_random_access_ctx;

bench_random_access_ctx* bench_random_access_setup(int n_total, int n_access);
void                     bench_random_access_iter(bench_random_access_ctx* ctx);
void                     bench_random_access_teardown(bench_random_access_ctx* ctx);
