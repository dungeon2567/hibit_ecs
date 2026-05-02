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

typedef struct bench_sum_pos_ctx bench_sum_pos_ctx;

bench_sum_pos_ctx* bench_sum_pos_setup(int n_total, int n_match);
void               bench_sum_pos_iter(bench_sum_pos_ctx* ctx);
void               bench_sum_pos_teardown(bench_sum_pos_ctx* ctx);

typedef struct bench_sum_pos_soa_ctx bench_sum_pos_soa_ctx;

bench_sum_pos_soa_ctx* bench_sum_pos_soa_setup(int n);
void                   bench_sum_pos_soa_iter(bench_sum_pos_soa_ctx* ctx);
void                   bench_sum_pos_soa_teardown(bench_sum_pos_soa_ctx* ctx);
