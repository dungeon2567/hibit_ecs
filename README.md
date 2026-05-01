# hibit_ecs

Fixed-point ECS in C11 with predict/rollback ticks, owned-component lifecycle, and a SIMD-accelerated math layer (SSE4.1+AVX2 / NEON / scalar).

[![build](https://github.com/dungeon2567/hibit_ecs/actions/workflows/build.yml/badge.svg)](https://github.com/dungeon2567/hibit_ecs/actions/workflows/build.yml)

## Performance

[![sparse](https://img.shields.io/endpoint?url=https://dungeon2567.github.io/hibit_ecs/dev/bench/badges/integrate_sparse.json)](https://dungeon2567.github.io/hibit_ecs/dev/bench/)
[![dense](https://img.shields.io/endpoint?url=https://dungeon2567.github.io/hibit_ecs/dev/bench/badges/integrate_dense.json)](https://dungeon2567.github.io/hibit_ecs/dev/bench/)
[![rand10k](https://img.shields.io/endpoint?url=https://dungeon2567.github.io/hibit_ecs/dev/bench/badges/random_access10k.json)](https://dungeon2567.github.io/hibit_ecs/dev/bench/)
[![rand100k](https://img.shields.io/endpoint?url=https://dungeon2567.github.io/hibit_ecs/dev/bench/badges/random_access100k.json)](https://dungeon2567.github.io/hibit_ecs/dev/bench/)

[**→ live dashboard**](https://dungeon2567.github.io/hibit_ecs/dev/bench/) · [**SUMMARY.md**](https://dungeon2567.github.io/hibit_ecs/dev/bench/SUMMARY.md)
