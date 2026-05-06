# hibit_ecs

A predict/rollback ECS in C11 for client-side networked simulation. Each component type lives in its own three-level radix tree (64×64×64 = 262 144 slots) with paired confirmed/predicted slots, sparse bitset traversal, and branchless hot paths. Designed around fixed-point math, deterministic ticks, and a SIMD-accelerated math layer (SSE4.1+AVX2 / NEON / scalar) so the same simulation can run on the server and every client.

Core features:

- **Predict / rollback ticks** — speculative writes go to the predicted half of each slot; `ecs_world_rollback` discards them when the server's confirmed state arrives. No undo log, no per-frame allocation in steady state.
- **Owned-component lifecycle** — components are acquired and released through the world, with strict mode separation between `CONFIRMED` (authoritative) and `PREDICT` (speculative) writes.
- **Compiled queries** — include / exclude terms are folded into bitmask ops over the radix tree; empty subtrees route to shared all-zero singletons so iteration skips them without null checks.
- **Networking primitives** — bit-level serializer, input buffer with redundancy, command stream that handles out-of-order and duplicate packets while preserving deterministic order.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the storage layout, mode rules, and query compiler.

[![build](https://github.com/dungeon2567/hibit_ecs/actions/workflows/build.yml/badge.svg)](https://github.com/dungeon2567/hibit_ecs/actions/workflows/build.yml)
