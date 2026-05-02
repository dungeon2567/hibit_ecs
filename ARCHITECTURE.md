# HibitEcs Architecture

A predict-mode ECS for client-side networked simulation. Two states per component (confirmed / predicted), no undo log, branchless hot paths, sparse bitset traversal. Allocation goes through mimalloc; no custom arena.

## 1. Storage layout

### 1.1 Three-level radix tree

Each component type lives in its own `ecs_tree_t`. Entity index is an 18-bit integer split into three 6-bit lanes:

```
index : [ l3 (6) | l2 (6) | l1 (6) ]   →  64 × 64 × 64 = 262 144 slots
```

- `ecs_l3_t` — root node, owned by `ecs_tree_t`. 64 child pointers to `ecs_l2_t`.
- `ecs_l2_t` — 64 child pointers to `ecs_l1_t`.
- `ecs_l1_t` — leaf node. Holds presence bitmaps and an inline 128-slot data tail (`[0..63]` confirmed, `[64..127]` predicted).

Empty subtrees point at two singletons: `ecs_default_l2` and `ecs_default_l1`. Both contain all-zero masks, so iteration and queries treat them as "no entities present" without a null check. Allocation only happens on first write.

### 1.2 Inline data tail

`ecs_l1_t` is allocated as `sizeof(ecs_l1_t) + 128 * data_size` bytes, 64-byte aligned. Slot `i`:

- confirmed slot pointer: `data + i * data_size`
- predicted slot pointer: `data + (i + 64) * data_size`

Both halves live in the same allocation. No pointer chasing between them.

For tag components (`data_size == 0`), the tail is empty; presence-only state lives in the masks.

### 1.3 Per-node bitmaps

Every node tracks four bitmaps:

| field                  | meaning                                                         |
|------------------------|-----------------------------------------------------------------|
| `confirmed_mask_any`   | authoritative presence — advances only via CONFIRMED writes     |
| `predicted_mask_any`   | live working set this frame — what reads/writes go through      |
| `dirty`                | slots written in PREDICT mode since last rollback               |
| `changed`              | slots written this tick; cleared by `ecs_tree_rollback`         |

L2 and L3 also carry `confirmed_mask_all` / `predicted_mask_all` to prune exclusion clauses without scanning. Bit set iff the corresponding subtree is fully populated: at L2, `mask_all[j] == 1` iff `children[j].mask_any == ~0ULL`; at L3, `mask_all[i] == 1` iff `children[i].mask_all == ~0ULL`. Iterators drop subtrees where `mask_all` indicates total fullness for an excluded term. Invariant verified by `ecs_tree_masks_valid`.

## 2. Modes, predict, rollback

Global VM mode is `ECS_MODE_CONFIRMED` (default) or `ECS_MODE_PREDICT`, set via `ecs_world_set_mode` / `ecs_tree_set_mode`. The mode dictates write semantics:

- **CONFIRMED**: `ecs_tree_get_mut` / `ecs_tree_remove` write directly to the confirmed slot, masks mirror, `dirty` stays 0. Used for server-acked state (or single-player authoritative sim).
- **PREDICT**: writes go to the predicted slot, set the corresponding `dirty` bit, leave confirmed untouched. Used for speculative client-side simulation.

Reads via `ecs_tree_get` / `ecs_iterator_get` use the "current frame" view — predicted bytes when `dirty` is set, confirmed bytes otherwise — branchless via the `+64` slot offset:

```c
off = slot + ((dirty >> slot) & 1) << 6;
```

There is **no `promote`**. Predicted state is always speculative; it never folds into confirmed. To advance confirmed state, switch to CONFIRMED mode and re-apply the writes (server messages do this naturally).

`ecs_tree_rollback` is the tick-end op:

- discards predicted bytes (`predicted_mask = confirmed_mask`, `dirty = 0`)
- clears `changed`
- releases L1/L2 nodes that just went empty
- if any CONFIRMED writes landed this tick, bumps `tree->tick`

`ecs_world_rollback` walks every populated tree.

Mode switching requires no in-flight prediction (`dirty == 0` everywhere) — asserted via `ecs_tree_no_dirty`.

## 3. Allocation

Direct mimalloc — `mi_malloc_aligned` for new nodes (`ecs_xmalloc_aligned` is a thin OOM-aborting wrapper), `mi_free` on release. L1 and L2 allocation paths are inline (`ecs_l1_acquire` / `ecs_l2_acquire`); empty L1/L2 nodes are released back to the allocator inline by `ecs_tree_remove` (CONFIRMED mode) or `ecs_tree_rollback` (PREDICT mode emptied subtrees). No custom arena, no per-tree node pool, no per-frame allocation in the steady state once L1/L2 nodes are populated.

mimalloc reuses freed slabs internally, so churning predict-only adds is cheap.

## 4. Queries

### 4.1 Compiled query

`ecs_compiled_query_t` carries up to 8 trees and up to 8 clauses. Each clause is a 3-tuple of 32-bit masks indexing into `trees[]`:

```c
typedef struct {
    uint32_t include;     // bits AND'd into result mask
    uint32_t exclude;     // bits AND'd with ~mask_any (L1) / ~mask_all (L2/L3)
    uint32_t changed;     // any of these trees must have a set `changed` bit
} ecs_compiled_clause_t;
```

Clauses are OR'd; terms within a clause are AND'd. This gives DNF expressivity over presence and tick-changed state without an interpreter. The query expression grammar lives in `ecs_query.c`.

### 4.2 Iterator

`ecs_iterator_t` walks all three levels in DFS order, carrying current `(l3_idx, l2_idx, l1_idx)` plus the remaining bitmap at each level (`l3_mask`, `l2_mask`, `l1_mask`).

The hot path (`ecs_iterator_next`, header-inline) is two instructions: CTZ + clear-low-bit on `l1_mask`. When `l1_mask == 0` it falls through to `ecs_iterator_next_slow` in `ecs.c`, which advances `l2_idx` (or `l3_idx`), reloads the L1 row across all trees, and recomputes `l1_mask` via `iter_compute_l1_mask`. Three near-identical routines (`iter_compute_l{1,2,3}_mask`) evaluate every clause against the current row by AND'ing presence/exclude bitmaps and OR'ing across clauses. They run only on level transitions, not per entity.

### 4.3 Branchless data access

`ecs_iterator_get` reads the current frame view (predicted-if-dirty, else confirmed) using the same `+64` slot trick as `ecs_tree_get`.

## 5. World

`ecs_world_t` is a thin wrapper: 64 trees in a fixed array, plus a `mask` of which slots are populated, a global `tick`, a `tick_id` counter bumped by `ecs_world_begin_tick`, and a `mode` mirrored onto every populated tree by `ecs_world_set_mode`. World-level rollback iterates the populated trees. `ecs_world_crc64` over confirmed state gives a deterministic checksum suitable for desync detection.

## 6. Fixed-point math (`fixed.h`)

Q16.16 fixed-point with three SIMD backends selected at compile time:

- `FIXED_BACKEND_X86` — SSE2/AVX2 (`__m128i`, `__m256i`)
- `FIXED_BACKEND_NEON` — ARM NEON (`int32x4_t`)
- `FIXED_BACKEND_SCALAR` — portable fallback

Add/sub/neg/mul/cmp are SIMD; `_div` falls back to per-lane scalar (no integer SIMD divide on either ISA). Multiply uses widen-to-int64 + logical right shift to recover Q16.16 without overflow.

Determinism: integer-only paths and the same shift semantics on every platform. No floating-point in the simulation hot paths.

## 7. Binary serializer (`ecs_serializer.h`)

Header-only bitpacked writer/reader (`ecs_serializer_t` / `ecs_deserializer_t`). Right-to-left scratch accumulator (`uint64_t`), 64-bit qword-aligned output, network byte order. `ecs_serializer_write_bytes` / `ecs_deserializer_read_bytes` align to qword boundary then `memcpy` the body, falling back to bit-level writes for head/tail bytes. Every function `static inline` for cross-TU inlining.

`ecs_tree_serialize(tree, ecs_serializer_t*)` dumps confirmed state mask-driven. Mask encoding picks per-mask between raw u64, an all-set shortcut, or an indexed list of bit positions (polarity + k(3) + k×idx(6)) — see `ecs_serialize_mask` in `ecs.c`. Per-L1 batch encoding goes through `tree->serialize_batch` (defaults to `ecs_serialize_batch_raw`, which packs only set-bit slots, coalescing runs via `ecs_mask_pop_run`).

## 8. Hot-path summary

| Operation                       | Cost                                            |
|---------------------------------|-------------------------------------------------|
| `ecs_iterator_next` (per entity)| 2 ops: CTZ + AND                                |
| `ecs_iterator_get` (per entity) | 1 shift, 1 mask, 1 add, 1 multiply, 1 load      |
| Level transition                | CTZ on level mask, recompute next-level mask    |
| `ecs_tree_get_mut` (CONFIRMED)  | 3 mask updates per level + node alloc on miss   |
| `ecs_tree_get_mut` (PREDICT)    | same + sets dirty/changed bits                  |
| `ecs_tree_rollback`             | walks dirty hierarchy, releases empty L1/L2     |
| Read of confirmed state         | 2 pointer loads + 1 indexed access (no branch)  |

No per-frame allocation in steady state. No locking. No archetype rebuilds. No virtual dispatch.

## 9. File map

| File                       | Responsibility                                       |
|----------------------------|------------------------------------------------------|
| `src/ecs.h`                | Public API + inline hot-path code                    |
| `src/ecs.c`                | Iterator slow path, rollback, CRC, serializer        |
| `src/ecs_query.c`          | Query expression parser → compiled clauses           |
| `src/ecs_serializer.h`     | Bitpacked stream reader/writer                       |
| `src/fixed.h`              | Q16.16 fixed-point + SIMD vector types               |
| `src/ecs_math.h`           | Fixed-point vector helpers                           |
| `tests/`                   | Unit tests (ECS, modes, iterator, serialize, fixed)  |
| `benchmarks/`              | Microbenchmarks (Release builds)                     |
| `main.c`                   | Test driver entry point                              |
