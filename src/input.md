# `ecs_input` — Architecture & Layout

Per-tick player input store for deterministic netcode. Server- and
client-symmetric. Holds opaque, fixed-stride input bytes for every
`(tick, slot)` cell in a wrap-around ring, with predicted/confirmed
flags and a sim-driven frontier.

The module is **slot-keyed**. Callers allocate slots via
`ecs_input_alloc_slot` and pass slots directly to `set` / `get`. Any
`pid → slot` mapping (if needed) lives outside this module.

---

## 1. Quality assessment

**Verdict: production-grade for its design envelope (≤1024 slots,
≤2048 buffered ticks).**

What's good:

| Aspect | Notes |
|---|---|
| **Documentation** | Module-level header comment is exhaustive: semantics, threading, endianness, write rules, allocation philosophy. Public API is fully commented. |
| **Layout** | Single contiguous 2D table. Per-tick op = one row touched = warm cache. Power-of-two dimensions everywhere → mask-and-shift addressing. |
| **Hot path** | Zero allocations on `set` / `get` / `get_view` / `clear`. Slot keying is O(1) — no lookup, no scan. |
| **Bitmaps** | `present`, `confirmed`, and `live_bm` are flat `u64` arrays — both fit a few cache lines for typical roster sizes. |
| **`all_confirmed`** | Computed on demand by ANDing the row's confirmed bits with `live_bm`. No cached counters to drift out of sync. |
| **Auto-grow** | Both dimensions double on demand and re-lay out under the new mask. Caller never has to size for worst case. |
| **Correctness details** | ABA defense on `alloc_slot` (scrubs stale bits in still-live rows). First-touch carry-forward (`in_advance_row`) implements input persistence. Predicted-after-confirmed writes are dropped, not silently overwritten. Tick `0` is reserved as a "no frontier" sentinel. |
| **Memory** | ~2 small allocs at init, zero per tick, columns recycled via lowest-free-bit reuse. |

Intentional non-features:

- **Not thread-safe.** By design — caller serializes. Atomics on every
  bitmap word would dominate the hot path with no benefit for the
  single-threaded apply model this module targets.

The module is internally consistent, well factored, and faithful to
its stated design.

---

## 2. Mental model

```
                      ┌──────────────────────┐
   tick & buf_mask ──►│ row[t]               │  one ring slot
                      ├──────────────────────┤
                      │ tick_in_slot  (u32)  │  absolute tick id here (+4B pad)
                      │ present[W]    (u64)  │  bit per dense column
                      │ confirmed[W]  (u64)  │  bit per dense column
                      │ payload[A*S]  (u8)   │  stride bytes per column
                      └──────────────────────┘
                                ▲
                        slot ── direct column index, no lookup
```

- `slot` is the engine-internal **dense column index**: returned by
  `ecs_input_alloc_slot`, used as the column in every row, and tested
  against `live_bm` for membership.
- `tick` is an absolute 32-bit id; the ring position is
  `tick & buf_mask`.
- `tick_in_slot[t]` disambiguates ring wrap: when a write for tick T
  finds a slot holding T' ≠ T, that's a first-touch and the row is
  reset (with carry-forward of predicted bytes).

---

## 3. Storage layout

Single allocation: `it->table`, `buf_size` rows × `row_bytes` each,
contiguous. Layout per row:

```
offset 0           u32 tick_in_slot   (+4 bytes pad to keep bitmaps 8B aligned)
offset 8           u64 present   [words_per_row]
offset 8 + W*8     u64 confirmed [words_per_row]
offset 8 + 2*W*8   u8  payload   [active_cap * stride]
```

Where:

- `W = words_per_row = ceil(active_cap / 64)`
- `A = active_cap` (column count, pow2)
- `S = stride` (bytes per input slot, frozen at init)
- `row_bytes = pad8(8 + 2*W*8 + A*S)`

A separate `live_bm[W]` array (sibling of the table) tracks slot
membership. It is realloc'd alongside the table when `active_cap`
grows.

Both dimensions grow pow2:

- **`active_cap` grow** → row width changes. `in_realloc_table`
  allocates a new table, walks every old row, re-lays bitmaps and
  payload into the wider row, leaving new columns uninitialized
  (presence bits gate reads). `live_bm` is re-allocated with old bits
  preserved and new words zeroed.
- **`buf_size` grow** → row count changes. Same `in_realloc_table`
  re-maps every live row to its new slot under the wider mask.

Auto-grow triggers:

- `set` detects a first-touch that would clobber a still-live row
  (`prev_tick > confirmed_frontier`) and doubles `buf_size` until
  capacity ≥ `tick - prev + 1` (capped at `ECS_INPUT_BUFSIZE_MAX`).
- `alloc_slot` detects `active_count >= active_cap` and doubles
  `active_cap`.

---

## 4. Slot membership

`live_bm[]` is a `uint64_t` array of length `W`. Bit `idx` set ⇔
slot `idx` is currently registered.

`alloc_slot` finds the lowest free slot via word-wise ctz on the
**inverted** live mask:

```c
for (uint32_t w = 0; w < W; w++) {
    uint64_t inv = ~live_bm[w];
    if (inv) {
        uint32_t bit  = ecs_ctz64(inv);
        uint32_t slot = (w << 6) + bit;
        if (slot < active_cap) return slot;
    }
}
```

Per-call cost: one cache-resident `u64` AND/test/ctz per 64 slots.
For typical `active_cap ≤ 64` this is a single instruction.

Consequences:

- No SIMD scan, no `pid → slot` reverse map. The cost of the old
  lookup (8-lane `fixed_8_t` equality) collapsed to one bit test.
- Holes are reused lowest-first. Reuse-stable iteration order is
  determined entirely by the alloc/free sequence — peers must replay
  the same membership stream for cross-peer determinism.
- `active_count = popcount(live_bm)` is maintained as a scalar.

---

## 5. Bitmaps and `all_confirmed`

`present[w]` bit `b` is set iff column `w*64+b` was written for the
current `tick_in_slot[t]`. `confirmed[w]` is the analogous flag for
authoritative writes.

`in_tick_all_confirmed` doesn't store an "all confirmed" cache. It
ANDs the row's confirmed bits with `live_bm` directly:

```c
for (uint32_t w = 0; w < W; w++) {
    uint64_t live = live_bm[w];
    if ((confirmed[w] & live) != live) return false;
}
```

This is robust against stale bits left over from past slot owners
because `live` reflects the current registration state.

ABA defense: when `alloc_slot` (re)allocates `slot`, it walks every
still-live row (`tick_in_slot != NIL`) and clears bit `slot` in both
`present` and `confirmed`. Without this, a re-alloc'd slot would
inherit the stale bits of its predecessor. The scrub runs on every
alloc — empty rows have `tick == NIL` and are skipped, so the cost is
proportional to the populated ring depth.

---

## 6. Hot path semantics

### `ecs_input_set(tick, slot, value, confirmed)`

1. Bail if `slot` is out of range or `live_bm` bit is clear.
2. Check first-touch eviction — if the target slot holds a
   still-live tick (`> frontier`), grow `buf_size`.
3. `t = tick & buf_mask`; if `tick_in_slot[t] != tick`, this is a
   first touch:
   - `in_advance_row` zeroes both bitmaps and, if the previous tick's
     row is still resident, carries forward each live slot's bytes
     (predicted) and sets their present bits.
   - `tick_in_slot[t] = tick`.
4. Write semantics:
   - `confirmed=false` + already confirmed → no-op (drop predicted).
   - Otherwise `memcpy` the value, set `present` bit.
   - `confirmed=true` additionally sets `confirmed` bit.

### `ecs_input_get(tick, slot)`

1. Bail if `slot` is dead.
2. `t = tick & buf_mask`; require `tick_in_slot[t] == tick`.
3. Require `present[idx>>6] & (1 << idx&63)`.
4. Return `payload + idx * stride`.

### `ecs_input_get_view(tick, slot)`

Same as `get` but additionally reports `present` and `confirmed`
flags so the caller can branch without two table walks.

### `ecs_input_clear(tick)`

Admin/rollback only. Zeroes both bitmaps and stamps
`tick_in_slot[t] = NIL`. Does **not** rewind the frontier.

---

## 7. Frontier

`confirmed_frontier` is a sim-driven low-water mark, not auto-tracked.
Sim calls `ecs_input_advance_to_tick(tick)` once it has finalized
everything `≤ tick`. Monotonic.

Effects:

- Ring slots holding ticks `≤ frontier` are eligible for eviction on
  first-touch by a newer write. They aren't scrubbed eagerly —
  queries on stale ticks still succeed until something overwrites.
- `set` only auto-grows the ring when the slot it would evict holds
  a tick **above** the frontier (i.e. still in the live rollback
  window).

`tick = 0` is reserved as the "no frontier yet" sentinel; valid
ticks start at `1`.

---

## 8. Lifecycle and growth costs

| Op | Allocs | Frees | Notes |
|---|---|---|---|
| `init` | 1 (table) + live_bm deferred to first alloc | 0 | row_bytes = 8 (header only) until first alloc_slot |
| `set` / `get` / `get_view` / `clear` | 0 | 0 | hot path is pure pointer math + bit test |
| `alloc_slot` (no growth) | 0 | 0 | reuse hole or extend within existing cap |
| `alloc_slot` (growth) | 2 (live_bm realloc + table realloc) | 1 (old table) | doubling, amortized O(1) |
| `free_slot` | 0 | 0 | clears one bit; column reused later |
| `set` (ring growth) | 1 (table) | 1 (old table) | rare; row remap under wider mask |
| `destroy` | 0 | 2 (table, live_bm) | |

Caps:

- `ECS_INPUT_PLAYER_MAX = 1024` (concurrently registered slots)
- `ECS_INPUT_BUFSIZE_MAX = 2048` (~34 seconds @ 60 Hz)
- `ECS_INPUT_SLOT_NIL = 0xFFFFFFFF` (returned by `alloc_slot` on cap)

---

## 9. Concurrency, endianness, portability

- **Not thread-safe.** Caller serializes apply path or fronts the
  table with a queue/mutex.
- **Payload is opaque.** Engine `memcpy`s `stride` bytes verbatim;
  cross-platform peers must serialize endian-explicitly upstream
  (e.g. `ecs_serializer_write_bits`).
- **No wall-time logic.** Server-side max-lag policy (sealing tick
  T after a deadline, synthesizing zero inputs) lives outside this
  module; this module is symmetric across server and client.

---

## 10. File map

| File | Role |
|---|---|
| `input.h` | Public API, design comment, type defs, inline iterator helpers |
| `input.c` | Row layout helpers, slot bitmap, resize, lifecycle, hot path |

Internal naming convention: `in_*` for static helpers, `ecs_input_*`
for public API.
