# `ecs_input` — Architecture & Layout

Per-tick player input store for deterministic netcode. Server- and
client-symmetric. Holds opaque, fixed-stride input bytes for every
`(tick, pid)` slot in a wrap-around ring, with predicted/confirmed
flags and a sim-driven frontier.

---

## 1. Quality assessment

**Verdict: production-grade for its design envelope (≤1024 players,
≤2048 buffered ticks).**

What's good:

| Aspect | Notes |
|---|---|
| **Documentation** | Module-level header comment is exhaustive: semantics, threading, endianness, write rules, allocation philosophy. Public API is fully commented. |
| **Layout** | Single contiguous 2D table. Per-tick op = one row touched = warm cache. Power-of-two dimensions everywhere → mask-and-shift addressing. |
| **Hot path** | Zero allocations on `set` / `get` / `get_view` / `clear`. SIMD-scan pid lookup via `fixed_8_t` (8-wide). |
| **Bitmaps** | `present` and `confirmed` are flat `u64` arrays inside the row — both fit a few cache lines for typical player counts. |
| **`all_confirmed`** | Computed on demand by ANDing the row's confirmed bits with a SIMD-derived live mask. No cached counters to drift out of sync. |
| **Auto-grow** | Both dimensions double on demand and re-lay out under the new mask. Caller never has to size for worst case. |
| **Correctness details** | ABA defense on `register_player` (scrubs stale bits in still-live rows). First-touch carry-forward (`in_advance_row`) implements input persistence. Predicted-after-confirmed writes are dropped, not silently overwritten. Tick `0` is reserved as a "no frontier" sentinel. |
| **Memory** | ~5 small allocs at init, zero per tick, columns recycled via dense-id holes (no free list overhead). |

Intentional non-features:

- **Not thread-safe.** By design — caller serializes. Atomics on every
  bitmap word would dominate the hot path with no benefit for the
  single-threaded apply model this module targets.
- **Pid lookup is a SIMD scan, not a reverse map.** `O(active_cap / 8)`
  vector ops per call is correct for the design envelope; a sparse
  `pid → idx` array would cost 512 KiB to save tens of nanoseconds.

The module is internally consistent, well factored, and faithful to
its stated design.

---

## 2. Mental model

```
                      ┌──────────────────────┐
   tick & buf_mask ──►│ row[t]               │  one ring slot
                      ├──────────────────────┤
                      │ tick_in_slot  (u64)  │  absolute tick id here
                      │ present[W]    (u64)  │  bit per dense column
                      │ confirmed[W]  (u64)  │  bit per dense column
                      │ payload[A*S]  (u8)   │  stride bytes per column
                      └──────────────────────┘
                                ▲
                       pid → SIMD scan dense_ids[] → idx (column)
```

- `pid` is the caller's 18-bit player id (matches `entity_t.id`).
- `idx` is the engine-internal **dense column index**: the position
  in `dense_ids[]` and the column in every row.
- `tick` is an absolute 64-bit id; the ring position is
  `tick & buf_mask`.
- `tick_in_slot[t]` disambiguates ring wrap: when a write for tick T
  finds a slot holding T' ≠ T, that's a first-touch and the row is
  reset (with carry-forward of predicted bytes).

---

## 3. Storage layout

Single allocation: `it->table`, `buf_size` rows × `row_bytes` each,
contiguous. Layout per row:

```
offset 0           u64 tick_in_slot
offset 8           u64 present   [words_per_row]
offset 8 + W*8     u64 confirmed [words_per_row]
offset 8 + 2*W*8   u8  payload   [active_cap * stride]
```

Where:

- `W = words_per_row = ceil(active_cap / 64)`
- `A = active_cap` (column count, pow2)
- `S = stride` (bytes per input slot, frozen at init)
- `row_bytes = pad8(8 + 2*W*8 + A*S)`

Both dimensions grow pow2:

- **`active_cap` grow** → row width changes. `in_realloc_table`
  allocates a new table, walks every old row, re-lays bitmaps and
  payload into the wider row, leaving new columns uninitialized
  (presence bits gate reads).
- **`buf_size` grow** → row count changes. Same `in_realloc_table`
  re-maps every live row to its new slot under the wider mask.

Auto-grow triggers:

- `set` detects a first-touch that would clobber a still-live row
  (`prev_tick > confirmed_frontier`) and doubles `buf_size` until
  capacity ≥ `tick - prev + 1` (capped at `ECS_INPUT_BUFSIZE_MAX`).
- `register_player` detects `dense_high >= active_cap` and doubles
  `active_cap`.

---

## 4. Pid → column resolution

`dense_ids[]` is a 32-byte-aligned `uint32_t` array of length
`active_cap` (pow2, ≥ 8). Each entry is either a registered pid or
`ECS_INPUT_PID_NIL = 0xFFFFFFFF`.

Both pid lookup and free-slot search are the same SIMD scan
(`in_scan_dense_ids`) using `fixed_8_t` (8-lane equality):

```c
for (uint32_t b = 0; b < blocks; b++) {
    uint8_t mask = fixed8_eq(ids[b], target);
    if (mask) return (b << 3) + ecs_ctz32(mask);
}
```

Consequences:

- No separate free list. Holes are `NIL` entries; the next register
  finds the lowest hole via the same scan with `target = NIL`.
- `dense_high` is a high-water mark of indices ever assigned. Only
  rows `[0, dense_high)` need to be scanned for live work
  (`in_advance_row`, iterator).
- `active_count` excludes holes; `dense_high − active_count` =
  recyclable holes available without growth.

---

## 5. Bitmaps and `all_confirmed`

`present[w]` bit `b` is set iff column `w*64+b` was written for the
current `tick_in_slot[t]`. `confirmed[w]` is the analogous flag for
authoritative writes.

`in_tick_all_confirmed` doesn't store a "all confirmed" cache. It
derives the **live mask** per word on demand:

```c
in_live_mask_word(it, w):
    for each 8-lane block in word w:
        mask = fixed8_eq(ids[block], NIL)   // bit set where slot is hole
        live |= ((~mask) & 0xFF) << offset   // invert: bit set where live
    return live
```

Then `confirmed[w] & live == live` for every word ⇒ all confirmed.
This is robust against stale bits left over from past registrations
because `live` is computed from `dense_ids[]` right now.

ABA defense: when `register_player` reuses a hole at `idx`, it walks
every still-live row (`tick_in_slot != NIL`) and clears bit `idx` in
both `present` and `confirmed`. Without this, a re-registered pid
would inherit the stale bits of its predecessor.

---

## 6. Hot path semantics

### `ecs_input_set(tick, pid, value, confirmed)`

1. `pid → idx` via SIMD scan; bail if unknown.
2. Check first-touch eviction — if the target slot holds a
   still-live tick (`> frontier`), grow `buf_size`.
3. `t = tick & buf_mask`; if `tick_in_slot[t] != tick`, this is a
   first touch:
   - `in_advance_row` zeroes both bitmaps and, if the previous tick's
     row is still resident, carries forward each live player's bytes
     (predicted) and sets their present bits.
   - `tick_in_slot[t] = tick`.
4. Write semantics:
   - `confirmed=false` + already confirmed → no-op (drop predicted).
   - Otherwise `memcpy` the value, set `present` bit.
   - `confirmed=true` additionally sets `confirmed` bit.

### `ecs_input_get(tick, pid)`

1. `pid → idx`.
2. `t = tick & buf_mask`; require `tick_in_slot[t] == tick`.
3. Require `present[idx>>6] & (1 << idx&63)`.
4. Return `payload + idx * stride`.

### `ecs_input_get_view(tick, pid)`

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
| `init` | 1 (table) + dense_ids deferred to first register | 0 | row_bytes = 8 (header only) until first register |
| `set` / `get` / `get_view` / `clear` | 0 | 0 | hot path is pure pointer math + SIMD |
| `register_player` (no growth) | 0 | 0 | reuse hole or extend `dense_high` |
| `register_player` (growth) | 2 (dense_ids realloc + table realloc) | 1 (old table) | doubling, amortized O(1) |
| `unregister_player` | 0 | 0 | hole left in dense_ids, reused later |
| `set` (ring growth) | 1 (table) | 1 (old table) | rare; row remap under wider mask |
| `destroy` | 0 | 2 (table, dense_ids) | |

Caps:

- `ECS_INPUT_PID_MAX = 1 << 18`
- `ECS_INPUT_PLAYER_MAX = 1024` (concurrently registered)
- `ECS_INPUT_BUFSIZE_MAX = 65536` (~18 minutes @ 60 Hz)

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
| `input.c` | Row layout helpers, SIMD scans, resize, lifecycle, hot path |

Internal naming convention: `in_*` for static helpers, `ecs_input_*`
for public API.
