#pragma once

#include <stdbool.h>
#include <stdint.h>

/* ==========================================================================
   ecs_input -- per-tick player input + command store.

   Design summary
   --------------
     pid             : 18-bit player id (matches entity_t.id width).
     dense idx       : engine-internal column index into the 2D table.
                       Stable across leaves -- never recycled until pid is
                       unregistered AND the slot is consumed from the free
                       list by a subsequent register.
     opaque payload  : caller defines stride at init; engine never inspects.

   pid -> dense idx  : linear scan of dense_ids[0..dense_high]. O(active)
                       per lookup. Cap is u16-bound (65535).

   Storage           : single 2D table, row-major. Each row = one tick
                       slot * active_cap * stride bytes (full roster for
                       that tick). Both buf_size and active_cap grow
                       pow2 on demand. Caller-supplied buf_size, must be
                       a power of two.

   Per-tick state    : flat row-major bitmaps of width words_per_row, indexed
                       by (tick & buf_mask):
                         confirmed_bm  -- bit set iff the (tick, idx) input
                                          arrived authoritative.
                         present_bm    -- bit set iff the (tick, idx) slot
                                          has been written at all (predicted
                                          or confirmed).
                       Plus per-tick scalars:
                         confirmed_count -- popcount of confirmed_bm row.
                         expected_count  -- server-stamped roster size for
                                            this tick. Set on first packet
                                            for the tick. Frontier advances
                                            when count == expected.
                         tick_in_slot    -- absolute tick id currently
                                            occupying ring slot t. Disambig
                                            wraparound; first-touch reset is
                                            triggered by mismatch.
                         all_confirmed_bm -- 1 bit per ring slot, set when
                                             count hits expected.
                         confirmed_frontier -- highest tick T such that all
                                               ticks <= T are fully confirmed.
                                               Greedily advanced.

   Operations (hot)  : set / get / get_view / clear -- all O(1) on the
                       per-tick scalars and O(active_cap / 64) memset on
                       clear's two bitmap rows.

   Server max-lag    : enforced by the SERVER side, not here. Server seals
                       tick T after a configurable wall deadline; missing
                       inputs synthesized as stride-zero bytes and broadcast
                       with confirmed=true. Caller treats the tick like any
                       other -- this module is server-side and client-side
                       symmetric and has no notion of wall time.

   Allocation philosophy
   ---------------------
     - Init: ~5 small bookkeeping allocations.
     - Tick (set/get/clear): zero.
     - Register: zero allocs (column already in table); table grows pow2
       only when high-water exceeds active_cap (rare, doubling).
     - Unregister: zero frees; column reused on next register.
     - active_cap grows pow2 (table realloc, no row re-layout).
     - buf_size grows pow2 (table re-laid-out per-section by mask remap).

   Threading
   ---------
   This module is NOT thread-safe. All ops mutate shared scalars (counts,
   frontier) and shared bitmaps without atomics. Callers serialize by
   either single-threading the apply path or fronting the table with a
   queue / mutex.

   Endianness
   ----------
   Slot bytes are opaque -- the engine memcpy's stride bytes verbatim
   on set / get. Cross-platform peers must serialize their input payload
   endian-explicitly (e.g. via the ecs_serializer write_bits API) before
   handing bytes to ecs_input_set; in-memory layout is not portable.

   Write semantics for predicted vs confirmed
   ------------------------------------------
   Once a (tick, pid) slot is confirmed, predicted writes are dropped --
   confirmed bytes are authoritative and a later prediction must NOT
   overwrite them. Confirmed writes always overwrite, regardless of
   prior state, and are idempotent (same bytes, same flags result).
   ========================================================================== */

/* 18-bit pid space (matches entity_t.id). */
#define ECS_INPUT_PID_BITS  18
#define ECS_INPUT_PID_MAX   (1u << ECS_INPUT_PID_BITS)
#define ECS_INPUT_PID_NIL   0xFFFFFFFFu

/* Hard cap on concurrently registered players. Player capacity grows
   pow2 on demand below this. */
#define ECS_INPUT_PLAYER_MAX 1024u

/* Hard cap on buf_size for auto-grow. Beyond this the caller must take
   the snapshot/resync path -- a tick lateness larger than this exceeds
   any reasonable rollback budget. 65536 ticks ~= 18 minutes @ 60fps. */
#define ECS_INPUT_BUFSIZE_MAX 65536u

/* Initial player capacity allocated on first register. Grows pow2. */
#define ECS_INPUT_PLAYER_CAP_INIT 16u

/* Default ring depth (tick capacity). Caller passes to ecs_input_init;
   grows pow2 on burst (see ecs_input_grow_buf). 32 ticks ~= 500ms @ 60fps. */
#define ECS_INPUT_BUFSIZE_INIT 32u

/* Sentinel for tick_in_slot[] meaning "ring slot empty / never written". */
#define ECS_INPUT_TICK_NIL  0xFFFFFFFFFFFFFFFFull

typedef uint32_t ecs_pid_t;

typedef struct ecs_input_t {
    /* Single packed table. One row per ring slot, layout:
         u64 tick_in_slot
         u64 present[words_per_row]
         u64 confirmed[words_per_row]
         u8  payload[active_cap * stride]
       row_bytes = 8 + 2*words_per_row*8 + active_cap*stride (rounded
       up to 8). Single contiguous allocation; per-slot ops touch one
       row = warm cache. Both dimensions grow pow2 on demand. */
    uint8_t*  table;
    uint32_t  row_bytes;     /* bytes per row, cached */

    /* dense_ids[i] = pid at column i, or ECS_INPUT_PID_NIL for free
       slots. Pid lookup AND free-slot search both SIMD-scan this same
       array (active_cap is pow2, multiple of 8). No separate free list. */
    uint32_t* dense_ids;
    uint32_t  active_count;  /* populated columns (does NOT include free holes) */
    uint32_t  active_cap;    /* table column count, pow2 */
    uint32_t  dense_high;    /* high-water of column indices ever assigned */
    uint32_t  words_per_row; /* ceil(active_cap / 64) */

    /* Frozen at init. */
    uint32_t  stride;        /* bytes per input slot */
    uint32_t  buf_size;      /* ring depth, pow2 */
    uint32_t  buf_mask;      /* buf_size - 1 */

    /* Confirmed frontier: highest tick T such that all live players are
       confirmed for every tick in [1..T]. 0 = no frontier yet (sim
       starts at tick 1; tick 0 is reserved as the "no frontier" sentinel
       and is invalid as an actual tick id). all_confirmed status per
       tick is computed on demand from dense_ids (live mask) AND
       confirmed bits in row -- no cached counts. */
    uint64_t  confirmed_frontier;
} ecs_input_t;

/* View returned by ecs_input_get_view -- caller can branch on flags
   without two separate table lookups. data is stride bytes, NULL if
   pid is unknown OR the ring slot does not currently hold this tick.
   When data is non-NULL the bytes belong to (tick, pid); use the flags
   to distinguish predicted-only vs confirmed. */
typedef struct ecs_input_view_t {
    const void* data;
    bool        present;     /* slot has been written for this tick */
    bool        confirmed;   /* slot is authoritative for this tick */
} ecs_input_view_t;

/* --- Lifecycle ----------------------------------------------------------- */

/* Initialize zero-filled `it` with given input stride and ring depth.
   buf_size must be a power of two and > 0. Caller-owned storage. */
void ecs_input_init(ecs_input_t* it, uint32_t stride, uint32_t buf_size);

/* Free all owned memory and reset to zeroed state. */
void ecs_input_destroy(ecs_input_t* it);

/* --- Membership ---------------------------------------------------------- */

/* Add a player. Returns true on success, false if pid already registered
   or the player cap (ECS_INPUT_PLAYER_MAX) is reached. Auto-grows
   active_cap (doubling) when the high-water exceeds the current
   capacity. Reuses a freed column when available; otherwise extends. */
bool ecs_input_register_player(ecs_input_t* it, ecs_pid_t pid);

/* Remove a player. Frees the player's slot ring and recycles the dense
   index onto the free list. The dense index is preserved in past tick
   bitmap rows -- with server-authoritative expected_count those rows
   were sealed before the leave took effect, so the stale bit is correct.
   No-op if pid unknown. */
void ecs_input_unregister_player(ecs_input_t* it, ecs_pid_t pid);

/* Number of currently-registered players. */
static inline uint32_t ecs_input_active_count(const ecs_input_t* it) {
    return it->active_count;
}

/* True iff `pid` is currently registered. */
bool ecs_input_is_registered(const ecs_input_t* it, ecs_pid_t pid);

/* --- Iteration ----------------------------------------------------------- */

/* Iterate live players in dense order. Holes (freed slots) skipped.
   Order is determined by register/unregister sequence -- callers that
   need cross-peer determinism must replay the same membership stream
   on every peer. */
typedef struct ecs_input_iter_t {
    const ecs_input_t* it;
    uint32_t           cursor;      /* next dense slot to inspect (private) */
    uint32_t           dense_idx;   /* dense_idx of the current live entry (set by next) */
    ecs_pid_t          pid;         /* live pid (set by next) */
} ecs_input_iter_t;

static inline ecs_input_iter_t ecs_input_iter_begin(const ecs_input_t* it) {
    ecs_input_iter_t r;
    r.it = it; r.cursor = 0u; r.dense_idx = 0u; r.pid = ECS_INPUT_PID_NIL;
    return r;
}

/* Advance iterator. Returns true and sets iter->pid / iter->dense_idx
   when a live entry was found, false when iteration is complete. */
bool ecs_input_iter_next(ecs_input_iter_t* iter);

/* --- Hot path: set / get / clear ----------------------------------------- */

/* Write `value` (stride bytes) into the slot for (tick, pid).
   If `confirmed` is true the confirmed bit is set; once every currently-
   registered player's confirmed bit is set for this tick, the frontier
   may advance.

   value MUST be non-NULL.

   Predicted-after-confirmed: if (tick, pid) is already confirmed and
   this call has confirmed=false, the call is a no-op (confirmed bytes
   are authoritative).

   No-op if pid unknown. */
void ecs_input_set(ecs_input_t* it, uint64_t tick, ecs_pid_t pid,
                   const void* value, bool confirmed);

/* Return slot pointer for (tick, pid) or NULL if pid unknown OR the
   slot has not been written for this tick (ring wrap / never set).
   When non-NULL the bytes were last written by ecs_input_set for THIS
   tick. Use ecs_input_get_view to additionally distinguish predicted
   vs confirmed. */
const void* ecs_input_get(const ecs_input_t* it, uint64_t tick, ecs_pid_t pid);

/* Combined lookup + presence/confirmed flags. */
ecs_input_view_t ecs_input_get_view(const ecs_input_t* it,
                                    uint64_t tick, ecs_pid_t pid);

/* Reset all per-player state for `tick`: zero confirmed/present rows,
   zero counts, mark slot empty (tick_in_slot = NIL). If `tick` is
   <= confirmed_frontier the frontier is rewound to tick - 1.
   This is admin/rollback API -- DO NOT call from sim path. */
void ecs_input_clear(ecs_input_t* it, uint64_t tick);

/* --- Frontier / status --------------------------------------------------- */

/* True iff every registered player's input for `tick` is confirmed
   (count == expected). */
bool ecs_input_tick_confirmed(const ecs_input_t* it, uint64_t tick);

/* Highest contiguous fully-confirmed tick. Returns 0 if no tick has
   ever fully confirmed yet (tick 0 is reserved as the "no frontier"
   sentinel; valid ticks start at 1). */
uint64_t ecs_input_frontier(const ecs_input_t* it);

/* Seed frontier at `tick` (mid-session join / snapshot bootstrap).
   `tick` must be >= 1. Subsequent confirmations of tick+1, tick+2, ...
   will extend it. */
void ecs_input_seed_frontier(ecs_input_t* it, uint64_t tick);

/* Mark `tick` as fully confirmed with zero expected players. Used by
   the server to seal ticks where the roster was empty -- the frontier
   advances past such ticks without any set() calls. */
void ecs_input_seal_empty_tick(ecs_input_t* it, uint64_t tick);

/* --- Auto-grow ----------------------------------------------------------- */

/* Grow the ring depth (table rows) to at least `new_buf_size` (rounded up
   to next pow2). Re-maps every live row to its new index under the wider
   mask. ecs_input_set auto-invokes this when a write would otherwise
   clobber a still-live row (predicted-but-not-yet-frontier-passed). */
void ecs_input_grow_buf(ecs_input_t* it, uint32_t new_buf_size);

/* Grow the player capacity (table cols) to at least `new_player_cap`
   (rounded up to next pow2). Both dimensions of the 2D table grow pow2
   doubling. ecs_input_register_player auto-invokes this on
   high-water-mark allocation; callers may grow preemptively before a
   known roster expansion. No-op if new_player_cap <= current. */
void ecs_input_grow_player_cap(ecs_input_t* it, uint32_t new_player_cap);
