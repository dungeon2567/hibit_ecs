#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ecs_serializer.h"

/* ==========================================================================
   ecs_input -- per-tick player input + command store.

   Design summary
   --------------
     slot            : engine-internal column index into the 2D table.
                       Allocated by ecs_input_alloc_slot, returned to the
                       caller. Stable across leaves -- never recycled until
                       freed by ecs_input_free_slot AND a subsequent alloc
                       picks the now-empty position.
     opaque payload  : caller defines stride at init; engine never inspects.

   Slot identity     : caller is responsible for any pid -> slot mapping
                       outside this module. The engine only knows slots.

   Storage           : single 2D table, row-major. Each row = one tick
                       slot * active_cap * stride bytes (full roster for
                       that tick). Both buf_size and active_cap grow
                       pow2 on demand. Caller-supplied buf_size, must be
                       a power of two.

   Per-tick state    : flat row-major bitmaps of width words_per_row, indexed
                       by (tick & buf_mask):
                         confirmed_bm  -- bit set iff the (tick, slot) input
                                          arrived authoritative.
                         present_bm    -- bit set iff the (tick, slot) slot
                                          has been written at all (predicted
                                          or confirmed).
                       Plus per-tick scalar:
                         tick_in_slot    -- absolute tick id currently
                                            occupying ring slot t. Disambig
                                            wraparound; first-touch reset is
                                            triggered by mismatch.

   Live mask         : live_bm is a single bitmap of width words_per_row.
                       Bit `idx` set <=> slot idx is currently registered.
                       Membership ops are O(1) on a single u64 word; alloc
                       finds the first 0 bit via ctz. No SIMD search.

   Frontier          : confirmed_frontier is sim-driven via
                       ecs_input_advance_to_tick. Input does NOT auto-track
                       it. Ring slots holding ticks <= frontier are eligible
                       for eviction on first-touch by a write to a newer
                       tick. Querying tick_confirmed for an arbitrary tick
                       still works (computed on demand).

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
     - Init: ~2 small bookkeeping allocations.
     - Tick (set/get/clear): zero.
     - Alloc slot: zero allocs (column already in table); table grows pow2
       only when high-water exceeds active_cap (rare, doubling).
     - Free slot: zero frees; column reused on next alloc.
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
   Once a (tick, slot) cell is confirmed, predicted writes are dropped --
   confirmed bytes are authoritative and a later prediction must NOT
   overwrite them. Confirmed writes always overwrite, regardless of
   prior state, and are idempotent (same bytes, same flags result).
   ========================================================================== */

/* Hard cap on concurrently registered slots. Player capacity grows
   pow2 on demand below this. */
#define ECS_INPUT_PLAYER_MAX 1024u

/* Hard cap on buf_size for auto-grow. Beyond this the caller must take
   the snapshot/resync path -- a tick lateness larger than this exceeds
   any reasonable rollback budget. 2048 ticks ~= 34 seconds @ 60fps. */
#define ECS_INPUT_BUFSIZE_MAX 2048u

/* Initial player capacity allocated on first alloc_slot. Grows pow2. */
#define ECS_INPUT_PLAYER_CAP_INIT 16u

/* Default ring depth (tick capacity). Caller passes to ecs_input_init;
   grows pow2 on burst (see ecs_input_grow_buf). 32 ticks ~= 500ms @ 60fps. */
#define ECS_INPUT_BUFSIZE_INIT 32u

/* Sentinel for tick_in_slot[] meaning "ring slot empty / never written". */
#define ECS_INPUT_TICK_NIL  0xFFFFFFFFu

/* Sentinel for "slot allocation failed / unknown slot". */
#define ECS_INPUT_SLOT_NIL  0xFFFFFFFFu

typedef struct ecs_input_t {
    /* Single packed table. One row per ring slot, layout:
         u32 tick_in_slot   (+ 4 bytes pad to keep bitmaps 8-byte aligned)
         u64 present[words_per_row]
         u64 confirmed[words_per_row]
         u8  payload[active_cap * stride]
       row_bytes = 8 + 2*words_per_row*8 + active_cap*stride (rounded
       up to 8). Single contiguous allocation; per-slot ops touch one
       row = warm cache. Both dimensions grow pow2 on demand. */
    uint8_t*  table;
    uint32_t  row_bytes;     /* bytes per row, cached */

    /* live_bm[w] bit b set <=> slot (w*64+b) is currently registered.
       Length = words_per_row u64 words. alloc_slot finds the first 0
       bit via word-wise ctz; free_slot clears the bit. */
    uint64_t* live_bm;
    uint32_t  active_count;  /* popcount(live_bm) */
    uint32_t  active_cap;    /* table column count, pow2 */
    uint32_t  words_per_row; /* ceil(active_cap / 64) */

    /* Frozen at init. */
    uint32_t  stride;        /* bytes per input slot */
    uint32_t  buf_size;      /* ring depth, pow2 */
    uint32_t  buf_mask;      /* buf_size - 1 */

    /* Confirmed frontier: sim-driven low-water mark. Set via
       ecs_input_advance_to_tick. Ticks <= confirmed_frontier are
       considered discarded -- their ring slots may be evicted on
       first-touch by a write to a newer tick. 0 = no frontier yet
       (sim starts at tick 1; tick 0 is reserved as the "no frontier"
       sentinel and is invalid as an actual tick id). */
    uint32_t  confirmed_frontier;
} ecs_input_t;

/* View returned by ecs_input_get_view -- caller can branch on flags
   without two separate table lookups. data is stride bytes, NULL if
   slot is unknown OR the ring slot does not currently hold this tick.
   When data is non-NULL the bytes belong to (tick, slot); use the flags
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

/* Allocate the lowest free slot. Returns the slot index, or
   ECS_INPUT_SLOT_NIL if ECS_INPUT_PLAYER_MAX would be exceeded.
   Auto-grows active_cap (pow2 doubling) when full. The recycled-slot
   case scrubs stale present/confirmed bits at this index in every
   still-live ring row (ABA defense). */
uint32_t ecs_input_alloc_slot(ecs_input_t* it);

/* Free a slot. No-op if slot is out of range or already free.
   Past tick rows still carry stale bits at this column; subsequent
   reads gate via the live mask, so they are inert. A re-alloc that
   picks the same slot will scrub them. */
void ecs_input_free_slot(ecs_input_t* it, uint32_t slot);

/* Number of currently-registered slots. */
static inline uint32_t ecs_input_active_count(const ecs_input_t* it) {
    return it->active_count;
}

/* True iff `slot` is currently allocated. */
bool ecs_input_slot_is_live(const ecs_input_t* it, uint32_t slot);

/* --- Iteration ----------------------------------------------------------- */

/* Iterate live slots in ascending order. Holes (freed slots) skipped.
   Order is determined by alloc_slot/free_slot sequence (lowest-free-
   first reuse) -- callers that need cross-peer determinism must replay
   the same membership stream on every peer. */
typedef struct ecs_input_iter_t {
    const ecs_input_t* it;
    uint32_t           cursor;      /* next slot to inspect (private) */
    uint32_t           slot;        /* current live slot (set by next) */
} ecs_input_iter_t;

static inline ecs_input_iter_t ecs_input_iter_begin(const ecs_input_t* it) {
    ecs_input_iter_t r;
    r.it = it; r.cursor = 0u; r.slot = ECS_INPUT_SLOT_NIL;
    return r;
}

/* Advance iterator. Returns true and sets iter->slot when a live slot
   was found, false when iteration is complete. */
bool ecs_input_iter_next(ecs_input_iter_t* iter);

/* --- Hot path: set / get / clear ----------------------------------------- */

/* Write `value` (stride bytes) into the cell for (tick, slot).
   If `confirmed` is true the confirmed bit is set. Frontier is NOT
   touched here -- the simulation drives it via
   ecs_input_advance_to_tick.

   value MUST be non-NULL.

   Predicted-after-confirmed: if (tick, slot) is already confirmed and
   this call has confirmed=false, the call is a no-op (confirmed bytes
   are authoritative).

   No-op if slot is out of range or not live. */
void ecs_input_set(ecs_input_t* it, uint32_t tick, uint32_t slot,
                   const void* value, bool confirmed);

/* Return cell pointer for (tick, slot) or NULL if slot is not live OR
   the cell has not been written for this tick (ring wrap / never set).
   When non-NULL the bytes were last written by ecs_input_set for THIS
   tick. Use ecs_input_get_view to additionally distinguish predicted
   vs confirmed. */
const void* ecs_input_get(const ecs_input_t* it, uint32_t tick, uint32_t slot);

/* Combined lookup + presence/confirmed flags. */
ecs_input_view_t ecs_input_get_view(const ecs_input_t* it,
                                    uint32_t tick, uint32_t slot);

/* Reset all per-slot state for `tick`: zero confirmed/present rows,
   mark slot empty (tick_in_slot = NIL). Does NOT touch frontier --
   caller is responsible for any frontier rewind via advance_to_tick
   if their semantics require it.
   Admin/rollback API -- DO NOT call from sim path. */
void ecs_input_clear(ecs_input_t* it, uint32_t tick);

/* --- Frontier / status --------------------------------------------------- */

/* True iff every currently-registered slot has a confirmed bit set
   for `tick`. Computed on demand from the row's confirmed bitmap and
   live_bm -- no cached counts. Returns true when active_count == 0
   (vacuous). Also returns true for any tick in [1, confirmed_frontier]:
   sim has sealed those, so they are authoritative regardless of
   whether their ring slot is still resident. */
bool ecs_input_tick_confirmed(const ecs_input_t* it, uint32_t tick);

/* Current sim-driven frontier. Returns 0 if no tick has ever been
   advanced past yet (tick 0 is reserved as the "no frontier" sentinel;
   valid ticks start at 1). */
uint32_t ecs_input_frontier(const ecs_input_t* it);

/* Set frontier to `tick`. Sim is responsible for invoking this after
   it has decided that everything <= `tick` is finalized; ring slots
   for those ticks are then eligible for eviction on first-touch by a
   newer write. `tick` must be >= 1 and >= current frontier (monotonic).
   Does NOT scrub ring contents -- queries on stale ticks still return
   data until the slot is overwritten. */
void ecs_input_advance_to_tick(ecs_input_t* it, uint32_t tick);

/* --- Auto-grow ----------------------------------------------------------- */

/* Grow the ring depth (table rows) to at least `new_buf_size` (rounded up
   to next pow2). Re-maps every live row to its new index under the wider
   mask. ecs_input_set auto-invokes this when a write would otherwise
   clobber a still-live row (predicted-but-not-yet-frontier-passed). */
void ecs_input_grow_buf(ecs_input_t* it, uint32_t new_buf_size);

/* Grow the player capacity (table cols) to at least `new_player_cap`
   (rounded up to next pow2). Both dimensions of the 2D table grow pow2
   doubling. ecs_input_alloc_slot auto-invokes this on
   high-water-mark allocation; callers may grow preemptively before a
   known roster expansion. No-op if new_player_cap <= current. */
void ecs_input_grow_player_cap(ecs_input_t* it, uint32_t new_player_cap);

/* --- Serialization ------------------------------------------------------ */

/* Serialize tick `tick` into `s` using cascading delta compression
   against up to `redundancy` past ticks (T-1, T-2, ..., T-redundancy)
   already held in the ring.

   Wire layout
   -----------
     u32 tick                                (header)
     u8  redundancy                          (header)
     for each slot idx in [0, active_cap):
       1 bit: differs from baseline (all zeros)?
         0 -> slot value IS zeros, done.
         1 -> for r in 1..redundancy:
                1 bit: differs from tick T-r?
                  0 -> slot value EQUALS tick T-r, done.
                  1 -> continue to next r.
              All redundancy+1 bits = 1 -> raw stride bytes (bit-packed).

   Past tick rows that are not currently resident in the ring are
   treated as "all zeros" by the encoder; the decoder uses the same
   rule, so caller need not coordinate visibility of past ticks beyond
   maintaining a peer ring of the same size.

   No slot id is emitted -- the wire is keyed by dense slot.
   Caller must guarantee both peers share active_cap and stride at
   decode time. Slots that are not live (live_bm bit clear) are
   serialized as if their value were zeros (same single bit as baseline
   match), so the wire size is independent of which columns are live.

   Performance notes
   -----------------
   - O(active_cap * redundancy) memcmps. For typical strides (4-16 B)
     each memcmp lowers to one or two integer compares.
   - One small stack scratch (256 B max) for the zero baseline; stride
     must fit. The redundancy refs table is `redundancy` pointer pairs
     on stack.

   Precondition: tick >= 1.
   The serializer must have room for the worst-case payload:
     header + cap * (redundancy + 1) bits + cap * stride * 8 bits.
*/
void ecs_input_serialize_tick(const ecs_input_t* it,
                              ecs_serializer_t* s,
                              uint32_t tick,
                              uint32_t redundancy);

/* Inverse of ecs_input_serialize_tick. Reads the wire format from `d`
   and applies every slot to `it` as a CONFIRMED write at the decoded
   tick. Slots whose live bit is clear on the receiver are skipped on
   apply -- their wire bits are still consumed.

   Past-tick references for the cascade are resolved against the
   receiver's own ring; any reference not currently resident is treated
   as zeros (same rule as the encoder).

   Caller must guarantee active_cap and stride match the sender at the
   moment of encode. */
void ecs_input_deserialize_tick(ecs_input_t* it, ecs_deserializer_t* d);
