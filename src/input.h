#pragma once

#include <stdbool.h>
#include <stdint.h>

/* ==========================================================================
   ecs_input -- per-tick player input + command store.

   Design summary
   --------------
     pid             : 18-bit player id (matches entity_t.id width).
     dense idx       : engine-internal stable index into dense arrays.
                       Stable across leaves -- never recycled until pid is
                       unregistered AND the slot is consumed from the free
                       list by a subsequent register.
     opaque payload  : caller defines stride at init; engine never inspects.

   pid -> dense idx  : 9+9 radix table. Top L1 = 512 page pointers (4 KB,
                       always present). Each populated L2 page = 512 * u32 +
                       small header. Lookup = 2 loads, branch on NULL page.
                       At-most ~1 MB if every pid populated; typical few KB.

   Per-player ring   : `slots` = buf_size * stride bytes, indexed by
                       (tick & buf_mask). Caller-supplied buf_size, must be
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
     - Init: ~5 allocations (table fields).
     - Tick (set/get/clear): zero.
     - Register: 1 buf-slots block + at most 1 pid_page (first pid in hi bucket).
     - Unregister: symmetric frees. Caller defers if a rollback window must
       still observe the player's slots.
     - active_cap grows pow2; bitmap rows migrated row-by-row on grow.
   ========================================================================== */

/* 18-bit pid space (matches entity_t.id). */
#define ECS_INPUT_PID_BITS  18
#define ECS_INPUT_PID_MAX   (1u << ECS_INPUT_PID_BITS)
#define ECS_INPUT_PID_NIL   0xFFFFFFFFu

/* Radix table: 9 high bits index L1, 9 low bits index L2 page entry. */
#define ECS_INPUT_L1_BITS   9
#define ECS_INPUT_L2_BITS   9
#define ECS_INPUT_L1_SIZE   (1u << ECS_INPUT_L1_BITS)
#define ECS_INPUT_L2_SIZE   (1u << ECS_INPUT_L2_BITS)

/* Sentinel for free list terminator in free_next[]. */
#define ECS_INPUT_FREE_NIL  0xFFFFFFFFu

/* Sentinel for tick_in_slot[] meaning "ring slot empty / never written". */
#define ECS_INPUT_TICK_NIL  0xFFFFFFFFFFFFFFFFull

typedef uint32_t ecs_pid_t;

typedef struct ecs_input_buf_t {
    /* buf_size * stride bytes. Indexed by (tick & buf_mask) * stride. */
    uint8_t* slots;
} ecs_input_buf_t;

typedef struct ecs_input_pid_page_t {
    /* pid_lo -> dense idx (or ECS_INPUT_PID_NIL). */
    uint32_t entry[ECS_INPUT_L2_SIZE];
    /* Live entries in this page; page freed when 0. */
    uint16_t used;
} ecs_input_pid_page_t;

typedef struct ecs_input_t {
    /* pid -> dense idx (radix). */
    ecs_input_pid_page_t* l1[ECS_INPUT_L1_SIZE];

    /* Dense storage. Stable idx with free-list reuse. */
    ecs_input_buf_t* dense;
    uint32_t*        dense_ids;     /* pid for live, ECS_INPUT_PID_NIL for free */
    uint32_t*        free_next;     /* free-list links (ECS_INPUT_FREE_NIL terminator) */
    uint32_t         free_head;
    uint32_t         active_count;  /* populated dense slots (does NOT include free holes) */
    uint32_t         active_cap;    /* capacity of dense / dense_ids / free_next; pow2 */
    uint32_t         dense_high;    /* high-water of indices ever assigned (<= active_cap) */

    /* Frozen at init. */
    uint32_t stride;     /* bytes per input slot */
    uint32_t buf_size;   /* ring depth, pow2 */
    uint32_t buf_mask;   /* buf_size - 1 */

    /* Per-tick row-major bitmaps. words_per_row = ceil(active_cap / 64). */
    uint64_t* confirmed_bm;     /* [buf_size * words_per_row] */
    uint64_t* present_bm;       /* [buf_size * words_per_row] */
    uint16_t* confirmed_count;  /* [buf_size] */
    uint16_t* expected_count;   /* [buf_size] */
    uint64_t* tick_in_slot;     /* [buf_size]; absolute tick id or ECS_INPUT_TICK_NIL */
    uint64_t* all_confirmed_bm; /* [ceil(buf_size / 64)] */
    uint64_t  confirmed_frontier;
    bool      frontier_valid;   /* false = no tick has ever advanced; frontier value undefined */

    uint32_t  words_per_row;
} ecs_input_t;

/* View returned by ecs_input_get_view -- caller can branch on flags
   without two separate table lookups. data is stride bytes; only valid
   while ecs_input_t is unchanged. */
typedef struct ecs_input_view_t {
    const void* data;        /* slot bytes (stride-sized) or NULL if pid unknown */
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

/* Add a player. Returns true on success, false if pid already registered.
   Allocates the player's slot ring (buf_size * stride bytes) and a pid_page
   if the high half of the pid was previously empty. */
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

/* --- Hot path: set / get / clear ----------------------------------------- */

/* Write `value` (stride bytes) into the slot for (tick, pid).
   `expected` is the server-stamped roster size for this tick; on first
   write to ring slot (tick & buf_mask) it is recorded as the threshold
   for frontier advance. If `confirmed` is true the confirmed bit is set
   and confirmed_count incremented (idempotent if already set).
   No-op if pid unknown. */
void ecs_input_set(ecs_input_t* it, uint64_t tick, ecs_pid_t pid,
                   const void* value, bool confirmed, uint16_t expected);

/* Return slot pointer for (tick, pid) or NULL if pid unknown.
   Bytes are valid only if the slot was previously set (caller's
   responsibility -- use ecs_input_get_view if presence matters). */
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

/* Highest contiguous fully-confirmed tick. Returns 0 with *out_valid
   = false if no tick has ever fully confirmed yet. */
uint64_t ecs_input_frontier(const ecs_input_t* it, bool* out_valid);

/* Seed frontier at `tick` (mid-session join / snapshot bootstrap).
   Subsequent confirmations of tick+1, tick+2, ... will extend it.
   Use after restoring world state from a server snapshot. */
void ecs_input_seed_frontier(ecs_input_t* it, uint64_t tick);
