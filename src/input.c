#include "input.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "ecs.h"   /* ecs_xmalloc_aligned, ecs_xcalloc, ecs_xrealloc, ecs_free, ecs_ctz64 */

/* ==========================================================================
   Row layout (packed, single allocation in it->table)

     offset 0                : u32 tick_in_slot  (+4 bytes pad)
     offset 8                : u64 present[W]
     offset 8+W*8            : u64 confirmed[W]
     offset 8+2*W*8          : u32 cmd_head[A]   (offset+1 into cmd_arenas[t].buf, 0=empty)
     offset 8+2*W*8+A*4      : u32 cmd_tail[A]   (offset+1 of FIFO tail node, 0=empty)
     offset 8+2*W*8+A*8      : u8  payload[A * stride]

   row_bytes = padded-up-to-8(8 + 2*W*8 + 2*A*4 + A*S).

   Per-slot ops (set/get/get_view, advance_row, reset_row, all_confirmed
   check) touch a single row = warm cache.

   "all_confirmed" for a tick is computed on demand by AND-checking
   the row's confirmed bits against live_bm (W u64 ops). No cached
   counts / bitmap.

   Slot membership: live_bm is a single bitmap of width W. Bit `idx`
   set <=> slot idx is registered. alloc_slot finds the lowest 0 bit
   via word-wise ctz; free_slot clears the bit. Callers pass slots
   directly -- the engine has no notion of external ids.
   ========================================================================== */

#define IN_HDR_BYTES 8u   /* tick_in_slot prefix size */

static inline uint32_t in_words_for_cap(uint32_t cap) {
    return (cap + 63u) >> 6;
}

static inline uint32_t in_compute_row_bytes(uint32_t W, uint32_t A, uint32_t S) {
    uint32_t b = IN_HDR_BYTES + 2u * W * 8u + 2u * A * 4u + A * S;
    return (b + 7u) & ~7u;
}

static inline uint8_t* in_row_ptr(const ecs_input_t* it, uint32_t slot) {
    return it->table + (size_t)slot * (size_t)it->row_bytes;
}
static inline uint32_t* in_row_tick(uint8_t* row) {
    return (uint32_t*)row;
}
static inline uint64_t* in_row_present(uint8_t* row) {
    return (uint64_t*)(row + IN_HDR_BYTES);
}
static inline uint64_t* in_row_confirmed(uint8_t* row, uint32_t W) {
    return (uint64_t*)(row + IN_HDR_BYTES + (size_t)W * 8u);
}
static inline uint32_t* in_row_cmd_head(uint8_t* row, uint32_t W) {
    return (uint32_t*)(row + IN_HDR_BYTES + (size_t)W * 16u);
}
static inline uint32_t* in_row_cmd_tail(uint8_t* row, uint32_t W, uint32_t A) {
    return (uint32_t*)(row + IN_HDR_BYTES + (size_t)W * 16u + (size_t)A * 4u);
}
static inline uint8_t* in_row_payload(uint8_t* row, uint32_t W, uint32_t A) {
    return row + IN_HDR_BYTES + (size_t)W * 16u + (size_t)A * 8u;
}
static inline uint8_t* in_row_cell(uint8_t* row, uint32_t W, uint32_t A,
                                   uint32_t idx, uint32_t stride) {
    return in_row_payload(row, W, A) + (size_t)idx * (size_t)stride;
}

static inline bool in_slot_live(const ecs_input_t* it, uint32_t slot) {
    if (slot >= it->active_cap) return false;
    return ((it->live_bm[slot >> 6] >> (slot & 63u)) & 1ull) != 0ull;
}

/* Find the lowest free slot via word-wise ctz on inverted live_bm.
   Returns ECS_INPUT_SLOT_NIL when every slot in [0, active_cap) is
   live. */
static uint32_t in_find_free_slot(const ecs_input_t* it) {
    uint32_t W = it->words_per_row;
    uint32_t cap = it->active_cap;
    for (uint32_t w = 0; w < W; w++) {
        uint64_t inv = ~it->live_bm[w];
        if (inv) {
            uint32_t bit = (uint32_t)ecs_ctz64(inv);
            uint32_t slot = (w << 6) + bit;
            if (slot < cap) return slot;
        }
    }
    return ECS_INPUT_SLOT_NIL;
}

/* True iff every currently-registered slot has its confirmed bit set
   at ring slot `t`. Computed on demand. O(W) word ops. */
static bool in_tick_all_confirmed(const ecs_input_t* it, uint32_t t) {
    if (it->active_count == 0u) return true;
    uint8_t* row = in_row_ptr(it, t);
    uint64_t* confirmed = in_row_confirmed(row, it->words_per_row);
    uint32_t W = it->words_per_row;
    for (uint32_t w = 0; w < W; w++) {
        uint64_t live = it->live_bm[w];
        if ((confirmed[w] & live) != live) return false;
    }
    return true;
}

/* Reset row: zero present + confirmed bits, zero cmd_head/tail, reset
   per-row cmd arena (len=0, buf retained), mark slot empty. Used by
   ecs_input_clear (no carry-forward). */
static void in_reset_row(ecs_input_t* it, uint32_t t) {
    uint8_t* row = in_row_ptr(it, t);
    uint32_t W = it->words_per_row;
    uint32_t A = it->active_cap;
    if (W) {
        memset(in_row_present(row),   0, (size_t)W * sizeof(uint64_t));
        memset(in_row_confirmed(row, W), 0, (size_t)W * sizeof(uint64_t));
    }
    if (A) {
        memset(in_row_cmd_head(row, W),    0, (size_t)A * sizeof(uint32_t));
        memset(in_row_cmd_tail(row, W, A), 0, (size_t)A * sizeof(uint32_t));
    }
    if (it->cmd_arenas) it->cmd_arenas[t].len = 0u;
    *in_row_tick(row) = ECS_INPUT_TICK_NIL;
}

/* First-touch advance: carry every live slot's input bytes forward
   from prev tick into the new ring slot, mark them present (predicted),
   clear confirmed. Implements input persistence.

   Commands are NEVER carried forward -- they are authoritative-only
   per-tick and reset on row eviction (cmd_head/tail zeroed,
   cmd_arenas[t].len = 0). */
static void in_advance_row(ecs_input_t* it, uint32_t t, uint32_t tick) {
    uint8_t* row = in_row_ptr(it, t);
    uint32_t W = it->words_per_row;
    uint32_t A = it->active_cap;

    if (W) {
        memset(in_row_present(row),    0, (size_t)W * sizeof(uint64_t));
        memset(in_row_confirmed(row, W), 0, (size_t)W * sizeof(uint64_t));
    }
    if (A) {
        memset(in_row_cmd_head(row, W),    0, (size_t)A * sizeof(uint32_t));
        memset(in_row_cmd_tail(row, W, A), 0, (size_t)A * sizeof(uint32_t));
    }
    if (it->cmd_arenas) it->cmd_arenas[t].len = 0u;

    bool has_prev = false;
    uint32_t prev_t = 0;
    if (tick > 0u) {
        prev_t = (tick - 1u) & it->buf_mask;
        uint8_t* prev_row = in_row_ptr(it, prev_t);
        has_prev = (*in_row_tick(prev_row) == tick - 1u);
    }

    if (has_prev && W) {
        uint8_t* prev_row = in_row_ptr(it, prev_t);
        const uint64_t* prev_pres = in_row_present(prev_row);
        uint64_t* new_pres = in_row_present(row);
        uint8_t* prev_payload = in_row_payload(prev_row, W, A);
        uint8_t* new_payload  = in_row_payload(row, W, A);
        size_t stride = (size_t)it->stride;

        for (uint32_t w = 0; w < W; w++) {
            uint64_t carry = it->live_bm[w] & prev_pres[w];
            new_pres[w] |= carry;
            uint64_t m = carry;
            while (m) {
                uint32_t b = (uint32_t)ecs_ctz64(m); m &= m - 1ull;
                uint32_t idx = (w << 6) + b;
                memcpy(new_payload  + (size_t)idx * stride,
                       prev_payload + (size_t)idx * stride,
                       stride);
            }
        }
    }
}

/* ==========================================================================
   Resize -- both dimensions grow pow2.

   active_cap grow: row width changes (bitmap words + payload). Walk old
   rows, copy contents into new (wider) row positions.

   buf_size grow: row count changes. Walk old slots, place each at its
   new slot under wider mask. Per-row data already contiguous, single
   memcpy per row.
   ========================================================================== */

static void in_realloc_table(ecs_input_t* it, uint32_t new_cap,
                             uint32_t new_buf_size) {
    uint32_t old_cap        = it->active_cap;
    uint32_t old_buf_size   = it->buf_size;
    uint32_t old_W          = it->words_per_row;
    uint32_t new_W          = in_words_for_cap(new_cap);
    uint32_t old_row_bytes  = it->row_bytes;
    uint32_t new_row_bytes  = in_compute_row_bytes(new_W, new_cap, it->stride);

    size_t   total          = (size_t)new_buf_size * (size_t)new_row_bytes;
    uint8_t* new_table      = (uint8_t*)ecs_xmalloc_aligned(total, 8);
    /* Mark all new rows empty. tick_in_slot at offset 0 of each row.
       Pad bytes [4..8) left undefined -- only the u32 at offset 0 is read. */
    for (uint32_t i = 0; i < new_buf_size; i++) {
        uint32_t* tick_p = (uint32_t*)(new_table + (size_t)i * new_row_bytes);
        *tick_p = ECS_INPUT_TICK_NIL;
    }

    if (it->table && old_cap > 0u && old_buf_size > 0u) {
        /* Walk old rows, place each at its new slot. */
        uint32_t common_buf = old_buf_size < new_buf_size ? old_buf_size : new_buf_size;
        uint32_t new_mask = new_buf_size - 1u;

        for (uint32_t old_slot = 0; old_slot < common_buf; old_slot++) {
            uint8_t* old_row = it->table + (size_t)old_slot * old_row_bytes;
            uint32_t T = *(uint32_t*)old_row;
            if (T == ECS_INPUT_TICK_NIL) continue;
            uint32_t new_slot = T & new_mask;
            uint8_t* new_row = new_table + (size_t)new_slot * new_row_bytes;

            /* tick_in_slot */
            *(uint32_t*)new_row = T;

            /* Bitmaps: present + confirmed. Copy old_W words into the
               low part of new_W; rest stays 0. The rest of the new row
               (beyond the regions we touch) must be zero -- we wrote
               tick_in_slot above and didn't zero the rest of the new
               row, so do it explicitly here for safety. */
            uint64_t* new_pres = (uint64_t*)(new_row + IN_HDR_BYTES);
            uint64_t* new_conf = (uint64_t*)(new_row + IN_HDR_BYTES + (size_t)new_W * 8u);
            memset(new_pres, 0, (size_t)new_W * sizeof(uint64_t));
            memset(new_conf, 0, (size_t)new_W * sizeof(uint64_t));

            const uint64_t* old_pres = (const uint64_t*)(old_row + IN_HDR_BYTES);
            const uint64_t* old_conf = (const uint64_t*)(old_row + IN_HDR_BYTES + (size_t)old_W * 8u);
            uint32_t copy_W = old_W < new_W ? old_W : new_W;
            memcpy(new_pres, old_pres, (size_t)copy_W * sizeof(uint64_t));
            memcpy(new_conf, old_conf, (size_t)copy_W * sizeof(uint64_t));

            /* cmd_head + cmd_tail: copy oldA u32s into the low part of
               newA u32s; high entries stay 0. */
            uint32_t* new_chead = (uint32_t*)(new_row + IN_HDR_BYTES + (size_t)new_W * 16u);
            uint32_t* new_ctail = (uint32_t*)(new_row + IN_HDR_BYTES + (size_t)new_W * 16u + (size_t)new_cap * 4u);
            memset(new_chead, 0, (size_t)new_cap * sizeof(uint32_t));
            memset(new_ctail, 0, (size_t)new_cap * sizeof(uint32_t));

            const uint32_t* old_chead = (const uint32_t*)(old_row + IN_HDR_BYTES + (size_t)old_W * 16u);
            const uint32_t* old_ctail = (const uint32_t*)(old_row + IN_HDR_BYTES + (size_t)old_W * 16u + (size_t)old_cap * 4u);
            uint32_t copy_cap = old_cap < new_cap ? old_cap : new_cap;
            memcpy(new_chead, old_chead, (size_t)copy_cap * sizeof(uint32_t));
            memcpy(new_ctail, old_ctail, (size_t)copy_cap * sizeof(uint32_t));

            /* Payload: copy old_cap * stride bytes into new payload region.
               New columns at [old_cap..new_cap) stay uninitialized
               (bitmap presence governs reads). */
            uint8_t* new_payload = new_row + IN_HDR_BYTES + (size_t)new_W * 16u + (size_t)new_cap * 8u;
            const uint8_t* old_payload = old_row + IN_HDR_BYTES + (size_t)old_W * 16u + (size_t)old_cap * 8u;
            memcpy(new_payload, old_payload, (size_t)copy_cap * (size_t)it->stride);
        }
        ecs_free(it->table);
    }
    it->table         = new_table;
    it->row_bytes     = new_row_bytes;
    it->words_per_row = new_W;
    it->active_cap    = new_cap;

    /* cmd_arenas: parallel array indexed by ring slot. Re-map only when
       buf_size changes; under same buf_size the array is unchanged.
       Each per-ring-slot arena is keyed by tick (the row's resident
       tick) just like the row table. Orphan arenas (NIL row, retained
       cache buf) are freed during re-map. */
    if (new_buf_size != old_buf_size) {
        ecs_input_cmd_arena_t* old_arenas = it->cmd_arenas;
        ecs_input_cmd_arena_t* new_arenas =
            (ecs_input_cmd_arena_t*)ecs_xcalloc(new_buf_size, sizeof(ecs_input_cmd_arena_t));

        if (old_arenas && old_buf_size > 0u) {
            /* Walk new table: every resident row's tick gives its new
               ring slot directly. Locate the source arena at
               old_t = T & old_mask and move it. */
            uint32_t old_mask = old_buf_size - 1u;
            for (uint32_t new_t = 0; new_t < new_buf_size; new_t++) {
                uint8_t* row = new_table + (size_t)new_t * new_row_bytes;
                uint32_t T = *(uint32_t*)row;
                if (T == ECS_INPUT_TICK_NIL) continue;
                uint32_t old_t = T & old_mask;
                if (old_arenas[old_t].buf) {
                    new_arenas[new_t] = old_arenas[old_t];
                    old_arenas[old_t].buf = NULL;
                    old_arenas[old_t].len = 0u;
                    old_arenas[old_t].cap = 0u;
                }
            }
            /* Free orphan arena bufs (NIL old rows that still held a
               retained cache buffer, or non-NIL rows whose tick mapped
               to a new ring slot already taken). */
            for (uint32_t old_t = 0; old_t < old_buf_size; old_t++) {
                if (old_arenas[old_t].buf) {
                    ecs_free(old_arenas[old_t].buf);
                }
            }
        }
        if (old_arenas) ecs_free(old_arenas);
        it->cmd_arenas = new_arenas;
    }

    it->buf_size      = new_buf_size;
    it->buf_mask      = new_buf_size - 1u;
}

static void in_grow_active_cap(ecs_input_t* it, uint32_t need) {
    uint32_t old_cap = it->active_cap;
    uint32_t old_W   = it->words_per_row;
    uint32_t new_cap = old_cap ? old_cap : ECS_INPUT_PLAYER_CAP_INIT;
    while (new_cap < need) new_cap <<= 1;
    if (new_cap == old_cap) return;

    uint32_t new_W = in_words_for_cap(new_cap);

    /* Live bitmap: realloc + zero-extend new words. Old bits preserved. */
    it->live_bm = (uint64_t*)ecs_xrealloc(it->live_bm,
                        (size_t)new_W * sizeof(uint64_t));
    for (uint32_t w = old_W; w < new_W; w++) it->live_bm[w] = 0ull;

    in_realloc_table(it, new_cap, it->buf_size);
}

static void in_grow_buf_size(ecs_input_t* it, uint32_t new_buf_size) {
    assert(new_buf_size > it->buf_size);
    assert((new_buf_size & (new_buf_size - 1u)) == 0u);
    assert(new_buf_size <= ECS_INPUT_BUFSIZE_MAX);
    in_realloc_table(it, it->active_cap, new_buf_size);
}

void ecs_input_grow_buf(ecs_input_t* it, uint32_t new_buf_size) {
    assert(it);
    if (new_buf_size <= it->buf_size) return;
    uint32_t ns = it->buf_size;
    while (ns < new_buf_size) ns <<= 1;
    assert(ns <= ECS_INPUT_BUFSIZE_MAX);
    in_grow_buf_size(it, ns);
}

void ecs_input_grow_player_cap(ecs_input_t* it, uint32_t new_player_cap) {
    assert(it);
    if (new_player_cap <= it->active_cap) return;
    assert(new_player_cap <= ECS_INPUT_PLAYER_MAX);
    in_grow_active_cap(it, new_player_cap);
}

/* ==========================================================================
   Lifecycle
   ========================================================================== */

void ecs_input_init(ecs_input_t* it, uint32_t stride, uint32_t buf_size) {
    assert(it);
    assert(stride > 0);
    assert(buf_size > 0 && (buf_size & (buf_size - 1u)) == 0u);

    memset(it, 0, sizeof(*it));
    it->stride   = stride;
    it->buf_size = buf_size;
    it->buf_mask = buf_size - 1u;

    /* Allocate table immediately with active_cap=0 so all rows have
       a valid tick_in_slot header. Payload region is empty until first
       alloc_slot grows active_cap. row_bytes = padded(8 + 0 + 0) = 8. */
    it->row_bytes     = in_compute_row_bytes(0u, 0u, stride);
    size_t total      = (size_t)buf_size * (size_t)it->row_bytes;
    it->table         = (uint8_t*)ecs_xmalloc_aligned(total, 8);
    for (uint32_t i = 0; i < buf_size; i++) {
        *(uint32_t*)(it->table + (size_t)i * it->row_bytes) = ECS_INPUT_TICK_NIL;
    }

    /* Per-ring-slot command arenas. Lazy bufs (NULL until first append). */
    it->cmd_arenas = (ecs_input_cmd_arena_t*)ecs_xcalloc(
        buf_size, sizeof(ecs_input_cmd_arena_t));

    it->confirmed_frontier = 0u;
}

void ecs_input_destroy(ecs_input_t* it) {
    if (!it) return;
    if (it->table)   ecs_free(it->table);
    if (it->live_bm) ecs_free(it->live_bm);
    if (it->cmd_arenas) {
        for (uint32_t t = 0; t < it->buf_size; t++) {
            if (it->cmd_arenas[t].buf) ecs_free(it->cmd_arenas[t].buf);
        }
        ecs_free(it->cmd_arenas);
    }
    memset(it, 0, sizeof(*it));
}

/* ==========================================================================
   Membership
   ========================================================================== */

uint32_t ecs_input_alloc_slot(ecs_input_t* it) {
    assert(it);
    if (it->active_count >= ECS_INPUT_PLAYER_MAX) return ECS_INPUT_SLOT_NIL;

    if (it->active_cap == 0u || it->active_count >= it->active_cap) {
        in_grow_active_cap(it, it->active_count + 1u);
    }
    uint32_t slot = in_find_free_slot(it);
    assert(slot != ECS_INPUT_SLOT_NIL);

    /* ABA defense: clear stale bits at slot in any still-live tick row.
       Safe to always run -- empty rows have tick == NIL and are skipped.
       cmd_head/tail at this column also scrubbed: stale chain bytes in
       the row's arena become unreachable garbage and get reclaimed on
       the next first-touch reset (arena.len = 0). */
    if (it->words_per_row) {
        uint32_t W   = it->words_per_row;
        uint32_t A   = it->active_cap;
        uint32_t word= slot >> 6;
        uint64_t bit = 1ull << (slot & 63u);
        uint64_t notbit = ~bit;
        for (uint32_t t = 0; t < it->buf_size; t++) {
            uint8_t* row = in_row_ptr(it, t);
            if (*in_row_tick(row) == ECS_INPUT_TICK_NIL) continue;
            uint64_t* pres = in_row_present(row);
            uint64_t* conf = in_row_confirmed(row, W);
            pres[word] &= notbit;
            conf[word] &= notbit;
            uint32_t* chead = in_row_cmd_head(row, W);
            uint32_t* ctail = in_row_cmd_tail(row, W, A);
            chead[slot] = 0u;
            ctail[slot] = 0u;
        }
    }

    it->live_bm[slot >> 6] |= 1ull << (slot & 63u);
    it->active_count++;
    return slot;
}

void ecs_input_free_slot(ecs_input_t* it, uint32_t slot) {
    assert(it);
    if (slot >= it->active_cap) return;
    uint64_t bit = 1ull << (slot & 63u);
    uint64_t* w = &it->live_bm[slot >> 6];
    if (!(*w & bit)) return;
    *w &= ~bit;
    it->active_count--;
}

bool ecs_input_slot_is_live(const ecs_input_t* it, uint32_t slot) {
    assert(it);
    return in_slot_live(it, slot);
}

/* ==========================================================================
   Hot path
   ========================================================================== */

void ecs_input_set(ecs_input_t* it, uint32_t tick, uint32_t slot,
                   const void* value, bool confirmed)
{
    assert(it);
    assert(value);
    assert(tick != ECS_INPUT_TICK_NIL);
    assert(tick >= 1u);   /* tick 0 reserved as "no frontier" sentinel */

    if (!in_slot_live(it, slot)) return;

    /* Auto-grow ring if first-touch would clobber a still-live slot. */
    {
        uint32_t t0   = tick & it->buf_mask;
        uint8_t* row0 = in_row_ptr(it, t0);
        uint32_t prev = *in_row_tick(row0);
        bool would_evict = (prev != ECS_INPUT_TICK_NIL && prev != tick &&
                            prev > it->confirmed_frontier);
        if (would_evict) {
            uint32_t diff = (tick > prev) ? (tick - prev) : (prev - tick);
            if (diff < ECS_INPUT_BUFSIZE_MAX) {
                uint32_t need = diff + 1u;
                uint32_t ns   = it->buf_size;
                while (ns < need) ns <<= 1;
                if (ns <= ECS_INPUT_BUFSIZE_MAX) in_grow_buf_size(it, ns);
            }
        }
    }

    uint32_t t  = tick & it->buf_mask;
    uint32_t W  = it->words_per_row;
    uint32_t A  = it->active_cap;
    uint8_t* row = in_row_ptr(it, t);

    if (*in_row_tick(row) != tick) {
        assert(tick > it->confirmed_frontier);
        in_advance_row(it, t, tick);
        *in_row_tick(row) = tick;
    }

    uint64_t* conf = in_row_confirmed(row, W);
    uint64_t* pres = in_row_present(row);
    uint64_t  bit  = 1ull << (slot & 63u);
    uint32_t  word = slot >> 6;
    bool was_confirmed = (conf[word] & bit) != 0ull;

    if (!confirmed && was_confirmed) return;

    memcpy(in_row_cell(row, W, A, slot, it->stride), value, it->stride);
    pres[word] |= bit;

    if (confirmed && !was_confirmed) {
        conf[word] |= bit;
    }

    /* Backward prediction fill. Out-of-order arrivals (tick N received
       before N-1) leave the in-between rows without this slot's bytes;
       backfill imprints `value` as PREDICTED at past ticks where this
       slot is currently absent. Walks T = tick-1, tick-2, ... and stops
       on the first row where the slot is already present, the row is
       sealed (T <= frontier), tick falls below 1, or the ring slot is
       holding a non-evictable foreign tick. Confirmed bit is never set
       on backfilled rows. */
    {
        uint32_t mask = it->buf_mask;
        uint32_t buf  = it->buf_size;
        uint32_t T    = tick;
        for (uint32_t step = 1u; step < buf; step++) {
            if (T == 0u) break;
            T--;
            if (T == 0u) break;                        /* tick 0 reserved */
            if (T <= it->confirmed_frontier) break;    /* sealed past */

            uint32_t rid    = T & mask;
            uint8_t* prow   = in_row_ptr(it, rid);
            uint32_t cur    = *in_row_tick(prow);

            if (cur == T) {
                /* Row resident for this past tick. Imprint slot if absent. */
                uint64_t* p_pres = in_row_present(prow);
                if ((p_pres[word] & bit) != 0ull) break;   /* already filled */
                memcpy(in_row_cell(prow, W, A, slot, it->stride), value, it->stride);
                p_pres[word] |= bit;
                continue;
            }

            /* Row holds something else. Only safe to claim if NIL or holds
               a sealed tick. Holding a different live tick (predicted future
               or a still-live past) -> stop, do not evict. */
            bool can_claim = (cur == ECS_INPUT_TICK_NIL) ||
                             (cur != tick && cur <= it->confirmed_frontier);
            if (!can_claim) break;

            in_advance_row(it, rid, T);
            *in_row_tick(prow) = T;
            uint64_t* p_pres = in_row_present(prow);
            memcpy(in_row_cell(prow, W, A, slot, it->stride), value, it->stride);
            p_pres[word] |= bit;
        }
    }
}

const void* ecs_input_get(const ecs_input_t* it, uint32_t tick, uint32_t slot) {
    if (!in_slot_live(it, slot)) return NULL;
    uint32_t t  = tick & it->buf_mask;
    uint8_t* row = in_row_ptr(it, t);
    if (*in_row_tick(row) != tick) return NULL;
    uint32_t W  = it->words_per_row;
    uint32_t A  = it->active_cap;
    if (!W) return NULL;
    const uint64_t* pres = in_row_present(row);
    if (!((pres[slot >> 6] >> (slot & 63u)) & 1u)) return NULL;
    return in_row_cell(row, W, A, slot, it->stride);
}

ecs_input_view_t ecs_input_get_view(const ecs_input_t* it,
                                    uint32_t tick, uint32_t slot)
{
    ecs_input_view_t v = { NULL, false, false };

    uint32_t t  = tick & it->buf_mask;
    uint32_t W  = it->words_per_row;
    uint32_t A  = it->active_cap;
    uint8_t* row = in_row_ptr(it, t);
    if (*in_row_tick(row) != tick || !W) return v;

    if (!in_slot_live(it, slot)) return v;

    const uint64_t* pres = in_row_present(row);
    const uint64_t* conf = in_row_confirmed(row, W);
    v.present   = (pres[slot >> 6] >> (slot & 63u)) & 1u;
    v.confirmed = (conf[slot >> 6] >> (slot & 63u)) & 1u;
    if (v.present) {
        v.data = in_row_cell(row, W, A, slot, it->stride);
    }
    return v;
}

void ecs_input_clear(ecs_input_t* it, uint32_t tick) {
    assert(it);
    assert(tick != ECS_INPUT_TICK_NIL);
    assert(tick >= 1u);
    uint32_t t = tick & it->buf_mask;
    in_reset_row(it, t);
}

bool ecs_input_tick_confirmed(const ecs_input_t* it, uint32_t tick) {
    /* Anything sim has already sealed is authoritative by definition --
       even if its ring slot has since been evicted by a newer write. */
    if (tick != 0u && tick <= it->confirmed_frontier) return true;
    if (it->active_count == 0u) return true;
    uint32_t t = tick & it->buf_mask;
    uint8_t* row = in_row_ptr(it, t);
    if (*in_row_tick(row) != tick) return false;
    return in_tick_all_confirmed(it, t);
}

uint32_t ecs_input_frontier(const ecs_input_t* it) {
    return it->confirmed_frontier;
}

void ecs_input_advance_to_tick(ecs_input_t* it, uint32_t tick) {
    assert(it);
    assert(tick != ECS_INPUT_TICK_NIL);
    assert(tick >= 1u);
    assert(tick >= it->confirmed_frontier);   /* frontier monotonic */
    it->confirmed_frontier = tick;
}

/* ==========================================================================
   Command stream

   Per-(tick, slot) FIFO list of opaque variable-bit-length commands.
   Authoritative-only -- never carried forward across ticks. Each ring
   row owns one bump arena; per-slot head/tail offsets in the row index
   into it. Append is O(1), iter is O(commands per slot). Reset on
   first-touch eviction (arena.len = 0, head/tail = 0; arena.buf retained).

   Node layout in arena (byte-packed):
     u32 next_off     offset+1 of next node in this slot's chain, 0 = end 
     u32 bit_len      exact payload bit count 
     u8  bytes[ceil(bit_len/8)]
   ========================================================================== */

void ecs_input_cmd_append(ecs_input_t* it, uint32_t tick, uint32_t slot,
                          const void* bytes, uint32_t bit_len)
{
    assert(it);
    assert(tick != ECS_INPUT_TICK_NIL);
    assert(tick >= 1u);
    assert(bit_len <= ECS_INPUT_CMD_MAX_BITS);
    assert(bytes || bit_len == 0u);

    if (tick <= it->confirmed_frontier) return;       /* sealed -- drop */
    if (!in_slot_live(it, slot)) return;

    /* Auto-grow ring if first-touch would clobber a still-live slot.
       Same gate as ecs_input_set. */
    {
        uint32_t t0   = tick & it->buf_mask;
        uint8_t* row0 = in_row_ptr(it, t0);
        uint32_t prev = *in_row_tick(row0);
        bool would_evict = (prev != ECS_INPUT_TICK_NIL && prev != tick &&
                            prev > it->confirmed_frontier);
        if (would_evict) {
            uint32_t diff = (tick > prev) ? (tick - prev) : (prev - tick);
            if (diff < ECS_INPUT_BUFSIZE_MAX) {
                uint32_t need = diff + 1u;
                uint32_t ns   = it->buf_size;
                while (ns < need) ns <<= 1;
                if (ns <= ECS_INPUT_BUFSIZE_MAX) in_grow_buf_size(it, ns);
            }
        }
    }

    uint32_t t = tick & it->buf_mask;
    uint32_t W = it->words_per_row;
    uint32_t A = it->active_cap;
    uint8_t* row = in_row_ptr(it, t);

    if (*in_row_tick(row) != tick) {
        assert(tick > it->confirmed_frontier);
        in_advance_row(it, t, tick);
        *in_row_tick(row) = tick;
    }

    ecs_input_cmd_arena_t* a = &it->cmd_arenas[t];
    uint32_t store_bytes = (bit_len + 7u) >> 3;
    uint32_t need = 8u + store_bytes;
    if (a->len + need > a->cap) {
        uint32_t nc = a->cap ? a->cap : 64u;
        while (nc < a->len + need) nc <<= 1;
        a->buf = (uint8_t*)ecs_xrealloc(a->buf, nc);
        a->cap = nc;
    }

    uint32_t off = a->len;
    uint32_t* heads = in_row_cmd_head(row, W);
    uint32_t* tails = in_row_cmd_tail(row, W, A);

    /* Write node: next=0 (becomes new tail), bit_len, payload. */
    uint32_t zero = 0u;
    memcpy(a->buf + off,     &zero,    4);
    memcpy(a->buf + off + 4, &bit_len, 4);
    if (store_bytes) memcpy(a->buf + off + 8, bytes, store_bytes);

    if (heads[slot] == 0u) {
        heads[slot] = off + 1u;
    } else {
        /* Update prev tail's next field -> point to this node. */
        uint32_t prev_off = tails[slot] - 1u;
        uint32_t next_plus = off + 1u;
        memcpy(a->buf + prev_off, &next_plus, 4);
    }
    tails[slot] = off + 1u;

    a->len += need;
}

ecs_input_cmd_iter_t ecs_input_cmd_iter_begin(const ecs_input_t* it,
                                              uint32_t tick, uint32_t slot)
{
    ecs_input_cmd_iter_t r = { it, NULL, 0u };
    if (!it || !it->cmd_arenas) return r;
    if (!in_slot_live(it, slot)) return r;
    uint32_t t = tick & it->buf_mask;
    uint8_t* row = in_row_ptr(it, t);
    if (*in_row_tick(row) != tick) return r;
    uint32_t W = it->words_per_row;
    if (slot >= it->active_cap) return r;
    const uint32_t* heads = in_row_cmd_head(row, W);
    r.arena    = it->cmd_arenas[t].buf;
    r.next_off = heads[slot];
    return r;
}

bool ecs_input_cmd_iter_next(ecs_input_cmd_iter_t* iter,
                             const void** out_bytes,
                             uint32_t* out_bit_len)
{
    if (!iter || !iter->arena || iter->next_off == 0u) {
        if (out_bytes)   *out_bytes   = NULL;
        if (out_bit_len) *out_bit_len = 0u;
        return false;
    }
    uint32_t off = iter->next_off - 1u;
    uint32_t next, bit_len;
    memcpy(&next,    iter->arena + off,     4);
    memcpy(&bit_len, iter->arena + off + 4, 4);
    if (out_bytes)   *out_bytes   = iter->arena + off + 8;
    if (out_bit_len) *out_bit_len = bit_len;
    iter->next_off = next;
    return true;
}

/* ==========================================================================
   Serialization

   Cascading delta compression. Per slot the encoder emits a unary
   prefix selecting the matching reference (baseline = zeros, then
   each of T-1 .. T-redundancy), followed by raw bit-packed bytes if
   no reference matches.

   Past ticks not currently resident in the ring are treated as the
   zero baseline by both encoder and decoder, so the wire is the same
   regardless of ring eviction state.
   ========================================================================== */

static inline void in_read_payload_bits(ecs_deserializer_t* d,
                                        uint8_t* p, uint32_t stride) {
    uint32_t left = stride;
    while (left >= 8u) {
        uint64_t v = ecs_deserializer_read_bits(d, 64);
        memcpy(p, &v, 8);
        p += 8;
        left -= 8u;
    }
    if (left) {
        uint64_t v = ecs_deserializer_read_bits(d, (int32_t)(left * 8u));
        memcpy(p, &v, left);
    }
}

static inline void in_write_payload_bits(ecs_serializer_t* s,
                                         const uint8_t* p, uint32_t stride) {
    uint32_t left = stride;
    while (left >= 8u) {
        uint64_t v;
        memcpy(&v, p, 8);
        ecs_serializer_write_bits(s, v, 64);
        p += 8;
        left -= 8u;
    }
    if (left) {
        uint64_t v = 0;
        memcpy(&v, p, left);
        ecs_serializer_write_bits(s, v, (int32_t)(left * 8u));
    }
}

/* Variable bit-length blob writer. Storage is byte-rounded; emits exact
   bit_len bits. Tail < 64 bits are masked so unused storage bits don't
   leak onto the wire. */
static inline void in_write_cmd_bits(ecs_serializer_t* s,
                                     const uint8_t* p, uint32_t bit_len) {
    uint32_t left = bit_len;
    while (left >= 64u) {
        uint64_t v;
        memcpy(&v, p, 8);
        ecs_serializer_write_bits(s, v, 64);
        p += 8;
        left -= 64u;
    }
    if (left) {
        uint64_t v = 0;
        uint32_t bytes_left = (left + 7u) >> 3;
        memcpy(&v, p, bytes_left);
        v &= (left == 64u) ? ~0ull : ((1ull << left) - 1ull);
        ecs_serializer_write_bits(s, v, (int32_t)left);
    }
}

/* Variable bit-length blob reader. Byte-rounded receive buffer; final
   byte's high bits (above bit_len) are zero-padded. */
static inline void in_read_cmd_bits(ecs_deserializer_t* d,
                                    uint8_t* p, uint32_t bit_len) {
    uint32_t left = bit_len;
    while (left >= 64u) {
        uint64_t v = ecs_deserializer_read_bits(d, 64);
        memcpy(p, &v, 8);
        p += 8;
        left -= 64u;
    }
    if (left) {
        uint64_t v = ecs_deserializer_read_bits(d, (int32_t)left);
        uint32_t bytes_left = (left + 7u) >> 3;
        /* Zero the destination bytes first so unread high bits of the
           final byte are 0, then memcpy only the bytes we have. */
        memset(p, 0, bytes_left);
        memcpy(p, &v, bytes_left);
    }
}

void ecs_input_serialize_tick(const ecs_input_t* it,
                              ecs_serializer_t* s,
                              uint32_t tick,
                              uint32_t redundancy)
{
    assert(it);
    assert(s);
    assert(tick >= 1u);

    /* Header. Caller is responsible for ensuring peer agrees on
       active_cap and stride out-of-band. */
    ecs_serializer_write_bits(s, tick, 32);
    ecs_serializer_write_bits(s, (uint64_t)redundancy, 8);

    uint32_t cap = it->active_cap;
    if (cap == 0u) return;

    uint32_t W      = it->words_per_row;
    uint32_t stride = it->stride;
    uint32_t mask   = it->buf_mask;

    /* Resolve current row. If the ring slot does not hold this tick
       every slot is treated as zeros (matches baseline -> 1 bit each). */
    uint32_t t_cur   = tick & mask;
    uint8_t* row_cur = in_row_ptr(it, t_cur);
    int cur_valid   = (*in_row_tick(row_cur) == tick);
    const uint64_t* pres_cur = cur_valid ? in_row_present(row_cur)         : NULL;
    const uint8_t*  pay_cur  = cur_valid ? in_row_payload(row_cur, W, cap) : NULL;

    uint8_t zero_buf[256] = {0};
    assert(stride <= sizeof(zero_buf));

    for (uint32_t idx = 0; idx < cap; idx++) {
        uint32_t w = idx >> 6;
        uint64_t b = 1ull << (idx & 63u);

        const uint8_t* cur_bytes = zero_buf;
        if (cur_valid && (pres_cur[w] & b)) {
            cur_bytes = pay_cur + (size_t)idx * stride;
        }

        /* Bit 0: differs from baseline (all zeros). */
        int diff_baseline = (cur_bytes != zero_buf) &&
                            (memcmp(cur_bytes, zero_buf, stride) != 0);
        if (!diff_baseline) {
            ecs_serializer_write_bits(s, 0u, 1);
            continue;
        }
        ecs_serializer_write_bits(s, 1u, 1);

        /* Cascade: emit one bit per redundancy ref, stop on match.
           Past-tick row resolved on demand; unresolved == zero baseline. */
        int matched = 0;
        for (uint32_t r = 0; r < redundancy; r++) {
            const uint8_t* ref_bytes = zero_buf;
            uint32_t off = r + 1u;
            if (tick > off) {
                uint32_t past   = tick - off;
                uint32_t tp     = past & mask;
                uint8_t* row_p  = in_row_ptr(it, tp);
                if (*in_row_tick(row_p) == past) {
                    const uint64_t* pres_p = in_row_present(row_p);
                    if (pres_p[w] & b) {
                        ref_bytes = in_row_payload(row_p, W, cap)
                                  + (size_t)idx * stride;
                    }
                }
            }
            int diff_ref = memcmp(cur_bytes, ref_bytes, stride) != 0;
            ecs_serializer_write_bits(s, diff_ref ? 1u : 0u, 1);
            if (!diff_ref) { matched = 1; break; }
        }

        if (!matched) {
            in_write_payload_bits(s, cur_bytes, stride);
        }
    }

    /* Commands. 1-bit gate: any commands at this tick? */
    int any_cmds = 0;
    if (cur_valid) {
        const uint32_t* heads = in_row_cmd_head(row_cur, W);
        for (uint32_t idx = 0; idx < cap; idx++) {
            if (heads[idx]) { any_cmds = 1; break; }
        }
    }
    ecs_serializer_write_bits(s, any_cmds ? 1u : 0u, 1);
    if (!any_cmds) return;

    const uint32_t* heads = in_row_cmd_head(row_cur, W);
    const uint8_t*  arena_buf = it->cmd_arenas[t_cur].buf;
    for (uint32_t idx = 0; idx < cap; idx++) {
        uint32_t off_plus = heads[idx];
        if (off_plus == 0u) {
            ecs_serializer_write_bits(s, 0u, 1);
            continue;
        }
        ecs_serializer_write_bits(s, 1u, 1);
        while (off_plus) {
            uint32_t off = off_plus - 1u;
            uint32_t next, bit_len;
            memcpy(&next,    arena_buf + off,     4);
            memcpy(&bit_len, arena_buf + off + 4, 4);
            const uint8_t* p = arena_buf + off + 8;
            ecs_serializer_write_bits(s, bit_len, ECS_INPUT_CMD_BIT_LEN_BITS);
            in_write_cmd_bits(s, p, bit_len);
            ecs_serializer_write_bits(s, next ? 1u : 0u, 1);
            off_plus = next;
        }
    }
}

void ecs_input_deserialize_tick(ecs_input_t* it, ecs_deserializer_t* d) {
    assert(it);
    assert(d);

    uint32_t tick       = (uint32_t)ecs_deserializer_read_bits(d, 32);
    uint32_t redundancy = (uint32_t)ecs_deserializer_read_bits(d, 8);
    assert(tick >= 1u);

    uint32_t cap = it->active_cap;
    if (cap == 0u) return;

    uint32_t W      = it->words_per_row;
    uint32_t stride = it->stride;
    uint32_t mask   = it->buf_mask;

    uint8_t zero_buf[256] = {0};
    assert(stride <= sizeof(zero_buf));
    uint8_t value_buf[256];

    for (uint32_t idx = 0; idx < cap; idx++) {
        uint32_t w = idx >> 6;
        uint64_t b = 1ull << (idx & 63u);
        const uint8_t* value_bytes;

        /* Bit 0: differs from baseline? */
        if (ecs_deserializer_read_bits(d, 1) == 0ull) {
            value_bytes = zero_buf;
        } else {
            int matched = 0;
            for (uint32_t r = 0; r < redundancy; r++) {
                if (ecs_deserializer_read_bits(d, 1) == 0ull) {
                    /* Slot equals tick T-r in receiver's ring; resolve. */
                    value_bytes = zero_buf;
                    uint32_t off = r + 1u;
                    if (tick > off) {
                        uint32_t past = tick - off;
                        uint32_t tp = past & mask;
                        uint8_t* row_p = in_row_ptr(it, tp);
                        if (*in_row_tick(row_p) == past) {
                            const uint64_t* pres_p = in_row_present(row_p);
                            if (pres_p[w] & b) {
                                value_bytes = in_row_payload(row_p, W, cap)
                                            + (size_t)idx * stride;
                            }
                        }
                    }
                    matched = 1;
                    break;
                }
            }
            if (!matched) {
                /* All cascade bits 1: raw payload follows. */
                in_read_payload_bits(d, value_buf, stride);
                value_bytes = value_buf;
            }
        }

        /* Apply as confirmed. Skip slots whose live bit is clear. */
        if (!in_slot_live(it, idx)) continue;
        ecs_input_set(it, tick, idx, value_bytes, true);
    }

    /* Commands. */
    if (ecs_deserializer_read_bits(d, 1) == 0ull) return;

    /* Idempotent re-receive: assume every packet for the same tick
       carries the same authoritative command set (server-authoritative
       model). If we already have a populated cmd arena at this tick
       row, treat the packet as a duplicate and walk the wire without
       applying. State stays untouched -- no clear, no double-append. */
    uint32_t t_cur = tick & mask;
    uint8_t* row_cur = in_row_ptr(it, t_cur);
    bool skip_apply = (*in_row_tick(row_cur) == tick) &&
                      it->cmd_arenas &&
                      it->cmd_arenas[t_cur].len > 0u;

    uint8_t cmd_buf[ECS_INPUT_CMD_MAX_BYTES];
    for (uint32_t idx = 0; idx < cap; idx++) {
        if (ecs_deserializer_read_bits(d, 1) == 0ull) continue;
        while (1) {
            uint32_t bit_len = (uint32_t)ecs_deserializer_read_bits(
                d, ECS_INPUT_CMD_BIT_LEN_BITS);
            if (bit_len) {
                in_read_cmd_bits(d, cmd_buf, bit_len);
            }
            if (!skip_apply) {
                /* Live-bit skip mirrors input apply (bits already consumed). */
                ecs_input_cmd_append(it, tick, idx, cmd_buf, bit_len);
            }
            if (ecs_deserializer_read_bits(d, 1) == 0ull) break;
        }
    }
}

bool ecs_input_iter_next(ecs_input_iter_t* iter) {
    assert(iter && iter->it);
    const ecs_input_t* it = iter->it;
    uint32_t cap = it->active_cap;
    uint32_t i   = iter->cursor;
    if (i >= cap) {
        iter->cursor = cap;
        iter->slot   = ECS_INPUT_SLOT_NIL;
        return false;
    }
    uint32_t W = it->words_per_row;
    uint32_t w = i >> 6;
    /* Mask off bits below i in the starting word, then ctz across words. */
    uint64_t m = it->live_bm[w] & (~0ull << (i & 63u));
    while (w < W) {
        if (m) {
            uint32_t b    = (uint32_t)ecs_ctz64(m);
            uint32_t slot = (w << 6) + b;
            if (slot >= cap) break;
            iter->slot   = slot;
            iter->cursor = slot + 1u;
            return true;
        }
        if (++w >= W) break;
        m = it->live_bm[w];
    }
    iter->cursor = cap;
    iter->slot   = ECS_INPUT_SLOT_NIL;
    return false;
}
