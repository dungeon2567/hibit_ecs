#include "input.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>   /* AVX2 */

#include "ecs.h"   /* ecs_xmalloc_aligned, ecs_xcalloc, ecs_xrealloc, ecs_free */

/* ==========================================================================
   Row layout (packed, single allocation in it->table)

     offset 0           : u64 tick_in_slot
     offset 8           : u64 present[words_per_row]
     offset 8+W*8       : u64 confirmed[words_per_row]
     offset 8+2*W*8     : u8  payload[active_cap * stride]

   row_bytes = padded-up-to-8(8 + 2*W*8 + A*S).

   Per-slot ops (set/get/get_view, advance_row, reset_row, all_confirmed
   check) touch a single row = warm cache.

   "all_confirmed" for a slot is computed on demand by SIMD-scanning
   dense_ids[] for the live mask and AND-checking against the row's
   confirmed bits. No cached counts / bitmap.
   ========================================================================== */

#define IN_HDR_BYTES 8u   /* tick_in_slot prefix size */

static inline uint32_t in_words_for_cap(uint32_t cap) {
    return (cap + 63u) >> 6;
}

static inline uint32_t in_compute_row_bytes(uint32_t W, uint32_t A, uint32_t S) {
    uint32_t b = IN_HDR_BYTES + 2u * W * 8u + A * S;
    return (b + 7u) & ~7u;
}

static inline uint8_t* in_row_ptr(const ecs_input_t* it, uint32_t slot) {
    return it->table + (size_t)slot * (size_t)it->row_bytes;
}
static inline uint64_t* in_row_tick(uint8_t* row) {
    return (uint64_t*)row;
}
static inline uint64_t* in_row_present(uint8_t* row) {
    return (uint64_t*)(row + IN_HDR_BYTES);
}
static inline uint64_t* in_row_confirmed(uint8_t* row, uint32_t W) {
    return (uint64_t*)(row + IN_HDR_BYTES + (size_t)W * 8u);
}
static inline uint8_t* in_row_payload(uint8_t* row, uint32_t W) {
    return row + IN_HDR_BYTES + (size_t)W * 16u;
}
static inline uint8_t* in_row_cell(uint8_t* row, uint32_t W,
                                   uint32_t idx, uint32_t stride) {
    return in_row_payload(row, W) + (size_t)idx * (size_t)stride;
}

/* SIMD scan dense_ids for first slot whose value == target_val. */
static uint32_t in_scan_dense_ids(const ecs_input_t* it, uint32_t target_val) {
    uint32_t cap = it->active_cap;
    if (!cap) return ECS_INPUT_PID_NIL;
    const uint32_t* ids = it->dense_ids;
    __m256i target = _mm256_set1_epi32((int)target_val);
    for (uint32_t i = 0; i < cap; i += 8) {
        __m256i v  = _mm256_loadu_si256((const __m256i*)(ids + i));
        __m256i eq = _mm256_cmpeq_epi32(v, target);
        int mask   = _mm256_movemask_ps(_mm256_castsi256_ps(eq));
        if (mask) return i + (uint32_t)_tzcnt_u32((unsigned)mask);
    }
    return ECS_INPUT_PID_NIL;
}

static inline uint32_t in_pid_lookup(const ecs_input_t* it, ecs_pid_t pid) {
    if (pid >= ECS_INPUT_PID_MAX) return ECS_INPUT_PID_NIL;
    return in_scan_dense_ids(it, (uint32_t)pid);
}

/* Build live mask per word: bits set where dense_ids[w*64+b] != NIL.
   SIMD-scans 8 ids per AVX2 op, packs result into uint64_t. */
static inline uint64_t in_live_mask_word(const ecs_input_t* it, uint32_t w) {
    uint64_t live = 0;
    uint32_t base = w * 64u;
    uint32_t cap  = it->active_cap;
    __m256i nil_v = _mm256_set1_epi32(-1);
    for (uint32_t k = 0; k < 64u; k += 8u) {
        uint32_t off = base + k;
        if (off >= cap) break;
        __m256i v  = _mm256_loadu_si256((const __m256i*)(it->dense_ids + off));
        __m256i eq = _mm256_cmpeq_epi32(v, nil_v);
        int mask   = _mm256_movemask_ps(_mm256_castsi256_ps(eq));
        /* mask: bit b set => dense_ids[off+b] == NIL.  live bit = !NIL. */
        uint64_t bits = (~(uint64_t)mask) & 0xFFull;
        live |= bits << k;
    }
    return live;
}

/* True iff every currently-registered player has its confirmed bit set
   at ring slot `t`. Computed on demand. O(active_cap / 8) SIMD ops. */
static bool in_tick_all_confirmed(const ecs_input_t* it, uint32_t t) {
    if (it->active_cap == 0u) return true;
    uint8_t* row = in_row_ptr(it, t);
    uint64_t* confirmed = in_row_confirmed(row, it->words_per_row);
    uint32_t W = it->words_per_row;
    for (uint32_t w = 0; w < W; w++) {
        uint64_t live = in_live_mask_word(it, w);
        if ((confirmed[w] & live) != live) return false;
    }
    return true;
}

/* Greedily advance confirmed_frontier. Frontier seeds when tick 1 fully
   confirms; from there it extends contiguously. Bound: buf_size.
   confirmed_frontier == 0 means "no frontier yet" (tick 0 is reserved
   sentinel; valid ticks start at 1). */
static void in_advance_frontier(ecs_input_t* it, uint64_t just_confirmed_tick) {
    if (just_confirmed_tick != it->confirmed_frontier + 1ull) return;
    it->confirmed_frontier = just_confirmed_tick;
    for (uint32_t i = 0; i < it->buf_size; i++) {
        uint64_t n = it->confirmed_frontier + 1ull;
        uint32_t t = (uint32_t)(n & it->buf_mask);
        uint8_t* row = in_row_ptr(it, t);
        if (*in_row_tick(row) != n) break;
        if (!in_tick_all_confirmed(it, t)) break;
        it->confirmed_frontier = n;
    }
}

/* Reset row: zero present + confirmed bits, mark slot empty. Used by
   ecs_input_clear and ecs_input_seal_empty_tick (no carry-forward). */
static void in_reset_row(ecs_input_t* it, uint32_t t) {
    uint8_t* row = in_row_ptr(it, t);
    uint32_t W = it->words_per_row;
    if (W) {
        memset(in_row_present(row),   0, (size_t)W * sizeof(uint64_t));
        memset(in_row_confirmed(row, W), 0, (size_t)W * sizeof(uint64_t));
    }
    *in_row_tick(row) = ECS_INPUT_TICK_NIL;
}

/* First-touch advance: carry every live player's bytes forward from
   prev tick into the new ring slot, mark them present (predicted),
   clear confirmed. Implements input persistence. */
static void in_advance_row(ecs_input_t* it, uint32_t t, uint64_t tick) {
    uint8_t* row = in_row_ptr(it, t);
    uint32_t W = it->words_per_row;

    if (W) {
        memset(in_row_present(row),    0, (size_t)W * sizeof(uint64_t));
        memset(in_row_confirmed(row, W), 0, (size_t)W * sizeof(uint64_t));
    }

    bool has_prev = false;
    uint32_t prev_t = 0;
    if (tick > 0ull) {
        prev_t = (uint32_t)((tick - 1ull) & it->buf_mask);
        uint8_t* prev_row = in_row_ptr(it, prev_t);
        has_prev = (*in_row_tick(prev_row) == tick - 1ull);
    }

    if (has_prev && W) {
        uint8_t* prev_row = in_row_ptr(it, prev_t);
        const uint64_t* prev_pres = in_row_present(prev_row);
        uint64_t* new_pres = in_row_present(row);
        uint8_t* prev_payload = in_row_payload(prev_row, W);
        uint8_t* new_payload  = in_row_payload(row, W);
        size_t stride = (size_t)it->stride;

        for (uint32_t idx = 0; idx < it->dense_high; idx++) {
            if (it->dense_ids[idx] == ECS_INPUT_PID_NIL) continue;
            uint64_t bit = 1ull << (idx & 63u);
            uint32_t word = idx >> 6;
            if (!(prev_pres[word] & bit)) continue;
            memcpy(new_payload  + (size_t)idx * stride,
                   prev_payload + (size_t)idx * stride,
                   stride);
            new_pres[word] |= bit;
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
    /* Mark all new rows empty. tick_in_slot at offset 0 of each row. */
    for (uint32_t i = 0; i < new_buf_size; i++) {
        uint64_t* tick_p = (uint64_t*)(new_table + (size_t)i * new_row_bytes);
        *tick_p = ECS_INPUT_TICK_NIL;
    }

    if (it->table && old_cap > 0u && old_buf_size > 0u) {
        /* Walk old rows, place each at its new slot. */
        uint32_t common_buf = old_buf_size < new_buf_size ? old_buf_size : new_buf_size;
        uint64_t new_mask = new_buf_size - 1u;

        for (uint32_t old_slot = 0; old_slot < common_buf; old_slot++) {
            uint8_t* old_row = it->table + (size_t)old_slot * old_row_bytes;
            uint64_t T = *(uint64_t*)old_row;
            if (T == ECS_INPUT_TICK_NIL) continue;
            uint32_t new_slot = (uint32_t)(T & new_mask);
            uint8_t* new_row = new_table + (size_t)new_slot * new_row_bytes;

            /* tick_in_slot */
            *(uint64_t*)new_row = T;

            /* Bitmaps: present + confirmed. Copy old_W words into the
               low part of new_W; rest stays 0 (alloc was xmalloc, but
               regions beyond what we touch must be zero -- we wrote
               tick_in_slot above and didn't zero the rest of the new
               row, so do it explicitly here for safety). */
            uint64_t* new_pres = (uint64_t*)(new_row + IN_HDR_BYTES);
            uint64_t* new_conf = (uint64_t*)(new_row + IN_HDR_BYTES + (size_t)new_W * 8u);
            memset(new_pres, 0, (size_t)new_W * sizeof(uint64_t));
            memset(new_conf, 0, (size_t)new_W * sizeof(uint64_t));

            const uint64_t* old_pres = (const uint64_t*)(old_row + IN_HDR_BYTES);
            const uint64_t* old_conf = (const uint64_t*)(old_row + IN_HDR_BYTES + (size_t)old_W * 8u);
            uint32_t copy_W = old_W < new_W ? old_W : new_W;
            memcpy(new_pres, old_pres, (size_t)copy_W * sizeof(uint64_t));
            memcpy(new_conf, old_conf, (size_t)copy_W * sizeof(uint64_t));

            /* Payload: copy old_cap * stride bytes into new payload region.
               New columns at [old_cap..new_cap) stay uninitialized
               (bitmap presence governs reads). */
            uint8_t* new_payload = new_row + IN_HDR_BYTES + (size_t)new_W * 16u;
            const uint8_t* old_payload = old_row + IN_HDR_BYTES + (size_t)old_W * 16u;
            uint32_t copy_cap = old_cap < new_cap ? old_cap : new_cap;
            memcpy(new_payload, old_payload, (size_t)copy_cap * (size_t)it->stride);
        }
        ecs_free(it->table);
    }
    it->table         = new_table;
    it->row_bytes     = new_row_bytes;
    it->words_per_row = new_W;
    it->active_cap    = new_cap;
    it->buf_size      = new_buf_size;
    it->buf_mask      = new_buf_size - 1u;
}

static void in_grow_active_cap(ecs_input_t* it, uint32_t need) {
    uint32_t old_cap = it->active_cap;
    uint32_t new_cap = old_cap ? old_cap : ECS_INPUT_PLAYER_CAP_INIT;
    while (new_cap < need) new_cap <<= 1;
    if (new_cap == old_cap) return;

    /* Dense bookkeeping: realloc + init NIL for new entries. */
    it->dense_ids = (uint32_t*)ecs_xrealloc(it->dense_ids,
                        (size_t)new_cap * sizeof(*it->dense_ids));
    for (uint32_t i = old_cap; i < new_cap; i++) {
        it->dense_ids[i] = ECS_INPUT_PID_NIL;
    }

    in_realloc_table(it, new_cap, it->buf_size);
}

static void in_grow_buf_size(ecs_input_t* it, uint32_t new_buf_size) {
    assert(new_buf_size > it->buf_size);
    assert((new_buf_size & (new_buf_size - 1u)) == 0u);
    assert(new_buf_size <= ECS_INPUT_BUFSIZE_MAX);
    in_realloc_table(it, it->active_cap ? it->active_cap : 0u, new_buf_size);
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
       register grows active_cap. row_bytes = padded(8 + 0 + 0) = 8. */
    it->row_bytes     = in_compute_row_bytes(0u, 0u, stride);
    size_t total      = (size_t)buf_size * (size_t)it->row_bytes;
    it->table         = (uint8_t*)ecs_xmalloc_aligned(total, 8);
    for (uint32_t i = 0; i < buf_size; i++) {
        *(uint64_t*)(it->table + (size_t)i * it->row_bytes) = ECS_INPUT_TICK_NIL;
    }

    it->confirmed_frontier = 0ull;
}

void ecs_input_destroy(ecs_input_t* it) {
    if (!it) return;
    if (it->table)     ecs_free(it->table);
    if (it->dense_ids) ecs_free(it->dense_ids);
    memset(it, 0, sizeof(*it));
}

/* ==========================================================================
   Membership
   ========================================================================== */

bool ecs_input_register_player(ecs_input_t* it, ecs_pid_t pid) {
    assert(it);
    if (pid >= ECS_INPUT_PID_MAX) return false;
    if (it->active_count >= ECS_INPUT_PLAYER_MAX) return false;

    if (in_pid_lookup(it, pid) != ECS_INPUT_PID_NIL) return false;

    if (it->active_cap == 0u || it->dense_high >= it->active_cap) {
        in_grow_active_cap(it, it->dense_high + 1u);
    }
    uint32_t idx = in_scan_dense_ids(it, ECS_INPUT_PID_NIL);
    assert(idx != ECS_INPUT_PID_NIL);

    bool reused = (idx < it->dense_high);
    if (!reused) it->dense_high = idx + 1u;

    /* ABA defense: clear stale bits at idx in any still-live tick row. */
    if (reused && it->words_per_row) {
        uint32_t W   = it->words_per_row;
        uint32_t word= idx >> 6;
        uint64_t bit = 1ull << (idx & 63u);
        uint64_t notbit = ~bit;
        for (uint32_t t = 0; t < it->buf_size; t++) {
            uint8_t* row = in_row_ptr(it, t);
            if (*in_row_tick(row) == ECS_INPUT_TICK_NIL) continue;
            uint64_t* pres = in_row_present(row);
            uint64_t* conf = in_row_confirmed(row, W);
            pres[word] &= notbit;
            conf[word] &= notbit;
        }
    }

    it->dense_ids[idx] = pid;
    it->active_count++;
    return true;
}

void ecs_input_unregister_player(ecs_input_t* it, ecs_pid_t pid) {
    assert(it);
    uint32_t idx = in_pid_lookup(it, pid);
    if (idx == ECS_INPUT_PID_NIL) return;
    it->dense_ids[idx] = ECS_INPUT_PID_NIL;
    it->active_count--;
}

/* ==========================================================================
   Hot path
   ========================================================================== */

void ecs_input_set(ecs_input_t* it, uint64_t tick, ecs_pid_t pid,
                   const void* value, bool confirmed)
{
    assert(it);
    assert(value);
    assert(tick != ECS_INPUT_TICK_NIL);
    assert(tick >= 1ull);   /* tick 0 reserved as "no frontier" sentinel */

    uint32_t idx = in_pid_lookup(it, pid);
    if (idx == ECS_INPUT_PID_NIL) return;

    /* Auto-grow ring if first-touch would clobber a still-live slot. */
    {
        uint32_t t0   = (uint32_t)(tick & it->buf_mask);
        uint8_t* row0 = in_row_ptr(it, t0);
        uint64_t prev = *in_row_tick(row0);
        bool would_evict = (prev != ECS_INPUT_TICK_NIL && prev != tick &&
                            prev > it->confirmed_frontier);
        if (would_evict) {
            uint64_t diff = (tick > prev) ? (tick - prev) : (prev - tick);
            if (diff < (uint64_t)ECS_INPUT_BUFSIZE_MAX) {
                uint32_t need = (uint32_t)diff + 1u;
                uint32_t ns   = it->buf_size;
                while (ns < need) ns <<= 1;
                if (ns <= ECS_INPUT_BUFSIZE_MAX) in_grow_buf_size(it, ns);
            }
        }
    }

    uint32_t t  = (uint32_t)(tick & it->buf_mask);
    uint32_t W  = it->words_per_row;
    uint8_t* row = in_row_ptr(it, t);

    if (*in_row_tick(row) != tick) {
        assert(tick > it->confirmed_frontier);
        in_advance_row(it, t, tick);
        *in_row_tick(row) = tick;
    }

    uint64_t* conf = in_row_confirmed(row, W);
    uint64_t* pres = in_row_present(row);
    uint64_t  bit  = 1ull << (idx & 63u);
    uint32_t  word = idx >> 6;
    bool was_confirmed = (conf[word] & bit) != 0ull;

    if (!confirmed && was_confirmed) return;

    memcpy(in_row_cell(row, W, idx, it->stride), value, it->stride);
    pres[word] |= bit;

    if (confirmed && !was_confirmed) {
        conf[word] |= bit;
        if (in_tick_all_confirmed(it, t)) {
            in_advance_frontier(it, tick);
        }
    }
}

const void* ecs_input_get(const ecs_input_t* it, uint64_t tick, ecs_pid_t pid) {
    uint32_t idx = in_pid_lookup(it, pid);
    if (idx == ECS_INPUT_PID_NIL) return NULL;
    uint32_t t  = (uint32_t)(tick & it->buf_mask);
    uint8_t* row = in_row_ptr(it, t);
    if (*in_row_tick(row) != tick) return NULL;
    uint32_t W  = it->words_per_row;
    if (!W) return NULL;
    const uint64_t* pres = in_row_present(row);
    if (!((pres[idx >> 6] >> (idx & 63u)) & 1u)) return NULL;
    return in_row_cell(row, W, idx, it->stride);
}

ecs_input_view_t ecs_input_get_view(const ecs_input_t* it,
                                    uint64_t tick, ecs_pid_t pid)
{
    ecs_input_view_t v = { NULL, false, false };
    uint32_t idx = in_pid_lookup(it, pid);
    if (idx == ECS_INPUT_PID_NIL) return v;

    uint32_t t  = (uint32_t)(tick & it->buf_mask);
    uint32_t W  = it->words_per_row;
    uint8_t* row = in_row_ptr(it, t);
    if (*in_row_tick(row) != tick || !W) return v;

    const uint64_t* pres = in_row_present(row);
    const uint64_t* conf = in_row_confirmed(row, W);
    v.present   = (pres[idx >> 6] >> (idx & 63u)) & 1u;
    v.confirmed = (conf[idx >> 6] >> (idx & 63u)) & 1u;
    if (v.present) {
        v.data = in_row_cell(row, W, idx, it->stride);
    }
    return v;
}

void ecs_input_clear(ecs_input_t* it, uint64_t tick) {
    assert(it);
    assert(tick != ECS_INPUT_TICK_NIL);
    assert(tick >= 1ull);
    uint32_t t = (uint32_t)(tick & it->buf_mask);
    in_reset_row(it, t);

    /* If cleared tick was within confirmed range, rewind frontier to
       tick - 1 (which may be 0 = no frontier). */
    if (tick <= it->confirmed_frontier) {
        it->confirmed_frontier = tick - 1ull;
    }
}

bool ecs_input_tick_confirmed(const ecs_input_t* it, uint64_t tick) {
    uint32_t t = (uint32_t)(tick & it->buf_mask);
    uint8_t* row = in_row_ptr(it, t);
    if (*in_row_tick(row) != tick) return false;
    return in_tick_all_confirmed(it, t);
}

uint64_t ecs_input_frontier(const ecs_input_t* it) {
    return it->confirmed_frontier;
}

void ecs_input_seed_frontier(ecs_input_t* it, uint64_t tick) {
    assert(it);
    assert(tick != ECS_INPUT_TICK_NIL);
    assert(tick >= 1ull);
    it->confirmed_frontier = tick;
}

void ecs_input_seal_empty_tick(ecs_input_t* it, uint64_t tick) {
    assert(it);
    assert(tick != ECS_INPUT_TICK_NIL);
    assert(tick >= 1ull);
    uint32_t t = (uint32_t)(tick & it->buf_mask);
    uint8_t* row = in_row_ptr(it, t);

    if (*in_row_tick(row) != tick) {
        in_reset_row(it, t);
        *in_row_tick(row) = tick;
    } else {
        /* Caller bug: sealing as empty a tick that already has writes. */
        uint32_t W = it->words_per_row;
        if (W) {
            uint64_t* conf = in_row_confirmed(row, W);
            for (uint32_t w = 0; w < W; w++) assert(conf[w] == 0ull);
        }
    }
    in_advance_frontier(it, tick);
}

bool ecs_input_is_registered(const ecs_input_t* it, ecs_pid_t pid) {
    return in_pid_lookup(it, pid) != ECS_INPUT_PID_NIL;
}

bool ecs_input_iter_next(ecs_input_iter_t* iter) {
    assert(iter && iter->it);
    const ecs_input_t* it = iter->it;
    uint32_t i = iter->cursor;
    while (i < it->dense_high) {
        if (it->dense_ids[i] != ECS_INPUT_PID_NIL) {
            iter->dense_idx = i;
            iter->pid       = it->dense_ids[i];
            iter->cursor    = i + 1u;
            return true;
        }
        i++;
    }
    iter->cursor    = i;
    iter->pid       = ECS_INPUT_PID_NIL;
    return false;
}
