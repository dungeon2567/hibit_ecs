#include "input.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "ecs.h"   /* ecs_xmalloc_aligned, ecs_xcalloc, ecs_xrealloc, ecs_free */

/* ==========================================================================
   Internal helpers
   ========================================================================== */

static inline uint32_t in_pid_hi(ecs_pid_t pid) { return pid >> ECS_INPUT_L2_BITS; }
static inline uint32_t in_pid_lo(ecs_pid_t pid) { return pid & (ECS_INPUT_L2_SIZE - 1u); }

static inline uint32_t in_words_for_cap(uint32_t cap) {
    return (cap + 63u) >> 6;
}

static inline uint64_t* in_row(uint64_t* bm, uint64_t t, uint32_t w) {
    return bm + (size_t)t * (size_t)w;
}

/* pid_page allocator. Pages are zero-init'd to PID_NIL via memset to 0xFF. */
static ecs_input_pid_page_t* in_alloc_page(void) {
    ecs_input_pid_page_t* p = (ecs_input_pid_page_t*)ecs_xcalloc(1, sizeof(*p));
    memset(p->entry, 0xFF, sizeof(p->entry));   /* every entry = PID_NIL */
    p->used = 0;
    return p;
}

static inline uint32_t in_pid_lookup(const ecs_input_t* it, ecs_pid_t pid) {
    if (pid >= ECS_INPUT_PID_MAX) return ECS_INPUT_PID_NIL;
    const ecs_input_pid_page_t* p = it->l1[in_pid_hi(pid)];
    if (!p) return ECS_INPUT_PID_NIL;
    return p->entry[in_pid_lo(pid)];
}

/* Reset all per-tick scalars + bitmap row for ring slot t (where t is
   already (tick & buf_mask), so it fits in 32 bits). Does NOT touch
   frontier (caller decides). Used both on first-touch and on
   ecs_input_clear. */
static void in_reset_row(ecs_input_t* it, uint32_t t) {
    uint32_t w = it->words_per_row;
    if (w) {
        memset(in_row(it->confirmed_bm, (uint64_t)t, w), 0, (size_t)w * sizeof(uint64_t));
        memset(in_row(it->present_bm,   (uint64_t)t, w), 0, (size_t)w * sizeof(uint64_t));
    }
    it->confirmed_count[t] = 0;
    it->expected_count[t]  = 0;
    it->all_confirmed_bm[t >> 6] &= ~(1ull << (t & 63u));
}

/* Greedily advance confirmed_frontier as long as the next tick's
   all_confirmed bit is set. Stops at any gap.

   Seeding rule: with no prior frontier, the only tick that can start
   the run is tick 0 (full-confirmation of every tick from 0 upward).
   Mid-session joins must call ecs_input_seed_frontier explicitly to
   skip past unobservable history. */
static void in_advance_frontier(ecs_input_t* it, uint64_t just_confirmed_tick) {
    if (!it->frontier_valid) {
        if (just_confirmed_tick != 0ull) return;
        it->frontier_valid     = true;
        it->confirmed_frontier = 0ull;
    } else if (just_confirmed_tick == it->confirmed_frontier + 1ull) {
        it->confirmed_frontier = just_confirmed_tick;
    } else {
        return;
    }
    /* Walk forward across any already-set bits. Bound: buf_size. */
    for (uint32_t i = 0; i < it->buf_size; i++) {
        uint64_t n = it->confirmed_frontier + 1ull;
        uint32_t t = (uint32_t)(n & it->buf_mask);
        if (it->tick_in_slot[t] != n) break;
        if (!((it->all_confirmed_bm[t >> 6] >> (t & 63u)) & 1u)) break;
        it->confirmed_frontier = n;
    }
}

/* ==========================================================================
   active_cap growth -- pow2. Reallocates dense + dense_ids + free_next +
   the per-tick bitmaps (rows migrated, since words_per_row changes).
   ========================================================================== */

static void in_grow_active_cap(ecs_input_t* it, uint32_t need) {
    uint32_t old_cap = it->active_cap;
    uint32_t new_cap = old_cap ? old_cap : 8u;
    while (new_cap < need) new_cap <<= 1;

    /* Dense arrays. */
    it->dense     = (ecs_input_buf_t*)ecs_xrealloc(it->dense,     (size_t)new_cap * sizeof(*it->dense));
    it->dense_ids = (uint32_t*)       ecs_xrealloc(it->dense_ids, (size_t)new_cap * sizeof(*it->dense_ids));
    it->free_next = (uint32_t*)       ecs_xrealloc(it->free_next, (size_t)new_cap * sizeof(*it->free_next));

    for (uint32_t i = old_cap; i < new_cap; i++) {
        it->dense[i].slots = NULL;
        it->dense_ids[i]   = ECS_INPUT_PID_NIL;
        it->free_next[i]   = ECS_INPUT_FREE_NIL;
    }

    /* Bitmap rows. words_per_row changes -- can't single-block realloc. */
    uint32_t old_w = it->words_per_row;
    uint32_t new_w = in_words_for_cap(new_cap);

    if (new_w != old_w) {
        size_t total = (size_t)it->buf_size * (size_t)new_w * sizeof(uint64_t);
        uint64_t* new_conf = (uint64_t*)ecs_xcalloc(1, total);
        uint64_t* new_pres = (uint64_t*)ecs_xcalloc(1, total);
        if (it->confirmed_bm && old_w) {
            for (uint32_t t = 0; t < it->buf_size; t++) {
                memcpy(new_conf + (size_t)t * new_w,
                       it->confirmed_bm + (size_t)t * old_w,
                       (size_t)old_w * sizeof(uint64_t));
                memcpy(new_pres + (size_t)t * new_w,
                       it->present_bm + (size_t)t * old_w,
                       (size_t)old_w * sizeof(uint64_t));
            }
        }
        if (it->confirmed_bm) ecs_free(it->confirmed_bm);
        if (it->present_bm)   ecs_free(it->present_bm);
        it->confirmed_bm = new_conf;
        it->present_bm   = new_pres;
        it->words_per_row = new_w;
    }

    it->active_cap = new_cap;
}

/* ==========================================================================
   Lifecycle
   ========================================================================== */

void ecs_input_init(ecs_input_t* it, uint32_t stride, uint32_t buf_size) {
    assert(it);
    assert(stride > 0);
    assert(buf_size > 0 && (buf_size & (buf_size - 1u)) == 0u);   /* pow2 */

    memset(it, 0, sizeof(*it));
    it->stride   = stride;
    it->buf_size = buf_size;
    it->buf_mask = buf_size - 1u;

    /* Per-tick scalars + all_confirmed bitmap. */
    it->confirmed_count  = (uint16_t*)ecs_xcalloc(buf_size, sizeof(uint16_t));
    it->expected_count   = (uint16_t*)ecs_xcalloc(buf_size, sizeof(uint16_t));
    it->tick_in_slot     = (uint64_t*)ecs_xmalloc_aligned((size_t)buf_size * sizeof(uint64_t), 8);
    for (uint32_t i = 0; i < buf_size; i++) it->tick_in_slot[i] = ECS_INPUT_TICK_NIL;

    uint32_t all_words = (buf_size + 63u) >> 6;
    it->all_confirmed_bm = (uint64_t*)ecs_xcalloc(all_words, sizeof(uint64_t));

    it->free_head        = ECS_INPUT_FREE_NIL;
    it->frontier_valid   = false;
    it->confirmed_frontier = 0ull;
    it->dense_high       = 0;

    /* Dense / bitmap rows allocated lazily on first register. */
}

void ecs_input_destroy(ecs_input_t* it) {
    if (!it) return;

    /* Player slot rings + radix pages. */
    for (uint32_t hi = 0; hi < ECS_INPUT_L1_SIZE; hi++) {
        if (it->l1[hi]) { ecs_free(it->l1[hi]); it->l1[hi] = NULL; }
    }
    if (it->dense) {
        for (uint32_t i = 0; i < it->active_cap; i++) {
            if (it->dense[i].slots) ecs_free(it->dense[i].slots);
        }
        ecs_free(it->dense);
    }
    if (it->dense_ids)        ecs_free(it->dense_ids);
    if (it->free_next)        ecs_free(it->free_next);
    if (it->confirmed_bm)     ecs_free(it->confirmed_bm);
    if (it->present_bm)       ecs_free(it->present_bm);
    if (it->confirmed_count)  ecs_free(it->confirmed_count);
    if (it->expected_count)   ecs_free(it->expected_count);
    if (it->tick_in_slot)     ecs_free(it->tick_in_slot);
    if (it->all_confirmed_bm) ecs_free(it->all_confirmed_bm);

    memset(it, 0, sizeof(*it));
}

/* ==========================================================================
   Membership
   ========================================================================== */

bool ecs_input_register_player(ecs_input_t* it, ecs_pid_t pid) {
    assert(it);
    if (pid >= ECS_INPUT_PID_MAX) return false;

    uint32_t hi = in_pid_hi(pid);
    uint32_t lo = in_pid_lo(pid);

    ecs_input_pid_page_t* page = it->l1[hi];
    if (!page) {
        page = in_alloc_page();
        it->l1[hi] = page;
    } else if (page->entry[lo] != ECS_INPUT_PID_NIL) {
        return false;   /* already registered */
    }

    /* Acquire dense idx: free list first, else extend high-water, growing
       cap as needed. */
    uint32_t idx;
    bool reused = false;
    if (it->free_head != ECS_INPUT_FREE_NIL) {
        idx = it->free_head;
        it->free_head = it->free_next[idx];
        it->free_next[idx] = ECS_INPUT_FREE_NIL;
        reused = true;
    } else {
        if (it->dense_high >= it->active_cap) {
            in_grow_active_cap(it, it->dense_high + 1u);
        }
        idx = it->dense_high++;
    }

    /* ABA defense: a recycled idx may carry stale bits from a previous
       owner in any still-live tick row (tick_in_slot != NIL). Past-row
       bits are otherwise meaningless for the new pid, and would surface
       false present/confirmed flags through get_view. Cost: O(buf_size)
       per re-register, only when the free list pops. */
    if (reused && it->words_per_row) {
        uint32_t w     = it->words_per_row;
        uint32_t word  = idx >> 6;
        uint64_t bit   = 1ull << (idx & 63u);
        uint64_t notbit= ~bit;
        for (uint32_t t = 0; t < it->buf_size; t++) {
            if (it->tick_in_slot[t] == ECS_INPUT_TICK_NIL) continue;
            uint64_t* conf = in_row(it->confirmed_bm, t, w);
            uint64_t* pres = in_row(it->present_bm,   t, w);
            if (conf[word] & bit) {
                conf[word] &= notbit;
                /* Only adjust live counts; past rows are sealed but the
                   counter decrement is harmless and keeps the invariant
                   confirmed_count[t] == popcount(confirmed_bm row at t). */
                if (it->confirmed_count[t]) it->confirmed_count[t]--;
            }
            pres[word] &= notbit;
        }
    }

    /* Allocate the player's slot ring. */
    if (!it->dense[idx].slots) {
        size_t sz = (size_t)it->buf_size * (size_t)it->stride;
        it->dense[idx].slots = (uint8_t*)ecs_xmalloc_aligned(sz, 8);
        memset(it->dense[idx].slots, 0, sz);
    } else {
        memset(it->dense[idx].slots, 0, (size_t)it->buf_size * (size_t)it->stride);
    }
    it->dense_ids[idx] = pid;

    page->entry[lo] = idx;
    page->used++;
    it->active_count++;
    return true;
}

void ecs_input_unregister_player(ecs_input_t* it, ecs_pid_t pid) {
    assert(it);
    if (pid >= ECS_INPUT_PID_MAX) return;

    uint32_t hi = in_pid_hi(pid);
    uint32_t lo = in_pid_lo(pid);

    ecs_input_pid_page_t* page = it->l1[hi];
    if (!page) return;
    uint32_t idx = page->entry[lo];
    if (idx == ECS_INPUT_PID_NIL) return;

    page->entry[lo] = ECS_INPUT_PID_NIL;
    if (--page->used == 0) {
        ecs_free(page);
        it->l1[hi] = NULL;
    }

    /* Free the player's slot ring. Caller must have already drained any
       in-flight rollback window. */
    if (it->dense[idx].slots) {
        ecs_free(it->dense[idx].slots);
        it->dense[idx].slots = NULL;
    }
    it->dense_ids[idx] = ECS_INPUT_PID_NIL;

    /* Push idx onto free list. Bits at idx in past tick bitmap rows are
       intentionally preserved -- with server-authoritative expected_count
       those rows are sealed and the stale bit is correct for the leaver's
       last live tick. */
    it->free_next[idx] = it->free_head;
    it->free_head      = idx;
    it->active_count--;
}

/* ==========================================================================
   Hot path
   ========================================================================== */

void ecs_input_set(ecs_input_t* it, uint64_t tick, ecs_pid_t pid,
                   const void* value, bool confirmed, uint16_t expected)
{
    uint32_t idx = in_pid_lookup(it, pid);
    if (idx == ECS_INPUT_PID_NIL) return;

    uint32_t t = (uint32_t)(tick & it->buf_mask);
    uint32_t w = it->words_per_row;

    /* First-touch: ring slot newly assigned to this tick. */
    if (it->tick_in_slot[t] != tick) {
        in_reset_row(it, t);
        it->tick_in_slot[t]   = tick;
        it->expected_count[t] = expected;
    } else {
        /* Subsequent packet for same tick: server-authoritative refresh. */
        it->expected_count[t] = expected;
    }

    /* Slot bytes. */
    memcpy(it->dense[idx].slots + (size_t)t * (size_t)it->stride,
           value, it->stride);

    /* present bit (always set on write). */
    uint64_t* pres = in_row(it->present_bm, t, w);
    pres[idx >> 6] |= 1ull << (idx & 63u);

    if (confirmed) {
        uint64_t* conf = in_row(it->confirmed_bm, t, w);
        uint64_t  bit  = 1ull << (idx & 63u);
        if (!(conf[idx >> 6] & bit)) {
            conf[idx >> 6] |= bit;
            uint16_t c = ++it->confirmed_count[t];
            if (c == it->expected_count[t]) {
                it->all_confirmed_bm[t >> 6] |= 1ull << (t & 63u);
                in_advance_frontier(it, tick);
            }
        }
    }

    /* Edge: expected may be 0 (no players in roster for this tick).
       Mark all_confirmed immediately on first-touch so frontier can pass.
       Re-check after any expected update. */
    if (it->expected_count[t] == 0 && it->confirmed_count[t] == 0) {
        if (!((it->all_confirmed_bm[t >> 6] >> (t & 63u)) & 1u)) {
            it->all_confirmed_bm[t >> 6] |= 1ull << (t & 63u);
            in_advance_frontier(it, tick);
        }
    }
}

const void* ecs_input_get(const ecs_input_t* it, uint64_t tick, ecs_pid_t pid) {
    uint32_t idx = in_pid_lookup(it, pid);
    if (idx == ECS_INPUT_PID_NIL) return NULL;
    uint32_t t = (uint32_t)(tick & it->buf_mask);
    return it->dense[idx].slots + (size_t)t * (size_t)it->stride;
}

ecs_input_view_t ecs_input_get_view(const ecs_input_t* it,
                                    uint64_t tick, ecs_pid_t pid)
{
    ecs_input_view_t v = { NULL, false, false };
    uint32_t idx = in_pid_lookup(it, pid);
    if (idx == ECS_INPUT_PID_NIL) return v;

    uint32_t t = (uint32_t)(tick & it->buf_mask);
    uint32_t w = it->words_per_row;

    v.data = it->dense[idx].slots + (size_t)t * (size_t)it->stride;

    if (it->tick_in_slot[t] == tick && w) {
        const uint64_t* pres = it->present_bm   + (size_t)t * w;
        const uint64_t* conf = it->confirmed_bm + (size_t)t * w;
        v.present   = (pres[idx >> 6] >> (idx & 63u)) & 1u;
        v.confirmed = (conf[idx >> 6] >> (idx & 63u)) & 1u;
    }
    return v;
}

void ecs_input_clear(ecs_input_t* it, uint64_t tick) {
    assert(it);
    uint32_t t = (uint32_t)(tick & it->buf_mask);
    in_reset_row(it, t);
    it->tick_in_slot[t] = ECS_INPUT_TICK_NIL;

    /* Frontier rewind if this tick was within the confirmed range. */
    if (it->frontier_valid && tick <= it->confirmed_frontier) {
        if (tick == 0ull) {
            it->frontier_valid = false;
            it->confirmed_frontier = 0ull;
        } else {
            it->confirmed_frontier = tick - 1ull;
        }
    }
}

bool ecs_input_tick_confirmed(const ecs_input_t* it, uint64_t tick) {
    uint32_t t = (uint32_t)(tick & it->buf_mask);
    if (it->tick_in_slot[t] != tick) return false;
    return ((it->all_confirmed_bm[t >> 6] >> (t & 63u)) & 1u) != 0u;
}

uint64_t ecs_input_frontier(const ecs_input_t* it, bool* out_valid) {
    if (out_valid) *out_valid = it->frontier_valid;
    return it->frontier_valid ? it->confirmed_frontier : 0ull;
}

void ecs_input_seed_frontier(ecs_input_t* it, uint64_t tick) {
    assert(it);
    it->frontier_valid     = true;
    it->confirmed_frontier = tick;
}
