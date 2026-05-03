#include "ecs.h"
#include <stddef.h>

ecs_l1_t ecs_default_l1 = {
    .confirmed_mask_any = 0,
    .predicted_mask_any = 0,
    .dirty              = 0,
    .changed            = 0
};

ecs_l2_t ecs_default_l2 = {
    .confirmed_mask_any  = 0,
    .confirmed_mask_all  = 0,
    .predicted_mask_any  = 0,
    .predicted_mask_all  = 0,
    .dirty               = 0,
    .changed             = 0,
    .children = {
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1,
        &ecs_default_l1, &ecs_default_l1, &ecs_default_l1, &ecs_default_l1
    }
};

/* ==========================================================================
   Iterator
   ========================================================================== */

/* Mutable scratch nodes used as initial targets for it->l1[*] / it->l2[*]
   in ecs_iterator_init. Lets the first ecs_iterator_next_block call run its
   flush + L1->L2 + L2->L3 propagation unconditionally (no first-call sentinel)
   while writing only zeros: scratch->changed is zero, so every `OR (ch != 0)`
   resolves to OR-with-zero. Engine is single-threaded; sharing across
   iterators is safe because no real flush ever stores a non-zero bit here
   (l1[*] / l2[*] are overwritten with real nodes on the first descent before
   any iter_get_mut write can occur).

   Kept separate from ecs_default_l1 / ecs_default_l2 so writers can't
   accidentally pollute the read-only defaults that every empty subtree
   shares (children[] arrays).  */
static ecs_l1_t iter_scratch_l1;        /* zero-init, module-private */
static ecs_l2_t iter_scratch_l2;        /* zero-init, module-private */

static uint64_t iter_compute_l3_mask(const ecs_compiled_query_t* q,
                                     const ecs_l3_t* const l3[]) {
    uint64_t result = 0;
    for (uint32_t c = 0; c < q->clause_count; c++) {
        const ecs_compiled_clause_t* cl = &q->clauses[c];
        if (!cl->include && !cl->exclude && !cl->changed) continue;

        uint64_t bits = ~0ULL;
        uint32_t mask = cl->include;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= l3[n]->predicted_mask_any; }
        /* Exclude prune: drop subtree where every slot has the term
           (mask_all bit set). Partial subtrees fall through to L1 for exact
           filtering. */
        mask = cl->exclude;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= ~l3[n]->predicted_mask_all; }
        if (cl->changed) {
            uint64_t any_changed = 0;
            mask = cl->changed;
            while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; any_changed |= l3[n]->changed; }
            bits &= any_changed;
        }
        result |= bits;
    }
    return result;
}

static uint64_t iter_compute_l2_mask(const ecs_compiled_query_t* q,
                                     const ecs_l2_t* const l2[]) {
    uint64_t result = 0;
    for (uint32_t c = 0; c < q->clause_count; c++) {
        const ecs_compiled_clause_t* cl = &q->clauses[c];
        if (!cl->include && !cl->exclude && !cl->changed) continue;
        uint64_t bits = ~0ULL;
        uint32_t mask = cl->include;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= l2[n]->predicted_mask_any; }
        mask = cl->exclude;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= ~l2[n]->predicted_mask_all; }
        if (cl->changed) {
            uint64_t any_changed = 0;
            mask = cl->changed;
            while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; any_changed |= l2[n]->changed; }
            bits &= any_changed;
        }
        result |= bits;
    }
    return result;
}

static uint64_t iter_compute_l1_mask(const ecs_compiled_query_t* q,
                                     const ecs_l1_t* const l1[]) {
    uint64_t result = 0;
    for (uint32_t c = 0; c < q->clause_count; c++) {
        const ecs_compiled_clause_t* cl = &q->clauses[c];
        if (!cl->include && !cl->exclude && !cl->changed) continue;

        uint64_t bits = ~0ULL;
        uint32_t mask = cl->include;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= l1[n]->predicted_mask_any; }
        mask = cl->exclude;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= ~l1[n]->predicted_mask_any; }
        if (cl->changed) {
            uint64_t any_changed = 0;
            mask = cl->changed;
            while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; any_changed |= l1[n]->changed; }
            bits &= any_changed;
        }
        result |= bits;
    }
    return result;
}

static inline void iter_load_l1(ecs_iterator_t* it, uint32_t tree_count) {
    for (uint32_t i = 0; i < tree_count; i++) {
        ecs_l1_t* l1 = (ecs_l1_t*)it->l2[i]->children[it->l2_idx];
        /* Prefetch L1 mask line. Address already in hand (loaded above), no
           extra work. Goal isn't to hide a single demand miss -- it's to
           expose DRAM parallelism: iter_compute_l1_mask reads l1[n]->mask
           through a serial `bits &= ...` AND chain, which CPU can't reorder.
           Issuing N prefetches here lets the misses fetch concurrently via
           LFBs instead of serializing. */
        ECS_PREFETCH(l1);
        ((ecs_l1_t**)it->l1)[i]  = l1;                                /* fields const post-init */
        ((void**)it->l1_data)[i] = (char*)l1 + sizeof(ecs_l1_t);
    }
}

void ecs_iterator_init(ecs_iterator_t* it, const ecs_compiled_query_t* query, uint32_t write_mask) {
    it->query      = query;
    it->l2_mask    = 0;
    /* No first-call sentinel. l*_idx start at 0 (any valid bit is fine);
       the first ecs_iterator_next_block call's flush / propagation runs
       unconditionally and writes only zeros to iter_scratch_l1 / l2 / real
       l3 until next_l2 swaps in real l1 / l2 pointers via iter_load_l1 and
       the L3 advance. */
    it->l3_idx     = 0;
    it->l2_idx     = 0;
    /* write_mask, mode, world, world_tree_idx: const post-init â€” written via cast.
       Same idiom as l1 / l1_data / data_size below. */
    *(uint32_t*)&it->write_mask      = write_mask;
    *(ecs_mode_t*)&it->mode          = query->tree_count ? query->trees[0]->mode : ECS_MODE_CONFIRMED;
    *(const ecs_world_t**)&it->world = query->world;

#ifndef NDEBUG
    /* `changed` clauses read tree->changed bitmap, which is per-tick. Require
       ecs_world_begin_tick (or ecs_tree_begin_tick on every involved tree) to
       have run this tick â€” i.e. tree->tick_id_at_begin == world->tick_id.
       Debug-only: hot path skips the scan in Release. */
    {
        uint32_t changed_terms = 0;
        for (uint32_t c = 0; c < query->clause_count; c++)
            changed_terms |= query->clauses[c].changed;
        if (changed_terms) {
            assert(query->world && "ecs_iterator_init: query has changed-clause but no world");
            uint64_t tick_id = query->world->tick_id;
            while (changed_terms) {
                int t = ecs_ctz32(changed_terms); changed_terms &= changed_terms - 1;
                assert(query->trees[t]->tick_id_at_begin == tick_id &&
                       "ecs_iterator_init: changed-clause used without ecs_world_begin_tick this tick");
            }
        }
    }
#endif

    /* Fused: world_tree_idx + per-slot pointer init in one pass. */
    const ecs_tree_t* base = query->world ? &query->world->trees[0] : NULL;
    uint32_t tc = query->tree_count;
    for (uint32_t i = 0; i < tc; i++) {
        ecs_tree_t* t = query->trees[i];
        assert(t->mode == it->mode &&
               "ecs_iterator_init: all query trees must share VM mode");
        if (base) {
            ptrdiff_t wt_idx = t - base;
            assert(wt_idx >= 0 && wt_idx < 64 &&
                   "ecs_iterator_init: query tree not in query->world->trees[]");
            ((uint8_t*)it->world_tree_idx)[i] = (uint8_t)wt_idx;
        } else {
            ((uint8_t*)it->world_tree_idx)[i] = 0;
        }
        it->l3[i]                   = t->root;
        it->l2[i]                   = &iter_scratch_l2;                  /* mutable scratch -- first flush ORs zero into it */
        ((ecs_l1_t**)it->l1)[i]     = &iter_scratch_l1;                  /* same; overwritten by first iter_load_l1 */
        ((void**)it->l1_data)[i]    = NULL;
        ((size_t*)it->data_size)[i] = t->data_size;                      /* field is const after init */
    }
    it->l3_mask = iter_compute_l3_mask(query, it->l3);
    /* L3->L2 descent and first L1 load are deferred to the first
       ecs_iterator_next_block call. */
}

uint64_t ecs_iterator_next_block(ecs_iterator_t* it) {
    const ecs_compiled_query_t* q  = it->query;
    int      predict = it->mode;
    uint32_t wm      = it->write_mask;

    /* FUSED block-boundary flush + L1->L2 propagation in one pass over
       write_mask trees. Flush realises mask updates that ecs_iterator_get_mut
       deferred (dirty / predicted_mask_any / confirmed_mask_any from
       l1->changed); propagation summarises the just-yielded L1's dirty/changed
       up to L2 at the prior l2_idx.

       No first-call gate: iter_init aims l1[*] / l2[*] at iter_scratch_l1 /
       iter_scratch_l2 (mutable, zero) so this loop's reads return zero and
       its writes are OR-with-zero on the first call. iter_load_l1 swaps in
       real pointers via the L3 / L2 descent before the first return. */
    {
        uint32_t bit = (uint32_t)it->l2_idx;
        uint32_t w   = wm;
        if (predict) {
            while (w) {
                uint32_t  t  = (uint32_t)ecs_ctz32(w); w &= w - 1;
                ecs_l1_t* l1 = it->l1[t];
                ecs_l2_t* l2 = (ecs_l2_t*)it->l2[t];
                uint64_t  ch = l1->changed;
                uint64_t  d  = l1->dirty | ch;                  /* flush dirty */
                l1->dirty               = d;
                l1->predicted_mask_any |= ch;
                l2->dirty   |= (uint64_t)(d  != 0) << bit;
                l2->changed |= (uint64_t)(ch != 0) << bit;
            }
        } else {
            /* CONFIRMED: dirty stays 0 everywhere (invariant). */
            while (w) {
                uint32_t  t  = (uint32_t)ecs_ctz32(w); w &= w - 1;
                ecs_l1_t* l1 = it->l1[t];
                ecs_l2_t* l2 = (ecs_l2_t*)it->l2[t];
                uint64_t  ch = l1->changed;
                l1->predicted_mask_any |= ch;
                l1->confirmed_mask_any |= ch;
                l2->changed |= (uint64_t)(ch != 0) << bit;
            }
        }
    }

next_l2:
    if (it->l2_mask) {
        it->l2_idx  = ecs_ctz64(it->l2_mask);
        it->l2_mask &= it->l2_mask - 1;
        iter_load_l1(it, q->tree_count);
        uint64_t mask = iter_compute_l1_mask(q, (const ecs_l1_t* const*)it->l1);
        if (mask) return mask;
        /* Empty L1 (query include filtered out): propagate just-loaded block's
           dirty/changed up to L2 at new l2_idx. Other-path writers (tree_get_mut)
           already eager-propagate, so this is idempotent for them; covers the
           case where iter_get_mut wrote prior blocks of the same L2. */
        uint32_t bit = (uint32_t)it->l2_idx;
        uint32_t w   = wm;
        if (predict) {
            while (w) {
                uint32_t t = (uint32_t)ecs_ctz32(w); w &= w - 1;
                ecs_l1_t* l1 = it->l1[t];
                ecs_l2_t* l2 = (ecs_l2_t*)it->l2[t];
                l2->dirty   |= (uint64_t)(l1->dirty   != 0) << bit;
                l2->changed |= (uint64_t)(l1->changed != 0) << bit;
            }
        } else {
            while (w) {
                uint32_t t = (uint32_t)ecs_ctz32(w); w &= w - 1;
                ecs_l2_t* l2 = (ecs_l2_t*)it->l2[t];
                l2->changed |= (uint64_t)(it->l1[t]->changed != 0) << bit;
            }
        }
        goto next_l2;
    }

    /* L2->L3 propagation. No first-call gate: l2[*] still points at
       iter_scratch_l2 (zero) on the very first call, so the OR resolves to
       OR-with-zero against real l3[*]->dirty/changed at l3_idx=0 (init dummy)
       -- a no-op write. After the L3 advance below replaces l3_idx with a
       real bit, subsequent calls see real l2[*] and propagate properly. */
    {
        uint32_t bit = (uint32_t)it->l3_idx;
        uint32_t w   = wm;
        if (predict) {
            while (w) {
                uint32_t t = (uint32_t)ecs_ctz32(w); w &= w - 1;
                const ecs_l2_t* l2 = it->l2[t];
                ecs_l3_t* l3 = (ecs_l3_t*)it->l3[t];
                l3->dirty   |= (uint64_t)(l2->dirty   != 0) << bit;
                l3->changed |= (uint64_t)(l2->changed != 0) << bit;
            }
        } else {
            while (w) {
                uint32_t t = (uint32_t)ecs_ctz32(w); w &= w - 1;
                const ecs_l2_t* l2 = it->l2[t];
                ecs_l3_t* l3 = (ecs_l3_t*)it->l3[t];
                l3->changed |= (uint64_t)(l2->changed != 0) << bit;
            }
        }
    }

    if (it->l3_mask) {
        it->l3_idx  = ecs_ctz64(it->l3_mask);
        it->l3_mask &= it->l3_mask - 1;
        uint32_t tc = q->tree_count;
        for (uint32_t i = 0; i < tc; i++) {
            const ecs_l2_t* l2 = it->l3[i]->children[it->l3_idx];
            it->l2[i] = l2;
            /* Same DRAM-parallelism rationale as iter_load_l1's L1 prefetch:
               iter_compute_l2_mask reads l2[n]->masks through a serial AND chain. */
            ECS_PREFETCH(l2);
        }
        it->l2_mask = iter_compute_l2_mask(q, it->l2);
        goto next_l2;
    }
    return 0;
}

/* ==========================================================================
   Stage writes -- predicted side
   ========================================================================== */

/* Internal: acquire L2/L1 + propagate masks for a write at `index`. Sets all
   masks (predicted/confirmed/dirty/changed) unconditionally â€” caller is
   asserting that the slot is being written this frame, no eq-check. Returns
   l1s (used by ecs_tree_buffer_push for direct slot access pre-write). */
static ecs_l1_t* tree_acquire_and_mark(ecs_tree_t* tree, int index,
                                        int* out_l1_idx) {
    int l3_idx = (index >> 12) & 0x3F;
    int l2_idx = (index >>  6) & 0x3F;
    int l1_idx =  index        & 0x3F;

    uint64_t m         = (uint64_t)tree->mode;          /* 0 = CONFIRMED, 1 = PREDICT */
    uint64_t conf_mask = -(uint64_t)(1ULL - m);         /* ~0 in CONFIRMED, 0 in PREDICT */
    uint64_t bit1      = 1ULL << l1_idx;

    ecs_l2_t* l2s = tree->root->children[l3_idx];
    if (l2s == &ecs_default_l2) {
        l2s = ecs_l2_acquire(tree);
        tree->root->children[l3_idx] = l2s;
    }
    ecs_l1_t* l1s = l2s->children[l2_idx];
    if (l1s == &ecs_default_l1) {
        l1s = ecs_l1_acquire(tree);
        l2s->children[l2_idx] = l1s;
    }

    l1s->predicted_mask_any |= bit1;
    l1s->confirmed_mask_any |= bit1 & conf_mask;
    l1s->dirty              |= bit1 * m;
    l1s->changed            |= bit1;

    uint64_t bit2          = 1ULL << l2_idx;
    uint64_t l1_pred_full  = (uint64_t)(l1s->predicted_mask_any == ~0ULL) << l2_idx;
    uint64_t l1_conf_full  = (uint64_t)(l1s->confirmed_mask_any == ~0ULL) << l2_idx;
    l2s->predicted_mask_any |= bit2;
    l2s->predicted_mask_all |= l1_pred_full;
    l2s->confirmed_mask_any |= bit2 & conf_mask;
    l2s->confirmed_mask_all |= l1_conf_full;
    l2s->dirty              |= bit2 * m;
    l2s->changed            |= bit2;

    uint64_t bit3          = 1ULL << l3_idx;
    uint64_t l2_pred_full  = (uint64_t)(l2s->predicted_mask_all == ~0ULL) << l3_idx;
    uint64_t l2_conf_full  = (uint64_t)(l2s->confirmed_mask_all == ~0ULL) << l3_idx;
    tree->root->predicted_mask_any |= bit3;
    tree->root->predicted_mask_all |= l2_pred_full;
    tree->root->confirmed_mask_any |= bit3 & conf_mask;
    tree->root->confirmed_mask_all |= l2_conf_full;
    tree->root->dirty              |= bit3 * m;
    tree->root->changed            |= bit3;

    *out_l1_idx = l1_idx;
    return l1s;
}

/* Get mutable pointer to slot, marking it as written-this-frame. Acquires
   L2/L1 if absent. POD trees: returns pointer to writable slot (predicted
   slot in PREDICT mode, confirmed slot in CONFIRMED mode). Tag trees
   (data_size == 0): returns NULL but still updates masks.

   No eq-check: engine assumes caller is writing different bytes. Caller
   should only call when it knows the value is changing (skip when value
   would be identical). BUFFER trees must use ecs_tree_buffer_push/pop/clear. */
void* ecs_tree_get_mut(ecs_tree_t* tree, int index) {
    assert(tree);
    assert(index >= 0 && index < (1 << 18));
    assert(!(tree->flags & ECS_TREE_FLAG_BUFFER) &&
           "ecs_tree_get_mut: BUFFER trees must use ecs_tree_buffer_push/pop/clear");

    int l1_idx;
    ecs_l1_t* l1s = tree_acquire_and_mark(tree, index, &l1_idx);

    size_t ds = tree->data_size;
    if (ds == 0) return NULL;     /* tag */

    uint64_t m = (uint64_t)tree->mode;
    return (char*)l1s + sizeof(ecs_l1_t) + (size_t)(l1_idx + 64 * (int)m) * ds;
}

/* ==========================================================================
   LIST element-level mutation. Single COW point: when in PREDICT mode and the
   target slot's dirty bit is unset (first predict touch this tick), deep-copy
   the confirmed slot's buffer into the predicted slot. Subsequent ops in the
   same tick mutate the predicted buffer in place (realloc only on grow inside
   ecs_buffer_push). Confirmed buffer survives across CONFIRMED ticks (no free
   on overwrite).
   ========================================================================== */

/* Internal: returns the live ecs_buffer_t* for index, performing the COW if
   needed. Marks all masks (predicted/confirmed/dirty/changed) hierarchically
   via tree_acquire_and_mark_. `wants_confirmed_seed` controls whether the
   COW seeds the predicted buffer from the confirmed buffer (push/pop) or
   leaves it empty (clear). */
static ecs_buffer_t* buffer_live_(ecs_tree_t* tree, int index, int wants_confirmed_seed) {
    assert(tree && (tree->flags & ECS_TREE_FLAG_BUFFER));
    assert(index >= 0 && index < (1 << 18));
    assert(tree->data_size == sizeof(ecs_buffer_t));

    int l3_idx = (index >> 12) & 0x3F;
    int l2_idx = (index >>  6) & 0x3F;
    int l1_idx =  index        & 0x3F;
    uint64_t m    = (uint64_t)tree->mode;
    uint64_t bit1 = 1ULL << l1_idx;

    /* Pre-mark snapshot â€” tree_acquire_and_mark_ would set dirty unconditionally
       and break the COW test. */
    ecs_l2_t* pre_l2 = tree->root->children[l3_idx];
    ecs_l1_t* pre_l1 = (pre_l2 == &ecs_default_l2) ? &ecs_default_l1
                                                    : pre_l2->children[l2_idx];
    int need_cow = (m == 1) && !(pre_l1->dirty & bit1);
    int conf_live = (pre_l1->confirmed_mask_any & bit1) != 0;

    int marked_idx;
    ecs_l1_t* l1 = tree_acquire_and_mark(tree, index, &marked_idx);

    char* data = (char*)l1 + sizeof(ecs_l1_t);
    ecs_buffer_t* conf = (ecs_buffer_t*)(data + (size_t)marked_idx * sizeof(ecs_buffer_t));
    ecs_buffer_t* pred = (ecs_buffer_t*)(data + (size_t)(marked_idx + 64) * sizeof(ecs_buffer_t));

    if (m == 1) {
        if (need_cow) {
            /* Pred slot bytes are NULL post-acquire (memset) or post-rollback
               (rollback NULLs them). Either way, no live header to free here. */
            pred->h = NULL;
            if (wants_confirmed_seed && conf_live && conf->h && conf->h->size > 0u) {
                uint32_t cap = conf->h->capacity;
                ecs_buffer_header_t* nh = (ecs_buffer_header_t*)ecs_xmalloc_aligned(
                    sizeof(ecs_buffer_header_t) + (size_t)cap, 8);
                nh->size     = conf->h->size;
                nh->capacity = cap;
                memcpy(nh->data, conf->h->data, conf->h->size);
                pred->h = nh;
            }
        }
        return pred;
    }
    return conf;
}

void ecs_tree_buffer_push(ecs_tree_t* tree, int index, size_t elem_size, const void* value) {
    assert(elem_size && value);
    ecs_buffer_t* live = buffer_live_(tree, index, /*wants_confirmed_seed=*/1);
    ecs_buffer_push(live, elem_size, value);
}

void ecs_tree_buffer_pop(ecs_tree_t* tree, int index, size_t elem_size) {
    assert(elem_size);
    ecs_buffer_t* live = buffer_live_(tree, index, /*wants_confirmed_seed=*/1);
    ecs_buffer_pop(*live, elem_size);
}

void ecs_tree_buffer_clear(ecs_tree_t* tree, int index) {
    /* Skip seeding; clear discards confirmed contents conceptually. CONFIRMED
       mode keeps the buffer (size=0, capacity preserved). PREDICT first touch
       leaves pred->h = NULL â€” next push allocs fresh. PREDICT subsequent
       touch on already-COWed pred buffer just zeroes size. */
    ecs_buffer_t* live = buffer_live_(tree, index, /*wants_confirmed_seed=*/0);
    if (live->h) live->h->size = 0u;
}

/* Remove a slot. Mode-dispatched branchlessly:
     PREDICT   -> clear predicted_mask only, set dirty (rolled back on rollback,
                  applied on promote).
     CONFIRMED -> clear both confirmed_mask and predicted_mask in lockstep, no
                  dirty. Releases L1/L2 nodes inline when emptied (no promote
                  walk to do it later). For BUFFER trees frees the live slot's
                  heap (ecs_free on its buffer header) before clearing presence.
   Returns 1 if a slot was present and removed, 0 otherwise. */
int ecs_tree_remove(ecs_tree_t* tree, int index) {
    assert(tree);
    assert(index >= 0 && index < (1 << 18));
    int l3_idx = (index >> 12) & 0x3F;
    int l2_idx = (index >>  6) & 0x3F;
    int l1_idx =  index        & 0x3F;

    ecs_l2_t* l2s = tree->root->children[l3_idx];
    if (l2s == &ecs_default_l2) return 0;

    ecs_l1_t* l1s = l2s->children[l2_idx];
    if (l1s == &ecs_default_l1) return 0;

    uint64_t bit1 = 1ULL << l1_idx;
    if ((l1s->predicted_mask_any & bit1) == 0) return 0;     /* slot already absent */

    uint64_t m         = (uint64_t)tree->mode;          /* 0 = CONFIRMED, 1 = PREDICT */
    uint64_t conf_mask = -(uint64_t)(1ULL - m);         /* ~0 in CONFIRMED, 0 in PREDICT */
    int      is_buffer   = (tree->flags & ECS_TREE_FLAG_BUFFER) != 0;

    /* CONFIRMED + BUFFER: free the live confirmed slot's buffer header now (no
       promote will see it). */
    if (is_buffer && m == 0 && tree->data_size) {
        ecs_buffer_t* slot = (ecs_buffer_t*)((char*)l1s + sizeof(ecs_l1_t)
                          + (size_t)l1_idx * tree->data_size);
        if (slot->h) { ecs_free(slot->h); slot->h = NULL; }
    }

    /* PREDICT + BUFFER: if predicted slot currently holds a live this-cycle
       write (predicted_mask & dirty bit set), free it inline. After this,
       predicted bytes are stale; rollback uses the same (predicted_mask &
       dirty) test to avoid double-free. Plain predict-remove on a confirmed-
       only slot (dirty was 0) leaves predicted bytes stale-POD; no free. */
    if (is_buffer && m == 1 && tree->data_size
        && (l1s->predicted_mask_any & l1s->dirty & bit1)) {
        ecs_buffer_t* pslot = (ecs_buffer_t*)((char*)l1s + sizeof(ecs_l1_t)
                           + (size_t)(l1_idx + 64) * tree->data_size);
        if (pslot->h) { ecs_free(pslot->h); pslot->h = NULL; }
    }

    l1s->predicted_mask_any &= ~bit1;
    l1s->confirmed_mask_any &= ~(bit1 & conf_mask);     /* clear bit only in CONFIRMED */
    l1s->dirty              |=  bit1 * m;
    l1s->changed            |=  bit1;

    /* L1 lost a bit -> can no longer be fully full. Clear l2->mask_all bit
       unconditionally (idempotent: clearing already-clear bit is a no-op).
       In PREDICT mode confirmed_mask_any wasn't touched, but bit2 & conf_mask
       == 0 leaves confirmed_mask_all bit untouched. */
    uint64_t bit2 = 1ULL << l2_idx;
    l2s->dirty               |=  bit2 * m;
    l2s->changed             |=  bit2;
    l2s->predicted_mask_all  &= ~bit2;
    l2s->confirmed_mask_all  &= ~(bit2 & conf_mask);
    if (!l1s->predicted_mask_any) {
        l2s->predicted_mask_any &= ~bit2;
        l2s->confirmed_mask_any &= ~(bit2 & conf_mask);

        /* CONFIRMED mode: no promote will run, so release the empty L1 inline. */
        if (m == 0) {
            l2s->children[l2_idx] = &ecs_default_l1;
            ecs_l1_release(tree, l1s);
        }
    }

    /* L2 lost fullness -> clear l3->mask_all bit. */
    uint64_t bit3 = 1ULL << l3_idx;
    tree->root->dirty               |=  bit3 * m;
    tree->root->changed             |=  bit3;
    tree->root->predicted_mask_all  &= ~bit3;
    tree->root->confirmed_mask_all  &= ~(bit3 & conf_mask);
    if (!l2s->predicted_mask_any) {
        tree->root->predicted_mask_any &= ~bit3;
        tree->root->confirmed_mask_any &= ~(bit3 & conf_mask);

        if (m == 0) {
            tree->root->children[l3_idx] = &ecs_default_l2;
            ecs_l2_release(tree, l2s);
        }
    }

    return 1;
}

/* ==========================================================================
   CRC64 -- over confirmed state only (post-promote observable state).
   ========================================================================== */

#define ECS_CRC64_POLY 0xC96C5795D7870F42ULL

static uint64_t ecs_crc64_table[256];

void ecs_crc64_init(void) {
    for (int i = 0; i < 256; i++) {
        uint64_t c = (uint64_t)i;
        for (int j = 0; j < 8; j++)
            c = (c >> 1) ^ ((c & 1) ? ECS_CRC64_POLY : 0);
        ecs_crc64_table[i] = c;
    }
}

static uint64_t ecs_crc64_feed(uint64_t crc, const void* data, size_t len) {
    const unsigned char* p = (const unsigned char*)data;
    while (len--)
        crc = (crc >> 8) ^ ecs_crc64_table[(crc ^ *p++) & 0xFF];
    return crc;
}

uint64_t ecs_tree_crc64(const ecs_tree_t* tree) {
    uint64_t crc = ~0ULL;
    const ecs_l3_t* l3 = tree->root;
    crc = ecs_crc64_feed(crc, &l3->predicted_mask_any, 8);
    uint64_t visit3 = l3->predicted_mask_any;
    while (visit3) {
        int i = ecs_ctz64(visit3); visit3 &= visit3 - 1;
        const ecs_l2_t* l2 = l3->children[i];
        crc = ecs_crc64_feed(crc, &l2->predicted_mask_any, 8);
        uint64_t visit2 = l2->predicted_mask_any;
        while (visit2) {
            int j = ecs_ctz64(visit2); visit2 &= visit2 - 1;
            const ecs_l1_t* l1 = l2->children[j];
            crc = ecs_crc64_feed(crc, &l1->predicted_mask_any, 8);
            if (tree->data_size > 0) {
                uint64_t alive = l1->predicted_mask_any;
                uint64_t dirty = l1->dirty;
                while (alive) {
                    int k = ecs_ctz64(alive); alive &= alive - 1;
                    const void* slot = (dirty & (1ULL << k))
                        ? ecs_l1_predicted(l1, k, tree->data_size)
                        : ecs_l1_confirmed(l1, k, tree->data_size);
                    crc = ecs_crc64_feed(crc, slot, tree->data_size);
                }
            }
        }
    }
    return ~crc;
}

uint64_t ecs_world_crc64(const ecs_world_t* world) {
    uint64_t crc = ~0ULL;
    crc = ecs_crc64_feed(crc, &world->mask, 8);
    uint64_t mask = world->mask;
    while (mask) {
        int i = ecs_ctz64(mask); mask &= mask - 1;
        uint8_t idx = (uint8_t)i;
        crc = ecs_crc64_feed(crc, &idx, 1);
        uint64_t tree_crc = ecs_tree_crc64(&world->trees[i]);
        crc = ecs_crc64_feed(crc, &tree_crc, 8);
    }
    return ~crc;
}

/* ==========================================================================
   Rollback / tick-end â€” discard predicted, clear changed, release empty nodes.
   Predicted state is always speculative; confirmed state advances only via
   CONFIRMED-mode writes. Walks union of dirty + changed: dirty drives
   predicted-bytes invalidation, changed drives per-tick reset. Predicted-only
   adds (no confirmed bytes) get released back to the L1/L2 pool.
   ========================================================================== */

void ecs_tree_rollback(ecs_tree_t* tree) {
    assert(tree);
    ecs_l3_t* l3s   = tree->root;
    if ((l3s->dirty | l3s->changed) == 0) return;     /* idle tick â€” no advance */
    uint64_t visit3 = l3s->dirty | l3s->changed;

    /* Confirmed state advanced iff any slot has changed&~dirty (CONFIRMED-mode
       writes set changed but not dirty; PREDICT-mode writes set both). */
    int confirmed_advanced = 0;

    while (visit3) {
        int i = ecs_ctz64(visit3); visit3 &= visit3 - 1;
        ecs_l2_t* l2s = l3s->children[i];
        if (l2s == &ecs_default_l2) continue;
        uint64_t visit2 = l2s->dirty | l2s->changed;

        while (visit2) {
            int j = ecs_ctz64(visit2); visit2 &= visit2 - 1;
            ecs_l1_t* l1s = l2s->children[j];
            if (l1s == &ecs_default_l1) continue;

            if (l1s->changed & ~l1s->dirty) confirmed_advanced = 1;

            /* BUFFER: predicted slot bytes hold a live this-cycle buffer header
               iff (predicted_mask_any & dirty) bit set â€” predict-set this
               cycle that wasn't subsequently predict-removed (which would
               have freed inline and cleared predicted_mask). Free live
               clones before mask reset, otherwise rollback leaks them. POD
               trees skip the walk via the flag check. */
            if ((tree->flags & ECS_TREE_FLAG_BUFFER) && tree->data_size) {
                uint64_t victims = l1s->predicted_mask_any & l1s->dirty;
                while (victims) {
                    int k = ecs_ctz64(victims); victims &= victims - 1;
                    ecs_buffer_t* pslot = (ecs_buffer_t*)((char*)l1s + sizeof(ecs_l1_t)
                                       + (size_t)(k + 64) * tree->data_size);
                    if (pslot->h) { ecs_free(pslot->h); pslot->h = NULL; }
                }
            }

            l1s->predicted_mask_any = l1s->confirmed_mask_any;
            l1s->dirty              = 0;
            l1s->changed            = 0;

            if (l1s->confirmed_mask_any == 0) {
                /* Predicted-only adds get cleaned up. */
                l2s->children[j] = &ecs_default_l1;
                ecs_l1_release(tree, l1s);
            }
        }

        l2s->predicted_mask_any  = l2s->confirmed_mask_any;
        l2s->predicted_mask_all  = l2s->confirmed_mask_all;
        l2s->dirty               = 0;
        l2s->changed             = 0;

        if (l2s->confirmed_mask_any == 0) {
            l3s->children[i] = &ecs_default_l2;
            ecs_l2_release(tree, l2s);
        }
    }

    l3s->predicted_mask_any  = l3s->confirmed_mask_any;
    l3s->predicted_mask_all  = l3s->confirmed_mask_all;
    l3s->dirty               = 0;
    l3s->changed             = 0;

    if (confirmed_advanced) tree->tick++;
    assert(ecs_tree_masks_valid(tree) && "ecs_tree_rollback: mask invariant broken post-rollback");
}

/* ==========================================================================
   Begin-tick â€” clears `changed` hierarchy. Walks `changed` mask sparsely.
   `changed` is system-facing per-tick delta; lifetime independent of dirty.
   Default-pointer children are skipped (they may be empty stubs left after
   removal that still carry parent-bit set in `changed`).
   ========================================================================== */
void ecs_tree_begin_tick(ecs_tree_t* tree) {
    assert(tree);
    if (tree->root->changed == 0) return;
    ecs_l3_t* l3s   = tree->root;
    uint64_t visit3 = l3s->changed;
    while (visit3) {
        int i = ecs_ctz64(visit3); visit3 &= visit3 - 1;
        ecs_l2_t* l2s = l3s->children[i];
        if (l2s != &ecs_default_l2) {
            uint64_t visit2 = l2s->changed;
            while (visit2) {
                int j = ecs_ctz64(visit2); visit2 &= visit2 - 1;
                ecs_l1_t* l1s = l2s->children[j];
                if (l1s != &ecs_default_l1) l1s->changed = 0;
            }
            l2s->changed = 0;
        }
    }
    l3s->changed = 0;
}

/* Bumps world->tick_id, clears `changed` everywhere, snapshots tick_id onto
   each populated tree. Pairs with the changed-clause invariant in
   ecs_iterator_init. Mode-agnostic â€” same call for CONFIRMED + PREDICT ticks. */
void ecs_world_begin_tick(ecs_world_t* world) {
    assert(world);
    world->tick_id++;
    uint64_t mask = world->mask;
    while (mask) {
        int i = ecs_ctz64(mask); mask &= mask - 1;
        ecs_tree_begin_tick(&world->trees[i]);
        world->trees[i].tick_id_at_begin = world->tick_id;
    }
}

void ecs_pipeline_run(ecs_pipeline_t* p, ecs_world_t* world) {
    assert(p && world);
    ecs_world_begin_tick(world);
    uint32_t n = p->count;
    ecs_system_fn* fns  = p->fns;
    void**         ctxs = p->ctxs;
    for (uint32_t i = 0; i < n; i++) {
        fns[i](world, ctxs[i]);
    }
}

/* ==========================================================================
   Mask invariant -- at-rest (no dirty) state. Verifies tree is internally
   consistent: predicted == confirmed everywhere, mask_any reflects child
   presence, mask_all reflects full-subtree fullness (bit set iff every slot
   in subtree carries the term).
   ========================================================================== */
int ecs_tree_masks_valid(const ecs_tree_t* tree) {
    const ecs_l3_t* l3s = tree->root;
    uint64_t l3_any = 0, l3_all = 0;

    for (int i = 0; i < 64; i++) {
        const ecs_l2_t* l2s = l3s->children[i];
        int claimed_live = (l3s->confirmed_mask_any >> i) & 1;

        if (l2s == &ecs_default_l2) {
            if (claimed_live) return 0;
            continue;
        }

        uint64_t l2_any = 0, l2_all = 0;
        for (int j = 0; j < 64; j++) {
            const ecs_l1_t* l1s = l2s->children[j];
            int l1_claimed_live = (l2s->confirmed_mask_any >> j) & 1;

            if (l1s == &ecs_default_l1) {
                if (l1_claimed_live) return 0;
                continue;
            }
            if (l1s->confirmed_mask_any == 0) {
                if (l1_claimed_live) return 0;
                continue;
            }
            if (!l1_claimed_live) return 0;
            /* At-rest invariant: predicted == confirmed, dirty == 0. */
            if (l1s->predicted_mask_any != l1s->confirmed_mask_any) return 0;
            if (l1s->dirty != 0) return 0;
            l2_any |= 1ULL << j;
            if (l1s->confirmed_mask_any == ~0ULL) l2_all |= 1ULL << j;
        }

        if (l2s->confirmed_mask_any != l2_any) return 0;
        if (l2s->confirmed_mask_all != l2_all) return 0;
        if (l2s->predicted_mask_any != l2_any) return 0;
        if (l2s->predicted_mask_all != l2_all) return 0;
        if (l2s->dirty              != 0)      return 0;

        if (l2_any == 0) {
            if (claimed_live) return 0;
        } else {
            if (!claimed_live) return 0;
            l3_any |= 1ULL << i;
            if (l2_all == ~0ULL) l3_all |= 1ULL << i;
        }
    }

    if (l3s->confirmed_mask_any != l3_any) return 0;
    if (l3s->confirmed_mask_all != l3_all) return 0;
    if (l3s->predicted_mask_any != l3_any) return 0;
    if (l3s->predicted_mask_all != l3_all) return 0;
    if (l3s->dirty              != 0)      return 0;
    return 1;
}

/* ==========================================================================
   World ops + lifecycle
   ========================================================================== */

void ecs_world_rollback(ecs_world_t* world) {
    assert(world);
    uint64_t mask = world->mask;
    int any_advanced = 0;
    while (mask) {
        int i = ecs_ctz64(mask); mask &= mask - 1;
        uint64_t before = world->trees[i].tick;
        ecs_tree_rollback(&world->trees[i]);
        if (world->trees[i].tick != before) any_advanced = 1;
    }
    if (any_advanced) world->tick++;
}

/* Switch a single tree's VM mode. Caller must guarantee no in-flight prediction
   on this tree (dirty == 0 everywhere). Cheap: just flips the tag -- masks are
   already in lockstep when dirty == 0, so the in-CONFIRMED invariant holds
   the moment the flip happens. */
void ecs_tree_set_mode(ecs_tree_t* tree, ecs_mode_t mode) {
    assert(tree);
    assert(ecs_tree_no_dirty(tree) &&
           "ecs_tree_set_mode: in-flight prediction -- promote or rollback first");
    tree->mode = mode;
}

void ecs_world_set_mode(ecs_world_t* world, ecs_mode_t mode) {
    assert(world);
    uint64_t mask = world->mask;
    while (mask) {
        int i = ecs_ctz64(mask); mask &= mask - 1;
        ecs_tree_set_mode(&world->trees[i], mode);
    }
    world->mode = mode;
}

void ecs_world_destroy(ecs_world_t* world) {
    if (!world) return;
    uint64_t mask = world->mask;
    while (mask) {
        int i = ecs_ctz64(mask); mask &= mask - 1;
        ecs_tree_destroy(&world->trees[i]);
    }
    world->mask = 0;
}

/* ==========================================================================
   Binary serializer -- confirmed state only, mask-driven sparse format,
   bitpacked via ecs_serializer.
   ========================================================================== */

void ecs_serialize_batch_raw(const void* l1_data, size_t block_size,
                             uint64_t mask, ecs_serializer_t* s) {
    if (!block_size || !mask) return;
    /* write_bytes self-aligns; coalesce runs of consecutive set bits into
       a single bulk write (shared with ecs_memcpy_sparse via ecs_mask_pop_run). */
    int idx, run;
    while ((run = ecs_mask_pop_run(&mask, &idx))) {
        ecs_serializer_write_bytes(s,
            (const uint8_t*)l1_data + (size_t)idx * block_size,
            (int32_t)((size_t)run * block_size));
    }
}

void ecs_deserialize_batch_raw(void* l1_data, size_t block_size,
                               uint64_t mask, ecs_deserializer_t* d) {
    if (!block_size || !mask) return;
    int idx, run;
    while ((run = ecs_mask_pop_run(&mask, &idx))) {
        ecs_deserializer_read_bytes(d,
            (uint8_t*)l1_data + (size_t)idx * block_size,
            (int32_t)((size_t)run * block_size));
    }
}

static void ecs_serialize_u64_(uint64_t v, ecs_serializer_t* s) {
    /* Fast path -- single 64-bit bit-write. Endian normalization happens
       inside the bitpacker on the qword scratch flush, so the wire format
       is identical to write_bytes(8) but skips the byte-alignment dance. */
    ecs_serializer_write_bits(s, v, 64);
}

/* Variable-bit mask encoding. Wire format:

     tag(1):
       0 -> raw u64 follows                                  65 bits
       1 -> compressed; sub-tag(1):
              1 -> all-set shortcut (mask == ~0)              2 bits
              0 -> indexed:
                   polarity(1) | k(3) | k * index(6)         6 + 6k bits

   Polarity 0 = encode set bits of mask; polarity 1 = encode clear bits
   (reader reconstructs as ~indexed). k = min(popcount, 64-popcount), so
   polarity is picked to minimise k. The indexed path is taken when the
   `min_count <= 8` test passes; k is stored in 3 bits so the effective
   range is 0..7 (note: min_count == 8 would overflow the field -- tighten
   to <= 7 or widen k(3) to k(4) if exact 8 is needed).

   Hot ECS shapes:
     * all-set L1 (64 entities present): handled by sub-tag=1, total 2 bits.
     * all-zero mask (root only): k=0, polarity=1, total 6 bits.
     * sparse L1 (1..7 entities): 12..48 bits.
     * dense random mask: raw 65 bits.

   Indexed beats raw whenever 6 + 6k < 65, i.e. k <= 9 mathematically; the
   3-bit k field caps it earlier. */
static void ecs_serialize_mask(uint64_t m, ecs_serializer_t* s) {
    int set_count = ecs_popcount64(m);

    if (set_count == 64) {
        ecs_serializer_write_bits(s, 0x3, 2);                       /* tag: indexed (LSB=1), sub: all-set (next=1) */
    }
    else
    {
        int clear_count = 64 - set_count;
        int min_count = set_count < clear_count ? set_count : clear_count;

        if (min_count <= 8) {
            ecs_serializer_write_bits(s, 0x1, 2);                   /* tag: indexed (LSB=1), sub: not-all-set (next=0) */
            int inverted = clear_count < set_count;
            ecs_serializer_write_bits(s, (uint64_t)inverted, 1);
            uint64_t encode = inverted ? ~m : m;

            ecs_serializer_write_bits(s, (uint64_t)min_count, 3);
            while (encode) {
                int i = ecs_ctz64(encode);
                encode &= encode - 1;
                ecs_serializer_write_bits(s, (uint64_t)i, 6);
            }
        }
        else {
            ecs_serializer_write_bits(s, 0, 1);                          /* tag: raw */
            ecs_serializer_write_bits(s, m, 64);
        }
    }
}

static void ecs_serialize_varint(uint64_t v, ecs_serializer_t* s) {
    /* Each LEB128 byte is 8 bits; write_bits skips the byte-alignment
       head/tail dance in write_bytes. bits_written stays %8 == 0. */
    do {
        uint64_t b = v & 0x7F;
        v >>= 7;
        if (v) b |= 0x80;
        ecs_serializer_write_bits(s, b, 8);
    } while (v);
}

void ecs_tree_serialize(const ecs_tree_t* tree, ecs_serializer_t* s) {
    assert(tree && s);

    uint8_t version = 1;
    uint8_t flags   = (tree->data_size > 0) ? 1u : 0u;
    ecs_serializer_write_bits(s, version, 8);
    ecs_serializer_write_bits(s, flags,   8);

    ecs_serialize_varint((uint64_t)tree->data_size, s);
    ecs_serialize_u64_(tree->tick, s);

    const ecs_l3_t* l3 = tree->root;

    /* ---- Pass 1: structural metadata (all masks) ----
       Walk the tree once writing l3, l2, l1 masks contiguously. Decoder
       can rebuild full topology from this block alone -- useful for
       streaming, partial parsing, or piping the payload block into a
       separate compressor (zstd, lz4) on top. */
    ecs_serialize_mask(l3->confirmed_mask_any, s);

    uint64_t v3 = l3->confirmed_mask_any;
    while (v3) {
        int i = ecs_ctz64(v3); v3 &= v3 - 1;
        const ecs_l2_t* l2 = l3->children[i];
        ecs_serialize_mask(l2->confirmed_mask_any, s);

        uint64_t v2 = l2->confirmed_mask_any;
        while (v2) {
            int j = ecs_ctz64(v2); v2 &= v2 - 1;
            const ecs_l1_t* l1 = l2->children[j];
            ecs_serialize_mask(l1->confirmed_mask_any, s);
        }
    }

    /* No payload at all? Tag tree (data_size==0) or empty tree. Done. */
    if (!tree->data_size || !l3->confirmed_mask_any) return;

    /* ---- Pass 2: payload (batch-encoded component data) ----
       write_bytes self-aligns at the byte boundary on its first call here,
       so we don't pre-pad -- the alignment cost (<=7 bits) lands inside
       write_bytes once at the pass-1/pass-2 boundary.
       Same walk order as pass 1 -- decoder traverses the reconstructed
       topology in the identical order to find each batch's payload.
       tree->serialize_batch is set by ecs_tree_init (defaults to
       ecs_serialize_batch_raw), so no NULL check needed here. */
    v3 = l3->confirmed_mask_any;
    while (v3) {
        int i = ecs_ctz64(v3); v3 &= v3 - 1;
        const ecs_l2_t* l2 = l3->children[i];

        uint64_t v2 = l2->confirmed_mask_any;
        while (v2) {
            int j = ecs_ctz64(v2); v2 &= v2 - 1;
            const ecs_l1_t* l1 = l2->children[j];
            tree->serialize_batch(ecs_l1_data(l1), tree->data_size,
                                  l1->confirmed_mask_any, s);
        }
    }
}

/* Confirmed-state variants of iter_compute_l{1,2,3}_mask. Serializer writes
   confirmed_mask_any; filters must read the same field so include/exclude
   semantics agree with the bits we actually emit (predicted-mode bits would
   diverge under in-flight prediction).

   L3/L2 variants are subtree-prune helpers: a cleared bit in the result
   means the entire L2 / L1 child can be skipped. Exclude uses _mask_all
   (only drop subtrees where every slot has the term) -- partial overlap
   falls through to the next level. */
static uint64_t serialize_compute_l3_filter(const ecs_compiled_query_t* q,
                                            const ecs_l3_t* const l3[]) {
    uint64_t result = 0;
    for (uint32_t c = 0; c < q->clause_count; c++) {
        const ecs_compiled_clause_t* cl = &q->clauses[c];
        if (!cl->include && !cl->exclude && !cl->changed) continue;

        uint64_t bits = ~0ULL;
        uint32_t mask = cl->include;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= l3[n]->confirmed_mask_any; }
        mask = cl->exclude;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= ~l3[n]->confirmed_mask_all; }
        if (cl->changed) {
            uint64_t any_changed = 0;
            mask = cl->changed;
            while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; any_changed |= l3[n]->changed; }
            bits &= any_changed;
        }
        result |= bits;
    }
    return result;
}

static uint64_t serialize_compute_l2_filter(const ecs_compiled_query_t* q,
                                            const ecs_l2_t* const l2[]) {
    uint64_t result = 0;
    for (uint32_t c = 0; c < q->clause_count; c++) {
        const ecs_compiled_clause_t* cl = &q->clauses[c];
        if (!cl->include && !cl->exclude && !cl->changed) continue;

        uint64_t bits = ~0ULL;
        uint32_t mask = cl->include;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= l2[n]->confirmed_mask_any; }
        mask = cl->exclude;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= ~l2[n]->confirmed_mask_all; }
        if (cl->changed) {
            uint64_t any_changed = 0;
            mask = cl->changed;
            while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; any_changed |= l2[n]->changed; }
            bits &= any_changed;
        }
        result |= bits;
    }
    return result;
}

static uint64_t serialize_compute_l1_filter(const ecs_compiled_query_t* q,
                                            const ecs_l1_t* const l1[]) {
    uint64_t result = 0;
    for (uint32_t c = 0; c < q->clause_count; c++) {
        const ecs_compiled_clause_t* cl = &q->clauses[c];
        if (!cl->include && !cl->exclude && !cl->changed) continue;

        uint64_t bits = ~0ULL;
        uint32_t mask = cl->include;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= l1[n]->confirmed_mask_any; }
        mask = cl->exclude;
        while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; bits &= ~l1[n]->confirmed_mask_any; }
        if (cl->changed) {
            uint64_t any_changed = 0;
            mask = cl->changed;
            while (mask) { int n = ecs_ctz32(mask); mask &= mask-1; any_changed |= l1[n]->changed; }
            bits &= any_changed;
        }
        result |= bits;
    }
    return result;
}

void ecs_tree_serialize_delta(const ecs_tree_t* tree,
                                 const ecs_compiled_query_t* query,
                                 ecs_serializer_t* s) {
    assert(tree && query && s);

    /* Wire format identical to ecs_tree_serialize -- ecs_tree_deserialize
       reads the output unchanged. L1 masks are intersected with the
       per-block query filter; L2/L3 masks are rebuilt bottom-up so a
       parent bit is set IFF its subtree contains at least one surviving
       slot (no false positives -- empty subtrees disappear from both the
       mask stream and the payload stream). */
    uint8_t version = 1;
    uint8_t flags   = (tree->data_size > 0) ? 1u : 0u;
    ecs_serializer_write_bits(s, version, 8);
    ecs_serializer_write_bits(s, flags,   8);

    ecs_serialize_varint((uint64_t)tree->data_size, s);
    ecs_serialize_u64_(tree->tick, s);

    const ecs_l3_t* l3 = tree->root;

    /* Pre-pass: compute delta L1 masks, propagate to L2/L3. Storage
       scoped to set L3/L2 bits only -- worst case 64*64 u64 = 32 KiB stack.
       L3/L2 filters short-circuit walk: cleared bits skip whole subtrees
       that the query already prunes (mirrors iter_compute_l{2,3}_mask). */
    uint64_t delta_l1[64][64];
    uint64_t delta_l2[64];
    uint64_t delta_l3 = 0;

    {
        const ecs_l3_t* l3_arr[ECS_QUERY_MAX_TERMS];
        for (uint32_t t = 0; t < query->tree_count; t++)
            l3_arr[t] = query->trees[t]->root;
        uint64_t l3_filter = serialize_compute_l3_filter(query, l3_arr);

        uint64_t v3 = l3->confirmed_mask_any & l3_filter;
        while (v3) {
            int i = ecs_ctz64(v3); v3 &= v3 - 1;
            const ecs_l2_t* l2 = l3->children[i];

            const ecs_l2_t* l2_arr[ECS_QUERY_MAX_TERMS];
            for (uint32_t t = 0; t < query->tree_count; t++)
                l2_arr[t] = l3_arr[t]->children[i];
            uint64_t l2_filter = serialize_compute_l2_filter(query, l2_arr);

            uint64_t fl2 = 0;
            uint64_t v2  = l2->confirmed_mask_any & l2_filter;
            while (v2) {
                int j = ecs_ctz64(v2); v2 &= v2 - 1;

                const ecs_l1_t* l1_arr[ECS_QUERY_MAX_TERMS];
                for (uint32_t t = 0; t < query->tree_count; t++)
                    l1_arr[t] = l2_arr[t]->children[j];
                uint64_t l1_filter = serialize_compute_l1_filter(query, l1_arr);
                uint64_t out       = l2->children[j]->confirmed_mask_any & l1_filter;

                if (out) {
                    delta_l1[i][j] = out;
                    fl2 |= (uint64_t)1 << j;
                }
            }
            delta_l2[i] = fl2;
            if (fl2) delta_l3 |= (uint64_t)1 << i;
        }
    }

    /* Pass 1: emit pruned masks. */
    ecs_serialize_mask(delta_l3, s);
    {
        uint64_t v3 = delta_l3;
        while (v3) {
            int i = ecs_ctz64(v3); v3 &= v3 - 1;
            ecs_serialize_mask(delta_l2[i], s);

            uint64_t v2 = delta_l2[i];
            while (v2) {
                int j = ecs_ctz64(v2); v2 &= v2 - 1;
                ecs_serialize_mask(delta_l1[i][j], s);
            }
        }
    }

    if (!tree->data_size || !delta_l3) return;

    /* Pass 2: payload. Iterate the same pruned topology so payload bytes
       align 1:1 with the masks emitted in pass 1. */
    {
        uint64_t v3 = delta_l3;
        while (v3) {
            int i = ecs_ctz64(v3); v3 &= v3 - 1;
            const ecs_l2_t* l2 = l3->children[i];

            uint64_t v2 = delta_l2[i];
            while (v2) {
                int j = ecs_ctz64(v2); v2 &= v2 - 1;
                const ecs_l1_t* l1_self = l2->children[j];
                tree->serialize_batch(ecs_l1_data(l1_self), tree->data_size,
                                      delta_l1[i][j], s);
            }
        }
    }
}

/* ==========================================================================
   Binary deserializer -- mirrors ecs_tree_serialize.
   ========================================================================== */

static uint64_t ecs_deserialize_varint(ecs_deserializer_t* d) {
    uint64_t v = 0;
    int shift = 0;
    for (;;) {
        uint64_t b = ecs_deserializer_read_bits(d, 8);
        v |= (b & 0x7Full) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    return v;
}

static uint64_t ecs_deserialize_mask(ecs_deserializer_t* d) {
    /* Inverse of ecs_serialize_mask. Format:
         tag(1) = 0 -> raw u64
         tag(1) = 1 -> sub(1) = 1 -> all-set
                       sub(1) = 0 -> indexed: polarity(1) | k(3) | k * idx(6) */
    uint64_t tag = ecs_deserializer_read_bits(d, 1);
    if (tag == 0) {
        return ecs_deserializer_read_bits(d, 64);
    }
    uint64_t sub = ecs_deserializer_read_bits(d, 1);
    if (sub == 1) return ~(uint64_t)0;

    uint64_t inverted = ecs_deserializer_read_bits(d, 1);
    uint64_t k        = ecs_deserializer_read_bits(d, 3);
    uint64_t encoded  = 0;
    for (uint64_t i = 0; i < k; i++) {
        uint64_t pos = ecs_deserializer_read_bits(d, 6);
        encoded |= (uint64_t)1 << pos;
    }
    return inverted ? ~encoded : encoded;
}

int ecs_tree_deserialize(ecs_tree_t* tree, ecs_deserializer_t* d) {
    assert(tree && d);

    uint8_t version = (uint8_t)ecs_deserializer_read_bits(d, 8);
    uint8_t flags   = (uint8_t)ecs_deserializer_read_bits(d, 8);
    if (version != 1) return -1;
    (void)flags;  /* bit0 is has_data; redundant with data_size */

    uint64_t data_size = ecs_deserialize_varint(d);
    if (!tree->root) {
        /* Uninitialized slot -- set up topology with the stream's data_size.
           Lets ecs_world_deserialize hydrate empty world tree-slots in
           one step. */
        ecs_tree_init(tree, (size_t)data_size, 0);
    } else if (data_size != tree->data_size) {
        return -1;
    }

    tree->tick = ecs_deserializer_read_bits(d, 64);

    /* ---- Pass 1: read masks, repurpose / acquire / release nodes. ----
       Walk every slot 0..63 testing the sentinel pointer (NOT the old
       confirmed mask) so we correctly reclaim nodes that the previous
       state held only via predicted writes. */
    uint64_t new_l3     = ecs_deserialize_mask(d);
    uint64_t new_l3_all = 0;
    ecs_l3_t* l3        = tree->root;

    for (int i = 0; i < 64; i++) {
        ecs_l2_t* l2  = l3->children[i];
        int allocated = (l2 != &ecs_default_l2);
        int new_set   = (new_l3 & ((uint64_t)1 << i)) != 0;

        if (!new_set) {
            if (allocated) {
                /* release every l1 actually held, regardless of old confirmed bits */
                for (int j = 0; j < 64; j++) {
                    ecs_l1_t* l1 = l2->children[j];
                    if (l1 != &ecs_default_l1) ecs_l1_release(tree, l1);
                    l2->children[j] = &ecs_default_l1;
                }
                ecs_l2_release(tree, l2);
            }
            l3->children[i] = &ecs_default_l2;
            continue;
        }

        if (!allocated) {
            l2 = ecs_l2_acquire(tree);
            l3->children[i] = l2;
        }

        uint64_t new_l2     = ecs_deserialize_mask(d);
        uint64_t new_l2_all = 0;

        for (int j = 0; j < 64; j++) {
            ecs_l1_t* l1   = l2->children[j];
            int allocated2 = (l1 != &ecs_default_l1);
            int new_set2   = (new_l2 & ((uint64_t)1 << j)) != 0;

            if (!new_set2) {
                if (allocated2) ecs_l1_release(tree, l1);
                l2->children[j] = &ecs_default_l1;
                continue;
            }

            if (!allocated2) {
                l1 = ecs_l1_acquire(tree);
                l2->children[j] = l1;
            }

            uint64_t new_l1 = ecs_deserialize_mask(d);
            l1->confirmed_mask_any = new_l1;
            l1->predicted_mask_any = new_l1;
            l1->dirty              = 0;
            l1->changed            = 0;
            if (new_l1 == ~0ULL) new_l2_all |= (uint64_t)1 << j;
        }

        l2->confirmed_mask_any  = new_l2;
        l2->confirmed_mask_all  = new_l2_all;
        l2->predicted_mask_any  = new_l2;
        l2->predicted_mask_all  = new_l2_all;
        l2->dirty               = 0;
        l2->changed             = 0;

        if (new_l2_all == ~0ULL) new_l3_all |= (uint64_t)1 << i;
    }

    l3->confirmed_mask_any  = new_l3;
    l3->confirmed_mask_all  = new_l3_all;
    l3->predicted_mask_any  = new_l3;
    l3->predicted_mask_all  = new_l3_all;
    l3->dirty               = 0;
    l3->changed             = 0;

    /* ---- Pass 2: payloads ---- */
    if (!tree->data_size || !new_l3) return 0;

    uint64_t v3 = new_l3;
    while (v3) {
        int i = ecs_ctz64(v3); v3 &= v3 - 1;
        ecs_l2_t* l2 = l3->children[i];
        uint64_t v2 = l2->confirmed_mask_any;
        while (v2) {
            int j = ecs_ctz64(v2); v2 &= v2 - 1;
            ecs_l1_t* l1 = l2->children[j];
            tree->deserialize_batch(ecs_l1_data(l1), tree->data_size,
                                    l1->confirmed_mask_any, d);
        }
    }
    return 0;
}

/* ==========================================================================
   World-level serialize/deserialize. Mask-driven: emits only populated
   tree slots; deserialize destroys vanished slots and auto-inits new ones.
   ========================================================================== */

void ecs_world_serialize(const ecs_world_t* world, ecs_serializer_t* s) {
    assert(world && s);
    uint8_t version = 1;
    ecs_serializer_write_bits(s, version, 8);
    ecs_serializer_write_bits(s, world->tick, 64);
    ecs_serialize_mask(world->mask, s);

    uint64_t m = world->mask;
    while (m) {
        int i = ecs_ctz64(m); m &= m - 1;
        ecs_tree_serialize(&world->trees[i], s);
    }
}

int ecs_world_deserialize(ecs_world_t* world, ecs_deserializer_t* d) {
    assert(world && d);
    uint8_t version = (uint8_t)ecs_deserializer_read_bits(d, 8);
    if (version != 1) return -1;

    world->tick = ecs_deserializer_read_bits(d, 64);
    uint64_t new_mask = ecs_deserialize_mask(d);
    uint64_t old_mask = world->mask;

    /* Slots in old but not new: destroy then zero so a future deserialize
       sees an uninitialized slot (root == NULL). */
    uint64_t drop = old_mask & ~new_mask;
    while (drop) {
        int i = ecs_ctz64(drop); drop &= drop - 1;
        ecs_tree_destroy(&world->trees[i]);
        memset(&world->trees[i], 0, sizeof(ecs_tree_t));
    }

    /* Slots in new mask: deserialize. ecs_tree_deserialize auto-inits
       when tree->root is NULL (fresh slot from the drop loop or zero-
       initialized world). */
    uint64_t bring = new_mask;
    while (bring) {
        int i = ecs_ctz64(bring); bring &= bring - 1;
        int rc = ecs_tree_deserialize(&world->trees[i], d);
        if (rc != 0) return rc;
    }

    world->mask  = new_mask;
    world->dirty = 0;
    return 0;
}

void ecs_tree_destroy(ecs_tree_t* tree) {
    assert(tree);
    if (tree->root) {
        /* Tree must be at rest â€” predicted == confirmed everywhere, dirty=0.
           That collapses confirmed_mask_any and predicted_mask_any to the
           same "any present" mask, so we just iterate one of them with ctz
           (skipping all-zero L2/L1 subtrees outright). Heap-owning
           components must promote or rollback before destroy. */
        assert(ecs_tree_no_dirty(tree) &&
               "ecs_tree_destroy: in-flight prediction â€” promote or rollback first");

        uint64_t v3 = tree->root->confirmed_mask_any;
        while (v3) {
            int i = ecs_ctz64(v3); v3 &= v3 - 1;
            ecs_l2_t* l2 = tree->root->children[i];
            uint64_t  v2 = l2->confirmed_mask_any;
            while (v2) {
                int j = ecs_ctz64(v2); v2 &= v2 - 1;
                ecs_l1_t* l1   = l2->children[j];
                uint64_t  mask = l1->confirmed_mask_any;
                if ((tree->flags & ECS_TREE_FLAG_BUFFER) && tree->data_size && mask) {
                    /* BUFFER teardown: free each live slot's buffer header. */
                    uint64_t alive = mask;
                    while (alive) {
                        int k = ecs_ctz64(alive); alive &= alive - 1;
                        ecs_buffer_t* slot = (ecs_buffer_t*)((char*)l1 + sizeof(ecs_l1_t)
                                          + (size_t)k * tree->data_size);
                        if (slot->h) ecs_free(slot->h);
                    }
                }
                ecs_free(l1);
            }
            ecs_free(l2);
        }
        ecs_free(tree->root);
    }
    tree->root = NULL;
}
