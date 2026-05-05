#include "broadphase.h"

#include <assert.h>
#include <mimalloc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__BMI2__)
#include <immintrin.h>
#endif

#define BP_NODE_ALIGN  32
#define BP_MORTON_BITS 10                       /* per axis -> 30-bit code */
#define BP_MORTON_MAX  ((1u << BP_MORTON_BITS) - 1u)

/* Average via int64 to avoid int32 overflow when both endpoints sit near the
   Q16.16 extremes. */
static inline fixed_t bp_avg(fixed_t a, fixed_t b) {
    return (fixed_t)(((int64_t)a + (int64_t)b) >> 1);
}

/* Quantize centroid into [0, 2^BP_MORTON_BITS) via precomputed reciprocal.
   `recip` = ((MORTON_MAX+1) << 32) / range, computed once per axis in the
   caller. Per-item cost: one i64 sub, one u64 mul, one shift -- replaces the
   per-item div in the original implementation. */
static inline uint32_t bp_quantize_recip(fixed_t v, fixed_t origin, uint64_t recip) {
    int64_t d = (int64_t)v - (int64_t)origin;
    if (d <= 0) return 0;
    uint64_t q = ((uint64_t)d * recip) >> 32;
    if (q > BP_MORTON_MAX) return BP_MORTON_MAX;
    return (uint32_t)q;
}

/* Spread the low 10 bits of x across every third bit slot -- produces
   0b001 001 001 ... when x = 0x3FF.

   BMI2 path: PDEP deposits the low 10 input bits into the 10 set positions of
   the destination mask (0x09249249 = bits 0,3,6,...,27). Single-uop, ~3-cycle
   latency on Intel Haswell+ and AMD Zen 3+. Caveat: Zen 1/2 PDEP is
   microcoded (~250 cycles) -- if you target those parts, undefine __BMI2__
   for this TU or add a runtime gate.

   Scalar fallback: standard magic-constant interleave, ~9 cycles. */
static inline uint32_t bp_morton_part(uint32_t x) {
#if defined(__BMI2__)
    return _pdep_u32(x & 0x3FFu, 0x09249249u);
#else
    x &= 0x3FFu;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x <<  8)) & 0x0300F00Fu;
    x = (x | (x <<  4)) & 0x030C30C3u;
    x = (x | (x <<  2)) & 0x09249249u;
    return x;
#endif
}

static inline uint32_t bp_morton30(uint32_t x, uint32_t y, uint32_t z) {
    return bp_morton_part(x) | (bp_morton_part(y) << 1) | (bp_morton_part(z) << 2);
}

/* Sentinel lanes in unoccupied slots: min lanes hold INT32_MAX, max lanes
   hold INT32_MIN. Two consequences fall out:
     1. Horizontal min/max reductions ignore unoccupied lanes naturally --
        no presence-mask gating, so the parent-build loop just chains
        fixed_min/fixed_max across the 8 lanes.
     2. Per-node overlap compares reject sentinel lanes for free: sentinel
        min > any qmax and sentinel max < any qmin, so neither fixed8_le nor
        fixed8_ge sets that lane bit. */

/* SIMD overlap: 8 child/leaf slots vs 1 query AABB. Six fixed8 compares +
   five ANDs. Caller passes pre-splatted query lanes (cached in the iterator)
   so the splat ops happen once per query, not once per visited node. AND with
   presence_mask zeros out unoccupied lanes so undefined slot data cannot
   produce false hits. Same routine works on internal nodes (lane = child
   AABB) and leaves (lane = item AABB). */
static inline uint8_t bp_overlap_mask(const broadphase_node_t* n,
    fixed_8_t qmin_x, fixed_8_t qmax_x,
    fixed_8_t qmin_y, fixed_8_t qmax_y,
    fixed_8_t qmin_z, fixed_8_t qmax_z) {
    uint8_t m = fixed8_le(n->min_x, qmax_x) & fixed8_ge(n->max_x, qmin_x)
              & fixed8_le(n->min_y, qmax_y) & fixed8_ge(n->max_y, qmin_y)
              & fixed8_le(n->min_z, qmax_z) & fixed8_ge(n->max_z, qmin_z);
    return m & n->presence_mask;
}

void broadphase_init(broadphase_t* bp, size_t item_cap) {
    assert(bp);
	assert(item_cap > 8 && item_cap % 8 == 0);

    bp->item_cap   = (uint32_t)item_cap;
    bp->n_items    = 0;
    bp->object_cap = (uint32_t)item_cap;
    bp->n_objects  = 0;

    /* Node count upper bound: leaves = ceil(N/8); internals form a geometric
       sum that converges below leaves/7. ceil(N/7) + 8 covers it for any N. */
    size_t node_cap = item_cap / 7 + 8;
    bp->node_cap = (uint32_t)node_cap;
    bp->n_nodes  = 0;
    bp->root     = 0;
    bp->has_tree = 0;

    /* All SoA item arrays are 32-byte aligned -- AVX2 aligned loads on the
       autovectorized world-AABB / centroid scans. Pad each axis allocation
       up to a multiple of 8 fixed_t so the SIMD tail can read junk safely. */
    size_t pad_cap = (item_cap + 7u) & ~(size_t)7u;
    size_t axis_bytes = pad_cap * sizeof(fixed_t);

    bp->item_ids   = (uint32_t*)mi_malloc_aligned(pad_cap * sizeof(uint32_t), BP_NODE_ALIGN);
    bp->objects    = (broadphase_object_t*)mi_malloc_aligned(pad_cap * sizeof(broadphase_object_t), BP_NODE_ALIGN);
    bp->item_min_x = (fixed_t*) mi_malloc_aligned(axis_bytes, BP_NODE_ALIGN);
    bp->item_min_y = (fixed_t*) mi_malloc_aligned(axis_bytes, BP_NODE_ALIGN);
    bp->item_min_z = (fixed_t*) mi_malloc_aligned(axis_bytes, BP_NODE_ALIGN);
    bp->item_max_x = (fixed_t*) mi_malloc_aligned(axis_bytes, BP_NODE_ALIGN);
    bp->item_max_y = (fixed_t*) mi_malloc_aligned(axis_bytes, BP_NODE_ALIGN);
    bp->item_max_z = (fixed_t*) mi_malloc_aligned(axis_bytes, BP_NODE_ALIGN);
    bp->morton     = (uint32_t*)mi_malloc_aligned(pad_cap * sizeof(uint32_t), BP_NODE_ALIGN);
    bp->perm       = (uint32_t*)mi_malloc_aligned(pad_cap * sizeof(uint32_t), BP_NODE_ALIGN);
    bp->perm_alt   = (uint32_t*)mi_malloc_aligned(pad_cap * sizeof(uint32_t), BP_NODE_ALIGN);
    bp->nodes      = (broadphase_node_t*)mi_malloc_aligned(
                         node_cap * sizeof(broadphase_node_t), BP_NODE_ALIGN);

    if (!bp->item_ids   || !bp->objects    || !bp->item_min_x || !bp->item_min_y || !bp->item_min_z ||
        !bp->item_max_x || !bp->item_max_y || !bp->item_max_z ||
        !bp->morton     || !bp->perm       || !bp->perm_alt   || !bp->nodes) {
        fprintf(stderr, "broadphase: OOM (item_cap=%zu node_cap=%zu)\n",
                item_cap, node_cap);
        abort();
    }
}

void broadphase_destroy(broadphase_t* bp) {
    assert(bp);
    if (bp->item_ids)   mi_free(bp->item_ids);
    if (bp->objects)    mi_free(bp->objects);
    if (bp->item_min_x) mi_free(bp->item_min_x);
    if (bp->item_min_y) mi_free(bp->item_min_y);
    if (bp->item_min_z) mi_free(bp->item_min_z);
    if (bp->item_max_x) mi_free(bp->item_max_x);
    if (bp->item_max_y) mi_free(bp->item_max_y);
    if (bp->item_max_z) mi_free(bp->item_max_z);
    if (bp->morton)     mi_free(bp->morton);
    if (bp->perm)       mi_free(bp->perm);
    if (bp->perm_alt)   mi_free(bp->perm_alt);
    if (bp->nodes)      mi_free(bp->nodes);
    bp->item_ids = bp->morton = bp->perm = bp->perm_alt = NULL;
    bp->item_min_x = bp->item_min_y = bp->item_min_z = NULL;
    bp->item_max_x = bp->item_max_y = bp->item_max_z = NULL;
    bp->nodes   = NULL;
    bp->objects = NULL;
    bp->item_cap   = bp->n_items   = 0;
    bp->object_cap = bp->n_objects = 0;
    bp->node_cap   = bp->n_nodes   = 0;
    bp->has_tree = 0;
}

void broadphase_clear(broadphase_t* bp) {
    assert(bp);
    bp->n_items   = 0;
    bp->n_objects = 0;
    bp->n_nodes   = 0;
    bp->has_tree  = 0;
}

static void bp_grow(broadphase_t* bp) {
    size_t new_cap      = (size_t)bp->item_cap * 2;
    size_t new_pad      = (new_cap + 7u) & ~(size_t)7u;
    size_t new_axis     = new_pad * sizeof(fixed_t);
    size_t new_node_cap = new_cap / 7 + 8;

    bp->item_ids   = (uint32_t*)mi_realloc_aligned(bp->item_ids,   new_pad * sizeof(uint32_t), BP_NODE_ALIGN);
    bp->objects    = (broadphase_object_t*)mi_realloc_aligned(bp->objects, new_pad * sizeof(broadphase_object_t), BP_NODE_ALIGN);
    bp->item_min_x = (fixed_t*) mi_realloc_aligned(bp->item_min_x, new_axis,                   BP_NODE_ALIGN);
    bp->item_min_y = (fixed_t*) mi_realloc_aligned(bp->item_min_y, new_axis,                   BP_NODE_ALIGN);
    bp->item_min_z = (fixed_t*) mi_realloc_aligned(bp->item_min_z, new_axis,                   BP_NODE_ALIGN);
    bp->item_max_x = (fixed_t*) mi_realloc_aligned(bp->item_max_x, new_axis,                   BP_NODE_ALIGN);
    bp->item_max_y = (fixed_t*) mi_realloc_aligned(bp->item_max_y, new_axis,                   BP_NODE_ALIGN);
    bp->item_max_z = (fixed_t*) mi_realloc_aligned(bp->item_max_z, new_axis,                   BP_NODE_ALIGN);
    bp->morton     = (uint32_t*)mi_realloc_aligned(bp->morton,     new_pad * sizeof(uint32_t), BP_NODE_ALIGN);
    bp->perm       = (uint32_t*)mi_realloc_aligned(bp->perm,       new_pad * sizeof(uint32_t), BP_NODE_ALIGN);
    bp->perm_alt   = (uint32_t*)mi_realloc_aligned(bp->perm_alt,   new_pad * sizeof(uint32_t), BP_NODE_ALIGN);
    bp->nodes      = (broadphase_node_t*)mi_realloc_aligned(
                         bp->nodes, new_node_cap * sizeof(broadphase_node_t), BP_NODE_ALIGN);

    if (!bp->item_ids   || !bp->objects    || !bp->item_min_x || !bp->item_min_y || !bp->item_min_z ||
        !bp->item_max_x || !bp->item_max_y || !bp->item_max_z ||
        !bp->morton     || !bp->perm       || !bp->perm_alt   || !bp->nodes) {
        fprintf(stderr, "broadphase: OOM grow (new_item_cap=%zu node_cap=%zu)\n",
                new_cap, new_node_cap);
        abort();
    }

    bp->item_cap   = (uint32_t)new_cap;
    bp->object_cap = (uint32_t)new_cap;
    bp->node_cap   = (uint32_t)new_node_cap;
}

void broadphase_insert(broadphase_t* bp, broadphase_object_t obj, aabb_t aabb) {
    assert(bp);
    if (bp->n_items >= bp->item_cap) {
        bp_grow(bp);
    }
    uint32_t i = bp->n_items++;
    bp->n_objects     = bp->n_items;
    bp->item_ids[i]   = obj.entity_id;
    bp->objects[i]    = obj;
    bp->item_min_x[i] = aabb.min.x; bp->item_max_x[i] = aabb.max.x;
    bp->item_min_y[i] = aabb.min.y; bp->item_max_y[i] = aabb.max.y;
    bp->item_min_z[i] = aabb.min.z; bp->item_max_z[i] = aabb.max.z;
    bp->has_tree = 0;
}

/* 4-pass × 8-bit LSD radix sort of perm by keys[perm[i]]. Covers full 32-bit
   keys. Stable. Even pass count → final scatter writes to perm directly, no
   trailing memcpy needed. */
static void bp_radix_sort_perm(const uint32_t* restrict keys,
    uint32_t* restrict perm,
    uint32_t* restrict perm_alt,
    uint32_t n) {

    uint32_t hist[4][256] = { {0} };

    // 1. Parallel Histogram Generation (Unrolled)
    for (uint32_t i = 0; i < n; i++) {
        uint32_t k = keys[perm[i]];
        hist[0][(k)       & 0xFFu]++;
        hist[1][(k >>  8) & 0xFFu]++;
        hist[2][(k >> 16) & 0xFFu]++;
        hist[3][(k >> 24) & 0xFFu]++;
    }

    // 2. Prefix Sum (Scan)
    for (int p = 0; p < 4; p++) {
        uint32_t sum = 0;
        for (int b = 0; b < 256; b++) {
            uint32_t count = hist[p][b];
            hist[p][b] = sum;
            sum += count;
        }
    }

    uint32_t* src = perm;
    uint32_t* dst = perm_alt;

    // 3. Four-Pass Scatter
    for (int p = 0; p < 4; p++) {
        uint32_t shift = (uint32_t)p * 8u;
        uint32_t* h = hist[p];

        for (uint32_t i = 0; i < n; i++) {
            uint32_t p_idx = src[i];
            uint32_t key = keys[p_idx];
            uint32_t bucket = (key >> shift) & 0xFFu;
            dst[h[bucket]++] = p_idx;
        }

        uint32_t* tmp = src;
        src = dst;
        dst = tmp;
    }
    // Even pass count: src points back to perm after final swap; sorted output
    // already in perm.
}

/* Pack one tree level. Reads child_lo..child_hi nodes (or items, when
   child_is_items=1), groups them 8-at-a-time into freshly appended internal
   (or leaf) nodes. Returns parent range as [out_lo, out_hi). */
static void bp_pack_level(broadphase_t* bp,
                          uint32_t  child_lo,
                          uint32_t  child_hi,
                          int       child_is_items,
                          uint32_t* out_lo,
                          uint32_t* out_hi) {
    *out_lo = bp->n_nodes;
    uint32_t c = child_lo;
    fixed_8_t sent_min = fixed8_set1(INT32_MAX);
    fixed_8_t sent_max = fixed8_set1(INT32_MIN);
    /* Hoist SoA bases out of the loop. restrict tells the compiler the SoA
       streams don't alias the node store, so reads can be vectorized through
       node writes. */
    const uint32_t* restrict perm  = bp->perm;
    const fixed_t*  restrict min_x = bp->item_min_x;
    const fixed_t*  restrict min_y = bp->item_min_y;
    const fixed_t*  restrict min_z = bp->item_min_z;
    const fixed_t*  restrict max_x = bp->item_max_x;
    const fixed_t*  restrict max_y = bp->item_max_y;
    const fixed_t*  restrict max_z = bp->item_max_z;
    while (c < child_hi) {
        assert(bp->n_nodes < bp->node_cap);
        broadphase_node_t* p = &bp->nodes[bp->n_nodes++];
        /* Prime all 8 lanes with sentinels; occupied slots overwrite below. */
        p->min_x = sent_min; p->max_x = sent_max;
        p->min_y = sent_min; p->max_y = sent_max;
        p->min_z = sent_min; p->max_z = sent_max;
        p->presence_mask = 0;
        p->object_mask   = 0;
        for (int s = 0; s < 8 && c < child_hi; s++, c++) {
            if (child_is_items) {
                /* Leaf: pull the original item via the sorted permutation. */
                uint32_t idx = perm[c];
                p->min_x.e[s] = min_x[idx];
                p->max_x.e[s] = max_x[idx];
                p->min_y.e[s] = min_y[idx];
                p->max_y.e[s] = max_y[idx];
                p->min_z.e[s] = min_z[idx];
                p->max_z.e[s] = max_z[idx];
                p->ids[s]     = idx;
                p->object_mask |= (uint8_t)(1u << s);
            } else {
                /* Internal: lane s collapses child c's 8 SoA lanes via
                   fixed8_min/fixed8_max horizontal reduction. Sentinels in
                   the child's unoccupied lanes are ignored automatically
                   (sentinel min = INT32_MAX dominated, sentinel max = INT32_MIN dominated). */
                const broadphase_node_t* ch = &bp->nodes[c];
                p->min_x.e[s] = fixed8_min(ch->min_x);
                p->max_x.e[s] = fixed8_max(ch->max_x);
                p->min_y.e[s] = fixed8_min(ch->min_y);
                p->max_y.e[s] = fixed8_max(ch->max_y);
                p->min_z.e[s] = fixed8_min(ch->min_z);
                p->max_z.e[s] = fixed8_max(ch->max_z);
                p->ids[s]     = c;
            }
            p->presence_mask |= (uint8_t)(1u << s);
        }
    }
    *out_hi = bp->n_nodes;
}

void broadphase_build(broadphase_t* bp) {
    assert(bp);
    bp->n_nodes  = 0;
    bp->has_tree = 0;
    uint32_t n = bp->n_items;
    if (n == 0) return;

    const fixed_t* restrict mn_x_a = bp->item_min_x;
    const fixed_t* restrict mn_y_a = bp->item_min_y;
    const fixed_t* restrict mn_z_a = bp->item_min_z;
    const fixed_t* restrict mx_x_a = bp->item_max_x;
    const fixed_t* restrict mx_y_a = bp->item_max_y;
    const fixed_t* restrict mx_z_a = bp->item_max_z;

    /* 1. World AABB. Six dense linear streams -- compiler turns the
          fixed_min/fixed_max chain into vmin/vmax SIMD loops with -O2 and
          AVX2/NEON enabled. The 6 reductions are independent; the compiler
          interleaves them to fill execution units. */
    fixed_t wmn_x = mn_x_a[0], wmx_x = mx_x_a[0];
    fixed_t wmn_y = mn_y_a[0], wmx_y = mx_y_a[0];
    fixed_t wmn_z = mn_z_a[0], wmx_z = mx_z_a[0];

    {
        uint32_t i = 0;

        while (i + 7 < n) {
            const fixed_8_t* mnx = (const fixed_8_t*)(mn_x_a + i);
            const fixed_8_t* mxx = (const fixed_8_t*)(mx_x_a + i);
            const fixed_8_t* mny = (const fixed_8_t*)(mn_y_a + i);
            const fixed_8_t* mxy = (const fixed_8_t*)(mx_y_a + i);
            const fixed_8_t* mnz = (const fixed_8_t*)(mn_z_a + i);
            const fixed_8_t* mxz = (const fixed_8_t*)(mx_z_a + i);

            wmn_x = fixed_min(wmn_x, fixed8_min(*mnx));
            wmx_x = fixed_max(wmx_x, fixed8_max(*mxx));

            wmn_y = fixed_min(wmn_y, fixed8_min(*mny));
            wmx_y = fixed_max(wmx_y, fixed8_max(*mxy));

            wmn_z = fixed_min(wmn_z, fixed8_min(*mnz));
            wmx_z = fixed_max(wmx_z, fixed8_max(*mxz));

            i += 8;
        }

        while (i < n) {
            wmn_x = fixed_min(wmn_x, mn_x_a[i]);
            wmx_x = fixed_max(wmx_x, mx_x_a[i]);
            wmn_y = fixed_min(wmn_y, mn_y_a[i]);
            wmx_y = fixed_max(wmx_y, mx_y_a[i]);
            wmn_z = fixed_min(wmn_z, mn_z_a[i]);
            wmx_z = fixed_max(wmx_z, mx_z_a[i]);

            i++;
        }
    }

    /* 2. Morton codes (centroid-quantized) and identity permutation. Range
          of 0 on any axis -> all centroids collapse to bin 0 on that axis,
          which is fine; the radix is stable so insertion order is preserved.
          Per-axis reciprocal precomputed once -> 3 i32 div hoisted out of
          the hot loop. Bit-spread is sequence-of-shifts -- not vector-
          friendly, leave scalar. */
    fixed_t rx = wmx_x - wmn_x; if (rx <= 0) rx = 1;
    fixed_t ry = wmx_y - wmn_y; if (ry <= 0) ry = 1;
    fixed_t rz = wmx_z - wmn_z; if (rz <= 0) rz = 1;
    uint64_t sx = ((uint64_t)(BP_MORTON_MAX + 1u) << 32) / (uint64_t)rx;
    uint64_t sy = ((uint64_t)(BP_MORTON_MAX + 1u) << 32) / (uint64_t)ry;
    uint64_t sz = ((uint64_t)(BP_MORTON_MAX + 1u) << 32) / (uint64_t)rz;
    uint32_t* restrict morton = bp->morton;
    uint32_t* restrict perm   = bp->perm;
    for (uint32_t i = 0; i < n; i++) {
        fixed_t cx = bp_avg(mn_x_a[i], mx_x_a[i]);
        fixed_t cy = bp_avg(mn_y_a[i], mx_y_a[i]);
        fixed_t cz = bp_avg(mn_z_a[i], mx_z_a[i]);
        uint32_t qx = bp_quantize_recip(cx, wmn_x, sx);
        uint32_t qy = bp_quantize_recip(cy, wmn_y, sy);
        uint32_t qz = bp_quantize_recip(cz, wmn_z, sz);
        morton[i] = bp_morton30(qx, qy, qz);
        perm[i]   = i;
    }

    /* 3. Sort permutation by morton. */
    bp_radix_sort_perm(bp->morton, bp->perm, bp->perm_alt, n);

    /* 4. Bottom-up pack. Leaves first from sorted items; then internal
          levels until one root remains. */
    uint32_t lo, hi;
    bp_pack_level(bp, 0, n, /*child_is_items=*/1, &lo, &hi);
    while (hi - lo > 1) {
        uint32_t plo, phi;
        bp_pack_level(bp, lo, hi, /*child_is_items=*/0, &plo, &phi);
        lo = plo; hi = phi;
    }
    bp->root     = lo;
    bp->has_tree = 1;
}

void broadphase_query_begin(broadphase_iter_t* it, const broadphase_t* bp, aabb_t q) {
    /* The struct has const-qualified `q` plus the cached splat lanes, so plain
       field assignment is illegal -- build the iterator on the stack and memcpy
       it across. The splats are computed exactly once here; bp_overlap_mask
       reuses them across every visited node. */
    broadphase_iter_t init = {
        .bp     = bp,
        .q      = q,
        .qmin_x = fixed8_set1(q.min.x), .qmax_x = fixed8_set1(q.max.x),
        .qmin_y = fixed8_set1(q.min.y), .qmax_y = fixed8_set1(q.max.y),
        .qmin_z = fixed8_set1(q.min.z), .qmax_z = fixed8_set1(q.max.z),
        .node   = NULL,
        .mask   = 0,
        .sp     = 0,
    };
    if (bp->has_tree && bp->n_nodes > 0) {
        init.stack[0] = bp->root;
        init.sp = 1;
    }
    memcpy(it, &init, sizeof(*it));
}

const broadphase_object_t* br(broadphase_iter_t* it) {
    /* Entered with it->mask == 0. Pop nodes from the stack and split each
       node's hit lanes by object_mask: child lanes get pushed for descent,
       object lanes are yielded (one now, rest stashed in it->mask). it->mask
       is always object-only -- the fast path never sees a child lane. */
    /* Hoist the cached splats out of the descent loop -- the compiler keeps
       them in SIMD registers across every bp_overlap_mask call. */
    const fixed_8_t qmin_x = it->qmin_x, qmax_x = it->qmax_x;
    const fixed_8_t qmin_y = it->qmin_y, qmax_y = it->qmax_y;
    const fixed_8_t qmin_z = it->qmin_z, qmax_z = it->qmax_z;
    while (it->sp > 0) {
        uint32_t idx = it->stack[--it->sp];
        const broadphase_node_t* n = &it->bp->nodes[idx];
        uint8_t m = bp_overlap_mask(n, qmin_x, qmax_x, qmin_y, qmax_y, qmin_z, qmax_z);
        if (!m) continue;
        uint8_t obj_hits   = (uint8_t)(m & n->object_mask);
        uint8_t child_hits = (uint8_t)(m & ~n->object_mask);
        /* Push every hit child lane. Stack depth bounded by tree depth
           (log_8 N) + 7 sibling pushes per descent step. BROADPHASE_STACK_MAX
           covers ~10^9 items. */
        while (child_hits) {
            int b = ecs_ctz32((uint32_t)child_hits);
            child_hits = (uint8_t)(child_hits & (child_hits - 1u));
            assert(it->sp < BROADPHASE_STACK_MAX);
            it->stack[it->sp++] = n->ids[b];
        }
        if (obj_hits) {
            int b = ecs_ctz32((uint32_t)obj_hits);
            const broadphase_object_t* obj = &it->bp->objects[n->ids[b]];
            it->node = n;
            it->mask = (uint8_t)(obj_hits & (obj_hits - 1u));
            return obj;
        }
    }
    it->node = NULL;
    it->mask = 0;
    return NULL;
}
