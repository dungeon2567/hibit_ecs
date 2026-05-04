#include "broadphase.h"

#include <assert.h>
#include <mimalloc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BP_NODE_ALIGN  32
#define BP_MORTON_BITS 10                       /* per axis -> 30-bit code */
#define BP_MORTON_MAX  ((1u << BP_MORTON_BITS) - 1u)

/* Average via int64 to avoid int32 overflow when both endpoints sit near the
   Q16.16 extremes. */
static inline fixed_t bp_avg(fixed_t a, fixed_t b) {
    return (fixed_t)(((int64_t)a + (int64_t)b) >> 1);
}

/* Quantize centroid into [0, 2^BP_MORTON_BITS). The world AABB is derived
   from the items themselves, so v is always in [origin, origin+range]; the
   clamp is defensive against the upper edge (q == 2^N) only. */
static inline uint32_t bp_quantize(fixed_t v, fixed_t origin, fixed_t range) {
    int64_t d = (int64_t)v - (int64_t)origin;
    if (d <= 0) return 0;
    int64_t q = (d * (int64_t)(BP_MORTON_MAX + 1u)) / (int64_t)range;
    if (q < 0) return 0;
    if (q > (int64_t)BP_MORTON_MAX) return BP_MORTON_MAX;
    return (uint32_t)q;
}

/* Spread the low 10 bits of x across every third bit slot. Standard magic-
   constant interleave; produces 0b001 001 001 ... when x=0x3FF. */
static inline uint32_t bp_morton_part(uint32_t x) {
    x &= 0x3FFu;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x <<  8)) & 0x0300F00Fu;
    x = (x | (x <<  4)) & 0x030C30C3u;
    x = (x | (x <<  2)) & 0x09249249u;
    return x;
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
   five ANDs. AND with presence_mask zeros out unoccupied lanes so undefined
   slot data cannot produce false hits. Same routine works on internal nodes
   (lane = child AABB) and leaves (lane = item AABB). */
static inline uint8_t bp_overlap_mask(const broadphase_node_t* n, aabb_t q) {
    fixed_8_t qmin_x = fixed8_set1(q.min.x), qmax_x = fixed8_set1(q.max.x);
    fixed_8_t qmin_y = fixed8_set1(q.min.y), qmax_y = fixed8_set1(q.max.y);
    fixed_8_t qmin_z = fixed8_set1(q.min.z), qmax_z = fixed8_set1(q.max.z);

    uint8_t m = fixed8_le(n->min_x, qmax_x) & fixed8_ge(n->max_x, qmin_x)
              & fixed8_le(n->min_y, qmax_y) & fixed8_ge(n->max_y, qmin_y)
              & fixed8_le(n->min_z, qmax_z) & fixed8_ge(n->max_z, qmin_z);
    return m & n->presence_mask;
}

void broadphase_init(broadphase_t* bp, size_t item_cap) {
    assert(bp);
    if (item_cap == 0) item_cap = 1;

    bp->item_cap = (uint32_t)item_cap;
    bp->n_items  = 0;

    /* Node count upper bound: leaves = ceil(N/8); internals form a geometric
       sum that converges below leaves/7. ceil(N/7) + 8 covers it for any N. */
    size_t node_cap = item_cap / 7 + 8;
    bp->node_cap = (uint32_t)node_cap;
    bp->n_nodes  = 0;
    bp->root     = 0;
    bp->has_tree = 0;

    bp->item_ids   = (uint32_t*)mi_malloc(item_cap * sizeof(uint32_t));
    bp->item_aabbs = (aabb_t*)  mi_malloc(item_cap * sizeof(aabb_t));
    bp->morton     = (uint32_t*)mi_malloc(item_cap * sizeof(uint32_t));
    bp->perm       = (uint32_t*)mi_malloc(item_cap * sizeof(uint32_t));
    bp->perm_alt   = (uint32_t*)mi_malloc(item_cap * sizeof(uint32_t));
    bp->nodes      = (broadphase_node_t*)mi_malloc_aligned(
                         node_cap * sizeof(broadphase_node_t), BP_NODE_ALIGN);

    if (!bp->item_ids || !bp->item_aabbs || !bp->morton ||
        !bp->perm || !bp->perm_alt || !bp->nodes) {
        fprintf(stderr, "broadphase: OOM (item_cap=%zu node_cap=%zu)\n",
                item_cap, node_cap);
        abort();
    }
}

void broadphase_destroy(broadphase_t* bp) {
    assert(bp);
    if (bp->item_ids)   mi_free(bp->item_ids);
    if (bp->item_aabbs) mi_free(bp->item_aabbs);
    if (bp->morton)     mi_free(bp->morton);
    if (bp->perm)       mi_free(bp->perm);
    if (bp->perm_alt)   mi_free(bp->perm_alt);
    if (bp->nodes)      mi_free(bp->nodes);
    bp->item_ids = bp->morton = bp->perm = bp->perm_alt = NULL;
    bp->item_aabbs = NULL;
    bp->nodes = NULL;
    bp->item_cap = bp->n_items = 0;
    bp->node_cap = bp->n_nodes = 0;
    bp->has_tree = 0;
}

void broadphase_clear(broadphase_t* bp) {
    assert(bp);
    bp->n_items  = 0;
    bp->n_nodes  = 0;
    bp->has_tree = 0;
}

void broadphase_insert(broadphase_t* bp, uint32_t id, aabb_t aabb) {
    assert(bp);
    if (bp->n_items >= bp->item_cap) {
        fprintf(stderr, "broadphase: item buffer full (cap=%u)\n", bp->item_cap);
        abort();
    }
    uint32_t i = bp->n_items++;
    bp->item_ids[i]   = id;
    bp->item_aabbs[i] = aabb;
    bp->has_tree = 0;
}

/* 4-pass × 8-bit LSD radix sort of perm by morton[perm[i]]. After 4 even-
   indexed passes, the sorted output lives back in perm; perm_alt is scratch.
   Stable, so equal-morton items keep insertion order. */
static void bp_radix_sort_perm(const uint32_t* keys,
                               uint32_t* perm,
                               uint32_t* perm_alt,
                               uint32_t n) {
    uint32_t hist[4][256] = {{0}};
    for (uint32_t i = 0; i < n; i++) {
        uint32_t k = keys[perm[i]];
        hist[0][(k >>  0) & 0xFFu]++;
        hist[1][(k >>  8) & 0xFFu]++;
        hist[2][(k >> 16) & 0xFFu]++;
        hist[3][(k >> 24) & 0xFFu]++;
    }
    for (int p = 0; p < 4; p++) {
        uint32_t s = 0;
        for (int b = 0; b < 256; b++) {
            uint32_t c = hist[p][b];
            hist[p][b] = s;
            s += c;
        }
    }

    uint32_t* src = perm;
    uint32_t* dst = perm_alt;
    for (int p = 0; p < 4; p++) {
        uint32_t shift = (uint32_t)(p * 8);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t idx = src[i];
            uint32_t b   = (keys[idx] >> shift) & 0xFFu;
            dst[hist[p][b]++] = idx;
        }
        uint32_t* tmp = src; src = dst; dst = tmp;
    }
    /* 4 passes => result lives in perm again. Defensive copy keeps the
       contract clean if someone later changes pass count. */
    if (src != perm) memcpy(perm, src, n * sizeof(uint32_t));
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
    while (c < child_hi) {
        assert(bp->n_nodes < bp->node_cap);
        broadphase_node_t* p = &bp->nodes[bp->n_nodes++];
        /* Prime all 8 lanes with sentinels; occupied slots overwrite below. */
        p->min_x = sent_min; p->max_x = sent_max;
        p->min_y = sent_min; p->max_y = sent_max;
        p->min_z = sent_min; p->max_z = sent_max;
        p->presence_mask = 0;
        p->is_leaf = (uint8_t)(child_is_items ? 1 : 0);
        for (int s = 0; s < 8 && c < child_hi; s++, c++) {
            if (child_is_items) {
                /* Leaf: pull the original item via the sorted permutation. */
                uint32_t idx = bp->perm[c];
                aabb_t   a   = bp->item_aabbs[idx];
                p->min_x.e[s] = a.min.x; p->max_x.e[s] = a.max.x;
                p->min_y.e[s] = a.min.y; p->max_y.e[s] = a.max.y;
                p->min_z.e[s] = a.min.z; p->max_z.e[s] = a.max.z;
                p->ids[s]     = bp->item_ids[idx];
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

    /* 1. World AABB from items. */
    aabb_t w;
    w.min.x = w.min.y = w.min.z = INT32_MAX;
    w.max.x = w.max.y = w.max.z = INT32_MIN;
    for (uint32_t i = 0; i < n; i++) {
        aabb_t a = bp->item_aabbs[i];
        if (a.min.x < w.min.x) w.min.x = a.min.x;
        if (a.min.y < w.min.y) w.min.y = a.min.y;
        if (a.min.z < w.min.z) w.min.z = a.min.z;
        if (a.max.x > w.max.x) w.max.x = a.max.x;
        if (a.max.y > w.max.y) w.max.y = a.max.y;
        if (a.max.z > w.max.z) w.max.z = a.max.z;
    }

    /* 2. Morton codes (centroid-quantized) and identity permutation. Range
          of 0 on any axis -> all centroids collapse to bin 0 on that axis,
          which is fine; the radix is stable so insertion order is preserved. */
    fixed_t rx = w.max.x - w.min.x; if (rx <= 0) rx = 1;
    fixed_t ry = w.max.y - w.min.y; if (ry <= 0) ry = 1;
    fixed_t rz = w.max.z - w.min.z; if (rz <= 0) rz = 1;
    for (uint32_t i = 0; i < n; i++) {
        aabb_t a  = bp->item_aabbs[i];
        fixed_t cx = bp_avg(a.min.x, a.max.x);
        fixed_t cy = bp_avg(a.min.y, a.max.y);
        fixed_t cz = bp_avg(a.min.z, a.max.z);
        uint32_t qx = bp_quantize(cx, w.min.x, rx);
        uint32_t qy = bp_quantize(cy, w.min.y, ry);
        uint32_t qz = bp_quantize(cz, w.min.z, rz);
        bp->morton[i] = bp_morton30(qx, qy, qz);
        bp->perm[i]   = i;
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
    /* The struct has a const-qualified `q`, so plain field assignment is
       illegal -- build the iterator on the stack and memcpy it across. */
    broadphase_iter_t init = {
        .bp   = bp,
        .q    = q,
        .node = NULL,
        .mask = 0,
        .sp   = 0,
    };
    if (bp->has_tree && bp->n_nodes > 0) {
        init.stack[0] = bp->root;
        init.sp = 1;
    }
    memcpy(it, &init, sizeof(*it));
}

int br(broadphase_iter_t* it, uint32_t* out_id) {
    /* Entered with it->mask == 0. Pop nodes from the stack: internal -> push
       hit children; leaf -> stash on iterator and yield first hit. */
    while (it->sp > 0) {
        uint32_t idx = it->stack[--it->sp];
        const broadphase_node_t* n = &it->bp->nodes[idx];
        uint8_t m = bp_overlap_mask(n, it->q);
        if (!m) continue;
        if (n->is_leaf) {
            int b = ecs_ctz32((uint32_t)m);
            *out_id  = n->ids[b];
            it->node = n;
            it->mask = (uint8_t)(m & (m - 1u));
            return 1;
        }
        /* Internal: push every hit child. Stack depth bounded by tree depth
           (log_8 N) + 7 sibling pushes per descent step. BROADPHASE_STACK_MAX
           covers ~10^9 items. */
        while (m) {
            int b = ecs_ctz32((uint32_t)m);
            m = (uint8_t)(m & (m - 1u));
            assert(it->sp < BROADPHASE_STACK_MAX);
            it->stack[it->sp++] = n->ids[b];
        }
    }
    it->node = NULL;
    it->mask = 0;
    return 0;
}
