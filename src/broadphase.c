#include "broadphase.h"

#include <assert.h>
#include <mimalloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BP_NODE_ALIGN 32

/* Floor((coord - origin) / cell_size_metres). The combined shift is
   FIXED_SHIFT (strip Q16.16 fraction) + CELL_LOG2 (divide by 4 metres). The
   subtraction goes through uint32 to avoid signed-overflow UB if the world
   straddles fixed_t's range; the cast back to int32 + arithmetic right shift
   produces the correct floor on two's-complement platforms. */
static inline int32_t bp_cell_axis(fixed_t coord, fixed_t origin) {
    int32_t d = (int32_t)((uint32_t)coord - (uint32_t)origin);
    return d >> (FIXED_SHIFT + BROADPHASE_CELL_LOG2);
}

/* Average via int64 to avoid int32 overflow when both endpoints sit near the
   Q16.16 extremes. */
static inline fixed_t bp_avg(fixed_t a, fixed_t b) {
    return (fixed_t)(((int64_t)a + (int64_t)b) >> 1);
}

static inline size_t bp_align_up(size_t v, size_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

static inline size_t bp_index(const broadphase_t* bp, int32_t cx, int32_t cy, int32_t cz) {
    return (size_t)cz * (size_t)bp->cells_x * (size_t)bp->cells_y
         + (size_t)cy * (size_t)bp->cells_x
         + (size_t)cx;
}

static broadphase_node_t* bp_arena_alloc_node(broadphase_arena_t* a) {
    size_t off = bp_align_up(a->off, BP_NODE_ALIGN);
    if (off + sizeof(broadphase_node_t) > a->cap) {
        fprintf(stderr, "broadphase: arena exhausted (cap=%zu)\n", a->cap);
        abort();
    }
    broadphase_node_t* n = (broadphase_node_t*)(a->base + off);
    a->off = off + sizeof(broadphase_node_t);
    n->presence_mask = 0;
    n->next = NULL;
    return n;
}

void broadphase_init(broadphase_t* bp, vec3_t origin, vec3_t size, size_t arena_cap) {
    assert(bp);
    bp->origin = origin;

    fixed_t cs = fixed_from_int(BROADPHASE_CELL_SIZE);
    /* ceil(size / cs) on Q16.16 fixed, both sides scaled by 2^16 cancel under
       integer division. */
    int32_t nx = (int32_t)((size.x + cs - 1) / cs);
    int32_t ny = (int32_t)((size.y + cs - 1) / cs);
    int32_t nz = (int32_t)((size.z + cs - 1) / cs);
    if (nx < 1) nx = 1;
    if (ny < 1) ny = 1;
    if (nz < 1) nz = 1;
    bp->cells_x = nx;
    bp->cells_y = ny;
    bp->cells_z = nz;

    size_t n_cells = (size_t)nx * (size_t)ny * (size_t)nz;
    bp->heads = (broadphase_node_t**)mi_calloc(n_cells, sizeof(broadphase_node_t*));
    if (!bp->heads) { fprintf(stderr, "broadphase: OOM heads\n"); abort(); }

    bp->arena.cap  = arena_cap;
    bp->arena.off  = 0;
    bp->arena.base = (uint8_t*)mi_malloc_aligned(arena_cap, BP_NODE_ALIGN);
    if (!bp->arena.base) { fprintf(stderr, "broadphase: OOM arena\n"); abort(); }
}

void broadphase_destroy(broadphase_t* bp) {
    assert(bp);
    if (bp->heads) mi_free(bp->heads);
    if (bp->arena.base) mi_free(bp->arena.base);
    bp->heads = NULL;
    bp->arena.base = NULL;
    bp->arena.cap = 0;
    bp->arena.off = 0;
    bp->cells_x = bp->cells_y = bp->cells_z = 0;
}

void broadphase_clear(broadphase_t* bp) {
    assert(bp);
    size_t n_cells = (size_t)bp->cells_x * (size_t)bp->cells_y * (size_t)bp->cells_z;
    memset(bp->heads, 0, n_cells * sizeof(broadphase_node_t*));
    bp->arena.off = 0;
}

void broadphase_insert(broadphase_t* bp, uint32_t id, aabb_t aabb) {
    fixed_t cx_f = bp_avg(aabb.min.x, aabb.max.x);
    fixed_t cy_f = bp_avg(aabb.min.y, aabb.max.y);
    fixed_t cz_f = bp_avg(aabb.min.z, aabb.max.z);
    int32_t cx = bp_cell_axis(cx_f, bp->origin.x);
    int32_t cy = bp_cell_axis(cy_f, bp->origin.y);
    int32_t cz = bp_cell_axis(cz_f, bp->origin.z);
    if (cx < 0 || cx >= bp->cells_x ||
        cy < 0 || cy >= bp->cells_y ||
        cz < 0 || cz >= bp->cells_z) return;

    size_t idx = bp_index(bp, cx, cy, cz);
    broadphase_node_t* head = bp->heads[idx];
    if (!head || head->presence_mask == (uint8_t)0xFFu) {
        broadphase_node_t* fresh = bp_arena_alloc_node(&bp->arena);
        fresh->next = head;
        bp->heads[idx] = fresh;
        head = fresh;
    }
    /* Lowest free slot. ctz on the inverted mask; inverted is non-zero since
       the node was guaranteed not-full above. */
    int i = ecs_ctz32((uint32_t)(uint8_t)~head->presence_mask);
    head->presence_mask |= (uint8_t)(1u << i);
    head->min_x[i] = aabb.min.x; head->max_x[i] = aabb.max.x;
    head->min_y[i] = aabb.min.y; head->max_y[i] = aabb.max.y;
    head->min_z[i] = aabb.min.z; head->max_z[i] = aabb.max.z;
    head->ids[i]   = id;
}

/* SIMD overlap: 8 items vs 1 query AABB. Six fixed8 compares + five ANDs.
   AND with presence_mask zeros out unoccupied lanes so undefined slot data
   cannot produce false hits. */
static inline uint8_t bp_overlap_mask(const broadphase_node_t* n, aabb_t q) {
    fixed_8_t nmin_x = fixed8_load_aligned(n->min_x);
    fixed_8_t nmax_x = fixed8_load_aligned(n->max_x);
    fixed_8_t nmin_y = fixed8_load_aligned(n->min_y);
    fixed_8_t nmax_y = fixed8_load_aligned(n->max_y);
    fixed_8_t nmin_z = fixed8_load_aligned(n->min_z);
    fixed_8_t nmax_z = fixed8_load_aligned(n->max_z);
    fixed_8_t qmin_x = fixed8_set1(q.min.x), qmax_x = fixed8_set1(q.max.x);
    fixed_8_t qmin_y = fixed8_set1(q.min.y), qmax_y = fixed8_set1(q.max.y);
    fixed_8_t qmin_z = fixed8_set1(q.min.z), qmax_z = fixed8_set1(q.max.z);

    uint8_t m = fixed8_le(nmin_x, qmax_x) & fixed8_ge(nmax_x, qmin_x)
              & fixed8_le(nmin_y, qmax_y) & fixed8_ge(nmax_y, qmin_y)
              & fixed8_le(nmin_z, qmax_z) & fixed8_ge(nmax_z, qmin_z);
    return m & n->presence_mask;
}

void broadphase_query_begin(broadphase_iter_t* it, const broadphase_t* bp, aabb_t q) {
    int32_t cx0 = bp_cell_axis(q.min.x, bp->origin.x);
    int32_t cy0 = bp_cell_axis(q.min.y, bp->origin.y);
    int32_t cz0 = bp_cell_axis(q.min.z, bp->origin.z);
    int32_t cx1 = bp_cell_axis(q.max.x, bp->origin.x);
    int32_t cy1 = bp_cell_axis(q.max.y, bp->origin.y);
    int32_t cz1 = bp_cell_axis(q.max.z, bp->origin.z);
    if (cx0 < 0) cx0 = 0;
    if (cy0 < 0) cy0 = 0;
    if (cz0 < 0) cz0 = 0;
    if (cx1 >= bp->cells_x) cx1 = bp->cells_x - 1;
    if (cy1 >= bp->cells_y) cy1 = bp->cells_y - 1;
    if (cz1 >= bp->cells_z) cz1 = bp->cells_z - 1;

    broadphase_node_t* node0 = NULL;
    uint8_t mask0 = 0;
    int range_ok = (cx0 <= cx1 && cy0 <= cy1 && cz0 <= cz1);
    if (range_ok) {
        node0 = bp->heads[bp_index(bp, cx0, cy0, cz0)];
        if (node0) mask0 = bp_overlap_mask(node0, q);
    } else {
        /* Force the slow path's advance to fall straight through to the
           "exhausted" branch: cx >= cx1, cy >= cy1, cz >= cz1. */
        cx0 = cy0 = cz0 = 0;
        cx1 = cy1 = cz1 = -1;
    }

    /* The struct has a const-qualified `q`, so plain field assignment is
       illegal -- build the iterator on the stack and memcpy it across. */
    broadphase_iter_t init = {
        .bp = bp,
        .q  = q,
        .cx0 = cx0, .cy0 = cy0, .cz0 = cz0,
        .cx1 = cx1, .cy1 = cy1, .cz1 = cz1,
        .cx  = cx0, .cy  = cy0, .cz  = cz0,
        .node = node0,
        .mask = mask0,
    };
    memcpy(it, &init, sizeof(*it));
}

int br(broadphase_iter_t* it, uint32_t* out_id) {
    /* Entered with it->mask == 0. Advance node-then-cell until a non-empty
       overlap mask appears; consume its lowest bit and return. */
    for (;;) {
        if (it->node && it->node->next) {
            it->node = it->node->next;
        } else {
            /* x is the inner axis -- matches the (cz, cy, cx) row-major head
               layout for cache-friendly stepping. */
            if (it->cx < it->cx1) {
                it->cx++;
            } else if (it->cy < it->cy1) {
                it->cx = it->cx0;
                it->cy++;
            } else if (it->cz < it->cz1) {
                it->cx = it->cx0;
                it->cy = it->cy0;
                it->cz++;
            } else {
                it->node = NULL;
                it->mask = 0;
                return 0;
            }
            it->node = it->bp->heads[bp_index(it->bp, it->cx, it->cy, it->cz)];
        }
        if (!it->node) continue;
        uint8_t m = bp_overlap_mask(it->node, it->q);
        if (!m) continue;
        int b = ecs_ctz32((uint32_t)m);
        *out_id = it->node->ids[b];
        it->mask = (uint8_t)(m & (m - 1u));
        return 1;
    }
}
