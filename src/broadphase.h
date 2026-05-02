#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include "ecs_math.h"

/* Uniform-grid broadphase. Cells are 4-metre cubes (BROADPHASE_CELL_LOG2 = 2,
   in real units; Q16.16 implementation shifts by FIXED_SHIFT + 2). Each cell
   owns a singly-linked list of nodes carved out of an internal bump arena;
   each node holds up to 8 items in SoA layout (six fixed_t[8] arrays for the
   AABB min/max plus an id[8]) so the per-node overlap test is one fixed8
   compare chain.

   Items are inserted by AABB centre only -- queries that need to find items
   whose extents poke into other cells must inflate the query AABB by the
   largest item half-extent before calling broadphase_query_begin.

   Clear is O(cells + 1): zero the head array and reset the arena offset; no
   per-node frees. The intended use is "rebuild from scratch every frame". */

#define BROADPHASE_CELL_LOG2  2
#define BROADPHASE_CELL_SIZE  (1 << BROADPHASE_CELL_LOG2)

/* fixed_8_t bundle plus an indexable view over the same 32 bytes. Loads
   pick the .simd lane for the SIMD overlap test; insertion writes a
   single slot through .e[i]. Backed by the same memory in every backend
   (sizeof(fixed_8_t) == 32). */
typedef union {
    fixed_8_t simd;
    fixed_t   e[8];
} bp_lane8_t;

/* SoA so each per-axis min/max occupies exactly one fixed8 register and
   the overlap test is six compares + five ANDs. _Alignas(32) on the
   first field promotes whole-struct alignment to 32 bytes (C11 rule:
   struct alignment >= max member alignment), keeping aligned loads
   legal on AVX2 across MSVC/clang/gcc; NEON and scalar are unaffected. */
typedef struct broadphase_node_t {
    _Alignas(32) bp_lane8_t min_x;
    bp_lane8_t max_x;
    bp_lane8_t min_y;
    bp_lane8_t max_y;
    bp_lane8_t min_z;
    bp_lane8_t max_z;
    uint8_t  presence_mask;            /* bit i set iff slot i occupied; full = 0xFF */
    uint32_t    ids[8];
    transform_t transforms[8];
    struct broadphase_node_t* next;
} broadphase_node_t;

typedef struct {
    uint8_t* base;
    size_t   cap;
    size_t   off;
} broadphase_arena_t;

typedef struct {
    vec3_t  origin;                    /* world-space corner of cell (0,0,0), Q16.16 */
    int32_t cells_x, cells_y, cells_z;
    broadphase_node_t** heads;         /* dense [cz][cy][cx], NULL = empty cell */
    broadphase_arena_t  arena;
} broadphase_t;

typedef struct {
    const broadphase_t* bp;
    const aabb_t q;                    /* cached for the per-node SIMD splat; init only via broadphase_query_begin */
    int32_t cx0, cy0, cz0;
    int32_t cx1, cy1, cz1;
    int32_t cx,  cy,  cz;              /* current cell */
    broadphase_node_t*  node;          /* current node within cell list */
    uint8_t  mask;                     /* unconsumed hits in current node */
} broadphase_iter_t;

/* size is the world extent in metres (Q16.16). cells_{x,y,z} = ceil(size/4),
   clamped to >=1. arena_cap is bytes; insertion past it aborts. */
void broadphase_init   (broadphase_t* bp, vec3_t origin, vec3_t size, size_t arena_cap);
void broadphase_destroy(broadphase_t* bp);

/* Reset to empty in O(cell_count): memset heads to NULL, arena offset to 0. */
void broadphase_clear  (broadphase_t* bp);

/* Inserts at the cell containing the AABB centre. Items whose centre falls
   outside the grid are silently dropped. */
void broadphase_insert (broadphase_t* bp, uint32_t id, aabb_t aabb);

/* Iterator yields each item id whose stored AABB overlaps q. The cell range
   is [floor(q.min/4) .. floor(q.max/4)] clamped to grid bounds. */
void broadphase_query_begin(broadphase_iter_t* it, const broadphase_t* bp, aabb_t q);

/* Slow path: advances node/cell until next non-empty hit mask, consumes its
   lowest bit, returns 1 (or 0 when exhausted). Inline fast path delegates
   here only when the cached mask is empty. */
int  br(broadphase_iter_t* it, uint32_t* out_id);

/* Fast path: pop the lowest set bit of the cached hit mask, return its id.
   Hot loop: branchless other than the empty-mask fallback. */
static inline int broadphase_query_next(broadphase_iter_t* it, uint32_t* out_id) {
    uint8_t m = it->mask;
    if (m) {
        int b = ecs_ctz32((uint32_t)m);
        *out_id = it->node->ids[b];
        it->mask = (uint8_t)(m & (m - 1u));
        return 1;
    }
    return br(it, out_id);
}
