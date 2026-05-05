#pragma once

#include <stdint.h>
#include "ecs_math.h"

/* Wide-8 LBVH broadphase. Items are inserted into a flat buffer; the tree is
   rebuilt from scratch every tick by broadphase_build:
     1. Scan items -> world AABB.
     2. Quantize each item centroid to a 30-bit morton code (10 bits/axis).
        Items whose centroid falls outside the world AABB cannot happen --
        the AABB was derived from the items themselves.
     3. 4-pass × 8-bit radix sort by morton.
     4. Bottom-up pack 8 sorted items per leaf node, then 8 child nodes per
        parent until one root remains.

   Each node is the same SoA shape as the prior grid implementation -- six
   fixed_8_t lanes for AABB min/max plus a presence_mask -- so the per-node
   overlap test is one fixed8 compare chain. Internal nodes reuse the SoA
   layout: each lane describes one child's AABB; ids[lane] holds the child
   node index. Leaves: ids[lane] holds the original item id.

   Clear is two stores (n_items=0, n_nodes=0) -- "array clear", no per-cell
   work. Same arrays reused next frame; no realloc churn. */

/* Self-bundled SoA node. _Alignas(32) on the first field promotes whole-
   struct alignment to 32 bytes (C11 rule: struct alignment >= max member
   alignment), keeping aligned loads legal on AVX2; NEON and scalar are
   unaffected. */
typedef struct broadphase_node_t {
    fixed_8_t min_x;
    fixed_8_t max_x;
    fixed_8_t min_y;
    fixed_8_t max_y;
    fixed_8_t min_z;
    fixed_8_t max_z;
    uint8_t   presence_mask;     /* bit i set iff slot i occupied */
    uint8_t   object_mask;       /* bit i set iff ids[i] indexes objects[]; (presence & ~object) bits index nodes[] */
    uint32_t  ids[8];
} broadphase_node_t;

typedef struct broadphase_object_t {
    transform_t transform;
    uint32_t entity_id;
} broadphase_object_t;

typedef struct {
    /* Item buffer: SoA, append-only between clear/build. Splitting AABBs
       into 6 fixed_t arrays lets the world-AABB / centroid scans run as
       dense linear streams the compiler turns into AVX2/NEON min/max loops. */
    uint32_t* item_ids;
    fixed_t*  item_min_x;
    fixed_t*  item_min_y;
    fixed_t*  item_min_z;
    fixed_t*  item_max_x;
    fixed_t*  item_max_y;
    fixed_t*  item_max_z;
    uint32_t  n_items;
    uint32_t  item_cap;

    /* Node arena: contiguous, indexed. Levels grow upward from index 0.
       n_nodes is also the count of valid nodes after build. */
    broadphase_node_t* nodes;
    uint32_t  n_nodes;
    uint32_t  node_cap;
    uint32_t  root;              /* index of root node (valid iff has_tree) */
    uint8_t   has_tree;

    /* Object arena: leaf node ids[] (where object_mask bit set) index here. */
    broadphase_object_t* objects;
    uint32_t  n_objects;
    uint32_t  object_cap;

    /* Sort scratch -- sized to item_cap. */
    uint32_t* morton;
    uint32_t* perm;
    uint32_t* perm_alt;
} broadphase_t;

/* Stack-based traversal. Depth is log_8(N) -- 11 for ~80M items, 32 plenty. */
#define BROADPHASE_STACK_MAX 32

typedef struct {
    const broadphase_t* bp;
    const aabb_t q;              /* original query AABB; init only via broadphase_query_begin */
    /* Pre-splatted query bounds. Splatting once at query_begin instead of on every
       per-node overlap call eliminates 6 SIMD broadcast ops per visited node. */
    const fixed_8_t qmin_x, qmax_x;
    const fixed_8_t qmin_y, qmax_y;
    const fixed_8_t qmin_z, qmax_z;
    const broadphase_node_t* node;   /* current node holding stashed object hits */
    uint8_t  mask;               /* unconsumed object-lane hits at it->node (child lanes already pushed) */
    int      sp;
    uint32_t stack[BROADPHASE_STACK_MAX];
} broadphase_iter_t;

/* item_cap caps how many items can be inserted between clears. The node
   arena and sort scratch are sized off it (one allocation each). */
void broadphase_init   (broadphase_t* bp, size_t item_cap);
void broadphase_destroy(broadphase_t* bp);

/* Reset to empty. Two stores -- the prior tick's nodes and items are simply
   overwritten on the next round of inserts/build. */
void broadphase_clear  (broadphase_t* bp);

/* Append (obj, aabb) to the item buffer. obj.id is the caller-facing external
   id; the tree's leaf ids[] index objects[] internally. The tree is invalidated;
   the caller must call broadphase_build before query_begin. */
void broadphase_insert (broadphase_t* bp, broadphase_object_t obj, aabb_t aabb);

/* Build the tree from the current item buffer. Must be called between the
   last insert of the tick and the first query. Cheap to call on an empty
   buffer (no-op marking has_tree=0). */
void broadphase_build  (broadphase_t* bp);

/* Iterator yields each item id whose stored AABB overlaps q. has_tree must
   be 1 (i.e. broadphase_build was called since the last clear). */
void broadphase_query_begin(broadphase_iter_t* it, const broadphase_t* bp, aabb_t q);

/* Slow path: pop from the descent stack, push hit children for internal
   nodes, stop on the first leaf with a non-empty hit mask. Returns NULL when
   the iterator is drained. */
const broadphase_object_t* br(broadphase_iter_t* it);

/* Fast path: pop the lowest set bit of the cached leaf hit mask, return its
   object. Hot loop: branchless other than the empty-mask fallback. NULL means
   no more hits. */
static inline const broadphase_object_t* broadphase_query_next(broadphase_iter_t* it) {
    uint8_t m = it->mask;
    if (m) {
        int b = ecs_ctz32((uint32_t)m);
        const broadphase_object_t* obj = &it->bp->objects[it->node->ids[b]];
        it->mask = (uint8_t)(m & (m - 1u));
        return obj;
    }
    return br(it);
}
