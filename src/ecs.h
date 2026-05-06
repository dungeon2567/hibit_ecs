#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "ecs_math.h"
#include "ecs_serializer.h"

/* Allocator wrappers. 'x' prefix = abort on OOM (xmalloc convention) --
   every allocation site treats failure as fatal so callers don't check.
   ecs_free is a thin pass-through for symmetry.

   ECS_NO_MIMALLOC (set on Debug builds): fall back to libc malloc.
   Picks the right aligned-alloc on each platform so all allocations are
   consistently freed by the matching free fn. Lets Valgrind / ASan see
   real heap state (mimalloc's pool layout confuses memcheck). */

#ifdef ECS_NO_MIMALLOC

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
  #include <malloc.h>
  #define ECS_NMI_ALIGNED_MALLOC(sz, al)        _aligned_malloc((sz), (al))
  #define ECS_NMI_ALIGNED_REALLOC(p, sz, al)    _aligned_realloc((p), (sz), (al))
  #define ECS_NMI_ALIGNED_FREE(p)               _aligned_free(p)
#else
  static inline void* ecs__nmi_aligned_malloc(size_t sz, size_t al) {
      /* C11 aligned_alloc requires size multiple of align. */
      size_t rounded = (sz + al - 1u) & ~(al - 1u);
      return aligned_alloc(al, rounded);
  }
  static inline void* ecs__nmi_aligned_realloc(void* p, size_t sz, size_t al) {
      /* No posix aligned_realloc. glibc malloc is 16B-aligned, so plain
         realloc keeps that for al <= 16. All call sites use al <= 16. */
      (void)al;
      assert(al <= 16u);
      return realloc(p, sz);
  }
  #define ECS_NMI_ALIGNED_MALLOC(sz, al)        ecs__nmi_aligned_malloc((sz), (al))
  #define ECS_NMI_ALIGNED_REALLOC(p, sz, al)    ecs__nmi_aligned_realloc((p), (sz), (al))
  #define ECS_NMI_ALIGNED_FREE(p)               free(p)
#endif

/* All ECS_NO_MIMALLOC allocations route through the aligned API so a
   single free fn handles everything. Non-aligned paths default to 8B
   alignment (matches mimalloc's default). */
static inline void* ecs_xmalloc_aligned(size_t size, size_t align) {
    void* p = ECS_NMI_ALIGNED_MALLOC(size, align);
    if (!p) { fprintf(stderr, "ecs: OOM xmalloc_aligned(%zu, %zu)\n", size, align); abort(); }
    return p;
}
static inline void* ecs_xcalloc(size_t n, size_t size) {
    size_t total = n * size;
    void* p = ECS_NMI_ALIGNED_MALLOC(total, 8u);
    if (!p) { fprintf(stderr, "ecs: OOM xcalloc(%zu, %zu)\n", n, size); abort(); }
    memset(p, 0, total);
    return p;
}
static inline void ecs_free(void* p) { ECS_NMI_ALIGNED_FREE(p); }
static inline void* ecs_xrealloc(void* p, size_t size) {
    void* q = ECS_NMI_ALIGNED_REALLOC(p, size, 8u);
    if (!q) { fprintf(stderr, "ecs: OOM xrealloc(%zu)\n", size); abort(); }
    return q;
}
static inline void* ecs_xrealloc_aligned(void* p, size_t size, size_t align) {
    void* q = ECS_NMI_ALIGNED_REALLOC(p, size, align);
    if (!q) { fprintf(stderr, "ecs: OOM xrealloc_aligned(%zu, %zu)\n", size, align); abort(); }
    return q;
}

#else

#include <mimalloc.h>

static inline void* ecs_xmalloc_aligned(size_t size, size_t align) {
    void* p = mi_malloc_aligned(size, align);
    if (!p) { fprintf(stderr, "ecs: OOM xmalloc_aligned(%zu, %zu)\n", size, align); abort(); }
    return p;
}
static inline void* ecs_xcalloc(size_t n, size_t size) {
    void* p = mi_calloc(n, size);
    if (!p) { fprintf(stderr, "ecs: OOM xcalloc(%zu, %zu)\n", n, size); abort(); }
    return p;
}
static inline void ecs_free(void* p) { mi_free(p); }
static inline void* ecs_xrealloc(void* p, size_t size) {
    void* q = mi_realloc(p, size);
    if (!q) { fprintf(stderr, "ecs: OOM xrealloc(%zu)\n", size); abort(); }
    return q;
}
static inline void* ecs_xrealloc_aligned(void* p, size_t size, size_t align) {
    void* q = mi_realloc_aligned(p, size, align);
    if (!q) { fprintf(stderr, "ecs: OOM xrealloc_aligned(%zu, %zu)\n", size, align); abort(); }
    return q;
}

#endif

/* ==========================================================================
   ecs_buffer_t -- typed dynamic array. Wraps a single `ecs_buffer_header_t*`
   that owns size + capacity + flexible-array data in one allocation. Both
   `size` and `capacity` are byte counts (NOT element counts) -- the list is
   element-size-agnostic at the storage level. Caller passes elem_size to
   every op that needs to address elements; the same value must be used for
   every call on a given buffer (no runtime check). Wrapper makes
   `ecs_buffer_t l = {0};` a valid empty buffer (h == NULL) and keeps the public
   type stable as a value type. Mutating ops take `ecs_buffer_t*` because
   growth may reallocate the header.
   Growth: 1.5Ãƒ- from base 8 bytes. push memcpy's the value (or leaves the
   slot uninitialized when value == NULL, returning the slot ptr).

   Usage:
       ecs_buffer_t xs = {0};
       ecs_buffer_push(&xs, sizeof(int), &(int){42});
       int* p = (int*)ecs_buffer_at(xs, sizeof(int), 0);
       ecs_buffer_destroy(&xs);
   ========================================================================== */
/* _Alignas(8) on size forces 8-byte alignment on the whole struct, which
   guarantees the FAM `data` (sitting at offset 8) is also 8-byte aligned --
   safe for any element up to 8-byte natural alignment without padding. */
typedef struct ecs_buffer_header_t {
    _Alignas(8) uint32_t size;              /* bytes used */
    uint32_t capacity;                      /* bytes allocated for data */
    char     data[];                        /* flexible array -- element bytes inline */
} ecs_buffer_header_t;

typedef struct ecs_buffer_t {
    ecs_buffer_header_t* h;                   /* NULL = empty */
} ecs_buffer_t;

static inline uint32_t ecs_buffer_size    (ecs_buffer_t l) { return l.h ? l.h->size     : 0u; }
static inline uint32_t ecs_buffer_capacity(ecs_buffer_t l) { return l.h ? l.h->capacity : 0u; }

static inline void ecs_buffer_destroy(ecs_buffer_t* l) {
    assert(l);
    if (l->h) ecs_free(l->h);
    l->h = NULL;
}

/* Reserve at least min_cap_bytes of data capacity. */
static inline void ecs_buffer_reserve(ecs_buffer_t* l, uint32_t min_cap_bytes) {
    assert(l);
    uint32_t cap = l->h ? l->h->capacity : 0u;
    if (min_cap_bytes <= cap) return;
    uint32_t newcap = cap ? cap : 8u;
    while (newcap < min_cap_bytes) newcap = newcap + (newcap >> 1) + 1u;  /* 1.5Ãƒ- */
    ecs_buffer_header_t* nh = (ecs_buffer_header_t*)ecs_xrealloc_aligned(
        l->h, sizeof(ecs_buffer_header_t) + (size_t)newcap, 8);
    if (!l->h) nh->size = 0u;               /* fresh alloc -- size was uninitialized */
    nh->capacity = newcap;
    l->h = nh;
}

static inline void ecs_buffer_clear(ecs_buffer_t l) { if (l.h) l.h->size = 0u; }

static inline void* ecs_buffer_at(ecs_buffer_t l, size_t elem_size, uint32_t i) {
    assert(l.h && elem_size && (size_t)i * elem_size < l.h->size);
    return (char*)l.h->data + (size_t)i * elem_size;
}

static inline void* ecs_buffer_push(ecs_buffer_t* l, size_t elem_size, const void* value) {
    assert(l && elem_size);
    uint32_t cur_size = l->h ? l->h->size : 0u;
    uint32_t need     = cur_size + (uint32_t)elem_size;
    if (!l->h || need > l->h->capacity) ecs_buffer_reserve(l, need);
    void* dst = (char*)l->h->data + cur_size;
    if (value) memcpy(dst, value, elem_size);
    l->h->size = need;
    return dst;
}

static inline void ecs_buffer_pop(ecs_buffer_t l, size_t elem_size) {
    assert(l.h && elem_size && l.h->size >= elem_size);
    l.h->size -= (uint32_t)elem_size;
}

/* O(1) unordered remove: copy last element over slot i, shrink size. */
static inline void ecs_buffer_swap_remove(ecs_buffer_t l, size_t elem_size, uint32_t i) {
    assert(l.h && elem_size);
    uint32_t off = (uint32_t)((size_t)i * elem_size);
    assert(off < l.h->size);
    uint32_t last = l.h->size - (uint32_t)elem_size;
    l.h->size = last;
    if (off != last) memcpy((char*)l.h->data + off, (char*)l.h->data + last, elem_size);
}

/* ==========================================================================
   Predict-mode ECS - two buffers per L1 (confirmed + predicted), no undo log.

     confirmed_mask_any  = authoritative presence (advances only via CONFIRMED writes)
     predicted_mask_any  = live working-set this frame (predicted is ALWAYS speculative)
     dirty               = slots written in PREDICT mode since last rollback

   Inline data tail = 128 slots: data[0..63] confirmed, data[64..127] predicted.
   Slot byte validity:
     - confirmed[i]: valid while bit i set in confirmed_mask_any
     - predicted[i]: valid while bit i set in dirty (stale otherwise)

   Predicted state never promotes. To advance confirmed state, write in
   CONFIRMED mode (server msgs / authoritative sim).

   ecs_tree_rollback = tick-end. predicted_mask = confirmed_mask, dirty = 0,
                       changed = 0, releases empty L1/L2 nodes. Returns 1 iff
                       a CONFIRMED-mode write landed this cycle, 0 otherwise.
                       (Predicted bytes go stale; dirty = 0 forbids reads.)
   ========================================================================== */

/* Global VM mode. Encoded as multiplier:
     CONFIRMED = 0  -> writes go to confirmed slot, no dirty bit, masks mirror.
     PREDICT   = 1  -> writes go to predicted slot, dirty bit set (existing behavior).
   Switching mode requires ecs_tree_no_dirty (no in-flight prediction). */
typedef enum {
    ECS_MODE_CONFIRMED = 0,
    ECS_MODE_PREDICT   = 1,
} ecs_mode_t;

typedef struct ecs_l1_t {
    uint64_t confirmed_mask_any;       /* committed bitmap, observed by readers */
    uint64_t changed;                  /* slots written this tick; cleared by ecs_tree_rollback */
    uint64_t predicted_mask_any;       /* live this frame; iterator masks come from here */
    uint64_t dirty;                    /* slots written in PREDICT mode since last rollback */
    /* tail: 128 * data_size bytes -- [0..63] confirmed slots, [64..127] predicted slots */
} ecs_l1_t;

typedef struct ecs_l2_t {
    uint64_t confirmed_mask_any;
    uint64_t confirmed_mask_all;       /* bit j set iff children[j].confirmed_mask_any == ~0ULL */
    uint64_t changed;
    uint64_t dirty;
    uint64_t predicted_mask_any;
    uint64_t predicted_mask_all;       /* bit j set iff children[j].predicted_mask_any == ~0ULL */
    ecs_l1_t* children[64];
} ecs_l2_t;

typedef struct ecs_l3_t {
    uint64_t confirmed_mask_any;
    uint64_t confirmed_mask_all;       /* bit i set iff children[i].confirmed_mask_all == ~0ULL */
    uint64_t dirty;
    uint64_t changed;

    uint64_t predicted_mask_any;
    uint64_t predicted_mask_all;       /* bit i set iff children[i].predicted_mask_all == ~0ULL */
    ecs_l2_t* children[64];
} ecs_l3_t;

/* Tree flags. Encodes component kind + future bools.
   Component lifecycle is fully determined by the BUFFER flag - no user-supplied
   destroy hook. POD = trivial copy + no destroy. LIST = element-level mutation
   via ecs_tree_buffer_push/pop/clear + ecs_free(slot->h) destroy.

   ECS_TREE_FLAG_BUFFER: slot type is ecs_buffer_t. Mutation goes through
                       ecs_tree_buffer_push/pop/clear (or iterator mirrors).
                       PREDICT mode COWs confirmedâ†’predicted on first dirty
                       this tick (single deep-copy), then mutates the
                       predicted buffer in place. tree_remove / rollback /
                       tree_destroy free slot->h. ecs_tree_get_mut and
                       ecs_iterator_get_mut assert this flag clear (POD-only). */
#define ECS_TREE_FLAG_BUFFER (1u << 0)

typedef struct ecs_tree_t {
    const char* name;          /* may be NULL */
    size_t      data_size;
    ecs_l3_t*   root;
    ecs_mode_t  mode;          /* CONFIRMED or PREDICT - set via ecs_world_set_mode */
    uint32_t    flags;         /* ECS_TREE_FLAG_* bitset */
} ecs_tree_t;

/* ==========================================================================
   Query + iterator
   ========================================================================== */

typedef struct {
    uint32_t id      : 18;
    uint32_t version : 12;
} entity_t;

#define ECS_QUERY_MAX_CLAUSES 4
#define ECS_QUERY_MAX_TERMS   8

typedef struct ecs_world_t ecs_world_t;

typedef struct {
    uint32_t include;
    uint32_t exclude;
    uint32_t changed;
} ecs_compiled_clause_t;

typedef struct {
    const ecs_world_t*    world;       /* set by ecs_compile_query, used for tick-id invariant */
    uint32_t              tree_count;
    uint32_t              clause_count;
    ecs_tree_t*           trees[ECS_QUERY_MAX_TERMS];
    ecs_compiled_clause_t clauses[ECS_QUERY_MAX_CLAUSES];
} ecs_compiled_query_t;

typedef struct ecs_iterator_t {
    const ecs_compiled_query_t* query;

    uint64_t l3_mask;
    uint64_t l2_mask;
    int      l3_idx;                 /* -1 sentinel = no L1 block loaded yet (init / first call) */
    int      l2_idx;
    /* write_mask: per-query-slot mask (one bit per query->trees[i]). Const
       post-init (compiler hoists across iter_set's l1->changed/dirty stores).
       Set via cast in ecs_iterator_init. */
    const uint32_t   write_mask;
    const ecs_mode_t mode;           /* cached from query trees at iterator_init */

    const ecs_world_t* const world;
    const uint8_t            world_tree_idx[ECS_QUERY_MAX_TERMS];

    const ecs_l3_t* l3[ECS_QUERY_MAX_TERMS];
    const ecs_l2_t* l2[ECS_QUERY_MAX_TERMS];
    /* Pointer arrays are const-after-block-load so the compiler can hoist
       loads across ecs_iterator_get_mut's memcpy (which writes through l1_data,
       not to it). Pointees are still mutable - l1->changed |= bit works.
       Updated via cast in iter_load_l1 / ecs_iterator_init only. */
    ecs_l1_t* const l1[ECS_QUERY_MAX_TERMS];
    void*     const l1_data[ECS_QUERY_MAX_TERMS];   /* inline data base = confirmed[0] */
    /* const so compiler can hoist size loads across iterator_set's writes
       through l1->dirty/changed (uint64 stores into l1, same type as size_t,
       would otherwise force reloads each call). Written once via cast in
       ecs_iterator_init. Never modified after that. */
    const size_t    data_size[ECS_QUERY_MAX_TERMS];
} ecs_iterator_t;

struct ecs_world_t {
    ecs_tree_t trees[64];
    uint64_t   mask;
    uint64_t   dirty;
    uint32_t   confirmed_tick; /* gameplay/network clock. Bumped by
                                  ecs_world_rollback when any tree advanced.
                                  Serialized; replicated across machines. */
    uint32_t   predicted_tick; /* local frame counter. Bumped every
                                  ecs_world_end_tick (CONFIRMED or PREDICT).
                                  Reset to confirmed_tick on rollback so
                                  predicted - confirmed = pending predict frames. */
    ecs_mode_t mode;           /* mirrored to every populated tree by ecs_world_set_mode */
};

#ifdef __cplusplus
extern "C" {
#endif

extern ecs_l1_t ecs_default_l1;
extern ecs_l2_t ecs_default_l2;

int      ecs_tree_masks_valid(const ecs_tree_t* tree);
ecs_compiled_query_t* ecs_compile_query(const ecs_world_t* world, const char* expr);
void                  ecs_tree_end_tick(ecs_tree_t* tree);
void                  ecs_world_end_tick(ecs_world_t* world);
void     ecs_crc64_init(void);

/* ==========================================================================
   Pipeline - ordered list of systems run once per tick. Single-threaded.
   Mode-agnostic: same pipeline runs in CONFIRMED and PREDICT ticks; the
   world's current mode dictates write semantics. Caller drives promote /
   rollback around ecs_pipeline_run.
   ========================================================================== */
typedef void (*ecs_system_fn)(ecs_world_t* world, void* ctx);

/* SOA: parallel arrays of fn / ctx. Hot loop in ecs_pipeline_run touches only
   fns[]; ctxs[] read once per call. No padding from struct-of-pointers AoS. */
typedef struct {
    ecs_system_fn* fns;
    void**         ctxs;
    uint32_t       count;
    uint32_t       cap;
} ecs_pipeline_t;

static inline void ecs_pipeline_init(ecs_pipeline_t* p) {
    assert(p);
    p->fns = NULL; p->ctxs = NULL; p->count = 0; p->cap = 0;
}
static inline void ecs_pipeline_destroy(ecs_pipeline_t* p) {
    assert(p);
    if (p->fns)  ecs_free(p->fns);
    if (p->ctxs) ecs_free(p->ctxs);
    p->fns = NULL; p->ctxs = NULL; p->count = 0; p->cap = 0;
}
static inline void ecs_pipeline_add(ecs_pipeline_t* p, ecs_system_fn fn,
                                    void* ctx) {
    assert(p && fn);
    if (p->count == p->cap) {
        uint32_t nc = p->cap ? p->cap * 2u : 8u;
        p->fns  = (ecs_system_fn*)ecs_xrealloc(p->fns,  nc * sizeof(ecs_system_fn));
        p->ctxs = (void**)         ecs_xrealloc(p->ctxs, nc * sizeof(void*));
        p->cap = nc;
    }
    p->fns [p->count] = fn;
    p->ctxs[p->count] = ctx;
    p->count++;
}
static inline uint32_t ecs_pipeline_count(const ecs_pipeline_t* p) {
    return p->count;
}

/* Run every registered system once, then end-of-tick: clear `changed` and
   bump predicted_tick. Caller drives rollback after return. */
void     ecs_pipeline_run(ecs_pipeline_t* p, ecs_world_t* world);

/* Serialize the confirmed state of a tree into the bitpacked serializer.
   Two-pass layout: structural metadata first, payload second. Lets a
   downstream consumer skim the topology without touching payload, or
   pipe just the payload block through a separate compressor.

   Format:

       version     = u8 (2)            byte-aligned
       flags       = u8 (bit0: has_data)
       data_size   = LEB128 varint

       --- Pass 1: masks (variable bits) ---
       mask = encoded l3 mask
       per set bit i in l3_mask:
           mask = encoded l2 mask
           per set bit j in l2_mask:
               mask = encoded l1 mask

       align-to-byte                       (single pad, <=7 bits)

       --- Pass 2: payloads (byte-aligned) ---
       per set bit i in l3_mask:
           per set bit j in l2_mask:
               batch_payload (raw bytes, runs coalesced via mask)

   Tag trees (data_size == 0) and empty trees skip pass 2 entirely.

   Mask encoding (variable bits, single tag table for root + child):

       tag(1) = 0 -> raw u64                                  65 bits
       tag(1) = 1, sub(1) = 1 -> all-set (mask == ~0)          2 bits
       tag(1) = 1, sub(1) = 0 -> indexed:
                  polarity(1) | k(3) | k * index(6)           6 + 6k bits

       Polarity 0 encodes set bits of mask; polarity 1 encodes clear bits
       (decoder reconstructs as ~indexed). k = min(popcount, 64-popcount)
       and the polarity is picked to minimise it. Indexed path is taken
       when min_count <= 7 (k field is 3 bits); larger min_counts fall
       through to raw. All-zero mask emits as polarity=1, k=0 (6 bits).

   Other compression: empty L2/L1 subtrees skipped entirely; only set-slot
   data emitted (no padding). Predicted/dirty state intentionally omitted
   -- transient, reconstructed by the simulator.

   Caller owns the backing buffer and is responsible for sizing it.
   Bitpacker asserts on overflow. */
void     ecs_tree_serialize(const ecs_tree_t* tree, ecs_serializer_t* s);

/* Deserialize a tree previously written by ecs_tree_serialize. Totally
   replaces tree state -- confirmed/predicted/dirty are overwritten
   from the stream, predicted_mask = confirmed_mask, dirty = 0.

   Reuses already-allocated nodes when possible: l2/l1 nodes that exist
   in both old and new state are repurposed in place; old-only nodes are
   freed.

   If tree->root is NULL (zero-initialized slot), auto-inits with the
   stream's data_size. Otherwise data_size must match.

   Returns 0 on success. Returns -1 on header mismatch (wrong version or
   data_size mismatch). Bitpacker asserts on truncation. */
int      ecs_tree_deserialize(ecs_tree_t* tree, ecs_deserializer_t* d);

/* Serialize an entire world. Format:
       version    = u8 (2)
       tick       = u32
       tree_mask  = encoded u64 (which of the 64 tree slots are populated)
       per set bit i in tree_mask:
           tree i serialized via ecs_tree_serialize
   Tree names are NOT serialized -- slot index is the stable identity. */
void     ecs_world_serialize(const ecs_world_t* world, ecs_serializer_t* s);

/* Deserialize a world. Trees in the old mask but missing from the new mask
   are destroyed; trees in the new mask are deserialized into the slot,
   auto-initializing zero-initialized slots. world->confirmed_tick +
   world->mask are overwritten. world->dirty is reset to 0. predicted_tick is
   set to confirmed_tick. Returns 0 on success, -1 on header / per-tree
   deserialize failure. */
int      ecs_world_deserialize(ecs_world_t* world, ecs_deserializer_t* d);

/* Tick-end. Discards predicted bytes (predicted is always speculative),
   clears changed, releases empty L1/L2 nodes. Returns 1 iff a CONFIRMED-mode
   write landed this cycle (caller can use it to drive world->confirmed_tick
   or per-tree freshness ack); 0 for predict-only / idle ticks. */
int      ecs_tree_rollback(ecs_tree_t* tree);
void     ecs_world_rollback(ecs_world_t* world);

/* Switch global VM mode. Asserts no in-flight prediction (dirty == 0
   everywhere). In CONFIRMED mode, get_mut/stage_remove write directly
   to confirmed slots, masks mirror, dirty stays 0. */
void     ecs_world_set_mode(ecs_world_t* world, ecs_mode_t mode);
void     ecs_tree_set_mode(ecs_tree_t* tree, ecs_mode_t mode);
void     ecs_tree_destroy(ecs_tree_t* tree);
void     ecs_world_destroy(ecs_world_t* world);

/* Get mutable pointer to slot, marking it as written-this-frame. Acquires
   L2/L1 if absent. Sets predicted/confirmed/dirty/changed masks
   unconditionally - caller asserts the slot is changing (no eq-check).
   Caller writes through the returned pointer.

   POD trees: returns ptr to writable slot (predicted slot in PREDICT mode,
   confirmed slot in CONFIRMED mode).
   Tag trees (data_size == 0): returns NULL but masks are still updated.
   BUFFER trees: must use ecs_tree_buffer_push/pop/clear (asserts).

   Caller should only call when value is actually changing - calling on an
   unchanged slot will spuriously promote dirty/changed bits, defeating
   prediction-mode rollback semantics. */
void*    ecs_tree_get_mut(ecs_tree_t* tree, int index);

/* BUFFER tree element-level mutation. PREDICT mode COWs confirmedâ†’predicted on
   first dirty bit this tick (one deep-copy of confirmed bytes), then mutates
   the predicted slot's buffer in place. CONFIRMED mode mutates the confirmed
   buffer directly; the buffer survives across ticks (realloc on grow only).
   All assert ECS_TREE_FLAG_BUFFER + tree->data_size == sizeof(ecs_buffer_t).

   ecs_tree_buffer_push  : append elem_size bytes pointed by value to slot's buffer.
                         Allocates the slot's buffer header on first push.
   ecs_tree_buffer_pop   : drop the last elem_size-sized element. Asserts non-empty.
   ecs_tree_buffer_clear : zero the slot's buffer size (keeps capacity in CONFIRMED;
                         drops the predicted buffer entirely on PREDICT first
                         touch so the next push starts fresh). */
void     ecs_tree_buffer_push (ecs_tree_t* tree, int index, size_t elem_size, const void* value);
void     ecs_tree_buffer_pop  (ecs_tree_t* tree, int index, size_t elem_size);
void     ecs_tree_buffer_clear(ecs_tree_t* tree, int index);

/* Remove a slot. For BUFFER trees frees the live slot's heap (ecs_free on its
   buffer header) before clearing presence. In CONFIRMED mode that's the
   confirmed slot; in PREDICT mode the predicted clone iff (predicted_mask &
   dirty) bit is set (i.e. a live predict-set this cycle). Plain predict-
   remove on a confirmed-only slot leaves predicted bytes as stale-POD, no
   free needed. POD trees: trivial mask clear, no free.
   Returns 1 if a slot was actually removed, 0 if it wasn't present. */
int      ecs_tree_remove(ecs_tree_t* tree, int index);

/* write_mask: per-query-slot bitmask (bit i set â‡’ writes through query->trees[i]).
   Pass 0 for read-only iterations. */
void     ecs_iterator_init(ecs_iterator_t* it, const ecs_compiled_query_t* query, uint32_t write_mask);
/* Returns the L1 hit mask for the next non-empty block (0 = iteration done).
   Caller iterates the mask: `while (mask) { int i = ecs_ctz64(mask); mask &= mask - 1; ... }`.
   Each call propagates dirty/changed bits from the prior L1 up to L2/L3, then
   advances forward until a non-empty L1 block is found. Tree base pointers
   referenced by ecs_iterator_get / get_mut / remove are updated in-place; the
   caller's local `mask` is the only per-iteration state. */
uint64_t ecs_iterator_next_block(ecs_iterator_t* it);
/* View-aware CRC. Alive set = predicted_mask_any (mirrors confirmed in
   CONFIRMED mode, is the live speculative set in PREDICT). Per-slot bytes:
   predicted bytes when dirty bit set, else confirmed bytes. Tick excluded
   so a confirmed-tick-N state and a predict-on-tick-M state with identical
   alive masks + view bytes hash equal - enabling determinism checks between
   predicted simulation and confirmed replay. */
uint64_t ecs_tree_crc64(const ecs_tree_t* tree);
uint64_t ecs_world_crc64(const ecs_world_t* world);


#ifdef __cplusplus
}
#endif

/* Inline-data accessors. Confirmed slots = data[0..63], predicted = data[64..127]. */
static inline char* ecs_l1_data(const ecs_l1_t* n) {
    return (char*)n + sizeof(ecs_l1_t);
}
/* Confirmed slot pointer for index i (0..63). */
static inline void* ecs_l1_confirmed(const ecs_l1_t* n, int i, size_t data_size) {
    return ecs_l1_data(n) + (size_t)i * data_size;
}
/* Predicted slot pointer for index i (0..63). Offset = (i + 64) * data_size. */
static inline void* ecs_l1_predicted(const ecs_l1_t* n, int i, size_t data_size) {
    return ecs_l1_data(n) + (size_t)(i + 64) * data_size;
}

static inline void ecs_tree_init(ecs_tree_t* tree, size_t data_size, uint32_t flags) {
    assert(tree);
    /* BUFFER trees must use sizeof(ecs_buffer_t) - engine reinterprets slot bytes
       as an ecs_buffer_t for assign/destroy. */
    assert(!(flags & ECS_TREE_FLAG_BUFFER) || data_size == sizeof(ecs_buffer_t));
    tree->name             = NULL;
    tree->data_size        = data_size;
    tree->mode             = ECS_MODE_CONFIRMED;
    tree->flags            = flags;
    tree->root             = (ecs_l3_t*)ecs_xmalloc_aligned(sizeof(ecs_l3_t), 64);
    tree->root->confirmed_mask_any  = 0;
    tree->root->confirmed_mask_all  = 0;
    tree->root->predicted_mask_any  = 0;
    tree->root->predicted_mask_all  = 0;
    tree->root->dirty               = 0;
    tree->root->changed             = 0;
    for (int k = 0; k < 64; k++)
        tree->root->children[k] = &ecs_default_l2;
}

static inline ecs_l2_t* ecs_l2_acquire(ecs_tree_t* tree) {
    assert(tree);
    ecs_l2_t* node = (ecs_l2_t*)ecs_xmalloc_aligned(sizeof(ecs_l2_t), 64);
    node->confirmed_mask_any  = 0;
    node->confirmed_mask_all  = 0;
    node->predicted_mask_any  = 0;
    node->predicted_mask_all  = 0;
    node->dirty               = 0;
    node->changed             = 0;
    for (int k = 0; k < 64; k++) node->children[k] = &ecs_default_l1;
    return node;
}

static inline void ecs_l2_release(ecs_tree_t* tree, ecs_l2_t* node) {
    assert(tree && node);
    assert(node != &ecs_default_l2 && "releasing default l2 node");
    (void)tree;
    ecs_free(node);
}

static inline ecs_l1_t* ecs_l1_acquire(ecs_tree_t* tree) {
    assert(tree);

    size_t data_size = tree->data_size;
    size_t total     = sizeof(ecs_l1_t) + 128 * data_size;
    ecs_l1_t* node = (ecs_l1_t*)ecs_xmalloc_aligned(total, 64);
    node->confirmed_mask_any = 0;
    node->predicted_mask_any = 0;
    node->dirty              = 0;
    node->changed            = 0;
    /* Zero the data tail so BUFFER slots start with NULL header pointers.
       POD trees don't depend on initial bytes (mask gates access), but the
       cost is a single memset at acquire which is rare. Keeps the BUFFER path
       branch-free in tree_set / rollback (always a valid ecs_buffer_t header). */
    if (data_size) memset((char*)node + sizeof(ecs_l1_t), 0, 128 * data_size);
    return node;
}

static inline void ecs_l1_release(ecs_tree_t* tree, ecs_l1_t* node) {
    assert(tree && node);
    assert(node != &ecs_default_l1 && "releasing default l1 node");
    (void)tree;
    ecs_free(node);
}

/* "Current frame" view: predicted when dirty, confirmed when not. Branchless via
   slot-offset shift (+64 if dirty bit set). */
static inline const void* ecs_tree_get(const ecs_tree_t* tree, int index) {
    assert(tree);
    assert(index >= 0 && index < (1 << 18));
    assert(tree->data_size && "ecs_tree_get called on tag component (data_size == 0)");
    const ecs_l2_t* l2s = tree->root->children[(index >> 12) & 0x3F];
    const ecs_l1_t* l1s = l2s->children[(index >> 6) & 0x3F];
    size_t   slot     = (size_t)(index & 0x3F);
    uint64_t dirty_hi = (l1s->dirty >> slot) & 1ULL;
    size_t   off      = slot + (size_t)(dirty_hi << 6);   /* +64 slots when dirty */
    return ecs_l1_data(l1s) + off * tree->data_size;
}

static inline int ecs_highest_bit64(uint64_t x) {
    assert(x);
#ifdef _MSC_VER
    unsigned long i; _BitScanReverse64(&i, x); return (int)i;
#else
    return 63 - __builtin_clzll(x);
#endif
}

/* Pop the next run of consecutive set bits from *mask. Returns run length;
   writes its starting bit index to *out_idx. Caller loops while result > 0:
       int idx, run;
       while ((run = ecs_mask_pop_run(&m, &idx))) { ... }
   Single source for the bit-run coalescing trick used by sparse memcpy and
   the batch serializer. */
static inline int ecs_mask_pop_run(uint64_t* mask, int* out_idx) {
    if (!*mask) return 0;
    int      i   = ecs_ctz64(*mask);
    uint64_t hi  = *mask >> i;
    int      run = hi == ~0ULL ? 64 - i : ecs_ctz64(~hi);
    *mask        = (hi & (hi + 1)) << i;
    *out_idx     = i;
    return run;
}

/* Sparse memcpy: src and dst both indexed by mask bit. Coalesces consecutive
   set bits into one memcpy call. __restrict -- no overlap. */
static inline void ecs_memcpy_sparse(void* __restrict dst, const void* __restrict src, size_t block_size, uint64_t mask) {
    assert(!mask || (dst && src && block_size));
    assert(mask == 0 || (src != dst && "ecs_memcpy_sparse does not support in-place copying"));
    assert(!mask ||
           (((const char*)src + (size_t)(ecs_highest_bit64(mask) + 1) * block_size <= (const char*)dst ||
             (const char*)dst + (size_t)(ecs_highest_bit64(mask) + 1) * block_size <= (const char*)src)
            && "ecs_memcpy_sparse: src and dst regions must not overlap"));

    int idx, run;
    while ((run = ecs_mask_pop_run(&mask, &idx))) {
        memcpy((char*)dst + (size_t)idx * block_size,
               (const char*)src + (size_t)idx * block_size,
               (size_t)run * block_size);
    }
}

static inline int ecs_tree_no_dirty(const ecs_tree_t* tree) {
    if (tree->root->dirty) return 0;
    uint64_t v3 = tree->root->confirmed_mask_any | tree->root->predicted_mask_any;
    while (v3) {
        int i = ecs_ctz64(v3); v3 &= v3 - 1;
        const ecs_l2_t* l2s = tree->root->children[i];
        if (l2s->dirty) return 0;
        uint64_t v2 = l2s->confirmed_mask_any | l2s->predicted_mask_any;
        while (v2) {
            int j = ecs_ctz64(v2); v2 &= v2 - 1;
            const ecs_l1_t* l1s = l2s->children[j];
            if (l1s->dirty) return 0;
        }
    }
    return 1;
}

/* "Current frame" data -- predicted when dirty, confirmed when not. Branchless
   via slot-offset shift (+64 when dirty), gated by mode multiplier (0 in
   CONFIRMED zeroes the contribution, 1 in PREDICT keeps it). */
static inline void* ecs_iterator_get(const ecs_iterator_t* it, uint32_t tree_idx, int slot) {
    assert(it);
    assert(tree_idx < it->query->tree_count);
    assert(slot >= 0 && slot < 64);
    const ecs_l1_t* l1 = it->l1[tree_idx];
    uint64_t dirty_hi = (l1->dirty >> slot) & 1ULL;
    size_t   off      = (size_t)slot + (size_t)(dirty_hi << 6) * (size_t)it->mode;
    return (char*)it->l1_data[tree_idx] + off * it->data_size[tree_idx];
}

/* Get mutable pointer to slot for write-this-frame. Caller writes through
   the returned ptr; engine assumes bytes are changing (no eq-check). Sets
   l1->changed bit eagerly; dirty / predicted_mask_any / confirmed_mask_any
   get derived from changed at the L1 block boundary in
   ecs_iterator_next_block. POD only - BUFFER/tag use the tree-level API.

   Caller should only call when the value is actually changing (calling on
   an unchanged slot spuriously promotes dirty/changed bits, breaking
   prediction-mode rollback semantics). */
static inline void* ecs_iterator_get_mut(ecs_iterator_t* it, uint32_t tree_idx, int slot) {
    assert(it);
    assert(tree_idx < it->query->tree_count);
    assert(slot >= 0 && slot < 64);
    assert(!(it->query->trees[tree_idx]->flags & ECS_TREE_FLAG_BUFFER) &&
           "ecs_iterator_get_mut: BUFFER components not supported on iterator path");

    ecs_l1_t* l1   = it->l1[tree_idx];
    size_t    ds   = it->data_size[tree_idx];
    assert(ds > 0 && "ecs_iterator_get_mut called on tag component (data_size == 0)");

    uint64_t  bit  = 1ULL << slot;
    char*     conf = (char*)it->l1_data[tree_idx] + (size_t)slot * ds;
    uint64_t  m    = (uint64_t)it->mode;                          /* 0 or 1 */
    l1->changed |= bit;
    return conf + (size_t)64 * ds * (size_t)m;                    /* conf when 0, pred when 1 */
}

/* Iterator-side remove. Mode-dispatched like ecs_iterator_get_mut:
     PREDICT   -> clear predicted_mask, set dirty (rolled back on rollback).
     CONFIRMED -> clear both confirmed_mask and predicted_mask in lockstep.
   Eager mask propagation up to L2/L3 - does not piggyback the deferred flush
   in ecs_iterator_next_block because that flush is OR-only and would re-add
   the cleared bit. Clears the slot's bit in l1->changed for the same reason
   (so a same-block iterator_set followed by iterator_remove on this slot
   leaves predicted_mask clear after the next L1 boundary). Trade-off: the
   removed slot does NOT register in changed-clauses for this tree (the slot
   wouldn't iterate post-remove anyway since predicted_mask_any cleared).
   POD-only; BUFFER trees must use ecs_tree_remove. Empty L1/L2 nodes are NOT
   released here - done by ecs_tree_rollback / ecs_tree_destroy.
   Returns 1 if a slot was present and removed, 0 otherwise. */
static inline int ecs_iterator_remove(ecs_iterator_t* it, uint32_t tree_idx, int slot) {
    assert(it);
    assert(tree_idx < it->query->tree_count);
    assert(slot >= 0 && slot < 64);
    assert(!(it->query->trees[tree_idx]->flags & ECS_TREE_FLAG_BUFFER) &&
           "ecs_iterator_remove: BUFFER components not supported on iterator path; use ecs_tree_remove");

    ecs_l1_t* l1   = it->l1[tree_idx];
    uint64_t  bit1 = 1ULL << slot;

    /* Pre-state for return value. Normal iteration flow guarantees this is 1
       (l1_mask only carries set bits); only zero on same-slot double-remove,
       in which case every store below is idempotent (mask clear of cleared
       bit, OR of already-set bit). */
    uint64_t was_present = (l1->predicted_mask_any >> slot) & 1ULL;

    uint64_t m         = (uint64_t)it->mode;            /* 0 = CONFIRMED, 1 = PREDICT */
    uint64_t conf_mask = -(uint64_t)(1ULL - m);         /* ~0 in CONFIRMED, 0 in PREDICT */
    uint64_t pred_bit1 = bit1 * m;                      /* 0 in CONFIRMED, bit1 in PREDICT */
    uint64_t conf_bit1 = bit1 & conf_mask;              /* bit1 in CONFIRMED, 0 in PREDICT */

    uint64_t l1_pred = l1->predicted_mask_any & ~bit1;
    l1->predicted_mask_any  = l1_pred;
    l1->confirmed_mask_any &= ~conf_bit1;
    l1->dirty              |=  pred_bit1;
    l1->changed            &= ~bit1;

    /* Branchless empty-propagation: l1_empty broadcast as ~0 / 0 mask. */
    uint64_t l1_empty = -(uint64_t)(l1_pred == 0);

    ecs_l2_t* l2   = (ecs_l2_t*)it->l2[tree_idx];
    uint64_t  bit2 = 1ULL << it->l2_idx;
    uint64_t  conf_bit2 = bit2 & conf_mask;
    l2->dirty              |=  bit2 * m;
    l2->changed            |=  bit2;
    l2->predicted_mask_all &= ~bit2;
    l2->confirmed_mask_all &= ~conf_bit2;
    uint64_t l2_pred = l2->predicted_mask_any & ~(bit2 & l1_empty);
    l2->predicted_mask_any  = l2_pred;
    l2->confirmed_mask_any &= ~(conf_bit2 & l1_empty);

    uint64_t l2_empty = -(uint64_t)(l2_pred == 0);

    ecs_l3_t* l3   = (ecs_l3_t*)it->l3[tree_idx];
    uint64_t  bit3 = 1ULL << it->l3_idx;
    uint64_t  conf_bit3 = bit3 & conf_mask;
    l3->dirty              |=  bit3 * m;
    l3->changed            |=  bit3;
    l3->predicted_mask_all &= ~bit3;
    l3->confirmed_mask_all &= ~conf_bit3;
    l3->predicted_mask_any &= ~(bit3 & l2_empty);
    l3->confirmed_mask_any &= ~(conf_bit3 & l2_empty);

    return (int)was_present;
}

/* ==========================================================================
   LIST iterator-side push/pop/clear. PREDICT mode COWs confirmedâ†’predicted
   on first dirty bit this tick (one deep-copy), then mutates in place.
   CONFIRMED mode mutates the confirmed buffer directly.

   Mask propagation: l1->changed and l1->dirty are set inline (dirty drives
   the COW guard). predicted_mask_any/confirmed_mask_any propagation uses the
   same deferred flush that iterator_get_mut relies on (ecs_iterator_next_block
   ORs l1->changed into them at the L1 boundary). Iterator only visits slots
   already in predicted_mask_any, so re-OR'ing is a no-op for these ops.

   Caller must set this tree's bit in the write_mask argument to
   ecs_iterator_init. Returned writes go through the slot's ecs_buffer_t header.
   POD-only ops like ecs_iterator_get_mut still assert non-BUFFER. */
static inline void ecs_iterator_buffer_push(ecs_iterator_t* it, uint32_t tree_idx,
                                          int slot, size_t elem_size, const void* value) {
    assert(it && elem_size && value);
    assert(tree_idx < it->query->tree_count);
    assert(slot >= 0 && slot < 64);
    assert((it->query->trees[tree_idx]->flags & ECS_TREE_FLAG_BUFFER) &&
           "ecs_iterator_buffer_push: tree must be BUFFER-flagged");

    ecs_l1_t* l1   = it->l1[tree_idx];
    char*     base = (char*)it->l1_data[tree_idx];
    uint64_t  bit  = 1ULL << slot;
    uint64_t  m    = (uint64_t)it->mode;

    ecs_buffer_t* live;
    if (m == 1) {
        ecs_buffer_t* pred = (ecs_buffer_t*)(base + (size_t)(slot + 64) * sizeof(ecs_buffer_t));
        if (!(l1->dirty & bit)) {
            ecs_buffer_t* conf = (ecs_buffer_t*)(base + (size_t)slot * sizeof(ecs_buffer_t));
            pred->h = NULL;
            if ((l1->confirmed_mask_any & bit) && conf->h && conf->h->size > 0u) {
                uint32_t cap = conf->h->capacity;
                ecs_buffer_header_t* nh = (ecs_buffer_header_t*)ecs_xmalloc_aligned(
                    sizeof(ecs_buffer_header_t) + (size_t)cap, 8);
                nh->size     = conf->h->size;
                nh->capacity = cap;
                memcpy(nh->data, conf->h->data, conf->h->size);
                pred->h = nh;
            }
        }
        live = pred;
    } else {
        live = (ecs_buffer_t*)(base + (size_t)slot * sizeof(ecs_buffer_t));
    }

    l1->changed |= bit;
    l1->dirty   |= bit * m;

    ecs_buffer_push(live, elem_size, value);
}

static inline void ecs_iterator_buffer_pop(ecs_iterator_t* it, uint32_t tree_idx,
                                         int slot, size_t elem_size) {
    assert(it && elem_size);
    assert(tree_idx < it->query->tree_count);
    assert(slot >= 0 && slot < 64);
    assert((it->query->trees[tree_idx]->flags & ECS_TREE_FLAG_BUFFER) &&
           "ecs_iterator_buffer_pop: tree must be BUFFER-flagged");

    ecs_l1_t* l1   = it->l1[tree_idx];
    char*     base = (char*)it->l1_data[tree_idx];
    uint64_t  bit  = 1ULL << slot;
    uint64_t  m    = (uint64_t)it->mode;

    ecs_buffer_t* live;
    if (m == 1) {
        ecs_buffer_t* pred = (ecs_buffer_t*)(base + (size_t)(slot + 64) * sizeof(ecs_buffer_t));
        if (!(l1->dirty & bit)) {
            ecs_buffer_t* conf = (ecs_buffer_t*)(base + (size_t)slot * sizeof(ecs_buffer_t));
            pred->h = NULL;
            if ((l1->confirmed_mask_any & bit) && conf->h && conf->h->size > 0u) {
                uint32_t cap = conf->h->capacity;
                ecs_buffer_header_t* nh = (ecs_buffer_header_t*)ecs_xmalloc_aligned(
                    sizeof(ecs_buffer_header_t) + (size_t)cap, 8);
                nh->size     = conf->h->size;
                nh->capacity = cap;
                memcpy(nh->data, conf->h->data, conf->h->size);
                pred->h = nh;
            }
        }
        live = pred;
    } else {
        live = (ecs_buffer_t*)(base + (size_t)slot * sizeof(ecs_buffer_t));
    }

    l1->changed |= bit;
    l1->dirty   |= bit * m;

    ecs_buffer_pop(*live, elem_size);
}

static inline void ecs_iterator_buffer_clear(ecs_iterator_t* it, uint32_t tree_idx, int slot) {
    assert(it);
    assert(tree_idx < it->query->tree_count);
    assert(slot >= 0 && slot < 64);
    assert((it->query->trees[tree_idx]->flags & ECS_TREE_FLAG_BUFFER) &&
           "ecs_iterator_buffer_clear: tree must be BUFFER-flagged");

    ecs_l1_t* l1   = it->l1[tree_idx];
    char*     base = (char*)it->l1_data[tree_idx];
    uint64_t  bit  = 1ULL << slot;
    uint64_t  m    = (uint64_t)it->mode;

    if (m == 1) {
        ecs_buffer_t* pred = (ecs_buffer_t*)(base + (size_t)(slot + 64) * sizeof(ecs_buffer_t));
        if (!(l1->dirty & bit)) {
            /* First predict touch: skip COW entirely (clear discards confirmed
               contents). Pred slot bytes were NULLed at acquire / prior rollback. */
            pred->h = NULL;
        } else if (pred->h) {
            pred->h->size = 0u;                             /* keep capacity */
        }
    } else {
        ecs_buffer_t* conf = (ecs_buffer_t*)(base + (size_t)slot * sizeof(ecs_buffer_t));
        if (conf->h) conf->h->size = 0u;
    }

    l1->changed |= bit;
    l1->dirty   |= bit * m;
}
