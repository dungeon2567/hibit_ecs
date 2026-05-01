#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mimalloc.h>
#include "ecs_serializer.h"

#ifdef _MSC_VER
#include <intrin.h>
#include <xmmintrin.h>
static inline int ecs_ctz64(uint64_t x) { unsigned long i; _BitScanForward64(&i, x); return (int)i; }
static inline int ecs_ctz32(uint32_t x) { unsigned long i; _BitScanForward(&i, x); return (int)i; }
static inline int ecs_popcount64(uint64_t x) { return (int)__popcnt64(x); }
#define ECS_PREFETCH(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
#define ECS_LIKELY(x)   (x)
#define ECS_UNLIKELY(x) (x)
#else
static inline int ecs_ctz64(uint64_t x) { return __builtin_ctzll(x); }
static inline int ecs_ctz32(uint32_t x) { return __builtin_ctz(x); }
static inline int ecs_popcount64(uint64_t x) { return __builtin_popcountll(x); }
#define ECS_PREFETCH(p) __builtin_prefetch((p), 0, 3)
#define ECS_LIKELY(x)   __builtin_expect(!!(x), 1)
#define ECS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

/* mimalloc wrappers. 'x' prefix = abort on OOM (xmalloc convention) -- every
   allocation site in this engine treats failure as fatal, so callers don't
   check. ecs_free is a thin pass-through for symmetry. */
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

/* ==========================================================================
   ecs_list_t -- typed dynamic array. Wraps a single `ecs_list_header_t*`
   that owns size + capacity + flexible-array data in one allocation. Both
   `size` and `capacity` are byte counts (NOT element counts) -- the list is
   element-size-agnostic at the storage level. Caller passes elem_size to
   every op that needs to address elements; the same value must be used for
   every call on a given list (no runtime check). Wrapper makes
   `ecs_list_t l = {0};` a valid empty list (h == NULL) and keeps the public
   type stable as a value type. Mutating ops take `ecs_list_t*` because
   growth may reallocate the header.
   Growth: 1.5Ã— from base 8 bytes. push memcpy's the value (or leaves the
   slot uninitialized when value == NULL, returning the slot ptr).

   Usage:
       ecs_list_t xs = {0};
       ecs_list_push(&xs, sizeof(int), &(int){42});
       int* p = (int*)ecs_list_at(xs, sizeof(int), 0);
       ecs_list_destroy(&xs);
   ========================================================================== */
/* _Alignas(8) on size forces 8-byte alignment on the whole struct, which
   guarantees the FAM `data` (sitting at offset 8) is also 8-byte aligned --
   safe for any element up to 8-byte natural alignment without padding. */
typedef struct ecs_list_header_t {
    _Alignas(8) uint32_t size;              /* bytes used */
    uint32_t capacity;                      /* bytes allocated for data */
    char     data[];                        /* flexible array -- element bytes inline */
} ecs_list_header_t;

typedef struct ecs_list_t {
    ecs_list_header_t* h;                   /* NULL = empty */
} ecs_list_t;

static inline uint32_t ecs_list_size    (ecs_list_t l) { return l.h ? l.h->size     : 0u; }
static inline uint32_t ecs_list_capacity(ecs_list_t l) { return l.h ? l.h->capacity : 0u; }

static inline void ecs_list_destroy(ecs_list_t* l) {
    assert(l);
    if (l->h) ecs_free(l->h);
    l->h = NULL;
}

/* Reserve at least min_cap_bytes of data capacity. */
static inline void ecs_list_reserve(ecs_list_t* l, uint32_t min_cap_bytes) {
    assert(l);
    uint32_t cap = l->h ? l->h->capacity : 0u;
    if (min_cap_bytes <= cap) return;
    uint32_t newcap = cap ? cap : 8u;
    while (newcap < min_cap_bytes) newcap = newcap + (newcap >> 1) + 1u;  /* 1.5Ã— */
    ecs_list_header_t* nh = (ecs_list_header_t*)ecs_xrealloc_aligned(
        l->h, sizeof(ecs_list_header_t) + (size_t)newcap, 8);
    if (!l->h) nh->size = 0u;               /* fresh alloc -- size was uninitialized */
    nh->capacity = newcap;
    l->h = nh;
}

static inline void ecs_list_clear(ecs_list_t l) { if (l.h) l.h->size = 0u; }

static inline void* ecs_list_at(ecs_list_t l, size_t elem_size, uint32_t i) {
    assert(l.h && elem_size && (size_t)i * elem_size < l.h->size);
    return (char*)l.h->data + (size_t)i * elem_size;
}

static inline void* ecs_list_push(ecs_list_t* l, size_t elem_size, const void* value) {
    assert(l && elem_size);
    uint32_t cur_size = l->h ? l->h->size : 0u;
    uint32_t need     = cur_size + (uint32_t)elem_size;
    if (!l->h || need > l->h->capacity) ecs_list_reserve(l, need);
    void* dst = (char*)l->h->data + cur_size;
    if (value) memcpy(dst, value, elem_size);
    l->h->size = need;
    return dst;
}

static inline void ecs_list_pop(ecs_list_t l, size_t elem_size) {
    assert(l.h && elem_size && l.h->size >= elem_size);
    l.h->size -= (uint32_t)elem_size;
}

/* O(1) unordered remove: copy last element over slot i, shrink size. */
static inline void ecs_list_swap_remove(ecs_list_t l, size_t elem_size, uint32_t i) {
    assert(l.h && elem_size);
    uint32_t off = (uint32_t)((size_t)i * elem_size);
    assert(off < l.h->size);
    uint32_t last = l.h->size - (uint32_t)elem_size;
    l.h->size = last;
    if (off != last) memcpy((char*)l.h->data + off, (char*)l.h->data + last, elem_size);
}

/* ==========================================================================
   Predict-mode ECS — two buffers per L1 (confirmed + predicted), no undo log.

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
                       changed = 0, releases empty L1/L2 nodes, tick++.
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
    uint64_t predicted_mask_any;       /* live this frame; iterator masks come from here */
    uint64_t dirty;                    /* slots written in PREDICT mode since last rollback */
    uint64_t changed;                  /* slots written this tick; cleared by ecs_tree_rollback */
    /* tail: 128 * data_size bytes -- [0..63] confirmed slots, [64..127] predicted slots */
} ecs_l1_t;

typedef struct ecs_l2_t {
    uint64_t confirmed_mask_any;
    uint64_t confirmed_mask_all;       /* bit j set iff children[j].confirmed_mask_any == ~0ULL */
    uint64_t predicted_mask_any;
    uint64_t predicted_mask_all;       /* bit j set iff children[j].predicted_mask_any == ~0ULL */
    uint64_t dirty;
    uint64_t changed;
    ecs_l1_t* children[64];
} ecs_l2_t;

typedef struct ecs_l3_t {
    uint64_t confirmed_mask_any;
    uint64_t confirmed_mask_all;       /* bit i set iff children[i].confirmed_mask_all == ~0ULL */
    uint64_t predicted_mask_any;
    uint64_t predicted_mask_all;       /* bit i set iff children[i].predicted_mask_all == ~0ULL */
    uint64_t dirty;
    uint64_t changed;
    ecs_l2_t* children[64];
} ecs_l3_t;

/* Per-L1 batch encoder. Called once per non-empty L1 during ecs_tree_serialize.
   Implementation must emit data for slots whose bit is set in `mask`, in
   ascending lane order, into the bitpacked serializer. Slot stride =
   `block_size`.

   ecs_tree_init defaults this to ecs_serialize_batch_raw: writes only the
   set slots back-to-back as raw bytes, coalescing runs of consecutive set
   bits into a single ecs_serializer_write_bytes call. Override on the
   tree post-init to plug in quantization, delta-encoding, or bit-packed
   component formats. */
typedef void (*ecs_serialize_batch_fn)(const void* l1_data,
                                       size_t      block_size,
                                       uint64_t    mask,
                                       ecs_serializer_t* s);

/* Mirror of ecs_serialize_batch_fn for the deserialization side. Reads the
   payload that the matching encoder produced, into the slots whose bit is
   set in `mask`. Default ecs_deserialize_batch_raw reads runs of bytes via
   ecs_deserializer_read_bytes. Override post-init to match a custom encoder. */
typedef void (*ecs_deserialize_batch_fn)(void* l1_data,
                                         size_t      block_size,
                                         uint64_t    mask,
                                         ecs_deserializer_t* d);

/* Tree flags. Encodes component kind + future bools.
   Component lifecycle is fully determined by the LIST flag — no user-supplied
   destroy hook. POD = trivial copy + no destroy. LIST = ecs_list_assign deep-
   copy + ecs_free(slot->h) destroy. Components requiring per-element teardown
   beyond list-header free are not supported; model heap as ecs_list_t and
   keep elements POD.

   ECS_TREE_FLAG_LIST: slot type is ecs_list_t. tree_set deep-copies via
                       ecs_list_assign; tree_remove / rollback / tree_destroy
                       free slot->h. ecs_iterator_set asserts this flag clear
                       (POD-only). */
#define ECS_TREE_FLAG_LIST (1u << 0)

typedef struct ecs_tree_t {
    const char* name;          /* may be NULL */
    size_t      data_size;
    uint64_t    data_mask;
    uint64_t    tick;
    uint64_t    tick_id_at_begin;  /* world->tick_id snapshot at last ecs_tree_begin_tick;
                                      iterator asserts equality when query has changed-clause */
    ecs_l3_t*   root;
    ecs_mode_t  mode;          /* CONFIRMED or PREDICT — set via ecs_world_set_mode */
    uint32_t    flags;         /* ECS_TREE_FLAG_* bitset */
    ecs_serialize_batch_fn   serialize_batch;             /* set by ecs_tree_init to raw default */
    ecs_deserialize_batch_fn deserialize_batch;           /* set by ecs_tree_init to raw default */
} ecs_tree_t;

/* ==========================================================================
   Query + iterator
   ========================================================================== */

typedef struct {
    uint64_t id      : 18;
    uint64_t version : 46;
} entity_t;

#define ECS_QUERY_MAX_CLAUSES 8
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
    uint64_t l1_mask;
    int      l3_idx;
    int      l2_idx;
    int      l1_idx;
    uint32_t write_mask;
    ecs_mode_t mode;           /* cached from query trees at iterator_init */

    ecs_world_t* world;
    uint8_t      world_tree_idx[ECS_QUERY_MAX_TERMS];

    const ecs_l3_t* l3[ECS_QUERY_MAX_TERMS];
    const ecs_l2_t* l2[ECS_QUERY_MAX_TERMS];
    /* Pointer arrays are const-after-block-load so the compiler can hoist
       loads across ecs_iterator_set's memcpy (which writes through l1_data,
       not to it). Pointees are still mutable — l1->changed |= bit works.
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
    uint64_t   tick;
    uint64_t   tick_id;        /* monotonic counter; bumped by ecs_world_begin_tick.
                                  Same value across CONFIRMED + PREDICT ticks (no mode distinction). */
    ecs_mode_t mode;           /* mirrored to every populated tree by ecs_world_set_mode */
};

#ifdef __cplusplus
extern "C" {
#endif

extern ecs_l1_t ecs_default_l1;
extern ecs_l2_t ecs_default_l2;

int      ecs_tree_masks_valid(const ecs_tree_t* tree);
ecs_compiled_query_t* ecs_compile_query(const ecs_world_t* world, const char* expr);
void                  ecs_tree_begin_tick(ecs_tree_t* tree);
void                  ecs_world_begin_tick(ecs_world_t* world);
void     ecs_crc64_init(void);

/* ==========================================================================
   Pipeline — ordered list of systems run once per tick. Single-threaded.
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

/* Begin a tick (clears `changed` everywhere, snapshots tick_id) and run every
   registered system once. Caller decides promote / rollback after return. */
void     ecs_pipeline_run(ecs_pipeline_t* p, ecs_world_t* world);

/* Default per-L1 batch encoder. Writes only the set-bit slots back-to-back,
   coalescing consecutive set bits into one ecs_serializer_write_bytes call.
   Compression here is positional: absent slots emit 0 bytes -- the mask is
   what reconstructs layout on read. write_bytes self-aligns, so callers
   need not pre-pad. */
void     ecs_serialize_batch_raw(const void* l1_data,
                                 size_t      block_size,
                                 uint64_t    mask,
                                 ecs_serializer_t* s);

/* Mirror of ecs_serialize_batch_raw -- reads runs of set-bit slots from the
   deserializer into the L1 inline data tail. */
void     ecs_deserialize_batch_raw(void* l1_data,
                                   size_t      block_size,
                                   uint64_t    mask,
                                   ecs_deserializer_t* d);

/* Serialize the confirmed state of a tree into the bitpacked serializer.
   Two-pass layout: structural metadata first, payload second. Lets a
   downstream consumer skim the topology without touching payload, or
   pipe just the payload block through a separate compressor.

   Format:

       version     = u8 (1)            byte-aligned
       flags       = u8 (bit0: has_data)
       data_size   = LEB128 varint
       tick        = u64

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
               batch_payload (via tree->serialize_batch or raw default)

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
   replaces tree state -- confirmed/predicted/dirty/tick are overwritten
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
       version    = u8 (1)
       tick       = u64
       tree_mask  = encoded u64 (which of the 64 tree slots are populated)
       per set bit i in tree_mask:
           tree i serialized via ecs_tree_serialize
   Tree names are NOT serialized -- slot index is the stable identity. */
void     ecs_world_serialize(const ecs_world_t* world, ecs_serializer_t* s);

/* Deserialize a world. Trees in the old mask but missing from the new mask
   are destroyed; trees in the new mask are deserialized into the slot,
   auto-initializing zero-initialized slots. world->tick + world->mask are
   overwritten. world->dirty is reset to 0. Returns 0 on success, -1 on
   header / per-tree deserialize failure. */
int      ecs_world_deserialize(ecs_world_t* world, ecs_deserializer_t* d);

/* Tick-end. Discards predicted bytes (predicted is always speculative),
   clears changed, releases empty L1/L2 nodes, bumps tree->tick. */
void     ecs_tree_rollback(ecs_tree_t* tree);
void     ecs_world_rollback(ecs_world_t* world);

/* Switch global VM mode. Asserts no in-flight prediction (dirty == 0
   everywhere). In CONFIRMED mode, get_mut/stage_remove write directly
   to confirmed slots, masks mirror, dirty stays 0. */
void     ecs_world_set_mode(ecs_world_t* world, ecs_mode_t mode);
void     ecs_tree_set_mode(ecs_tree_t* tree, ecs_mode_t mode);
void     ecs_tree_destroy(ecs_tree_t* tree);
void     ecs_world_destroy(ecs_world_t* world);
void     ecs_tree_set(ecs_tree_t* tree, int index, const void* new_value);

/* Remove a slot. For LIST trees frees the live slot's heap (ecs_free on its
   list header) before clearing presence. In CONFIRMED mode that's the
   confirmed slot; in PREDICT mode the predicted clone iff (predicted_mask &
   dirty) bit is set (i.e. a live predict-set this cycle). Plain predict-
   remove on a confirmed-only slot leaves predicted bytes as stale-POD, no
   free needed. POD trees: trivial mask clear, no free.
   Returns 1 if a slot was actually removed, 0 if it wasn't present. */
int      ecs_tree_remove(ecs_tree_t* tree, int index);

void     ecs_iterator_init(ecs_iterator_t* it, const ecs_compiled_query_t* query);
int      ecs_iterator_next_slow(ecs_iterator_t* it);
/* View-aware CRC. Alive set = predicted_mask_any (mirrors confirmed in
   CONFIRMED mode, is the live speculative set in PREDICT). Per-slot bytes:
   predicted bytes when dirty bit set, else confirmed bytes. Tick excluded
   so a confirmed-tick-N state and a predict-on-tick-M state with identical
   alive masks + view bytes hash equal — enabling determinism checks between
   predicted simulation and confirmed replay. */
uint64_t ecs_tree_crc64(const ecs_tree_t* tree);
uint64_t ecs_world_crc64(const ecs_world_t* world);

/* ecs_list_t deep-copy. Caller MUST ensure dst is empty (dst->h == NULL or
   dst->h freed already). Used by ecs_tree_set on LIST trees. */
static inline void ecs_list_assign(ecs_list_t* dst, const ecs_list_t* src) {
    assert(dst);
    if (src && src->h && src->h->size > 0u) {
        uint32_t cap = src->h->capacity;
        ecs_list_header_t* nh = (ecs_list_header_t*)ecs_xmalloc_aligned(
            sizeof(ecs_list_header_t) + (size_t)cap, 8);
        nh->size     = src->h->size;
        nh->capacity = cap;
        memcpy(nh->data, src->h->data, src->h->size);
        dst->h = nh;
    } else {
        dst->h = NULL;
    }
}

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
    /* LIST trees must use sizeof(ecs_list_t) — engine reinterprets slot bytes
       as an ecs_list_t for assign/destroy. */
    assert(!(flags & ECS_TREE_FLAG_LIST) || data_size == sizeof(ecs_list_t));
    tree->name             = NULL;
    tree->data_size        = data_size;
    tree->data_mask        = (uint64_t)-(int64_t)(data_size != 0);
    tree->tick             = 0;
    tree->tick_id_at_begin = 0;
    tree->mode             = ECS_MODE_CONFIRMED;
    tree->flags            = flags;
    tree->serialize_batch             = ecs_serialize_batch_raw;
    tree->deserialize_batch           = ecs_deserialize_batch_raw;
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
    /* Zero the data tail so LIST slots start with NULL header pointers.
       POD trees don't depend on initial bytes (mask gates access), but the
       cost is a single memset at acquire which is rare. Keeps the LIST path
       branch-free in tree_set / rollback (always a valid ecs_list_t header). */
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

/* Fast path: pop next entity from current L1 block. Slow path (L1 exhausted ->
   advance L2/L3 + dirty propagation) lives in ecs.c. */
static inline int ecs_iterator_next(ecs_iterator_t* it) {
    if (it->l1_mask) {
        it->l1_idx   = ecs_ctz64(it->l1_mask);
        it->l1_mask &= it->l1_mask - 1;
        return 1;
    }
    return ecs_iterator_next_slow(it);
}

/* "Current frame" data -- predicted when dirty, confirmed when not. Branchless
   via slot-offset shift (+64 when dirty), gated by mode multiplier (0 in
   CONFIRMED zeroes the contribution, 1 in PREDICT keeps it). */
static inline void* ecs_iterator_get(const ecs_iterator_t* it, uint32_t tree_idx) {
    assert(it);
    assert(tree_idx < it->query->tree_count);
    const ecs_l1_t* l1 = it->l1[tree_idx];
    size_t   slot     = (size_t)it->l1_idx;
    uint64_t dirty_hi = (l1->dirty >> slot) & 1ULL;
    size_t   off      = slot + (size_t)(dirty_hi << 6) * (size_t)it->mode;
    return (char*)it->l1_data[tree_idx] + off * it->data_size[tree_idx];
}

/* Branchless mode-multiplexed write. POD-only — asserts the term's tree is
   not LIST. Always memcpy, conditionally skip the mask update via memcmp
   against the current visible value: if bytes match, the predicted slot
   doesn't get promoted (dirty/changed unchanged → reader still sees conf).
   Wasted store on equal case is one cache-line write; the elimination of
   the eq-vs-neq branch on the I-cache hot path pays for it.

   Mask propagation: only l1->changed is touched eagerly. dirty /
   predicted_mask_any / confirmed_mask_any get derived from changed at the
   L1 block boundary in ecs_iterator_next_slow / ecs_iterator_next_block. */
static inline void ecs_iterator_set(ecs_iterator_t* it, uint32_t tree_idx, const void* new_value) {
    assert(it);
    assert(new_value);
    assert(tree_idx < it->query->tree_count);
    assert(!(it->query->trees[tree_idx]->flags & ECS_TREE_FLAG_LIST) &&
           "ecs_iterator_set: LIST components not supported on iterator path; use ecs_tree_set");

    ecs_l1_t* l1   = it->l1[tree_idx];
    size_t    ds   = it->data_size[tree_idx];
    assert(ds > 0 && "ecs_iterator_set called on tag component (data_size == 0)");

    uint64_t  bit  = 1ULL << it->l1_idx;
    char*     conf = (char*)it->l1_data[tree_idx] + (size_t)it->l1_idx * ds;
    uint64_t  m    = (uint64_t)it->mode;                          /* 0 or 1 */
    char*     dst  = conf + (size_t)64 * ds * (size_t)m;          /* conf when 0, pred when 1 */

    /* Visible value = predicted when dirty, else confirmed. */
    uint64_t    dirty_hi = (l1->dirty >> it->l1_idx) & 1ULL;
    const char* cur      = conf + (size_t)64 * ds * (size_t)dirty_hi;
    int eq = (memcmp(cur, new_value, ds) == 0);

    memcpy(dst, new_value, ds);
    if (eq) return;
    l1->changed |= bit;
}
