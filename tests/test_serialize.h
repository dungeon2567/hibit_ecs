#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>
#include "ecs.h"
#include "test_ecs.h"   /* g_passed/g_failed/EXPECT/make_tree/free_tree/set_val/comp_t */

/* Round-trip ecs_tree_serialize -> ecs_tree_deserialize and confirm the
   reconstructed tree's CRC64 matches the original. CRC64 reads only
   confirmed state, which is exactly what serialize emits, so equal CRC =
   bit-exact replay of serialized state. */

static const int32_t TS_BUFFER_BYTES = 1 << 16;   /* 64 KiB, %8 == 0 */

static void ts_round_trip(ecs_tree_t* src, const char* tag) {
    /* Encode src */
    uint8_t* buf = (uint8_t*)ecs_xcalloc(1, (size_t)TS_BUFFER_BYTES);
    ecs_serializer_t s;
    ecs_serializer_init(&s, buf, TS_BUFFER_BYTES);
    ecs_tree_serialize(src, &s);
    ecs_serializer_flush_bits(&s);
    int32_t bytes = ecs_serializer_get_bytes_written(&s);
    EXPECT(bytes > 0, tag);

    uint64_t crc_src = ecs_tree_crc64(src);

    /* Decode into a fresh tree of matching data_size. */
    ecs_tree_t* dst = (ecs_tree_t*)ecs_xcalloc(1, sizeof(ecs_tree_t));
    ecs_tree_init(dst, src->data_size, 0);

    ecs_deserializer_t d;
    ecs_deserializer_init(&d, buf, ((bytes + 7) / 8) * 8);
    int rc = ecs_tree_deserialize(dst, &d);
    EXPECT(rc == 0, tag);

    EXPECT(ecs_tree_masks_valid(dst), tag);
    EXPECT(ecs_tree_crc64(dst) == crc_src, tag);

    free_tree(dst);
    ecs_free(buf);
}

static void test_serialize_empty_tree(void) {
    ecs_tree_t* t = make_tree();
    ts_round_trip(t, "serialize empty tree round-trip");
    free_tree(t);
}

static void test_serialize_single_entity(void) {
    ecs_tree_t* t = make_tree();
    set_val(t, 42, 12345);
    ecs_tree_rollback(t);
    ts_round_trip(t, "serialize single entity round-trip");
    free_tree(t);
}

static void test_serialize_sparse_multi_l1(void) {
    ecs_tree_t* t = make_tree();
    /* Indices spread across multiple L1 buckets within one L2. */
    set_val(t, 0,    0xA);
    set_val(t, 63,   0xB);     /* same L1[0], boundary */
    set_val(t, 64,   0xC);     /* L1[1] */
    set_val(t, 4095, 0xD);     /* L1[63], same L2[0] */
    ecs_tree_rollback(t);
    ts_round_trip(t, "serialize sparse multi-L1 round-trip");
    free_tree(t);
}

static void test_serialize_dense_full_l1(void) {
    ecs_tree_t* t = make_tree();
    /* Fill an entire L1 (64 consecutive indices) - exercises all-set
       mask shortcut on the encoder side. */
    for (int i = 0; i < 64; i++) set_val(t, i, (comp_t)(i * 7 + 1));
    ecs_tree_rollback(t);
    ts_round_trip(t, "serialize full-L1 round-trip");
    free_tree(t);
}

static void test_serialize_multi_l2(void) {
    ecs_tree_t* t = make_tree();
    /* Index layout (18 bits): L3 = bits 12..17, L2 = bits 6..11, L1 = bits 0..5.
       Max valid index = (1 << 18) - 1 = 262143. */
    set_val(t, 0,      0x111);   /* L3[0] L2[0] L1[0] */
    set_val(t, 64,     0x222);   /* L3[0] L2[1] L1[0] */
    set_val(t, 4096,   0x333);   /* L3[1] L2[0] L1[0] */
    set_val(t, 100000, 0x444);   /* L3[24] L2[26] L1[32], a deep slot */
    ecs_tree_rollback(t);
    ts_round_trip(t, "serialize multi-L2 round-trip");
    free_tree(t);
}

static void test_serialize_tag_tree(void) {
    /* data_size == 0: presence-only, payload pass entirely skipped. */
    ecs_tree_t* t = (ecs_tree_t*)ecs_xcalloc(1, sizeof(ecs_tree_t));
    ecs_tree_init(t, 0, 0);
    /* Tag trees can't store data - flip presence bits via get_mut + remove. */
    ecs_tree_get_mut(t, 7);
    ecs_tree_get_mut(t, 200);
    ecs_tree_get_mut(t, 4097);
    ecs_tree_rollback(t);
    ts_round_trip(t, "serialize tag tree round-trip");
    free_tree(t);
}

/* Reuse path: deserialize INTO an already-populated tree. The deserializer
   must release stale L2/L1 nodes back to the pool and acquire (or repurpose)
   for the new state. CRC parity proves no leftover state polluted the dest. */
static void test_serialize_override_existing_tree(void) {
    ecs_tree_t* src = make_tree();
    set_val(src, 5,    0x1);
    set_val(src, 1000, 0x2);
    set_val(src, 4096, 0x3);
    ecs_tree_rollback(src);
    uint64_t crc_src = ecs_tree_crc64(src);

    /* Encode src */
    uint8_t* buf = (uint8_t*)ecs_xcalloc(1, (size_t)TS_BUFFER_BYTES);
    ecs_serializer_t s;
    ecs_serializer_init(&s, buf, TS_BUFFER_BYTES);
    ecs_tree_serialize(src, &s);
    ecs_serializer_flush_bits(&s);
    int32_t bytes = ecs_serializer_get_bytes_written(&s);

    /* Build a different state in dst - overlapping + non-overlapping indices. */
    ecs_tree_t* dst = make_tree();
    set_val(dst, 7,     0xAA);
    set_val(dst, 1000,  0xBB);     /* overlap with src */
    set_val(dst, 9999,  0xCC);
    set_val(dst, 50000, 0xDD);
    ecs_tree_rollback(dst);
    EXPECT(ecs_tree_crc64(dst) != crc_src, "pre-deserialize state differs");

    /* Deserialize over dst - must reuse pool nodes correctly. */
    ecs_deserializer_t d;
    ecs_deserializer_init(&d, buf, ((bytes + 7) / 8) * 8);
    int rc = ecs_tree_deserialize(dst, &d);
    EXPECT(rc == 0, "deserialize override returns 0");
    EXPECT(ecs_tree_masks_valid(dst), "override yields valid masks");
    EXPECT(ecs_tree_crc64(dst) == crc_src, "override yields src CRC");

    free_tree(src);
    free_tree(dst);
    ecs_free(buf);
}

/* Re-deserialize twice into the same tree - second round must fully replace
   the first round's state without leaks. */
static void test_serialize_back_to_back_overrides(void) {
    /* Build state A */
    ecs_tree_t* a = make_tree();
    set_val(a, 1, 0x10); set_val(a, 130, 0x11);
    ecs_tree_rollback(a);
    uint64_t crc_a = ecs_tree_crc64(a);

    /* Build state B (different) */
    ecs_tree_t* b = make_tree();
    set_val(b, 2000, 0x20); set_val(b, 5000, 0x21); set_val(b, 5001, 0x22);
    ecs_tree_rollback(b);
    uint64_t crc_b = ecs_tree_crc64(b);
    EXPECT(crc_a != crc_b, "states A and B differ");

    uint8_t* buf_a = (uint8_t*)ecs_xcalloc(1, (size_t)TS_BUFFER_BYTES);
    uint8_t* buf_b = (uint8_t*)ecs_xcalloc(1, (size_t)TS_BUFFER_BYTES);
    ecs_serializer_t sa, sb;
    ecs_serializer_init(&sa, buf_a, TS_BUFFER_BYTES);
    ecs_serializer_init(&sb, buf_b, TS_BUFFER_BYTES);
    ecs_tree_serialize(a, &sa); ecs_serializer_flush_bits(&sa);
    ecs_tree_serialize(b, &sb); ecs_serializer_flush_bits(&sb);
    int32_t bytes_a = ecs_serializer_get_bytes_written(&sa);
    int32_t bytes_b = ecs_serializer_get_bytes_written(&sb);

    ecs_tree_t* dst = make_tree();
    ecs_deserializer_t d;

    /* Load A */
    ecs_deserializer_init(&d, buf_a, ((bytes_a + 7) / 8) * 8);
    EXPECT(ecs_tree_deserialize(dst, &d) == 0, "load A");
    EXPECT(ecs_tree_crc64(dst) == crc_a, "after load A: crc == crc_a");

    /* Load B over A */
    ecs_deserializer_init(&d, buf_b, ((bytes_b + 7) / 8) * 8);
    EXPECT(ecs_tree_deserialize(dst, &d) == 0, "load B over A");
    EXPECT(ecs_tree_crc64(dst) == crc_b, "after load B: crc == crc_b");

    /* Reload A on top of B */
    ecs_deserializer_init(&d, buf_a, ((bytes_a + 7) / 8) * 8);
    EXPECT(ecs_tree_deserialize(dst, &d) == 0, "reload A over B");
    EXPECT(ecs_tree_crc64(dst) == crc_a, "after reload A: crc == crc_a");

    free_tree(a); free_tree(b); free_tree(dst);
    ecs_free(buf_a); ecs_free(buf_b);
}

/* Bad header rejection - wrong version. Tree state must NOT be touched on
   a -1 return (header validated before topology mutation). */
static void test_serialize_rejects_bad_header(void) {
    ecs_tree_t* t = make_tree();
    set_val(t, 99, 0xDEAD);
    ecs_tree_rollback(t);
    uint64_t crc_before = ecs_tree_crc64(t);

    /* version=99 (unknown), rest zeros - must reject before mutating tree. */
    uint8_t buf[32] = { 99, 0, 0 };
    ecs_deserializer_t d;
    ecs_deserializer_init(&d, buf, sizeof(buf));
    int rc = ecs_tree_deserialize(t, &d);
    EXPECT(rc == -1, "bad version returns -1");
    EXPECT(ecs_tree_crc64(t) == crc_before, "bad version leaves tree unchanged");

    free_tree(t);
}

/* ============================================================
   World-level serialize/deserialize round-trips.
   ============================================================ */

typedef uint16_t comp_b_t;   /* second component type with different size */

static ecs_world_t* ts_make_world(int n_trees, const size_t* sizes) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    for (int i = 0; i < n_trees; i++) {
        ecs_tree_init(&w->trees[i], sizes[i], 0);
        w->mask |= (uint64_t)1 << i;
    }
    return w;
}

static void ts_world_round_trip(ecs_world_t* src, const char* tag) {
    uint8_t* buf = (uint8_t*)ecs_xcalloc(1, (size_t)TS_BUFFER_BYTES);
    ecs_serializer_t s;
    ecs_serializer_init(&s, buf, TS_BUFFER_BYTES);
    ecs_world_serialize(src, &s);
    ecs_serializer_flush_bits(&s);
    int32_t bytes = ecs_serializer_get_bytes_written(&s);
    EXPECT(bytes > 0, tag);

    uint64_t crc_src = ecs_world_crc64(src);

    ecs_world_t* dst = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_deserializer_t d;
    ecs_deserializer_init(&d, buf, ((bytes + 7) / 8) * 8);
    int rc = ecs_world_deserialize(dst, &d);
    EXPECT(rc == 0, tag);
    EXPECT(ecs_world_crc64(dst) == crc_src, tag);
    EXPECT(dst->tick == src->tick, tag);  /* world->tick survives round-trip */
    EXPECT(dst->mask == src->mask, tag);

    ecs_world_destroy(dst); ecs_free(dst);
    ecs_free(buf);
}

static void test_world_serialize_empty(void) {
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ts_world_round_trip(w, "world serialize empty round-trip");
    ecs_world_destroy(w); ecs_free(w);
}

static void test_world_serialize_single_tree(void) {
    size_t sizes[1] = { sizeof(comp_t) };
    ecs_world_t* w = ts_make_world(1, sizes);
    set_val(&w->trees[0], 5,  0xAA);
    set_val(&w->trees[0], 70, 0xBB);
    ecs_world_rollback(w);
    ts_world_round_trip(w, "world serialize single tree");
    ecs_world_destroy(w); ecs_free(w);
}

static void test_world_serialize_multi_tree_mixed_sizes(void) {
    /* Trees at slots 0, 2, 5 with different data sizes - exercises
       tree mask sparsity + per-tree data_size round-trip. */
    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(comp_t), 0);      /* 4 bytes */
    ecs_tree_init(&w->trees[2], sizeof(comp_b_t), 0);    /* 2 bytes */
    ecs_tree_init(&w->trees[5], 0, 0);                    /* tag tree */
    w->mask = (1ULL << 0) | (1ULL << 2) | (1ULL << 5);

    set_val(&w->trees[0], 1,    0x1111);
    set_val(&w->trees[0], 2000, 0x2222);
    *(comp_b_t*)ecs_tree_get_mut(&w->trees[2], 9)    = (comp_b_t){0xABCD};
    *(comp_b_t*)ecs_tree_get_mut(&w->trees[2], 5000) = (comp_b_t){0xBEEF};
    ecs_tree_get_mut(&w->trees[5], 42);
    ecs_tree_get_mut(&w->trees[5], 4096);
    ecs_world_rollback(w);

    ts_world_round_trip(w, "world serialize multi-tree mixed sizes");
    ecs_world_destroy(w); ecs_free(w);
}

/* Override path: deserialize world A onto a world that previously held
   world B with overlapping AND non-overlapping tree slots. */
static void test_world_serialize_override(void) {
    /* Source world: trees at slots {0, 3} */
    ecs_world_t* src = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&src->trees[0], sizeof(comp_t), 0);
    ecs_tree_init(&src->trees[3], sizeof(comp_t), 0);
    src->mask = (1ULL << 0) | (1ULL << 3);
    set_val(&src->trees[0], 10, 0xA1);
    set_val(&src->trees[3], 50, 0xB2);
    ecs_world_rollback(src);
    uint64_t crc_src = ecs_world_crc64(src);

    /* Encode src */
    uint8_t* buf = (uint8_t*)ecs_xcalloc(1, (size_t)TS_BUFFER_BYTES);
    ecs_serializer_t s;
    ecs_serializer_init(&s, buf, TS_BUFFER_BYTES);
    ecs_world_serialize(src, &s);
    ecs_serializer_flush_bits(&s);
    int32_t bytes = ecs_serializer_get_bytes_written(&s);

    /* Destination: trees at slots {0, 1, 7} - overlap at 0, drop 1+7, add 3 */
    ecs_world_t* dst = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&dst->trees[0], sizeof(comp_t), 0);
    ecs_tree_init(&dst->trees[1], sizeof(comp_t), 0);
    ecs_tree_init(&dst->trees[7], sizeof(comp_t), 0);
    dst->mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 7);
    set_val(&dst->trees[0], 99, 0xFF);
    set_val(&dst->trees[1], 1,  0x01);
    set_val(&dst->trees[7], 8000, 0x80);
    ecs_world_rollback(dst);
    EXPECT(ecs_world_crc64(dst) != crc_src, "pre-deserialize world differs");

    ecs_deserializer_t d;
    ecs_deserializer_init(&d, buf, ((bytes + 7) / 8) * 8);
    int rc = ecs_world_deserialize(dst, &d);
    EXPECT(rc == 0, "world deserialize override returns 0");
    EXPECT(dst->mask == src->mask, "world mask matches src");
    EXPECT(ecs_world_crc64(dst) == crc_src, "world CRC matches src after override");

    ecs_world_destroy(src); ecs_free(src);
    ecs_world_destroy(dst); ecs_free(dst);
    ecs_free(buf);
}

/* ============================================================
   Wire-size sanity probe: 4 trees (pos fixed_4_t, vel fixed_4_t,
   transform TRS, flags tag), 5000 entities sparsely placed across
   100k slot range. Prints serialized world byte count for inspection.
   ============================================================ */

static void test_world_serialize_size_5k_4trees(void) {
    enum { N_ENT = 5000, IDX_RANGE = 100000, STRIDE = IDX_RANGE / N_ENT };
    enum { BIG_BUF = 2 * 1024 * 1024 };

    ecs_world_t* w = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_tree_init(&w->trees[0], sizeof(fixed_4_t),   0);  /* pos    */
    ecs_tree_init(&w->trees[1], sizeof(fixed_4_t),   0);  /* vel    */
    ecs_tree_init(&w->trees[2], sizeof(transform_t), 0);  /* transform  */
    ecs_tree_init(&w->trees[3], 0,                   0);  /* flags  */
    w->mask = 0xFu;

    for (int i = 0; i < N_ENT; i++) {
        int idx = i * STRIDE;
        fixed_4_t*   p = (fixed_4_t*)  ecs_tree_get_mut(&w->trees[0], idx);
        fixed_4_t*   v = (fixed_4_t*)  ecs_tree_get_mut(&w->trees[1], idx);
        transform_t* x = (transform_t*)ecs_tree_get_mut(&w->trees[2], idx);
        ecs_tree_get_mut(&w->trees[3], idx);
        *p = fixed4_vec3((fixed_t)idx, (fixed_t)(idx + 1), (fixed_t)(idx + 2));
        *v = fixed4_vec3((fixed_t)i,   (fixed_t)(i * 2),   (fixed_t)(i * 3));
        *x = transform_identity();
    }
    ecs_world_rollback(w);

    uint8_t* buf = (uint8_t*)ecs_xcalloc(1, (size_t)BIG_BUF);
    ecs_serializer_t s;
    ecs_serializer_init(&s, buf, BIG_BUF);
    ecs_world_serialize(w, &s);
    ecs_serializer_flush_bits(&s);
    int32_t bytes = ecs_serializer_get_bytes_written(&s);

    printf("  world_serialize: %d bytes (entities=%d, trees=4 [fixed4+fixed4+transform+tag], range=%d slots)\n",
           bytes, N_ENT, IDX_RANGE);
    EXPECT(bytes > 0, "world serialize produced output");

    /* Naive baseline: per tree, write 18-bit count then for each entry an
       18-bit raw index (covers IDX_RANGE up to 2^18) immediately followed by
       the raw payload bytes. Index width 18 bits is reused for the count
       header too -- max entries per tree bounded by index range, so 18 bits
       suffice. No bitmask hierarchy, no run encoding -- this is the dumb
       wire format the bitmask serializer is meant to beat. */
    uint8_t* nbuf = (uint8_t*)ecs_xcalloc(1, (size_t)BIG_BUF);
    ecs_serializer_t ns;
    ecs_serializer_init(&ns, nbuf, BIG_BUF);
    for (int t = 0; t < 4; ++t) {
        ecs_serializer_write_bits(&ns, (uint64_t)N_ENT, 18);
        for (int i = 0; i < N_ENT; ++i) {
            int idx = i * STRIDE;
            ecs_serializer_write_bits(&ns, (uint64_t)idx, 18);
            if (t == 0) {
                fixed_4_t p = fixed4_vec3((fixed_t)idx, (fixed_t)(idx + 1), (fixed_t)(idx + 2));
                ecs_serializer_write_bytes(&ns, (const uint8_t*)&p, (int32_t)sizeof(p));
            } else if (t == 1) {
                fixed_4_t v = fixed4_vec3((fixed_t)i, (fixed_t)(i * 2), (fixed_t)(i * 3));
                ecs_serializer_write_bytes(&ns, (const uint8_t*)&v, (int32_t)sizeof(v));
            } else if (t == 2) {
                transform_t x = transform_identity();
                ecs_serializer_write_bytes(&ns, (const uint8_t*)&x, (int32_t)sizeof(x));
            }
            /* t == 3: tag tree, presence-only -- no payload */
        }
    }
    ecs_serializer_flush_bits(&ns);
    int32_t naive_bytes = ecs_serializer_get_bytes_written(&ns);
    printf("  naive_serialize: %d bytes (18-bit count + 18-bit idx + payload per entry, per tree)\n", naive_bytes);

    /* zstd compression probe on bitmask serializer output + bit-exact round-trip. */
    size_t zbound = ZSTD_compressBound((size_t)bytes);
    uint8_t* zbuf = (uint8_t*)ecs_xcalloc(1, zbound);
    size_t zsize = ZSTD_compress(zbuf, zbound, buf, (size_t)bytes, ZSTD_CLEVEL_DEFAULT);
    EXPECT(!ZSTD_isError(zsize), "zstd compress ok");
    printf("  zstd(world):     %zu bytes (ratio %.2fx vs raw, level=%d)\n",
           zsize, (double)bytes / (double)zsize, ZSTD_CLEVEL_DEFAULT);

    /* zstd on the naive baseline -- shows whether zstd alone closes the gap. */
    size_t nzbound = ZSTD_compressBound((size_t)naive_bytes);
    uint8_t* nzbuf = (uint8_t*)ecs_xcalloc(1, nzbound);
    size_t nzsize = ZSTD_compress(nzbuf, nzbound, nbuf, (size_t)naive_bytes, ZSTD_CLEVEL_DEFAULT);
    EXPECT(!ZSTD_isError(nzsize), "zstd compress (naive) ok");
    printf("  zstd(naive):     %zu bytes (ratio %.2fx vs raw)\n",
           nzsize, (double)naive_bytes / (double)nzsize);
    printf("  bitmask vs naive: raw %.2fx smaller, zstd %.2fx smaller\n",
           (double)naive_bytes / (double)bytes,
           (double)nzsize / (double)zsize);
    ecs_free(nbuf);
    ecs_free(nzbuf);

    uint8_t* dbuf = (uint8_t*)ecs_xcalloc(1, (size_t)BIG_BUF);
    size_t dsize = ZSTD_decompress(dbuf, (size_t)BIG_BUF, zbuf, zsize);
    EXPECT(!ZSTD_isError(dsize), "zstd decompress ok");
    EXPECT(dsize == (size_t)bytes, "zstd decompressed size matches");
    EXPECT(memcmp(dbuf, buf, (size_t)bytes) == 0, "zstd round-trip bit-exact");

    /* Sanity: round-trip CRC parity (decode from zstd-decompressed buffer). */
    uint64_t crc_src = ecs_world_crc64(w);
    ecs_world_t* dst = (ecs_world_t*)ecs_xcalloc(1, sizeof(ecs_world_t));
    ecs_deserializer_t d;
    ecs_deserializer_init(&d, dbuf, ((bytes + 7) / 8) * 8);
    int rc = ecs_world_deserialize(dst, &d);
    EXPECT(rc == 0, "5k 4-tree world deserializes");
    EXPECT(ecs_world_crc64(dst) == crc_src, "5k 4-tree world CRC matches");

    ecs_world_destroy(dst); ecs_free(dst);
    ecs_world_destroy(w);   ecs_free(w);
    ecs_free(buf);
    ecs_free(zbuf);
    ecs_free(dbuf);
}

static int test_serialize_all(void) {
    int before = g_failed;
    printf("=== serialize/deserialize tests ===\n\n");
    RUN_TEST(test_serialize_empty_tree);
    RUN_TEST(test_serialize_single_entity);
    RUN_TEST(test_serialize_sparse_multi_l1);
    RUN_TEST(test_serialize_dense_full_l1);
    RUN_TEST(test_serialize_multi_l2);
    RUN_TEST(test_serialize_tag_tree);
    RUN_TEST(test_serialize_override_existing_tree);
    RUN_TEST(test_serialize_back_to_back_overrides);
    RUN_TEST(test_serialize_rejects_bad_header);
    RUN_TEST(test_world_serialize_empty);
    RUN_TEST(test_world_serialize_single_tree);
    RUN_TEST(test_world_serialize_multi_tree_mixed_sizes);
    RUN_TEST(test_world_serialize_override);
    RUN_TEST(test_world_serialize_size_5k_4trees);
    int failed = g_failed - before;
    printf("\nserialize: %d failed\n", failed);
    return failed ? 1 : 0;
}
