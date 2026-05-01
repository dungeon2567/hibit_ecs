#pragma once

/*
    ecs_serializer — bitpacked writer/reader (C port of serialize.hpp,
    Mas Bandwidth LLC, BSD-3). Right-to-left scratch accumulator, 64-bit
    qword-aligned output, network byte order = little-endian.

    Header-only: every function static inline for cross-TU inlining.
    Word size: 64-bit.

    Naming:
      ecs_serializer_t   = writer
      ecs_deserializer_t = reader
*/

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
    #define ECS_SER_RESTRICT       __restrict__
    #define ECS_SER_FORCE_INLINE   static inline __attribute__((always_inline))
    #define ECS_SER_LIKELY(x)      __builtin_expect(!!(x), 1)
    #define ECS_SER_UNLIKELY(x)    __builtin_expect(!!(x), 0)
    #define ECS_SER_BSWAP64(x)     __builtin_bswap64(x)
    #define ECS_SER_BSWAP32(x)     __builtin_bswap32(x)
    #define ECS_SER_BSWAP16(x)     __builtin_bswap16(x)
#elif defined(_MSC_VER)
    #include <stdlib.h>
    #define ECS_SER_RESTRICT       __restrict
    #define ECS_SER_FORCE_INLINE   static __forceinline
    #define ECS_SER_LIKELY(x)      (x)
    #define ECS_SER_UNLIKELY(x)    (x)
    #define ECS_SER_BSWAP64(x)     _byteswap_uint64(x)
    #define ECS_SER_BSWAP32(x)     _byteswap_ulong(x)
    #define ECS_SER_BSWAP16(x)     _byteswap_ushort(x)
#else
    #define ECS_SER_RESTRICT
    #define ECS_SER_FORCE_INLINE   static inline
    #define ECS_SER_LIKELY(x)      (x)
    #define ECS_SER_UNLIKELY(x)    (x)
    ECS_SER_FORCE_INLINE uint64_t ECS_SER_BSWAP64(uint64_t v) {
        v = (v & 0x00000000FFFFFFFFULL) << 32 | (v & 0xFFFFFFFF00000000ULL) >> 32;
        v = (v & 0x0000FFFF0000FFFFULL) << 16 | (v & 0xFFFF0000FFFF0000ULL) >> 16;
        v = (v & 0x00FF00FF00FF00FFULL) <<  8 | (v & 0xFF00FF00FF00FF00ULL) >>  8;
        return v;
    }
    ECS_SER_FORCE_INLINE uint32_t ECS_SER_BSWAP32(uint32_t v) {
        return (v & 0x000000FFu) << 24 | (v & 0x0000FF00u) <<  8
             | (v & 0x00FF0000u) >>  8 | (v & 0xFF000000u) >> 24;
    }
    ECS_SER_FORCE_INLINE uint16_t ECS_SER_BSWAP16(uint16_t v) {
        return (uint16_t)((v & 0x00FFu) << 8 | (v & 0xFF00u) >> 8);
    }
#endif

#ifndef ecs_ser_assert
    #define ecs_ser_assert assert
#endif

#if !defined(ECS_SER_LITTLE_ENDIAN) && !defined(ECS_SER_BIG_ENDIAN)
    #if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        #define ECS_SER_BIG_ENDIAN 1
    #else
        #define ECS_SER_LITTLE_ENDIAN 1
    #endif
#endif
#ifndef ECS_SER_LITTLE_ENDIAN
    #define ECS_SER_LITTLE_ENDIAN 0
#endif
#ifndef ECS_SER_BIG_ENDIAN
    #define ECS_SER_BIG_ENDIAN 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- byte order ---------- */

ECS_SER_FORCE_INLINE uint64_t ecs_ser_host_to_network64(uint64_t v) {
#if ECS_SER_BIG_ENDIAN
    return ECS_SER_BSWAP64(v);
#else
    return v;
#endif
}
ECS_SER_FORCE_INLINE uint32_t ecs_ser_host_to_network32(uint32_t v) {
#if ECS_SER_BIG_ENDIAN
    return ECS_SER_BSWAP32(v);
#else
    return v;
#endif
}
ECS_SER_FORCE_INLINE uint16_t ecs_ser_host_to_network16(uint16_t v) {
#if ECS_SER_BIG_ENDIAN
    return ECS_SER_BSWAP16(v);
#else
    return v;
#endif
}
ECS_SER_FORCE_INLINE uint64_t ecs_ser_network_to_host64(uint64_t v) { return ecs_ser_host_to_network64(v); }
ECS_SER_FORCE_INLINE uint32_t ecs_ser_network_to_host32(uint32_t v) { return ecs_ser_host_to_network32(v); }
ECS_SER_FORCE_INLINE uint16_t ecs_ser_network_to_host16(uint16_t v) { return ecs_ser_host_to_network16(v); }

/* ---------- zig-zag ---------- */

ECS_SER_FORCE_INLINE uint32_t ecs_ser_signed_to_unsigned(int32_t n) {
    return (uint32_t)((n << 1) ^ (n >> 31));
}
ECS_SER_FORCE_INLINE int32_t ecs_ser_unsigned_to_signed(uint32_t n) {
    return (int32_t)((n >> 1) ^ (uint32_t)(-(int32_t)(n & 1)));
}

/* ---------- writer ---------- */

typedef struct {
    uint64_t* data;           /* qword buffer */
    uint64_t  scratch;        /* right-to-left accumulator, < 64 bits live */
    int32_t   num_bits;
    int32_t   num_words;
    int32_t   bits_written;
    int32_t   word_index;
    int32_t   scratch_bits;
} ecs_serializer_t;

ECS_SER_FORCE_INLINE void ecs_serializer_init(ecs_serializer_t* w, void* data, int32_t bytes) {
    ecs_ser_assert(data);
    ecs_ser_assert((bytes % 8) == 0);
    w->data         = (uint64_t*)data;
    w->scratch      = 0;
    w->num_words    = bytes / 8;
    w->num_bits     = w->num_words * 64;
    w->bits_written = 0;
    w->word_index   = 0;
    w->scratch_bits = 0;
}

ECS_SER_FORCE_INLINE void ecs_serializer_write_bits(ecs_serializer_t* w, uint64_t value, int32_t bits) {
    ecs_ser_assert(bits > 0);
    ecs_ser_assert(bits <= 64);
    ecs_ser_assert(w->bits_written + bits <= w->num_bits);
    /* (1ULL << 64) is UB, so skip the fits-in-bits check for the full-width case. */
    ecs_ser_assert(bits == 64 || value <= ((1ULL << bits) - 1));

    if (w->scratch_bits + bits >= 64) {
        const int32_t low = 64 - w->scratch_bits; /* bits placed in current scratch */
        /* For 64-bit value with scratch_bits>0 the shift drops high bits — fine,
           they're recovered from `value >> low` below. For scratch_bits==0 the
           shift is by 0, no truncation. scratch_bits is always in [0, 63] by
           invariant, so `low` is always in [1, 64]. */
        w->scratch |= value << w->scratch_bits;
        ecs_ser_assert(w->word_index < w->num_words);
        w->data[w->word_index++] = ecs_ser_host_to_network64(w->scratch);
        if (low < bits) {
            /* low < bits implies low < 64, so the shift is well-defined. */
            w->scratch      = value >> low;
            w->scratch_bits = bits - low;
        } else {
            w->scratch      = 0;
            w->scratch_bits = 0;
        }
    } else {
        w->scratch      |= value << w->scratch_bits;
        w->scratch_bits += bits;
    }
    w->bits_written += bits;
}

ECS_SER_FORCE_INLINE void ecs_serializer_flush_bits(ecs_serializer_t* w) {
    if (w->scratch_bits != 0) {
        ecs_ser_assert(w->scratch_bits <= 64);
        ecs_ser_assert(w->word_index < w->num_words);
        w->data[w->word_index++] = ecs_ser_host_to_network64(w->scratch);
        w->scratch      = 0;
        w->scratch_bits = 0;
    }
}

ECS_SER_FORCE_INLINE int32_t ecs_serializer_get_align_bits(const ecs_serializer_t* w) {
    return (8 - (w->bits_written % 8)) % 8;
}

ECS_SER_FORCE_INLINE void ecs_serializer_write_align(ecs_serializer_t* w) {
    const int32_t rem = w->bits_written % 8;
    if (rem != 0) {
        ecs_serializer_write_bits(w, 0u, 8 - rem);
        ecs_ser_assert((w->bits_written % 8) == 0);
    }
}

ECS_SER_FORCE_INLINE int32_t ecs_serializer_get_bits_written(const ecs_serializer_t* w)   { return w->bits_written; }
ECS_SER_FORCE_INLINE int32_t ecs_serializer_get_bits_available(const ecs_serializer_t* w) { return w->num_bits - w->bits_written; }
ECS_SER_FORCE_INLINE const uint8_t* ecs_serializer_get_data(const ecs_serializer_t* w)    { return (const uint8_t*)w->data; }
ECS_SER_FORCE_INLINE int32_t ecs_serializer_get_bytes_written(const ecs_serializer_t* w)  { return (w->bits_written + 7) / 8; }

ECS_SER_FORCE_INLINE void ecs_serializer_write_bytes(ecs_serializer_t* w, const uint8_t* ECS_SER_RESTRICT data, int32_t bytes) {
    /* Self-align to next byte boundary if currently misaligned. No-op when
       already aligned. Lets callers mix bit-level and byte-level writes
       without tracking alignment themselves. */
    ecs_serializer_write_align(w);
    ecs_ser_assert(w->bits_written + bytes * 8 <= w->num_bits);
    ecs_ser_assert((w->bits_written % 8) == 0);

    int32_t head_bytes = (8 - (w->bits_written % 64) / 8) % 8;
    if (head_bytes > bytes) head_bytes = bytes;
    for (int32_t i = 0; i < head_bytes; ++i)
        ecs_serializer_write_bits(w, data[i], 8);
    if (head_bytes == bytes) return;

    ecs_serializer_flush_bits(w);
    ecs_ser_assert(ecs_serializer_get_align_bits(w) == 0);

    int32_t num_words = (bytes - head_bytes) / 8;
    if (num_words > 0) {
        ecs_ser_assert((w->bits_written % 64) == 0);
        memcpy(&w->data[w->word_index], data + head_bytes, (size_t)num_words * 8);
        w->bits_written += num_words * 64;
        w->word_index   += num_words;
        w->scratch       = 0;
    }

    ecs_ser_assert(ecs_serializer_get_align_bits(w) == 0);

    int32_t tail_start = head_bytes + num_words * 8;
    int32_t tail_bytes = bytes - tail_start;
    ecs_ser_assert(tail_bytes >= 0 && tail_bytes < 8);
    for (int32_t i = 0; i < tail_bytes; ++i)
        ecs_serializer_write_bits(w, data[tail_start + i], 8);

    ecs_ser_assert(ecs_serializer_get_align_bits(w) == 0);
    ecs_ser_assert(head_bytes + num_words * 8 + tail_bytes == bytes);
}

/* ---------- reader ---------- */

typedef struct {
    const uint64_t* data;
    uint64_t        scratch;
    int32_t         num_bits;
    int32_t         num_bytes;
    int32_t         num_words;
    int32_t         bits_read;
    int32_t         scratch_bits;
    int32_t         word_index;
} ecs_deserializer_t;

ECS_SER_FORCE_INLINE void ecs_deserializer_init(ecs_deserializer_t* r, const void* data, int32_t bytes) {
    ecs_ser_assert(data);
    r->data         = (const uint64_t*)data;
    r->scratch      = 0;
    r->num_bytes    = bytes;
    r->num_words    = (bytes + 7) / 8;
    r->num_bits     = bytes * 8;
    r->bits_read    = 0;
    r->scratch_bits = 0;
    r->word_index   = 0;
}

ECS_SER_FORCE_INLINE int ecs_deserializer_would_read_past_end(const ecs_deserializer_t* r, int32_t bits) {
    return r->bits_read + bits > r->num_bits;
}
ECS_SER_FORCE_INLINE int ecs_deserializer_has_data(const ecs_deserializer_t* r) {
    return r->bits_read < r->num_bits;
}

ECS_SER_FORCE_INLINE uint64_t ecs_deserializer_read_bits(ecs_deserializer_t* r, int32_t bits) {
    ecs_ser_assert(bits > 0);
    ecs_ser_assert(bits <= 64);
    ecs_ser_assert(r->bits_read + bits <= r->num_bits);

    r->bits_read += bits;
    ecs_ser_assert(r->scratch_bits >= 0 && r->scratch_bits <= 64);

    uint64_t out;
    if (r->scratch_bits < bits) {
        ecs_ser_assert(r->word_index < r->num_words);
        const uint64_t next = ecs_ser_network_to_host64(r->data[r->word_index++]);
        const int32_t  need = bits - r->scratch_bits;
        if (need == 64) {
            /* scratch_bits == 0 here. Take full word. (1ULL<<64)-1 / shift-by-64 are UB. */
            out             = next;
            r->scratch      = 0;
            r->scratch_bits = 0;
        } else {
            const uint64_t mask = ((uint64_t)1 << need) - 1;
            out             = r->scratch | ((next & mask) << r->scratch_bits);
            r->scratch      = next >> need;
            r->scratch_bits = 64 - need;
        }
    } else {
        /* scratch_bits >= bits, with scratch_bits in [0, 63] by invariant, so
           bits is in [1, 63] here — (1ULL<<bits)-1 is well-defined. */
        out = r->scratch & (((uint64_t)1 << bits) - 1);
        r->scratch      >>= bits;
        r->scratch_bits  -= bits;
    }
    return out;
}

ECS_SER_FORCE_INLINE int32_t ecs_deserializer_get_align_bits(const ecs_deserializer_t* r) {
    return (8 - r->bits_read % 8) % 8;
}

ECS_SER_FORCE_INLINE int ecs_deserializer_read_align(ecs_deserializer_t* r) {
    const int32_t rem = r->bits_read % 8;
    if (rem != 0) {
        const uint64_t v = ecs_deserializer_read_bits(r, 8 - rem);
        ecs_ser_assert(r->bits_read % 8 == 0);
        if (v != 0) return 0;
    }
    return 1;
}

ECS_SER_FORCE_INLINE int32_t ecs_deserializer_get_bits_read(const ecs_deserializer_t* r)      { return r->bits_read; }
ECS_SER_FORCE_INLINE int32_t ecs_deserializer_get_bits_remaining(const ecs_deserializer_t* r) { return r->num_bits - r->bits_read; }

ECS_SER_FORCE_INLINE void ecs_deserializer_read_bytes(ecs_deserializer_t* r, uint8_t* ECS_SER_RESTRICT data, int32_t bytes) {
    /* Self-align to next byte boundary, mirroring ecs_serializer_write_bytes. */
    ecs_deserializer_read_align(r);
    ecs_ser_assert(r->bits_read + bytes * 8 <= r->num_bits);
    ecs_ser_assert((r->bits_read % 8) == 0);

    int32_t head_bytes = (8 - (r->bits_read % 64) / 8) % 8;
    if (head_bytes > bytes) head_bytes = bytes;
    for (int32_t i = 0; i < head_bytes; ++i)
        data[i] = (uint8_t)ecs_deserializer_read_bits(r, 8);
    if (head_bytes == bytes) return;

    ecs_ser_assert(ecs_deserializer_get_align_bits(r) == 0);

    int32_t num_words = (bytes - head_bytes) / 8;
    if (num_words > 0) {
        ecs_ser_assert((r->bits_read % 64) == 0);
        memcpy(data + head_bytes, &r->data[r->word_index], (size_t)num_words * 8);
        r->bits_read    += num_words * 64;
        r->word_index   += num_words;
        r->scratch       = 0;
        r->scratch_bits  = 0;
    }

    ecs_ser_assert(ecs_deserializer_get_align_bits(r) == 0);

    int32_t tail_start = head_bytes + num_words * 8;
    int32_t tail_bytes = bytes - tail_start;
    ecs_ser_assert(tail_bytes >= 0 && tail_bytes < 8);
    for (int32_t i = 0; i < tail_bytes; ++i)
        data[tail_start + i] = (uint8_t)ecs_deserializer_read_bits(r, 8);

    ecs_ser_assert(ecs_deserializer_get_align_bits(r) == 0);
    ecs_ser_assert(head_bytes + num_words * 8 + tail_bytes == bytes);
}

#ifdef __cplusplus
}
#endif
