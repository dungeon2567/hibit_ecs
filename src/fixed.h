#pragma once

#include <assert.h>
#include <stdint.h>
#include <string.h>

/* Q16.16 fixed-point. fixed_t is int32, 16 fractional bits.
   Range: [-32768, 32768), resolution 1/65536. */

#define FIXED_SHIFT 16
#define FIXED_ONE   ((int32_t)(1 << FIXED_SHIFT))
#define FIXED_HALF  ((int32_t)(1 << (FIXED_SHIFT - 1)))
#define FIXED_MASK  ((int32_t)(FIXED_ONE - 1))

typedef int32_t fixed_t;

/* ---------- scalar ---------- */

/* Cast through uint32 to avoid signed-shift UB when x is negative or
   when (x << 16) overflows int32 (e.g. x >= 32768). */
static inline fixed_t fixed_from_int(int32_t x)   { return (fixed_t)((uint32_t)x << FIXED_SHIFT); }
static inline int32_t fixed_to_int(fixed_t x)     { return x >> FIXED_SHIFT; }
static inline fixed_t fixed_from_float(float f)   { return (fixed_t)(f * (float)FIXED_ONE); }
static inline float   fixed_to_float(fixed_t x)   { return (float)x / (float)FIXED_ONE; }

/* Build a Q16.16 value from a decimal-style integer pair: whole + frac/10^digits.
   fixed_from_parts(3, 14)   -> 3.14
   fixed_from_parts(3, 5)    -> 3.5
   fixed_from_parts(3, 7071) -> 3.7071
   fixed_from_parts(-2, 25)  -> -2.25
   fixed_from_parts(3, 0)    -> 3.0

   Fractional digit count is inferred from `frac` itself, so DO NOT pass
   leading zeros: 014 is octal in C and 09 is a syntax error.  Result is
   the floor of (whole + frac/10^digits) * 65536; not bit-exact for
   non-terminating binary fractions. */
static inline fixed_t fixed_from_parts(int32_t whole, uint32_t frac) {
    fixed_t w = fixed_from_int(whole);
    if (frac == 0) return w;
    uint32_t den = 1;
    for (uint32_t f = frac; f > 0; f /= 10) den *= 10;
    fixed_t fp = (fixed_t)(((uint64_t)frac << FIXED_SHIFT) / den);
    return whole < 0 ? w - fp : w + fp;
}

static inline fixed_t fixed_add(fixed_t a, fixed_t b) { return a + b; }
static inline fixed_t fixed_sub(fixed_t a, fixed_t b) { return a - b; }

static inline fixed_t fixed_mul(fixed_t a, fixed_t b) {
    return (fixed_t)(((int64_t)a * (int64_t)b) >> FIXED_SHIFT);
}

/* Multiply by FIXED_ONE rather than shifting left — signed left-shift of a
   negative value is UB in C, multiplication is not. Compiler folds the
   constant multiply back into a shift on the unsigned representation. */
static inline fixed_t fixed_div(fixed_t a, fixed_t b) {
    return (fixed_t)(((int64_t)a * (int64_t)FIXED_ONE) / (int64_t)b);
}

static inline fixed_t fixed_min(fixed_t a, fixed_t b) { return a < b ? a : b; }
static inline fixed_t fixed_max(fixed_t a, fixed_t b) { return a > b ? a : b; }

/* ---------- backend selection ---------- */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define FIXED_BACKEND_X86 1
#  include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
#  define FIXED_BACKEND_NEON 1
#  include <arm_neon.h>
#elif defined(__wasm_simd128__)
#  define FIXED_BACKEND_WASM 1
#  include <wasm_simd128.h>
#else
#  define FIXED_BACKEND_SCALAR 1
#endif

/* ============================================================
   4-wide SIMD: 4 packed int32 fixed_t.
   Backends provide *_impl_ for arithmetic/compare ops; load/store/set are
   public unchanged. Verified public wrappers below.
   ============================================================ */

/* fixed_4_t is a union: `.simd` for SIMD intrinsic plumbing, `.e[4]` flat lane
   view, anonymous struct `.x/.y/.z/.w` for vec3/vec4 named-axis access.
   All four backends keep the same union shape so callers see a single type. */

#if FIXED_BACKEND_X86

typedef union {
    __m128i simd;
    fixed_t e[4];
    struct { fixed_t x, y, z, w; };
} fixed_4_t;

static inline fixed_4_t fixed4_zero(void)                            { fixed_4_t r; r.simd = _mm_setzero_si128(); return r; }
static inline fixed_4_t fixed4_set1(fixed_t x)                       { fixed_4_t r; r.simd = _mm_set1_epi32(x); return r; }
static inline fixed_4_t fixed4_set(fixed_t a, fixed_t b, fixed_t c, fixed_t d) { fixed_4_t r; r.simd = _mm_set_epi32(d, c, b, a); return r; }
static inline fixed_4_t fixed4_load(const fixed_t* p)                { fixed_4_t r; r.simd = _mm_loadu_si128((const __m128i*)p); return r; }
static inline fixed_4_t fixed4_load_aligned(const fixed_t* p)        { fixed_4_t r; r.simd = _mm_load_si128((const __m128i*)p); return r; }
static inline void      fixed4_store(fixed_t* p, fixed_4_t v)        { _mm_storeu_si128((__m128i*)p, v.simd); }
static inline void      fixed4_store_aligned(fixed_t* p, fixed_4_t v){ _mm_store_si128((__m128i*)p, v.simd); }

static inline fixed_4_t fixed4_add_impl_(fixed_4_t a, fixed_4_t b) { fixed_4_t r; r.simd = _mm_add_epi32(a.simd, b.simd); return r; }
static inline fixed_4_t fixed4_sub_impl_(fixed_4_t a, fixed_4_t b) { fixed_4_t r; r.simd = _mm_sub_epi32(a.simd, b.simd); return r; }
static inline fixed_4_t fixed4_neg_impl_(fixed_4_t a)              { fixed_4_t r; r.simd = _mm_sub_epi32(_mm_setzero_si128(), a.simd); return r; }

/* mul: 32x32 -> 64-bit per lane, shift right by FIXED_SHIFT, repack low 32 bits.
   Logical srli on int64 product preserves the correct low-32 result whenever
   the final value fits in int32 (sign extension above bit 47 is shifted out). */
static inline fixed_4_t fixed4_mul_impl_(fixed_4_t a, fixed_4_t b) {
    __m128i ae = _mm_mul_epi32(a.simd, b.simd);                      /* lanes 0,2 -> int64 */
    __m128i ash = _mm_shuffle_epi32(a.simd, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i bsh = _mm_shuffle_epi32(b.simd, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i ao = _mm_mul_epi32(ash, bsh);                            /* lanes 1,3 -> int64 */
    ae = _mm_srli_epi64(ae, FIXED_SHIFT);
    ao = _mm_srli_epi64(ao, FIXED_SHIFT);
    __m128i ao_shifted = _mm_shuffle_epi32(ao, _MM_SHUFFLE(2, 2, 0, 0));
    fixed_4_t r; r.simd = _mm_blend_epi16(ae, ao_shifted, 0xCC); return r;
}

static inline fixed_4_t fixed4_div_impl_(fixed_4_t a, fixed_4_t b) {
    fixed_4_t r;
    for (int i = 0; i < 4; ++i) r.e[i] = fixed_div(a.e[i], b.e[i]);
    return r;
}

/* Compare ops return a 4-bit mask: bit i set iff lane i true. */
static inline uint8_t fixed4_eq_impl_(fixed_4_t a, fixed_4_t b) { return (uint8_t)_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(a.simd, b.simd))); }
static inline uint8_t fixed4_gt_impl_(fixed_4_t a, fixed_4_t b) { return (uint8_t)_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a.simd, b.simd))); }
static inline uint8_t fixed4_lt_impl_(fixed_4_t a, fixed_4_t b) { return (uint8_t)_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(b.simd, a.simd))); }
static inline uint8_t fixed4_ge_impl_(fixed_4_t a, fixed_4_t b) { return (uint8_t)((~_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(b.simd, a.simd)))) & 0xF); }
static inline uint8_t fixed4_le_impl_(fixed_4_t a, fixed_4_t b) { return (uint8_t)((~_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a.simd, b.simd)))) & 0xF); }

#elif FIXED_BACKEND_NEON

typedef union {
    int32x4_t simd;
    fixed_t e[4];
    struct { fixed_t x, y, z, w; };
} fixed_4_t;

static inline fixed_4_t fixed4_zero(void)                            { fixed_4_t r; r.simd = vdupq_n_s32(0); return r; }
static inline fixed_4_t fixed4_set1(fixed_t x)                       { fixed_4_t r; r.simd = vdupq_n_s32(x); return r; }
static inline fixed_4_t fixed4_set(fixed_t a, fixed_t b, fixed_t c, fixed_t d) {
    int32_t v[4] = { a, b, c, d };
    fixed_4_t r; r.simd = vld1q_s32(v); return r;
}
static inline fixed_4_t fixed4_load(const fixed_t* p)                { fixed_4_t r; r.simd = vld1q_s32(p); return r; }
static inline fixed_4_t fixed4_load_aligned(const fixed_t* p)        { fixed_4_t r; r.simd = vld1q_s32(p); return r; }
static inline void      fixed4_store(fixed_t* p, fixed_4_t v)        { vst1q_s32(p, v.simd); }
static inline void      fixed4_store_aligned(fixed_t* p, fixed_4_t v){ vst1q_s32(p, v.simd); }

static inline fixed_4_t fixed4_add_impl_(fixed_4_t a, fixed_4_t b) { fixed_4_t r; r.simd = vaddq_s32(a.simd, b.simd); return r; }
static inline fixed_4_t fixed4_sub_impl_(fixed_4_t a, fixed_4_t b) { fixed_4_t r; r.simd = vsubq_s32(a.simd, b.simd); return r; }
static inline fixed_4_t fixed4_neg_impl_(fixed_4_t a)              { fixed_4_t r; r.simd = vnegq_s32(a.simd); return r; }

/* Widen int32x4 -> int64x2 lo/hi, multiply, logical-shift-right by FIXED_SHIFT, narrow back. */
static inline fixed_4_t fixed4_mul_impl_(fixed_4_t a, fixed_4_t b) {
    int64x2_t lo64 = vmull_s32(vget_low_s32(a.simd),  vget_low_s32(b.simd));
    int64x2_t hi64 = vmull_s32(vget_high_s32(a.simd), vget_high_s32(b.simd));
    uint64x2_t lo_u = vshrq_n_u64(vreinterpretq_u64_s64(lo64), FIXED_SHIFT);
    uint64x2_t hi_u = vshrq_n_u64(vreinterpretq_u64_s64(hi64), FIXED_SHIFT);
    uint32x2_t lo32 = vmovn_u64(lo_u);
    uint32x2_t hi32 = vmovn_u64(hi_u);
    fixed_4_t r; r.simd = vreinterpretq_s32_u32(vcombine_u32(lo32, hi32)); return r;
}

static inline fixed_4_t fixed4_div_impl_(fixed_4_t a, fixed_4_t b) {
    fixed_4_t r;
    for (int i = 0; i < 4; ++i) r.e[i] = fixed_div(a.e[i], b.e[i]);
    return r;
}

/* Pack a 4x32 all-ones/zero compare result into a 4-bit mask. AND with
   {1,2,4,8} then horizontal-add. Works on both AArch64 and ARMv7 NEON. */
static inline uint8_t fixed4_movemask_neon_(uint32x4_t m) {
    static const uint32_t pow2_data[4] = { 1u, 2u, 4u, 8u };
    uint32x4_t bits = vandq_u32(m, vld1q_u32(pow2_data));
    uint32x2_t s = vadd_u32(vget_low_u32(bits), vget_high_u32(bits));
    s = vpadd_u32(s, s);
    return (uint8_t)vget_lane_u32(s, 0);
}

/* Compare ops return a 4-bit mask: bit i set iff lane i true. */
static inline uint8_t fixed4_eq_impl_(fixed_4_t a, fixed_4_t b) { return fixed4_movemask_neon_(vceqq_s32(a.simd, b.simd)); }
static inline uint8_t fixed4_gt_impl_(fixed_4_t a, fixed_4_t b) { return fixed4_movemask_neon_(vcgtq_s32(a.simd, b.simd)); }
static inline uint8_t fixed4_lt_impl_(fixed_4_t a, fixed_4_t b) { return fixed4_movemask_neon_(vcltq_s32(a.simd, b.simd)); }
static inline uint8_t fixed4_ge_impl_(fixed_4_t a, fixed_4_t b) { return fixed4_movemask_neon_(vcgeq_s32(a.simd, b.simd)); }
static inline uint8_t fixed4_le_impl_(fixed_4_t a, fixed_4_t b) { return fixed4_movemask_neon_(vcleq_s32(a.simd, b.simd)); }

#elif FIXED_BACKEND_WASM

typedef union {
    v128_t simd;
    fixed_t e[4];
    struct { fixed_t x, y, z, w; };
} fixed_4_t;

static inline fixed_4_t fixed4_zero(void)                            { fixed_4_t r; r.simd = wasm_i32x4_splat(0); return r; }
static inline fixed_4_t fixed4_set1(fixed_t x)                       { fixed_4_t r; r.simd = wasm_i32x4_splat(x); return r; }
static inline fixed_4_t fixed4_set(fixed_t a, fixed_t b, fixed_t c, fixed_t d) { fixed_4_t r; r.simd = wasm_i32x4_make(a, b, c, d); return r; }
static inline fixed_4_t fixed4_load(const fixed_t* p)                { fixed_4_t r; r.simd = wasm_v128_load(p); return r; }
static inline fixed_4_t fixed4_load_aligned(const fixed_t* p)        { fixed_4_t r; r.simd = wasm_v128_load(p); return r; }
static inline void      fixed4_store(fixed_t* p, fixed_4_t v)        { wasm_v128_store(p, v.simd); }
static inline void      fixed4_store_aligned(fixed_t* p, fixed_4_t v){ wasm_v128_store(p, v.simd); }

static inline fixed_4_t fixed4_add_impl_(fixed_4_t a, fixed_4_t b) { fixed_4_t r; r.simd = wasm_i32x4_add(a.simd, b.simd); return r; }
static inline fixed_4_t fixed4_sub_impl_(fixed_4_t a, fixed_4_t b) { fixed_4_t r; r.simd = wasm_i32x4_sub(a.simd, b.simd); return r; }
static inline fixed_4_t fixed4_neg_impl_(fixed_4_t a)              { fixed_4_t r; r.simd = wasm_i32x4_neg(a.simd); return r; }

/* mul: widen 32x32 -> 64 via extmul_low/high, logical >> FIXED_SHIFT, narrow.
   wasm_i32x4_shuffle picks the low int32 of each i64 lane (lanes 0,2,4,6). */
static inline fixed_4_t fixed4_mul_impl_(fixed_4_t a, fixed_4_t b) {
    v128_t lo64 = wasm_i64x2_extmul_low_i32x4(a.simd, b.simd);
    v128_t hi64 = wasm_i64x2_extmul_high_i32x4(a.simd, b.simd);
    lo64 = wasm_u64x2_shr(lo64, FIXED_SHIFT);
    hi64 = wasm_u64x2_shr(hi64, FIXED_SHIFT);
    fixed_4_t r; r.simd = wasm_i32x4_shuffle(lo64, hi64, 0, 2, 4, 6); return r;
}

static inline fixed_4_t fixed4_div_impl_(fixed_4_t a, fixed_4_t b) {
    fixed_4_t r;
    for (int i = 0; i < 4; ++i) r.e[i] = fixed_div(a.e[i], b.e[i]);
    return r;
}

/* WASM SIMD bitmask returns lanes packed into low bits of a 32-bit int. */
static inline uint8_t fixed4_eq_impl_(fixed_4_t a, fixed_4_t b) { return (uint8_t)wasm_i32x4_bitmask(wasm_i32x4_eq(a.simd, b.simd)); }
static inline uint8_t fixed4_gt_impl_(fixed_4_t a, fixed_4_t b) { return (uint8_t)wasm_i32x4_bitmask(wasm_i32x4_gt(a.simd, b.simd)); }
static inline uint8_t fixed4_lt_impl_(fixed_4_t a, fixed_4_t b) { return (uint8_t)wasm_i32x4_bitmask(wasm_i32x4_lt(a.simd, b.simd)); }
static inline uint8_t fixed4_ge_impl_(fixed_4_t a, fixed_4_t b) { return (uint8_t)wasm_i32x4_bitmask(wasm_i32x4_ge(a.simd, b.simd)); }
static inline uint8_t fixed4_le_impl_(fixed_4_t a, fixed_4_t b) { return (uint8_t)wasm_i32x4_bitmask(wasm_i32x4_le(a.simd, b.simd)); }

#else /* FIXED_BACKEND_SCALAR */

typedef union {
    fixed_t e[4];
    struct { fixed_t x, y, z, w; };
} fixed_4_t;

static inline fixed_4_t fixed4_zero(void)                       { fixed_4_t r = {{0,0,0,0}}; return r; }
static inline fixed_4_t fixed4_set1(fixed_t x)                  { fixed_4_t r = {{x,x,x,x}}; return r; }
static inline fixed_4_t fixed4_set(fixed_t a, fixed_t b, fixed_t c, fixed_t d) { fixed_4_t r = {{a,b,c,d}}; return r; }
static inline fixed_4_t fixed4_load(const fixed_t* p)           { fixed_4_t r; memcpy(r.e, p, 4 * sizeof(fixed_t)); return r; }
static inline fixed_4_t fixed4_load_aligned(const fixed_t* p)   { fixed_4_t r; memcpy(r.e, p, 4 * sizeof(fixed_t)); return r; }
static inline void      fixed4_store(fixed_t* p, fixed_4_t v)        { memcpy(p, v.e, 4 * sizeof(fixed_t)); }
static inline void      fixed4_store_aligned(fixed_t* p, fixed_4_t v){ memcpy(p, v.e, 4 * sizeof(fixed_t)); }

#define FIXED4_LANEWISE_(OP) do { fixed_4_t r; for (int i = 0; i < 4; ++i) r.e[i] = (OP); return r; } while (0)

static inline fixed_4_t fixed4_add_impl_(fixed_4_t a, fixed_4_t b) { FIXED4_LANEWISE_((fixed_t)((uint32_t)a.e[i] + (uint32_t)b.e[i])); }
static inline fixed_4_t fixed4_sub_impl_(fixed_4_t a, fixed_4_t b) { FIXED4_LANEWISE_((fixed_t)((uint32_t)a.e[i] - (uint32_t)b.e[i])); }
static inline fixed_4_t fixed4_neg_impl_(fixed_4_t a)              { FIXED4_LANEWISE_((fixed_t)(0u - (uint32_t)a.e[i])); }
static inline fixed_4_t fixed4_mul_impl_(fixed_4_t a, fixed_4_t b) { FIXED4_LANEWISE_(fixed_mul(a.e[i], b.e[i])); }
static inline fixed_4_t fixed4_div_impl_(fixed_4_t a, fixed_4_t b) { FIXED4_LANEWISE_(fixed_div(a.e[i], b.e[i])); }

#define FIXED4_MASK_(OP) do { uint8_t m = 0; for (int i = 0; i < 4; ++i) if (OP) m |= (uint8_t)(1u << i); return m; } while (0)

static inline uint8_t fixed4_eq_impl_(fixed_4_t a, fixed_4_t b)  { FIXED4_MASK_(a.e[i] == b.e[i]); }
static inline uint8_t fixed4_gt_impl_(fixed_4_t a, fixed_4_t b)  { FIXED4_MASK_(a.e[i] >  b.e[i]); }
static inline uint8_t fixed4_lt_impl_(fixed_4_t a, fixed_4_t b)  { FIXED4_MASK_(a.e[i] <  b.e[i]); }
static inline uint8_t fixed4_ge_impl_(fixed_4_t a, fixed_4_t b)  { FIXED4_MASK_(a.e[i] >= b.e[i]); }
static inline uint8_t fixed4_le_impl_(fixed_4_t a, fixed_4_t b)  { FIXED4_MASK_(a.e[i] <= b.e[i]); }

#undef FIXED4_MASK_
#undef FIXED4_LANEWISE_

#endif

/* ---------- 4-wide verified public wrappers ----------
   Debug: run SIMD op, then assert lane-by-lane match against scalar reference.
   Catches backend regressions and intrinsic surprises early. NDEBUG: direct
   call, zero overhead — compiler inlines through. */

#ifndef NDEBUG
#  define FIXED4_VERIFY_BIN_(name, ref_expr) \
    static inline fixed_4_t fixed4_##name(fixed_4_t a, fixed_4_t b) { \
        fixed_4_t r_ = fixed4_##name##_impl_(a, b); \
        fixed_t ax_[4], bx_[4], rx_[4]; \
        fixed4_store(ax_, a); fixed4_store(bx_, b); fixed4_store(rx_, r_); \
        for (int i_ = 0; i_ < 4; ++i_) { \
            fixed_t exp_ = (fixed_t)(ref_expr); \
            assert(rx_[i_] == exp_ && "fixed4_" #name " SIMD/scalar lane mismatch"); \
        } \
        return r_; \
    }
#  define FIXED4_VERIFY_UN_(name, ref_expr) \
    static inline fixed_4_t fixed4_##name(fixed_4_t a) { \
        fixed_4_t r_ = fixed4_##name##_impl_(a); \
        fixed_t ax_[4], rx_[4]; \
        fixed4_store(ax_, a); fixed4_store(rx_, r_); \
        for (int i_ = 0; i_ < 4; ++i_) { \
            fixed_t exp_ = (fixed_t)(ref_expr); \
            assert(rx_[i_] == exp_ && "fixed4_" #name " SIMD/scalar lane mismatch"); \
        } \
        return r_; \
    }
#  define FIXED4_VERIFY_CMP_(name, ref_expr) \
    static inline uint8_t fixed4_##name(fixed_4_t a, fixed_4_t b) { \
        uint8_t r_ = fixed4_##name##_impl_(a, b); \
        fixed_t ax_[4], bx_[4]; \
        fixed4_store(ax_, a); fixed4_store(bx_, b); \
        uint8_t exp_ = 0; \
        for (int i_ = 0; i_ < 4; ++i_) if (ref_expr) exp_ |= (uint8_t)(1u << i_); \
        assert(r_ == exp_ && "fixed4_" #name " SIMD/scalar mask mismatch"); \
        return r_; \
    }
#else
#  define FIXED4_VERIFY_BIN_(name, ref_expr) \
    static inline fixed_4_t fixed4_##name(fixed_4_t a, fixed_4_t b) { return fixed4_##name##_impl_(a, b); }
#  define FIXED4_VERIFY_UN_(name, ref_expr) \
    static inline fixed_4_t fixed4_##name(fixed_4_t a) { return fixed4_##name##_impl_(a); }
#  define FIXED4_VERIFY_CMP_(name, ref_expr) \
    static inline uint8_t fixed4_##name(fixed_4_t a, fixed_4_t b) { return fixed4_##name##_impl_(a, b); }
#endif

FIXED4_VERIFY_BIN_(add, (uint32_t)ax_[i_] + (uint32_t)bx_[i_])
FIXED4_VERIFY_BIN_(sub, (uint32_t)ax_[i_] - (uint32_t)bx_[i_])
FIXED4_VERIFY_UN_ (neg, 0u - (uint32_t)ax_[i_])
FIXED4_VERIFY_BIN_(mul, fixed_mul(ax_[i_], bx_[i_]))
FIXED4_VERIFY_BIN_(div, fixed_div(ax_[i_], bx_[i_]))

FIXED4_VERIFY_CMP_(eq, ax_[i_] == bx_[i_])
FIXED4_VERIFY_CMP_(gt, ax_[i_] >  bx_[i_])
FIXED4_VERIFY_CMP_(lt, ax_[i_] <  bx_[i_])
FIXED4_VERIFY_CMP_(ge, ax_[i_] >= bx_[i_])
FIXED4_VERIFY_CMP_(le, ax_[i_] <= bx_[i_])

#undef FIXED4_VERIFY_BIN_
#undef FIXED4_VERIFY_UN_
#undef FIXED4_VERIFY_CMP_

/* ============================================================
   8-wide SIMD: 8 packed int32 fixed_t.
   ============================================================ */

#if FIXED_BACKEND_X86

/* Self-bundled SIMD value: `.simd` is the typed __m256i for intrinsic
   plumbing; `.e[8]` is the lane-indexed view callers use to read/write a
   single slot. Same 32 bytes, same alignment (inherited from __m256i). */
typedef union {
    __m256i simd;
    fixed_t e[8];
} fixed_8_t;

static inline fixed_8_t fixed8_zero(void)                            { fixed_8_t r; r.simd = _mm256_setzero_si256(); return r; }
static inline fixed_8_t fixed8_set1(fixed_t x)                       { fixed_8_t r; r.simd = _mm256_set1_epi32(x); return r; }
static inline fixed_8_t fixed8_load(const fixed_t* p)                { fixed_8_t r; r.simd = _mm256_loadu_si256((const __m256i*)p); return r; }
static inline fixed_8_t fixed8_load_aligned(const fixed_t* p)        { fixed_8_t r; r.simd = _mm256_load_si256((const __m256i*)p); return r; }
static inline void      fixed8_store(fixed_t* p, fixed_8_t v)        { _mm256_storeu_si256((__m256i*)p, v.simd); }
static inline void      fixed8_store_aligned(fixed_t* p, fixed_8_t v){ _mm256_store_si256((__m256i*)p, v.simd); }

static inline fixed_8_t fixed8_add_impl_(fixed_8_t a, fixed_8_t b) { fixed_8_t r; r.simd = _mm256_add_epi32(a.simd, b.simd); return r; }
static inline fixed_8_t fixed8_sub_impl_(fixed_8_t a, fixed_8_t b) { fixed_8_t r; r.simd = _mm256_sub_epi32(a.simd, b.simd); return r; }
static inline fixed_8_t fixed8_neg_impl_(fixed_8_t a)              { fixed_8_t r; r.simd = _mm256_sub_epi32(_mm256_setzero_si256(), a.simd); return r; }

/* Same trick as fixed4_mul, applied per 128-bit lane (AVX2 shuffle/blend are lane-local). */
static inline fixed_8_t fixed8_mul_impl_(fixed_8_t a, fixed_8_t b) {
    __m256i ae  = _mm256_mul_epi32(a.simd, b.simd);
    __m256i ash = _mm256_shuffle_epi32(a.simd, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i bsh = _mm256_shuffle_epi32(b.simd, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i ao  = _mm256_mul_epi32(ash, bsh);
    ae = _mm256_srli_epi64(ae, FIXED_SHIFT);
    ao = _mm256_srli_epi64(ao, FIXED_SHIFT);
    __m256i ao_shifted = _mm256_shuffle_epi32(ao, _MM_SHUFFLE(2, 2, 0, 0));
    fixed_8_t r; r.simd = _mm256_blend_epi16(ae, ao_shifted, 0xCC); return r;
}

static inline fixed_8_t fixed8_div_impl_(fixed_8_t a, fixed_8_t b) {
    fixed_8_t r;
    for (int i = 0; i < 8; ++i) r.e[i] = fixed_div(a.e[i], b.e[i]);
    return r;
}

/* Compare ops return an 8-bit mask: bit i set iff lane i true. */
static inline uint8_t fixed8_eq_impl_(fixed_8_t a, fixed_8_t b) { return (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(a.simd, b.simd))); }
static inline uint8_t fixed8_gt_impl_(fixed_8_t a, fixed_8_t b) { return (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(a.simd, b.simd))); }
static inline uint8_t fixed8_lt_impl_(fixed_8_t a, fixed_8_t b) { return (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(b.simd, a.simd))); }
static inline uint8_t fixed8_ge_impl_(fixed_8_t a, fixed_8_t b) { return (uint8_t)~_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(b.simd, a.simd))); }
static inline uint8_t fixed8_le_impl_(fixed_8_t a, fixed_8_t b) { return (uint8_t)~_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(a.simd, b.simd))); }

/* Horizontal min/max: collapse all 8 lanes to a scalar. Reduction ladder
   8 -> 4 -> 2 -> 1: AVX2 split into two 128-bit halves, SSE4.1 _mm_min_epi32
   pair-min, then byte-shift the 128-bit reg to feed lower lanes against
   upper. _mm_srli_si128 by 8 / 4 bytes equals 2 / 1 int32-lane shifts. */
static inline fixed_t fixed8_min_impl_(fixed_8_t v) {
    __m128i lo = _mm256_castsi256_si128(v.simd);
    __m128i hi = _mm256_extracti128_si256(v.simd, 1);
    __m128i m  = _mm_min_epi32(lo, hi);
    m = _mm_min_epi32(m, _mm_srli_si128(m, 8));
    m = _mm_min_epi32(m, _mm_srli_si128(m, 4));
    return (fixed_t)_mm_cvtsi128_si32(m);
}
static inline fixed_t fixed8_max_impl_(fixed_8_t v) {
    __m128i lo = _mm256_castsi256_si128(v.simd);
    __m128i hi = _mm256_extracti128_si256(v.simd, 1);
    __m128i m  = _mm_max_epi32(lo, hi);
    m = _mm_max_epi32(m, _mm_srli_si128(m, 8));
    m = _mm_max_epi32(m, _mm_srli_si128(m, 4));
    return (fixed_t)_mm_cvtsi128_si32(m);
}

#elif FIXED_BACKEND_NEON

/* NEON is 128-bit; 8-wide is two stacked fixed_4_t. The anonymous struct
   keeps `a.lo` / `a.hi` working in the impl_ functions while the union
   also exposes a flat `.e[8]` lane view for the public API. */
typedef union {
    struct { fixed_4_t lo, hi; };
    fixed_t e[8];
} fixed_8_t;

static inline fixed_8_t fixed8_zero(void)                            { fixed_8_t r; r.lo = fixed4_zero(); r.hi = fixed4_zero(); return r; }
static inline fixed_8_t fixed8_set1(fixed_t x)                       { fixed_8_t r; r.lo = fixed4_set1(x); r.hi = fixed4_set1(x); return r; }
static inline fixed_8_t fixed8_load(const fixed_t* p)                { fixed_8_t r; r.lo = fixed4_load(p); r.hi = fixed4_load(p + 4); return r; }
static inline fixed_8_t fixed8_load_aligned(const fixed_t* p)        { fixed_8_t r; r.lo = fixed4_load_aligned(p); r.hi = fixed4_load_aligned(p + 4); return r; }
static inline void      fixed8_store(fixed_t* p, fixed_8_t v)        { fixed4_store(p, v.lo); fixed4_store(p + 4, v.hi); }
static inline void      fixed8_store_aligned(fixed_t* p, fixed_8_t v){ fixed4_store_aligned(p, v.lo); fixed4_store_aligned(p + 4, v.hi); }

#define FIXED8_PAIR_(EXPR_LO, EXPR_HI) do { fixed_8_t r; r.lo = (EXPR_LO); r.hi = (EXPR_HI); return r; } while (0)

/* fixed8 impls call fixed4 impls (not verified wrappers) so the assertion runs
   once at the fixed8 level — avoids redundant scalar reference work. */
static inline fixed_8_t fixed8_add_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_PAIR_(fixed4_add_impl_(a.lo, b.lo), fixed4_add_impl_(a.hi, b.hi)); }
static inline fixed_8_t fixed8_sub_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_PAIR_(fixed4_sub_impl_(a.lo, b.lo), fixed4_sub_impl_(a.hi, b.hi)); }
static inline fixed_8_t fixed8_neg_impl_(fixed_8_t a)              { FIXED8_PAIR_(fixed4_neg_impl_(a.lo), fixed4_neg_impl_(a.hi)); }
static inline fixed_8_t fixed8_mul_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_PAIR_(fixed4_mul_impl_(a.lo, b.lo), fixed4_mul_impl_(a.hi, b.hi)); }
static inline fixed_8_t fixed8_div_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_PAIR_(fixed4_div_impl_(a.lo, b.lo), fixed4_div_impl_(a.hi, b.hi)); }

/* Compare ops return an 8-bit mask: bit i set iff lane i true. */
static inline uint8_t fixed8_eq_impl_(fixed_8_t a, fixed_8_t b)  { return (uint8_t)(fixed4_eq_impl_(a.lo, b.lo) | (fixed4_eq_impl_(a.hi, b.hi) << 4)); }
static inline uint8_t fixed8_gt_impl_(fixed_8_t a, fixed_8_t b)  { return (uint8_t)(fixed4_gt_impl_(a.lo, b.lo) | (fixed4_gt_impl_(a.hi, b.hi) << 4)); }
static inline uint8_t fixed8_lt_impl_(fixed_8_t a, fixed_8_t b)  { return (uint8_t)(fixed4_lt_impl_(a.lo, b.lo) | (fixed4_lt_impl_(a.hi, b.hi) << 4)); }
static inline uint8_t fixed8_ge_impl_(fixed_8_t a, fixed_8_t b)  { return (uint8_t)(fixed4_ge_impl_(a.lo, b.lo) | (fixed4_ge_impl_(a.hi, b.hi) << 4)); }
static inline uint8_t fixed8_le_impl_(fixed_8_t a, fixed_8_t b)  { return (uint8_t)(fixed4_le_impl_(a.lo, b.lo) | (fixed4_le_impl_(a.hi, b.hi) << 4)); }

/* Horizontal min/max via NEON pairwise reduction: 8 -> 4 -> 2 -> 1. */
static inline fixed_t fixed8_min_impl_(fixed_8_t v) {
    int32x4_t m  = vminq_s32(v.lo.simd, v.hi.simd);
    int32x2_t m2 = vmin_s32(vget_low_s32(m), vget_high_s32(m));
    m2 = vpmin_s32(m2, m2);
    return (fixed_t)vget_lane_s32(m2, 0);
}
static inline fixed_t fixed8_max_impl_(fixed_8_t v) {
    int32x4_t m  = vmaxq_s32(v.lo.simd, v.hi.simd);
    int32x2_t m2 = vmax_s32(vget_low_s32(m), vget_high_s32(m));
    m2 = vpmax_s32(m2, m2);
    return (fixed_t)vget_lane_s32(m2, 0);
}

#undef FIXED8_PAIR_

#elif FIXED_BACKEND_WASM

/* WASM SIMD is 128-bit; 8-wide is two stacked fixed_4_t. */
typedef union {
    struct { fixed_4_t lo, hi; };
    fixed_t e[8];
} fixed_8_t;

static inline fixed_8_t fixed8_zero(void)                            { fixed_8_t r; r.lo = fixed4_zero(); r.hi = fixed4_zero(); return r; }
static inline fixed_8_t fixed8_set1(fixed_t x)                       { fixed_8_t r; r.lo = fixed4_set1(x); r.hi = fixed4_set1(x); return r; }
static inline fixed_8_t fixed8_load(const fixed_t* p)                { fixed_8_t r; r.lo = fixed4_load(p); r.hi = fixed4_load(p + 4); return r; }
static inline fixed_8_t fixed8_load_aligned(const fixed_t* p)        { fixed_8_t r; r.lo = fixed4_load_aligned(p); r.hi = fixed4_load_aligned(p + 4); return r; }
static inline void      fixed8_store(fixed_t* p, fixed_8_t v)        { fixed4_store(p, v.lo); fixed4_store(p + 4, v.hi); }
static inline void      fixed8_store_aligned(fixed_t* p, fixed_8_t v){ fixed4_store_aligned(p, v.lo); fixed4_store_aligned(p + 4, v.hi); }

#define FIXED8_PAIR_(EXPR_LO, EXPR_HI) do { fixed_8_t r; r.lo = (EXPR_LO); r.hi = (EXPR_HI); return r; } while (0)

static inline fixed_8_t fixed8_add_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_PAIR_(fixed4_add_impl_(a.lo, b.lo), fixed4_add_impl_(a.hi, b.hi)); }
static inline fixed_8_t fixed8_sub_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_PAIR_(fixed4_sub_impl_(a.lo, b.lo), fixed4_sub_impl_(a.hi, b.hi)); }
static inline fixed_8_t fixed8_neg_impl_(fixed_8_t a)              { FIXED8_PAIR_(fixed4_neg_impl_(a.lo), fixed4_neg_impl_(a.hi)); }
static inline fixed_8_t fixed8_mul_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_PAIR_(fixed4_mul_impl_(a.lo, b.lo), fixed4_mul_impl_(a.hi, b.hi)); }
static inline fixed_8_t fixed8_div_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_PAIR_(fixed4_div_impl_(a.lo, b.lo), fixed4_div_impl_(a.hi, b.hi)); }

static inline uint8_t fixed8_eq_impl_(fixed_8_t a, fixed_8_t b)  { return (uint8_t)(fixed4_eq_impl_(a.lo, b.lo) | (fixed4_eq_impl_(a.hi, b.hi) << 4)); }
static inline uint8_t fixed8_gt_impl_(fixed_8_t a, fixed_8_t b)  { return (uint8_t)(fixed4_gt_impl_(a.lo, b.lo) | (fixed4_gt_impl_(a.hi, b.hi) << 4)); }
static inline uint8_t fixed8_lt_impl_(fixed_8_t a, fixed_8_t b)  { return (uint8_t)(fixed4_lt_impl_(a.lo, b.lo) | (fixed4_lt_impl_(a.hi, b.hi) << 4)); }
static inline uint8_t fixed8_ge_impl_(fixed_8_t a, fixed_8_t b)  { return (uint8_t)(fixed4_ge_impl_(a.lo, b.lo) | (fixed4_ge_impl_(a.hi, b.hi) << 4)); }
static inline uint8_t fixed8_le_impl_(fixed_8_t a, fixed_8_t b)  { return (uint8_t)(fixed4_le_impl_(a.lo, b.lo) | (fixed4_le_impl_(a.hi, b.hi) << 4)); }

/* Horizontal reduction: pairwise WASM SIMD min/max, then collapse via
   shuffle. wasm_i32x4_min_s / max_s give pairwise lane ops; reduction
   ladder 8 -> 4 -> 2 -> 1. */
static inline fixed_t fixed8_min_impl_(fixed_8_t v) {
    v128_t m = wasm_i32x4_min(v.lo.simd, v.hi.simd);
    v128_t s = wasm_i32x4_shuffle(m, m, 2, 3, 2, 3);
    m = wasm_i32x4_min(m, s);
    s = wasm_i32x4_shuffle(m, m, 1, 1, 1, 1);
    m = wasm_i32x4_min(m, s);
    return (fixed_t)wasm_i32x4_extract_lane(m, 0);
}
static inline fixed_t fixed8_max_impl_(fixed_8_t v) {
    v128_t m = wasm_i32x4_max(v.lo.simd, v.hi.simd);
    v128_t s = wasm_i32x4_shuffle(m, m, 2, 3, 2, 3);
    m = wasm_i32x4_max(m, s);
    s = wasm_i32x4_shuffle(m, m, 1, 1, 1, 1);
    m = wasm_i32x4_max(m, s);
    return (fixed_t)wasm_i32x4_extract_lane(m, 0);
}

#undef FIXED8_PAIR_

#else /* FIXED_BACKEND_SCALAR */

typedef struct { fixed_t e[8]; } fixed_8_t;

static inline fixed_8_t fixed8_zero(void)                       { fixed_8_t r = {{0}}; return r; }
static inline fixed_8_t fixed8_set1(fixed_t x)                  { fixed_8_t r = {{x,x,x,x,x,x,x,x}}; return r; }
static inline fixed_8_t fixed8_load(const fixed_t* p)           { fixed_8_t r; memcpy(r.e, p, 8 * sizeof(fixed_t)); return r; }
static inline fixed_8_t fixed8_load_aligned(const fixed_t* p)   { fixed_8_t r; memcpy(r.e, p, 8 * sizeof(fixed_t)); return r; }
static inline void      fixed8_store(fixed_t* p, fixed_8_t v)        { memcpy(p, v.e, 8 * sizeof(fixed_t)); }
static inline void      fixed8_store_aligned(fixed_t* p, fixed_8_t v){ memcpy(p, v.e, 8 * sizeof(fixed_t)); }

#define FIXED8_LANEWISE_(OP) do { fixed_8_t r; for (int i = 0; i < 8; ++i) r.e[i] = (OP); return r; } while (0)

static inline fixed_8_t fixed8_add_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_LANEWISE_((fixed_t)((uint32_t)a.e[i] + (uint32_t)b.e[i])); }
static inline fixed_8_t fixed8_sub_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_LANEWISE_((fixed_t)((uint32_t)a.e[i] - (uint32_t)b.e[i])); }
static inline fixed_8_t fixed8_neg_impl_(fixed_8_t a)              { FIXED8_LANEWISE_((fixed_t)(0u - (uint32_t)a.e[i])); }
static inline fixed_8_t fixed8_mul_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_LANEWISE_(fixed_mul(a.e[i], b.e[i])); }
static inline fixed_8_t fixed8_div_impl_(fixed_8_t a, fixed_8_t b) { FIXED8_LANEWISE_(fixed_div(a.e[i], b.e[i])); }

#define FIXED8_MASK_(OP) do { uint8_t m = 0; for (int i = 0; i < 8; ++i) if (OP) m |= (uint8_t)(1u << i); return m; } while (0)

static inline uint8_t fixed8_eq_impl_(fixed_8_t a, fixed_8_t b)  { FIXED8_MASK_(a.e[i] == b.e[i]); }
static inline uint8_t fixed8_gt_impl_(fixed_8_t a, fixed_8_t b)  { FIXED8_MASK_(a.e[i] >  b.e[i]); }
static inline uint8_t fixed8_lt_impl_(fixed_8_t a, fixed_8_t b)  { FIXED8_MASK_(a.e[i] <  b.e[i]); }
static inline uint8_t fixed8_ge_impl_(fixed_8_t a, fixed_8_t b)  { FIXED8_MASK_(a.e[i] >= b.e[i]); }
static inline uint8_t fixed8_le_impl_(fixed_8_t a, fixed_8_t b)  { FIXED8_MASK_(a.e[i] <= b.e[i]); }

static inline fixed_t fixed8_min_impl_(fixed_8_t v) {
    fixed_t r = v.e[0];
    for (int i = 1; i < 8; i++) r = fixed_min(r, v.e[i]);
    return r;
}
static inline fixed_t fixed8_max_impl_(fixed_8_t v) {
    fixed_t r = v.e[0];
    for (int i = 1; i < 8; i++) r = fixed_max(r, v.e[i]);
    return r;
}

#undef FIXED8_MASK_
#undef FIXED8_LANEWISE_

#endif

/* ---------- 8-wide verified public wrappers ---------- */

#ifndef NDEBUG
#  define FIXED8_VERIFY_BIN_(name, ref_expr) \
    static inline fixed_8_t fixed8_##name(fixed_8_t a, fixed_8_t b) { \
        fixed_8_t r_ = fixed8_##name##_impl_(a, b); \
        fixed_t ax_[8], bx_[8], rx_[8]; \
        fixed8_store(ax_, a); fixed8_store(bx_, b); fixed8_store(rx_, r_); \
        for (int i_ = 0; i_ < 8; ++i_) { \
            fixed_t exp_ = (fixed_t)(ref_expr); \
            assert(rx_[i_] == exp_ && "fixed8_" #name " SIMD/scalar lane mismatch"); \
        } \
        return r_; \
    }
#  define FIXED8_VERIFY_UN_(name, ref_expr) \
    static inline fixed_8_t fixed8_##name(fixed_8_t a) { \
        fixed_8_t r_ = fixed8_##name##_impl_(a); \
        fixed_t ax_[8], rx_[8]; \
        fixed8_store(ax_, a); fixed8_store(rx_, r_); \
        for (int i_ = 0; i_ < 8; ++i_) { \
            fixed_t exp_ = (fixed_t)(ref_expr); \
            assert(rx_[i_] == exp_ && "fixed8_" #name " SIMD/scalar lane mismatch"); \
        } \
        return r_; \
    }
#  define FIXED8_VERIFY_CMP_(name, ref_expr) \
    static inline uint8_t fixed8_##name(fixed_8_t a, fixed_8_t b) { \
        uint8_t r_ = fixed8_##name##_impl_(a, b); \
        fixed_t ax_[8], bx_[8]; \
        fixed8_store(ax_, a); fixed8_store(bx_, b); \
        uint8_t exp_ = 0; \
        for (int i_ = 0; i_ < 8; ++i_) if (ref_expr) exp_ |= (uint8_t)(1u << i_); \
        assert(r_ == exp_ && "fixed8_" #name " SIMD/scalar mask mismatch"); \
        return r_; \
    }
#else
#  define FIXED8_VERIFY_BIN_(name, ref_expr) \
    static inline fixed_8_t fixed8_##name(fixed_8_t a, fixed_8_t b) { return fixed8_##name##_impl_(a, b); }
#  define FIXED8_VERIFY_UN_(name, ref_expr) \
    static inline fixed_8_t fixed8_##name(fixed_8_t a) { return fixed8_##name##_impl_(a); }
#  define FIXED8_VERIFY_CMP_(name, ref_expr) \
    static inline uint8_t fixed8_##name(fixed_8_t a, fixed_8_t b) { return fixed8_##name##_impl_(a, b); }
#endif

FIXED8_VERIFY_BIN_(add, (uint32_t)ax_[i_] + (uint32_t)bx_[i_])
FIXED8_VERIFY_BIN_(sub, (uint32_t)ax_[i_] - (uint32_t)bx_[i_])
FIXED8_VERIFY_UN_ (neg, 0u - (uint32_t)ax_[i_])
FIXED8_VERIFY_BIN_(mul, fixed_mul(ax_[i_], bx_[i_]))
FIXED8_VERIFY_BIN_(div, fixed_div(ax_[i_], bx_[i_]))

FIXED8_VERIFY_CMP_(eq, ax_[i_] == bx_[i_])
FIXED8_VERIFY_CMP_(gt, ax_[i_] >  bx_[i_])
FIXED8_VERIFY_CMP_(lt, ax_[i_] <  bx_[i_])
FIXED8_VERIFY_CMP_(ge, ax_[i_] >= bx_[i_])
FIXED8_VERIFY_CMP_(le, ax_[i_] <= bx_[i_])

/* Horizontal reductions: fixed_8_t -> fixed_t. Verified against scalar
   chain: r0 = v.e[0]; r_{i+1} = OP(r_i, v.e[i+1]). */
#ifndef NDEBUG
#  define FIXED8_VERIFY_RED_(name, ref_step) \
    static inline fixed_t fixed8_##name(fixed_8_t a) { \
        fixed_t r_ = fixed8_##name##_impl_(a); \
        fixed_t ax_[8]; \
        fixed8_store(ax_, a); \
        fixed_t exp_ = ax_[0]; \
        for (int i_ = 1; i_ < 8; ++i_) exp_ = (ref_step); \
        assert(r_ == exp_ && "fixed8_" #name " SIMD/scalar reduction mismatch"); \
        return r_; \
    }
#else
#  define FIXED8_VERIFY_RED_(name, ref_step) \
    static inline fixed_t fixed8_##name(fixed_8_t a) { return fixed8_##name##_impl_(a); }
#endif

FIXED8_VERIFY_RED_(min, fixed_min(exp_, ax_[i_]))
FIXED8_VERIFY_RED_(max, fixed_max(exp_, ax_[i_]))

#undef FIXED8_VERIFY_BIN_
#undef FIXED8_VERIFY_UN_
#undef FIXED8_VERIFY_CMP_
#undef FIXED8_VERIFY_RED_
