#pragma once

#include "fixed.h"

/* ============================================================
   bitops + branch hints
   ============================================================ */

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

/* Q16.16 vec3 / quaternion math. Storage is fixed_4_t (16 bytes) so SIMD
   ops on add/sub/scale/neg are direct; scalar paths are used where the
   horizontal extraction would cost more than three plain imuls. */

/* ============================================================
   Scalar helpers: sqrt, rsqrt, sin, cos
   ============================================================ */

#define FIXED_PI       ((fixed_t)205887)   /* pi    in Q16.16 */
#define FIXED_TWO_PI   ((fixed_t)411775)   /* 2*pi  in Q16.16 */
#define FIXED_HALF_PI  ((fixed_t)102944)   /* pi/2  in Q16.16 */

/* Digit-by-digit unsigned 64-bit isqrt. Branch-on-bit, no division. */
static inline uint32_t fixed_isqrt64_(uint64_t v) {
    uint64_t r = 0;
    uint64_t bit = (uint64_t)1 << 62;
    while (bit > v) bit >>= 2;
    while (bit) {
        if (v >= r + bit) { v -= r + bit; r = (r >> 1) + bit; }
        else              { r >>= 1; }
        bit >>= 2;
    }
    return (uint32_t)r;
}

/* sqrt on Q16.16. Negative inputs clamp to 0. */
static inline fixed_t fixed_sqrt(fixed_t x) {
    if (x <= 0) return 0;
    /* sqrt(x_real) * 2^16 = sqrt(x * 2^16). Cast through uint32 first to
       avoid signed left-shift UB; we already know x > 0. */
    return (fixed_t)fixed_isqrt64_((uint64_t)(uint32_t)x << FIXED_SHIFT);
}

static inline fixed_t fixed_rsqrt(fixed_t x) {
    fixed_t s = fixed_sqrt(x);
    return s > 0 ? fixed_div(FIXED_ONE, s) : 0;
}

/* sin via range reduction to [-pi/2, pi/2] then degree-5 minimax-style
   polynomial:  sin(z) ≈ z - z^3/6 + z^5/120.  Max abs error ~5e-3 on the
   reduced range, plenty for game-grade Q16.16. */
static inline fixed_t fixed_sin(fixed_t x) {
    int64_t v = (int64_t)x % (int64_t)FIXED_TWO_PI;
    if (v >  FIXED_PI)      v -= FIXED_TWO_PI;
    else if (v < -FIXED_PI) v += FIXED_TWO_PI;
    if (v >  FIXED_HALF_PI)      v =  FIXED_PI - v;
    else if (v < -FIXED_HALF_PI) v = -FIXED_PI - v;

    fixed_t z   = (fixed_t)v;
    fixed_t z2  = fixed_mul(z, z);
    fixed_t inv6   = (fixed_t)10923;   /* 1/6   in Q16.16 */
    fixed_t inv120 = (fixed_t)546;     /* 1/120 in Q16.16 */
    fixed_t t = fixed_sub(inv6, fixed_mul(z2, inv120));
    fixed_t s = fixed_sub(FIXED_ONE, fixed_mul(z2, t));
    return fixed_mul(z, s);
}

static inline fixed_t fixed_cos(fixed_t x) {
    return fixed_sin(fixed_add(x, FIXED_HALF_PI));
}

/* ============================================================
   vec3
   ============================================================ */

/* Anonymous-struct-in-union (C11). simd lane 0..2 = x,y,z; lane 3 is
   padding kept at 0 by every constructor / op. */
typedef union {
    fixed_4_t simd;
    struct { fixed_t x, y, z, _pad; };
    fixed_t e[4];
} vec3_t;

static inline vec3_t vec3_make(fixed_t x, fixed_t y, fixed_t z) {
    vec3_t r; r.simd = fixed4_set(x, y, z, 0); return r;
}
static inline vec3_t vec3_zero(void) {
    vec3_t r; r.simd = fixed4_zero(); return r;
}
static inline vec3_t vec3_load(const fixed_t* p) {
    return vec3_make(p[0], p[1], p[2]);
}
static inline void vec3_store(fixed_t* p, vec3_t v) {
    p[0] = v.x; p[1] = v.y; p[2] = v.z;
}

static inline vec3_t vec3_add(vec3_t a, vec3_t b) {
    vec3_t r; r.simd = fixed4_add(a.simd, b.simd); return r;
}
static inline int vec3_eq(vec3_t a, vec3_t b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
static inline vec3_t vec3_sub(vec3_t a, vec3_t b) {
    vec3_t r; r.simd = fixed4_sub(a.simd, b.simd); return r;
}
static inline vec3_t vec3_neg(vec3_t a) {
    vec3_t r; r.simd = fixed4_neg(a.simd); return r;
}
static inline vec3_t vec3_scale(vec3_t a, fixed_t s) {
    vec3_t r; r.simd = fixed4_mul(a.simd, fixed4_set1(s)); return r;
}
static inline vec3_t vec3_hadamard(vec3_t a, vec3_t b) {
    vec3_t r; r.simd = fixed4_mul(a.simd, b.simd); return r;
}

/* Dot/cross are 3-lane ops; scalar imul + iadd is faster than SIMD mul +
   horizontal extract on Q16.16 because fixed4_mul itself costs ~4 uops. */
static inline fixed_t vec3_dot(vec3_t a, vec3_t b) {
    return fixed_add(fixed_add(fixed_mul(a.x, b.x), fixed_mul(a.y, b.y)),
                     fixed_mul(a.z, b.z));
}
static inline vec3_t vec3_cross(vec3_t a, vec3_t b) {
    return vec3_make(
        fixed_sub(fixed_mul(a.y, b.z), fixed_mul(a.z, b.y)),
        fixed_sub(fixed_mul(a.z, b.x), fixed_mul(a.x, b.z)),
        fixed_sub(fixed_mul(a.x, b.y), fixed_mul(a.y, b.x))
    );
}

static inline fixed_t vec3_length_sq(vec3_t a) { return vec3_dot(a, a); }
static inline fixed_t vec3_length(vec3_t a)    { return fixed_sqrt(vec3_length_sq(a)); }

static inline vec3_t vec3_normalize(vec3_t a) {
    fixed_t r = fixed_rsqrt(vec3_length_sq(a));
    return vec3_scale(a, r);
}
static inline vec3_t vec3_lerp(vec3_t a, vec3_t b, fixed_t t) {
    return vec3_add(a, vec3_scale(vec3_sub(b, a), t));
}

/* ============================================================
   aabb — axis-aligned box, Q16.16 metres
   ============================================================ */

typedef struct {
    vec3_t min;
    vec3_t max;
} aabb_t;

static inline aabb_t aabb_make(vec3_t min, vec3_t max) {
    aabb_t b = { min, max }; return b;
}
static inline aabb_t aabb_from_center_extents(vec3_t c, vec3_t e) {
    aabb_t b = { vec3_sub(c, e), vec3_add(c, e) }; return b;
}
static inline vec3_t aabb_center(aabb_t b) {
    return vec3_scale(vec3_add(b.min, b.max), FIXED_HALF);
}
static inline int aabb_overlaps(aabb_t a, aabb_t b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x
        && a.min.y <= b.max.y && a.max.y >= b.min.y
        && a.min.z <= b.max.z && a.max.z >= b.min.z;
}

/* ============================================================
   quat — (x, y, z, w), w is real
   ============================================================ */

typedef union {
    fixed_4_t simd;
    struct { fixed_t x, y, z, w; };
    fixed_t e[4];
} quat_t;

static inline quat_t quat_identity(void) {
    quat_t r; r.simd = fixed4_set(0, 0, 0, FIXED_ONE); return r;
}
static inline quat_t quat_make(fixed_t x, fixed_t y, fixed_t z, fixed_t w) {
    quat_t r; r.simd = fixed4_set(x, y, z, w); return r;
}
static inline quat_t quat_load(const fixed_t* p) {
    return quat_make(p[0], p[1], p[2], p[3]);
}
static inline void quat_store(fixed_t* p, quat_t q) {
    p[0] = q.x; p[1] = q.y; p[2] = q.z; p[3] = q.w;
}

static inline quat_t quat_add(quat_t a, quat_t b) {
    quat_t r; r.simd = fixed4_add(a.simd, b.simd); return r;
}
static inline quat_t quat_sub(quat_t a, quat_t b) {
    quat_t r; r.simd = fixed4_sub(a.simd, b.simd); return r;
}
static inline quat_t quat_scale(quat_t a, fixed_t s) {
    quat_t r; r.simd = fixed4_mul(a.simd, fixed4_set1(s)); return r;
}
static inline quat_t quat_neg(quat_t a) {
    quat_t r; r.simd = fixed4_neg(a.simd); return r;
}
static inline quat_t quat_conjugate(quat_t a) {
    return quat_make(-a.x, -a.y, -a.z, a.w);
}

/* Hamilton product. Fully scalar: 16 fixed_mul + 12 add/sub. SIMD shuffle
   plumbing for this on Q16.16 ends up with more uops because every per-
   lane mul is itself a multi-uop sequence. */
static inline quat_t quat_mul(quat_t a, quat_t b) {
    fixed_t ax = a.x, ay = a.y, az = a.z, aw = a.w;
    fixed_t bx = b.x, by = b.y, bz = b.z, bw = b.w;
    fixed_t x = fixed_sub(fixed_add(fixed_add(fixed_mul(aw, bx), fixed_mul(ax, bw)),
                                    fixed_mul(ay, bz)),
                          fixed_mul(az, by));
    fixed_t y = fixed_add(fixed_sub(fixed_add(fixed_mul(aw, by), fixed_mul(ay, bw)),
                                    fixed_mul(ax, bz)),
                          fixed_mul(az, bx));
    fixed_t z = fixed_add(fixed_sub(fixed_add(fixed_mul(aw, bz), fixed_mul(ax, by)),
                                    fixed_mul(ay, bx)),
                          fixed_mul(az, bw));
    fixed_t w = fixed_sub(fixed_sub(fixed_sub(fixed_mul(aw, bw), fixed_mul(ax, bx)),
                                    fixed_mul(ay, by)),
                          fixed_mul(az, bz));
    return quat_make(x, y, z, w);
}

static inline fixed_t quat_dot(quat_t a, quat_t b) {
    return fixed_add(fixed_add(fixed_add(fixed_mul(a.x, b.x), fixed_mul(a.y, b.y)),
                               fixed_mul(a.z, b.z)),
                     fixed_mul(a.w, b.w));
}
static inline fixed_t quat_length_sq(quat_t a) { return quat_dot(a, a); }
static inline fixed_t quat_length(quat_t a)    { return fixed_sqrt(quat_length_sq(a)); }

static inline quat_t quat_normalize(quat_t a) {
    fixed_t r = fixed_rsqrt(quat_length_sq(a));
    return quat_scale(a, r);
}

/* axis must be unit-length. Half-angle sin/cos via the polynomial sin/cos. */
static inline quat_t quat_from_axis_angle(vec3_t axis, fixed_t angle) {
    fixed_t half = angle / 2;
    fixed_t s = fixed_sin(half);
    fixed_t c = fixed_cos(half);
    return quat_make(fixed_mul(axis.x, s),
                     fixed_mul(axis.y, s),
                     fixed_mul(axis.z, s),
                     c);
}

/* Rotate v by q (q assumed unit). Uses the identity
       v' = v + q.w * t + cross(qv, t),   t = 2 * cross(qv, v)
   Same result as q * v * q^-1 but with one cross saved vs the naive form. */
static inline vec3_t quat_rotate_vec3(quat_t q, vec3_t v) {
    vec3_t qv = vec3_make(q.x, q.y, q.z);
    vec3_t t  = vec3_scale(vec3_cross(qv, v), fixed_from_int(2));
    return vec3_add(v, vec3_add(vec3_scale(t, q.w), vec3_cross(qv, t)));
}

/* Normalised lerp — cheap stand-in for slerp. Picks the shortest arc by
   flipping b when a·b < 0. */
static inline quat_t quat_nlerp(quat_t a, quat_t b, fixed_t t) {
    if (quat_dot(a, b) < 0) b = quat_neg(b);
    quat_t r; r.simd = fixed4_add(a.simd,
                                  fixed4_mul(fixed4_sub(b.simd, a.simd),
                                             fixed4_set1(t)));
    return quat_normalize(r);
}
