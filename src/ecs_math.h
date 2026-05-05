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

/* Q16.16 vec3 / quaternion math.
   - fixed_4_t is the canonical 4-lane SIMD vector; vec3 ops treat lane 3 as
     padding (invariant: zero) and lane 0..2 as x/y/z. Add/sub/scale/neg/lerp
     use direct SIMD; dot3/cross3/length3 mix SIMD with a scalar tail because
     the 3-lane reduction beats horizontal-extract on Q16.16.
   - quat_t aliases fixed_4_t (union with .x/.y/.z/.w/.simd/.e[]). Add/sub/
     scale/neg/nlerp stay direct SIMD; quat_mul is scalar by design. */

/* ============================================================
   Scalar helpers: sqrt, rsqrt, sin, cos
   ============================================================ */

#define FIXED_PI       ((fixed_t)205887)   /* pi    in Q16.16 */
#define FIXED_TWO_PI   ((fixed_t)411775)   /* 2*pi  in Q16.16 */
#define FIXED_HALF_PI  ((fixed_t)102944)   /* pi/2  in Q16.16 */

/* Digit-by-digit unsigned 64-bit isqrt. Branch-on-bit, no division. */
static inline uint32_t fixed_isqrt64(uint64_t v) {
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
    return (fixed_t)fixed_isqrt64((uint64_t)(uint32_t)x << FIXED_SHIFT);
}

/* rsqrt without isqrt-loop or 64-bit divide. Bitscan gives a seed within √2
   of truth; one sqrt(2) bias on odd exponents tightens that to ~19%, after
   which Newton-Raphson y' = y*(3 - x*y^2)/2 converges quadratically — 4
   iterations clear Q16.16 ULP. y stays in [1, 2^24] for valid Q16.16 inputs,
   so the int64 intermediates (y*y up to 2^48, y*t up to 2^41) cannot overflow. */
static inline fixed_t fixed_rsqrt(fixed_t x) {
    if (x <= 0) return 0;
    uint32_t ux = (uint32_t)x;
#ifdef _MSC_VER
    unsigned long n; _BitScanReverse(&n, ux);
#else
    unsigned n = 31u - (unsigned)__builtin_clz(ux);
#endif
    int e = 48 - (int)n;
    int64_t y = (int64_t)1 << (e >> 1);
    if (e & 1) y = (y * 46341) >> 16;  /* 46341 ≈ 2^16 / sqrt(2) */
    for (int i = 0; i < 4; ++i) {
        int64_t y2  = (y * y) >> FIXED_SHIFT;
        int64_t xy2 = ((int64_t)x * y2) >> FIXED_SHIFT;
        int64_t t   = 3LL * FIXED_ONE - xy2;
        y = (y * t) >> (FIXED_SHIFT + 1);
    }
    return (fixed_t)y;
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
   fixed_4_t vec3/vec4 helpers

   fixed_4_t is a 4-lane SIMD union (defined in fixed.h) with .simd / .e[4]
   / .x .y .z .w views. vec3-style ops use lanes 0..2; lane 3 is padding
   and the convention is to keep it zero so fixed4_eq compares cleanly
   across all four lanes.
   ============================================================ */

/* Build a 3-lane vector (lane 3 = 0 by convention). */
static inline fixed_4_t fixed4_vec3(fixed_t x, fixed_t y, fixed_t z) {
    return fixed4_set(x, y, z, 0);
}
static inline fixed_4_t fixed4_load3(const fixed_t* p) {
    return fixed4_set(p[0], p[1], p[2], 0);
}
static inline void fixed4_store3(fixed_t* p, fixed_4_t v) {
    p[0] = v.x; p[1] = v.y; p[2] = v.z;
}

/* Scalar broadcast multiply: SIMD scale is one mul + a splat. */
static inline fixed_4_t fixed4_scale(fixed_4_t a, fixed_t s) {
    return fixed4_mul(a, fixed4_set1(s));
}
/* Hadamard product == per-lane mul. Alias kept for readability at call site. */
static inline fixed_4_t fixed4_hadamard(fixed_4_t a, fixed_4_t b) {
    return fixed4_mul(a, b);
}

/* 3-lane dot. Mul is SIMD; horizontal sum stays scalar — the 3-lane reduction
   on Q16.16 beats shuffle-and-add via lane reads, since fixed4_mul itself
   is multi-uop and the union view is free. Lane 3 is ignored. */
static inline fixed_t fixed4_dot3(fixed_4_t a, fixed_4_t b) {
    fixed_4_t m = fixed4_mul(a, b);
    return fixed_add(fixed_add(m.x, m.y), m.z);
}
/* 4-lane dot. Used by quat_dot. */
static inline fixed_t fixed4_dot4(fixed_4_t a, fixed_4_t b) {
    fixed_4_t m = fixed4_mul(a, b);
    return fixed_add(fixed_add(m.x, m.y), fixed_add(m.z, m.w));
}

/* 3D cross. Shuffles + per-lane scalar muls — Q16.16 SIMD shuffle plumbing
   loses to plain scalar here for the same reason quat_mul stays scalar. */
static inline fixed_4_t fixed4_cross3(fixed_4_t a, fixed_4_t b) {
    return fixed4_set(
        fixed_sub(fixed_mul(a.y, b.z), fixed_mul(a.z, b.y)),
        fixed_sub(fixed_mul(a.z, b.x), fixed_mul(a.x, b.z)),
        fixed_sub(fixed_mul(a.x, b.y), fixed_mul(a.y, b.x)),
        0);
}

static inline fixed_t fixed4_length3_sq(fixed_4_t a) { return fixed4_dot3(a, a); }
static inline fixed_t fixed4_length3(fixed_4_t a)    { return fixed_sqrt(fixed4_length3_sq(a)); }

static inline fixed_4_t fixed4_normalize3(fixed_4_t a) {
    fixed_t r = fixed_rsqrt(fixed4_length3_sq(a));
    return fixed4_scale(a, r);
}
static inline fixed_4_t fixed4_lerp(fixed_4_t a, fixed_4_t b, fixed_t t) {
    return fixed4_add(a, fixed4_scale(fixed4_sub(b, a), t));
}

/* ============================================================
   aabb — axis-aligned box, Q16.16 metres

   min / max are fixed_4_t with lane 3 = 0 by convention. SIMD storage
   (16-byte aligned) lets aabb_overlaps fold into a single 3-lane compare;
   broadphase reads min.x/.y/.z directly via the union view.
   ============================================================ */

typedef struct {
    fixed_4_t min;
    fixed_4_t max;
} aabb_t;

static inline aabb_t aabb_make(fixed_4_t min, fixed_4_t max) {
    aabb_t b = { min, max }; return b;
}
static inline aabb_t aabb_from_center_extents(fixed_4_t c, fixed_4_t e) {
    aabb_t b = { fixed4_sub(c, e), fixed4_add(c, e) }; return b;
}
static inline fixed_4_t aabb_center(aabb_t b) {
    return fixed4_scale(fixed4_add(b.min, b.max), FIXED_HALF);
}
static inline int aabb_overlaps(aabb_t a, aabb_t b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x
        && a.min.y <= b.max.y && a.max.y >= b.min.y
        && a.min.z <= b.max.z && a.max.z >= b.min.z;
}

/* ============================================================
   quat — (x, y, z, w), w is real

   quat_t aliases fixed_4_t. Ops that map cleanly to lane-wise SIMD
   (add/sub/scale/neg/nlerp) reuse fixed4_*; quat_mul stays scalar.
   ============================================================ */

typedef fixed_4_t quat_t;

static inline quat_t quat_identity(void) { return fixed4_set(0, 0, 0, FIXED_ONE); }
static inline quat_t quat_make(fixed_t x, fixed_t y, fixed_t z, fixed_t w) {
    return fixed4_set(x, y, z, w);
}
static inline quat_t quat_load(const fixed_t* p)        { return fixed4_load(p); }
static inline void   quat_store(fixed_t* p, quat_t q)   { fixed4_store(p, q); }

static inline quat_t quat_add(quat_t a, quat_t b)       { return fixed4_add(a, b); }
static inline quat_t quat_sub(quat_t a, quat_t b)       { return fixed4_sub(a, b); }
static inline quat_t quat_scale(quat_t a, fixed_t s)    { return fixed4_scale(a, s); }
static inline quat_t quat_neg(quat_t a)                 { return fixed4_neg(a); }
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

static inline fixed_t quat_dot(quat_t a, quat_t b)    { return fixed4_dot4(a, b); }
static inline fixed_t quat_length_sq(quat_t a)        { return quat_dot(a, a); }
static inline fixed_t quat_length(quat_t a)           { return fixed_sqrt(quat_length_sq(a)); }

static inline quat_t quat_normalize(quat_t a) {
    fixed_t r = fixed_rsqrt(quat_length_sq(a));
    return quat_scale(a, r);
}

/* axis must be unit-length. Half-angle sin/cos via the polynomial sin/cos. */
static inline quat_t quat_from_axis_angle(fixed_4_t axis, fixed_t angle) {
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
   Same result as q * v * q^-1 but with one cross saved vs the naive form.
   qv is q with w forced to 0 so the cross/scale stay 3-lane clean. */
static inline fixed_4_t quat_rotate_vec3(quat_t q, fixed_4_t v) {
    fixed_4_t qv = fixed4_set(q.x, q.y, q.z, 0);
    fixed_4_t t  = fixed4_scale(fixed4_cross3(qv, v), fixed_from_int(2));
    return fixed4_add(v, fixed4_add(fixed4_scale(t, q.w), fixed4_cross3(qv, t)));
}

/* Normalised lerp — cheap stand-in for slerp. Picks the shortest arc by
   flipping b when a·b < 0. */
static inline quat_t quat_nlerp(quat_t a, quat_t b, fixed_t t) {
    if (quat_dot(a, b) < 0) b = quat_neg(b);
    quat_t r = fixed4_add(a, fixed4_scale(fixed4_sub(b, a), t));
    return quat_normalize(r);
}

/* ============================================================
   transform — TRS (translation, rotation, non-uniform scale)
   ============================================================ */

/* Apply order is scale -> rotate -> translate, matching the convention
   `world = parent * local`. Non-uniform scale composes lossily under
   rotation (introduces shear); transform_mul therefore stays in TRS form
   only when child rotation is axis-aligned with parent scale -- callers
   that need exact composition under arbitrary rotation should bake to a
   matrix. For uniform scale the composition is exact. */
typedef struct {
    fixed_4_t position;
    quat_t    rotation;
    fixed_4_t scale;       /* per-axis */
} transform_t;

static inline transform_t transform_identity(void) {
    transform_t t = {
        fixed4_zero(),
        quat_identity(),
        fixed4_set(FIXED_ONE, FIXED_ONE, FIXED_ONE, 0),
    };
    return t;
}

static inline transform_t transform_make(fixed_4_t position, quat_t rotation, fixed_4_t scale) {
    transform_t t = { position, rotation, scale };
    return t;
}

/* World point from local point: p_w = R * (S ⊙ p_l) + T. */
static inline fixed_4_t transform_point(transform_t t, fixed_4_t p) {
    return fixed4_add(t.position, quat_rotate_vec3(t.rotation, fixed4_hadamard(t.scale, p)));
}

/* Direction transform: ignores translation, applies scale + rotation.
   For a unit normal under non-uniform scale callers should renormalise. */
static inline fixed_4_t transform_dir(transform_t t, fixed_4_t d) {
    return quat_rotate_vec3(t.rotation, fixed4_hadamard(t.scale, d));
}

/* Compose: out(v) = a(b(v)). Position is a's TRS applied to b.position. */
static inline transform_t transform_mul(transform_t a, transform_t b) {
    transform_t r;
    r.scale    = fixed4_hadamard(a.scale, b.scale);
    r.rotation = quat_mul(a.rotation, b.rotation);
    r.position = fixed4_add(a.position,
                            quat_rotate_vec3(a.rotation, fixed4_hadamard(a.scale, b.position)));
    return r;
}

/* Inverse, valid for any non-zero scale. Order: undo translation, undo
   rotation, undo scale. Per-axis reciprocal of scale uses fixed_div, so
   precision degrades for very small components. */
static inline transform_t transform_inverse(transform_t t) {
    fixed_4_t inv_scale = fixed4_set(fixed_div(FIXED_ONE, t.scale.x),
                                     fixed_div(FIXED_ONE, t.scale.y),
                                     fixed_div(FIXED_ONE, t.scale.z),
                                     0);
    quat_t    inv_rot = quat_conjugate(t.rotation);
    fixed_4_t inv_pos = fixed4_hadamard(inv_scale,
                                        quat_rotate_vec3(inv_rot, fixed4_neg(t.position)));
    transform_t r = { inv_pos, inv_rot, inv_scale };
    return r;
}
