#pragma once

#include <stdio.h>
#include "ecs_math.h"

/* Reuses g_passed / g_failed / EXPECT defined in test_ecs.h.
   test_ecs.h must be included before this file in the TU.

   DETERMINISM: every expected value below is the bit-exact output of the
   current implementation. No tolerance, no near-equal. If any test fails,
   either the algorithm changed (intentional — re-pin the literal) or
   something drifted (regression — fix the code). */

/* ============================================================
   Helpers: sqrt, rsqrt, sin, cos
   ============================================================ */

static void test_fixed_sqrt_known(void) {
    /* Perfect squares — isqrt is exact. */
    EXPECT(fixed_sqrt(0) == 0,                                   "sqrt(0)");
    EXPECT(fixed_sqrt(FIXED_ONE) == FIXED_ONE,                   "sqrt(1)");
    EXPECT(fixed_sqrt(fixed_from_int(4))   == fixed_from_int(2), "sqrt(4)");
    EXPECT(fixed_sqrt(fixed_from_int(9))   == fixed_from_int(3), "sqrt(9)");
    EXPECT(fixed_sqrt(fixed_from_int(16))  == fixed_from_int(4), "sqrt(16)");
    EXPECT(fixed_sqrt(fixed_from_int(100)) == fixed_from_int(10),"sqrt(100)");

    /* Non-square: isqrt floors. Hand-traced bit-exact outputs. */
    EXPECT(fixed_sqrt(fixed_from_int(2)) == 92681,  "sqrt(2)  pinned");
    EXPECT(fixed_sqrt(fixed_from_int(3)) == 113511, "sqrt(3)  pinned");
    EXPECT(fixed_sqrt(FIXED_HALF)        == 46340,  "sqrt(.5) pinned");

    /* sqrt(0.25) = 0.5 exactly. */
    EXPECT(fixed_sqrt(FIXED_ONE >> 2) == FIXED_HALF, "sqrt(.25)");

    /* Negative -> 0 by spec. */
    EXPECT(fixed_sqrt(fixed_from_int(-9)) == 0, "sqrt(neg) -> 0");
}

static void test_fixed_rsqrt_known(void) {
    EXPECT(fixed_rsqrt(FIXED_ONE)         == FIXED_ONE,         "rsqrt(1)");
    EXPECT(fixed_rsqrt(fixed_from_int(4)) == FIXED_HALF,        "rsqrt(4)");
    EXPECT(fixed_rsqrt(FIXED_ONE >> 2)    == fixed_from_int(2), "rsqrt(.25)");
    EXPECT(fixed_rsqrt(0) == 0,                                 "rsqrt(0) -> 0");
    EXPECT(fixed_rsqrt(fixed_from_int(-1)) == 0,                "rsqrt(neg) -> 0");
}

static void test_fixed_sincos_known(void) {
    /* All values are bit-exact outputs of the current poly. */
    EXPECT(fixed_sin(0) == 0,                       "sin(0)");
    EXPECT(fixed_sin(FIXED_HALF_PI)  == 65830,      "sin(pi/2)  pinned");
    EXPECT(fixed_sin(FIXED_PI)       == 0,          "sin(pi)    -> 0 via reduction");
    EXPECT(fixed_sin(-FIXED_HALF_PI) == -65831,     "sin(-pi/2) pinned (floor asymmetry)");
    EXPECT(fixed_sin(FIXED_TWO_PI)   == 0,          "sin(2pi)   -> 0 via mod");

    EXPECT(fixed_cos(0)              == 65830,      "cos(0) = sin(pi/2)");
    EXPECT(fixed_cos(FIXED_HALF_PI)  == 0,          "cos(pi/2)");
    EXPECT(fixed_cos(FIXED_PI)       == -65831,     "cos(pi) pinned");

    fixed_t pi_4 = FIXED_HALF_PI / 2;   /* = 51472 */
    EXPECT(fixed_sin(pi_4) == 46343, "sin(pi/4) pinned");
    EXPECT(fixed_cos(pi_4) == 46342, "cos(pi/4) pinned");

    fixed_t pi_6 = FIXED_PI / 6;        /* = 34314 (truncated) */
    EXPECT(fixed_sin(pi_6) == 32767, "sin(pi/6) pinned");
    EXPECT(fixed_cos(pi_6) == 56772, "cos(pi/6) pinned");

    /* Periodicity is exact: mod 2pi maps both inputs to the same v. */
    EXPECT(fixed_sin(fixed_add(pi_4, FIXED_TWO_PI)) == fixed_sin(pi_4),
           "sin periodic 2pi exact");
}

/* ============================================================
   vec3 (fixed_4_t lane 0..2) — bit-exact known values
   ============================================================ */

static void test_vec3_basic(void) {
    fixed_4_t v = fixed4_vec3(fixed_from_int(1), fixed_from_int(2), fixed_from_int(3));
    EXPECT(v.x == fixed_from_int(1), "fixed4_vec3 x");
    EXPECT(v.y == fixed_from_int(2), "fixed4_vec3 y");
    EXPECT(v.z == fixed_from_int(3), "fixed4_vec3 z");
    EXPECT(v.w == 0,                 "fixed4_vec3 lane 3 = 0 invariant");
    EXPECT(sizeof(fixed_4_t) == 4 * sizeof(fixed_t), "fixed_4_t packs to 4 lanes");

    fixed_4_t z = fixed4_zero();
    EXPECT(z.x == 0 && z.y == 0 && z.z == 0 && z.w == 0, "fixed4_zero");

    fixed_t buf[3] = { fixed_from_int(7), fixed_from_int(-8), FIXED_HALF };
    fixed_4_t l = fixed4_load3(buf);
    EXPECT(l.x == buf[0] && l.y == buf[1] && l.z == buf[2] && l.w == 0, "fixed4_load3");
    fixed_t out[3] = {0};
    fixed4_store3(out, l);
    EXPECT(out[0] == buf[0] && out[1] == buf[1] && out[2] == buf[2], "fixed4_store3");
}

static void test_vec3_arith_known(void) {
    fixed_4_t a = fixed4_vec3(fixed_from_int(1), fixed_from_int(2), fixed_from_int(3));
    fixed_4_t b = fixed4_vec3(fixed_from_int(4), fixed_from_int(5), fixed_from_int(6));

    fixed_4_t s = fixed4_add(a, b);
    EXPECT(s.x == fixed_from_int(5) && s.y == fixed_from_int(7) && s.z == fixed_from_int(9),
           "fixed4_add (1,2,3)+(4,5,6)");

    fixed_4_t d = fixed4_sub(b, a);
    EXPECT(d.x == fixed_from_int(3) && d.y == fixed_from_int(3) && d.z == fixed_from_int(3),
           "fixed4_sub (4,5,6)-(1,2,3)");

    fixed_4_t n = fixed4_neg(a);
    EXPECT(n.x == fixed_from_int(-1) && n.y == fixed_from_int(-2) && n.z == fixed_from_int(-3),
           "fixed4_neg");

    fixed_4_t k = fixed4_scale(a, fixed_from_int(2));
    EXPECT(k.x == fixed_from_int(2) && k.y == fixed_from_int(4) && k.z == fixed_from_int(6),
           "fixed4_scale by 2");

    fixed_4_t h = fixed4_hadamard(a, b);
    EXPECT(h.x == fixed_from_int(4)  && h.y == fixed_from_int(10) && h.z == fixed_from_int(18),
           "fixed4_hadamard");
}

static void test_vec3_dot_cross_known(void) {
    fixed_4_t a = fixed4_vec3(fixed_from_int(1), fixed_from_int(2), fixed_from_int(3));
    fixed_4_t b = fixed4_vec3(fixed_from_int(4), fixed_from_int(5), fixed_from_int(6));

    /* 1*4 + 2*5 + 3*6 = 32 */
    EXPECT(fixed4_dot3(a, b) == fixed_from_int(32), "fixed4_dot3 = 32");

    fixed_4_t c = fixed4_cross3(a, b);
    EXPECT(c.x == fixed_from_int(-3) && c.y == fixed_from_int(6) && c.z == fixed_from_int(-3),
           "fixed4_cross3 (1,2,3)x(4,5,6)");

    fixed_4_t ex = fixed4_vec3(FIXED_ONE, 0, 0);
    fixed_4_t ey = fixed4_vec3(0, FIXED_ONE, 0);
    fixed_4_t ez = fixed4_vec3(0, 0, FIXED_ONE);
    fixed_4_t cz = fixed4_cross3(ex, ey);
    EXPECT(cz.x == ez.x && cz.y == ez.y && cz.z == ez.z, "x_hat x y_hat = z_hat");

    fixed_4_t self = fixed4_cross3(a, a);
    EXPECT(self.x == 0 && self.y == 0 && self.z == 0, "a x a = 0");

    EXPECT(fixed4_dot3(ex, ey) == 0, "x_hat . y_hat = 0");
}

static void test_vec3_length_normalize_known(void) {
    /* (3,4,0): 9+16=25 exactly; sqrt is exact. */
    fixed_4_t v = fixed4_vec3(fixed_from_int(3), fixed_from_int(4), 0);
    EXPECT(fixed4_length3_sq(v) == fixed_from_int(25), "len_sq (3,4,0) = 25");
    EXPECT(fixed4_length3(v)    == fixed_from_int(5),  "len (3,4,0) = 5");

    /* rsqrt(25) = floor(65536^2 / 327680) = 13107 (truncating div).
       Then scale: 3*13107>>16 == 0, 196608*13107>>16 == 39321,
                   262144*13107>>16 == 52428. */
    fixed_4_t n = fixed4_normalize3(v);
    EXPECT(n.x == fixed_from_parts(0, 6), "normalize x = .6 pinned (39321)");
    EXPECT(n.y == fixed_from_parts(0, 8), "normalize y = .8 pinned (52428)");
    EXPECT(n.z == 0,                       "normalize z = 0");

    /* Unit-axis length is exact 1. */
    fixed_4_t ex = fixed4_vec3(FIXED_ONE, 0, 0);
    EXPECT(fixed4_length3(ex) == FIXED_ONE, "len (1,0,0) = 1");
}

static void test_vec3_lerp_known(void) {
    fixed_4_t a = fixed4_vec3(0, 0, 0);
    fixed_4_t b = fixed4_vec3(fixed_from_int(10), fixed_from_int(10), fixed_from_int(10));

    fixed_4_t m = fixed4_lerp(a, b, FIXED_HALF);
    EXPECT(m.x == fixed_from_int(5) && m.y == fixed_from_int(5) && m.z == fixed_from_int(5),
           "lerp t=.5 -> midpoint");

    fixed_4_t z = fixed4_lerp(a, b, 0);
    EXPECT(z.x == 0 && z.y == 0 && z.z == 0, "lerp t=0 -> a");

    fixed_4_t o = fixed4_lerp(a, b, FIXED_ONE);
    EXPECT(o.x == fixed_from_int(10) && o.y == fixed_from_int(10) && o.z == fixed_from_int(10),
           "lerp t=1 -> b");
}

/* ============================================================
   quat — bit-exact known values
   ============================================================ */

static void test_quat_basic(void) {
    quat_t id = quat_identity();
    EXPECT(id.x == 0 && id.y == 0 && id.z == 0 && id.w == FIXED_ONE, "identity");

    quat_t q = quat_make(fixed_from_int(1), fixed_from_int(2), fixed_from_int(3), fixed_from_int(4));
    EXPECT(q.x == fixed_from_int(1) && q.y == fixed_from_int(2) &&
           q.z == fixed_from_int(3) && q.w == fixed_from_int(4), "quat_make");

    quat_t c = quat_conjugate(q);
    EXPECT(c.x == fixed_from_int(-1) && c.y == fixed_from_int(-2) &&
           c.z == fixed_from_int(-3) && c.w == fixed_from_int(4), "conjugate");

    /* 1+4+9+16 = 30 */
    EXPECT(quat_dot(q, q) == fixed_from_int(30), "dot self = 30");
    EXPECT(quat_length_sq(id) == FIXED_ONE, "len_sq(id) = 1");
    EXPECT(quat_length(id)    == FIXED_ONE, "len(id) = 1");
}

static void test_quat_mul_known(void) {
    quat_t id = quat_identity();
    quat_t q  = quat_make(fixed_from_int(1), fixed_from_int(2), fixed_from_int(3), fixed_from_int(4));

    quat_t r1 = quat_mul(id, q);
    EXPECT(r1.x == q.x && r1.y == q.y && r1.z == q.z && r1.w == q.w, "id*q = q");

    quat_t r2 = quat_mul(q, id);
    EXPECT(r2.x == q.x && r2.y == q.y && r2.z == q.z && r2.w == q.w, "q*id = q");

    quat_t i  = quat_make(FIXED_ONE, 0, 0, 0);
    quat_t ii = quat_mul(i, i);
    EXPECT(ii.x == 0 && ii.y == 0 && ii.z == 0 && ii.w == -FIXED_ONE, "i*i = -1");

    quat_t j  = quat_make(0, FIXED_ONE, 0, 0);
    quat_t jj = quat_mul(j, j);
    EXPECT(jj.x == 0 && jj.y == 0 && jj.z == 0 && jj.w == -FIXED_ONE, "j*j = -1");

    quat_t k  = quat_make(0, 0, FIXED_ONE, 0);
    quat_t kk = quat_mul(k, k);
    EXPECT(kk.x == 0 && kk.y == 0 && kk.z == 0 && kk.w == -FIXED_ONE, "k*k = -1");

    quat_t ij = quat_mul(i, j);
    EXPECT(ij.x == 0 && ij.y == 0 && ij.z == FIXED_ONE && ij.w == 0, "i*j = k");

    quat_t ji = quat_mul(j, i);
    EXPECT(ji.x == 0 && ji.y == 0 && ji.z == -FIXED_ONE && ji.w == 0, "j*i = -k");
}

static void test_quat_axis_angle_rotate_known(void) {
    /* 90° around Z. half = 51472. sin(half)=46343, cos(half)=46342.
       qz90 = (0, 0, 46343, 46342). Bit-exact rotate of (1,0,0): (-5, 65540, 0). */
    fixed_4_t z_axis = fixed4_vec3(0, 0, FIXED_ONE);
    quat_t qz90 = quat_from_axis_angle(z_axis, FIXED_HALF_PI);

    EXPECT(qz90.x == 0     && qz90.y == 0 &&
           qz90.z == 46343 && qz90.w == 46342, "qz90 pinned");

    fixed_4_t v = fixed4_vec3(FIXED_ONE, 0, 0);
    fixed_4_t r = quat_rotate_vec3(qz90, v);
    EXPECT(r.x == -5 && r.y == 65540 && r.z == 0, "rotate (1,0,0) by qz90 pinned");

    /* 180° around Y. half = FIXED_PI/2 = 102943. sin(half)=65830, cos(half)=0.
       qy180 = (0, 65830, 0, 0). Rotate (1,0,0): (-66708, 0, 0). */
    fixed_4_t y_axis = fixed4_vec3(0, FIXED_ONE, 0);
    quat_t qy180 = quat_from_axis_angle(y_axis, FIXED_PI);

    EXPECT(qy180.x == 0     && qy180.y == 65830 &&
           qy180.z == 0     && qy180.w == 0,     "qy180 pinned");

    fixed_4_t v2 = fixed4_vec3(FIXED_ONE, 0, 0);
    fixed_4_t r2 = quat_rotate_vec3(qy180, v2);
    EXPECT(r2.x == -66715 && r2.y == 0 && r2.z == 0, "rotate (1,0,0) by qy180 pinned");

    /* Identity rotation is exact. */
    fixed_4_t v3 = fixed4_vec3(fixed_from_int(7), fixed_from_int(-3), FIXED_HALF);
    fixed_4_t r3 = quat_rotate_vec3(quat_identity(), v3);
    EXPECT(r3.x == v3.x && r3.y == v3.y && r3.z == v3.z, "identity rotate = v");
}

static void test_quat_normalize_known(void) {
    /* (2,0,0,0) -> (1,0,0,0) exact. */
    quat_t q = quat_make(fixed_from_int(2), 0, 0, 0);
    quat_t n = quat_normalize(q);
    EXPECT(n.x == FIXED_ONE && n.y == 0 && n.z == 0 && n.w == 0, "normalize (2,0,0,0) = (1,0,0,0)");

    /* Identity is already unit; round-trip exact. */
    quat_t ni = quat_normalize(quat_identity());
    EXPECT(ni.x == 0 && ni.y == 0 && ni.z == 0 && ni.w == FIXED_ONE, "normalize(identity) = identity");
}

static void test_quat_nlerp_known(void) {
    quat_t a = quat_identity();
    quat_t b = quat_make(0, 0, FIXED_ONE, 0);    /* k */

    /* t=0: r=a, normalize(a)=a. */
    quat_t r0 = quat_nlerp(a, b, 0);
    EXPECT(r0.x == 0 && r0.y == 0 && r0.z == 0 && r0.w == FIXED_ONE, "nlerp t=0 = identity");

    /* t=1: r=b, normalize(b)=b (unit). */
    quat_t r1 = quat_nlerp(a, b, FIXED_ONE);
    EXPECT(r1.x == 0 && r1.y == 0 && r1.z == FIXED_ONE && r1.w == 0, "nlerp t=1 = b");
}

/* ============================================================
   Runner
   ============================================================ */

static int test_math_all(void) {
    int before = g_failed;
    printf("=== math.h tests ===\n\n");
    RUN_TEST(test_fixed_sqrt_known);
    RUN_TEST(test_fixed_rsqrt_known);
    RUN_TEST(test_fixed_sincos_known);
    RUN_TEST(test_vec3_basic);
    RUN_TEST(test_vec3_arith_known);
    RUN_TEST(test_vec3_dot_cross_known);
    RUN_TEST(test_vec3_length_normalize_known);
    RUN_TEST(test_vec3_lerp_known);
    RUN_TEST(test_quat_basic);
    RUN_TEST(test_quat_mul_known);
    RUN_TEST(test_quat_axis_angle_rotate_known);
    RUN_TEST(test_quat_normalize_known);
    RUN_TEST(test_quat_nlerp_known);
    int failed = g_failed - before;
    printf("\nmath: %d failed\n", failed);
    return failed ? 1 : 0;
}
