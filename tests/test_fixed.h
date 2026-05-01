#pragma once

#include <stdio.h>
#include "fixed.h"

/* Reuses g_passed / g_failed / EXPECT defined in test_ecs.h.
   test_ecs.h must be included before this file in the TU. */

static void test_fixed_scalar(void) {
    EXPECT(fixed_from_int(0) == 0, "fixed_from_int(0)");
    EXPECT(fixed_from_int(1) == FIXED_ONE, "fixed_from_int(1)");
    EXPECT(fixed_from_int(-1) == -FIXED_ONE, "fixed_from_int(-1)");
    EXPECT(fixed_to_int(fixed_from_int(7)) == 7, "fixed_to_int round-trip");

    fixed_t a = fixed_from_int(2);
    fixed_t b = fixed_from_int(3);
    EXPECT(fixed_add(a, b) == fixed_from_int(5), "scalar add");
    EXPECT(fixed_sub(b, a) == fixed_from_int(1), "scalar sub");
    EXPECT(fixed_mul(a, b) == fixed_from_int(6), "scalar mul int");
    EXPECT(fixed_div(fixed_from_int(6), b) == fixed_from_int(2), "scalar div int");

    /* Fractional */
    EXPECT(fixed_mul(FIXED_HALF, FIXED_HALF) == (FIXED_ONE >> 2), "0.5 * 0.5 = 0.25");
    EXPECT(fixed_div(FIXED_ONE, fixed_from_int(2)) == FIXED_HALF, "1 / 2 = 0.5");
    EXPECT(fixed_mul(fixed_from_int(-2), fixed_from_int(3)) == fixed_from_int(-6), "neg mul");
    EXPECT(fixed_div(fixed_from_int(-6), fixed_from_int(2)) == fixed_from_int(-3), "neg div");

    /* fixed_from_parts: bit-exact decimal-style construction.
       (3, 14): den = 100, fp = (14<<16)/100 = 9175, w = 196608, total 205783.
       (0, 5):  den = 10,  fp = (5<<16)/10  = 32768 = FIXED_HALF.
       (0, 6):  den = 10,  fp = (6<<16)/10  = 39321.
       (0, 8):  den = 10,  fp = (8<<16)/10  = 52428.
       (-2, 25): w = -131072, fp = (25<<16)/100 = 16384, total -147456. */
    EXPECT(fixed_from_parts(0, 0)    == 0,                  "fixed_from_parts(0,0) = 0");
    EXPECT(fixed_from_parts(3, 0)    == fixed_from_int(3),  "fixed_from_parts(3,0) = 3.0");
    EXPECT(fixed_from_parts(3, 14)   == 205783,             "fixed_from_parts(3,14) = 3.14");
    EXPECT(fixed_from_parts(0, 5)    == FIXED_HALF,         "fixed_from_parts(0,5) = 0.5");
    EXPECT(fixed_from_parts(0, 6)    == 39321,              "fixed_from_parts(0,6) = 0.6 pinned");
    EXPECT(fixed_from_parts(0, 8)    == 52428,              "fixed_from_parts(0,8) = 0.8 pinned");
    EXPECT(fixed_from_parts(-2, 25)  == -147456,            "fixed_from_parts(-2,25) = -2.25");
    /* (3, 7071): den=10000, fp = (7071<<16)/10000 = 463405056/10000 = 46340 (trunc).
                  total = 196608 + 46340 = 242948. */
    EXPECT(fixed_from_parts(3, 7071) == 242948, "fixed_from_parts(3,7071) pinned");
}

static void test_fixed4_arith(void) {
    fixed_t a[4] = {
        fixed_from_int(1), fixed_from_int(2),
        fixed_from_int(-3), fixed_from_int(4)
    };
    fixed_t b[4] = {
        fixed_from_int(5), FIXED_HALF,
        fixed_from_int(-2), fixed_from_int(8)
    };
    fixed_t out[4];

    fixed_4_t va = fixed4_load(a);
    fixed_4_t vb = fixed4_load(b);

    fixed4_store(out, fixed4_add(va, vb));
    for (int i = 0; i < 4; ++i) EXPECT(out[i] == fixed_add(a[i], b[i]), "fixed4_add lane");

    fixed4_store(out, fixed4_sub(va, vb));
    for (int i = 0; i < 4; ++i) EXPECT(out[i] == fixed_sub(a[i], b[i]), "fixed4_sub lane");

    fixed4_store(out, fixed4_mul(va, vb));
    for (int i = 0; i < 4; ++i) EXPECT(out[i] == fixed_mul(a[i], b[i]), "fixed4_mul lane");

    fixed4_store(out, fixed4_div(va, vb));
    for (int i = 0; i < 4; ++i) EXPECT(out[i] == fixed_div(a[i], b[i]), "fixed4_div lane");

    fixed4_store(out, fixed4_neg(va));
    for (int i = 0; i < 4; ++i) EXPECT(out[i] == -a[i], "fixed4_neg lane");

    fixed4_store(out, fixed4_set1(fixed_from_int(7)));
    for (int i = 0; i < 4; ++i) EXPECT(out[i] == fixed_from_int(7), "fixed4_set1 lane");

    fixed4_store(out, fixed4_zero());
    for (int i = 0; i < 4; ++i) EXPECT(out[i] == 0, "fixed4_zero lane");
}

static void test_fixed4_cmp(void) {
    fixed_t a[4] = {
        fixed_from_int(1), fixed_from_int(2),
        fixed_from_int(3), fixed_from_int(-4)
    };
    fixed_t b[4] = {
        fixed_from_int(2), fixed_from_int(2),
        fixed_from_int(1), fixed_from_int(5)
    };
    fixed_4_t va = fixed4_load(a);
    fixed_4_t vb = fixed4_load(b);

    uint8_t expect_eq = 0, expect_lt = 0, expect_gt = 0, expect_le = 0, expect_ge = 0;
    for (int i = 0; i < 4; ++i) {
        if (a[i] == b[i]) expect_eq |= (uint8_t)(1u << i);
        if (a[i] <  b[i]) expect_lt |= (uint8_t)(1u << i);
        if (a[i] >  b[i]) expect_gt |= (uint8_t)(1u << i);
        if (a[i] <= b[i]) expect_le |= (uint8_t)(1u << i);
        if (a[i] >= b[i]) expect_ge |= (uint8_t)(1u << i);
    }
    EXPECT(fixed4_eq(va, vb) == expect_eq, "fixed4_eq mask");
    EXPECT(fixed4_lt(va, vb) == expect_lt, "fixed4_lt mask");
    EXPECT(fixed4_gt(va, vb) == expect_gt, "fixed4_gt mask");
    EXPECT(fixed4_le(va, vb) == expect_le, "fixed4_le mask");
    EXPECT(fixed4_ge(va, vb) == expect_ge, "fixed4_ge mask");
}

static void test_fixed8_arith(void) {
    fixed_t a[8] = {
        fixed_from_int(1), fixed_from_int(2), fixed_from_int(-3), fixed_from_int(4),
        FIXED_HALF,        fixed_from_int(-5), FIXED_ONE,         fixed_from_int(10)
    };
    fixed_t b[8] = {
        fixed_from_int(5),  FIXED_HALF,        fixed_from_int(-2), fixed_from_int(8),
        fixed_from_int(2),  fixed_from_int(3), fixed_from_int(-1), fixed_from_int(4)
    };
    fixed_t out[8];

    fixed_8_t va = fixed8_load(a);
    fixed_8_t vb = fixed8_load(b);

    fixed8_store(out, fixed8_add(va, vb));
    for (int i = 0; i < 8; ++i) EXPECT(out[i] == fixed_add(a[i], b[i]), "fixed8_add lane");

    fixed8_store(out, fixed8_sub(va, vb));
    for (int i = 0; i < 8; ++i) EXPECT(out[i] == fixed_sub(a[i], b[i]), "fixed8_sub lane");

    fixed8_store(out, fixed8_mul(va, vb));
    for (int i = 0; i < 8; ++i) EXPECT(out[i] == fixed_mul(a[i], b[i]), "fixed8_mul lane");

    fixed8_store(out, fixed8_div(va, vb));
    for (int i = 0; i < 8; ++i) EXPECT(out[i] == fixed_div(a[i], b[i]), "fixed8_div lane");

    fixed8_store(out, fixed8_neg(va));
    for (int i = 0; i < 8; ++i) EXPECT(out[i] == -a[i], "fixed8_neg lane");

    fixed8_store(out, fixed8_set1(fixed_from_int(13)));
    for (int i = 0; i < 8; ++i) EXPECT(out[i] == fixed_from_int(13), "fixed8_set1 lane");

    fixed8_store(out, fixed8_zero());
    for (int i = 0; i < 8; ++i) EXPECT(out[i] == 0, "fixed8_zero lane");
}

static void test_fixed8_cmp(void) {
    fixed_t a[8] = {
        fixed_from_int(1), fixed_from_int(2), fixed_from_int(3), fixed_from_int(-4),
        fixed_from_int(0), fixed_from_int(7), fixed_from_int(7), fixed_from_int(-8)
    };
    fixed_t b[8] = {
        fixed_from_int(2), fixed_from_int(2), fixed_from_int(1), fixed_from_int(5),
        fixed_from_int(0), fixed_from_int(6), fixed_from_int(7), fixed_from_int(-9)
    };
    fixed_8_t va = fixed8_load(a);
    fixed_8_t vb = fixed8_load(b);

    uint8_t expect_eq = 0, expect_lt = 0, expect_gt = 0, expect_le = 0, expect_ge = 0;
    for (int i = 0; i < 8; ++i) {
        if (a[i] == b[i]) expect_eq |= (uint8_t)(1u << i);
        if (a[i] <  b[i]) expect_lt |= (uint8_t)(1u << i);
        if (a[i] >  b[i]) expect_gt |= (uint8_t)(1u << i);
        if (a[i] <= b[i]) expect_le |= (uint8_t)(1u << i);
        if (a[i] >= b[i]) expect_ge |= (uint8_t)(1u << i);
    }
    EXPECT(fixed8_eq(va, vb) == expect_eq, "fixed8_eq mask");
    EXPECT(fixed8_lt(va, vb) == expect_lt, "fixed8_lt mask");
    EXPECT(fixed8_gt(va, vb) == expect_gt, "fixed8_gt mask");
    EXPECT(fixed8_le(va, vb) == expect_le, "fixed8_le mask");
    EXPECT(fixed8_ge(va, vb) == expect_ge, "fixed8_ge mask");
}

/* Known-result FMA-style tests with hand-computed expected values per lane.
   Catch bugs that "compare against scalar" tests miss when both paths share
   the same defect. */

static void test_fixed4_fma_known(void) {
    /* result = a + b * c, all positive ints. */
    fixed_t a[4]        = { fixed_from_int(1), fixed_from_int(2), fixed_from_int(3), fixed_from_int(4) };
    fixed_t b[4]        = { fixed_from_int(2), fixed_from_int(3), fixed_from_int(4), fixed_from_int(5) };
    fixed_t c[4]        = { fixed_from_int(3), fixed_from_int(4), fixed_from_int(5), fixed_from_int(6) };
    fixed_t expected[4] = { fixed_from_int(7), fixed_from_int(14), fixed_from_int(23), fixed_from_int(34) };
    fixed_t out[4];

    fixed_4_t va = fixed4_load(a);
    fixed_4_t vb = fixed4_load(b);
    fixed_4_t vc = fixed4_load(c);
    fixed4_store(out, fixed4_add(va, fixed4_mul(vb, vc)));

    for (int i = 0; i < 4; ++i) EXPECT(out[i] == expected[i], "fixed4 a + b*c == known");
}

static void test_fixed4_frac_known(void) {
    /* result = a * b + c, fractional + signed.
       Lane 0: 0.5 * 0.5 + 0.25 = 0.5
       Lane 1: 2   * 0.5 + 1    = 2
       Lane 2: -0.5* 0.5 + 0.25 = 0
       Lane 3: -3  * 0.5 + 2    = 0.5 */
    fixed_t a[4]        = { FIXED_HALF, fixed_from_int(2), -FIXED_HALF, fixed_from_int(-3) };
    fixed_t b[4]        = { FIXED_HALF, FIXED_HALF, FIXED_HALF, FIXED_HALF };
    fixed_t c[4]        = { FIXED_ONE >> 2, FIXED_ONE, FIXED_ONE >> 2, fixed_from_int(2) };
    fixed_t expected[4] = { FIXED_HALF, fixed_from_int(2), 0, FIXED_HALF };
    fixed_t out[4];

    fixed_4_t va = fixed4_load(a);
    fixed_4_t vb = fixed4_load(b);
    fixed_4_t vc = fixed4_load(c);
    fixed4_store(out, fixed4_add(fixed4_mul(va, vb), vc));

    for (int i = 0; i < 4; ++i) EXPECT(out[i] == expected[i], "fixed4 a*b + c == known frac");
}

static void test_fixed8_fma_known(void) {
    /* result = (a + b) * c, mixed signs, includes 0 multiplier. */
    fixed_t a[8] = { fixed_from_int(1), fixed_from_int(-2), fixed_from_int(3),  fixed_from_int(-4),
                     fixed_from_int(5), fixed_from_int(-6), fixed_from_int(7),  fixed_from_int(-8) };
    fixed_t b[8] = { fixed_from_int(2), fixed_from_int(3),  fixed_from_int(-1), fixed_from_int(0),
                     fixed_from_int(1), fixed_from_int(2),  fixed_from_int(-3), fixed_from_int(4) };
    fixed_t c[8] = { fixed_from_int(10), fixed_from_int(20), fixed_from_int(5), fixed_from_int(7),
                     fixed_from_int(0),  fixed_from_int(-1), fixed_from_int(2), fixed_from_int(3) };
    fixed_t expected[8] = { fixed_from_int(30), fixed_from_int(20), fixed_from_int(10), fixed_from_int(-28),
                            fixed_from_int(0),  fixed_from_int(4),  fixed_from_int(8),  fixed_from_int(-12) };
    fixed_t out[8];

    fixed_8_t va = fixed8_load(a);
    fixed_8_t vb = fixed8_load(b);
    fixed_8_t vc = fixed8_load(c);
    fixed8_store(out, fixed8_mul(fixed8_add(va, vb), vc));

    for (int i = 0; i < 8; ++i) EXPECT(out[i] == expected[i], "fixed8 (a+b)*c == known");
}

static void test_fixed8_poly_known(void) {
    /* result = a*b - c/d, with b=0.5 and d=2 chosen so all div results land
       on exact integers. Expected = [-1,-2,...,-8]. */
    fixed_t a[8] = { fixed_from_int(2),  fixed_from_int(4),  fixed_from_int(6),  fixed_from_int(8),
                     fixed_from_int(10), fixed_from_int(12), fixed_from_int(14), fixed_from_int(16) };
    fixed_t b[8] = { FIXED_HALF, FIXED_HALF, FIXED_HALF, FIXED_HALF,
                     FIXED_HALF, FIXED_HALF, FIXED_HALF, FIXED_HALF };
    fixed_t c[8] = { fixed_from_int(4),  fixed_from_int(8),  fixed_from_int(12), fixed_from_int(16),
                     fixed_from_int(20), fixed_from_int(24), fixed_from_int(28), fixed_from_int(32) };
    fixed_t d[8] = { fixed_from_int(2), fixed_from_int(2), fixed_from_int(2), fixed_from_int(2),
                     fixed_from_int(2), fixed_from_int(2), fixed_from_int(2), fixed_from_int(2) };
    fixed_t expected[8] = { fixed_from_int(-1), fixed_from_int(-2), fixed_from_int(-3), fixed_from_int(-4),
                            fixed_from_int(-5), fixed_from_int(-6), fixed_from_int(-7), fixed_from_int(-8) };
    fixed_t out[8];

    fixed_8_t va = fixed8_load(a);
    fixed_8_t vb = fixed8_load(b);
    fixed_8_t vc = fixed8_load(c);
    fixed_8_t vd = fixed8_load(d);
    fixed8_store(out, fixed8_sub(fixed8_mul(va, vb), fixed8_div(vc, vd)));

    for (int i = 0; i < 8; ++i) EXPECT(out[i] == expected[i], "fixed8 a*b - c/d == known");
}

static int test_fixed_all(void) {
    int before = g_failed;
    printf("=== fixed.h tests ===\n\n");
    test_fixed_scalar();
    test_fixed4_arith();
    test_fixed4_cmp();
    test_fixed8_arith();
    test_fixed8_cmp();
    test_fixed4_fma_known();
    test_fixed4_frac_known();
    test_fixed8_fma_known();
    test_fixed8_poly_known();
    int failed = g_failed - before;
    printf("\nfixed: %d failed\n", failed);
    return failed ? 1 : 0;
}
