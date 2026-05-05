#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ecs.h"
#include "input.h"

/* Reuses g_passed / g_failed / EXPECT / RUN_TEST from test_ecs.h.
   test_ecs.h must be included before this file. */

/* --- helpers ------------------------------------------------------------ */

typedef struct {
    uint32_t buttons;
    int16_t  axis_x;
    int16_t  axis_y;
} ti_input_t;   /* 8 bytes */

static inline ti_input_t ti_make(uint32_t b, int16_t x, int16_t y) {
    ti_input_t v = { b, x, y };
    return v;
}

/* Read slot bytes back as ti_input_t. */
static inline ti_input_t ti_get(const ecs_input_t* it, uint64_t tick, ecs_pid_t pid) {
    const void* p = ecs_input_get(it, tick, pid);
    ti_input_t v = {0};
    if (p) memcpy(&v, p, sizeof(v));
    return v;
}

static inline bool ti_eq(ti_input_t a, ti_input_t b) {
    return a.buttons == b.buttons && a.axis_x == b.axis_x && a.axis_y == b.axis_y;
}

/* --- basic set/get roundtrip ------------------------------------------- */

static void test_input_basic_roundtrip(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 64);

    EXPECT(ecs_input_register_player(&it, 7),  "register pid 7 ok");
    EXPECT(ecs_input_active_count(&it) == 1u,   "active count 1 after register");

    ti_input_t v = ti_make(0xABCD, 100, -50);
    ecs_input_set(&it, /*tick*/3, /*pid*/7, &v, /*confirmed*/true, /*expected*/1);

    EXPECT(ti_eq(ti_get(&it, 3, 7), v),         "get returns what set wrote");

    ecs_input_view_t view = ecs_input_get_view(&it, 3, 7);
    EXPECT(view.data != NULL,                    "view has data ptr");
    EXPECT(view.present,                         "view marks present");
    EXPECT(view.confirmed,                       "view marks confirmed");

    ecs_input_destroy(&it);
}

/* --- get on unknown pid returns NULL ----------------------------------- */

static void test_input_get_unknown(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);

    EXPECT(ecs_input_get(&it, 0, 12345) == NULL,                 "get unknown pid -> NULL");
    ecs_input_view_t v = ecs_input_get_view(&it, 0, 12345);
    EXPECT(v.data == NULL && !v.present && !v.confirmed,         "view unknown pid -> empty");

    /* Set on unknown pid is a no-op (no crash, no state). */
    ti_input_t in = ti_make(1, 2, 3);
    ecs_input_set(&it, 5, 999, &in, true, 1);
    EXPECT(ecs_input_get(&it, 5, 999) == NULL,                   "set on unknown pid did nothing");

    ecs_input_destroy(&it);
}

/* --- idempotent confirm (same value, twice) ---------------------------- */

static void test_input_set_idempotent_same_value(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 32);
    ecs_input_register_player(&it, 1);
    ecs_input_register_player(&it, 2);

    ti_input_t a = ti_make(0x10, 5, 5);
    ti_input_t b = ti_make(0x20, -1, 1);

    /* Tick 0: confirm both, expected = 2. */
    ecs_input_set(&it, 0, 1, &a, true, 2);
    EXPECT(!ecs_input_tick_confirmed(&it, 0),    "1 of 2 confirmed -> tick not yet confirmed");

    ecs_input_set(&it, 0, 2, &b, true, 2);
    EXPECT(ecs_input_tick_confirmed(&it, 0),     "2 of 2 confirmed -> tick confirmed");

    bool valid = false;
    EXPECT(ecs_input_frontier(&it, &valid) == 0ull && valid, "frontier == 0 after first full tick");

    /* Re-apply same packet: confirmed_count must NOT double-count, frontier
       still 0, tick still confirmed, slot bytes unchanged. */
    ecs_input_set(&it, 0, 1, &a, true, 2);
    ecs_input_set(&it, 0, 2, &b, true, 2);

    EXPECT(ecs_input_tick_confirmed(&it, 0),     "tick still confirmed after replay");
    EXPECT(ecs_input_frontier(&it, &valid) == 0ull && valid, "frontier unchanged on replay");
    EXPECT(ti_eq(ti_get(&it, 0, 1), a),          "pid 1 bytes preserved on replay");
    EXPECT(ti_eq(ti_get(&it, 0, 2), b),          "pid 2 bytes preserved on replay");

    ecs_input_destroy(&it);
}

/* --- idempotent confirm (overwrite with new bytes) --------------------- */

static void test_input_set_overwrite(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    ecs_input_register_player(&it, 5);

    ti_input_t v0 = ti_make(0xAA, 1, 2);
    ti_input_t v1 = ti_make(0xBB, 9, 9);

    /* Predicted first, then confirmed override with different bytes. */
    ecs_input_set(&it, 1, 5, &v0, false, 1);
    ecs_input_view_t view = ecs_input_get_view(&it, 1, 5);
    EXPECT(view.present && !view.confirmed,      "predicted set marks present, not confirmed");
    EXPECT(ti_eq(ti_get(&it, 1, 5), v0),         "predicted bytes visible");

    ecs_input_set(&it, 1, 5, &v1, true, 1);
    view = ecs_input_get_view(&it, 1, 5);
    EXPECT(view.confirmed && view.present,       "confirmed set marks both flags");
    EXPECT(ti_eq(ti_get(&it, 1, 5), v1),         "bytes overwritten by confirmed packet");

    ecs_input_destroy(&it);
}

/* --- multi-packet split for same tick, any order ----------------------- */

static void test_input_multi_packet_split(void) {
    /* 4 players, server splits across 2 UDP packets per tick.
       Apply in different orders -- final state must match. */
    const uint32_t pids[4] = { 10, 11, 12, 13 };
    const ti_input_t vals[4] = {
        { 0x1, 1, 1 },
        { 0x2, 2, 2 },
        { 0x3, 3, 3 },
        { 0x4, 4, 4 },
    };

    ecs_input_t a; ecs_input_init(&a, sizeof(ti_input_t), 16);
    ecs_input_t b; ecs_input_init(&b, sizeof(ti_input_t), 16);
    for (int i = 0; i < 4; i++) {
        ecs_input_register_player(&a, pids[i]);
        ecs_input_register_player(&b, pids[i]);
    }

    /* a: packet1 = {0,1}, packet2 = {2,3}. */
    ecs_input_set(&a, 0, pids[0], &vals[0], true, 4);
    ecs_input_set(&a, 0, pids[1], &vals[1], true, 4);
    ecs_input_set(&a, 0, pids[2], &vals[2], true, 4);
    ecs_input_set(&a, 0, pids[3], &vals[3], true, 4);

    /* b: reverse order. */
    ecs_input_set(&b, 0, pids[3], &vals[3], true, 4);
    ecs_input_set(&b, 0, pids[2], &vals[2], true, 4);
    ecs_input_set(&b, 0, pids[1], &vals[1], true, 4);
    ecs_input_set(&b, 0, pids[0], &vals[0], true, 4);

    EXPECT(ecs_input_tick_confirmed(&a, 0),      "a: tick 0 confirmed after both packets");
    EXPECT(ecs_input_tick_confirmed(&b, 0),      "b: tick 0 confirmed after reverse-order packets");

    bool va = false, vb = false;
    EXPECT(ecs_input_frontier(&a, &va) == ecs_input_frontier(&b, &vb),
                                                 "frontiers match across orderings");
    EXPECT(va && vb,                              "both frontiers valid");

    for (int i = 0; i < 4; i++) {
        EXPECT(ti_eq(ti_get(&a, 0, pids[i]), ti_get(&b, 0, pids[i])),
                                                 "per-pid bytes match across orderings");
    }

    ecs_input_destroy(&a);
    ecs_input_destroy(&b);
}

/* --- frontier advances only when contiguous ---------------------------- */

static void test_input_frontier_contiguous(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 32);
    ecs_input_register_player(&it, 1);

    ti_input_t v = ti_make(0, 0, 0);

    /* Confirm tick 2 first -- frontier cannot start (only tick 0 seeds). */
    ecs_input_set(&it, 2, 1, &v, true, 1);
    bool valid = false;
    (void)ecs_input_frontier(&it, &valid);
    EXPECT(!valid,                                "frontier not seeded by tick > 0 alone");

    /* Confirm tick 0 -- frontier seeds, then walks forward to 2 since the
       all_confirmed bit for tick 2 is already set. Tick 1 still missing,
       so it stops at 0. */
    ecs_input_set(&it, 0, 1, &v, true, 1);
    uint64_t f = ecs_input_frontier(&it, &valid);
    EXPECT(valid && f == 0ull,                    "frontier == 0, gap at tick 1 blocks");

    /* Fill tick 1 -> walks to 2. */
    ecs_input_set(&it, 1, 1, &v, true, 1);
    f = ecs_input_frontier(&it, &valid);
    EXPECT(valid && f == 2ull,                    "frontier extends to 2 after gap fills");

    ecs_input_destroy(&it);
}

/* --- partial confirm does NOT advance frontier ------------------------- */

static void test_input_partial_no_advance(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    ecs_input_register_player(&it, 1);
    ecs_input_register_player(&it, 2);

    ti_input_t v = ti_make(0, 0, 0);
    ecs_input_set(&it, 0, 1, &v, true, 2);

    bool valid = false;
    (void)ecs_input_frontier(&it, &valid);
    EXPECT(!valid,                                "frontier not advanced on partial confirm");
    EXPECT(!ecs_input_tick_confirmed(&it, 0),     "tick 0 not confirmed -- pid 2 missing");

    ecs_input_destroy(&it);
}

/* --- ring wrap: tick T and T+buf_size occupy same slot ----------------- */

static void test_input_ring_wrap(void) {
    const uint32_t BUF = 8;
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), BUF);
    ecs_input_register_player(&it, 1);

    ti_input_t old_v = ti_make(0xDEAD, 1, 1);
    ti_input_t new_v = ti_make(0xBEEF, 9, 9);

    /* Confirm ticks 0..7 contiguously to bring frontier to 7. */
    for (uint64_t t = 0; t < BUF; t++) {
        ecs_input_set(&it, t, 1, &old_v, true, 1);
    }
    bool valid = false;
    EXPECT(ecs_input_frontier(&it, &valid) == (uint64_t)(BUF - 1) && valid,
                                                  "frontier == buf_size-1 after 8 confirms");

    /* Now write tick 8 -- same ring slot as tick 0, must rewrite cleanly. */
    ecs_input_set(&it, BUF, 1, &new_v, true, 1);
    EXPECT(ti_eq(ti_get(&it, BUF, 1), new_v),     "wrap: new tick bytes visible");
    EXPECT(ecs_input_tick_confirmed(&it, BUF),    "wrap: new tick confirmed");
    /* The slot's tick_in_slot now is 8; querying tick 0 returns false. */
    EXPECT(!ecs_input_tick_confirmed(&it, 0),     "wrap: old tick no longer confirmed at this slot");

    ecs_input_destroy(&it);
}

/* --- clear semantics --------------------------------------------------- */

static void test_input_clear(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    ecs_input_register_player(&it, 1);

    ti_input_t v = ti_make(7, 7, 7);
    ecs_input_set(&it, 0, 1, &v, true, 1);
    ecs_input_set(&it, 1, 1, &v, true, 1);

    bool valid = false;
    EXPECT(ecs_input_frontier(&it, &valid) == 1ull && valid, "frontier == 1 after 2 confirms");

    /* Clear tick 1. View should report not-present, not-confirmed.
       Frontier rewinds to 0. */
    ecs_input_clear(&it, 1);
    ecs_input_view_t view = ecs_input_get_view(&it, 1, 1);
    EXPECT(!view.present && !view.confirmed,                  "after clear: not present");
    EXPECT(!ecs_input_tick_confirmed(&it, 1),                 "after clear: tick not confirmed");
    EXPECT(ecs_input_frontier(&it, &valid) == 0ull && valid,  "frontier rewound to 0");

    /* Clear tick 0 too -> frontier becomes invalid. */
    ecs_input_clear(&it, 0);
    (void)ecs_input_frontier(&it, &valid);
    EXPECT(!valid,                                            "frontier invalid after clearing tick 0");

    ecs_input_destroy(&it);
}

/* --- register/unregister cycle freelist & ABA defense ------------------ */

static void test_input_register_unregister_cycle(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);

    EXPECT(ecs_input_register_player(&it, 100),  "register 100");
    EXPECT(!ecs_input_register_player(&it, 100), "double-register rejected");
    EXPECT(ecs_input_active_count(&it) == 1u,     "active still 1 after dup register");

    ti_input_t a = ti_make(0xA, 1, 1);
    ecs_input_set(&it, 0, 100, &a, true, 1);
    EXPECT(ecs_input_tick_confirmed(&it, 0),      "tick 0 confirmed for old pid 100");

    /* Unregister 100 then register a NEW pid (200). The freelist will pop
       the same dense idx that 100 used. ABA defense must clear past-row
       bits at that idx so pid 200 doesn't see stale present/confirmed. */
    ecs_input_unregister_player(&it, 100);
    EXPECT(ecs_input_active_count(&it) == 0u,     "active 0 after unregister");

    EXPECT(ecs_input_register_player(&it, 200),   "register 200 -- reuses freed dense idx");
    ecs_input_view_t v = ecs_input_get_view(&it, 0, 200);
    EXPECT(!v.present,                             "ABA defense: tick 0 not present for new pid 200");
    EXPECT(!v.confirmed,                           "ABA defense: tick 0 not confirmed for new pid 200");

    /* New writes for 200 work normally. */
    ti_input_t b = ti_make(0xB, 2, 2);
    ecs_input_set(&it, 0, 200, &b, true, 1);
    EXPECT(ti_eq(ti_get(&it, 0, 200), b),         "new pid bytes visible after reuse");

    ecs_input_destroy(&it);
}

/* --- determinism: replay the exact same op sequence -> identical state - */

static void test_input_deterministic_replay(void) {
    /* Drive two independent ecs_input_t instances with the SAME sequence
       of ops in the SAME order. Their final byte-for-byte slot contents,
       confirmed bitmap rows, and frontier must match. */
    const uint32_t BUF = 32;
    ecs_input_t a; ecs_input_init(&a, sizeof(ti_input_t), BUF);
    ecs_input_t b; ecs_input_init(&b, sizeof(ti_input_t), BUF);

    const uint32_t pids[3] = { 50, 51, 52 };
    for (int i = 0; i < 3; i++) {
        ecs_input_register_player(&a, pids[i]);
        ecs_input_register_player(&b, pids[i]);
    }

    /* Mixed predicted + confirmed across 5 ticks, multiple players each. */
    for (uint64_t t = 0; t < 5; t++) {
        for (int i = 0; i < 3; i++) {
            ti_input_t v = ti_make((uint32_t)(t * 10 + i), (int16_t)t, (int16_t)i);
            bool conf = ((t + i) & 1u) != 0u;
            ecs_input_set(&a, t, pids[i], &v, conf, 3);
            ecs_input_set(&b, t, pids[i], &v, conf, 3);
        }
    }

    /* Compare slot bytes for every tick & pid. */
    for (uint64_t t = 0; t < 5; t++) {
        for (int i = 0; i < 3; i++) {
            EXPECT(ti_eq(ti_get(&a, t, pids[i]), ti_get(&b, t, pids[i])),
                                                  "deterministic: slot bytes match");
            ecs_input_view_t va = ecs_input_get_view(&a, t, pids[i]);
            ecs_input_view_t vb = ecs_input_get_view(&b, t, pids[i]);
            EXPECT(va.present   == vb.present,    "deterministic: present flag matches");
            EXPECT(va.confirmed == vb.confirmed,  "deterministic: confirmed flag matches");
        }
    }

    bool av = false, bv = false;
    EXPECT(ecs_input_frontier(&a, &av) == ecs_input_frontier(&b, &bv) && av == bv,
                                                  "deterministic: frontier matches");

    ecs_input_destroy(&a);
    ecs_input_destroy(&b);
}

/* --- determinism: order of inputs WITHIN a tick must not affect state -- */

static void test_input_intratick_order_invariance(void) {
    const uint32_t pids[4] = { 70, 71, 72, 73 };
    const ti_input_t vals[4] = {
        { 0xAA, 1, 1 },
        { 0xBB, 2, 2 },
        { 0xCC, 3, 3 },
        { 0xDD, 4, 4 },
    };

    /* All 4! = 24 permutations are too many; sample 4 distinct orders. */
    const int orders[4][4] = {
        { 0, 1, 2, 3 },
        { 3, 2, 1, 0 },
        { 1, 3, 0, 2 },
        { 2, 0, 3, 1 },
    };

    /* Reference run: order 0. */
    ecs_input_t ref; ecs_input_init(&ref, sizeof(ti_input_t), 16);
    for (int i = 0; i < 4; i++) ecs_input_register_player(&ref, pids[i]);
    for (int k = 0; k < 4; k++) {
        int i = orders[0][k];
        ecs_input_set(&ref, 0, pids[i], &vals[i], true, 4);
    }

    for (int o = 1; o < 4; o++) {
        ecs_input_t cur; ecs_input_init(&cur, sizeof(ti_input_t), 16);
        for (int i = 0; i < 4; i++) ecs_input_register_player(&cur, pids[i]);
        for (int k = 0; k < 4; k++) {
            int i = orders[o][k];
            ecs_input_set(&cur, 0, pids[i], &vals[i], true, 4);
        }

        for (int i = 0; i < 4; i++) {
            EXPECT(ti_eq(ti_get(&cur, 0, pids[i]), ti_get(&ref, 0, pids[i])),
                                                  "intra-tick order invariance: bytes");
        }
        EXPECT(ecs_input_tick_confirmed(&cur, 0) == ecs_input_tick_confirmed(&ref, 0),
                                                  "intra-tick order invariance: confirmed");
        bool rv = false, cv = false;
        EXPECT(ecs_input_frontier(&cur, &cv) == ecs_input_frontier(&ref, &rv) && cv == rv,
                                                  "intra-tick order invariance: frontier");

        ecs_input_destroy(&cur);
    }
    ecs_input_destroy(&ref);
}

/* --- seed_frontier for mid-session join -------------------------------- */

static void test_input_seed_frontier(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 32);
    ecs_input_register_player(&it, 1);

    /* Bootstrap as if we joined at tick 1000. */
    ecs_input_seed_frontier(&it, 1000);
    bool valid = false;
    EXPECT(ecs_input_frontier(&it, &valid) == 1000ull && valid,
                                                  "seed sets frontier to 1000");

    /* Confirming tick 1001 advances. */
    ti_input_t v = ti_make(0, 0, 0);
    ecs_input_set(&it, 1001, 1, &v, true, 1);
    EXPECT(ecs_input_frontier(&it, &valid) == 1001ull && valid,
                                                  "frontier extends past seeded value");

    /* Confirming tick 999 (before seed) does NOT pull frontier backward. */
    ecs_input_set(&it, 999, 1, &v, true, 1);
    EXPECT(ecs_input_frontier(&it, &valid) == 1001ull && valid,
                                                  "old confirmation does not regress frontier");

    ecs_input_destroy(&it);
}

/* --- entry point ------------------------------------------------------- */

static int test_input_all(void) {
    int before = g_failed;
    printf("=== ecs_input_t tests ===\n\n");
    RUN_TEST(test_input_basic_roundtrip);
    RUN_TEST(test_input_get_unknown);
    RUN_TEST(test_input_set_idempotent_same_value);
    RUN_TEST(test_input_set_overwrite);
    RUN_TEST(test_input_multi_packet_split);
    RUN_TEST(test_input_frontier_contiguous);
    RUN_TEST(test_input_partial_no_advance);
    RUN_TEST(test_input_ring_wrap);
    RUN_TEST(test_input_clear);
    RUN_TEST(test_input_register_unregister_cycle);
    RUN_TEST(test_input_deterministic_replay);
    RUN_TEST(test_input_intratick_order_invariance);
    RUN_TEST(test_input_seed_frontier);
    int failed = g_failed - before;
    printf("\ninput: %d failed\n", failed);
    return failed ? 1 : 0;
}
