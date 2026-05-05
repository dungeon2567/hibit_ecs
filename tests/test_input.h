#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ecs.h"
#include "input.h"

/* Reuses g_passed / g_failed / EXPECT / RUN_TEST from test_ecs.h.
   test_ecs.h must be included before this file.

   Conventions:
     - Tick 0 is reserved as the "no frontier" sentinel; valid ticks
       start at 1.
     - all_confirmed status is computed on demand from current live
       roster (dense_ids != NIL) AND confirmed bits.
     - ecs_input_set has no `expected` parameter; the live roster
       implicitly defines it.
     - ecs_input_frontier returns 0 = no frontier yet. */

typedef struct {
    uint32_t buttons;
    int16_t  axis_x;
    int16_t  axis_y;
} ti_input_t;   /* 8 bytes */

static inline ti_input_t ti_make(uint32_t b, int16_t x, int16_t y) {
    ti_input_t v = { b, x, y };
    return v;
}

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
    EXPECT(ecs_input_active_count(&it) == 1u,  "active count 1 after register");

    ti_input_t v = ti_make(0xABCD, 100, -50);
    ecs_input_set(&it, 3, 7, &v, true);

    EXPECT(ti_eq(ti_get(&it, 3, 7), v),        "get returns what set wrote");
    ecs_input_view_t view = ecs_input_get_view(&it, 3, 7);
    EXPECT(view.data != NULL,                  "view has data ptr");
    EXPECT(view.present,                       "view marks present");
    EXPECT(view.confirmed,                     "view marks confirmed");

    ecs_input_destroy(&it);
}

/* --- get on unknown pid returns NULL ----------------------------------- */

static void test_input_get_unknown(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);

    EXPECT(ecs_input_get(&it, 1, 12345) == NULL,                 "get unknown pid -> NULL");
    ecs_input_view_t v = ecs_input_get_view(&it, 1, 12345);
    EXPECT(v.data == NULL && !v.present && !v.confirmed,         "view unknown pid -> empty");

    ti_input_t in = ti_make(1, 2, 3);
    ecs_input_set(&it, 5, 999, &in, true);
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

    ecs_input_set(&it, 1, 1, &a, true);
    EXPECT(!ecs_input_tick_confirmed(&it, 1),   "1 of 2 confirmed -> tick not yet confirmed");
    ecs_input_set(&it, 1, 2, &b, true);
    EXPECT(ecs_input_tick_confirmed(&it, 1),    "2 of 2 confirmed -> tick confirmed");
    EXPECT(ecs_input_frontier(&it) == 1ull,     "frontier == 1 after first full tick");

    /* Replay: must not double-confirm. */
    ecs_input_set(&it, 1, 1, &a, true);
    ecs_input_set(&it, 1, 2, &b, true);
    EXPECT(ecs_input_tick_confirmed(&it, 1),    "tick still confirmed after replay");
    EXPECT(ecs_input_frontier(&it) == 1ull,     "frontier unchanged on replay");
    EXPECT(ti_eq(ti_get(&it, 1, 1), a),         "pid 1 bytes preserved on replay");
    EXPECT(ti_eq(ti_get(&it, 1, 2), b),         "pid 2 bytes preserved on replay");

    ecs_input_destroy(&it);
}

/* --- predicted then confirmed (overwrite with new bytes) --------------- */

static void test_input_set_overwrite(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    ecs_input_register_player(&it, 5);

    ti_input_t v0 = ti_make(0xAA, 1, 2);
    ti_input_t v1 = ti_make(0xBB, 9, 9);

    ecs_input_set(&it, 1, 5, &v0, false);
    ecs_input_view_t view = ecs_input_get_view(&it, 1, 5);
    EXPECT(view.present && !view.confirmed,    "predicted set marks present, not confirmed");
    EXPECT(ti_eq(ti_get(&it, 1, 5), v0),       "predicted bytes visible");

    ecs_input_set(&it, 1, 5, &v1, true);
    view = ecs_input_get_view(&it, 1, 5);
    EXPECT(view.confirmed && view.present,     "confirmed set marks both flags");
    EXPECT(ti_eq(ti_get(&it, 1, 5), v1),       "bytes overwritten by confirmed packet");

    ecs_input_destroy(&it);
}

/* --- multi-packet split, any order ------------------------------------- */

static void test_input_multi_packet_split(void) {
    const uint32_t pids[4] = { 10, 11, 12, 13 };
    const ti_input_t vals[4] = {
        { 0x1, 1, 1 }, { 0x2, 2, 2 }, { 0x3, 3, 3 }, { 0x4, 4, 4 },
    };

    ecs_input_t a; ecs_input_init(&a, sizeof(ti_input_t), 16);
    ecs_input_t b; ecs_input_init(&b, sizeof(ti_input_t), 16);
    for (int i = 0; i < 4; i++) {
        ecs_input_register_player(&a, pids[i]);
        ecs_input_register_player(&b, pids[i]);
    }

    for (int i = 0; i < 4; i++) ecs_input_set(&a, 1, pids[i], &vals[i], true);
    for (int i = 3; i >= 0; i--) ecs_input_set(&b, 1, pids[i], &vals[i], true);

    EXPECT(ecs_input_tick_confirmed(&a, 1),   "a: tick confirmed");
    EXPECT(ecs_input_tick_confirmed(&b, 1),   "b: tick confirmed (reverse order)");
    EXPECT(ecs_input_frontier(&a) == ecs_input_frontier(&b),
                                              "frontiers match across orderings");
    for (int i = 0; i < 4; i++) {
        EXPECT(ti_eq(ti_get(&a, 1, pids[i]), ti_get(&b, 1, pids[i])),
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

    /* Confirm tick 3 first -- frontier cannot start (only tick 1 seeds). */
    ecs_input_set(&it, 3, 1, &v, true);
    EXPECT(ecs_input_frontier(&it) == 0ull,    "frontier not seeded by tick > 1 alone");

    /* Confirm tick 1 -- seeds, walks forward. Tick 2 missing -> stops at 1. */
    ecs_input_set(&it, 1, 1, &v, true);
    EXPECT(ecs_input_frontier(&it) == 1ull,    "frontier == 1, gap at tick 2 blocks");

    /* Fill tick 2 -> walks to 3. */
    ecs_input_set(&it, 2, 1, &v, true);
    EXPECT(ecs_input_frontier(&it) == 3ull,    "frontier extends to 3 after gap fills");

    ecs_input_destroy(&it);
}

/* --- partial confirm does NOT advance frontier ------------------------- */

static void test_input_partial_no_advance(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    ecs_input_register_player(&it, 1);
    ecs_input_register_player(&it, 2);

    ti_input_t v = ti_make(0, 0, 0);
    ecs_input_set(&it, 1, 1, &v, true);

    EXPECT(ecs_input_frontier(&it) == 0ull,    "frontier not advanced on partial confirm");
    EXPECT(!ecs_input_tick_confirmed(&it, 1),  "tick 1 not confirmed -- pid 2 missing");

    ecs_input_destroy(&it);
}

/* --- ring wrap: tick T and T+buf_size occupy same slot ----------------- */

static void test_input_ring_wrap(void) {
    const uint32_t BUF = 8;
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), BUF);
    ecs_input_register_player(&it, 1);

    ti_input_t old_v = ti_make(0xDEAD, 1, 1);
    ti_input_t new_v = ti_make(0xBEEF, 9, 9);

    /* Confirm ticks 1..8 contiguously -> frontier == 8. */
    for (uint64_t t = 1; t <= BUF; t++) {
        ecs_input_set(&it, t, 1, &old_v, true);
    }
    EXPECT(ecs_input_frontier(&it) == (uint64_t)BUF,
                                              "frontier == buf_size after BUF confirms");

    /* Tick 9 aliases slot of tick 1; tick 1 at frontier so safe to evict. */
    ecs_input_set(&it, BUF + 1u, 1, &new_v, true);
    EXPECT(ti_eq(ti_get(&it, BUF + 1u, 1), new_v),
                                              "wrap: new tick bytes visible");
    EXPECT(ecs_input_tick_confirmed(&it, BUF + 1u),
                                              "wrap: new tick confirmed");
    EXPECT(!ecs_input_tick_confirmed(&it, 1), "wrap: old tick no longer confirmed at slot");

    ecs_input_destroy(&it);
}

/* --- clear semantics --------------------------------------------------- */

static void test_input_clear(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    ecs_input_register_player(&it, 1);

    ti_input_t v = ti_make(7, 7, 7);
    ecs_input_set(&it, 1, 1, &v, true);
    ecs_input_set(&it, 2, 1, &v, true);
    EXPECT(ecs_input_frontier(&it) == 2ull,    "frontier == 2 after 2 confirms");

    ecs_input_clear(&it, 2);
    ecs_input_view_t view = ecs_input_get_view(&it, 2, 1);
    EXPECT(!view.present && !view.confirmed,   "after clear: not present");
    EXPECT(!ecs_input_tick_confirmed(&it, 2),  "after clear: tick not confirmed");
    EXPECT(ecs_input_frontier(&it) == 1ull,    "frontier rewound to 1");

    ecs_input_clear(&it, 1);
    EXPECT(ecs_input_frontier(&it) == 0ull,    "frontier rewound to 0 (no frontier)");

    ecs_input_destroy(&it);
}

/* --- register/unregister cycle: ABA defense ---------------------------- */

static void test_input_register_unregister_cycle(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);

    EXPECT(ecs_input_register_player(&it, 100),  "register 100");
    EXPECT(!ecs_input_register_player(&it, 100), "double-register rejected");
    EXPECT(ecs_input_active_count(&it) == 1u,    "active still 1 after dup register");

    ti_input_t a = ti_make(0xA, 1, 1);
    ecs_input_set(&it, 1, 100, &a, true);
    EXPECT(ecs_input_tick_confirmed(&it, 1),     "tick 1 confirmed for old pid 100");

    ecs_input_unregister_player(&it, 100);
    EXPECT(ecs_input_active_count(&it) == 0u,    "active 0 after unregister");

    EXPECT(ecs_input_register_player(&it, 200),  "register 200 reuses freed col");
    ecs_input_view_t v = ecs_input_get_view(&it, 1, 200);
    EXPECT(!v.present,                            "ABA defense: tick 1 not present for new pid 200");
    EXPECT(!v.confirmed,                          "ABA defense: tick 1 not confirmed for new pid 200");

    ti_input_t b = ti_make(0xB, 2, 2);
    ecs_input_set(&it, 2, 200, &b, true);
    EXPECT(ti_eq(ti_get(&it, 2, 200), b),        "new pid bytes visible after reuse");

    ecs_input_destroy(&it);
}

/* --- determinism: same op stream -> identical state ------------------- */

static void test_input_deterministic_replay(void) {
    const uint32_t BUF = 32;
    ecs_input_t a; ecs_input_init(&a, sizeof(ti_input_t), BUF);
    ecs_input_t b; ecs_input_init(&b, sizeof(ti_input_t), BUF);

    const uint32_t pids[3] = { 50, 51, 52 };
    for (int i = 0; i < 3; i++) {
        ecs_input_register_player(&a, pids[i]);
        ecs_input_register_player(&b, pids[i]);
    }

    for (uint64_t t = 1; t <= 5; t++) {
        for (int i = 0; i < 3; i++) {
            ti_input_t v = ti_make((uint32_t)(t * 10 + i), (int16_t)t, (int16_t)i);
            bool conf = ((t + i) & 1u) != 0u;
            ecs_input_set(&a, t, pids[i], &v, conf);
            ecs_input_set(&b, t, pids[i], &v, conf);
        }
    }

    for (uint64_t t = 1; t <= 5; t++) {
        for (int i = 0; i < 3; i++) {
            EXPECT(ti_eq(ti_get(&a, t, pids[i]), ti_get(&b, t, pids[i])),
                                                "deterministic: bytes match");
        }
    }
    EXPECT(ecs_input_frontier(&a) == ecs_input_frontier(&b),
                                                "deterministic: frontier matches");

    ecs_input_destroy(&a);
    ecs_input_destroy(&b);
}

/* --- seed_frontier for mid-session join -------------------------------- */

static void test_input_seed_frontier(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 32);
    ecs_input_register_player(&it, 1);

    ecs_input_seed_frontier(&it, 1000);
    EXPECT(ecs_input_frontier(&it) == 1000ull, "seed sets frontier to 1000");

    ti_input_t v = ti_make(0, 0, 0);
    ecs_input_set(&it, 1001, 1, &v, true);
    EXPECT(ecs_input_frontier(&it) == 1001ull, "frontier extends past seeded value");

    ecs_input_destroy(&it);
}

/* --- predicted-after-confirmed dropped --------------------------------- */

static void test_input_predicted_after_confirmed_dropped(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    ecs_input_register_player(&it, 1);

    ti_input_t conf_v = ti_make(0xC0, 10, 10);
    ti_input_t pred_v = ti_make(0xFF, -1, -1);

    ecs_input_set(&it, 2, 1, &conf_v, true);
    EXPECT(ti_eq(ti_get(&it, 2, 1), conf_v),    "confirmed bytes recorded");

    ecs_input_set(&it, 2, 1, &pred_v, false);
    EXPECT(ti_eq(ti_get(&it, 2, 1), conf_v),    "predicted overwrite dropped");
    ecs_input_view_t view = ecs_input_get_view(&it, 2, 1);
    EXPECT(view.present && view.confirmed,      "flags unchanged");

    ecs_input_destroy(&it);
}

/* --- seal_empty_tick advances frontier without sets ------------------- */

static void test_input_seal_empty_tick(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);

    /* No players registered. Server seals ticks 1..3 as empty. */
    ecs_input_seal_empty_tick(&it, 1);
    EXPECT(ecs_input_frontier(&it) == 1ull,   "seal_empty(1) -> frontier 1");
    ecs_input_seal_empty_tick(&it, 2);
    EXPECT(ecs_input_frontier(&it) == 2ull,   "seal_empty(2) -> frontier 2");
    ecs_input_seal_empty_tick(&it, 3);
    EXPECT(ecs_input_frontier(&it) == 3ull,   "seal_empty(3) -> frontier 3");

    EXPECT(ecs_input_tick_confirmed(&it, 1),  "tick 1 reported confirmed");
    EXPECT(ecs_input_tick_confirmed(&it, 2),  "tick 2 reported confirmed");
    EXPECT(ecs_input_tick_confirmed(&it, 3),  "tick 3 reported confirmed");

    ecs_input_destroy(&it);
}

/* --- iterator visits live pids exactly once --------------------------- */

static void test_input_iterator(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    const uint32_t pids[5] = { 10, 20, 30, 40, 50 };
    for (int i = 0; i < 5; i++) ecs_input_register_player(&it, pids[i]);

    ecs_input_unregister_player(&it, 30);

    ecs_input_iter_t iter = ecs_input_iter_begin(&it);
    uint32_t seen[8] = {0};
    int n = 0;
    while (ecs_input_iter_next(&iter)) {
        EXPECT(n < 8, "iterator does not exceed expected size");
        seen[n++] = iter.pid;
    }
    EXPECT(n == 4,         "iterator visits 4 live entries");
    EXPECT(seen[0] == 10,  "iter[0] == 10");
    EXPECT(seen[1] == 20,  "iter[1] == 20");
    EXPECT(seen[2] == 40,  "iter[2] == 40");
    EXPECT(seen[3] == 50,  "iter[3] == 50");

    EXPECT(ecs_input_is_registered(&it, 10),    "10 registered");
    EXPECT(!ecs_input_is_registered(&it, 30),   "30 not registered after unregister");
    EXPECT(!ecs_input_is_registered(&it, 999),  "unknown pid not registered");

    ecs_input_destroy(&it);
}

/* --- player cap auto-grow --------------------------------------------- */

static void test_input_player_cap(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);

    EXPECT(it.active_cap == 0u, "active_cap starts at 0 (lazy alloc)");

    /* Register 17 players: forces grow past initial 16. */
    for (uint32_t i = 1; i <= 17u; i++) {
        EXPECT(ecs_input_register_player(&it, i),  "register grows player cap pow2");
    }
    EXPECT(it.active_cap >= 32u,               "active_cap doubled past 16");
    EXPECT(ecs_input_active_count(&it) == 17u, "active count tracks registers");

    ecs_input_grow_player_cap(&it, 128);
    EXPECT(it.active_cap >= 128u,              "preemptive grow rounds to >= request");

    uint32_t cap_before = it.active_cap;
    ecs_input_grow_player_cap(&it, 32);
    EXPECT(it.active_cap == cap_before,        "below-current grow is no-op");

    ti_input_t v = ti_make(0xCAFE, 7, 7);
    ecs_input_set(&it, 1, 5, &v, true);
    EXPECT(ti_eq(ti_get(&it, 1, 5), v),        "set/get works after grow");

    ecs_input_destroy(&it);
}

/* --- input persistence: predicted carries forward --------------------- */

static void test_input_persistence(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 32);
    ecs_input_register_player(&it, 1);
    ecs_input_register_player(&it, 2);

    ti_input_t a = ti_make(0xA, 1, 1);
    ti_input_t b = ti_make(0xB, 2, 2);

    /* Confirm tick 1 for both players. */
    ecs_input_set(&it, 1, 1, &a, true);
    ecs_input_set(&it, 1, 2, &b, true);
    EXPECT(ecs_input_frontier(&it) == 1ull,    "tick 1 confirmed");

    /* Predict tick 2 for pid 1 only; pid 2's prev bytes (b) carry forward
       as predicted via advance_row. */
    ti_input_t a2 = ti_make(0xAA, 5, 5);
    ecs_input_set(&it, 2, 1, &a2, false);

    EXPECT(ti_eq(ti_get(&it, 2, 1), a2),       "pid 1's predicted bytes");
    EXPECT(ti_eq(ti_get(&it, 2, 2), b),        "pid 2's bytes carried from tick 1");
    ecs_input_view_t v2 = ecs_input_get_view(&it, 2, 2);
    EXPECT(v2.present && !v2.confirmed,        "carried bytes marked predicted");

    ecs_input_destroy(&it);
}

/* --- auto-grow on burst of predictions -------------------------------- */

static void test_input_auto_grow_burst(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 8);
    ecs_input_register_player(&it, 1);

    /* Predict 600 ticks. Frontier stays 0 (no confirm), so every aliased
       slot is "live" -> auto-grow fires. */
    ti_input_t pred_v[600];
    for (uint64_t t = 1; t <= 600; t++) {
        pred_v[t-1] = ti_make((uint32_t)(t * 7u), (int16_t)t, (int16_t)(t & 0xff));
        ecs_input_set(&it, t, 1, &pred_v[t-1], false);
    }

    EXPECT(it.buf_size >= 1024u,
                                              "ring grew to >= 1024 during 600-tick predict burst");
    EXPECT(ti_eq(ti_get(&it, 1, 1), pred_v[0]),
                                              "tick 1 predicted bytes preserved");
    EXPECT(ti_eq(ti_get(&it, 600, 1), pred_v[599]),
                                              "tick 600 predicted bytes preserved");

    /* Confirmed burst: frontier walks to 600. */
    for (uint64_t t = 1; t <= 600; t++) {
        ti_input_t v = ti_make((uint32_t)t, (int16_t)t, 0);
        ecs_input_set(&it, t, 1, &v, true);
    }
    EXPECT(ecs_input_frontier(&it) == 600ull,
                                              "frontier reached 600 after confirm burst");

    ecs_input_destroy(&it);
}

/* --- ecs_input_grow_buf preemptive ----------------------------------- */

static void test_input_grow_buf_preemptive(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 4);
    ecs_input_register_player(&it, 1);

    ecs_input_grow_buf(&it, 64);
    EXPECT(it.buf_size == 64u, "preemptive grow rounds to pow2 >= request");

    ecs_input_grow_buf(&it, 8);
    EXPECT(it.buf_size == 64u, "below-current grow is no-op");

    ti_input_t v = ti_make(0x1234, 1, 1);
    ecs_input_set(&it, 50, 1, &v, true);
    EXPECT(ti_eq(ti_get(&it, 50, 1), v), "tick 50 set/get works in grown ring");

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
    RUN_TEST(test_input_seed_frontier);
    RUN_TEST(test_input_predicted_after_confirmed_dropped);
    RUN_TEST(test_input_seal_empty_tick);
    RUN_TEST(test_input_iterator);
    RUN_TEST(test_input_player_cap);
    RUN_TEST(test_input_persistence);
    RUN_TEST(test_input_auto_grow_burst);
    RUN_TEST(test_input_grow_buf_preemptive);
    int failed = g_failed - before;
    printf("\ninput: %d failed\n", failed);
    return failed ? 1 : 0;
}
