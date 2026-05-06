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
       roster (live_bm) AND confirmed bits.
     - ecs_input_set has no `expected` parameter; the live roster
       implicitly defines it.
     - Frontier is sim-driven. ecs_input_set never advances it; tests
       call ecs_input_advance_to_tick explicitly to model the sim.
     - The module is slot-keyed -- ecs_input_alloc_slot returns the
       slot index; tests use slots directly. Any pid <-> slot mapping
       lives outside this module. */

typedef struct {
    uint32_t buttons;
    int16_t  axis_x;
    int16_t  axis_y;
} ti_input_t;   /* 8 bytes */

static inline ti_input_t ti_make(uint32_t b, int16_t x, int16_t y) {
    ti_input_t v = { b, x, y };
    return v;
}

static inline ti_input_t ti_get(const ecs_input_t* it, uint32_t tick, uint32_t slot) {
    const void* p = ecs_input_get(it, tick, slot);
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

    uint32_t s = ecs_input_alloc_slot(&it);
    EXPECT(s != ECS_INPUT_SLOT_NIL,            "alloc slot ok");
    EXPECT(ecs_input_active_count(&it) == 1u,  "active count 1 after alloc");
    EXPECT(ecs_input_slot_is_live(&it, s),     "alloc'd slot reports live");

    ti_input_t v = ti_make(0xABCD, 100, -50);
    ecs_input_set(&it, 3, s, &v, true);

    EXPECT(ti_eq(ti_get(&it, 3, s), v),        "get returns what set wrote");
    ecs_input_view_t view = ecs_input_get_view(&it, 3, s);
    EXPECT(view.data != NULL,                  "view has data ptr");
    EXPECT(view.present,                       "view marks present");
    EXPECT(view.confirmed,                     "view marks confirmed");

    ecs_input_destroy(&it);
}

/* --- get on unknown slot returns NULL ---------------------------------- */

static void test_input_get_unknown(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);

    EXPECT(ecs_input_get(&it, 1, 0) == NULL,                     "get on dead slot 0 -> NULL");
    EXPECT(ecs_input_get(&it, 1, 12345) == NULL,                 "get on out-of-range slot -> NULL");
    ecs_input_view_t v = ecs_input_get_view(&it, 1, 0);
    EXPECT(v.data == NULL && !v.present && !v.confirmed,         "view dead slot -> empty");

    ti_input_t in = ti_make(1, 2, 3);
    ecs_input_set(&it, 5, 999, &in, true);
    EXPECT(ecs_input_get(&it, 5, 999) == NULL,                   "set on dead slot did nothing");

    ecs_input_destroy(&it);
}

/* --- idempotent confirm (same value, twice) ---------------------------- */

static void test_input_set_idempotent_same_value(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 32);
    uint32_t s1 = ecs_input_alloc_slot(&it);
    uint32_t s2 = ecs_input_alloc_slot(&it);

    ti_input_t a = ti_make(0x10, 5, 5);
    ti_input_t b = ti_make(0x20, -1, 1);

    ecs_input_set(&it, 1, s1, &a, true);
    EXPECT(!ecs_input_tick_confirmed(&it, 1),   "1 of 2 confirmed -> tick not yet confirmed");
    ecs_input_set(&it, 1, s2, &b, true);
    EXPECT(ecs_input_tick_confirmed(&it, 1),    "2 of 2 confirmed -> tick confirmed");
    EXPECT(ecs_input_frontier(&it) == 0ull,     "frontier untouched -- sim drives it");

    /* Replay: must not double-confirm. */
    ecs_input_set(&it, 1, s1, &a, true);
    ecs_input_set(&it, 1, s2, &b, true);
    EXPECT(ecs_input_tick_confirmed(&it, 1),    "tick still confirmed after replay");
    EXPECT(ti_eq(ti_get(&it, 1, s1), a),        "slot 1 bytes preserved on replay");
    EXPECT(ti_eq(ti_get(&it, 1, s2), b),        "slot 2 bytes preserved on replay");

    ecs_input_destroy(&it);
}

/* --- predicted then confirmed (overwrite with new bytes) --------------- */

static void test_input_set_overwrite(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    uint32_t s = ecs_input_alloc_slot(&it);

    ti_input_t v0 = ti_make(0xAA, 1, 2);
    ti_input_t v1 = ti_make(0xBB, 9, 9);

    ecs_input_set(&it, 1, s, &v0, false);
    ecs_input_view_t view = ecs_input_get_view(&it, 1, s);
    EXPECT(view.present && !view.confirmed,    "predicted set marks present, not confirmed");
    EXPECT(ti_eq(ti_get(&it, 1, s), v0),       "predicted bytes visible");

    ecs_input_set(&it, 1, s, &v1, true);
    view = ecs_input_get_view(&it, 1, s);
    EXPECT(view.confirmed && view.present,     "confirmed set marks both flags");
    EXPECT(ti_eq(ti_get(&it, 1, s), v1),       "bytes overwritten by confirmed packet");

    ecs_input_destroy(&it);
}

/* --- multi-packet split, any order ------------------------------------- */

static void test_input_multi_packet_split(void) {
    const ti_input_t vals[4] = {
        { 0x1, 1, 1 }, { 0x2, 2, 2 }, { 0x3, 3, 3 }, { 0x4, 4, 4 },
    };

    ecs_input_t a; ecs_input_init(&a, sizeof(ti_input_t), 16);
    ecs_input_t b; ecs_input_init(&b, sizeof(ti_input_t), 16);
    uint32_t sa[4], sb[4];
    for (int i = 0; i < 4; i++) {
        sa[i] = ecs_input_alloc_slot(&a);
        sb[i] = ecs_input_alloc_slot(&b);
        EXPECT(sa[i] == sb[i],                "slots aligned across peers (parallel allocs)");
    }

    for (int i = 0; i < 4; i++) ecs_input_set(&a, 1, sa[i], &vals[i], true);
    for (int i = 3; i >= 0; i--) ecs_input_set(&b, 1, sb[i], &vals[i], true);

    EXPECT(ecs_input_tick_confirmed(&a, 1),   "a: tick confirmed");
    EXPECT(ecs_input_tick_confirmed(&b, 1),   "b: tick confirmed (reverse order)");
    for (int i = 0; i < 4; i++) {
        EXPECT(ti_eq(ti_get(&a, 1, sa[i]), ti_get(&b, 1, sb[i])),
                                              "per-slot bytes match across orderings");
    }

    ecs_input_destroy(&a);
    ecs_input_destroy(&b);
}

/* --- frontier is sim-driven, not auto-tracked -------------------------- */

static void test_input_frontier_sim_driven(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 32);
    uint32_t s = ecs_input_alloc_slot(&it);
    ti_input_t v = ti_make(0, 0, 0);

    ecs_input_set(&it, 1, s, &v, true);
    ecs_input_set(&it, 2, s, &v, true);
    ecs_input_set(&it, 3, s, &v, true);
    EXPECT(ecs_input_frontier(&it) == 0ull,    "set never moves frontier");

    /* Sim decides ticks 1..3 are done. */
    ecs_input_advance_to_tick(&it, 3);
    EXPECT(ecs_input_frontier(&it) == 3ull,    "advance_to_tick(3) -> frontier 3");

    ecs_input_advance_to_tick(&it, 5);
    EXPECT(ecs_input_frontier(&it) == 5ull,    "advance is monotonic, can skip ticks");

    ecs_input_destroy(&it);
}

/* --- past-frontier ticks are vacuously confirmed ----------------------- */

static void test_input_past_frontier_confirmed(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 8);
    uint32_t s1 = ecs_input_alloc_slot(&it);
    uint32_t s2 = ecs_input_alloc_slot(&it);

    /* Tick 5 partial-confirmed (only s1). Without frontier advance,
       tick_confirmed must report false. */
    ti_input_t v = ti_make(0xAA, 1, 1);
    ecs_input_set(&it, 5, s1, &v, true);
    EXPECT(!ecs_input_tick_confirmed(&it, 5),    "partial tick: not confirmed pre-frontier");

    /* Sim seals tick 10. Now ticks 1..10 must report confirmed even
       though s2 was never written and bitmaps disagree. */
    ecs_input_advance_to_tick(&it, 10);
    EXPECT(ecs_input_tick_confirmed(&it, 5),     "tick 5 <= frontier: confirmed by seal");
    EXPECT(ecs_input_tick_confirmed(&it, 10),    "tick 10 == frontier: confirmed");
    EXPECT(ecs_input_tick_confirmed(&it, 1),     "tick 1 <= frontier: confirmed even if never written");
    EXPECT(!ecs_input_tick_confirmed(&it, 11),   "tick 11 > frontier: still gated by bitmap");

    /* Evicted slot: write tick 100 -> aliases slot of tick 100 % 8 = 4.
       Old occupants of any slot whose stored tick <= frontier are evictable.
       After this write, tick 5 may or may not still be resident depending
       on slot, but tick_confirmed(5) must still return true since 5 <= frontier. */
    ti_input_t w = ti_make(0xBB, 2, 2);
    ecs_input_set(&it, 100, s1, &w, true);
    ecs_input_set(&it, 100, s2, &w, true);
    EXPECT(ecs_input_tick_confirmed(&it, 5),     "evicted past-frontier tick still reports confirmed");
    EXPECT(ecs_input_tick_confirmed(&it, 3),     "never-written past-frontier tick reports confirmed");

    /* Tick 0 is the "no frontier" sentinel and must NOT be claimed confirmed. */
    EXPECT(!ecs_input_tick_confirmed(&it, 0),    "tick 0 sentinel: not confirmed");

    ecs_input_destroy(&it);
}

/* --- partial confirm reports tick as not-confirmed --------------------- */

static void test_input_partial_no_advance(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    uint32_t s1 = ecs_input_alloc_slot(&it);
    (void)ecs_input_alloc_slot(&it);

    ti_input_t v = ti_make(0, 0, 0);
    ecs_input_set(&it, 1, s1, &v, true);

    EXPECT(!ecs_input_tick_confirmed(&it, 1),  "tick 1 not confirmed -- second slot missing");

    ecs_input_destroy(&it);
}

/* --- ring wrap: tick T and T+buf_size occupy same slot ----------------- */

static void test_input_ring_wrap(void) {
    const uint32_t BUF = 8;
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), BUF);
    uint32_t s = ecs_input_alloc_slot(&it);

    ti_input_t old_v = ti_make(0xDEAD, 1, 1);
    ti_input_t new_v = ti_make(0xBEEF, 9, 9);

    /* Confirm ticks 1..8; sim then advances frontier past them. */
    for (uint32_t t = 1; t <= BUF; t++) {
        ecs_input_set(&it, t, s, &old_v, true);
    }
    ecs_input_advance_to_tick(&it, BUF);
    EXPECT(ecs_input_frontier(&it) == (uint64_t)BUF,
                                              "frontier == buf_size after sim advance");

    /* Tick 9 aliases slot of tick 1; tick 1 at frontier so safe to evict. */
    ecs_input_set(&it, BUF + 1u, s, &new_v, true);
    EXPECT(ti_eq(ti_get(&it, BUF + 1u, s), new_v),
                                              "wrap: new tick bytes visible");
    EXPECT(ecs_input_tick_confirmed(&it, BUF + 1u),
                                              "wrap: new tick confirmed");
    /* Old tick row was evicted but tick 1 <= frontier, so it is sealed
       and tick_confirmed reports true (sim authority overrides bitmap). */
    EXPECT(ecs_input_tick_confirmed(&it, 1),  "wrap: evicted past-frontier tick still confirmed by seal");
    /* Data eviction is unchanged: ti_get returns NULL because the slot
       no longer holds tick 1. */
    EXPECT(ecs_input_get(&it, 1, s) == NULL,  "wrap: evicted tick no longer resident in its slot");

    ecs_input_destroy(&it);
}

/* --- clear semantics --------------------------------------------------- */

static void test_input_clear(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    uint32_t s = ecs_input_alloc_slot(&it);

    ti_input_t v = ti_make(7, 7, 7);
    ecs_input_set(&it, 1, s, &v, true);
    ecs_input_set(&it, 2, s, &v, true);

    ecs_input_clear(&it, 2);
    ecs_input_view_t view = ecs_input_get_view(&it, 2, s);
    EXPECT(!view.present && !view.confirmed,   "after clear: not present");
    EXPECT(!ecs_input_tick_confirmed(&it, 2),  "after clear: tick not confirmed");
    EXPECT(ecs_input_frontier(&it) == 0ull,    "clear does NOT touch frontier");

    /* Tick 1 still intact since clear only resets one row. */
    EXPECT(ti_eq(ti_get(&it, 1, s), v),        "untouched tick survives clear");

    ecs_input_destroy(&it);
}

/* --- alloc/free cycle: ABA defense ------------------------------------- */

static void test_input_alloc_free_cycle(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);

    uint32_t s = ecs_input_alloc_slot(&it);
    EXPECT(s != ECS_INPUT_SLOT_NIL,              "alloc ok");
    EXPECT(ecs_input_active_count(&it) == 1u,    "active 1 after alloc");
    EXPECT(ecs_input_slot_is_live(&it, s),       "slot reports live");

    ti_input_t a = ti_make(0xA, 1, 1);
    ecs_input_set(&it, 1, s, &a, true);
    EXPECT(ecs_input_tick_confirmed(&it, 1),     "tick 1 confirmed for first owner");

    ecs_input_free_slot(&it, s);
    EXPECT(ecs_input_active_count(&it) == 0u,    "active 0 after free");
    EXPECT(!ecs_input_slot_is_live(&it, s),      "freed slot not live");

    /* Free of out-of-range / already-freed slot is a no-op. */
    ecs_input_free_slot(&it, s);
    EXPECT(ecs_input_active_count(&it) == 0u,    "double-free is no-op");
    ecs_input_free_slot(&it, 99999);
    EXPECT(ecs_input_active_count(&it) == 0u,    "out-of-range free is no-op");

    uint32_t s2 = ecs_input_alloc_slot(&it);
    EXPECT(s2 == s,                              "second alloc reuses lowest free slot");
    ecs_input_view_t v = ecs_input_get_view(&it, 1, s2);
    EXPECT(!v.present,                           "ABA defense: tick 1 not present for new owner");
    EXPECT(!v.confirmed,                         "ABA defense: tick 1 not confirmed for new owner");

    ti_input_t b = ti_make(0xB, 2, 2);
    ecs_input_set(&it, 2, s2, &b, true);
    EXPECT(ti_eq(ti_get(&it, 2, s2), b),         "new-owner bytes visible after reuse");

    ecs_input_destroy(&it);
}

/* --- determinism: same op stream -> identical state ------------------- */

static void test_input_deterministic_replay(void) {
    const uint32_t BUF = 32;
    ecs_input_t a; ecs_input_init(&a, sizeof(ti_input_t), BUF);
    ecs_input_t b; ecs_input_init(&b, sizeof(ti_input_t), BUF);

    uint32_t sa[3], sb[3];
    for (int i = 0; i < 3; i++) {
        sa[i] = ecs_input_alloc_slot(&a);
        sb[i] = ecs_input_alloc_slot(&b);
    }

    for (uint32_t t = 1; t <= 5; t++) {
        for (int i = 0; i < 3; i++) {
            ti_input_t v = ti_make((uint32_t)(t * 10 + i), (int16_t)t, (int16_t)i);
            bool conf = ((t + i) & 1u) != 0u;
            ecs_input_set(&a, t, sa[i], &v, conf);
            ecs_input_set(&b, t, sb[i], &v, conf);
        }
    }

    for (uint32_t t = 1; t <= 5; t++) {
        for (int i = 0; i < 3; i++) {
            EXPECT(ti_eq(ti_get(&a, t, sa[i]), ti_get(&b, t, sb[i])),
                                                "deterministic: bytes match");
        }
    }
    /* Frontier never advances on its own, so both stay 0. */
    EXPECT(ecs_input_frontier(&a) == 0ull && ecs_input_frontier(&b) == 0ull,
                                                "deterministic: both frontiers untouched");

    ecs_input_destroy(&a);
    ecs_input_destroy(&b);
}

/* --- advance_to_tick for mid-session join ------------------------------ */

static void test_input_advance_to_tick(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 32);
    uint32_t s = ecs_input_alloc_slot(&it);

    ecs_input_advance_to_tick(&it, 1000);
    EXPECT(ecs_input_frontier(&it) == 1000ull, "advance_to_tick(1000) sets frontier");

    ti_input_t v = ti_make(0, 0, 0);
    ecs_input_set(&it, 1001, s, &v, true);
    EXPECT(ecs_input_frontier(&it) == 1000ull, "set does not move frontier");

    ecs_input_advance_to_tick(&it, 1001);
    EXPECT(ecs_input_frontier(&it) == 1001ull, "subsequent advance bumps frontier");

    ecs_input_destroy(&it);
}

/* --- predicted-after-confirmed dropped --------------------------------- */

static void test_input_predicted_after_confirmed_dropped(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    uint32_t s = ecs_input_alloc_slot(&it);

    ti_input_t conf_v = ti_make(0xC0, 10, 10);
    ti_input_t pred_v = ti_make(0xFF, -1, -1);

    ecs_input_set(&it, 2, s, &conf_v, true);
    EXPECT(ti_eq(ti_get(&it, 2, s), conf_v),    "confirmed bytes recorded");

    ecs_input_set(&it, 2, s, &pred_v, false);
    EXPECT(ti_eq(ti_get(&it, 2, s), conf_v),    "predicted overwrite dropped");
    ecs_input_view_t view = ecs_input_get_view(&it, 2, s);
    EXPECT(view.present && view.confirmed,      "flags unchanged");

    ecs_input_destroy(&it);
}

/* --- empty roster: tick_confirmed vacuously true ----------------------- */

static void test_input_empty_roster(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);

    /* No slots allocated -- every tick reports confirmed (vacuous). */
    EXPECT(ecs_input_tick_confirmed(&it, 1),  "no roster -> tick 1 vacuously confirmed");
    EXPECT(ecs_input_tick_confirmed(&it, 42), "no roster -> arbitrary tick confirmed");

    ecs_input_destroy(&it);
}

/* --- iterator visits live slots exactly once -------------------------- */

static void test_input_iterator(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);
    uint32_t s[5];
    for (int i = 0; i < 5; i++) s[i] = ecs_input_alloc_slot(&it);

    /* Free middle slot: hole left at s[2]. */
    ecs_input_free_slot(&it, s[2]);

    ecs_input_iter_t iter = ecs_input_iter_begin(&it);
    uint32_t seen[8] = {0};
    int n = 0;
    while (ecs_input_iter_next(&iter)) {
        EXPECT(n < 8, "iterator does not exceed expected size");
        seen[n++] = iter.slot;
    }
    EXPECT(n == 4,            "iterator visits 4 live slots");
    EXPECT(seen[0] == s[0],   "iter[0] == s[0]");
    EXPECT(seen[1] == s[1],   "iter[1] == s[1]");
    EXPECT(seen[2] == s[3],   "iter[2] == s[3]");
    EXPECT(seen[3] == s[4],   "iter[3] == s[4]");

    EXPECT(ecs_input_slot_is_live(&it, s[0]),    "s[0] live");
    EXPECT(!ecs_input_slot_is_live(&it, s[2]),   "s[2] not live after free");
    EXPECT(!ecs_input_slot_is_live(&it, 999),    "out-of-range slot not live");

    ecs_input_destroy(&it);
}

/* --- player cap auto-grow --------------------------------------------- */

static void test_input_player_cap(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 16);

    EXPECT(it.active_cap == 0u, "active_cap starts at 0 (lazy alloc)");

    /* Alloc 17 slots: forces grow past initial 16. */
    uint32_t slots[17];
    for (uint32_t i = 0; i < 17u; i++) {
        slots[i] = ecs_input_alloc_slot(&it);
        EXPECT(slots[i] != ECS_INPUT_SLOT_NIL,  "alloc grows player cap pow2");
    }
    EXPECT(it.active_cap >= 32u,               "active_cap doubled past 16");
    EXPECT(ecs_input_active_count(&it) == 17u, "active count tracks allocs");

    ecs_input_grow_player_cap(&it, 128);
    EXPECT(it.active_cap >= 128u,              "preemptive grow rounds to >= request");

    uint32_t cap_before = it.active_cap;
    ecs_input_grow_player_cap(&it, 32);
    EXPECT(it.active_cap == cap_before,        "below-current grow is no-op");

    ti_input_t v = ti_make(0xCAFE, 7, 7);
    ecs_input_set(&it, 1, slots[5], &v, true);
    EXPECT(ti_eq(ti_get(&it, 1, slots[5]), v), "set/get works after grow");

    ecs_input_destroy(&it);
}

/* --- input persistence: predicted carries forward --------------------- */

static void test_input_persistence(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 32);
    uint32_t s1 = ecs_input_alloc_slot(&it);
    uint32_t s2 = ecs_input_alloc_slot(&it);

    ti_input_t a = ti_make(0xA, 1, 1);
    ti_input_t b = ti_make(0xB, 2, 2);

    /* Confirm tick 1 for both slots. */
    ecs_input_set(&it, 1, s1, &a, true);
    ecs_input_set(&it, 1, s2, &b, true);
    EXPECT(ecs_input_tick_confirmed(&it, 1),   "tick 1 confirmed");

    /* Predict tick 2 for s1 only; s2's prev bytes (b) carry forward
       as predicted via advance_row. */
    ti_input_t a2 = ti_make(0xAA, 5, 5);
    ecs_input_set(&it, 2, s1, &a2, false);

    EXPECT(ti_eq(ti_get(&it, 2, s1), a2),      "s1 predicted bytes");
    EXPECT(ti_eq(ti_get(&it, 2, s2), b),       "s2 bytes carried from tick 1");
    ecs_input_view_t v2 = ecs_input_get_view(&it, 2, s2);
    EXPECT(v2.present && !v2.confirmed,        "carried bytes marked predicted");

    ecs_input_destroy(&it);
}

/* --- auto-grow on burst of predictions -------------------------------- */

static void test_input_auto_grow_burst(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 8);
    uint32_t s = ecs_input_alloc_slot(&it);

    /* Predict 600 ticks. Frontier stays 0 (no confirm), so every aliased
       slot is "live" -> auto-grow fires. */
    ti_input_t pred_v[600];
    for (uint32_t t = 1; t <= 600; t++) {
        pred_v[t-1] = ti_make((uint32_t)(t * 7u), (int16_t)t, (int16_t)(t & 0xff));
        ecs_input_set(&it, t, s, &pred_v[t-1], false);
    }

    EXPECT(it.buf_size >= 1024u,
                                              "ring grew to >= 1024 during 600-tick predict burst");
    EXPECT(ti_eq(ti_get(&it, 1, s), pred_v[0]),
                                              "tick 1 predicted bytes preserved");
    EXPECT(ti_eq(ti_get(&it, 600, s), pred_v[599]),
                                              "tick 600 predicted bytes preserved");

    /* Confirmed burst, then sim advances frontier. */
    for (uint32_t t = 1; t <= 600; t++) {
        ti_input_t v = ti_make((uint32_t)t, (int16_t)t, 0);
        ecs_input_set(&it, t, s, &v, true);
    }
    ecs_input_advance_to_tick(&it, 600);
    EXPECT(ecs_input_frontier(&it) == 600ull,
                                              "sim advanced frontier to 600");

    /* All 600 ticks must report confirmed and yield the confirm-pass bytes
       (NOT the earlier predicted bytes). */
    bool all_conf = true;
    bool all_bytes = true;
    bool no_pred_leak = true;
    for (uint32_t t = 1; t <= 600; t++) {
        if (!ecs_input_tick_confirmed(&it, t)) { all_conf = false; break; }
        ti_input_t expect_v = ti_make((uint32_t)t, (int16_t)t, 0);
        ti_input_t got = ti_get(&it, t, s);
        if (!ti_eq(got, expect_v))      { all_bytes = false; break; }
        if (ti_eq(got, pred_v[t-1]))    { no_pred_leak = false; break; }
    }
    EXPECT(all_conf,                          "every tick 1..600 reports confirmed");
    EXPECT(all_bytes,                         "every tick 1..600 returns confirm-pass bytes");
    EXPECT(no_pred_leak,                      "predicted bytes overwritten by confirm pass");

    /* view flags consistent on a spot-check tick. */
    ecs_input_view_t mid = ecs_input_get_view(&it, 300, s);
    EXPECT(mid.present && mid.confirmed,      "mid-burst view: present + confirmed");
    EXPECT(mid.data != NULL,                  "mid-burst view: data ptr non-null");

    /* Frontier must be untouched by all the prior set() calls. */
    EXPECT(ecs_input_frontier(&it) == 600ull, "frontier still 600 (set never moves it)");

    /* Past-frontier write into a slot whose old tick is now <= frontier
       must succeed without further ring growth. tick 700 aliases slot of
       tick (700 % buf_size); old occupant is evictable. */
    uint32_t buf_before = it.buf_size;
    ti_input_t fresh = ti_make(0xFEEDF00Du, 1234, -1234);
    ecs_input_set(&it, 700, s, &fresh, true);
    EXPECT(it.buf_size == buf_before,         "no extra grow when victim slot <= frontier");
    EXPECT(ti_eq(ti_get(&it, 700, s), fresh), "tick 700 confirmed bytes visible");
    EXPECT(ecs_input_tick_confirmed(&it, 700),"tick 700 reports confirmed");

    /* Old tick whose ring slot was just stolen is no longer the row's
       resident -- ti_get returns NULL because tick_in_slot != requested. */
    uint64_t evicted = 700ull - (uint64_t)it.buf_size;
    if (evicted >= 1ull && evicted <= 600ull) {
        EXPECT(ecs_input_get(&it, evicted, s) == NULL,
                                              "evicted tick no longer resident in its slot");
    }

    ecs_input_destroy(&it);
}

/* --- ecs_input_grow_buf preemptive ----------------------------------- */

static void test_input_grow_buf_preemptive(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 4);
    uint32_t s = ecs_input_alloc_slot(&it);

    ecs_input_grow_buf(&it, 64);
    EXPECT(it.buf_size == 64u, "preemptive grow rounds to pow2 >= request");

    ecs_input_grow_buf(&it, 8);
    EXPECT(it.buf_size == 64u, "below-current grow is no-op");

    ti_input_t v = ti_make(0x1234, 1, 1);
    ecs_input_set(&it, 50, s, &v, true);
    EXPECT(ti_eq(ti_get(&it, 50, s), v), "tick 50 set/get works in grown ring");

    ecs_input_destroy(&it);
}

/* --- grow consistency: active_cap (player dimension) ------------------ */

static void test_input_grow_player_cap_consistency(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 32);

    uint32_t slots[5];
    for (int i = 0; i < 5; i++) slots[i] = ecs_input_alloc_slot(&it);

    EXPECT(it.active_cap == 16u, "initial cap is ECS_INPUT_PLAYER_CAP_INIT (16)");

    /* Populate ticks 1..4 with mixed predicted/confirmed. Tick 3 is
       intentionally partial so tick_confirmed must report false. */
    ti_input_t snap_data[4][5];
    bool snap_present[4][5];
    bool snap_confirmed[4][5];
    bool snap_tick_conf[4];

    for (uint32_t t = 1; t <= 4; t++) {
        bool all_conf = true;
        for (int i = 0; i < 5; i++) {
            ti_input_t v = ti_make((uint32_t)(t * 100u + i),
                                   (int16_t)(t + i), (int16_t)(t * i));
            bool conf = !(t == 3 && i >= 3);
            ecs_input_set(&it, t, slots[i], &v, conf);
            snap_data[t-1][i]      = v;
            snap_present[t-1][i]   = true;
            snap_confirmed[t-1][i] = conf;
            if (!conf) all_conf = false;
        }
        snap_tick_conf[t-1] = all_conf;
    }

    /* Sanity on snapshot itself before any grow. */
    for (uint32_t t = 1; t <= 4; t++) {
        EXPECT(ecs_input_tick_confirmed(&it, t) == snap_tick_conf[t-1],
                                                  "snapshot: tick_confirmed matches expected");
    }

    /* Grow active_cap: triggers row-width realloc + per-row remap. */
    ecs_input_grow_player_cap(&it, 64);
    EXPECT(it.active_cap == 64u,                  "active_cap grown to 64");

    for (uint32_t t = 1; t <= 4; t++) {
        EXPECT(ecs_input_tick_confirmed(&it, t) == snap_tick_conf[t-1],
                                                  "tick_confirmed preserved across active_cap grow");
        for (int i = 0; i < 5; i++) {
            ecs_input_view_t v = ecs_input_get_view(&it, t, slots[i]);
            EXPECT(v.present   == snap_present[t-1][i],
                                                  "view.present preserved across active_cap grow");
            EXPECT(v.confirmed == snap_confirmed[t-1][i],
                                                  "view.confirmed preserved across active_cap grow");
            EXPECT(ti_eq(ti_get(&it, t, slots[i]), snap_data[t-1][i]),
                                                  "bytes preserved across active_cap grow");
        }
    }

    /* Second grow: re-runs the same remap with already-grown layout. */
    ecs_input_grow_player_cap(&it, 256);
    EXPECT(it.active_cap == 256u,                 "active_cap grown to 256");
    for (uint32_t t = 1; t <= 4; t++) {
        EXPECT(ecs_input_tick_confirmed(&it, t) == snap_tick_conf[t-1],
                                                  "tick_confirmed preserved across second grow");
        for (int i = 0; i < 5; i++) {
            EXPECT(ti_eq(ti_get(&it, t, slots[i]), snap_data[t-1][i]),
                                                  "bytes preserved across second grow");
        }
    }

    /* Alloc a new slot post-grow; must not disturb prior state. */
    uint32_t fresh_s = ecs_input_alloc_slot(&it);
    EXPECT(fresh_s != ECS_INPUT_SLOT_NIL,         "alloc new slot after grow");
    ti_input_t fresh = ti_make(0xFEED, 5, 5);
    ecs_input_set(&it, 5, fresh_s, &fresh, true);
    EXPECT(ti_eq(ti_get(&it, 5, fresh_s), fresh), "new-slot bytes visible after alloc-post-grow");
    for (int i = 0; i < 5; i++) {
        EXPECT(ti_eq(ti_get(&it, 1, slots[i]), snap_data[0][i]),
                                                  "old-slot data intact after new-slot alloc");
    }

    ecs_input_destroy(&it);
}

/* --- grow consistency: buf_size (ring dimension) ----------------------- */

static void test_input_grow_buf_consistency(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 8);

    uint32_t slots[3];
    for (int i = 0; i < 3; i++) slots[i] = ecs_input_alloc_slot(&it);

    /* Fill entire ring (ticks 1..8). Frontier stays at 0 so every slot
       counts as "live" — auto-grow on first-touch must NOT be triggered
       inside this loop because we only write within the existing ring. */
    ti_input_t snap_data[8][3];
    bool snap_present[8][3];
    bool snap_confirmed[8][3];
    bool snap_tick_conf[8];

    for (uint32_t t = 1; t <= 8; t++) {
        bool all_conf = true;
        for (int i = 0; i < 3; i++) {
            ti_input_t v = ti_make((uint32_t)(t * 11u + i),
                                   (int16_t)(t * 2), (int16_t)(i + 1));
            bool conf = !(t == 5 && i == 2);   /* tick 5 partial-confirmed */
            ecs_input_set(&it, t, slots[i], &v, conf);
            snap_data[t-1][i]      = v;
            snap_present[t-1][i]   = true;
            snap_confirmed[t-1][i] = conf;
            if (!conf) all_conf = false;
        }
        snap_tick_conf[t-1] = all_conf;
    }

    EXPECT(it.buf_size == 8u,                     "ring at original size before grow");

    /* Grow buf_size: triggers row-count realloc + per-row remap under wider mask. */
    ecs_input_grow_buf(&it, 32);
    EXPECT(it.buf_size == 32u,                    "ring grown to 32");

    /* All ticks 1..8 still readable; their slot indices changed only
       for tick 8 (old slot 0, new slot 8) but content must be identical. */
    for (uint32_t t = 1; t <= 8; t++) {
        EXPECT(ecs_input_tick_confirmed(&it, t) == snap_tick_conf[t-1],
                                                  "tick_confirmed preserved across buf_size grow");
        for (int i = 0; i < 3; i++) {
            ecs_input_view_t v = ecs_input_get_view(&it, t, slots[i]);
            EXPECT(v.present   == snap_present[t-1][i],
                                                  "view.present preserved across buf_size grow");
            EXPECT(v.confirmed == snap_confirmed[t-1][i],
                                                  "view.confirmed preserved across buf_size grow");
            EXPECT(ti_eq(ti_get(&it, t, slots[i]), snap_data[t-1][i]),
                                                  "bytes preserved across buf_size grow");
        }
    }

    /* Second grow. */
    ecs_input_grow_buf(&it, 128);
    EXPECT(it.buf_size == 128u,                   "ring grown to 128");
    for (uint32_t t = 1; t <= 8; t++) {
        EXPECT(ecs_input_tick_confirmed(&it, t) == snap_tick_conf[t-1],
                                                  "tick_confirmed preserved across second grow");
        for (int i = 0; i < 3; i++) {
            EXPECT(ti_eq(ti_get(&it, t, slots[i]), snap_data[t-1][i]),
                                                  "bytes preserved across second grow");
        }
    }

    /* Far-future write into newly available slots works. */
    ti_input_t v = ti_make(0xDADA, 9, 9);
    ecs_input_set(&it, 100, slots[0], &v, true);
    EXPECT(ti_eq(ti_get(&it, 100, slots[0]), v),  "post-grow far-tick write works");
    /* Old data still intact. */
    for (uint32_t t = 1; t <= 8; t++) {
        for (int i = 0; i < 3; i++) {
            EXPECT(ti_eq(ti_get(&it, t, slots[i]), snap_data[t-1][i]),
                                                  "old data intact after far-tick write");
        }
    }

    ecs_input_destroy(&it);
}

/* --- grow consistency: combined (cap then buf) ------------------------- */

static void test_input_grow_combined_consistency(void) {
    ecs_input_t it; ecs_input_init(&it, sizeof(ti_input_t), 8);

    uint32_t slots[4];
    for (int i = 0; i < 4; i++) slots[i] = ecs_input_alloc_slot(&it);

    ti_input_t snap_data[6][4];
    bool snap_confirmed[6][4];
    for (uint32_t t = 1; t <= 6; t++) {
        for (int i = 0; i < 4; i++) {
            ti_input_t v = ti_make((uint32_t)(t * 1000u + i * 7u),
                                   (int16_t)(t - i), (int16_t)(t + i));
            bool conf = ((t + i) & 1u) == 0u;
            ecs_input_set(&it, t, slots[i], &v, conf);
            snap_data[t-1][i] = v;
            snap_confirmed[t-1][i] = conf;
        }
    }

    /* Interleaved grows on both axes. */
    ecs_input_grow_player_cap(&it, 64);
    ecs_input_grow_buf(&it, 64);
    ecs_input_grow_player_cap(&it, 128);
    ecs_input_grow_buf(&it, 256);

    for (uint32_t t = 1; t <= 6; t++) {
        for (int i = 0; i < 4; i++) {
            ecs_input_view_t v = ecs_input_get_view(&it, t, slots[i]);
            EXPECT(v.present,                     "present preserved through combined grows");
            EXPECT(v.confirmed == snap_confirmed[t-1][i],
                                                  "confirmed preserved through combined grows");
            EXPECT(ti_eq(ti_get(&it, t, slots[i]), snap_data[t-1][i]),
                                                  "bytes preserved through combined grows");
        }
    }

    ecs_input_destroy(&it);
}

/* --- serialize/deserialize round-trip --------------------------------- */

static void test_input_serialize_roundtrip(void) {
    /* Two peers. Mirror past ticks so cascade refs resolve identically
       on both sides. Slots aligned via parallel allocs. */
    ecs_input_t a, b;
    ecs_input_init(&a, sizeof(ti_input_t), 32);
    ecs_input_init(&b, sizeof(ti_input_t), 32);

    uint32_t sa[4], sb[4];
    for (int i = 0; i < 4; i++) {
        sa[i] = ecs_input_alloc_slot(&a);
        sb[i] = ecs_input_alloc_slot(&b);
        EXPECT(sa[i] == sb[i],                "slots aligned across peers");
    }

    ti_input_t v1 = ti_make(0xAA, 1, 1);
    ti_input_t v2 = ti_make(0xBB, 2, 2);
    ti_input_t v3 = ti_make(0xCC, 3, 3);
    ti_input_t v4 = ti_make(0xDD, 4, 4);

    /* Both peers know tick 10. */
    ti_input_t* vs[4] = { &v1, &v2, &v3, &v4 };
    for (int i = 0; i < 4; i++) {
        ecs_input_set(&a, 10, sa[i], vs[i], true);
        ecs_input_set(&b, 10, sb[i], vs[i], true);
    }

    /* Tick 11 on sender:
         slot 0: same as tick 10  -> cascade match (1+1 bit)
         slot 1: zero              -> baseline match (1 bit)
         slot 2: new payload       -> all-1 bits + raw stride bytes
         slot 3: same as tick 10   -> cascade match (1+1 bit). */
    ti_input_t zero = ti_make(0, 0, 0);
    ti_input_t v3b  = ti_make(0xCAFEu, -100, 200);
    ecs_input_set(&a, 11, sa[0], &v1,   true);
    ecs_input_set(&a, 11, sa[1], &zero, true);
    ecs_input_set(&a, 11, sa[2], &v3b,  true);
    ecs_input_set(&a, 11, sa[3], &v4,   true);

    uint64_t buf[64] = {0};
    ecs_serializer_t s;
    ecs_serializer_init(&s, buf, sizeof(buf));
    ecs_input_serialize_tick(&a, &s, 11, 8);
    ecs_serializer_flush_bits(&s);
    int32_t bits_written = ecs_serializer_get_bits_written(&s);
    EXPECT(bits_written > 0,                  "serializer wrote some bits");

    ecs_deserializer_t d;
    ecs_deserializer_init(&d, buf, sizeof(buf));
    ecs_input_deserialize_tick(&b, &d);

    EXPECT(ti_eq(ti_get(&b, 11, sb[0]), v1),  "slot 0: cascade-match round-trip");
    EXPECT(ti_eq(ti_get(&b, 11, sb[1]), zero),"slot 1: baseline-match round-trip");
    EXPECT(ti_eq(ti_get(&b, 11, sb[2]), v3b), "slot 2: raw-payload round-trip");
    EXPECT(ti_eq(ti_get(&b, 11, sb[3]), v4),  "slot 3: cascade-match round-trip");

    /* Deserialized tick must be marked confirmed (per spec). */
    for (int i = 0; i < 4; i++) {
        ecs_input_view_t v = ecs_input_get_view(&b, 11, sb[i]);
        EXPECT(v.present && v.confirmed,      "deserialized slot: present + confirmed");
    }
    EXPECT(ecs_input_tick_confirmed(&b, 11),  "tick fully confirmed after deserialize");

    ecs_input_destroy(&a);
    ecs_input_destroy(&b);
}

/* --- serialize: redundancy=0 (just baseline vs raw) ------------------- */

static void test_input_serialize_redundancy_zero(void) {
    ecs_input_t a, b;
    ecs_input_init(&a, sizeof(ti_input_t), 16);
    ecs_input_init(&b, sizeof(ti_input_t), 16);
    uint32_t sa = ecs_input_alloc_slot(&a);
    uint32_t sb = ecs_input_alloc_slot(&b);
    EXPECT(sa == sb,                          "slots aligned");

    ti_input_t v = ti_make(0xBEEFu, 7, 7);
    ecs_input_set(&a, 5, sa, &v, true);

    uint64_t buf[16] = {0};
    ecs_serializer_t s;   ecs_serializer_init(&s, buf, sizeof(buf));
    ecs_input_serialize_tick(&a, &s, 5, 0);
    ecs_serializer_flush_bits(&s);

    ecs_deserializer_t d; ecs_deserializer_init(&d, buf, sizeof(buf));
    ecs_input_deserialize_tick(&b, &d);

    EXPECT(ti_eq(ti_get(&b, 5, sb), v),       "redundancy=0: raw round-trip");
    EXPECT(ecs_input_tick_confirmed(&b, 5),   "tick 5 confirmed on receiver");

    ecs_input_destroy(&a);
    ecs_input_destroy(&b);
}

/* --- serialize: all-zero tick compresses to 1 bit per slot ------------ */

static void test_input_serialize_all_zero_compact(void) {
    ecs_input_t it;
    ecs_input_init(&it, sizeof(ti_input_t), 16);
    uint32_t slots[5];
    for (int i = 0; i < 5; i++) slots[i] = ecs_input_alloc_slot(&it);

    ti_input_t z = ti_make(0, 0, 0);
    for (int i = 0; i < 5; i++) ecs_input_set(&it, 1, slots[i], &z, true);

    uint64_t buf[16] = {0};
    ecs_serializer_t s;
    ecs_serializer_init(&s, buf, sizeof(buf));
    ecs_input_serialize_tick(&it, &s, 1, 3);
    ecs_serializer_flush_bits(&s);

    int32_t bits = ecs_serializer_get_bits_written(&s);
    /* Header: 32 (tick) + 8 (redundancy) = 40. Per slot: 1 bit baseline.
       active_cap = 16 (init grow). Total = 40 + 16 = 56 bits. */
    EXPECT(bits == 56,                        "all-zero tick: header(40) + cap(16) = 56 bits");

    ecs_input_destroy(&it);
}

/* --- serialize: dead slot columns consume bits but don't apply -------- */

static void test_input_serialize_unregistered_skipped(void) {
    ecs_input_t a, b;
    ecs_input_init(&a, sizeof(ti_input_t), 16);
    ecs_input_init(&b, sizeof(ti_input_t), 16);

    /* Sender allocs 3 slots. Receiver allocs 2 then frees the second
       and re-allocs once more, leaving slot index 1 dead and slot 2
       live -- but we want sender's slot 2 to be dead on receiver. So
       receiver allocs slot 0, slot 1, then nothing else; sender allocs
       slot 0, 1, 2. b's active_cap stays 16 after first alloc on b. */
    uint32_t sa0 = ecs_input_alloc_slot(&a);
    uint32_t sa1 = ecs_input_alloc_slot(&a);
    uint32_t sa2 = ecs_input_alloc_slot(&a);
    uint32_t sb0 = ecs_input_alloc_slot(&b);
    uint32_t sb1 = ecs_input_alloc_slot(&b);
    EXPECT(sa0 == sb0 && sa1 == sb1,           "slots 0,1 aligned");
    EXPECT(sa2 == 2u,                          "sender slot 2 == 2");
    EXPECT(!ecs_input_slot_is_live(&b, 2),     "receiver slot 2 not live");

    ti_input_t v1 = ti_make(0x11u, 1, 1);
    ti_input_t v2 = ti_make(0x22u, 2, 2);
    ti_input_t v3 = ti_make(0x33u, 3, 3);
    ecs_input_set(&a, 1, sa0, &v1, true);
    ecs_input_set(&a, 1, sa1, &v2, true);
    ecs_input_set(&a, 1, sa2, &v3, true);

    uint64_t buf[16] = {0};
    ecs_serializer_t s;   ecs_serializer_init(&s, buf, sizeof(buf));
    ecs_input_serialize_tick(&a, &s, 1, 0);
    ecs_serializer_flush_bits(&s);

    ecs_deserializer_t d; ecs_deserializer_init(&d, buf, sizeof(buf));
    ecs_input_deserialize_tick(&b, &d);

    EXPECT(ti_eq(ti_get(&b, 1, sb0), v1),     "slot 0 applied on receiver");
    EXPECT(ti_eq(ti_get(&b, 1, sb1), v2),     "slot 1 applied on receiver");
    EXPECT(ecs_input_get(&b, 1, 2) == NULL,   "slot 2 not on receiver -- dead");

    ecs_input_destroy(&a);
    ecs_input_destroy(&b);
}

/* --- serialize: redundancy=8 wire size per cascade level -------------- */

static void test_input_serialize_redundancy_eight(void) {
    /* 1 allocated slot -> active_cap = 16 (init grow). 15 dead slots
       always emit 1 baseline bit each. Header is 32 (tick) + 8
       (redundancy) = 40 bits. So the per-test wire size is:
           40 + 15 (dead slots) + slot_0_cost
       slot_0_cost depends on which cascade level matches. */
    ecs_input_t a, b;
    ecs_input_init(&a, sizeof(ti_input_t), 32);
    ecs_input_init(&b, sizeof(ti_input_t), 32);
    uint32_t sa = ecs_input_alloc_slot(&a);
    uint32_t sb = ecs_input_alloc_slot(&b);
    EXPECT(sa == sb,                          "slots aligned");

    /* Mirror ticks 1..9 on both peers, all distinct payloads. */
    ti_input_t vs[10];
    vs[0] = ti_make(0, 0, 0);   /* unused, tick 0 reserved */
    for (uint32_t t = 1; t <= 9; t++) {
        vs[t] = ti_make((uint32_t)(t * 0x1111u), (int16_t)t, (int16_t)(-(int)t));
        ecs_input_set(&a, t, sa, &vs[t], true);
        ecs_input_set(&b, t, sb, &vs[t], true);
    }

    uint64_t buf[16];
    ecs_serializer_t s;
    ecs_deserializer_t d;
    int32_t bits;

    /* (A) Unique payload at tick 10: no cascade match -> 9 cascade
           bits (8 ones + signal-end one) + 64 raw payload bits.
       40 header + 15 dead + 9 cascade + 64 raw = 128 bits. */
    ti_input_t unique = ti_make(0xDEADBEEFu, 999, -999);
    ecs_input_set(&a, 10, sa, &unique, true);

    memset(buf, 0, sizeof(buf));
    ecs_serializer_init(&s, buf, sizeof(buf));
    ecs_input_serialize_tick(&a, &s, 10, 8);
    ecs_serializer_flush_bits(&s);
    bits = ecs_serializer_get_bits_written(&s);
    EXPECT(bits == 128,                       "redundancy=8 unique slot: 128 bits");

    ecs_deserializer_init(&d, buf, sizeof(buf));
    ecs_input_deserialize_tick(&b, &d);
    EXPECT(ti_eq(ti_get(&b, 10, sb), unique), "redundancy=8 unique: round-trip");

    /* (B) Tick 11 == tick 10 -> matches T-1: 2 cascade bits (1, 0).
       40 + 15 + 2 = 57 bits. */
    ecs_input_set(&a, 11, sa, &unique, true);

    memset(buf, 0, sizeof(buf));
    ecs_serializer_init(&s, buf, sizeof(buf));
    ecs_input_serialize_tick(&a, &s, 11, 8);
    ecs_serializer_flush_bits(&s);
    bits = ecs_serializer_get_bits_written(&s);
    EXPECT(bits == 57,                        "redundancy=8 T-1 match: 57 bits");

    ecs_deserializer_init(&d, buf, sizeof(buf));
    ecs_input_deserialize_tick(&b, &d);
    EXPECT(ti_eq(ti_get(&b, 11, sb), unique), "redundancy=8 T-1 match: round-trip");

    /* (C) Tick 12 == tick 4 -> matches T-8 (deepest cascade): 9
           cascade bits (8 ones + ending zero). 40 + 15 + 9 = 64 bits. */
    ecs_input_set(&a, 12, sa, &vs[4], true);

    memset(buf, 0, sizeof(buf));
    ecs_serializer_init(&s, buf, sizeof(buf));
    ecs_input_serialize_tick(&a, &s, 12, 8);
    ecs_serializer_flush_bits(&s);
    bits = ecs_serializer_get_bits_written(&s);
    EXPECT(bits == 64,                        "redundancy=8 T-8 match: 64 bits");

    ecs_deserializer_init(&d, buf, sizeof(buf));
    ecs_input_deserialize_tick(&b, &d);
    EXPECT(ti_eq(ti_get(&b, 12, sb), vs[4]),  "redundancy=8 T-8 match: round-trip");

    /* (D) Tick 13 == zero -> baseline match: 1 cascade bit.
       40 + 16 = 56 bits (slot 0 collapses to a single baseline bit
       just like the dead slots). */
    ti_input_t zero = ti_make(0, 0, 0);
    ecs_input_set(&a, 13, sa, &zero, true);

    memset(buf, 0, sizeof(buf));
    ecs_serializer_init(&s, buf, sizeof(buf));
    ecs_input_serialize_tick(&a, &s, 13, 8);
    ecs_serializer_flush_bits(&s);
    bits = ecs_serializer_get_bits_written(&s);
    EXPECT(bits == 56,                        "redundancy=8 baseline match: 56 bits");

    ecs_deserializer_init(&d, buf, sizeof(buf));
    ecs_input_deserialize_tick(&b, &d);
    EXPECT(ti_eq(ti_get(&b, 13, sb), zero),   "redundancy=8 baseline match: round-trip");

    /* Sanity: across all four scenarios receiver tick is fully confirmed. */
    EXPECT(ecs_input_tick_confirmed(&b, 10),  "tick 10 confirmed on receiver");
    EXPECT(ecs_input_tick_confirmed(&b, 11),  "tick 11 confirmed on receiver");
    EXPECT(ecs_input_tick_confirmed(&b, 12),  "tick 12 confirmed on receiver");
    EXPECT(ecs_input_tick_confirmed(&b, 13),  "tick 13 confirmed on receiver");

    ecs_input_destroy(&a);
    ecs_input_destroy(&b);
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
    RUN_TEST(test_input_frontier_sim_driven);
    RUN_TEST(test_input_past_frontier_confirmed);
    RUN_TEST(test_input_partial_no_advance);
    RUN_TEST(test_input_ring_wrap);
    RUN_TEST(test_input_clear);
    RUN_TEST(test_input_alloc_free_cycle);
    RUN_TEST(test_input_deterministic_replay);
    RUN_TEST(test_input_advance_to_tick);
    RUN_TEST(test_input_predicted_after_confirmed_dropped);
    RUN_TEST(test_input_empty_roster);
    RUN_TEST(test_input_iterator);
    RUN_TEST(test_input_player_cap);
    RUN_TEST(test_input_persistence);
    RUN_TEST(test_input_auto_grow_burst);
    RUN_TEST(test_input_grow_buf_preemptive);
    RUN_TEST(test_input_grow_player_cap_consistency);
    RUN_TEST(test_input_grow_buf_consistency);
    RUN_TEST(test_input_grow_combined_consistency);
    RUN_TEST(test_input_serialize_roundtrip);
    RUN_TEST(test_input_serialize_redundancy_zero);
    RUN_TEST(test_input_serialize_all_zero_compact);
    RUN_TEST(test_input_serialize_unregistered_skipped);
    RUN_TEST(test_input_serialize_redundancy_eight);
    int failed = g_failed - before;
    printf("\ninput: %d failed\n", failed);
    return failed ? 1 : 0;
}
