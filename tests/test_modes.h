#pragma once

#include "ecs.h"
#include "test_ecs.h"

/* Tests for VM mode dispatch (CONFIRMED vs PREDICT) and the byte-layout
   contract: predicted writes land at slot+64, confirmed at slot+0; reads
   pick predicted iff dirty bit is set.

   ecs_tree_set / ecs_tree_remove branch on tree->mode (0 = CONFIRMED,
   1 = PREDICT) — both halves must be exercised. ecs_tree_set_mode
   asserts no in-flight prediction (dirty == 0 everywhere). */

/* ---- byte-layout: PREDICT writes to slot+64, CONFIRMED to slot+0.
   L1 backing memory is allocated uninitialized (mi_malloc_aligned), so we
   prime both halves of every slot with known sentinels (CONFIRMED-mode
   writes both halves via two staged writes) before observing layout. ---- */

static void test_mode_predict_byte_layout(void) {
    ecs_tree_t* t = make_tree();
    EXPECT(t->mode == ECS_MODE_CONFIRMED, "make_tree returns CONFIRMED-mode tree (default)");

    /* Prime confirmed slot via CONFIRMED write. */
    set_val(t, 5, 0x11111111);
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);

    /* PREDICT write — should land at slot+64, leave confirmed slot alone. */
    set_val(t, 5, 0xAAAAAAAA);
    ecs_l1_t* l1 = l1_of(t, 0, 0);
    const comp_t* conf = (const comp_t*)ecs_l1_confirmed(l1, 5, sizeof(comp_t));
    const comp_t* pred = (const comp_t*)ecs_l1_predicted(l1, 5, sizeof(comp_t));
    EXPECT(*pred == 0xAAAAAAAA,  "predicted slot holds new value");
    EXPECT(*conf == 0x11111111,  "confirmed slot bytes preserved");
    EXPECT((l1->dirty & (1ULL << 5)) != 0, "dirty bit set");
    EXPECT(get_current_val(t, 5) == 0xAAAAAAAA, "get reads predicted (dirty)");

    /* Predicted is always speculative — rollback discards it, confirmed stays. */
    ecs_tree_rollback(t);
    EXPECT(*conf == 0x11111111, "post-rollback: confirmed slot preserved");
    EXPECT(l1->dirty == 0, "post-rollback: dirty cleared");
    EXPECT(get_current_val(t, 5) == 0x11111111, "get reads confirmed after rollback");

    free_tree(t);
}

static void test_mode_confirmed_byte_layout(void) {
    ecs_tree_t* t = make_tree();
    ecs_tree_set_mode(t, ECS_MODE_CONFIRMED);
    EXPECT(t->mode == ECS_MODE_CONFIRMED, "mode is CONFIRMED");

    set_val(t, 5, 0xBBBBBBBB);
    ecs_l1_t* l1 = l1_of(t, 0, 0);
    const comp_t* conf = (const comp_t*)ecs_l1_confirmed(l1, 5, sizeof(comp_t));
    EXPECT(*conf == 0xBBBBBBBB, "CONFIRMED write goes to confirmed slot");
    EXPECT(l1->dirty == 0, "CONFIRMED write does not set dirty");
    EXPECT((l1->predicted_mask_any & (1ULL << 5)) != 0, "mask mirrors in CONFIRMED");
    EXPECT((l1->confirmed_mask_any & (1ULL << 5)) != 0, "confirmed_mask set");

    free_tree(t);
}

static void test_mode_confirmed_get_reads_confirmed(void) {
    ecs_tree_t* t = make_tree();
    ecs_tree_set_mode(t, ECS_MODE_CONFIRMED);
    set_val(t, 0, 0xCCCCCCCC);
    /* dirty stays 0 in CONFIRMED, so get's dirty_hi shift is 0 — reads slot+0. */
    EXPECT(get_current_val(t, 0) == 0xCCCCCCCC, "get reads confirmed slot");
    free_tree(t);
}

/* ---- CONFIRMED-mode remove releases L1/L2 inline (no promote walk) ---- */

static void test_mode_confirmed_remove_releases_l1(void) {
    ecs_tree_t* t = make_tree();
    ecs_tree_set_mode(t, ECS_MODE_CONFIRMED);
    set_val(t, ENT(0, 0, 3), 7);
    EXPECT(l1_of(t, 0, 0) != &ecs_default_l1, "l1 allocated");

    int rc = ecs_tree_remove(t, ENT(0, 0, 3));
    EXPECT(rc == 1, "remove returns 1");
    EXPECT(l1_of(t, 0, 0) == &ecs_default_l1, "CONFIRMED remove released empty l1 inline");
    EXPECT(l2_of(t, 0)    == &ecs_default_l2, "CONFIRMED remove released empty l2 inline");
    EXPECT(t->root->confirmed_mask_any == 0, "l3 confirmed_mask cleared");
    EXPECT(t->root->predicted_mask_any == 0, "l3 predicted_mask cleared in lockstep");
    EXPECT(t->root->dirty == 0, "no dirty raised");

    free_tree(t);
}

static void test_mode_confirmed_remove_returns_zero_for_absent(void) {
    ecs_tree_t* t = make_tree();
    ecs_tree_set_mode(t, ECS_MODE_CONFIRMED);
    int rc = ecs_tree_remove(t, ENT(0, 0, 0));
    EXPECT(rc == 0, "remove of absent slot returns 0");
    free_tree(t);
}

/* ---- promote in CONFIRMED is no-op tick++ ---- */

static void test_mode_confirmed_promote_is_tick_only(void) {
    ecs_tree_t* t = make_tree();
    ecs_tree_set_mode(t, ECS_MODE_CONFIRMED);
    set_val(t, 0, 1);

    uint64_t tick_before = t->tick;
    ecs_tree_rollback(t);
    EXPECT(t->tick == tick_before + 1, "promote in CONFIRMED still bumps tick");
    EXPECT(get_current_val(t, 0) == 1, "value unchanged");

    free_tree(t);
}

/* ---- mode switch: PREDICT -> CONFIRMED -> PREDICT ----
   set_mode requires no in-flight prediction. After switch, writes route
   to the new mode's slot. Pre-switch confirmed bytes remain visible. */

static void test_mode_switch_predict_to_confirmed_to_predict(void) {
    ecs_tree_t* t = make_tree();

    /* Phase 1: PREDICT — write + promote so dirty == 0 before switch. */
    set_val(t, 0, 100);
    ecs_tree_rollback(t);
    ecs_l1_t* l1 = l1_of(t, 0, 0);
    EXPECT(l1->dirty == 0, "dirty cleared before switch");

    /* Switch to CONFIRMED. */
    ecs_tree_set_mode(t, ECS_MODE_CONFIRMED);
    set_val(t, 0, 200);
    const comp_t* conf = (const comp_t*)ecs_l1_confirmed(l1, 0, sizeof(comp_t));
    const comp_t* pred = (const comp_t*)ecs_l1_predicted(l1, 0, sizeof(comp_t));
    EXPECT(*conf == 200, "CONFIRMED write overwrote confirmed slot directly");
    EXPECT(l1->dirty == 0, "no dirty raised");

    /* Switch back to PREDICT. PREDICT write must overwrite the predicted slot
       and leave confirmed bytes alone, regardless of pre-existing pred bytes. */
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    set_val(t, 0, 300);
    EXPECT(*pred == 300, "PREDICT write went to predicted slot");
    EXPECT(*conf == 200, "confirmed slot preserved");
    EXPECT((l1->dirty & 1ULL) != 0, "dirty bit set");
    EXPECT(get_current_val(t, 0) == 300, "current view = predicted");

    ecs_tree_rollback(t);
    EXPECT(get_current_val(t, 0) == 200, "rollback restores confirmed");

    free_tree(t);
}

/* ---- multi-tick lifecycle: CONFIRMED writes advance state; PREDICT writes
   are speculative and always discarded by rollback. ---- */

static void test_mode_predict_multi_tick_lifecycle(void) {
    ecs_tree_t* t = make_tree();

    /* Tick 1: CONFIRMED add. */
    set_val(t, 0, 10);
    ecs_tree_rollback(t);
    EXPECT(t->tick == 1, "tick=1 after first rollback");
    EXPECT(get_current_val(t, 0) == 10, "tick 1 confirmed = 10");

    /* Tick 2: CONFIRMED overwrite. */
    set_val(t, 0, 20);
    ecs_tree_rollback(t);
    EXPECT(t->tick == 2, "tick=2");
    EXPECT(get_current_val(t, 0) == 20, "tick 2 confirmed=20");

    /* PREDICT modify, rollback discards — confirmed stays 20, tick unchanged. */
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    set_val(t, 0, 30);
    EXPECT(get_current_val(t, 0) == 30, "predicted=30 mid-tick");
    ecs_tree_rollback(t);
    EXPECT(t->tick == 2, "predict-only rollback does not advance tick");
    EXPECT(get_current_val(t, 0) == 20, "confirmed unchanged — predict discarded");

    /* Tick 3: CONFIRMED-add second slot. */
    ecs_tree_set_mode(t, ECS_MODE_CONFIRMED);
    set_val(t, 0, 40);
    set_val(t, ENT(0, 0, 5), 50);
    ecs_tree_rollback(t);
    EXPECT(t->tick == 3, "tick=3");
    EXPECT(get_current_val(t, 0)            == 40, "slot 0 = 40");
    EXPECT(get_current_val(t, ENT(0, 0, 5)) == 50, "slot 5 = 50");

    /* PREDICT-remove + PREDICT-modify, rollback restores both. Tick unchanged. */
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    ecs_tree_remove(t, 0);
    set_val(t, ENT(0, 0, 5), 999);
    ecs_tree_rollback(t);
    EXPECT(t->tick == 3, "predict-only rollback does not advance tick");
    EXPECT(get_current_val(t, 0)            == 40, "rollback restored slot 0");
    EXPECT(get_current_val(t, ENT(0, 0, 5)) == 50, "rollback restored slot 5");

    /* Tick 4: CONFIRMED-remove + CONFIRMED-add. */
    ecs_tree_set_mode(t, ECS_MODE_CONFIRMED);
    ecs_tree_remove(t, 0);
    set_val(t, ENT(0, 0, 7), 70);
    ecs_tree_rollback(t);
    EXPECT(t->tick == 4, "tick=4");
    EXPECT(((const ecs_l1_t*)l1_of(t, 0, 0))->confirmed_mask_any == ((1ULL << 5) | (1ULL << 7)),
           "confirmed_mask = slots {5,7}");

    /* Mask invariants must hold across the whole run. */
    EXPECT(ecs_tree_masks_valid(t), "masks valid after full lifecycle");

    free_tree(t);
}

/* ---- memcmp short-circuit on a fresh slot — first write must NOT be
   short-circuited (slot bit is absent, so `cur` would alias uninitialized
   bytes). Covers the guard at the top of ecs_tree_set. ---- */

static void test_mode_first_write_no_short_circuit(void) {
    ecs_tree_t* t = make_tree();
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    /* First write to a fresh tree — slot bit absent. Even if the new value
       happens to equal 0 (the uninitialized byte content), the write must
       go through and set masks. */
    comp_t zero = 0;
    ecs_tree_set(t, 7, &zero);
    EXPECT((l1_of(t, 0, 0)->predicted_mask_any & (1ULL << 7)) != 0,
           "first write sets predicted mask even when value bytes equal 0");
    EXPECT((l1_of(t, 0, 0)->dirty & (1ULL << 7)) != 0, "dirty set on fresh-slot write");
    ecs_tree_rollback(t);   /* clear in-flight prediction so destroy is dirty-free */
    free_tree(t);
}

/* ---- repeated identical write hits memcmp short-circuit (existing slot) ---- */

static void test_mode_repeat_write_short_circuits(void) {
    ecs_tree_t* t = make_tree();
    set_val(t, 0, 42);
    ecs_tree_rollback(t);
    ecs_l1_t* l1 = l1_of(t, 0, 0);
    EXPECT(l1->dirty == 0 && l1->changed == 0, "clean post-promote");

    set_val(t, 0, 42);  /* identical to confirmed */
    EXPECT(l1->dirty   == 0, "no dirty for identical predict write");
    EXPECT(l1->changed == 0, "no changed for identical predict write");

    free_tree(t);
}

/* ---- determinism: predicted simulation must reproduce confirmed simulation.
   3 confirmed integration ticks, capture view CRC. Then 3 predicted "ticks"
   on the predicted overlay running same integration logic — view CRC at this
   point is the would-be tick-6 fingerprint. Rollback discards predicted ⇒
   view collapses to confirmed-tick-3. Replaying the same 3 steps in CONFIRMED
   mode must land on the predicted result. View CRC excludes tick by design,
   so predict-on-tick-3 vs confirmed-at-tick-6 hash equal when state matches.
   ---- */
static void test_mode_predict_matches_confirmed_simulation(void) {
    ecs_tree_t* t = make_tree();

    /* Phase A: 3 confirmed ticks. Integration logic = slot 0 := tick. */
    set_val(t, 0, 1); ecs_tree_rollback(t);
    set_val(t, 0, 2); ecs_tree_rollback(t);
    set_val(t, 0, 3); ecs_tree_rollback(t);
    EXPECT(t->tick == 3, "tick=3 after 3 confirmed integration ticks");
    uint64_t crc_tick_3 = ecs_tree_crc64(t);

    /* Phase B: 3 predicted ticks — same logic, predicted slot. No rollback
       between steps; each predict-write reads previous predicted (dirty). */
    ecs_tree_set_mode(t, ECS_MODE_PREDICT);
    set_val(t, 0, 4);
    set_val(t, 0, 5);
    set_val(t, 0, 6);
    uint64_t crc_tick_6 = ecs_tree_crc64(t);
    EXPECT(crc_tick_6 != crc_tick_3, "predicted view diverges from confirmed");
    EXPECT(get_current_val(t, 0) == 6, "predicted value = 6");

    /* Rollback discards predicted overlay → view = confirmed = crc_tick_3. */
    ecs_tree_rollback(t);
    EXPECT(ecs_tree_crc64(t) == crc_tick_3,
           "post-rollback view CRC matches confirmed-tick-3");
    EXPECT(get_current_val(t, 0) == 3, "value restored to confirmed 3");

    /* Phase C: 3 confirmed ticks replay → must reach crc_tick_6. */
    ecs_tree_set_mode(t, ECS_MODE_CONFIRMED);
    set_val(t, 0, 4); ecs_tree_rollback(t);
    set_val(t, 0, 5); ecs_tree_rollback(t);
    set_val(t, 0, 6); ecs_tree_rollback(t);
    EXPECT(t->tick == 6, "tick=6 after 3 more confirmed ticks");
    EXPECT(ecs_tree_crc64(t) == crc_tick_6,
           "confirmed-replay view CRC matches predicted result");

    free_tree(t);
}

static int test_modes_all(void) {
    int before = g_failed;
    printf("=== mode / phase tests ===\n\n");
    test_mode_predict_byte_layout();
    test_mode_confirmed_byte_layout();
    test_mode_confirmed_get_reads_confirmed();
    test_mode_confirmed_remove_releases_l1();
    test_mode_confirmed_remove_returns_zero_for_absent();
    test_mode_confirmed_promote_is_tick_only();
    test_mode_switch_predict_to_confirmed_to_predict();
    test_mode_predict_multi_tick_lifecycle();
    test_mode_first_write_no_short_circuit();
    test_mode_repeat_write_short_circuits();
    test_mode_predict_matches_confirmed_simulation();
    int failed = g_failed - before;
    printf("\nmodes: %d failed\n", failed);
    return failed ? 1 : 0;
}
