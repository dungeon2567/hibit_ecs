#include "ecs.h"
#include "test_ecs.h"
#include "test_fixed.h"
#include "test_math.h"
#include "test_serialize.h"
#include "test_iterator.h"
#include "test_changed.h"
#include "test_pipeline.h"
#include "test_modes.h"
#include "test_owned.h"
#include "test_buffer.h"
#include "test_broadphase.h"
#include "test_input.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } } while(0)

/* --- ecs_memcpy_sparse ---------------------------------------------------- */

static void test_memcpy(void) {
    float src[8] = { 0,1,2,3,4,5,6,7 };
    float dst[8] = {0};

    ecs_memcpy_sparse(dst, src, sizeof(float), 0xB5ULL);  /* 0b10110101 */
    CHECK(dst[0] == 0.f);
    CHECK(dst[2] == 2.f);
    CHECK(dst[4] == 4.f);
    CHECK(dst[5] == 5.f);
    CHECK(dst[7] == 7.f);
    CHECK(dst[1] == 0.f);
    CHECK(dst[3] == 0.f);

    printf("  ecs_memcpy_sparse   OK\n");
}

/* --- helpers -------------------------------------------------------------- */

typedef struct { float x, y; } Vec2;

static int entity_exists_confirmed(const ecs_tree_t* tree, int index) {
    const ecs_l2_t* l2s = tree->root->children[(index >> 12) & 0x3F];
    if (l2s == &ecs_default_l2) return 0;
    const ecs_l1_t* l1s = l2s->children[(index >> 6) & 0x3F];
    if (l1s == &ecs_default_l1) return 0;
    return (int)((l1s->confirmed_mask_any >> (index & 0x3F)) & 1);
}

/* --- promote -------------------------------------------------------------- */

static void test_promote(void) {
    ecs_tree_t tree = {0};
    ecs_tree_init(&tree, sizeof(Vec2), 0);

    *(Vec2*)ecs_tree_get_mut(&tree, 3) = (Vec2){ 1.f, 2.f };
    *(Vec2*)ecs_tree_get_mut(&tree, 7) = (Vec2){ 3.f, 4.f };
    ecs_tree_rollback(&tree);

    const Vec2* got3 = (const Vec2*)ecs_tree_get(&tree, 3);
    const Vec2* got7 = (const Vec2*)ecs_tree_get(&tree, 7);
    CHECK(got3->x == 1.f && got3->y == 2.f);
    CHECK(got7->x == 3.f && got7->y == 4.f);

    ecs_tree_destroy(&tree);
    printf("  promote      OK\n");
}

/* --- predict + rollback (single ack window) ------------------------------- */

static void test_predict_rollback(void) {
    ecs_tree_t tree = {0};
    ecs_tree_init(&tree, sizeof(Vec2), 0);

    uint64_t crc0 = ecs_tree_crc64(&tree);

    /* Seed confirmed via CONFIRMED write. */
    *(Vec2*)ecs_tree_get_mut(&tree, 3) = (Vec2){ 1.f, 2.f };
    CHECK(ecs_tree_rollback(&tree) == 1);
    uint64_t crc1 = ecs_tree_crc64(&tree);
    CHECK(crc1 != crc0);

    /* Switch to PREDICT - modifications are speculative. */
    ecs_tree_set_mode(&tree, ECS_MODE_PREDICT);
    *(Vec2*)ecs_tree_get_mut(&tree, 3) = (Vec2){ 5.f, 6.f };
    *(Vec2*)ecs_tree_get_mut(&tree, 7) = (Vec2){ 9.f, 9.f };
    ecs_tree_remove(&tree, 3);

    /* CRC is view-aware - predicted writes shift it. */
    CHECK(ecs_tree_crc64(&tree) != crc1);

    CHECK(ecs_tree_rollback(&tree) == 0);     /* predict-only rollback reports no advance */
    CHECK(ecs_tree_crc64(&tree) == crc1);     /* rollback restores view to confirmed */
    const Vec2* cur = (const Vec2*)ecs_tree_get(&tree, 3);
    CHECK(cur->x == 1.f && cur->y == 2.f);
    CHECK(!entity_exists_confirmed(&tree, 7));

    ecs_tree_destroy(&tree);
    printf("  predict/rollback  OK\n");
}

/* --- predict-mode with mixed add/remove ----------------------------------- */

static void test_predict_rollback_mixed(void) {
    ecs_tree_t tree = {0};
    ecs_tree_init(&tree, sizeof(Vec2), 0);

    /* Seed confirmed via CONFIRMED writes. */
    *(Vec2*)ecs_tree_get_mut(&tree,  3) = (Vec2){ 1.f, 2.f };
    *(Vec2*)ecs_tree_get_mut(&tree,  7) = (Vec2){ 3.f, 4.f };
    CHECK(ecs_tree_rollback(&tree) == 1);
    uint64_t crc1 = ecs_tree_crc64(&tree);

    /* Switch to PREDICT for speculative mutations. */
    ecs_tree_set_mode(&tree, ECS_MODE_PREDICT);
    *(Vec2*)ecs_tree_get_mut(&tree,  3) = (Vec2){ 5.f, 6.f };
    ecs_tree_remove(&tree, 7);
    *(Vec2*)ecs_tree_get_mut(&tree, 20) = (Vec2){ 7.f, 8.f };

    /* Rollback: discards all predicted, confirmed unchanged. */
    ecs_tree_rollback(&tree);
    CHECK(ecs_tree_crc64(&tree) == crc1);
    CHECK( entity_exists_confirmed(&tree, 3));
    CHECK( entity_exists_confirmed(&tree, 7));
    CHECK(!entity_exists_confirmed(&tree, 20));
    CHECK(((const Vec2*)ecs_tree_get(&tree, 3))->x == 1.f);
    CHECK(((const Vec2*)ecs_tree_get(&tree, 7))->x == 3.f);

    /* CONFIRMED writes to actually advance state. */
    ecs_tree_set_mode(&tree, ECS_MODE_CONFIRMED);
    *(Vec2*)ecs_tree_get_mut(&tree,  3) = (Vec2){ 5.f, 6.f };
    ecs_tree_remove(&tree, 7);
    *(Vec2*)ecs_tree_get_mut(&tree, 20) = (Vec2){ 7.f, 8.f };
    CHECK(ecs_tree_rollback(&tree) == 1);     /* second confirmed advance */
    CHECK( entity_exists_confirmed(&tree,  3));
    CHECK(!entity_exists_confirmed(&tree,  7));
    CHECK( entity_exists_confirmed(&tree, 20));
    CHECK(((const Vec2*)ecs_tree_get(&tree,  3))->x == 5.f);
    CHECK(((const Vec2*)ecs_tree_get(&tree, 20))->x == 7.f);

    ecs_tree_destroy(&tree);
    printf("  predict/rollback mixed  OK\n");
}

/* -------------------------------------------------------------------------- */

int main(void) {
    ecs_crc64_init();
    printf("tests\n");
    test_memcpy();
    test_promote();
    test_predict_rollback();
    test_predict_rollback_mixed();
    if (failures)
        printf("\n%d FAILURE(S)\n", failures);
    else
        printf("\nAll tests passed\n");

    int result = test_all()
               | test_fixed_all()
               | test_math_all()
               | test_serialize_all()
               | test_iterator_all()
               | test_changed_all()
               | test_pipeline_all()
               | test_modes_all()
               | test_owned_all()
               | test_buffer_all()
               | test_broadphase_all()
               | test_input_all()
               | (failures ? 1 : 0);

    return result;
}
