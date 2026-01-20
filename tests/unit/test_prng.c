/**
 * @file test_prng.c
 * @project Certifiable Training
 * @brief Unit tests for counter-based PRNG
 *
 * @traceability SRS-002, CT-MATH-001 §6
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 */

#include <stdio.h>
#include <string.h>
#include "ct_types.h"
#include "prng.h"
#include "dvm.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(fn) do { \
    printf("  %-50s ", #fn); \
    tests_run++; \
    if (fn()) { printf("PASS\n"); tests_passed++; } \
    else { printf("FAIL\n"); } \
} while(0)

/* ============================================================================
 * Test: Initialization
 * ============================================================================ */

static int test_init_sets_fields(void)
{
    ct_prng_t prng;
    ct_prng_init(&prng, 0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL);

    if (prng.seed != 0x123456789ABCDEF0ULL) return 0;
    if (prng.op_id != 0xFEDCBA9876543210ULL) return 0;
    if (prng.step != 0) return 0;

    return 1;
}

static int test_init_null_safe(void)
{
    /* Should not crash */
    ct_prng_init(NULL, 0, 0);
    return 1;
}

/* ============================================================================
 * Test: Determinism - Core Property
 * ============================================================================ */

static int test_same_seed_same_sequence(void)
{
    ct_prng_t prng1, prng2;

    ct_prng_init(&prng1, 12345, 100);
    ct_prng_init(&prng2, 12345, 100);

    /* Same seed + op_id must produce identical sequence */
    for (int i = 0; i < 1000; i++) {
        uint32_t v1 = ct_prng_next(&prng1);
        uint32_t v2 = ct_prng_next(&prng2);
        if (v1 != v2) return 0;
    }

    return 1;
}

static int test_different_seed_different_sequence(void)
{
    ct_prng_t prng1, prng2;

    ct_prng_init(&prng1, 12345, 100);
    ct_prng_init(&prng2, 12346, 100);  /* Different seed */

    /* Should produce different sequences */
    int different_count = 0;
    for (int i = 0; i < 100; i++) {
        uint32_t v1 = ct_prng_next(&prng1);
        uint32_t v2 = ct_prng_next(&prng2);
        if (v1 != v2) different_count++;
    }

    /* Expect most values to differ */
    return different_count > 90;
}

static int test_different_opid_different_sequence(void)
{
    ct_prng_t prng1, prng2;

    ct_prng_init(&prng1, 12345, 100);
    ct_prng_init(&prng2, 12345, 101);  /* Different op_id */

    int different_count = 0;
    for (int i = 0; i < 100; i++) {
        uint32_t v1 = ct_prng_next(&prng1);
        uint32_t v2 = ct_prng_next(&prng2);
        if (v1 != v2) different_count++;
    }

    return different_count > 90;
}

static int test_peek_matches_next(void)
{
    ct_prng_t prng;
    ct_prng_init(&prng, 42, 0);

    /* Peek at steps 0-9 before advancing */
    uint32_t peeked[10];
    for (int i = 0; i < 10; i++) {
        peeked[i] = ct_prng_peek(&prng, (uint64_t)i);
    }

    /* Now generate with next() and compare */
    for (int i = 0; i < 10; i++) {
        uint32_t actual = ct_prng_next(&prng);
        if (actual != peeked[i]) return 0;
    }

    return 1;
}

static int test_peek_does_not_advance(void)
{
    ct_prng_t prng;
    ct_prng_init(&prng, 42, 0);

    /* Peek multiple times */
    ct_prng_peek(&prng, 0);
    ct_prng_peek(&prng, 100);
    ct_prng_peek(&prng, 1000);

    /* Step should still be 0 */
    return prng.step == 0;
}

/* ============================================================================
 * Test: Distribution Quality (basic sanity checks)
 * ============================================================================ */

static int test_all_bits_vary(void)
{
    ct_prng_t prng;
    ct_prng_init(&prng, 0xDEADBEEF, 0);

    uint32_t or_accum = 0;
    uint32_t and_accum = 0xFFFFFFFF;

    for (int i = 0; i < 10000; i++) {
        uint32_t v = ct_prng_next(&prng);
        or_accum |= v;
        and_accum &= v;
    }

    /* All bits should be 1 in OR (each bit set at least once) */
    if (or_accum != 0xFFFFFFFF) return 0;

    /* All bits should be 0 in AND (each bit clear at least once) */
    if (and_accum != 0) return 0;

    return 1;
}

static int test_not_constant(void)
{
    ct_prng_t prng;
    ct_prng_init(&prng, 1, 1);

    uint32_t first = ct_prng_next(&prng);

    for (int i = 0; i < 100; i++) {
        uint32_t v = ct_prng_next(&prng);
        if (v != first) return 1;  /* Found a different value - good */
    }

    return 0;  /* All same - bad */
}

static int test_not_incrementing(void)
{
    ct_prng_t prng;
    ct_prng_init(&prng, 999, 0);

    uint32_t prev = ct_prng_next(&prng);
    int incrementing = 1;

    for (int i = 0; i < 100; i++) {
        uint32_t v = ct_prng_next(&prng);
        if (v != prev + 1) incrementing = 0;
        prev = v;
    }

    return !incrementing;  /* Should NOT be simply incrementing */
}

/* ============================================================================
 * Test: Stochastic Rounding
 * ============================================================================ */

static int test_stochastic_round_deterministic(void)
{
    ct_prng_t prng1, prng2;
    ct_fault_flags_t faults1 = {0}, faults2 = {0};

    ct_prng_init(&prng1, 12345, 500);
    ct_prng_init(&prng2, 12345, 500);

    /* Same input + same PRNG state = same output */
    for (int i = 0; i < 100; i++) {
        int64_t x = (int64_t)i * 0x8000 + 0x4000;  /* i.25 in Q16.16 */

        int32_t r1 = ct_stochastic_round(x, 16, &prng1, &faults1);
        int32_t r2 = ct_stochastic_round(x, 16, &prng2, &faults2);

        if (r1 != r2) return 0;
    }

    return 1;
}

static int test_stochastic_round_shift_zero(void)
{
    ct_prng_t prng;
    ct_fault_flags_t faults = {0};
    ct_prng_init(&prng, 1, 1);

    /* Shift 0 should just clamp */
    int32_t result = ct_stochastic_round(12345, 0, &prng, &faults);
    return result == 12345;
}

static int test_stochastic_round_shift_bounds(void)
{
    ct_prng_t prng;
    ct_fault_flags_t faults = {0};
    ct_prng_init(&prng, 1, 1);

    /* Shift > 62 should fault */
    int32_t result = ct_stochastic_round(12345, 63, &prng, &faults);

    return (result == 0) && (faults.domain == 1);
}

static int test_stochastic_round_probabilistic_behavior(void)
{
    ct_prng_t prng;
    ct_fault_flags_t faults = {0};
    ct_prng_init(&prng, 42, 0);

    /* Value = 0.5 in Q16.16 = 0x8000 */
    /* Should round up ~50% of the time */
    int64_t half = 0x8000;
    int round_up_count = 0;

    for (int i = 0; i < 10000; i++) {
        int32_t r = ct_stochastic_round(half, 16, &prng, &faults);
        if (r == 1) round_up_count++;
    }

    /* Expect ~50% round up (allow 45-55% range) */
    return (round_up_count >= 4500) && (round_up_count <= 5500);
}

static int test_stochastic_round_zero_always_zero(void)
{
    ct_prng_t prng;
    ct_fault_flags_t faults = {0};
    ct_prng_init(&prng, 99, 99);

    /* Zero fraction should always round to quotient */
    for (int i = 0; i < 100; i++) {
        int64_t x = (int64_t)i << 16;  /* Exact integer in Q16.16 */
        int32_t r = ct_stochastic_round(x, 16, &prng, &faults);
        if (r != i) return 0;
    }

    return 1;
}

/* ============================================================================
 * Test: op_id generation
 * ============================================================================ */

static int test_opid_different_for_different_inputs(void)
{
    uint64_t id1 = ct_prng_make_op_id(0, 0, 0);
    uint64_t id2 = ct_prng_make_op_id(0, 0, 1);
    uint64_t id3 = ct_prng_make_op_id(0, 1, 0);
    uint64_t id4 = ct_prng_make_op_id(1, 0, 0);

    /* All should be different */
    if (id1 == id2) return 0;
    if (id1 == id3) return 0;
    if (id1 == id4) return 0;
    if (id2 == id3) return 0;
    if (id2 == id4) return 0;
    if (id3 == id4) return 0;

    return 1;
}

static int test_opid_deterministic(void)
{
    uint64_t id1 = ct_prng_make_op_id(5, 10, 15);
    uint64_t id2 = ct_prng_make_op_id(5, 10, 15);

    return id1 == id2;
}

/* ============================================================================
 * Test: Known test vectors (for cross-platform verification)
 * ============================================================================ */

static int test_known_vectors(void)
{
    ct_prng_t prng;
    ct_prng_init(&prng, 0, 0);

    /* Generate first 5 values and record for cross-platform check */
    uint32_t v0 = ct_prng_next(&prng);
    uint32_t v1 = ct_prng_next(&prng);
    uint32_t v2 = ct_prng_next(&prng);
    uint32_t v3 = ct_prng_next(&prng);
    uint32_t v4 = ct_prng_next(&prng);

    /*
     * These values are deterministic and must match on all platforms.
     * If this test fails on a new platform, the PRNG is broken there.
     *
     * Reference values established on x86_64 Linux:
     */

    printf("\n    PRNG(seed=0, op_id=0): v0=0x%08X v1=0x%08X v2=0x%08X v3=0x%08X v4=0x%08X\n",
           v0, v1, v2, v3, v4);

    /* Verify against reference values */
    if (v0 != 0x24F74A49) return 0;
    if (v1 != 0xA96E3F40) return 0;
    if (v2 != 0xC1C8ECFB) return 0;
    if (v3 != 0xE2E62252) return 0;
    if (v4 != 0x0AAD3C4D) return 0;

    return 1;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void)
{
    printf("==============================================\n");
    printf("Certifiable Training - PRNG Tests\n");
    printf("Traceability: SRS-002, CT-MATH-001 §6\n");
    printf("==============================================\n\n");

    printf("Initialization:\n");
    RUN_TEST(test_init_sets_fields);
    RUN_TEST(test_init_null_safe);

    printf("\nDeterminism (core property):\n");
    RUN_TEST(test_same_seed_same_sequence);
    RUN_TEST(test_different_seed_different_sequence);
    RUN_TEST(test_different_opid_different_sequence);
    RUN_TEST(test_peek_matches_next);
    RUN_TEST(test_peek_does_not_advance);

    printf("\nDistribution quality:\n");
    RUN_TEST(test_all_bits_vary);
    RUN_TEST(test_not_constant);
    RUN_TEST(test_not_incrementing);

    printf("\nStochastic rounding:\n");
    RUN_TEST(test_stochastic_round_deterministic);
    RUN_TEST(test_stochastic_round_shift_zero);
    RUN_TEST(test_stochastic_round_shift_bounds);
    RUN_TEST(test_stochastic_round_probabilistic_behavior);
    RUN_TEST(test_stochastic_round_zero_always_zero);

    printf("\nOperation ID generation:\n");
    RUN_TEST(test_opid_different_for_different_inputs);
    RUN_TEST(test_opid_deterministic);

    printf("\nKnown test vectors:\n");
    RUN_TEST(test_known_vectors);

    printf("\n==============================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("==============================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
