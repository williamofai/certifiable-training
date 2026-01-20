/**
 * @file test_primitives.c
 * @project Certifiable Training
 * @brief Unit tests for DVM arithmetic primitives
 *
 * @traceability CT-MATH-001 §3, CT-SPEC-001 §3
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 */

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "ct_types.h"
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
 * Test: dvm_add - Saturating Addition (CT-MATH-001 §3.1)
 * ============================================================================ */

static int test_add_basic(void)
{
    ct_fault_flags_t faults = {0};

    /* Simple addition */
    fixed_t result = dvm_add(FIXED_ONE, FIXED_ONE, &faults);
    if (result != (2 * FIXED_ONE)) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_add_negative(void)
{
    ct_fault_flags_t faults = {0};

    /* Negative numbers */
    fixed_t result = dvm_add(-FIXED_ONE, -FIXED_ONE, &faults);
    if (result != (-2 * FIXED_ONE)) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_add_mixed_sign(void)
{
    ct_fault_flags_t faults = {0};

    /* Mixed signs */
    fixed_t result = dvm_add(FIXED_ONE, -FIXED_HALF, &faults);
    if (result != FIXED_HALF) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_add_overflow_saturates(void)
{
    ct_fault_flags_t faults = {0};

    /* Overflow should saturate to MAX and set flag */
    fixed_t result = dvm_add(FIXED_MAX, FIXED_ONE, &faults);
    if (result != FIXED_MAX) return 0;
    if (!faults.overflow) return 0;

    return 1;
}

static int test_add_underflow_saturates(void)
{
    ct_fault_flags_t faults = {0};

    /* Underflow should saturate to MIN and set flag */
    fixed_t result = dvm_add(FIXED_MIN, -FIXED_ONE, &faults);
    if (result != FIXED_MIN) return 0;
    if (!faults.underflow) return 0;

    return 1;
}

static int test_add_null_faults_safe(void)
{
    /* Should not crash with NULL faults pointer */
    fixed_t result = dvm_add(FIXED_ONE, FIXED_ONE, NULL);
    return result == (2 * FIXED_ONE);
}

/* ============================================================================
 * Test: dvm_sub - Saturating Subtraction (CT-MATH-001 §3.2)
 * ============================================================================ */

static int test_sub_basic(void)
{
    ct_fault_flags_t faults = {0};

    fixed_t result = dvm_sub(2 * FIXED_ONE, FIXED_ONE, &faults);
    if (result != FIXED_ONE) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_sub_negative_result(void)
{
    ct_fault_flags_t faults = {0};

    fixed_t result = dvm_sub(FIXED_ONE, 2 * FIXED_ONE, &faults);
    if (result != -FIXED_ONE) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_sub_overflow_saturates(void)
{
    ct_fault_flags_t faults = {0};

    /* MAX - (-1) overflows */
    fixed_t result = dvm_sub(FIXED_MAX, -FIXED_ONE, &faults);
    if (result != FIXED_MAX) return 0;
    if (!faults.overflow) return 0;

    return 1;
}

static int test_sub_underflow_saturates(void)
{
    ct_fault_flags_t faults = {0};

    /* MIN - 1 underflows */
    fixed_t result = dvm_sub(FIXED_MIN, FIXED_ONE, &faults);
    if (result != FIXED_MIN) return 0;
    if (!faults.underflow) return 0;

    return 1;
}

/* ============================================================================
 * Test: dvm_mul - Fixed-Point Multiplication (CT-MATH-001 §3.3)
 * ============================================================================ */

static int test_mul_one(void)
{
    ct_fault_flags_t faults = {0};

    /* Multiply by 1.0 should return same value */
    fixed_t result = dvm_mul(FIXED_ONE * 5, FIXED_ONE, &faults);
    if (result != FIXED_ONE * 5) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_mul_half(void)
{
    ct_fault_flags_t faults = {0};

    /* 2.0 * 0.5 = 1.0 */
    fixed_t result = dvm_mul(2 * FIXED_ONE, FIXED_HALF, &faults);
    if (result != FIXED_ONE) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_mul_negative(void)
{
    ct_fault_flags_t faults = {0};

    /* 2.0 * -1.0 = -2.0 */
    fixed_t result = dvm_mul(2 * FIXED_ONE, -FIXED_ONE, &faults);
    if (result != -2 * FIXED_ONE) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_mul_two_negatives(void)
{
    ct_fault_flags_t faults = {0};

    /* -2.0 * -3.0 = 6.0 */
    fixed_t result = dvm_mul(-2 * FIXED_ONE, -3 * FIXED_ONE, &faults);
    if (result != 6 * FIXED_ONE) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_mul_overflow_saturates(void)
{
    ct_fault_flags_t faults = {0};

    /* Large values should saturate */
    fixed_t big = FIXED_ONE * 1000;
    fixed_t result = dvm_mul(big, big, &faults);
    if (result != FIXED_MAX) return 0;
    if (!faults.overflow) return 0;

    return 1;
}

static int test_mul_zero(void)
{
    ct_fault_flags_t faults = {0};

    /* Anything * 0 = 0 */
    fixed_t result = dvm_mul(FIXED_MAX, FIXED_ZERO, &faults);
    if (result != FIXED_ZERO) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

/* ============================================================================
 * Test: dvm_div_int32 - Integer Division (CT-MATH-001 §3.4)
 * ============================================================================ */

static int test_div_int32_basic(void)
{
    ct_fault_flags_t faults = {0};

    int32_t result = dvm_div_int32(10, 3, &faults);
    if (result != 3) return 0;  /* Truncates toward zero */
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_div_int32_exact(void)
{
    ct_fault_flags_t faults = {0};

    int32_t result = dvm_div_int32(12, 4, &faults);
    if (result != 3) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_div_int32_negative(void)
{
    ct_fault_flags_t faults = {0};

    int32_t result = dvm_div_int32(-10, 3, &faults);
    if (result != -3) return 0;  /* Truncates toward zero */
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_div_int32_by_zero(void)
{
    ct_fault_flags_t faults = {0};

    int32_t result = dvm_div_int32(100, 0, &faults);
    if (result != 0) return 0;
    if (!faults.div_zero) return 0;

    return 1;
}

/* ============================================================================
 * Test: dvm_div_q - Fixed-Point Division (CT-MATH-001 §3.5)
 * ============================================================================ */

static int test_div_q_basic(void)
{
    ct_fault_flags_t faults = {0};

    /* 2.0 / 2.0 = 1.0 */
    fixed_t result = dvm_div_q(2 * FIXED_ONE, 2 * FIXED_ONE, FIXED_FRAC_BITS, &faults);
    if (result != FIXED_ONE) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_div_q_half(void)
{
    ct_fault_flags_t faults = {0};

    /* 1.0 / 2.0 = 0.5 */
    fixed_t result = dvm_div_q(FIXED_ONE, 2 * FIXED_ONE, FIXED_FRAC_BITS, &faults);
    if (result != FIXED_HALF) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_div_q_by_zero(void)
{
    ct_fault_flags_t faults = {0};

    fixed_t result = dvm_div_q(FIXED_ONE, 0, FIXED_FRAC_BITS, &faults);
    if (result != 0) return 0;
    if (!faults.div_zero) return 0;

    return 1;
}

static int test_div_q_shift_too_large(void)
{
    ct_fault_flags_t faults = {0};

    fixed_t result = dvm_div_q(FIXED_ONE, FIXED_ONE, CT_MAX_SHIFT + 1, &faults);
    if (result != 0) return 0;
    if (!faults.domain) return 0;

    return 1;
}

/* ============================================================================
 * Test: dvm_clamp32 - Saturation (CT-MATH-001 §3.6)
 * ============================================================================ */

static int test_clamp32_in_range(void)
{
    ct_fault_flags_t faults = {0};

    int32_t result = dvm_clamp32(12345LL, &faults);
    if (result != 12345) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_clamp32_negative_in_range(void)
{
    ct_fault_flags_t faults = {0};

    int32_t result = dvm_clamp32(-12345LL, &faults);
    if (result != -12345) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_clamp32_overflow(void)
{
    ct_fault_flags_t faults = {0};

    int32_t result = dvm_clamp32((int64_t)INT32_MAX + 1, &faults);
    if (result != INT32_MAX) return 0;
    if (!faults.overflow) return 0;

    return 1;
}

static int test_clamp32_underflow(void)
{
    ct_fault_flags_t faults = {0};

    int32_t result = dvm_clamp32((int64_t)INT32_MIN - 1, &faults);
    if (result != INT32_MIN) return 0;
    if (!faults.underflow) return 0;

    return 1;
}

static int test_clamp32_boundary_max(void)
{
    ct_fault_flags_t faults = {0};

    int32_t result = dvm_clamp32((int64_t)INT32_MAX, &faults);
    if (result != INT32_MAX) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_clamp32_boundary_min(void)
{
    ct_fault_flags_t faults = {0};

    int32_t result = dvm_clamp32((int64_t)INT32_MIN, &faults);
    if (result != INT32_MIN) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

/* ============================================================================
 * Test: dvm_abs64_sat - Safe Absolute Value (CT-MATH-001 §3.7)
 * ============================================================================ */

static int test_abs64_positive(void)
{
    ct_fault_flags_t faults = {0};

    int64_t result = dvm_abs64_sat(12345LL, &faults);
    if (result != 12345LL) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_abs64_negative(void)
{
    ct_fault_flags_t faults = {0};

    int64_t result = dvm_abs64_sat(-12345LL, &faults);
    if (result != 12345LL) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_abs64_zero(void)
{
    ct_fault_flags_t faults = {0};

    int64_t result = dvm_abs64_sat(0LL, &faults);
    if (result != 0LL) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_abs64_int64_min_saturates(void)
{
    ct_fault_flags_t faults = {0};

    /* INT64_MIN has no positive representation - must saturate */
    int64_t result = dvm_abs64_sat(INT64_MIN, &faults);
    if (result != INT64_MAX) return 0;
    if (!faults.overflow) return 0;

    return 1;
}

/* ============================================================================
 * Test: dvm_round_shift_rne - Round-to-Nearest-Even (CT-MATH-001 §8)
 * MANDATORY TEST VECTORS from specification
 * ============================================================================ */

static int test_rne_below_halfway(void)
{
    ct_fault_flags_t faults = {0};

    /* 1.25 in Q16.16 = 0x14000, should round to 1 */
    int32_t result = dvm_round_shift_rne(0x14000LL, 16, &faults);
    if (result != 1) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_rne_above_halfway(void)
{
    ct_fault_flags_t faults = {0};

    /* 1.75 in Q16.16 = 0x1C000, should round to 2 */
    int32_t result = dvm_round_shift_rne(0x1C000LL, 16, &faults);
    if (result != 2) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

/* MANDATORY: CT-MATH-001 Table 8.1 Test Vectors */

static int test_rne_vector_1_5_rounds_to_2(void)
{
    ct_fault_flags_t faults = {0};

    /* 1.5 → 2 (even): 0x18000 >> 16 = 1.5, rounds to 2 */
    int32_t result = dvm_round_shift_rne(0x00018000LL, 16, &faults);
    if (result != 2) {
        printf("\n    FAIL: 1.5 rounded to %d, expected 2\n", result);
        return 0;
    }
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_rne_vector_2_5_rounds_to_2(void)
{
    ct_fault_flags_t faults = {0};

    /* 2.5 → 2 (even): 0x28000 >> 16 = 2.5, rounds to 2 */
    int32_t result = dvm_round_shift_rne(0x00028000LL, 16, &faults);
    if (result != 2) {
        printf("\n    FAIL: 2.5 rounded to %d, expected 2\n", result);
        return 0;
    }
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_rne_vector_3_5_rounds_to_4(void)
{
    ct_fault_flags_t faults = {0};

    /* 3.5 → 4 (even): 0x38000 >> 16 = 3.5, rounds to 4 */
    int32_t result = dvm_round_shift_rne(0x00038000LL, 16, &faults);
    if (result != 4) {
        printf("\n    FAIL: 3.5 rounded to %d, expected 4\n", result);
        return 0;
    }
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_rne_vector_neg_1_5_rounds_to_neg_2(void)
{
    ct_fault_flags_t faults = {0};

    /* -1.5 → -2 (even): 0xFFFE8000 (sign-extended) >> 16 = -1.5, rounds to -2 */
    int64_t neg_1_5 = -0x18000LL;  /* -1.5 in Q16.16 */
    int32_t result = dvm_round_shift_rne(neg_1_5, 16, &faults);
    if (result != -2) {
        printf("\n    FAIL: -1.5 rounded to %d, expected -2\n", result);
        return 0;
    }
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_rne_vector_neg_2_5_rounds_to_neg_2(void)
{
    ct_fault_flags_t faults = {0};

    /* -2.5 → -2 (even): rounds to -2 */
    int64_t neg_2_5 = -0x28000LL;  /* -2.5 in Q16.16 */
    int32_t result = dvm_round_shift_rne(neg_2_5, 16, &faults);
    if (result != -2) {
        printf("\n    FAIL: -2.5 rounded to %d, expected -2\n", result);
        return 0;
    }
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_rne_shift_zero(void)
{
    ct_fault_flags_t faults = {0};

    /* Shift 0 should just clamp */
    int32_t result = dvm_round_shift_rne(12345LL, 0, &faults);
    if (result != 12345) return 0;
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_rne_shift_too_large(void)
{
    ct_fault_flags_t faults = {0};

    /* Shift > 62 should fault */
    int32_t result = dvm_round_shift_rne(12345LL, 63, &faults);
    if (result != 0) return 0;
    if (!faults.domain) return 0;

    return 1;
}

static int test_rne_overflow_saturates(void)
{
    ct_fault_flags_t faults = {0};

    /* Result > INT32_MAX should saturate */
    int64_t big = ((int64_t)INT32_MAX + 100) << 16;
    int32_t result = dvm_round_shift_rne(big, 16, &faults);
    if (result != INT32_MAX) return 0;
    if (!faults.overflow) return 0;

    return 1;
}

/* ============================================================================
 * Test: Fault flag behavior
 * ============================================================================ */

static int test_fault_flags_cleared(void)
{
    ct_fault_flags_t faults = {0};

    /* Generate overflow */
    dvm_add(FIXED_MAX, FIXED_ONE, &faults);
    if (!faults.overflow) return 0;

    /* Clear and verify */
    ct_clear_faults(&faults);
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

static int test_fault_flags_accumulate(void)
{
    ct_fault_flags_t faults = {0};

    /* Generate overflow */
    dvm_add(FIXED_MAX, FIXED_ONE, &faults);

    /* Generate underflow (without clearing) */
    dvm_add(FIXED_MIN, -FIXED_ONE, &faults);

    /* Both should be set */
    if (!faults.overflow) return 0;
    if (!faults.underflow) return 0;

    return 1;
}

static int test_ct_has_fault_function(void)
{
    ct_fault_flags_t faults = {0};

    if (ct_has_fault(&faults)) return 0;  /* Should be false initially */

    faults.overflow = 1;
    if (!ct_has_fault(&faults)) return 0;  /* Should be true now */

    faults.overflow = 0;
    faults.div_zero = 1;
    if (!ct_has_fault(&faults)) return 0;  /* Should still be true */

    return 1;
}

/* ============================================================================
 * Test: Determinism (cross-platform verification)
 * ============================================================================ */

static int test_deterministic_sequence(void)
{
    ct_fault_flags_t faults = {0};

    /* Same operations must produce same results */
    fixed_t a = dvm_mul(FIXED_ONE * 3, FIXED_HALF, &faults);  /* 1.5 */
    fixed_t b = dvm_add(a, FIXED_ONE, &faults);               /* 2.5 */
    fixed_t c = dvm_mul(b, FIXED_ONE * 2, &faults);           /* 5.0 */

    if (a != (FIXED_ONE + FIXED_HALF)) return 0;  /* 1.5 = 0x18000 */
    if (b != (2 * FIXED_ONE + FIXED_HALF)) return 0;  /* 2.5 = 0x28000 */
    if (c != (5 * FIXED_ONE)) return 0;  /* 5.0 = 0x50000 */
    if (ct_has_fault(&faults)) return 0;

    return 1;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void)
{
    printf("==============================================\n");
    printf("Certifiable Training - DVM Primitives Tests\n");
    printf("Traceability: CT-MATH-001 §3, CT-SPEC-001 §3\n");
    printf("==============================================\n\n");

    printf("dvm_add (saturating addition):\n");
    RUN_TEST(test_add_basic);
    RUN_TEST(test_add_negative);
    RUN_TEST(test_add_mixed_sign);
    RUN_TEST(test_add_overflow_saturates);
    RUN_TEST(test_add_underflow_saturates);
    RUN_TEST(test_add_null_faults_safe);

    printf("\ndvm_sub (saturating subtraction):\n");
    RUN_TEST(test_sub_basic);
    RUN_TEST(test_sub_negative_result);
    RUN_TEST(test_sub_overflow_saturates);
    RUN_TEST(test_sub_underflow_saturates);

    printf("\ndvm_mul (fixed-point multiplication):\n");
    RUN_TEST(test_mul_one);
    RUN_TEST(test_mul_half);
    RUN_TEST(test_mul_negative);
    RUN_TEST(test_mul_two_negatives);
    RUN_TEST(test_mul_overflow_saturates);
    RUN_TEST(test_mul_zero);

    printf("\ndvm_div_int32 (integer division):\n");
    RUN_TEST(test_div_int32_basic);
    RUN_TEST(test_div_int32_exact);
    RUN_TEST(test_div_int32_negative);
    RUN_TEST(test_div_int32_by_zero);

    printf("\ndvm_div_q (fixed-point division):\n");
    RUN_TEST(test_div_q_basic);
    RUN_TEST(test_div_q_half);
    RUN_TEST(test_div_q_by_zero);
    RUN_TEST(test_div_q_shift_too_large);

    printf("\ndvm_clamp32 (saturation):\n");
    RUN_TEST(test_clamp32_in_range);
    RUN_TEST(test_clamp32_negative_in_range);
    RUN_TEST(test_clamp32_overflow);
    RUN_TEST(test_clamp32_underflow);
    RUN_TEST(test_clamp32_boundary_max);
    RUN_TEST(test_clamp32_boundary_min);

    printf("\ndvm_abs64_sat (safe absolute value):\n");
    RUN_TEST(test_abs64_positive);
    RUN_TEST(test_abs64_negative);
    RUN_TEST(test_abs64_zero);
    RUN_TEST(test_abs64_int64_min_saturates);

    printf("\ndvm_round_shift_rne (round-to-nearest-even):\n");
    RUN_TEST(test_rne_below_halfway);
    RUN_TEST(test_rne_above_halfway);

    printf("\n  ** MANDATORY CT-MATH-001 §8 Test Vectors **\n");
    RUN_TEST(test_rne_vector_1_5_rounds_to_2);
    RUN_TEST(test_rne_vector_2_5_rounds_to_2);
    RUN_TEST(test_rne_vector_3_5_rounds_to_4);
    RUN_TEST(test_rne_vector_neg_1_5_rounds_to_neg_2);
    RUN_TEST(test_rne_vector_neg_2_5_rounds_to_neg_2);

    printf("\n  ** Edge cases **\n");
    RUN_TEST(test_rne_shift_zero);
    RUN_TEST(test_rne_shift_too_large);
    RUN_TEST(test_rne_overflow_saturates);

    printf("\nFault flag behavior:\n");
    RUN_TEST(test_fault_flags_cleared);
    RUN_TEST(test_fault_flags_accumulate);
    RUN_TEST(test_ct_has_fault_function);

    printf("\nDeterminism verification:\n");
    RUN_TEST(test_deterministic_sequence);

    printf("\n==============================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("==============================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
