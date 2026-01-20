/**
 * @file test_bit_identity.c
 * @project Certifiable Training
 * @brief Cross-platform bit-identity verification tests
 *
 * @details Verifies that all operations produce identical outputs across
 *          platforms (x86_64, ARM64, RISC-V). Tests use known reference
 *          values established on x86_64 Linux. Any platform that produces
 *          different results is non-compliant with the DVM specification.
 *
 * @traceability CT-MATH-001 §3, §6, §8, §16; CT-SPEC-001 Theorem 1 (Bit Identity)
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "ct_types.h"
#include "dvm.h"
#include "prng.h"
#include "forward.h"
#include "merkle.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(fn) do { \
    printf("  %-50s ", #fn); \
    tests_run++; \
    if (fn()) { printf("PASS\n"); tests_passed++; } \
    else { printf("FAIL\n"); } \
} while(0)

/* Helper to print hash for debugging */
static void print_hash(const char *label, const uint8_t hash[CT_HASH_SIZE])
{
    printf("%s: ", label);
    for (int i = 0; i < 8; i++) {
        printf("%02x", hash[i]);
    }
    printf("...\n");
}

/* ============================================================================
 * DVM Primitives - Reference Values (CT-MATH-001 §3)
 * These values are the CANONICAL results across all platforms.
 * ============================================================================ */

static int test_dvm_add_reference(void)
{
    ct_fault_flags_t faults = {0};

    /* Reference: 0x10000 + 0x8000 = 0x18000 (1.0 + 0.5 = 1.5) */
    fixed_t result = dvm_add(FIXED_ONE, FIXED_HALF, &faults);
    if (result != 0x00018000) return 0;

    /* Reference: 0x7FFFFFFF + 0x10000 saturates to 0x7FFFFFFF */
    ct_clear_faults(&faults);
    result = dvm_add(FIXED_MAX, FIXED_ONE, &faults);
    if (result != 0x7FFFFFFF) return 0;
    if (!faults.overflow) return 0;

    return 1;
}

static int test_dvm_mul_reference(void)
{
    ct_fault_flags_t faults = {0};

    /* Reference: 3.0 * 0.5 = 1.5 → 0x18000 */
    fixed_t three = 3 * FIXED_ONE;
    fixed_t result = dvm_mul(three, FIXED_HALF, &faults);
    if (result != 0x00018000) return 0;

    /* Reference: 2.0 * 2.0 = 4.0 → 0x40000 */
    ct_clear_faults(&faults);
    result = dvm_mul(2 * FIXED_ONE, 2 * FIXED_ONE, &faults);
    if (result != 0x00040000) return 0;

    return 1;
}

static int test_dvm_rne_reference_vectors(void)
{
    ct_fault_flags_t faults = {0};

    /* MANDATORY CT-MATH-001 §8 Table: Round-to-Nearest-Even test vectors */

    /* 1.5 → 2 (rounds to even) */
    int32_t r = dvm_round_shift_rne(0x00018000LL, 16, &faults);
    if (r != 2) {
        printf("\n    1.5 -> %d (expected 2)\n", r);
        return 0;
    }

    /* 2.5 → 2 (rounds to even) */
    r = dvm_round_shift_rne(0x00028000LL, 16, &faults);
    if (r != 2) {
        printf("\n    2.5 -> %d (expected 2)\n", r);
        return 0;
    }

    /* 3.5 → 4 (rounds to even) */
    r = dvm_round_shift_rne(0x00038000LL, 16, &faults);
    if (r != 4) {
        printf("\n    3.5 -> %d (expected 4)\n", r);
        return 0;
    }

    /* 4.5 → 4 (rounds to even) */
    r = dvm_round_shift_rne(0x00048000LL, 16, &faults);
    if (r != 4) {
        printf("\n    4.5 -> %d (expected 4)\n", r);
        return 0;
    }

    /* 5.5 → 6 (rounds to even) */
    r = dvm_round_shift_rne(0x00058000LL, 16, &faults);
    if (r != 6) {
        printf("\n    5.5 -> %d (expected 6)\n", r);
        return 0;
    }

    return 1;
}

static int test_dvm_rne_negative_reference(void)
{
    ct_fault_flags_t faults = {0};

    /* -1.5 → -2 (rounds to even) */
    int32_t r = dvm_round_shift_rne(-0x18000LL, 16, &faults);
    if (r != -2) {
        printf("\n    -1.5 -> %d (expected -2)\n", r);
        return 0;
    }

    /* -2.5 → -2 (rounds to even) */
    r = dvm_round_shift_rne(-0x28000LL, 16, &faults);
    if (r != -2) {
        printf("\n    -2.5 -> %d (expected -2)\n", r);
        return 0;
    }

    /* -3.5 → -4 (rounds to even) */
    r = dvm_round_shift_rne(-0x38000LL, 16, &faults);
    if (r != -4) {
        printf("\n    -3.5 -> %d (expected -4)\n", r);
        return 0;
    }

    return 1;
}

/* ============================================================================
 * PRNG - Reference Values (CT-MATH-001 §6)
 * ============================================================================ */

static int test_prng_reference_vectors(void)
{
    ct_prng_t prng;
    ct_prng_init(&prng, 0, 0);

    /* Reference values established on x86_64 Linux */
    uint32_t expected[] = {
        0x24F74A49,
        0xA96E3F40,
        0xC1C8ECFB,
        0xE2E62252,
        0x0AAD3C4D
    };

    for (int i = 0; i < 5; i++) {
        uint32_t v = ct_prng_next(&prng);
        if (v != expected[i]) {
            printf("\n    PRNG[%d] = 0x%08X (expected 0x%08X)\n", i, v, expected[i]);
            return 0;
        }
    }

    return 1;
}

static int test_prng_different_seeds(void)
{
    ct_prng_t prng;

    /* Seed 12345, op_id 100: first value */
    ct_prng_init(&prng, 12345, 100);
    uint32_t v1 = ct_prng_next(&prng);

    /* Same seed/op_id must produce same sequence */
    ct_prng_init(&prng, 12345, 100);
    uint32_t v2 = ct_prng_next(&prng);

    if (v1 != v2) {
        printf("\n    Determinism failed: 0x%08X != 0x%08X\n", v1, v2);
        return 0;
    }

    return 1;
}

static int test_prng_op_id_reference(void)
{
    /* op_id generation must be deterministic */
    uint64_t id1 = ct_prng_make_op_id(0, 0, 0);
    uint64_t id2 = ct_prng_make_op_id(0, 0, 0);

    if (id1 != id2) {
        printf("\n    op_id not deterministic\n");
        return 0;
    }

    /* Different inputs must produce different op_ids */
    uint64_t id3 = ct_prng_make_op_id(1, 0, 0);
    uint64_t id4 = ct_prng_make_op_id(0, 1, 0);
    uint64_t id5 = ct_prng_make_op_id(0, 0, 1);

    if (id1 == id3 || id1 == id4 || id1 == id5) {
        printf("\n    op_id collision detected\n");
        return 0;
    }

    return 1;
}

/* ============================================================================
 * SHA256 - Reference Values (NIST test vectors)
 * ============================================================================ */

static int test_sha256_empty_reference(void)
{
    /* SHA256("") = e3b0c442...  (NIST) */
    uint8_t expected[CT_HASH_SIZE] = {
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
        0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
        0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
        0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55
    };

    uint8_t hash[CT_HASH_SIZE];
    ct_sha256("", 0, hash);

    if (!ct_hash_equal(hash, expected)) {
        printf("\n    SHA256('') mismatch\n");
        print_hash("    Got", hash);
        return 0;
    }

    return 1;
}

static int test_sha256_abc_reference(void)
{
    /* SHA256("abc") = ba7816bf... (NIST) */
    uint8_t expected[CT_HASH_SIZE] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
    };

    uint8_t hash[CT_HASH_SIZE];
    ct_sha256("abc", 3, hash);

    if (!ct_hash_equal(hash, expected)) {
        printf("\n    SHA256('abc') mismatch\n");
        print_hash("    Got", hash);
        return 0;
    }

    return 1;
}

static int test_sha256_determinism(void)
{
    const char *data = "Certifiable Training";
    uint8_t hash1[CT_HASH_SIZE], hash2[CT_HASH_SIZE];

    ct_sha256(data, strlen(data), hash1);
    ct_sha256(data, strlen(data), hash2);

    if (!ct_hash_equal(hash1, hash2)) {
        printf("\n    SHA256 not deterministic\n");
        return 0;
    }

    return 1;
}

/* ============================================================================
 * Fixed-Point Arithmetic Chain (CT-MATH-001 §7)
 * ============================================================================ */

static int test_arithmetic_chain_reference(void)
{
    ct_fault_flags_t faults = {0};

    /* Compute: ((3.0 * 0.5) + 1.0) * 2.0 = 5.0 */
    /* Step by step with reference values */

    fixed_t a = dvm_mul(3 * FIXED_ONE, FIXED_HALF, &faults);   /* 1.5 */
    if (a != 0x00018000) {
        printf("\n    Step 1: 0x%08X (expected 0x00018000)\n", a);
        return 0;
    }

    fixed_t b = dvm_add(a, FIXED_ONE, &faults);                 /* 2.5 */
    if (b != 0x00028000) {
        printf("\n    Step 2: 0x%08X (expected 0x00028000)\n", b);
        return 0;
    }

    fixed_t c = dvm_mul(b, 2 * FIXED_ONE, &faults);            /* 5.0 */
    if (c != 0x00050000) {
        printf("\n    Step 3: 0x%08X (expected 0x00050000)\n", c);
        return 0;
    }

    if (ct_has_fault(&faults)) {
        printf("\n    Unexpected fault\n");
        return 0;
    }

    return 1;
}

static int test_division_chain_reference(void)
{
    ct_fault_flags_t faults = {0};

    /* 4.0 / 2.0 = 2.0 */
    fixed_t a = dvm_div_q(4 * FIXED_ONE, 2 * FIXED_ONE, FIXED_FRAC_BITS, &faults);
    if (a != 2 * FIXED_ONE) {
        printf("\n    4/2 = 0x%08X (expected 0x%08X)\n", a, 2 * FIXED_ONE);
        return 0;
    }

    /* 1.0 / 4.0 = 0.25 */
    fixed_t b = dvm_div_q(FIXED_ONE, 4 * FIXED_ONE, FIXED_FRAC_BITS, &faults);
    if (b != FIXED_ONE / 4) {
        printf("\n    1/4 = 0x%08X (expected 0x%08X)\n", b, FIXED_ONE / 4);
        return 0;
    }

    return 1;
}

/* ============================================================================
 * Activation Functions (CT-MATH-001 §12)
 * ============================================================================ */

static int test_relu_reference(void)
{
    /* ReLU is trivial but must be consistent */
    if (ct_relu(FIXED_ONE) != FIXED_ONE) return 0;
    if (ct_relu(-FIXED_ONE) != 0) return 0;
    if (ct_relu(0) != 0) return 0;
    if (ct_relu(FIXED_MAX) != FIXED_MAX) return 0;

    return 1;
}

static int test_sigmoid_lut_reference(void)
{
    ct_activation_lut_t lut;
    ct_activation_init_sigmoid_lut(&lut);

    /* sigmoid(0) ≈ 0.5 */
    fixed_t s0 = ct_sigmoid(0, &lut);
    /* Allow small tolerance due to LUT interpolation */
    if (s0 < FIXED_HALF - 100 || s0 > FIXED_HALF + 100) {
        printf("\n    sigmoid(0) = 0x%08X (expected ~0x%08X)\n", s0, FIXED_HALF);
        return 0;
    }

    /* sigmoid(large positive) ≈ 1.0 */
    fixed_t s_big = ct_sigmoid(8 * FIXED_ONE, &lut);
    if (s_big < FIXED_ONE - 1000) {
        printf("\n    sigmoid(8) = 0x%08X (expected ~0x%08X)\n", s_big, FIXED_ONE);
        return 0;
    }

    /* sigmoid(large negative) ≈ 0.0 */
    fixed_t s_neg = ct_sigmoid(-8 * FIXED_ONE, &lut);
    if (s_neg > 1000) {
        printf("\n    sigmoid(-8) = 0x%08X (expected ~0)\n", s_neg);
        return 0;
    }

    return 1;
}

/* ============================================================================
 * Matrix Operations (CT-MATH-001 §7.1)
 * ============================================================================ */

static int test_matvec_reference(void)
{
    ct_fault_flags_t faults = {0};

    /* Simple 2x2 matrix × 2-vector */
    /* A = [[1, 2], [3, 4]] in Q16.16 */
    /* x = [1, 1] in Q16.16 */
    /* y = A*x = [3, 7] in Q16.16 */

    fixed_t A[4] = {
        1 * FIXED_ONE, 2 * FIXED_ONE,
        3 * FIXED_ONE, 4 * FIXED_ONE
    };
    fixed_t x[2] = { FIXED_ONE, FIXED_ONE };
    fixed_t y[2] = { 0, 0 };

    ct_matvec_mul(A, x, y, 2, 2, &faults);

    if (y[0] != 3 * FIXED_ONE) {
        printf("\n    y[0] = 0x%08X (expected 0x%08X)\n", y[0], 3 * FIXED_ONE);
        return 0;
    }
    if (y[1] != 7 * FIXED_ONE) {
        printf("\n    y[1] = 0x%08X (expected 0x%08X)\n", y[1], 7 * FIXED_ONE);
        return 0;
    }

    return 1;
}

static int test_dot_product_reference(void)
{
    ct_fault_flags_t faults = {0};

    /* [1, 2, 3] · [4, 5, 6] = 4 + 10 + 18 = 32 */
    fixed_t a[3] = { 1 * FIXED_ONE, 2 * FIXED_ONE, 3 * FIXED_ONE };
    fixed_t b[3] = { 4 * FIXED_ONE, 5 * FIXED_ONE, 6 * FIXED_ONE };

    fixed_t dot = ct_dot_product(a, b, 3, &faults);

    if (dot != 32 * FIXED_ONE) {
        printf("\n    dot = 0x%08X (expected 0x%08X)\n", dot, 32 * FIXED_ONE);
        return 0;
    }

    return 1;
}

/* ============================================================================
 * Tensor Hashing (CT-MATH-001 §17)
 * ============================================================================ */

static int test_tensor_hash_determinism(void)
{
    fixed_t data[4] = { FIXED_ONE, 2 * FIXED_ONE, 3 * FIXED_ONE, 4 * FIXED_ONE };
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, data, 4);

    uint8_t hash1[CT_HASH_SIZE], hash2[CT_HASH_SIZE];

    ct_tensor_hash(&tensor, hash1);
    ct_tensor_hash(&tensor, hash2);

    if (!ct_hash_equal(hash1, hash2)) {
        printf("\n    Tensor hash not deterministic\n");
        return 0;
    }

    return 1;
}

static int test_tensor_hash_changes_with_data(void)
{
    fixed_t data1[4] = { FIXED_ONE, 2 * FIXED_ONE, 3 * FIXED_ONE, 4 * FIXED_ONE };
    fixed_t data2[4] = { FIXED_ONE, 2 * FIXED_ONE, 3 * FIXED_ONE, 5 * FIXED_ONE };

    ct_tensor_t tensor1, tensor2;
    ct_tensor_init_1d(&tensor1, data1, 4);
    ct_tensor_init_1d(&tensor2, data2, 4);

    uint8_t hash1[CT_HASH_SIZE], hash2[CT_HASH_SIZE];

    ct_tensor_hash(&tensor1, hash1);
    ct_tensor_hash(&tensor2, hash2);

    if (ct_hash_equal(hash1, hash2)) {
        printf("\n    Different tensors produced same hash!\n");
        return 0;
    }

    return 1;
}

/* ============================================================================
 * Stochastic Rounding (CT-MATH-001 §8.4)
 * ============================================================================ */

static int test_stochastic_round_determinism(void)
{
    ct_prng_t prng1, prng2;
    ct_fault_flags_t faults = {0};

    ct_prng_init(&prng1, 12345, 500);
    ct_prng_init(&prng2, 12345, 500);

    /* Same PRNG state must produce same stochastic rounding result */
    int64_t x = 0x18000LL;  /* 1.5 in Q16.16 */

    int32_t r1 = ct_stochastic_round(x, 16, &prng1, &faults);
    int32_t r2 = ct_stochastic_round(x, 16, &prng2, &faults);

    if (r1 != r2) {
        printf("\n    Stochastic round not deterministic: %d != %d\n", r1, r2);
        return 0;
    }

    return 1;
}

/* ============================================================================
 * Full Pipeline Test
 * ============================================================================ */

static int test_full_forward_pass_determinism(void)
{
    ct_fault_flags_t faults = {0};

    /* Create a simple linear layer: 2 inputs → 2 outputs */
    fixed_t weights_buf[4] = {
        FIXED_ONE,      FIXED_HALF,     /* W[0,0], W[0,1] */
        FIXED_HALF,     FIXED_ONE       /* W[1,0], W[1,1] */
    };
    fixed_t bias_buf[2] = { 0, 0 };

    ct_linear_t layer;
    ct_linear_init(&layer, weights_buf, bias_buf, 2, 2);

    /* Input: [1.0, 1.0] */
    fixed_t input_buf[2] = { FIXED_ONE, FIXED_ONE };
    ct_tensor_t input;
    ct_tensor_init_1d(&input, input_buf, 2);

    /* Output buffers */
    fixed_t output_buf1[2], output_buf2[2];
    ct_tensor_t output1, output2;
    ct_tensor_init_1d(&output1, output_buf1, 2);
    ct_tensor_init_1d(&output2, output_buf2, 2);

    /* Run forward pass twice */
    ct_linear_forward(&layer, &input, &output1, &faults);
    ct_linear_forward(&layer, &input, &output2, &faults);

    /* Results must be identical */
    if (output_buf1[0] != output_buf2[0] || output_buf1[1] != output_buf2[1]) {
        printf("\n    Forward pass not deterministic\n");
        printf("    Run 1: [0x%08X, 0x%08X]\n", output_buf1[0], output_buf1[1]);
        printf("    Run 2: [0x%08X, 0x%08X]\n", output_buf2[0], output_buf2[1]);
        return 0;
    }

    /* Verify expected values: y = Wx = [[1, 0.5], [0.5, 1]] * [1, 1] = [1.5, 1.5] */
    if (output_buf1[0] != (FIXED_ONE + FIXED_HALF)) {
        printf("\n    y[0] = 0x%08X (expected 0x%08X)\n",
               output_buf1[0], FIXED_ONE + FIXED_HALF);
        return 0;
    }

    return 1;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void)
{
    printf("==============================================\n");
    printf("Certifiable Training - Bit Identity Tests\n");
    printf("Traceability: CT-MATH-001, CT-SPEC-001 Theorem 1\n");
    printf("==============================================\n\n");

    printf("** These tests verify cross-platform bit-identity **\n");
    printf("** Failure indicates DVM non-compliance **\n\n");

    printf("DVM Primitives (CT-MATH-001 §3):\n");
    RUN_TEST(test_dvm_add_reference);
    RUN_TEST(test_dvm_mul_reference);

    printf("\nRound-to-Nearest-Even (CT-MATH-001 §8):\n");
    RUN_TEST(test_dvm_rne_reference_vectors);
    RUN_TEST(test_dvm_rne_negative_reference);

    printf("\nPRNG Reference Vectors (CT-MATH-001 §6):\n");
    RUN_TEST(test_prng_reference_vectors);
    RUN_TEST(test_prng_different_seeds);
    RUN_TEST(test_prng_op_id_reference);

    printf("\nSHA256 Reference Vectors (NIST FIPS 180-4):\n");
    RUN_TEST(test_sha256_empty_reference);
    RUN_TEST(test_sha256_abc_reference);
    RUN_TEST(test_sha256_determinism);

    printf("\nArithmetic Chains (CT-MATH-001 §7):\n");
    RUN_TEST(test_arithmetic_chain_reference);
    RUN_TEST(test_division_chain_reference);

    printf("\nActivation Functions (CT-MATH-001 §12):\n");
    RUN_TEST(test_relu_reference);
    RUN_TEST(test_sigmoid_lut_reference);

    printf("\nMatrix Operations (CT-MATH-001 §7.1):\n");
    RUN_TEST(test_matvec_reference);
    RUN_TEST(test_dot_product_reference);

    printf("\nTensor Hashing (CT-MATH-001 §17):\n");
    RUN_TEST(test_tensor_hash_determinism);
    RUN_TEST(test_tensor_hash_changes_with_data);

    printf("\nStochastic Rounding (CT-MATH-001 §8.4):\n");
    RUN_TEST(test_stochastic_round_determinism);

    printf("\nFull Pipeline Determinism:\n");
    RUN_TEST(test_full_forward_pass_determinism);

    printf("\n==============================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    if (tests_passed == tests_run) {
        printf("STATUS: Platform is DVM-compliant\n");
    } else {
        printf("STATUS: *** PLATFORM NON-COMPLIANT ***\n");
    }
    printf("==============================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
