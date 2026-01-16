/**
 * @file test_permutation.c
 * @project Certifiable Training
 * @brief Unit tests for Cycle-Walking Feistel permutation.
 *
 * @traceability SRS-009-PERMUTATION
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust. All rights reserved.
 * @license Licensed under the GPL-3.0 (Open Source) or Commercial License.
 *          For commercial licensing: william@fstopify.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "permutation.h"

/* ============================================================================
 * Test Framework
 * ============================================================================ */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN_TEST(name) do { \
    printf("  Testing %s... ", #name); \
    test_##name(); \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAIL\n    Assertion failed: %s\n    at %s:%d\n", \
               #cond, __FILE__, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_LT(a, b) ASSERT((a) < (b))

/* ============================================================================
 * Test: Initialization
 * ============================================================================ */

TEST(init_basic) {
    ct_permutation_t perm;
    ct_error_t err = ct_permutation_init(&perm, 12345, 0, 100);
    
    ASSERT_EQ(err, CT_OK);
    ASSERT(perm.initialized);
    ASSERT_EQ(perm.seed, 12345);
    ASSERT_EQ(perm.epoch, 0);
    ASSERT_EQ(perm.dataset_size, 100);
}

TEST(init_power_of_two) {
    ct_permutation_t perm;
    ct_error_t err = ct_permutation_init(&perm, 42, 0, 256);
    
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(perm.range, 256);  /* 256 = 2^8, already even bits */
}

TEST(init_non_power_of_two) {
    ct_permutation_t perm;
    ct_error_t err = ct_permutation_init(&perm, 42, 0, 100);
    
    ASSERT_EQ(err, CT_OK);
    /* 100 needs 7 bits (ceil(log2(100))=7), rounded up to 8 bits for balanced Feistel */
    ASSERT_EQ(perm.range, 256);  /* 2^8 = 256 */
}

TEST(init_null) {
    ASSERT_EQ(ct_permutation_init(NULL, 0, 0, 100), CT_ERR_NULL);
}

TEST(init_zero_size) {
    ct_permutation_t perm;
    ASSERT_EQ(ct_permutation_init(&perm, 0, 0, 0), CT_ERR_DIMENSION);
}

TEST(init_size_one) {
    ct_permutation_t perm;
    ct_error_t err = ct_permutation_init(&perm, 42, 0, 1);
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(perm.dataset_size, 1);
}

/* ============================================================================
 * Test: Feistel Hash
 * ============================================================================ */

TEST(feistel_hash_deterministic) {
    uint32_t h1 = ct_feistel_hash(12345, 0, 0, 42);
    uint32_t h2 = ct_feistel_hash(12345, 0, 0, 42);
    
    ASSERT_EQ(h1, h2);
}

TEST(feistel_hash_varies_with_seed) {
    uint32_t h1 = ct_feistel_hash(12345, 0, 0, 42);
    uint32_t h2 = ct_feistel_hash(12346, 0, 0, 42);
    
    ASSERT_NE(h1, h2);
}

TEST(feistel_hash_varies_with_epoch) {
    uint32_t h1 = ct_feistel_hash(12345, 0, 0, 42);
    uint32_t h2 = ct_feistel_hash(12345, 1, 0, 42);
    
    ASSERT_NE(h1, h2);
}

TEST(feistel_hash_varies_with_round) {
    uint32_t h1 = ct_feistel_hash(12345, 0, 0, 42);
    uint32_t h2 = ct_feistel_hash(12345, 0, 1, 42);
    
    ASSERT_NE(h1, h2);
}

TEST(feistel_hash_varies_with_value) {
    uint32_t h1 = ct_feistel_hash(12345, 0, 0, 42);
    uint32_t h2 = ct_feistel_hash(12345, 0, 0, 43);
    
    ASSERT_NE(h1, h2);
}

/* ============================================================================
 * Test: Permutation Apply
 * ============================================================================ */

TEST(apply_in_range) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 42, 0, 100);
    ct_fault_flags_t faults = {0};
    
    for (uint32_t i = 0; i < 100; i++) {
        uint32_t j = ct_permutation_apply(&perm, i, &faults);
        ASSERT_LT(j, 100);
    }
    
    ASSERT(!ct_has_fault(&faults));
}

TEST(apply_deterministic) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 42, 0, 100);
    ct_fault_flags_t faults = {0};
    
    uint32_t j1 = ct_permutation_apply(&perm, 50, &faults);
    uint32_t j2 = ct_permutation_apply(&perm, 50, &faults);
    
    ASSERT_EQ(j1, j2);
}

TEST(apply_size_one) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 42, 0, 1);
    ct_fault_flags_t faults = {0};
    
    uint32_t j = ct_permutation_apply(&perm, 0, &faults);
    ASSERT_EQ(j, 0);
}

TEST(apply_shuffles) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 42, 0, 10);
    ct_fault_flags_t faults = {0};
    
    /* Count how many indices stay in place */
    int unchanged = 0;
    for (uint32_t i = 0; i < 10; i++) {
        if (ct_permutation_apply(&perm, i, &faults) == i) {
            unchanged++;
        }
    }
    
    /* Very unlikely all stay in place (1/10! chance) */
    ASSERT(unchanged < 10);
}

/* ============================================================================
 * Test: Bijection Verification
 * ============================================================================ */

TEST(bijection_small) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 42, 0, 10);
    ct_fault_flags_t faults = {0};
    
    ASSERT(ct_permutation_verify_bijection(&perm, &faults));
}

TEST(bijection_power_of_two) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 12345, 0, 64);
    ct_fault_flags_t faults = {0};
    
    ASSERT(ct_permutation_verify_bijection(&perm, &faults));
}

TEST(bijection_non_power_of_two) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 12345, 0, 100);
    ct_fault_flags_t faults = {0};
    
    ASSERT(ct_permutation_verify_bijection(&perm, &faults));
}

TEST(bijection_prime) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 42, 0, 97);  /* Prime number */
    ct_fault_flags_t faults = {0};
    
    ASSERT(ct_permutation_verify_bijection(&perm, &faults));
}

TEST(bijection_different_epochs) {
    ct_permutation_t perm;
    ct_fault_flags_t faults = {0};
    
    for (uint32_t epoch = 0; epoch < 5; epoch++) {
        ct_permutation_init(&perm, 42, epoch, 50);
        ASSERT(ct_permutation_verify_bijection(&perm, &faults));
    }
}

/* ============================================================================
 * Test: Inverse
 * ============================================================================ */

TEST(inverse_roundtrip) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 42, 0, 100);
    ct_fault_flags_t faults = {0};
    
    for (uint32_t i = 0; i < 100; i++) {
        uint32_t j = ct_permutation_apply(&perm, i, &faults);
        uint32_t k = ct_permutation_inverse(&perm, j, &faults);
        ASSERT_EQ(k, i);
    }
}

TEST(inverse_roundtrip_reverse) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 42, 0, 100);
    ct_fault_flags_t faults = {0};
    
    for (uint32_t j = 0; j < 100; j++) {
        uint32_t i = ct_permutation_inverse(&perm, j, &faults);
        uint32_t k = ct_permutation_apply(&perm, i, &faults);
        ASSERT_EQ(k, j);
    }
}

/* ============================================================================
 * Test: Epoch Changes Permutation
 * ============================================================================ */

TEST(epoch_changes_output) {
    ct_permutation_t perm1, perm2;
    ct_permutation_init(&perm1, 42, 0, 100);
    ct_permutation_init(&perm2, 42, 1, 100);
    ct_fault_flags_t faults = {0};
    
    /* Count differences */
    int different = 0;
    for (uint32_t i = 0; i < 100; i++) {
        if (ct_permutation_apply(&perm1, i, &faults) != 
            ct_permutation_apply(&perm2, i, &faults)) {
            different++;
        }
    }
    
    /* Most should be different */
    ASSERT(different > 50);
}

TEST(set_epoch) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 42, 0, 100);
    ct_fault_flags_t faults = {0};
    
    uint32_t j1 = ct_permutation_apply(&perm, 50, &faults);
    
    ct_permutation_set_epoch(&perm, 1);
    uint32_t j2 = ct_permutation_apply(&perm, 50, &faults);
    
    /* Different epoch should give different result */
    ASSERT_NE(j1, j2);
}

/* ============================================================================
 * Test: Batch Operations
 * ============================================================================ */

TEST(batch_init) {
    ct_batch_ctx_t ctx;
    ct_error_t err = ct_batch_init(&ctx, 42, 0, 100, 10);
    
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(ctx.batch_size, 10);
    ASSERT_EQ(ctx.steps_per_epoch, 10);  /* 100 / 10 */
}

TEST(batch_init_non_divisible) {
    ct_batch_ctx_t ctx;
    ct_error_t err = ct_batch_init(&ctx, 42, 0, 100, 30);
    
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(ctx.steps_per_epoch, 4);  /* ceil(100 / 30) */
}

TEST(batch_get_indices) {
    ct_batch_ctx_t ctx;
    ct_batch_init(&ctx, 42, 0, 100, 10);
    ct_fault_flags_t faults = {0};
    
    uint32_t indices[10];
    ct_error_t err = ct_batch_get_indices(&ctx, 0, indices, &faults);
    
    ASSERT_EQ(err, CT_OK);
    
    /* All indices should be in range */
    for (int i = 0; i < 10; i++) {
        ASSERT_LT(indices[i], 100);
    }
}

TEST(batch_indices_deterministic) {
    ct_batch_ctx_t ctx;
    ct_batch_init(&ctx, 42, 0, 100, 10);
    ct_fault_flags_t faults = {0};
    
    uint32_t indices1[10], indices2[10];
    ct_batch_get_indices(&ctx, 5, indices1, &faults);
    ct_batch_get_indices(&ctx, 5, indices2, &faults);
    
    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(indices1[i], indices2[i]);
    }
}

TEST(batch_indices_different_steps) {
    ct_batch_ctx_t ctx;
    ct_batch_init(&ctx, 42, 0, 100, 10);
    ct_fault_flags_t faults = {0};
    
    uint32_t indices1[10], indices2[10];
    ct_batch_get_indices(&ctx, 0, indices1, &faults);
    ct_batch_get_indices(&ctx, 1, indices2, &faults);
    
    /* At least some should differ */
    int same = 0;
    for (int i = 0; i < 10; i++) {
        if (indices1[i] == indices2[i]) same++;
    }
    ASSERT(same < 10);
}

TEST(batch_get_size_full) {
    ct_batch_ctx_t ctx;
    ct_batch_init(&ctx, 42, 0, 100, 10);
    
    ASSERT_EQ(ct_batch_get_size(&ctx, 0), 10);
    ASSERT_EQ(ct_batch_get_size(&ctx, 5), 10);
}

TEST(batch_get_size_partial) {
    ct_batch_ctx_t ctx;
    ct_batch_init(&ctx, 42, 0, 95, 10);  /* 95 samples, batch 10 */
    
    /* steps_per_epoch = ceil(95/10) = 10 */
    /* Last batch (step 9) has only 5 samples */
    ASSERT_EQ(ct_batch_get_size(&ctx, 0), 10);
    ASSERT_EQ(ct_batch_get_size(&ctx, 9), 5);
}

TEST(batch_step_in_epoch) {
    ct_batch_ctx_t ctx;
    ct_batch_init(&ctx, 42, 0, 100, 10);
    
    ASSERT_EQ(ct_batch_step_in_epoch(&ctx, 0), 0);
    ASSERT_EQ(ct_batch_step_in_epoch(&ctx, 5), 5);
    ASSERT_EQ(ct_batch_step_in_epoch(&ctx, 10), 0);  /* New epoch */
    ASSERT_EQ(ct_batch_step_in_epoch(&ctx, 15), 5);
}

TEST(batch_get_epoch) {
    ct_batch_ctx_t ctx;
    ct_batch_init(&ctx, 42, 0, 100, 10);
    
    ASSERT_EQ(ct_batch_get_epoch(&ctx, 0), 0);
    ASSERT_EQ(ct_batch_get_epoch(&ctx, 9), 0);
    ASSERT_EQ(ct_batch_get_epoch(&ctx, 10), 1);
    ASSERT_EQ(ct_batch_get_epoch(&ctx, 25), 2);
}

/* ============================================================================
 * Test: Larger Dataset
 * ============================================================================ */

TEST(large_dataset_bijection) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 12345, 0, 10000);
    ct_fault_flags_t faults = {0};
    
    ASSERT(ct_permutation_verify_bijection(&perm, &faults));
}

TEST(large_dataset_inverse) {
    ct_permutation_t perm;
    ct_permutation_init(&perm, 12345, 0, 1000);
    ct_fault_flags_t faults = {0};
    
    /* Spot check some indices */
    for (uint32_t i = 0; i < 1000; i += 100) {
        uint32_t j = ct_permutation_apply(&perm, i, &faults);
        uint32_t k = ct_permutation_inverse(&perm, j, &faults);
        ASSERT_EQ(k, i);
    }
}

/* ============================================================================
 * Test: Error Handling
 * ============================================================================ */

TEST(null_pointer_handling) {
    ct_fault_flags_t faults = {0};
    
    ASSERT_EQ(ct_permutation_init(NULL, 0, 0, 100), CT_ERR_NULL);
    ASSERT_EQ(ct_batch_init(NULL, 0, 0, 100, 10), CT_ERR_NULL);
    ASSERT_EQ(ct_batch_get_indices(NULL, 0, NULL, &faults), CT_ERR_NULL);
}

TEST(uninitialized_context) {
    ct_permutation_t perm = {0};
    ct_fault_flags_t faults = {0};
    
    uint32_t result = ct_permutation_apply(&perm, 0, &faults);
    (void)result;  /* Silence unused variable warning */
    ASSERT(faults.domain);
}

/* ============================================================================
 * Test: Determinism
 * ============================================================================ */

TEST(full_determinism) {
    ct_batch_ctx_t ctx1, ctx2;
    ct_batch_init(&ctx1, 42, 0, 100, 10);
    ct_batch_init(&ctx2, 42, 0, 100, 10);
    ct_fault_flags_t faults1 = {0}, faults2 = {0};
    
    /* Run several batches */
    for (uint64_t step = 0; step < 20; step++) {
        uint32_t indices1[10], indices2[10];
        ct_batch_get_indices(&ctx1, step, indices1, &faults1);
        ct_batch_get_indices(&ctx2, step, indices2, &faults2);
        
        for (int i = 0; i < 10; i++) {
            ASSERT_EQ(indices1[i], indices2[i]);
        }
    }
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== SRS-009: Permutation Tests ===\n\n");
    
    printf("Initialization Tests:\n");
    RUN_TEST(init_basic);
    RUN_TEST(init_power_of_two);
    RUN_TEST(init_non_power_of_two);
    RUN_TEST(init_null);
    RUN_TEST(init_zero_size);
    RUN_TEST(init_size_one);
    
    printf("\nFeistel Hash Tests:\n");
    RUN_TEST(feistel_hash_deterministic);
    RUN_TEST(feistel_hash_varies_with_seed);
    RUN_TEST(feistel_hash_varies_with_epoch);
    RUN_TEST(feistel_hash_varies_with_round);
    RUN_TEST(feistel_hash_varies_with_value);
    
    printf("\nPermutation Apply Tests:\n");
    RUN_TEST(apply_in_range);
    RUN_TEST(apply_deterministic);
    RUN_TEST(apply_size_one);
    RUN_TEST(apply_shuffles);
    
    printf("\nBijection Tests:\n");
    RUN_TEST(bijection_small);
    RUN_TEST(bijection_power_of_two);
    RUN_TEST(bijection_non_power_of_two);
    RUN_TEST(bijection_prime);
    RUN_TEST(bijection_different_epochs);
    
    printf("\nInverse Tests:\n");
    RUN_TEST(inverse_roundtrip);
    RUN_TEST(inverse_roundtrip_reverse);
    
    printf("\nEpoch Tests:\n");
    RUN_TEST(epoch_changes_output);
    RUN_TEST(set_epoch);
    
    printf("\nBatch Tests:\n");
    RUN_TEST(batch_init);
    RUN_TEST(batch_init_non_divisible);
    RUN_TEST(batch_get_indices);
    RUN_TEST(batch_indices_deterministic);
    RUN_TEST(batch_indices_different_steps);
    RUN_TEST(batch_get_size_full);
    RUN_TEST(batch_get_size_partial);
    RUN_TEST(batch_step_in_epoch);
    RUN_TEST(batch_get_epoch);
    
    printf("\nLarge Dataset Tests:\n");
    RUN_TEST(large_dataset_bijection);
    RUN_TEST(large_dataset_inverse);
    
    printf("\nError Handling Tests:\n");
    RUN_TEST(null_pointer_handling);
    RUN_TEST(uninitialized_context);
    
    printf("\nDeterminism Tests:\n");
    RUN_TEST(full_determinism);
    
    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    
    return tests_failed > 0 ? 1 : 0;
}
