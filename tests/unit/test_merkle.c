/**
 * @file test_merkle.c
 * @project Certifiable Training
 * @brief Unit tests for Merkle training chain.
 *
 * @traceability SRS-008-MERKLE
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
#include "merkle.h"
#include "forward.h"
#include "dvm.h"

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

/* ============================================================================
 * Test: SHA256
 * ============================================================================ */

TEST(sha256_empty) {
    /* SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 */
    uint8_t hash[CT_HASH_SIZE];
    ct_sha256("", 0, hash);
    
    ASSERT_EQ(hash[0], 0xe3);
    ASSERT_EQ(hash[1], 0xb0);
    ASSERT_EQ(hash[2], 0xc4);
    ASSERT_EQ(hash[3], 0x42);
    ASSERT_EQ(hash[31], 0x55);
}

TEST(sha256_abc) {
    /* SHA256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad */
    uint8_t hash[CT_HASH_SIZE];
    ct_sha256("abc", 3, hash);
    
    ASSERT_EQ(hash[0], 0xba);
    ASSERT_EQ(hash[1], 0x78);
    ASSERT_EQ(hash[2], 0x16);
    ASSERT_EQ(hash[3], 0xbf);
    ASSERT_EQ(hash[31], 0xad);
}

TEST(sha256_incremental) {
    /* Test incremental hashing produces same result */
    uint8_t hash1[CT_HASH_SIZE];
    uint8_t hash2[CT_HASH_SIZE];
    
    /* One-shot */
    ct_sha256("hello world", 11, hash1);
    
    /* Incremental */
    ct_sha256_ctx_t ctx;
    ct_sha256_init(&ctx);
    ct_sha256_update(&ctx, "hello", 5);
    ct_sha256_update(&ctx, " ", 1);
    ct_sha256_update(&ctx, "world", 5);
    ct_sha256_final(&ctx, hash2);
    
    ASSERT(ct_hash_equal(hash1, hash2));
}

/* ============================================================================
 * Test: Hash Utilities
 * ============================================================================ */

TEST(hash_equal) {
    uint8_t a[CT_HASH_SIZE] = {0};
    uint8_t b[CT_HASH_SIZE] = {0};
    
    ASSERT(ct_hash_equal(a, b));
    
    b[0] = 1;
    ASSERT(!ct_hash_equal(a, b));
    
    b[0] = 0;
    b[31] = 0xff;
    ASSERT(!ct_hash_equal(a, b));
}

TEST(hash_copy) {
    uint8_t src[CT_HASH_SIZE];
    uint8_t dst[CT_HASH_SIZE] = {0};
    
    for (int i = 0; i < CT_HASH_SIZE; i++) {
        src[i] = (uint8_t)i;
    }
    
    ct_hash_copy(dst, src);
    ASSERT(ct_hash_equal(dst, src));
}

TEST(hash_zero) {
    uint8_t hash[CT_HASH_SIZE];
    for (int i = 0; i < CT_HASH_SIZE; i++) {
        hash[i] = 0xff;
    }
    
    ct_hash_zero(hash);
    
    for (int i = 0; i < CT_HASH_SIZE; i++) {
        ASSERT_EQ(hash[i], 0);
    }
}

/* ============================================================================
 * Test: Tensor Serialization
 * ============================================================================ */

TEST(tensor_is_contiguous) {
    fixed_t data[6];
    ct_tensor_t tensor;
    
    /* 1D tensor is always contiguous */
    ct_tensor_init_1d(&tensor, data, 6);
    ASSERT(ct_tensor_is_contiguous(&tensor));
    
    /* 2D contiguous */
    ct_tensor_init_2d(&tensor, data, 2, 3);
    ASSERT(ct_tensor_is_contiguous(&tensor));
}

TEST(tensor_serial_size) {
    fixed_t data[10];
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, data, 10);
    
    /* Header: 4 + 4 + 4 + 4*4 + 8 = 36 bytes */
    /* Data: 10 * 4 = 40 bytes */
    /* Total: 76 bytes */
    size_t size = ct_tensor_serial_size(&tensor);
    ASSERT_EQ(size, 36 + 40);
}

TEST(tensor_serialize) {
    fixed_t data[4] = {FIXED_ONE, -FIXED_ONE, FIXED_HALF, 0};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, data, 4);
    
    uint8_t buffer[100];
    int32_t written = ct_tensor_serialize(&tensor, buffer, sizeof(buffer));
    
    ASSERT(written > 0);
    ASSERT_EQ((size_t)written, ct_tensor_serial_size(&tensor));
    
    /* Check version (little-endian) */
    ASSERT_EQ(buffer[0], CT_SERIALIZE_VERSION);
    ASSERT_EQ(buffer[1], 0);
    ASSERT_EQ(buffer[2], 0);
    ASSERT_EQ(buffer[3], 0);
}

TEST(tensor_hash_deterministic) {
    fixed_t data[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, data, 4);
    
    uint8_t hash1[CT_HASH_SIZE];
    uint8_t hash2[CT_HASH_SIZE];
    
    ct_tensor_hash(&tensor, hash1);
    ct_tensor_hash(&tensor, hash2);
    
    /* Same data must produce same hash */
    ASSERT(ct_hash_equal(hash1, hash2));
}

TEST(tensor_hash_different) {
    fixed_t data1[4] = {1, 2, 3, 4};
    fixed_t data2[4] = {1, 2, 3, 5};
    ct_tensor_t tensor1, tensor2;
    ct_tensor_init_1d(&tensor1, data1, 4);
    ct_tensor_init_1d(&tensor2, data2, 4);
    
    uint8_t hash1[CT_HASH_SIZE];
    uint8_t hash2[CT_HASH_SIZE];
    
    ct_tensor_hash(&tensor1, hash1);
    ct_tensor_hash(&tensor2, hash2);
    
    /* Different data must produce different hash */
    ASSERT(!ct_hash_equal(hash1, hash2));
}

/* ============================================================================
 * Test: Merkle Chain Init
 * ============================================================================ */

TEST(merkle_init) {
    ct_merkle_ctx_t ctx;
    fixed_t weights[4] = {FIXED_ONE, FIXED_HALF, 0, -FIXED_HALF};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    uint8_t config[] = "test config";
    
    ct_error_t err = ct_merkle_init(&ctx, &tensor, config, sizeof(config), 12345);
    
    ASSERT_EQ(err, CT_OK);
    ASSERT(ctx.initialized);
    ASSERT(!ctx.faulted);
    ASSERT_EQ(ctx.step, 0);
    
    /* Initial hash should be non-zero */
    uint8_t zero[CT_HASH_SIZE] = {0};
    ASSERT(!ct_hash_equal(ctx.current_hash, zero));
    
    /* Initial hash should equal stored initial */
    ASSERT(ct_hash_equal(ctx.current_hash, ctx.initial_hash));
}

TEST(merkle_init_deterministic) {
    ct_merkle_ctx_t ctx1, ctx2;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx1, &tensor, "cfg", 3, 42);
    ct_merkle_init(&ctx2, &tensor, "cfg", 3, 42);
    
    /* Same inputs must produce same initial hash */
    ASSERT(ct_hash_equal(ctx1.current_hash, ctx2.current_hash));
}

TEST(merkle_init_seed_matters) {
    ct_merkle_ctx_t ctx1, ctx2;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx1, &tensor, "cfg", 3, 42);
    ct_merkle_init(&ctx2, &tensor, "cfg", 3, 43);  /* Different seed */
    
    /* Different seed must produce different hash */
    ASSERT(!ct_hash_equal(ctx1.current_hash, ctx2.current_hash));
}

/* ============================================================================
 * Test: Merkle Step
 * ============================================================================ */

TEST(merkle_step_basic) {
    ct_merkle_ctx_t ctx;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx, &tensor, NULL, 0, 1);
    
    uint8_t initial_hash[CT_HASH_SIZE];
    ct_hash_copy(initial_hash, ctx.current_hash);
    
    uint32_t batch[] = {0, 1, 2, 3};
    ct_training_step_t step;
    
    ct_error_t err = ct_merkle_step(&ctx, &tensor, batch, 4, &step, NULL);
    
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(ctx.step, 1);
    
    /* Hash should have changed */
    ASSERT(!ct_hash_equal(ctx.current_hash, initial_hash));
    
    /* Step record should be filled */
    ASSERT(ct_hash_equal(step.prev_hash, initial_hash));
    ASSERT_EQ(step.step, 0);
}

TEST(merkle_step_chain) {
    ct_merkle_ctx_t ctx;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx, &tensor, NULL, 0, 1);
    
    uint32_t batch[] = {0, 1};
    ct_training_step_t steps[3];
    
    /* Run 3 steps */
    for (int i = 0; i < 3; i++) {
        ct_error_t err = ct_merkle_step(&ctx, &tensor, batch, 2, &steps[i], NULL);
        ASSERT_EQ(err, CT_OK);
    }
    
    ASSERT_EQ(ctx.step, 3);
    
    /* Verify chain: step[i].step_hash == step[i+1].prev_hash */
    ASSERT(ct_hash_equal(steps[0].step_hash, steps[1].prev_hash));
    ASSERT(ct_hash_equal(steps[1].step_hash, steps[2].prev_hash));
}

TEST(merkle_step_deterministic) {
    ct_merkle_ctx_t ctx1, ctx2;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx1, &tensor, NULL, 0, 1);
    ct_merkle_init(&ctx2, &tensor, NULL, 0, 1);
    
    uint32_t batch[] = {0, 1, 2};
    
    ct_merkle_step(&ctx1, &tensor, batch, 3, NULL, NULL);
    ct_merkle_step(&ctx2, &tensor, batch, 3, NULL, NULL);
    
    /* Same sequence must produce same hash */
    ASSERT(ct_hash_equal(ctx1.current_hash, ctx2.current_hash));
}

/* ============================================================================
 * Test: Fault Handling
 * ============================================================================ */

TEST(merkle_fault_invalidates) {
    ct_merkle_ctx_t ctx;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx, &tensor, NULL, 0, 1);
    ASSERT(ct_merkle_is_valid(&ctx));
    
    /* Step with fault flag set */
    ct_fault_flags_t faults = {0};
    faults.overflow = 1;
    
    uint32_t batch[] = {0};
    ct_error_t err = ct_merkle_step(&ctx, &tensor, batch, 1, NULL, &faults);
    
    ASSERT_EQ(err, CT_ERR_FAULT);
    ASSERT(!ct_merkle_is_valid(&ctx));
    ASSERT(ctx.faulted);
}

TEST(merkle_faulted_rejects_steps) {
    ct_merkle_ctx_t ctx;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx, &tensor, NULL, 0, 1);
    ct_merkle_invalidate(&ctx);
    
    uint32_t batch[] = {0};
    ct_error_t err = ct_merkle_step(&ctx, &tensor, batch, 1, NULL, NULL);
    
    ASSERT_EQ(err, CT_ERR_FAULT);
}

/* ============================================================================
 * Test: Checkpoints
 * ============================================================================ */

TEST(checkpoint_create) {
    ct_merkle_ctx_t ctx;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx, &tensor, NULL, 0, 1);
    
    uint32_t batch[] = {0, 1};
    ct_merkle_step(&ctx, &tensor, batch, 2, NULL, NULL);
    
    ct_prng_t prng;
    ct_prng_init(&prng, 42, 0);
    
    uint8_t config_hash[CT_HASH_SIZE] = {0};
    ct_checkpoint_t checkpoint;
    
    ct_error_t err = ct_checkpoint_create(&ctx, &prng, 0, &tensor, config_hash, &checkpoint);
    
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(checkpoint.step, 1);
    ASSERT_EQ(checkpoint.epoch, 0);
    ASSERT_EQ(checkpoint.version, CT_CHECKPOINT_VERSION);
    ASSERT(ct_hash_equal(checkpoint.merkle_hash, ctx.current_hash));
}

TEST(checkpoint_verify) {
    ct_merkle_ctx_t ctx;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx, &tensor, NULL, 0, 1);
    
    ct_prng_t prng;
    ct_prng_init(&prng, 42, 0);
    
    uint8_t config_hash[CT_HASH_SIZE] = {0};
    ct_checkpoint_t checkpoint;
    ct_checkpoint_create(&ctx, &prng, 0, &tensor, config_hash, &checkpoint);
    
    /* Verify with same weights */
    ct_error_t err = ct_checkpoint_verify(&checkpoint, &tensor);
    ASSERT_EQ(err, CT_OK);
    
    /* Verify with different weights should fail */
    weights[0] = 999;
    err = ct_checkpoint_verify(&checkpoint, &tensor);
    ASSERT_EQ(err, CT_ERR_HASH);
}

TEST(checkpoint_restore) {
    ct_merkle_ctx_t ctx1, ctx2;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx1, &tensor, NULL, 0, 1);
    
    uint32_t batch[] = {0, 1};
    ct_merkle_step(&ctx1, &tensor, batch, 2, NULL, NULL);
    ct_merkle_step(&ctx1, &tensor, batch, 2, NULL, NULL);
    
    /* Create checkpoint */
    ct_prng_t prng;
    ct_prng_init(&prng, 42, 0);
    uint8_t config_hash[CT_HASH_SIZE] = {0};
    ct_checkpoint_t checkpoint;
    ct_checkpoint_create(&ctx1, &prng, 0, &tensor, config_hash, &checkpoint);
    
    /* Restore to new context */
    ct_error_t err = ct_merkle_restore(&ctx2, &checkpoint);
    ASSERT_EQ(err, CT_OK);
    
    ASSERT(ct_hash_equal(ctx2.current_hash, ctx1.current_hash));
    ASSERT_EQ(ctx2.step, ctx1.step);
    ASSERT(ctx2.initialized);
}

/* ============================================================================
 * Test: Step Verification
 * ============================================================================ */

TEST(verify_step) {
    ct_merkle_ctx_t ctx;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx, &tensor, NULL, 0, 1);
    
    uint8_t prev_hash[CT_HASH_SIZE];
    ct_hash_copy(prev_hash, ctx.current_hash);
    
    uint32_t batch[] = {0, 1, 2};
    ct_training_step_t step;
    ct_merkle_step(&ctx, &tensor, batch, 3, &step, NULL);
    
    /* Verify the step */
    ct_error_t err = ct_merkle_verify_step(&step, prev_hash, &tensor, batch, 3);
    ASSERT_EQ(err, CT_OK);
}

TEST(verify_step_wrong_prev) {
    ct_merkle_ctx_t ctx;
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    ct_merkle_init(&ctx, &tensor, NULL, 0, 1);
    
    uint32_t batch[] = {0, 1};
    ct_training_step_t step;
    ct_merkle_step(&ctx, &tensor, batch, 2, &step, NULL);
    
    /* Wrong prev_hash should fail */
    uint8_t wrong_prev[CT_HASH_SIZE] = {0};
    ct_error_t err = ct_merkle_verify_step(&step, wrong_prev, &tensor, batch, 2);
    ASSERT_EQ(err, CT_ERR_HASH);
}

/* ============================================================================
 * Test: Error Handling
 * ============================================================================ */

TEST(null_pointer_handling) {
    ASSERT_EQ(ct_merkle_init(NULL, NULL, NULL, 0, 0), CT_ERR_NULL);
    ASSERT_EQ(ct_merkle_step(NULL, NULL, NULL, 0, NULL, NULL), CT_ERR_NULL);
    ASSERT_EQ(ct_tensor_hash(NULL, NULL), CT_ERR_NULL);
    ASSERT_EQ(ct_checkpoint_create(NULL, NULL, 0, NULL, NULL, NULL), CT_ERR_NULL);
    ASSERT_EQ(ct_checkpoint_verify(NULL, NULL), CT_ERR_NULL);
    ASSERT_EQ(ct_merkle_restore(NULL, NULL), CT_ERR_NULL);
}

TEST(uninitialized_context) {
    ct_merkle_ctx_t ctx = {0};
    fixed_t weights[4] = {1, 2, 3, 4};
    ct_tensor_t tensor;
    ct_tensor_init_1d(&tensor, weights, 4);
    
    uint32_t batch[] = {0};
    ct_error_t err = ct_merkle_step(&ctx, &tensor, batch, 1, NULL, NULL);
    ASSERT_EQ(err, CT_ERR_STATE);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== SRS-008: Merkle Chain Tests ===\n\n");
    
    printf("SHA256 Tests:\n");
    RUN_TEST(sha256_empty);
    RUN_TEST(sha256_abc);
    RUN_TEST(sha256_incremental);
    
    printf("\nHash Utility Tests:\n");
    RUN_TEST(hash_equal);
    RUN_TEST(hash_copy);
    RUN_TEST(hash_zero);
    
    printf("\nTensor Serialization Tests:\n");
    RUN_TEST(tensor_is_contiguous);
    RUN_TEST(tensor_serial_size);
    RUN_TEST(tensor_serialize);
    RUN_TEST(tensor_hash_deterministic);
    RUN_TEST(tensor_hash_different);
    
    printf("\nMerkle Init Tests:\n");
    RUN_TEST(merkle_init);
    RUN_TEST(merkle_init_deterministic);
    RUN_TEST(merkle_init_seed_matters);
    
    printf("\nMerkle Step Tests:\n");
    RUN_TEST(merkle_step_basic);
    RUN_TEST(merkle_step_chain);
    RUN_TEST(merkle_step_deterministic);
    
    printf("\nFault Handling Tests:\n");
    RUN_TEST(merkle_fault_invalidates);
    RUN_TEST(merkle_faulted_rejects_steps);
    
    printf("\nCheckpoint Tests:\n");
    RUN_TEST(checkpoint_create);
    RUN_TEST(checkpoint_verify);
    RUN_TEST(checkpoint_restore);
    
    printf("\nVerification Tests:\n");
    RUN_TEST(verify_step);
    RUN_TEST(verify_step_wrong_prev);
    
    printf("\nError Handling Tests:\n");
    RUN_TEST(null_pointer_handling);
    RUN_TEST(uninitialized_context);
    
    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    
    return tests_failed > 0 ? 1 : 0;
}
