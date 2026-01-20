/**
 * @file verify_step.c
 * @project Certifiable Training
 * @brief Single step verification demo - proves Merkle chain integrity
 *
 * Demonstrates the auditability property of certifiable training:
 * - Initialize a Merkle chain
 * - Commit training steps
 * - Verify steps via replay
 * - Detect tampering
 *
 * NO FLOATING POINT - All computation and display uses integer arithmetic.
 *
 * This is the core primitive for post-hoc audit: given the previous hash,
 * weights, batch indices, and step metadata, anyone can verify that the
 * committed hash was computed correctly.
 *
 * @traceability CT-MATH-001 S16, SRS-008-MERKLE
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 * @license GPL-3.0
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "ct_types.h"
#include "merkle.h"
#include "prng.h"

/*===========================================================================
 * Configuration
 *===========================================================================*/

#define NUM_WEIGHTS     16
#define BATCH_SIZE      4
#define SEED            0x123456789ABCDEF0ULL

/*===========================================================================
 * Test Data (signed decimal for -Wsign-conversion compliance)
 *===========================================================================*/

static fixed_t weights_data[NUM_WEIGHTS] = {
    65536,       /*  1.0   (0x00010000) */
    -65536,      /* -1.0   (0xFFFF0000) */
    32768,       /*  0.5   (0x00008000) */
    -32768,      /* -0.5   (0xFFFF8000) */
    131072,      /*  2.0   (0x00020000) */
    -131072,     /* -2.0   (0xFFFE0000) */
    16384,       /*  0.25  (0x00004000) */
    -16384,      /* -0.25  (0xFFFFC000) */
    1,           /*  epsilon */
    -1,          /* -epsilon */
    0,           /*  0 */
    FIXED_MAX,   /*  max (0x7FFFFFFF) */
    FIXED_MIN,   /*  min (0x80000000) */
    196608,      /*  3.0   (0x00030000) */
    262144,      /*  4.0   (0x00040000) */
    327680,      /*  5.0   (0x00050000) */
};

static uint32_t batch_indices[BATCH_SIZE] = { 42, 17, 99, 3 };

static const uint8_t config_data[] = "verify_step_demo_v1";

/*===========================================================================
 * Helpers (Integer Only)
 *===========================================================================*/

static void print_hash(const uint8_t *hash, int n) {
    for (int i = 0; i < n; i++) {
        printf("%02x", hash[i]);
    }
}

static ct_tensor_t make_weights_tensor(void) {
    ct_tensor_t t;
    t.data = weights_data;
    t.dims[0] = NUM_WEIGHTS;
    t.strides[0] = 1;
    t.ndims = 1;
    t.total_size = NUM_WEIGHTS;
    return t;
}

/*===========================================================================
 * Demo 1: Genesis Verification
 *===========================================================================*/

static bool demo_genesis_verification(void) {
    printf("\n+-------------------------------------------------------------+\n");
    printf("|  Demo 1: Genesis Hash Verification                         |\n");
    printf("+-------------------------------------------------------------+\n\n");

    ct_tensor_t weights = make_weights_tensor();

    /* Initialize first chain */
    ct_merkle_ctx_t merkle1;
    ct_error_t err = ct_merkle_init(&merkle1, &weights, config_data,
                                     sizeof(config_data), SEED);

    if (err != CT_OK) {
        printf("ERROR: Failed to init merkle1 (%d)\n", err);
        return false;
    }

    printf("Genesis Parameters:\n");
    printf("  Seed: 0x%016llX\n", (unsigned long long)SEED);
    printf("  Weights: %d values\n", NUM_WEIGHTS);
    printf("  Config: \"%s\"\n", config_data);
    printf("  Genesis hash (h_0): ");
    print_hash(merkle1.current_hash, 16);
    printf("...\n\n");

    /* Verify by re-initializing with same parameters */
    ct_merkle_ctx_t merkle2;
    err = ct_merkle_init(&merkle2, &weights, config_data,
                         sizeof(config_data), SEED);

    if (err != CT_OK) {
        printf("ERROR: Failed to init merkle2 (%d)\n", err);
        return false;
    }

    bool match = ct_hash_equal(merkle1.current_hash, merkle2.current_hash);

    printf("Verification (re-init with same params):\n");
    printf("  Re-computed hash:   ");
    print_hash(merkle2.current_hash, 16);
    printf("...\n");
    printf("  Match: %s\n", match ? "[OK] YES" : "[FAIL] NO");

    return match;
}

/*===========================================================================
 * Demo 2: Single Step Verification
 *===========================================================================*/

static bool demo_step_verification(void) {
    printf("\n+-------------------------------------------------------------+\n");
    printf("|  Demo 2: Training Step Verification                        |\n");
    printf("+-------------------------------------------------------------+\n\n");

    ct_tensor_t weights = make_weights_tensor();
    ct_fault_flags_t faults = {0};

    /* Initialize chain */
    ct_merkle_ctx_t merkle;
    ct_error_t err = ct_merkle_init(&merkle, &weights, config_data,
                                     sizeof(config_data), SEED);
    if (err != CT_OK) return false;

    /* Save genesis */
    uint8_t h0[CT_HASH_SIZE];
    ct_hash_copy(h0, merkle.current_hash);

    printf("Step 0 Parameters:\n");
    printf("  Previous hash (h_0): ");
    print_hash(h0, 16);
    printf("...\n");
    printf("  Batch indices: [%u, %u, %u, %u]\n",
           batch_indices[0], batch_indices[1], batch_indices[2], batch_indices[3]);

    /* Commit step */
    ct_training_step_t step_record;
    err = ct_merkle_step(&merkle, &weights, batch_indices, BATCH_SIZE,
                         &step_record, &faults);

    if (err != CT_OK) {
        printf("ERROR: Merkle step failed (%d)\n", err);
        return false;
    }

    printf("\nResult:\n");
    printf("  New hash (h_1): ");
    print_hash(merkle.current_hash, 16);
    printf("...\n\n");

    /* Verify by replaying from genesis */
    ct_merkle_ctx_t replay;
    err = ct_merkle_init(&replay, &weights, config_data,
                         sizeof(config_data), SEED);
    if (err != CT_OK) return false;

    ct_training_step_t replay_record;
    err = ct_merkle_step(&replay, &weights, batch_indices, BATCH_SIZE,
                         &replay_record, &faults);
    if (err != CT_OK) return false;

    bool match = ct_hash_equal(merkle.current_hash, replay.current_hash);

    printf("Verification (replay from genesis):\n");
    printf("  Replayed hash:      ");
    print_hash(replay.current_hash, 16);
    printf("...\n");
    printf("  Match: %s\n", match ? "[OK] YES" : "[FAIL] NO");

    return match && !ct_has_fault(&faults);
}

/*===========================================================================
 * Demo 3: Chain Extension Verification
 *===========================================================================*/

static bool demo_chain_verification(void) {
    printf("\n+-------------------------------------------------------------+\n");
    printf("|  Demo 3: Multi-Step Chain Verification                     |\n");
    printf("+-------------------------------------------------------------+\n\n");

    ct_tensor_t weights = make_weights_tensor();
    ct_fault_flags_t faults = {0};

    /* Build a chain of 5 steps */
    ct_merkle_ctx_t merkle;
    ct_error_t err = ct_merkle_init(&merkle, &weights, config_data,
                                     sizeof(config_data), SEED);
    if (err != CT_OK) return false;

    printf("Building chain (5 steps):\n");
    printf("  h_0: ");
    print_hash(merkle.current_hash, 8);
    printf("...\n");

    /* Store intermediate hashes */
    uint8_t chain_hashes[6][CT_HASH_SIZE];
    ct_hash_copy(chain_hashes[0], merkle.current_hash);

    /* Store weight snapshots (simplified: just modify first weight) */
    fixed_t weight_snapshots[6];
    weight_snapshots[0] = weights_data[0];

    for (int step = 0; step < 5; step++) {
        /* Simulate weight update: subtract 4096 (0.0625 in Q16.16) */
        weights_data[0] = weights_data[0] - 4096;
        weight_snapshots[step + 1] = weights_data[0];

        ct_training_step_t record;
        err = ct_merkle_step(&merkle, &weights, batch_indices, BATCH_SIZE,
                             &record, &faults);
        if (err != CT_OK) return false;

        ct_hash_copy(chain_hashes[step + 1], merkle.current_hash);

        printf("  h_%d: ", step + 1);
        print_hash(merkle.current_hash, 8);
        printf("...\n");
    }

    printf("\nVerifying chain integrity...\n");

    /* Reset weights and replay */
    weights_data[0] = weight_snapshots[0];

    ct_merkle_ctx_t replay;
    err = ct_merkle_init(&replay, &weights, config_data,
                         sizeof(config_data), SEED);
    if (err != CT_OK) return false;

    bool all_match = true;

    /* Verify genesis */
    if (!ct_hash_equal(replay.current_hash, chain_hashes[0])) {
        printf("  [FAIL] Genesis mismatch!\n");
        all_match = false;
    } else {
        printf("  [OK] h_0 verified\n");
    }

    /* Replay each step */
    for (int step = 0; step < 5; step++) {
        weights_data[0] = weight_snapshots[step + 1];

        ct_training_step_t record;
        err = ct_merkle_step(&replay, &weights, batch_indices, BATCH_SIZE,
                             &record, &faults);
        if (err != CT_OK) {
            printf("  [FAIL] h_%d replay failed!\n", step + 1);
            all_match = false;
            continue;
        }

        if (!ct_hash_equal(replay.current_hash, chain_hashes[step + 1])) {
            printf("  [FAIL] h_%d mismatch!\n", step + 1);
            all_match = false;
        } else {
            printf("  [OK] h_%d verified\n", step + 1);
        }
    }

    return all_match && !ct_has_fault(&faults);
}

/*===========================================================================
 * Demo 4: Tamper Detection
 *===========================================================================*/

static bool demo_tamper_detection(void) {
    printf("\n+-------------------------------------------------------------+\n");
    printf("|  Demo 4: Tamper Detection                                  |\n");
    printf("+-------------------------------------------------------------+\n\n");

    ct_fault_flags_t faults = {0};

    /* Reset weights to known state: 1.0 in Q16.16 */
    weights_data[0] = 65536;
    ct_tensor_t weights = make_weights_tensor();

    /* Build legitimate chain */
    ct_merkle_ctx_t merkle;
    ct_error_t err = ct_merkle_init(&merkle, &weights, config_data,
                                     sizeof(config_data), SEED);
    if (err != CT_OK) return false;

    ct_training_step_t record;
    err = ct_merkle_step(&merkle, &weights, batch_indices, BATCH_SIZE,
                         &record, &faults);
    if (err != CT_OK) return false;

    uint8_t legitimate_hash[CT_HASH_SIZE];
    ct_hash_copy(legitimate_hash, merkle.current_hash);

    printf("Legitimate step hash: ");
    print_hash(legitimate_hash, 16);
    printf("...\n\n");

    /* Test 1: Tampered weight */
    printf("Test 1: Tampered weight (changed by 1 LSB)...\n");

    weights_data[0] = 65537;  /* Changed by 1 LSB */

    ct_merkle_ctx_t tampered1;
    err = ct_merkle_init(&tampered1, &weights, config_data,
                         sizeof(config_data), SEED);
    if (err != CT_OK) return false;

    ct_training_step_t tampered_record1;
    err = ct_merkle_step(&tampered1, &weights, batch_indices, BATCH_SIZE,
                         &tampered_record1, &faults);
    if (err != CT_OK) return false;

    printf("  Tampered hash: ");
    print_hash(tampered1.current_hash, 16);
    printf("...\n");

    bool detected1 = !ct_hash_equal(legitimate_hash, tampered1.current_hash);
    printf("  Tamper detected: %s\n\n", detected1 ? "[OK] YES" : "[FAIL] NO - SECURITY FAILURE");

    /* Test 2: Tampered batch index */
    printf("Test 2: Tampered batch index...\n");

    weights_data[0] = 65536;  /* Restore */
    uint32_t tampered_indices[BATCH_SIZE] = { 42, 17, 99, 4 };  /* Changed last index */

    ct_merkle_ctx_t tampered2;
    err = ct_merkle_init(&tampered2, &weights, config_data,
                         sizeof(config_data), SEED);
    if (err != CT_OK) return false;

    ct_training_step_t tampered_record2;
    err = ct_merkle_step(&tampered2, &weights, tampered_indices, BATCH_SIZE,
                         &tampered_record2, &faults);
    if (err != CT_OK) return false;

    printf("  Tampered hash: ");
    print_hash(tampered2.current_hash, 16);
    printf("...\n");

    bool detected2 = !ct_hash_equal(legitimate_hash, tampered2.current_hash);
    printf("  Tamper detected: %s\n\n", detected2 ? "[OK] YES" : "[FAIL] NO - SECURITY FAILURE");

    /* Test 3: Tampered seed */
    printf("Test 3: Tampered seed...\n");

    ct_merkle_ctx_t tampered3;
    err = ct_merkle_init(&tampered3, &weights, config_data,
                         sizeof(config_data), SEED + 1);  /* Different seed */
    if (err != CT_OK) return false;

    ct_training_step_t tampered_record3;
    err = ct_merkle_step(&tampered3, &weights, batch_indices, BATCH_SIZE,
                         &tampered_record3, &faults);
    if (err != CT_OK) return false;

    printf("  Tampered hash: ");
    print_hash(tampered3.current_hash, 16);
    printf("...\n");

    bool detected3 = !ct_hash_equal(legitimate_hash, tampered3.current_hash);
    printf("  Tamper detected: %s\n", detected3 ? "[OK] YES" : "[FAIL] NO - SECURITY FAILURE");

    return detected1 && detected2 && detected3;
}

/*===========================================================================
 * Main
 *===========================================================================*/

int main(void) {
    printf("===============================================================\n");
    printf("  Certifiable Training - Step Verification Demo\n");
    printf("===============================================================\n");
    printf("\nThis demo proves the auditability property of Merkle training\n");
    printf("chains: any step can be independently verified by replaying\n");
    printf("the computation and comparing hashes.\n");

    bool pass1 = demo_genesis_verification();
    bool pass2 = demo_step_verification();
    bool pass3 = demo_chain_verification();
    bool pass4 = demo_tamper_detection();

    printf("\n===============================================================\n");
    printf("  Summary\n");
    printf("===============================================================\n\n");

    printf("  Demo 1 (Genesis):        %s\n", pass1 ? "[PASS]" : "[FAIL]");
    printf("  Demo 2 (Single Step):    %s\n", pass2 ? "[PASS]" : "[FAIL]");
    printf("  Demo 3 (Chain):          %s\n", pass3 ? "[PASS]" : "[FAIL]");
    printf("  Demo 4 (Tamper Detect):  %s\n", pass4 ? "[PASS]" : "[FAIL]");

    bool all_pass = pass1 && pass2 && pass3 && pass4;

    printf("\n===============================================================\n");
    if (all_pass) {
        printf("  [PASS] All verification demos passed\n");
        printf("  [PASS] Merkle chain provides cryptographic auditability\n");
        printf("===============================================================\n");
        return 0;
    } else {
        printf("  [FAIL] Some demos failed\n");
        printf("===============================================================\n");
        return 1;
    }
}
