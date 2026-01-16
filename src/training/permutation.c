/**
 * @file permutation.c
 * @project Certifiable Training
 * @brief Cycle-Walking Feistel permutation implementation.
 *
 * @traceability SRS-009-PERMUTATION, CT-MATH-001 §5
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304, DO-178C
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust. All rights reserved.
 * @license Licensed under the GPL-3.0 (Open Source) or Commercial License.
 *          For commercial licensing: william@fstopify.com
 */

#include "permutation.h"
#include <string.h>

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

/**
 * @brief Compute ceil(log2(n)), minimum 1
 */
static uint32_t ceil_log2(uint32_t n) {
    if (n <= 1) return 1;
    
    uint32_t bits = 0;
    uint32_t val = n - 1;
    while (val > 0) {
        val >>= 1;
        bits++;
    }
    return bits;
}

/**
 * @brief Compute cached parameters for permutation
 * 
 * @note We round k up to even for balanced Feistel (equal L and R halves).
 *       This may expand the range slightly but guarantees bijection.
 */
static void compute_params(ct_permutation_t *perm) {
    uint32_t k = ceil_log2(perm->dataset_size);
    
    /* Round up to even number of bits for balanced Feistel */
    if (k % 2 == 1) {
        k++;
    }
    
    perm->range = (uint32_t)1 << k;
    perm->half_bits = k / 2;
    perm->half_mask = ((uint32_t)1 << perm->half_bits) - 1;
}

/* ============================================================================
 * Feistel Hash Function
 * ============================================================================ */

uint32_t ct_feistel_hash(uint64_t seed,
                         uint32_t epoch,
                         uint32_t round,
                         uint32_t value) {
    /*
     * Per CT-MATH-001 §5.4:
     * h = seed
     * h = (h × 0x9E3779B9 + epoch) & 0xFFFFFFFF
     * h = (h × 0x85EBCA6B + round) & 0xFFFFFFFF
     * h = (h × 0xC2B2AE35 + value) & 0xFFFFFFFF
     * h = h ⊕ (h >> 16)
     * h = (h × 0x85EBCA6B) & 0xFFFFFFFF
     * h = h ⊕ (h >> 13)
     */
    uint32_t h = (uint32_t)(seed & 0xFFFFFFFF);
    
    h = (uint32_t)((uint64_t)h * 0x9E3779B9 + epoch);
    h = (uint32_t)((uint64_t)h * 0x85EBCA6B + round);
    h = (uint32_t)((uint64_t)h * 0xC2B2AE35 + value);
    h ^= (h >> 16);
    h = (uint32_t)((uint64_t)h * 0x85EBCA6B);
    h ^= (h >> 13);
    
    return h;
}

/* ============================================================================
 * Permutation Operations
 * ============================================================================ */

ct_error_t ct_permutation_init(ct_permutation_t *perm,
                               uint64_t seed,
                               uint32_t epoch,
                               uint32_t dataset_size) {
    if (!perm) {
        return CT_ERR_NULL;
    }
    
    if (dataset_size == 0 || dataset_size > CT_PERM_MAX_DATASET_SIZE) {
        return CT_ERR_DIMENSION;
    }
    
    perm->seed = seed;
    perm->epoch = epoch;
    perm->dataset_size = dataset_size;
    
    compute_params(perm);
    
    perm->initialized = true;
    
    return CT_OK;
}

void ct_permutation_set_epoch(ct_permutation_t *perm, uint32_t epoch) {
    if (perm && perm->initialized) {
        perm->epoch = epoch;
    }
}

/**
 * @brief Core Feistel network (forward direction)
 * 
 * @note Uses balanced Feistel with equal-sized L and R halves.
 */
static uint32_t feistel_forward(const ct_permutation_t *perm, uint32_t input) {
    uint32_t L = input & perm->half_mask;
    uint32_t R = (input >> perm->half_bits) & perm->half_mask;
    
    /* 4-round Feistel network */
    for (uint32_t r = 0; r < CT_PERM_FEISTEL_ROUNDS; r++) {
        uint32_t F = ct_feistel_hash(perm->seed, perm->epoch, r, R);
        uint32_t temp = R;
        R = L ^ (F & perm->half_mask);
        L = temp;
    }
    
    return (R << perm->half_bits) | L;
}

/**
 * @brief Core Feistel network (inverse direction)
 */
static uint32_t feistel_inverse(const ct_permutation_t *perm, uint32_t input) {
    uint32_t L = input & perm->half_mask;
    uint32_t R = (input >> perm->half_bits) & perm->half_mask;
    
    /* 4-round Feistel network in reverse */
    for (int r = CT_PERM_FEISTEL_ROUNDS - 1; r >= 0; r--) {
        uint32_t F = ct_feistel_hash(perm->seed, perm->epoch, (uint32_t)r, L);
        uint32_t temp = L;
        L = R ^ (F & perm->half_mask);
        R = temp;
    }
    
    return (R << perm->half_bits) | L;
}

uint32_t ct_permutation_apply(const ct_permutation_t *perm,
                              uint32_t index,
                              ct_fault_flags_t *faults) {
    if (!perm || !perm->initialized) {
        if (faults) faults->domain = 1;
        return 0;
    }
    
    if (index >= perm->dataset_size) {
        if (faults) faults->domain = 1;
        return index % perm->dataset_size;
    }
    
    /* Special case: N = 1 */
    if (perm->dataset_size == 1) {
        return 0;
    }
    
    uint32_t i = index;
    uint32_t max_iterations = perm->range;  /* Safety bound */
    uint32_t iterations = 0;
    
    /* Cycle-walk until result is in valid range [0, N-1] */
    do {
        if (iterations >= max_iterations) {
            /* Safety bound hit - should never happen */
            if (faults) faults->domain = 1;
            return index % perm->dataset_size;  /* Fallback */
        }
        iterations++;
        
        i = feistel_forward(perm, i);
        
    } while (i >= perm->dataset_size);
    
    return i;
}

uint32_t ct_permutation_inverse(const ct_permutation_t *perm,
                                uint32_t permuted_index,
                                ct_fault_flags_t *faults) {
    if (!perm || !perm->initialized) {
        if (faults) faults->domain = 1;
        return 0;
    }
    
    if (permuted_index >= perm->dataset_size) {
        if (faults) faults->domain = 1;
        return permuted_index % perm->dataset_size;
    }
    
    /* Special case: N = 1 */
    if (perm->dataset_size == 1) {
        return 0;
    }
    
    uint32_t i = permuted_index;
    uint32_t max_iterations = perm->range;
    uint32_t iterations = 0;
    
    /* Cycle-walk inverse */
    do {
        if (iterations >= max_iterations) {
            if (faults) faults->domain = 1;
            return permuted_index % perm->dataset_size;
        }
        iterations++;
        
        i = feistel_inverse(perm, i);
        
    } while (i >= perm->dataset_size);
    
    return i;
}

/* ============================================================================
 * Batch Operations
 * ============================================================================ */

ct_error_t ct_batch_init(ct_batch_ctx_t *ctx,
                         uint64_t seed,
                         uint32_t epoch,
                         uint32_t dataset_size,
                         uint32_t batch_size) {
    if (!ctx) {
        return CT_ERR_NULL;
    }
    
    if (batch_size == 0) {
        return CT_ERR_DIMENSION;
    }
    
    ct_error_t err = ct_permutation_init(&ctx->perm, seed, epoch, dataset_size);
    if (err != CT_OK) {
        return err;
    }
    
    ctx->batch_size = batch_size;
    
    /* steps_per_epoch = ceil(N / B) */
    ctx->steps_per_epoch = (dataset_size + batch_size - 1) / batch_size;
    
    return CT_OK;
}

void ct_batch_set_epoch(ct_batch_ctx_t *ctx, uint32_t epoch) {
    if (ctx) {
        ct_permutation_set_epoch(&ctx->perm, epoch);
    }
}

ct_error_t ct_batch_get_indices(const ct_batch_ctx_t *ctx,
                                uint64_t step,
                                uint32_t *indices_out,
                                ct_fault_flags_t *faults) {
    if (!ctx || !indices_out) {
        return CT_ERR_NULL;
    }
    
    if (!ctx->perm.initialized) {
        return CT_ERR_STATE;
    }
    
    uint32_t N = ctx->perm.dataset_size;
    uint32_t B = ctx->batch_size;
    
    /* Compute step within epoch */
    uint32_t step_in_epoch = (uint32_t)(step % ctx->steps_per_epoch);
    
    /* B_t = { d_{π(t*B + j)} : j ∈ [0, B-1] } */
    uint64_t base_index = (uint64_t)step_in_epoch * B;
    
    for (uint32_t j = 0; j < B; j++) {
        uint64_t linear_idx = base_index + j;
        
        if (linear_idx >= N) {
            /* Partial last batch - wrap or mark invalid */
            indices_out[j] = ct_permutation_apply(&ctx->perm, 
                                                   (uint32_t)(linear_idx % N), 
                                                   faults);
        } else {
            indices_out[j] = ct_permutation_apply(&ctx->perm, 
                                                   (uint32_t)linear_idx, 
                                                   faults);
        }
    }
    
    return CT_OK;
}

uint32_t ct_batch_get_size(const ct_batch_ctx_t *ctx, uint64_t step) {
    if (!ctx || !ctx->perm.initialized) {
        return 0;
    }
    
    uint32_t step_in_epoch = (uint32_t)(step % ctx->steps_per_epoch);
    uint32_t N = ctx->perm.dataset_size;
    uint32_t B = ctx->batch_size;
    
    /* Check if this is the last step in epoch */
    if (step_in_epoch == ctx->steps_per_epoch - 1) {
        /* Last batch may be partial */
        uint32_t remaining = N - (step_in_epoch * B);
        return (remaining < B) ? remaining : B;
    }
    
    return B;
}

uint32_t ct_batch_step_in_epoch(const ct_batch_ctx_t *ctx, uint64_t global_step) {
    if (!ctx || ctx->steps_per_epoch == 0) {
        return 0;
    }
    return (uint32_t)(global_step % ctx->steps_per_epoch);
}

uint32_t ct_batch_get_epoch(const ct_batch_ctx_t *ctx, uint64_t global_step) {
    if (!ctx || ctx->steps_per_epoch == 0) {
        return 0;
    }
    return (uint32_t)(global_step / ctx->steps_per_epoch);
}

/* ============================================================================
 * Verification Utilities
 * ============================================================================ */

bool ct_permutation_verify_bijection(const ct_permutation_t *perm,
                                     ct_fault_flags_t *faults) {
    if (!perm || !perm->initialized) {
        return false;
    }
    
    uint32_t N = perm->dataset_size;
    
    /* Allocate visited array (only for small N in testing) */
    if (N > 100000) {
        /* Too large for simple verification */
        return false;
    }
    
    /* Static buffer for small tests */
    static uint8_t visited[100000];
    memset(visited, 0, N);
    
    /* Check each index maps to unique output */
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = ct_permutation_apply(perm, i, faults);
        
        if (j >= N) {
            return false;  /* Out of range */
        }
        
        if (visited[j]) {
            return false;  /* Collision - not bijective */
        }
        
        visited[j] = 1;
    }
    
    /* Verify all outputs were hit */
    for (uint32_t j = 0; j < N; j++) {
        if (!visited[j]) {
            return false;  /* Missing output */
        }
    }
    
    return true;
}
