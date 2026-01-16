/**
 * @file permutation.h
 * @project Certifiable Training
 * @brief Deterministic data permutation using Cycle-Walking Feistel.
 *
 * @details Implements bijective permutation for dataset shuffling:
 *          - Guarantees true bijection on [0, N-1] for any N
 *          - Deterministic: same (seed, epoch, index) → same output
 *          - Bounded iterations for safety-critical compliance
 *          - Batch construction for training
 *
 * @traceability SRS-009-PERMUTATION, CT-MATH-001 §5
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304, DO-178C
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust. All rights reserved.
 * @license Licensed under the GPL-3.0 (Open Source) or Commercial License.
 *          For commercial licensing: william@fstopify.com
 */

#ifndef CERTIFIABLE_TRAINING_PERMUTATION_H
#define CERTIFIABLE_TRAINING_PERMUTATION_H

#include "ct_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

/** Number of Feistel rounds per CT-MATH-001 §5.3 */
#define CT_PERM_FEISTEL_ROUNDS      4

/** Maximum dataset size (2^30) */
#define CT_PERM_MAX_DATASET_SIZE    ((uint32_t)1 << 30)

/* ============================================================================
 * Permutation Context
 * ============================================================================ */

/**
 * @brief Cycle-Walking Feistel permutation state
 * @ref CT-STRUCT-001 §8
 */
typedef struct {
    uint64_t seed;              /**< Master seed for permutation */
    uint32_t epoch;             /**< Current epoch (changes permutation) */
    uint32_t dataset_size;      /**< N: number of samples */
    uint32_t half_bits;         /**< Cached: floor(ceil(log2(N))/2) */
    uint32_t half_mask;         /**< Cached: 2^half_bits - 1 */
    uint32_t range;             /**< Cached: 2^k where k = ceil(log2(N)) */
    bool initialized;           /**< Initialization flag */
} ct_permutation_t;

/**
 * @brief Batch generation context
 */
typedef struct {
    ct_permutation_t perm;      /**< Permutation state */
    uint32_t batch_size;        /**< B: samples per batch */
    uint32_t steps_per_epoch;   /**< ceil(N/B) */
} ct_batch_ctx_t;

/* ============================================================================
 * Permutation Operations
 * ============================================================================ */

/**
 * @brief Initialize permutation context
 * @param perm Permutation context
 * @param seed Master seed
 * @param epoch Current epoch
 * @param dataset_size N: number of samples
 * @return CT_OK on success
 *
 * @note Dataset size must be > 0 and <= CT_PERM_MAX_DATASET_SIZE
 */
ct_error_t ct_permutation_init(ct_permutation_t *perm,
                               uint64_t seed,
                               uint32_t epoch,
                               uint32_t dataset_size);

/**
 * @brief Set epoch (changes permutation)
 * @param perm Permutation context
 * @param epoch New epoch
 */
void ct_permutation_set_epoch(ct_permutation_t *perm, uint32_t epoch);

/**
 * @brief Compute permuted index: π(index) → [0, N-1]
 * @param perm Permutation context
 * @param index Original index [0, N-1]
 * @param faults Fault accumulator
 * @return Permuted index [0, N-1]
 *
 * @details Uses Cycle-Walking Feistel construction:
 *          1. Apply 4-round Feistel network
 *          2. If result >= N, repeat (cycle-walk)
 *          3. Guaranteed to terminate with expected iterations < 2
 *
 * @ref CT-MATH-001 §5.3
 */
uint32_t ct_permutation_apply(const ct_permutation_t *perm,
                              uint32_t index,
                              ct_fault_flags_t *faults);

/**
 * @brief Compute inverse permutation: π⁻¹(permuted_index) → original
 * @param perm Permutation context
 * @param permuted_index Permuted index [0, N-1]
 * @param faults Fault accumulator
 * @return Original index [0, N-1]
 *
 * @note Feistel networks are self-inverting with reversed round order
 */
uint32_t ct_permutation_inverse(const ct_permutation_t *perm,
                                uint32_t permuted_index,
                                ct_fault_flags_t *faults);

/* ============================================================================
 * Batch Operations
 * ============================================================================ */

/**
 * @brief Initialize batch generation context
 * @param ctx Batch context
 * @param seed Master seed
 * @param epoch Current epoch
 * @param dataset_size N: number of samples
 * @param batch_size B: samples per batch
 * @return CT_OK on success
 */
ct_error_t ct_batch_init(ct_batch_ctx_t *ctx,
                         uint64_t seed,
                         uint32_t epoch,
                         uint32_t dataset_size,
                         uint32_t batch_size);

/**
 * @brief Set epoch for batch context
 */
void ct_batch_set_epoch(ct_batch_ctx_t *ctx, uint32_t epoch);

/**
 * @brief Get batch indices for a training step
 * @param ctx Batch context
 * @param step Training step t
 * @param indices_out Output array [batch_size]
 * @param faults Fault accumulator
 * @return CT_OK on success
 *
 * @details Computes: indices[j] = π(t*B + j, seed, epoch, N) for j in [0, B-1]
 *          Per CT-MATH-001 §5.6 canonical formula.
 *
 * @note Last batch of epoch may have fewer than batch_size valid indices
 *       if N is not divisible by B.
 */
ct_error_t ct_batch_get_indices(const ct_batch_ctx_t *ctx,
                                uint64_t step,
                                uint32_t *indices_out,
                                ct_fault_flags_t *faults);

/**
 * @brief Get actual batch size for a step (handles partial last batch)
 * @param ctx Batch context
 * @param step Training step
 * @return Number of valid samples in this batch
 */
uint32_t ct_batch_get_size(const ct_batch_ctx_t *ctx, uint64_t step);

/**
 * @brief Get step within current epoch
 * @param ctx Batch context
 * @param global_step Global training step
 * @return Step within epoch [0, steps_per_epoch-1]
 */
uint32_t ct_batch_step_in_epoch(const ct_batch_ctx_t *ctx, uint64_t global_step);

/**
 * @brief Get epoch from global step
 * @param ctx Batch context
 * @param global_step Global training step
 * @return Epoch number
 */
uint32_t ct_batch_get_epoch(const ct_batch_ctx_t *ctx, uint64_t global_step);

/* ============================================================================
 * Feistel Hash Function
 * ============================================================================ */

/**
 * @brief Hash function for Feistel network
 * @param seed Master seed
 * @param epoch Current epoch
 * @param round Feistel round [0-3]
 * @param value Input value
 * @return Hashed value
 *
 * @ref CT-MATH-001 §5.4
 */
uint32_t ct_feistel_hash(uint64_t seed,
                         uint32_t epoch,
                         uint32_t round,
                         uint32_t value);

/* ============================================================================
 * Verification Utilities
 * ============================================================================ */

/**
 * @brief Verify permutation is bijective (for testing)
 * @param perm Permutation context
 * @param faults Fault accumulator
 * @return true if verified bijective
 *
 * @warning O(N) time and memory - only for small N in testing
 */
bool ct_permutation_verify_bijection(const ct_permutation_t *perm,
                                     ct_fault_flags_t *faults);

#ifdef __cplusplus
}
#endif

#endif /* CERTIFIABLE_TRAINING_PERMUTATION_H */
