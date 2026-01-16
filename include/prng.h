/**
 * @file prng.h
 * @project Certifiable Training
 * @brief Counter-based PRNG for deterministic randomness
 *
 * @details Provides deterministic pseudo-random number generation as a
 *          pure function of (seed, op_id, step). Used for:
 *          - Deterministic stochastic rounding
 *          - Data shuffling (via Feistel permutation)
 *          - Dropout (if implemented)
 *
 * @traceability CT-MATH-001 §6, CT-STRUCT-001 §4.1
 * @compliance MISRA-C:2012, DO-178C, IEC 62304, ISO 26262
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 */

#ifndef CT_PRNG_H
#define CT_PRNG_H

#include "ct_types.h"

/**
 * @brief PRNG state structure
 *
 * @details Holds the parameters needed to generate deterministic
 *          pseudo-random sequences. The actual PRNG is a pure function;
 *          this structure just tracks the current position.
 *
 * @invariant seed is immutable after init
 * @invariant op_id is immutable after init
 * @invariant step increases monotonically via ct_prng_next()
 *
 * @ref CT-STRUCT-001 §4.1
 */
typedef struct {
    uint64_t seed;      /**< Master seed (immutable after init) */
    uint64_t op_id;     /**< Operation identifier (64-bit minimum) */
    uint64_t step;      /**< Current step counter */
} ct_prng_t;

/**
 * @brief Initialize PRNG state
 *
 * @param prng  Pointer to PRNG state structure
 * @param seed  Master seed (determines entire sequence)
 * @param op_id Operation identifier (unique per operation context)
 *
 * Complexity: O(1)
 * Determinism: N/A (initialization)
 */
void ct_prng_init(ct_prng_t *prng, uint64_t seed, uint64_t op_id);

/**
 * @brief Generate next pseudo-random value and advance state
 *
 * @param prng Pointer to PRNG state
 * @return 32-bit pseudo-random value uniformly distributed in [0, 2^32-1]
 *
 * Complexity: O(1)
 * Determinism: Bit-perfect - same (seed, op_id, step) always produces same output
 *
 * @ref CT-MATH-001 §6.2
 */
uint32_t ct_prng_next(ct_prng_t *prng);

/**
 * @brief Generate value at specific step without modifying state
 *
 * @param prng Pointer to PRNG state (const - not modified)
 * @param step Step to query
 * @return 32-bit pseudo-random value at that step
 *
 * @note Allows random access to any point in the sequence
 *       without sequential generation.
 *
 * Complexity: O(1)
 * Determinism: Bit-perfect
 */
uint32_t ct_prng_peek(const ct_prng_t *prng, uint64_t step);

/**
 * @brief Deterministic stochastic rounding
 *
 * @param x      Value to round (64-bit intermediate)
 * @param shift  Number of fractional bits to remove (0-62)
 * @param prng   PRNG state (will be advanced by 1)
 * @param faults Fault flags (set on domain error if shift > 62)
 * @return Stochastically rounded 32-bit result
 *
 * @details Provides the regularization benefits of stochastic rounding
 *          while remaining fully deterministic. The probability of
 *          rounding up equals the fractional part value.
 *
 * Complexity: O(1)
 * Determinism: Bit-perfect - same inputs always produce same output
 *
 * @ref CT-MATH-001 §8.4
 */
int32_t ct_stochastic_round(int64_t x, uint32_t shift, ct_prng_t *prng,
                            ct_fault_flags_t *faults);

/**
 * @brief Compute 64-bit op_id from context
 *
 * @param layer_id    Layer index in model
 * @param tensor_id   Tensor index within layer  
 * @param element_idx Element index within tensor
 * @return 64-bit operation identifier
 *
 * @details Combines multiple indices into a single unique identifier
 *          suitable for use with ct_prng_init().
 *
 * @note For very large models, consider using a 128-bit hash instead.
 *
 * @ref CT-MATH-001 §6.3
 */
uint64_t ct_prng_make_op_id(uint32_t layer_id, uint32_t tensor_id,
                            uint32_t element_idx);

#endif /* CT_PRNG_H */
