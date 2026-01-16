/**
 * @file compensated.h
 * @project Certifiable Training
 * @brief Neumaier compensated summation for deterministic gradient reduction
 *
 * @details Provides high-precision summation using compensated arithmetic.
 *          The Neumaier algorithm tracks rounding errors and compensates,
 *          achieving near-double precision using only integer operations.
 *
 *          Used for gradient aggregation where many small values must be
 *          summed without accumulating rounding errors.
 *
 * @traceability CT-MATH-001 §9, CT-STRUCT-001 §4.2
 * @compliance MISRA-C:2012, DO-178C, IEC 62304, ISO 26262
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 */

#ifndef CT_COMPENSATED_H
#define CT_COMPENSATED_H

#include "ct_types.h"

/**
 * @brief Compensated accumulator structure
 *
 * @details Holds (sum, error) pair for Neumaier compensated summation.
 *          The error term tracks rounding discrepancies, enabling
 *          near-double precision from single-precision operations.
 *
 * @invariant sum + err represents the true accumulated value
 * @invariant err is typically much smaller than sum
 *
 * @ref CT-MATH-001 §9.2, CT-STRUCT-001 §4.2
 */
typedef struct {
    int64_t sum;    /**< Running sum (Q32.32 or Q16.16 extended) */
    int64_t err;    /**< Compensation term (tracks rounding error) */
} ct_comp_accum_t;

/**
 * @brief Initialize compensated accumulator to zero
 *
 * @param accum Pointer to accumulator structure
 *
 * @pre accum != NULL
 * @post accum->sum = 0, accum->err = 0
 *
 * Complexity: O(1)
 */
void ct_comp_init(ct_comp_accum_t *accum);

/**
 * @brief Initialize compensated accumulator with a value
 *
 * @param accum Pointer to accumulator structure
 * @param initial_value Starting value for the sum
 *
 * @pre accum != NULL
 * @post accum->sum = initial_value, accum->err = 0
 *
 * Complexity: O(1)
 */
void ct_comp_init_value(ct_comp_accum_t *accum, int64_t initial_value);

/**
 * @brief Add a value using Neumaier compensated summation
 *
 * @param accum  Pointer to accumulator (modified in place)
 * @param value  Value to add
 * @param faults Fault flags (set on overflow detection)
 *
 * @details Implements the Neumaier algorithm which is more robust than
 *          Kahan summation when the new value may be larger than the
 *          running sum. Tracks rounding errors in the compensation term.
 *
 *          Algorithm:
 *          1. Compute t = sum + value
 *          2. If |sum| >= |value|: e = (sum - t) + value
 *             Else: e = (value - t) + sum
 *          3. Update: sum = t, err = err + e
 *
 * @pre accum != NULL
 * @post accum contains compensated sum including value
 *
 * Complexity: O(1)
 * Determinism: Bit-perfect across platforms
 *
 * @ref CT-MATH-001 §9.3
 */
void ct_comp_add(ct_comp_accum_t *accum, int64_t value, ct_fault_flags_t *faults);

/**
 * @brief Merge two compensated accumulators
 *
 * @param dest   Destination accumulator (modified in place)
 * @param src    Source accumulator to merge
 * @param faults Fault flags
 *
 * @details Combines two compensated sums while preserving error tracking.
 *          Used in tree reduction to merge child node results.
 *
 *          Process:
 *          1. Add src.sum to dest using compensated addition
 *          2. Add src.err to dest.err (error terms combine additively)
 *
 * @pre dest != NULL, src != NULL
 * @post dest contains merged result
 *
 * Complexity: O(1)
 * Determinism: Bit-perfect
 *
 * @ref CT-MATH-001 §9.3
 */
void ct_comp_merge(ct_comp_accum_t *dest, const ct_comp_accum_t *src,
                   ct_fault_flags_t *faults);

/**
 * @brief Extract final sum with error compensation
 *
 * @param accum  Pointer to accumulator
 * @param faults Fault flags (set on overflow)
 * @return Final compensated sum (sum + err)
 *
 * @details Returns the full-precision result by adding the compensation
 *          term back to the sum. This should be done only once at the
 *          end of accumulation.
 *
 * Complexity: O(1)
 * Determinism: Bit-perfect
 */
int64_t ct_comp_finalize(const ct_comp_accum_t *accum, ct_fault_flags_t *faults);

/**
 * @brief Extract sum without compensation (for inspection)
 *
 * @param accum Pointer to accumulator
 * @return Current sum value (without error term)
 *
 * @note Use ct_comp_finalize() for the true accumulated value.
 *       This function is for debugging/inspection only.
 */
int64_t ct_comp_get_sum(const ct_comp_accum_t *accum);

/**
 * @brief Extract error term (for inspection/analysis)
 *
 * @param accum Pointer to accumulator
 * @return Current error compensation term
 *
 * @note For debugging and numerical analysis. A large error term
 *       relative to sum indicates significant rounding compensation.
 */
int64_t ct_comp_get_error(const ct_comp_accum_t *accum);

/**
 * @brief Sum an array using compensated arithmetic
 *
 * @param values Array of values to sum
 * @param count  Number of elements
 * @param faults Fault flags
 * @return Compensated sum of all values
 *
 * @details Convenience function that initializes an accumulator,
 *          adds all values, and returns the finalized result.
 *
 * @pre values != NULL or count == 0
 * @pre count <= CT_MAX_BATCH_SIZE (65536)
 *
 * Complexity: O(count)
 * Determinism: Bit-perfect - same input array always produces same output
 *
 * @ref CT-MATH-001 §9.5 (batch size limits)
 */
int64_t ct_comp_sum_array(const int64_t *values, uint32_t count,
                          ct_fault_flags_t *faults);

/**
 * @brief Sum an array of 32-bit fixed-point values
 *
 * @param values Array of Q16.16 or Q8.24 values
 * @param count  Number of elements
 * @param faults Fault flags
 * @return Compensated sum (64-bit to avoid overflow)
 *
 * @details Widens 32-bit values to 64-bit before accumulation.
 *          Result is in the same fixed-point format, extended precision.
 *
 * @pre values != NULL or count == 0
 * @pre count <= CT_MAX_BATCH_SIZE
 *
 * Complexity: O(count)
 * Determinism: Bit-perfect
 */
int64_t ct_comp_sum_array_32(const int32_t *values, uint32_t count,
                             ct_fault_flags_t *faults);

/**
 * @brief Compute mean using compensated sum
 *
 * @param values Array of values
 * @param count  Number of elements (must be > 0)
 * @param faults Fault flags
 * @return Mean value (sum / count)
 *
 * @details Computes compensated sum, then divides by count.
 *          Division uses truncation toward zero.
 *
 * @pre values != NULL, count > 0
 *
 * Complexity: O(count)
 * Determinism: Bit-perfect
 */
int64_t ct_comp_mean_array(const int64_t *values, uint32_t count,
                           ct_fault_flags_t *faults);

#endif /* CT_COMPENSATED_H */
