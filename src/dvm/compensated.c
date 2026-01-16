/**
 * @file compensated.c
 * @project Certifiable Training
 * @brief Neumaier compensated summation for deterministic gradient reduction
 *
 * @details Implements high-precision summation using the Neumaier algorithm.
 *          This tracks rounding errors during accumulation and compensates,
 *          achieving near-double precision using only integer operations.
 *
 *          Critical for gradient aggregation where many small values are
 *          summed - without compensation, rounding errors would accumulate
 *          and break determinism guarantees.
 *
 * @traceability CT-MATH-001 §9
 * @compliance MISRA-C:2012, DO-178C, IEC 62304, ISO 26262
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 */

#include "compensated.h"
#include "dvm.h"
#include <stddef.h>

/**
 * @brief Safe 64-bit absolute value with saturation
 *
 * @param x Input value
 * @param faults Fault flags (optional, set on INT64_MIN)
 * @return Absolute value, saturated to INT64_MAX for INT64_MIN input
 *
 * @details Handles the edge case where abs(INT64_MIN) would overflow.
 *          Returns INT64_MAX instead and optionally sets fault flag.
 *
 * @ref CT-MATH-001 §9.3 (DVM_Abs64_Sat)
 */
static int64_t abs64_sat(int64_t x, ct_fault_flags_t *faults)
{
    if (x == INT64_MIN) {
        /* abs(INT64_MIN) overflows - saturate */
        if (faults != NULL) {
            faults->overflow = 1;
        }
        return INT64_MAX;
    }
    return (x < 0) ? -x : x;
}

/**
 * @brief Safe 64-bit addition with overflow detection
 *
 * @param a First operand
 * @param b Second operand
 * @param faults Fault flags (set on overflow)
 * @return Sum, saturated on overflow
 *
 * @details Detects overflow before it occurs using the standard
 *          two's complement overflow check pattern.
 */
static int64_t safe_add64(int64_t a, int64_t b, ct_fault_flags_t *faults)
{
    /* Check for overflow before adding */
    if (b > 0 && a > INT64_MAX - b) {
        /* Positive overflow */
        if (faults != NULL) {
            faults->overflow = 1;
        }
        return INT64_MAX;
    }
    if (b < 0 && a < INT64_MIN - b) {
        /* Negative overflow */
        if (faults != NULL) {
            faults->underflow = 1;
        }
        return INT64_MIN;
    }
    return a + b;
}

/**
 * @brief Initialize compensated accumulator to zero
 */
void ct_comp_init(ct_comp_accum_t *accum)
{
    if (accum == NULL) {
        return;
    }
    
    accum->sum = 0;
    accum->err = 0;
}

/**
 * @brief Initialize compensated accumulator with a value
 */
void ct_comp_init_value(ct_comp_accum_t *accum, int64_t initial_value)
{
    if (accum == NULL) {
        return;
    }
    
    accum->sum = initial_value;
    accum->err = 0;
}

/**
 * @brief Add a value using Neumaier compensated summation
 *
 * @details The Neumaier algorithm improves on Kahan summation by handling
 *          the case where the new value is larger than the running sum.
 *
 *          Mathematical basis:
 *          - Let t = sum + v (computed with rounding)
 *          - The error e = (larger - t) + smaller
 *          - This captures the rounding error exactly (in exact arithmetic)
 *
 *          In fixed-point, we still accumulate most of the error, providing
 *          much better precision than naive summation.
 *
 * @ref CT-MATH-001 §9.3
 */
void ct_comp_add(ct_comp_accum_t *accum, int64_t value, ct_fault_flags_t *faults)
{
    if (accum == NULL) {
        return;
    }
    
    /* Compute t = sum + value */
    int64_t t = safe_add64(accum->sum, value, faults);
    
    /* Compute error term based on relative magnitudes */
    int64_t e;
    
    if (abs64_sat(accum->sum, faults) >= abs64_sat(value, faults)) {
        /*
         * |sum| >= |value|: sum is the "big" number
         * e = (sum - t) + value
         * 
         * In exact arithmetic: sum - t = -value, so e = 0
         * With rounding: e captures the lost bits
         */
        e = (accum->sum - t) + value;
    } else {
        /*
         * |value| > |sum|: value is the "big" number  
         * e = (value - t) + sum
         */
        e = (value - t) + accum->sum;
    }
    
    /* Update accumulator */
    accum->sum = t;
    accum->err = safe_add64(accum->err, e, faults);
}

/**
 * @brief Merge two compensated accumulators
 *
 * @details Used in tree reduction. When merging two child nodes:
 *          1. Add the child's sum using compensated addition
 *          2. Add the child's error term to our error term
 *
 *          This preserves the error tracking through the reduction tree.
 */
void ct_comp_merge(ct_comp_accum_t *dest, const ct_comp_accum_t *src,
                   ct_fault_flags_t *faults)
{
    if (dest == NULL || src == NULL) {
        return;
    }
    
    /* Add src's sum to dest using compensated addition */
    ct_comp_add(dest, src->sum, faults);
    
    /* Merge error terms */
    dest->err = safe_add64(dest->err, src->err, faults);
}

/**
 * @brief Extract final sum with error compensation
 *
 * @details The true accumulated value is sum + err. This should be
 *          called once at the end of all additions to get the final
 *          high-precision result.
 */
int64_t ct_comp_finalize(const ct_comp_accum_t *accum, ct_fault_flags_t *faults)
{
    if (accum == NULL) {
        return 0;
    }
    
    return safe_add64(accum->sum, accum->err, faults);
}

/**
 * @brief Extract sum without compensation
 */
int64_t ct_comp_get_sum(const ct_comp_accum_t *accum)
{
    if (accum == NULL) {
        return 0;
    }
    return accum->sum;
}

/**
 * @brief Extract error term
 */
int64_t ct_comp_get_error(const ct_comp_accum_t *accum)
{
    if (accum == NULL) {
        return 0;
    }
    return accum->err;
}

/**
 * @brief Sum an array using compensated arithmetic
 *
 * @details Simple sequential sum using Neumaier algorithm.
 *          For tree-based parallel reduction, use the reduction
 *          tree module instead.
 */
int64_t ct_comp_sum_array(const int64_t *values, uint32_t count,
                          ct_fault_flags_t *faults)
{
    if (values == NULL || count == 0) {
        return 0;
    }
    
    /* Check batch size limit */
    if (count > CT_MAX_BATCH_SIZE) {
        if (faults != NULL) {
            faults->domain = 1;
        }
        /* Process anyway but flag the violation */
    }
    
    ct_comp_accum_t accum;
    ct_comp_init(&accum);
    
    for (uint32_t i = 0; i < count; i++) {
        ct_comp_add(&accum, values[i], faults);
    }
    
    return ct_comp_finalize(&accum, faults);
}

/**
 * @brief Sum an array of 32-bit fixed-point values
 *
 * @details Widens each value to 64-bit before accumulation.
 *          This prevents overflow within individual additions
 *          and provides extended precision for the result.
 */
int64_t ct_comp_sum_array_32(const int32_t *values, uint32_t count,
                             ct_fault_flags_t *faults)
{
    if (values == NULL || count == 0) {
        return 0;
    }
    
    /* Check batch size limit */
    if (count > CT_MAX_BATCH_SIZE) {
        if (faults != NULL) {
            faults->domain = 1;
        }
    }
    
    ct_comp_accum_t accum;
    ct_comp_init(&accum);
    
    for (uint32_t i = 0; i < count; i++) {
        /* Widen to 64-bit before adding */
        ct_comp_add(&accum, (int64_t)values[i], faults);
    }
    
    return ct_comp_finalize(&accum, faults);
}

/**
 * @brief Compute mean using compensated sum
 */
int64_t ct_comp_mean_array(const int64_t *values, uint32_t count,
                           ct_fault_flags_t *faults)
{
    if (values == NULL || count == 0) {
        if (faults != NULL) {
            faults->div_zero = 1;
        }
        return 0;
    }
    
    int64_t sum = ct_comp_sum_array(values, count, faults);
    
    /* Integer division - truncate toward zero */
    return sum / (int64_t)count;
}
