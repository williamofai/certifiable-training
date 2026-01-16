/**
 * @file primitives.c
 * @project Certifiable Training
 * @brief DVM arithmetic primitives
 * @traceability CT-MATH-001 §3
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 */

#include "dvm.h"

fixed_t dvm_add(fixed_t a, fixed_t b, ct_fault_flags_t *faults) {
    int64_t wide = (int64_t)a + (int64_t)b;
    return dvm_clamp32(wide, faults);
}

fixed_t dvm_sub(fixed_t a, fixed_t b, ct_fault_flags_t *faults) {
    int64_t wide = (int64_t)a - (int64_t)b;
    return dvm_clamp32(wide, faults);
}

fixed_t dvm_mul(fixed_t a, fixed_t b, ct_fault_flags_t *faults) {
    int64_t wide = (int64_t)a * (int64_t)b;
    return dvm_round_shift_rne(wide, FIXED_FRAC_BITS, faults);
}

int32_t dvm_div_int32(int32_t a, int32_t b, ct_fault_flags_t *faults) {
    if (b == 0) {
        if (faults) faults->div_zero = 1;
        return 0;
    }
    return a / b;
}

fixed_t dvm_div_q(fixed_t a, fixed_t b, uint32_t frac_bits, ct_fault_flags_t *faults) {
    if (b == 0) {
        if (faults) faults->div_zero = 1;
        return 0;
    }
    if (frac_bits > CT_MAX_SHIFT) {
        if (faults) faults->domain = 1;
        return 0;
    }
    int64_t wide = (int64_t)a << frac_bits;
    return dvm_clamp32(wide / b, faults);
}

int32_t dvm_clamp32(int64_t x, ct_fault_flags_t *faults) {
    if (x > INT32_MAX) {
        if (faults) faults->overflow = 1;
        return INT32_MAX;
    }
    if (x < INT32_MIN) {
        if (faults) faults->underflow = 1;
        return INT32_MIN;
    }
    return (int32_t)x;
}

int64_t dvm_abs64_sat(int64_t x, ct_fault_flags_t *faults) {
    if (x == INT64_MIN) {
        if (faults) faults->overflow = 1;
        return INT64_MAX;
    }
    return (x < 0) ? -x : x;
}

int32_t dvm_round_shift_rne(int64_t x, uint32_t shift, ct_fault_flags_t *faults) {
    if (shift > CT_MAX_SHIFT) {
        if (faults) faults->domain = 1;
        return 0;
    }
    if (shift == 0) {
        return dvm_clamp32(x, faults);
    }
    
    int64_t mask = (1LL << shift) - 1;
    int64_t halfway = 1LL << (shift - 1);
    int64_t fraction = x & mask;
    int64_t quotient = x >> shift;
    
    int64_t result;
    if (fraction < halfway) {
        result = quotient;
    } else if (fraction > halfway) {
        result = quotient + 1;
    } else {
        result = quotient + (quotient & 1);
    }
    
    return dvm_clamp32(result, faults);
}
