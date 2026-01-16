/**
 * @file dvm.h
 * @project Certifiable Training
 * @brief Deterministic Virtual Machine primitives
 * @traceability CT-MATH-001 §3, CT-SPEC-001 §3
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 */

#ifndef CT_DVM_H
#define CT_DVM_H

#include "ct_types.h"

fixed_t dvm_add(fixed_t a, fixed_t b, ct_fault_flags_t *faults);
fixed_t dvm_sub(fixed_t a, fixed_t b, ct_fault_flags_t *faults);
fixed_t dvm_mul(fixed_t a, fixed_t b, ct_fault_flags_t *faults);
int32_t dvm_div_int32(int32_t a, int32_t b, ct_fault_flags_t *faults);
fixed_t dvm_div_q(fixed_t a, fixed_t b, uint32_t frac_bits, ct_fault_flags_t *faults);
int32_t dvm_clamp32(int64_t x, ct_fault_flags_t *faults);
int64_t dvm_abs64_sat(int64_t x, ct_fault_flags_t *faults);
int32_t dvm_round_shift_rne(int64_t x, uint32_t shift, ct_fault_flags_t *faults);

#endif /* CT_DVM_H */
