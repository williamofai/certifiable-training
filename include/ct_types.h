/**
 * @file ct_types.h
 * @project Certifiable Training
 * @brief Core type definitions for deterministic ML training
 *
 * @traceability CT-MATH-001, CT-STRUCT-001
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304, DO-178C
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 * @license GPL-3.0 or Commercial License (william@fstopify.com)
 */

#ifndef CT_TYPES_H
#define CT_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <limits.h>

/* Fixed-Point Types (CT-MATH-001 §2.2) */
typedef int32_t fixed_t;
typedef int32_t fixed_hp_t;
typedef int64_t fixed_acc_t;

/* Q16.16 Constants */
#define FIXED_FRAC_BITS     16
#define FIXED_ONE           ((fixed_t)(1 << FIXED_FRAC_BITS))
#define FIXED_HALF          ((fixed_t)(1 << (FIXED_FRAC_BITS - 1)))
#define FIXED_ZERO          ((fixed_t)0)
#define FIXED_MAX           ((fixed_t)INT32_MAX)
#define FIXED_MIN           ((fixed_t)INT32_MIN)
#define FIXED_EPS           ((fixed_t)1)

/* Q8.24 Constants */
#define FIXED_HP_FRAC_BITS  24
#define FIXED_HP_ONE        ((fixed_hp_t)(1 << FIXED_HP_FRAC_BITS))

/* Limits */
#define CT_MAX_BATCH_SIZE   65536
#define CT_MAX_DIMS         4
#define CT_MAX_SHIFT        62

/* Static Assert */
#define CT_STATIC_ASSERT(cond, msg) \
    typedef char ct_static_assert_##__LINE__[(cond) ? 1 : -1]

/* Error Codes (CT-STRUCT-001 §11.2) */
typedef enum {
    CT_OK               =  0,
    CT_ERR_NULL         = -1,
    CT_ERR_DIMENSION    = -2,
    CT_ERR_OVERFLOW     = -3,
    CT_ERR_UNDERFLOW    = -4,
    CT_ERR_DIV_ZERO     = -5,
    CT_ERR_DOMAIN       = -6,
    CT_ERR_CONFIG       = -7,
    CT_ERR_STATE        = -8,
    CT_ERR_MEMORY       = -9,
    CT_ERR_HASH         = -10,
    CT_ERR_FAULT        = -11
} ct_error_t;

/* Fault Flags (CT-STRUCT-001 §11.1) */
typedef struct {
    uint32_t overflow    : 1;
    uint32_t underflow   : 1;
    uint32_t div_zero    : 1;
    uint32_t domain      : 1;
    uint32_t grad_floor  : 1;
    uint32_t _reserved   : 27;
} ct_fault_flags_t;

static inline bool ct_has_fault(const ct_fault_flags_t *f) {
    return f->overflow || f->underflow || f->div_zero || f->domain;
}

static inline void ct_clear_faults(ct_fault_flags_t *f) {
    f->overflow = 0;
    f->underflow = 0;
    f->div_zero = 0;
    f->domain = 0;
    f->grad_floor = 0;
}

#endif /* CT_TYPES_H */
