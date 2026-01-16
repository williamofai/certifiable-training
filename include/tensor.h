/**
 * @file tensor.h
 * @project Certifiable Training
 * @brief Tensor data structures
 * @traceability CT-STRUCT-001 §5
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 */

#ifndef CT_TENSOR_H
#define CT_TENSOR_H

#include "ct_types.h"

typedef struct {
    fixed_t *data;
    uint32_t dims[CT_MAX_DIMS];
    uint32_t strides[CT_MAX_DIMS];
    uint32_t ndims;
    uint32_t total_size;
} ct_tensor_t;

typedef struct {
    fixed_hp_t *data;
    uint32_t dims[CT_MAX_DIMS];
    uint32_t strides[CT_MAX_DIMS];
    uint32_t ndims;
    uint32_t total_size;
} ct_grad_tensor_t;

ct_error_t ct_tensor_init(ct_tensor_t *t, fixed_t *buffer, uint32_t ndims, const uint32_t *dims);
bool ct_tensor_is_contiguous(const ct_tensor_t *t);

#endif /* CT_TENSOR_H */
