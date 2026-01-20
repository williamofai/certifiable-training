/**
 * @file linear.c
 * @project Certifiable Training
 * @brief Extended linear layer with gradient accumulation
 *
 * @details This module provides a training-aware linear layer that:
 *          - Caches input for backward pass
 *          - Accumulates gradients across batch
 *          - Supports gradient averaging
 *
 *          Core linear forward/backward are in forward.c and backward.c.
 *          This module adds batch accumulation and state management.
 *
 * @traceability CT-MATH-001 §7, SRS-005-FORWARD, SRS-006-BACKWARD
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304, DO-178C
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 * @license GPL-3.0 or Commercial License (william@fstopify.com)
 */

#include "ct_types.h"
#include "forward.h"
#include "backward.h"
#include "dvm.h"
#include <string.h>

/* ============================================================================
 * Extended Linear Layer
 * ============================================================================ */

/**
 * @brief Extended linear layer for training
 *
 * @details Wraps ct_linear_t with gradient buffers and input caching.
 */
typedef struct {
    ct_linear_t layer;              /**< Base linear layer */
    ct_linear_grad_t grad;          /**< Gradient cache */
    ct_tensor_t input_cache;        /**< Cached input for backward */
    fixed_t *input_cache_buf;       /**< Input cache buffer */
    uint32_t batch_count;           /**< Samples accumulated */
    bool grad_initialized;          /**< Gradient buffers zeroed */
} ct_linear_layer_t;

/**
 * @brief Initialize extended linear layer
 *
 * @param ext Layer to initialize
 * @param weights_buf Buffer for weights [output_size * input_size]
 * @param bias_buf Buffer for bias [output_size]
 * @param grad_weights_buf Buffer for weight gradients (Q8.24)
 * @param grad_bias_buf Buffer for bias gradients (Q8.24)
 * @param input_cache_buf Buffer for input caching
 * @param input_size Input dimension
 * @param output_size Output dimension
 * @return CT_OK on success
 */
ct_error_t ct_linear_layer_init(ct_linear_layer_t *ext,
                                fixed_t *weights_buf,
                                fixed_t *bias_buf,
                                fixed_hp_t *grad_weights_buf,
                                fixed_hp_t *grad_bias_buf,
                                fixed_t *input_cache_buf,
                                uint32_t input_size,
                                uint32_t output_size)
{
    if (ext == NULL || weights_buf == NULL || bias_buf == NULL) {
        return CT_ERR_NULL;
    }

    /* Initialize base layer */
    ct_error_t err = ct_linear_init(&ext->layer, weights_buf, bias_buf,
                                    input_size, output_size);
    if (err != CT_OK) return err;

    /* Initialize gradient cache */
    if (grad_weights_buf != NULL && grad_bias_buf != NULL) {
        err = ct_linear_grad_init(&ext->grad, grad_weights_buf, grad_bias_buf,
                                  &ext->input_cache, input_size, output_size);
        if (err != CT_OK) return err;
    }

    /* Initialize input cache */
    if (input_cache_buf != NULL) {
        ct_tensor_init_1d(&ext->input_cache, input_cache_buf, input_size);
        ext->input_cache_buf = input_cache_buf;
    } else {
        ext->input_cache_buf = NULL;
    }

    ext->batch_count = 0;
    ext->grad_initialized = false;

    return CT_OK;
}

/**
 * @brief Zero accumulated gradients
 *
 * @param ext Extended linear layer
 */
void ct_linear_layer_zero_grad(ct_linear_layer_t *ext)
{
    if (ext == NULL) return;

    ct_grad_tensor_zero(&ext->grad.grad_weights);
    ct_grad_tensor_zero(&ext->grad.grad_bias);
    ext->batch_count = 0;
    ext->grad_initialized = true;
}

/**
 * @brief Forward pass with input caching
 *
 * @param ext Extended linear layer
 * @param input Input tensor [input_size]
 * @param output Output tensor [output_size]
 * @param faults Fault accumulator
 * @return CT_OK on success
 */
ct_error_t ct_linear_layer_forward(ct_linear_layer_t *ext,
                                   const ct_tensor_t *input,
                                   ct_tensor_t *output,
                                   ct_fault_flags_t *faults)
{
    if (ext == NULL || input == NULL || output == NULL) {
        return CT_ERR_NULL;
    }

    /* Cache input for backward pass */
    if (ext->input_cache_buf != NULL) {
        for (uint32_t i = 0; i < input->total_size && i < ext->layer.input_size; i++) {
            ext->input_cache.data[i] = input->data[i];
        }
    }

    /* Forward pass */
    return ct_linear_forward(&ext->layer, input, output, faults);
}

/**
 * @brief Backward pass with gradient accumulation
 *
 * @param ext Extended linear layer
 * @param grad_output Upstream gradient (Q8.24)
 * @param grad_input Output gradient for previous layer (Q8.24), can be NULL
 * @param faults Fault accumulator
 * @return CT_OK on success
 */
ct_error_t ct_linear_layer_backward(ct_linear_layer_t *ext,
                                    const ct_grad_tensor_t *grad_output,
                                    ct_grad_tensor_t *grad_input,
                                    ct_fault_flags_t *faults)
{
    if (ext == NULL || grad_output == NULL) {
        return CT_ERR_NULL;
    }

    if (!ext->grad_initialized) {
        ct_linear_layer_zero_grad(ext);
    }

    /* Compute gradients and accumulate */
    ct_error_t err = ct_linear_backward(&ext->layer, &ext->grad,
                                        grad_output, grad_input, faults);
    if (err != CT_OK) return err;

    ext->batch_count++;
    return CT_OK;
}

/**
 * @brief Get accumulated gradients (averaged over batch)
 *
 * @param ext Extended linear layer
 * @param avg_grad_weights Output: averaged weight gradients
 * @param avg_grad_bias Output: averaged bias gradients
 * @param faults Fault accumulator
 * @return CT_OK on success
 *
 * @note Caller provides output buffers; this computes grad / batch_count
 */
ct_error_t ct_linear_layer_get_avg_grad(const ct_linear_layer_t *ext,
                                        ct_grad_tensor_t *avg_grad_weights,
                                        ct_grad_tensor_t *avg_grad_bias,
                                        ct_fault_flags_t *faults)
{
    if (ext == NULL || avg_grad_weights == NULL || avg_grad_bias == NULL) {
        return CT_ERR_NULL;
    }

    if (ext->batch_count == 0) {
        return CT_ERR_STATE;  /* No gradients accumulated */
    }

    /* Compute 1/batch_count in Q8.24 */
    /* Note: For small batch sizes, this is straightforward integer division */
    int64_t scale = CT_GRAD_ONE / (int64_t)ext->batch_count;

    /* Average weight gradients */
    for (uint32_t i = 0; i < ext->grad.grad_weights.total_size; i++) {
        int64_t prod = (int64_t)ext->grad.grad_weights.data[i] * scale;
        avg_grad_weights->data[i] = dvm_round_shift_rne(prod, CT_GRAD_FRAC_BITS, faults);
    }

    /* Average bias gradients */
    for (uint32_t i = 0; i < ext->grad.grad_bias.total_size; i++) {
        int64_t prod = (int64_t)ext->grad.grad_bias.data[i] * scale;
        avg_grad_bias->data[i] = dvm_round_shift_rne(prod, CT_GRAD_FRAC_BITS, faults);
    }

    return CT_OK;
}

/**
 * @brief Get batch count
 *
 * @param ext Extended linear layer
 * @return Number of samples accumulated
 */
uint32_t ct_linear_layer_get_batch_count(const ct_linear_layer_t *ext)
{
    return (ext != NULL) ? ext->batch_count : 0;
}

/**
 * @brief Get underlying linear layer (for optimizer updates)
 *
 * @param ext Extended linear layer
 * @return Pointer to base layer, or NULL
 */
ct_linear_t *ct_linear_layer_get_base(ct_linear_layer_t *ext)
{
    return (ext != NULL) ? &ext->layer : NULL;
}

/**
 * @brief Get gradient cache (for optimizer access)
 *
 * @param ext Extended linear layer
 * @return Pointer to gradient cache, or NULL
 */
ct_linear_grad_t *ct_linear_layer_get_grad(ct_linear_layer_t *ext)
{
    return (ext != NULL) ? &ext->grad : NULL;
}
