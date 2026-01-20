/**
 * @file activation.c
 * @project Certifiable Training
 * @brief Extended activation layer operations
 *
 * @details This module provides activation layer state management for
 *          training with gradient caching. The core activation functions
 *          (ReLU, sigmoid, tanh) are implemented in forward.c.
 *
 *          This module adds:
 *          - Activation layer context with gradient cache
 *          - Pre-activation caching for backward pass
 *          - Activation derivative computation
 *
 * @traceability CT-MATH-001 §12, SRS-005-FORWARD
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

/* ============================================================================
 * Activation Layer Context
 * ============================================================================ */

/**
 * @brief Activation layer context for training
 *
 * @details Wraps ct_activation_t with gradient caching for backprop.
 */
typedef struct {
    ct_activation_t base;           /**< Base activation layer */
    ct_tensor_t *pre_activation;    /**< Cached pre-activation values */
    ct_tensor_t *activation_output; /**< Cached post-activation values */
    bool cache_valid;               /**< Cache validity flag */
} ct_activation_layer_t;

/**
 * @brief Initialize activation layer context
 *
 * @param layer Layer to initialize
 * @param type Activation type (ReLU, sigmoid, tanh)
 * @param lut LUT for sigmoid/tanh (NULL for ReLU)
 * @param pre_act_buffer Buffer for pre-activation cache [size]
 * @param post_act_buffer Buffer for post-activation cache [size]
 * @param size Number of elements
 * @return CT_OK on success
 */
ct_error_t ct_activation_layer_init(ct_activation_layer_t *layer,
                                    ct_activation_type_t type,
                                    const ct_activation_lut_t *lut,
                                    fixed_t *pre_act_buffer,
                                    fixed_t *post_act_buffer,
                                    uint32_t size)
{
    if (layer == NULL) return CT_ERR_NULL;

    ct_activation_init(&layer->base, type, lut);

    if (pre_act_buffer != NULL) {
        layer->pre_activation = (ct_tensor_t *)pre_act_buffer;  /* Reinterpret */
        ct_tensor_init_1d(layer->pre_activation, pre_act_buffer, size);
    } else {
        layer->pre_activation = NULL;
    }

    if (post_act_buffer != NULL) {
        layer->activation_output = (ct_tensor_t *)post_act_buffer;
        ct_tensor_init_1d(layer->activation_output, post_act_buffer, size);
    } else {
        layer->activation_output = NULL;
    }

    layer->cache_valid = false;

    return CT_OK;
}

/**
 * @brief Forward pass with caching for backprop
 *
 * @param layer Activation layer
 * @param input Input tensor
 * @param output Output tensor
 * @param faults Fault accumulator
 * @return CT_OK on success
 */
ct_error_t ct_activation_layer_forward(ct_activation_layer_t *layer,
                                       const ct_tensor_t *input,
                                       ct_tensor_t *output,
                                       ct_fault_flags_t *faults)
{
    if (layer == NULL || input == NULL || output == NULL) {
        return CT_ERR_NULL;
    }

    /* Cache pre-activation values for backward pass */
    if (layer->pre_activation != NULL && input->total_size <= layer->pre_activation->total_size) {
        for (uint32_t i = 0; i < input->total_size; i++) {
            layer->pre_activation->data[i] = input->data[i];
        }
    }

    /* Apply activation */
    ct_error_t err = ct_activation_forward(&layer->base, input, output, faults);
    if (err != CT_OK) return err;

    /* Cache post-activation values */
    if (layer->activation_output != NULL && output->total_size <= layer->activation_output->total_size) {
        for (uint32_t i = 0; i < output->total_size; i++) {
            layer->activation_output->data[i] = output->data[i];
        }
        layer->cache_valid = true;
    }

    return CT_OK;
}

/**
 * @brief Backward pass through activation layer
 *
 * @param layer Activation layer with cached values
 * @param grad_output Upstream gradient (Q8.24)
 * @param grad_input Output gradient (Q8.24)
 * @param faults Fault accumulator
 * @return CT_OK on success
 */
ct_error_t ct_activation_layer_backward(const ct_activation_layer_t *layer,
                                        const ct_grad_tensor_t *grad_output,
                                        ct_grad_tensor_t *grad_input,
                                        ct_fault_flags_t *faults)
{
    if (layer == NULL || grad_output == NULL || grad_input == NULL) {
        return CT_ERR_NULL;
    }

    switch (layer->base.type) {
        case CT_ACT_NONE:
            /* Identity: grad_input = grad_output */
            for (uint32_t i = 0; i < grad_output->total_size; i++) {
                grad_input->data[i] = grad_output->data[i];
            }
            break;

        case CT_ACT_RELU:
            /* ReLU backward uses pre-activation cache */
            if (layer->pre_activation == NULL) {
                return CT_ERR_STATE;
            }
            return ct_activation_relu_backward(grad_output, layer->pre_activation,
                                               grad_input, faults);

        case CT_ACT_SIGMOID:
            /* Sigmoid backward uses post-activation cache */
            if (layer->activation_output == NULL) {
                return CT_ERR_STATE;
            }
            return ct_activation_sigmoid_backward(grad_output, layer->activation_output,
                                                  grad_input, faults);

        case CT_ACT_TANH:
            /* Tanh backward uses post-activation cache */
            if (layer->activation_output == NULL) {
                return CT_ERR_STATE;
            }
            return ct_activation_tanh_backward(grad_output, layer->activation_output,
                                               grad_input, faults);

        default:
            return CT_ERR_CONFIG;
    }

    return CT_OK;
}

/**
 * @brief Invalidate activation cache
 *
 * @param layer Activation layer
 */
void ct_activation_layer_invalidate_cache(ct_activation_layer_t *layer)
{
    if (layer != NULL) {
        layer->cache_valid = false;
    }
}

/**
 * @brief Check if cache is valid
 *
 * @param layer Activation layer
 * @return true if cache is valid
 */
bool ct_activation_layer_cache_valid(const ct_activation_layer_t *layer)
{
    return (layer != NULL) && layer->cache_valid;
}
