/**
 * @file backward.c
 * @project Certifiable Training
 * @brief Deterministic backward pass (backpropagation) implementation.
 *
 * @details All gradient computation in Q8.24 high-precision format.
 *          Uses DVM primitives for deterministic arithmetic.
 *
 * @traceability SRS-006-BACKWARD, CT-MATH-001 §7
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust. All rights reserved.
 * @license Licensed under the GPL-3.0 (Open Source) or Commercial License.
 *          For commercial licensing: william@fstopify.com
 */

#include "backward.h"
#include "dvm.h"
#include "compensated.h"
#include <string.h>

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

/**
 * @brief Multiply two Q8.24 values, return Q8.24
 * Uses 64-bit intermediate with proper rounding
 */
static fixed_hp_t grad_mul(fixed_hp_t a, fixed_hp_t b, ct_fault_flags_t *faults) {
    int64_t wide = (int64_t)a * (int64_t)b;
    /* Round to nearest, ties to even */
    int64_t half = (int64_t)1 << (CT_GRAD_FRAC_BITS - 1);
    wide += half;
    int64_t shifted = wide >> CT_GRAD_FRAC_BITS;
    
    /* Clamp to Q8.24 range (same as int32) */
    if (shifted > INT32_MAX) {
        if (faults) faults->overflow = 1;
        return INT32_MAX;
    }
    if (shifted < INT32_MIN) {
        if (faults) faults->underflow = 1;
        return INT32_MIN;
    }
    return (fixed_hp_t)shifted;
}

/**
 * @brief Multiply Q8.24 by Q16.16, return Q8.24
 * Handles the mixed-precision case
 */
static fixed_hp_t grad_mul_fixed(fixed_hp_t grad, fixed_t value, ct_fault_flags_t *faults) {
    int64_t wide = (int64_t)grad * (int64_t)value;
    /* Need to shift by 16 (the Q16.16 fraction bits) */
    int64_t half = (int64_t)1 << (FIXED_FRAC_BITS - 1);
    wide += half;
    int64_t shifted = wide >> FIXED_FRAC_BITS;
    
    if (shifted > INT32_MAX) {
        if (faults) faults->overflow = 1;
        return INT32_MAX;
    }
    if (shifted < INT32_MIN) {
        if (faults) faults->underflow = 1;
        return INT32_MIN;
    }
    return (fixed_hp_t)shifted;
}



/* ============================================================================
 * Format Conversion
 * ============================================================================ */

fixed_t ct_grad_to_fixed(fixed_hp_t grad, ct_fault_flags_t *faults) {
    /* Shift right by 8 bits (24-16=8) with rounding */
    int32_t shift = CT_GRAD_FRAC_BITS - FIXED_FRAC_BITS;
    int32_t half = 1 << (shift - 1);
    int64_t rounded = (int64_t)grad + half;
    int64_t shifted = rounded >> shift;
    
    if (shifted > INT32_MAX) {
        if (faults) faults->overflow = 1;
        return INT32_MAX;
    }
    if (shifted < INT32_MIN) {
        if (faults) faults->underflow = 1;
        return INT32_MIN;
    }
    return (fixed_t)shifted;
}

/* ============================================================================
 * Gradient Tensor Operations
 * ============================================================================ */

ct_error_t ct_grad_tensor_init(ct_grad_tensor_t *grad,
                               fixed_hp_t *buffer,
                               uint32_t rows,
                               uint32_t cols) {
    if (!grad || !buffer) {
        return CT_ERR_NULL;
    }
    
    grad->data = buffer;
    
    if (cols == 0) {
        /* 1D tensor */
        grad->ndims = 1;
        grad->dims[0] = rows;
        grad->dims[1] = 0;
        grad->strides[0] = 1;
        grad->strides[1] = 0;
        grad->total_size = rows;
    } else {
        /* 2D tensor */
        grad->ndims = 2;
        grad->dims[0] = rows;
        grad->dims[1] = cols;
        grad->strides[0] = cols;
        grad->strides[1] = 1;
        grad->total_size = rows * cols;
    }
    
    return CT_OK;
}

void ct_grad_tensor_zero(ct_grad_tensor_t *grad) {
    if (grad && grad->data) {
        memset(grad->data, 0, grad->total_size * sizeof(fixed_hp_t));
    }
}

/* ============================================================================
 * Loss Functions
 * ============================================================================ */

ct_error_t ct_loss_mse_forward(const ct_tensor_t *output,
                               const ct_tensor_t *target,
                               fixed_t *loss_out,
                               ct_fault_flags_t *faults) {
    if (!output || !target || !loss_out) {
        return CT_ERR_NULL;
    }
    if (output->total_size != target->total_size) {
        return CT_ERR_DIMENSION;
    }
    
    uint32_t n = output->total_size;
    ct_comp_accum_t acc;
    ct_comp_init(&acc);
    
    /* Sum of squared differences */
    for (uint32_t i = 0; i < n; i++) {
        fixed_t diff = dvm_sub(output->data[i], target->data[i], faults);
        /* Square the difference - use 64-bit intermediate */
        int64_t sq = (int64_t)diff * (int64_t)diff;
        /* Convert to accumulator format (Q32.32) */
        fixed_acc_t sq_acc = sq;  /* Already effectively Q32.32 after squaring Q16.16 */
        ct_comp_add(&acc, sq_acc, faults);
    }
    
    /* Divide by N */
    fixed_acc_t sum = ct_comp_get_sum(&acc);
    if (n > 0) {
        sum = sum / (int64_t)n;
    }
    
    /* Convert back to Q16.16 */
    *loss_out = dvm_clamp32(sum >> FIXED_FRAC_BITS, faults);
    
    return CT_OK;
}

ct_error_t ct_loss_mse_backward(const ct_tensor_t *output,
                                const ct_tensor_t *target,
                                ct_grad_tensor_t *grad_output,
                                ct_fault_flags_t *faults) {
    if (!output || !target || !grad_output) {
        return CT_ERR_NULL;
    }
    if (output->total_size != target->total_size ||
        output->total_size != grad_output->total_size) {
        return CT_ERR_DIMENSION;
    }
    
    uint32_t n = output->total_size;
    
    /* Gradient: ∂L/∂ŷ = (2/N)(ŷ - y) */
    /* Compute 2/N in Q8.24 */
    fixed_hp_t two_over_n;
    if (n > 0) {
        /* 2 in Q8.24 */
        int64_t two_q24 = (int64_t)2 << CT_GRAD_FRAC_BITS;
        two_over_n = (fixed_hp_t)(two_q24 / (int64_t)n);
    } else {
        two_over_n = 0;
    }
    
    for (uint32_t i = 0; i < n; i++) {
        /* diff = ŷ - y (in Q16.16) */
        fixed_t diff = dvm_sub(output->data[i], target->data[i], faults);
        /* Convert to Q8.24 */
        fixed_hp_t diff_hp = ct_fixed_to_grad(diff);
        /* Multiply by 2/N */
        fixed_hp_t grad = grad_mul(diff_hp, two_over_n, faults);
        grad_output->data[i] = grad;
    }
    
    return CT_OK;
}

/* ============================================================================
 * Activation Derivatives
 * ============================================================================ */

ct_error_t ct_activation_relu_backward(const ct_grad_tensor_t *grad_output,
                                       const ct_tensor_t *pre_activation,
                                       ct_grad_tensor_t *grad_input,
                                       ct_fault_flags_t *faults) {
    (void)faults;  /* ReLU backward cannot overflow */
    
    if (!grad_output || !pre_activation || !grad_input) {
        return CT_ERR_NULL;
    }
    if (grad_output->total_size != pre_activation->total_size ||
        grad_output->total_size != grad_input->total_size) {
        return CT_ERR_DIMENSION;
    }
    
    uint32_t n = grad_output->total_size;
    
    /* ReLU derivative: 1 if x > 0, 0 otherwise */
    for (uint32_t i = 0; i < n; i++) {
        if (pre_activation->data[i] > 0) {
            grad_input->data[i] = grad_output->data[i];
        } else {
            grad_input->data[i] = 0;
        }
    }
    
    return CT_OK;
}

ct_error_t ct_activation_sigmoid_backward(const ct_grad_tensor_t *grad_output,
                                          const ct_tensor_t *activation,
                                          ct_grad_tensor_t *grad_input,
                                          ct_fault_flags_t *faults) {
    if (!grad_output || !activation || !grad_input) {
        return CT_ERR_NULL;
    }
    if (grad_output->total_size != activation->total_size ||
        grad_output->total_size != grad_input->total_size) {
        return CT_ERR_DIMENSION;
    }
    
    uint32_t n = grad_output->total_size;
    
    /* σ'(x) = σ(x) · (1 - σ(x)) */
    for (uint32_t i = 0; i < n; i++) {
        fixed_t sig = activation->data[i];  /* σ(x) in Q16.16 */
        fixed_t one_minus_sig = dvm_sub(FIXED_ONE, sig, faults);  /* 1 - σ(x) */
        
        /* Compute σ(x) * (1 - σ(x)) in Q16.16 */
        fixed_t deriv = dvm_mul(sig, one_minus_sig, faults);
        
        /* Convert derivative to Q8.24 */
        fixed_hp_t deriv_hp = ct_fixed_to_grad(deriv);
        
        /* Multiply by upstream gradient */
        grad_input->data[i] = grad_mul(grad_output->data[i], deriv_hp, faults);
    }
    
    return CT_OK;
}

ct_error_t ct_activation_tanh_backward(const ct_grad_tensor_t *grad_output,
                                       const ct_tensor_t *activation,
                                       ct_grad_tensor_t *grad_input,
                                       ct_fault_flags_t *faults) {
    if (!grad_output || !activation || !grad_input) {
        return CT_ERR_NULL;
    }
    if (grad_output->total_size != activation->total_size ||
        grad_output->total_size != grad_input->total_size) {
        return CT_ERR_DIMENSION;
    }
    
    uint32_t n = grad_output->total_size;
    
    /* tanh'(x) = 1 - tanh²(x) */
    for (uint32_t i = 0; i < n; i++) {
        fixed_t tanh_x = activation->data[i];  /* tanh(x) in Q16.16 */
        
        /* Compute tanh²(x) */
        fixed_t tanh_sq = dvm_mul(tanh_x, tanh_x, faults);
        
        /* 1 - tanh²(x) */
        fixed_t deriv = dvm_sub(FIXED_ONE, tanh_sq, faults);
        
        /* Convert to Q8.24 */
        fixed_hp_t deriv_hp = ct_fixed_to_grad(deriv);
        
        /* Multiply by upstream gradient */
        grad_input->data[i] = grad_mul(grad_output->data[i], deriv_hp, faults);
    }
    
    return CT_OK;
}

/* ============================================================================
 * Linear Layer Backward
 * ============================================================================ */

ct_error_t ct_linear_grad_init(ct_linear_grad_t *grad,
                               fixed_hp_t *weight_buffer,
                               fixed_hp_t *bias_buffer,
                               ct_tensor_t *input_cache,
                               uint32_t input_size,
                               uint32_t output_size) {
    if (!grad || !weight_buffer || !bias_buffer) {
        return CT_ERR_NULL;
    }
    
    grad->input_size = input_size;
    grad->output_size = output_size;
    grad->input_cache = input_cache;
    
    /* Initialize weight gradient tensor [output_size, input_size] */
    ct_grad_tensor_init(&grad->grad_weights, weight_buffer,
                        output_size, input_size);
    
    /* Initialize bias gradient tensor [output_size] */
    ct_grad_tensor_init(&grad->grad_bias, bias_buffer,
                        output_size, 0);
    
    return CT_OK;
}

ct_error_t ct_linear_backward(const ct_linear_t *layer,
                              ct_linear_grad_t *grad,
                              const ct_grad_tensor_t *grad_output,
                              ct_grad_tensor_t *grad_input,
                              ct_fault_flags_t *faults) {
    if (!layer || !grad || !grad_output) {
        return CT_ERR_NULL;
    }
    
    uint32_t in_size = grad->input_size;
    uint32_t out_size = grad->output_size;
    
    /* Zero the weight and bias gradients first */
    ct_grad_tensor_zero(&grad->grad_weights);
    ct_grad_tensor_zero(&grad->grad_bias);
    
    /* For single sample (batch_size = 1):
     * 
     * grad_input[i] = Σ_j W[j,i] * grad_output[j]
     * grad_weights[j,i] = grad_output[j] * input[i]
     * grad_bias[j] = grad_output[j]
     */
    
    /* Compute grad_input (if requested) */
    if (grad_input) {
        /* grad_input = W^T @ grad_output */
        for (uint32_t i = 0; i < in_size; i++) {
            ct_comp_accum_t acc;
            ct_comp_init(&acc);
            
            for (uint32_t j = 0; j < out_size; j++) {
                /* W[j, i] in Q16.16 */
                fixed_t w = ct_tensor_get_2d(&layer->weights, j, i);
                /* grad_output[j] in Q8.24 */
                fixed_hp_t go = ct_grad_get_1d(grad_output, j);
                
                /* Product: Q8.24 * Q16.16 -> need to align */
                /* Multiply and accumulate */
                int64_t prod = (int64_t)go * (int64_t)w;
                ct_comp_add(&acc, prod, faults);
            }
            
            /* Result is in high-precision, convert */
            fixed_acc_t sum = ct_comp_get_sum(&acc);
            /* Shift down by Q16.16 frac bits to get Q8.24 result */
            int64_t result = sum >> FIXED_FRAC_BITS;
            ct_grad_set_1d(grad_input, i, dvm_clamp32(result, faults));
        }
    }
    
    /* Compute grad_weights and grad_bias */
    if (grad->input_cache) {
        /* grad_weights[j,i] = grad_output[j] * input[i] */
        for (uint32_t j = 0; j < out_size; j++) {
            fixed_hp_t go = ct_grad_get_1d(grad_output, j);
            
            /* Bias gradient is just the output gradient */
            ct_grad_set_1d(&grad->grad_bias, j, go);
            
            for (uint32_t i = 0; i < in_size; i++) {
                /* input[i] in Q16.16 */
                fixed_t inp = ct_tensor_get_1d(grad->input_cache, i);
                
                /* grad_output[j] * input[i] */
                fixed_hp_t gw = grad_mul_fixed(go, inp, faults);
                ct_grad_set_2d(&grad->grad_weights, j, i, gw);
            }
        }
    }
    
    return CT_OK;
}

/* ============================================================================
 * Gradient Processing
 * ============================================================================ */

uint32_t ct_grad_clip(ct_grad_tensor_t *grad,
                      fixed_hp_t min_val,
                      fixed_hp_t max_val,
                      ct_fault_flags_t *faults) {
    (void)faults;
    
    if (!grad || !grad->data) {
        return 0;
    }
    
    uint32_t clipped_count = 0;
    
    for (uint32_t i = 0; i < grad->total_size; i++) {
        fixed_hp_t val = grad->data[i];
        if (val < min_val) {
            grad->data[i] = min_val;
            clipped_count++;
        } else if (val > max_val) {
            grad->data[i] = max_val;
            clipped_count++;
        }
    }
    
    return clipped_count;
}

void ct_grad_scale(ct_grad_tensor_t *grad,
                   fixed_hp_t scale,
                   ct_fault_flags_t *faults) {
    if (!grad || !grad->data) {
        return;
    }
    
    for (uint32_t i = 0; i < grad->total_size; i++) {
        grad->data[i] = grad_mul(grad->data[i], scale, faults);
    }
}

ct_error_t ct_grad_norm(const ct_grad_tensor_t *grad,
                        fixed_hp_t *norm_out,
                        ct_fault_flags_t *faults) {
    if (!grad || !norm_out) {
        return CT_ERR_NULL;
    }
    
    ct_comp_accum_t acc;
    ct_comp_init(&acc);
    
    /* Sum of squares */
    /* Input: Q8.24 values, squared gives Q16.48 */
    for (uint32_t i = 0; i < grad->total_size; i++) {
        fixed_hp_t val = grad->data[i];
        int64_t sq = (int64_t)val * (int64_t)val;
        ct_comp_add(&acc, sq, faults);
    }
    
    fixed_acc_t sum_sq = ct_comp_get_sum(&acc);
    
    /* sum_sq is in Q16.48 (Q8.24 squared)
     * We want sqrt(sum_sq) in Q8.24
     * 
     * If sum_sq represents value V in Q16.48, then sum_sq = V * 2^48
     * sqrt(sum_sq) = sqrt(V) * 2^24 which is sqrt(V) in Q8.24
     * 
     * So we compute integer sqrt of sum_sq directly
     */
    
    if (sum_sq <= 0) {
        *norm_out = 0;
        return CT_OK;
    }
    
    /* Integer sqrt using Newton-Raphson on the raw sum_sq value */
    /* Result will be in Q8.24 automatically */
    uint64_t x = (uint64_t)sum_sq;
    uint64_t guess = x >> 1;
    if (guess == 0) guess = 1;
    
    /* More iterations for 64-bit values */
    for (int iter = 0; iter < 32; iter++) {
        if (guess == 0) break;
        uint64_t div = x / guess;
        uint64_t new_guess = (guess + div) >> 1;
        if (new_guess >= guess) break;  /* Converged */
        guess = new_guess;
    }
    
    *norm_out = dvm_clamp32((int64_t)guess, faults);
    
    return CT_OK;
}

/* ============================================================================
 * Gradient Health Monitoring
 * ============================================================================ */

void ct_grad_health_init(ct_grad_health_t *health) {
    if (!health) return;
    
    health->zero_grad_count = 0;
    health->total_grad_count = 0;
    health->min_nonzero_grad = INT32_MAX;
    health->max_grad = 0;
}

void ct_grad_health_update(ct_grad_health_t *health,
                           const ct_grad_tensor_t *grad) {
    if (!health || !grad || !grad->data) {
        return;
    }
    
    for (uint32_t i = 0; i < grad->total_size; i++) {
        fixed_hp_t val = grad->data[i];
        fixed_hp_t abs_val = (val < 0) ? -val : val;
        
        health->total_grad_count++;
        
        if (val == 0) {
            health->zero_grad_count++;
        } else {
            if (abs_val < health->min_nonzero_grad) {
                health->min_nonzero_grad = abs_val;
            }
        }
        
        if (abs_val > health->max_grad) {
            health->max_grad = abs_val;
        }
    }
}

bool ct_grad_health_is_vanishing(const ct_grad_health_t *health) {
    if (!health || health->total_grad_count == 0) {
        return false;
    }
    
    /* Check if more than 5% of gradients are zero */
    uint64_t threshold = (health->total_grad_count * CT_GRAD_FLOOR_THRESHOLD_PERCENT) / 100;
    return health->zero_grad_count > threshold;
}

fixed_t ct_grad_health_zero_ratio(const ct_grad_health_t *health) {
    if (!health || health->total_grad_count == 0) {
        return 0;
    }
    
    /* Return ratio as Q16.16 (0.0 to 1.0) */
    uint64_t ratio = (health->zero_grad_count * (uint64_t)FIXED_ONE) / health->total_grad_count;
    if (ratio > (uint64_t)FIXED_ONE) {
        ratio = FIXED_ONE;
    }
    return (fixed_t)ratio;
}

/* ============================================================================
 * Backward Context
 * ============================================================================ */

ct_backward_config_t ct_backward_config_default(void) {
    ct_backward_config_t config;
    config.grad_clip_min = -CT_GRAD_CLIP_DEFAULT;
    config.grad_clip_max = CT_GRAD_CLIP_DEFAULT;
    config.enable_grad_health = true;
    config.batch_size = 1;
    return config;
}

ct_error_t ct_backward_ctx_init(ct_backward_ctx_t *ctx,
                                const ct_backward_config_t *config,
                                ct_fault_flags_t *faults) {
    if (!ctx || !config) {
        return CT_ERR_NULL;
    }
    
    ctx->config = *config;
    ctx->faults = faults;
    ct_grad_health_init(&ctx->health);
    
    return CT_OK;
}
