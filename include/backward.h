/**
 * @file backward.h
 * @project Certifiable Training
 * @brief Deterministic backward pass (backpropagation) for neural network training.
 *
 * @details Implements gradient computation in Q8.24 high-precision format:
 *          - Loss gradients (MSE, Cross-Entropy)
 *          - Layer gradients via backpropagation
 *          - Activation derivatives (ReLU, Sigmoid, Tanh)
 *          - Gradient health monitoring for vanishing detection
 *
 * @traceability SRS-006-BACKWARD, CT-MATH-001 §7
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust. All rights reserved.
 * @license Licensed under the GPL-3.0 (Open Source) or Commercial License.
 *          For commercial licensing: william@fstopify.com
 */

#ifndef CERTIFIABLE_TRAINING_BACKWARD_H
#define CERTIFIABLE_TRAINING_BACKWARD_H

#include "ct_types.h"
#include "dvm.h"
#include "forward.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

/** Q8.24 fractional bits for high-precision gradients */
#define CT_GRAD_FRAC_BITS       24

/** Gradient scale factor (Q8.24 ONE) */
#define CT_GRAD_ONE             ((fixed_hp_t)(1 << CT_GRAD_FRAC_BITS))

/** Gradient half for rounding */
#define CT_GRAD_HALF            ((fixed_hp_t)(1 << (CT_GRAD_FRAC_BITS - 1)))

/** Vanishing gradient warning threshold (5% zeros) */
#define CT_GRAD_FLOOR_THRESHOLD_PERCENT  5

/** Gradient clipping default (in Q8.24, represents ~100.0) */
#define CT_GRAD_CLIP_DEFAULT    ((fixed_hp_t)(100 << CT_GRAD_FRAC_BITS))

/* ============================================================================
 * Gradient Tensor
 * ============================================================================ */

/**
 * @brief High-precision gradient tensor (Q8.24)
 * @ref CT-STRUCT-001 §5.2
 */
typedef struct {
    fixed_hp_t *data;               /**< High-precision gradient data */
    uint32_t dims[CT_MAX_DIMS];     /**< Dimension sizes */
    uint32_t strides[CT_MAX_DIMS];  /**< Element strides */
    uint32_t ndims;                 /**< Number of dimensions */
    uint32_t total_size;            /**< Total elements */
} ct_grad_tensor_t;

/**
 * @brief Gradient health monitor for vanishing detection
 * @ref CT-MATH-001 §7.4
 */
typedef struct {
    uint64_t zero_grad_count;       /**< Gradients that hit floor (became zero) */
    uint64_t total_grad_count;      /**< Total gradient updates */
    fixed_hp_t min_nonzero_grad;    /**< Smallest non-zero gradient seen */
    fixed_hp_t max_grad;            /**< Largest gradient magnitude seen */
} ct_grad_health_t;

/* ============================================================================
 * Linear Layer Gradients
 * ============================================================================ */

/**
 * @brief Gradient cache for linear layer backward pass
 */
typedef struct {
    ct_grad_tensor_t grad_weights;  /**< ∂L/∂W: [output_size, input_size] */
    ct_grad_tensor_t grad_bias;     /**< ∂L/∂b: [output_size] */
    ct_tensor_t *input_cache;       /**< Cached input from forward pass */
    uint32_t input_size;
    uint32_t output_size;
} ct_linear_grad_t;

/* ============================================================================
 * Backward Context
 * ============================================================================ */

/**
 * @brief Configuration for backward pass
 */
typedef struct {
    fixed_hp_t grad_clip_min;       /**< Minimum gradient (negative) */
    fixed_hp_t grad_clip_max;       /**< Maximum gradient (positive) */
    bool enable_grad_health;        /**< Track vanishing gradient stats */
    uint32_t batch_size;            /**< For gradient averaging */
} ct_backward_config_t;

/**
 * @brief Backward pass context
 */
typedef struct {
    ct_backward_config_t config;
    ct_grad_health_t health;
    ct_fault_flags_t *faults;       /**< Fault accumulator (shared) */
} ct_backward_ctx_t;

/* ============================================================================
 * Gradient Tensor Operations
 * ============================================================================ */

/**
 * @brief Initialize gradient tensor with pre-allocated buffer
 * @param grad Gradient tensor to initialize
 * @param buffer Pre-allocated Q8.24 data buffer
 * @param rows Number of rows (or total size for 1D)
 * @param cols Number of columns (0 for 1D)
 * @return CT_OK on success
 */
ct_error_t ct_grad_tensor_init(ct_grad_tensor_t *grad,
                               fixed_hp_t *buffer,
                               uint32_t rows,
                               uint32_t cols);

/**
 * @brief Zero all gradient values
 * @param grad Gradient tensor to clear
 */
void ct_grad_tensor_zero(ct_grad_tensor_t *grad);

/**
 * @brief Get gradient element (1D)
 */
static inline fixed_hp_t ct_grad_get_1d(const ct_grad_tensor_t *grad,
                                        uint32_t i) {
    return grad->data[i * grad->strides[0]];
}

/**
 * @brief Set gradient element (1D)
 */
static inline void ct_grad_set_1d(ct_grad_tensor_t *grad,
                                  uint32_t i,
                                  fixed_hp_t value) {
    grad->data[i * grad->strides[0]] = value;
}

/**
 * @brief Get gradient element (2D)
 */
static inline fixed_hp_t ct_grad_get_2d(const ct_grad_tensor_t *grad,
                                        uint32_t row,
                                        uint32_t col) {
    return grad->data[row * grad->strides[0] + col * grad->strides[1]];
}

/**
 * @brief Set gradient element (2D)
 */
static inline void ct_grad_set_2d(ct_grad_tensor_t *grad,
                                  uint32_t row,
                                  uint32_t col,
                                  fixed_hp_t value) {
    grad->data[row * grad->strides[0] + col * grad->strides[1]] = value;
}

/* ============================================================================
 * Format Conversion
 * ============================================================================ */

/**
 * @brief Convert Q16.16 to Q8.24 (widen precision)
 * @param value Q16.16 fixed-point value
 * @return Q8.24 high-precision value
 * 
 * @note This is a left shift by 8 bits (24-16=8)
 */
static inline fixed_hp_t ct_fixed_to_grad(fixed_t value) {
    return (fixed_hp_t)value << (CT_GRAD_FRAC_BITS - FIXED_FRAC_BITS);
}

/**
 * @brief Convert Q8.24 to Q16.16 (narrow precision with rounding)
 * @param grad Q8.24 gradient value
 * @param faults Fault flags for overflow detection
 * @return Q16.16 fixed-point value
 */
fixed_t ct_grad_to_fixed(fixed_hp_t grad, ct_fault_flags_t *faults);

/* ============================================================================
 * Loss Functions
 * ============================================================================ */

/**
 * @brief Compute MSE loss gradient: ∂L/∂ŷ = (2/N)(ŷ - y)
 * @param output Predicted output tensor (Q16.16)
 * @param target Target tensor (Q16.16)
 * @param grad_output Output gradient tensor (Q8.24)
 * @param faults Fault accumulator
 * @return CT_OK on success
 * 
 * @ref CT-MATH-001 §14.1
 */
ct_error_t ct_loss_mse_backward(const ct_tensor_t *output,
                                const ct_tensor_t *target,
                                ct_grad_tensor_t *grad_output,
                                ct_fault_flags_t *faults);

/**
 * @brief Compute MSE loss value: L = (1/N) Σ(ŷ - y)²
 * @param output Predicted output tensor (Q16.16)
 * @param target Target tensor (Q16.16)
 * @param loss_out Output loss value (Q16.16)
 * @param faults Fault accumulator
 * @return CT_OK on success
 */
ct_error_t ct_loss_mse_forward(const ct_tensor_t *output,
                               const ct_tensor_t *target,
                               fixed_t *loss_out,
                               ct_fault_flags_t *faults);

/* ============================================================================
 * Activation Derivatives
 * ============================================================================ */

/**
 * @brief ReLU backward: grad_input = grad_output if pre_act > 0, else 0
 * @param grad_output Upstream gradient (Q8.24)
 * @param pre_activation Pre-activation values from forward pass (Q16.16)
 * @param grad_input Output gradient (Q8.24)
 * @param faults Fault accumulator
 * @return CT_OK on success
 * 
 * @note ReLU derivative is 1 if x > 0, 0 otherwise
 */
ct_error_t ct_activation_relu_backward(const ct_grad_tensor_t *grad_output,
                                       const ct_tensor_t *pre_activation,
                                       ct_grad_tensor_t *grad_input,
                                       ct_fault_flags_t *faults);

/**
 * @brief Sigmoid backward: grad_input = grad_output * σ(x) * (1 - σ(x))
 * @param grad_output Upstream gradient (Q8.24)
 * @param activation Sigmoid output from forward pass (Q16.16)
 * @param grad_input Output gradient (Q8.24)
 * @param faults Fault accumulator
 * @return CT_OK on success
 * 
 * @ref CT-MATH-001 §12.5: σ'(x) = σ(x) · (1 - σ(x))
 */
ct_error_t ct_activation_sigmoid_backward(const ct_grad_tensor_t *grad_output,
                                          const ct_tensor_t *activation,
                                          ct_grad_tensor_t *grad_input,
                                          ct_fault_flags_t *faults);

/**
 * @brief Tanh backward: grad_input = grad_output * (1 - tanh²(x))
 * @param grad_output Upstream gradient (Q8.24)
 * @param activation Tanh output from forward pass (Q16.16)
 * @param grad_input Output gradient (Q8.24)
 * @param faults Fault accumulator
 * @return CT_OK on success
 * 
 * @note tanh'(x) = 1 - tanh²(x)
 */
ct_error_t ct_activation_tanh_backward(const ct_grad_tensor_t *grad_output,
                                       const ct_tensor_t *activation,
                                       ct_grad_tensor_t *grad_input,
                                       ct_fault_flags_t *faults);

/* ============================================================================
 * Linear Layer Backward
 * ============================================================================ */

/**
 * @brief Initialize linear layer gradient cache
 * @param grad Gradient cache to initialize
 * @param weight_buffer Buffer for weight gradients [output_size * input_size]
 * @param bias_buffer Buffer for bias gradients [output_size]
 * @param input_cache Pointer to cached input from forward pass
 * @param input_size Number of input features
 * @param output_size Number of output features
 * @return CT_OK on success
 */
ct_error_t ct_linear_grad_init(ct_linear_grad_t *grad,
                               fixed_hp_t *weight_buffer,
                               fixed_hp_t *bias_buffer,
                               ct_tensor_t *input_cache,
                               uint32_t input_size,
                               uint32_t output_size);

/**
 * @brief Linear layer backward pass
 * @param layer Linear layer (weights, bias)
 * @param grad Layer gradient cache
 * @param grad_output Upstream gradient (Q8.24) [batch_size, output_size]
 * @param grad_input Output gradient for previous layer (Q8.24) [batch_size, input_size]
 * @param faults Fault accumulator
 * @return CT_OK on success
 * 
 * @details Computes:
 *          - grad_input = grad_output @ W (for backprop to previous layer)
 *          - grad_weights = grad_output.T @ input (weight update)
 *          - grad_bias = sum(grad_output, axis=0) (bias update)
 * 
 * @ref CT-MATH-001 §7.2
 */
ct_error_t ct_linear_backward(const ct_linear_t *layer,
                              ct_linear_grad_t *grad,
                              const ct_grad_tensor_t *grad_output,
                              ct_grad_tensor_t *grad_input,
                              ct_fault_flags_t *faults);

/* ============================================================================
 * Gradient Processing
 * ============================================================================ */

/**
 * @brief Clip gradient to specified range
 * @param grad Gradient tensor to clip in-place
 * @param min_val Minimum allowed value
 * @param max_val Maximum allowed value
 * @param faults Fault accumulator (records if clipping occurred)
 * @return Number of values clipped
 */
uint32_t ct_grad_clip(ct_grad_tensor_t *grad,
                      fixed_hp_t min_val,
                      fixed_hp_t max_val,
                      ct_fault_flags_t *faults);

/**
 * @brief Scale gradient by constant factor
 * @param grad Gradient tensor to scale in-place
 * @param scale Scale factor (Q8.24)
 * @param faults Fault accumulator
 */
void ct_grad_scale(ct_grad_tensor_t *grad,
                   fixed_hp_t scale,
                   ct_fault_flags_t *faults);

/**
 * @brief Compute gradient L2 norm
 * @param grad Gradient tensor
 * @param norm_out Output norm value (Q8.24)
 * @param faults Fault accumulator
 * @return CT_OK on success
 */
ct_error_t ct_grad_norm(const ct_grad_tensor_t *grad,
                        fixed_hp_t *norm_out,
                        ct_fault_flags_t *faults);

/* ============================================================================
 * Gradient Health Monitoring
 * ============================================================================ */

/**
 * @brief Initialize gradient health monitor
 */
void ct_grad_health_init(ct_grad_health_t *health);

/**
 * @brief Update health statistics with gradient tensor
 * @param health Health monitor
 * @param grad Gradient tensor to analyze
 */
void ct_grad_health_update(ct_grad_health_t *health,
                           const ct_grad_tensor_t *grad);

/**
 * @brief Check if vanishing gradient threshold exceeded
 * @param health Health monitor
 * @return true if zero_ratio > threshold
 */
bool ct_grad_health_is_vanishing(const ct_grad_health_t *health);

/**
 * @brief Get zero gradient ratio (0.0 to 1.0 in Q16.16)
 * @param health Health monitor
 * @return Ratio of zero gradients
 */
fixed_t ct_grad_health_zero_ratio(const ct_grad_health_t *health);

/* ============================================================================
 * Backward Context
 * ============================================================================ */

/**
 * @brief Initialize backward pass context
 * @param ctx Context to initialize
 * @param config Configuration
 * @param faults Shared fault accumulator
 * @return CT_OK on success
 */
ct_error_t ct_backward_ctx_init(ct_backward_ctx_t *ctx,
                                const ct_backward_config_t *config,
                                ct_fault_flags_t *faults);

/**
 * @brief Get default backward configuration
 */
ct_backward_config_t ct_backward_config_default(void);

#ifdef __cplusplus
}
#endif

#endif /* CERTIFIABLE_TRAINING_BACKWARD_H */
