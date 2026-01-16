/**
 * @file optimizer.h
 * @project Certifiable Training
 * @brief Deterministic optimizers for neural network training.
 *
 * @details Implements fixed-point optimizers:
 *          - SGD (Stochastic Gradient Descent)
 *          - SGD with Momentum
 *          - Adam (Adaptive Moment Estimation)
 *          All using DVM primitives for bit-identical results.
 *
 * @traceability SRS-007-OPTIMIZER, CT-MATH-001 §10, §13
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust. All rights reserved.
 * @license Licensed under the GPL-3.0 (Open Source) or Commercial License.
 *          For commercial licensing: william@fstopify.com
 */

#ifndef CERTIFIABLE_TRAINING_OPTIMIZER_H
#define CERTIFIABLE_TRAINING_OPTIMIZER_H

#include "ct_types.h"
#include "dvm.h"
#include "forward.h"
#include "backward.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

/** Default learning rate: 0.01 in Q16.16 */
#define CT_OPT_DEFAULT_LR           ((fixed_t)(655))

/** Default momentum: 0.9 in Q16.16 */
#define CT_OPT_DEFAULT_MOMENTUM     ((fixed_t)(58982))

/** Default weight decay: 0.0001 in Q16.16 */
#define CT_OPT_DEFAULT_WEIGHT_DECAY ((fixed_t)(7))

/** Adam beta1: 0.9 in Q16.16 */
#define CT_OPT_ADAM_BETA1           ((fixed_t)(58982))

/** Adam beta2: 0.999 in Q16.16 */
#define CT_OPT_ADAM_BETA2           ((fixed_t)(65471))

/** Adam epsilon: 1e-8 in Q16.16 (minimum representable ~1.5e-5) */
#define CT_OPT_ADAM_EPSILON         ((fixed_t)(1))

/** Fixed sqrt iterations per CT-MATH-001 §13 */
#define CT_OPT_SQRT_ITERATIONS      8

/* ============================================================================
 * Optimizer Type Enum
 * ============================================================================ */

typedef enum {
    CT_OPT_SGD          = 0,
    CT_OPT_SGD_MOMENTUM = 1,
    CT_OPT_ADAM         = 2
} ct_optimizer_type_t;

/* ============================================================================
 * SGD Optimizer
 * ============================================================================ */

/**
 * @brief SGD configuration
 * @ref CT-STRUCT-001 §7.1
 */
typedef struct {
    fixed_t learning_rate;      /**< η: step size */
    fixed_t weight_decay;       /**< λ: L2 regularization */
} ct_sgd_config_t;

/**
 * @brief SGD optimizer state
 */
typedef struct {
    ct_sgd_config_t config;
    uint64_t step;              /**< Update count */
} ct_sgd_t;

/**
 * @brief Get default SGD configuration
 */
ct_sgd_config_t ct_sgd_config_default(void);

/**
 * @brief Initialize SGD optimizer
 * @param opt Optimizer to initialize
 * @param config Configuration (NULL for defaults)
 * @return CT_OK on success
 */
ct_error_t ct_sgd_init(ct_sgd_t *opt, const ct_sgd_config_t *config);

/**
 * @brief SGD update step: θ = θ - η * (g + λ * θ)
 * @param opt Optimizer state
 * @param params Parameter tensor to update (Q16.16)
 * @param grads Gradient tensor (Q8.24)
 * @param faults Fault accumulator
 * @return CT_OK on success
 *
 * @ref CT-MATH-001 §10.2
 */
ct_error_t ct_sgd_step(ct_sgd_t *opt,
                       ct_tensor_t *params,
                       const ct_grad_tensor_t *grads,
                       ct_fault_flags_t *faults);

/* ============================================================================
 * SGD with Momentum
 * ============================================================================ */

/**
 * @brief SGD with Momentum configuration
 * @ref CT-STRUCT-001 §7.2
 */
typedef struct {
    fixed_t learning_rate;      /**< η: step size */
    fixed_t momentum;           /**< β: momentum coefficient (typically 0.9) */
    fixed_t weight_decay;       /**< λ: L2 regularization */
} ct_sgd_momentum_config_t;

/**
 * @brief SGD with Momentum optimizer state
 */
typedef struct {
    ct_sgd_momentum_config_t config;
    ct_tensor_t velocity;       /**< v: velocity buffer (caller-provided) */
    uint32_t num_params;        /**< Number of parameters */
    uint64_t step;              /**< Update count */
    bool initialized;           /**< Velocity initialized flag */
} ct_sgd_momentum_t;

/**
 * @brief Get default SGD+Momentum configuration
 */
ct_sgd_momentum_config_t ct_sgd_momentum_config_default(void);

/**
 * @brief Initialize SGD with Momentum optimizer
 * @param opt Optimizer to initialize
 * @param config Configuration (NULL for defaults)
 * @param velocity_buffer Pre-allocated buffer for velocity [num_params]
 * @param num_params Number of parameters
 * @return CT_OK on success
 */
ct_error_t ct_sgd_momentum_init(ct_sgd_momentum_t *opt,
                                const ct_sgd_momentum_config_t *config,
                                fixed_t *velocity_buffer,
                                uint32_t num_params);

/**
 * @brief SGD+Momentum update step
 *        v = β * v + g
 *        θ = θ - η * (v + λ * θ)
 * @param opt Optimizer state
 * @param params Parameter tensor to update (Q16.16)
 * @param grads Gradient tensor (Q8.24)
 * @param faults Fault accumulator
 * @return CT_OK on success
 *
 * @ref CT-MATH-001 §10.3
 */
ct_error_t ct_sgd_momentum_step(ct_sgd_momentum_t *opt,
                                ct_tensor_t *params,
                                const ct_grad_tensor_t *grads,
                                ct_fault_flags_t *faults);

/* ============================================================================
 * Adam Optimizer
 * ============================================================================ */

/**
 * @brief Adam configuration
 * @ref CT-STRUCT-001 §7.3
 */
typedef struct {
    fixed_t learning_rate;      /**< η: step size */
    fixed_t beta1;              /**< β₁: first moment decay (0.9) */
    fixed_t beta2;              /**< β₂: second moment decay (0.999) */
    fixed_t epsilon;            /**< ε: numerical stability */
    fixed_t weight_decay;       /**< λ: L2 regularization (AdamW style) */
} ct_adam_config_t;

/**
 * @brief Adam optimizer state
 */
typedef struct {
    ct_adam_config_t config;
    ct_tensor_t m;              /**< First moment estimate (caller-provided) */
    ct_tensor_t v;              /**< Second moment estimate (caller-provided) */
    fixed_t beta1_power;        /**< β₁^t for bias correction */
    fixed_t beta2_power;        /**< β₂^t for bias correction */
    uint32_t num_params;        /**< Number of parameters */
    uint64_t step;              /**< Update count (t) */
    bool initialized;           /**< Moments initialized flag */
} ct_adam_t;

/**
 * @brief Get default Adam configuration
 */
ct_adam_config_t ct_adam_config_default(void);

/**
 * @brief Initialize Adam optimizer
 * @param opt Optimizer to initialize
 * @param config Configuration (NULL for defaults)
 * @param m_buffer Pre-allocated buffer for first moment [num_params]
 * @param v_buffer Pre-allocated buffer for second moment [num_params]
 * @param num_params Number of parameters
 * @return CT_OK on success
 */
ct_error_t ct_adam_init(ct_adam_t *opt,
                        const ct_adam_config_t *config,
                        fixed_t *m_buffer,
                        fixed_t *v_buffer,
                        uint32_t num_params);

/**
 * @brief Adam update step
 *        m = β₁ * m + (1-β₁) * g
 *        v = β₂ * v + (1-β₂) * g²
 *        m̂ = m / (1 - β₁^t)
 *        v̂ = v / (1 - β₂^t)
 *        θ = θ - η * m̂ / (√v̂ + ε)
 * @param opt Optimizer state
 * @param params Parameter tensor to update (Q16.16)
 * @param grads Gradient tensor (Q8.24)
 * @param faults Fault accumulator
 * @return CT_OK on success
 *
 * @ref CT-MATH-001 §10.4, §13
 */
ct_error_t ct_adam_step(ct_adam_t *opt,
                        ct_tensor_t *params,
                        const ct_grad_tensor_t *grads,
                        ct_fault_flags_t *faults);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Fixed-point square root using Newton-Raphson
 * @param x Input value (Q16.16, must be non-negative)
 * @param faults Fault accumulator
 * @return sqrt(x) in Q16.16
 *
 * @details Uses fixed 8 iterations per CT-MATH-001 §13.
 *          No data-dependent branching for determinism.
 *
 * @ref CT-MATH-001 §13.1
 */
fixed_t ct_opt_sqrt(fixed_t x, ct_fault_flags_t *faults);

/**
 * @brief Reset optimizer state (for retraining)
 */
void ct_sgd_reset(ct_sgd_t *opt);
void ct_sgd_momentum_reset(ct_sgd_momentum_t *opt);
void ct_adam_reset(ct_adam_t *opt);

#ifdef __cplusplus
}
#endif

#endif /* CERTIFIABLE_TRAINING_OPTIMIZER_H */
