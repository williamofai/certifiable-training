/**
 * @file optimizer.c
 * @project Certifiable Training
 * @brief Deterministic optimizer implementations.
 *
 * @details All updates use DVM primitives for bit-identical results.
 *
 * @traceability SRS-007-OPTIMIZER, CT-MATH-001 §10, §13
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust. All rights reserved.
 * @license Licensed under the GPL-3.0 (Open Source) or Commercial License.
 *          For commercial licensing: william@fstopify.com
 */

#include "optimizer.h"
#include <string.h>

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

/**
 * @brief Convert Q8.24 gradient to Q16.16 for weight update
 */
static fixed_t grad_to_param(fixed_hp_t grad, ct_fault_flags_t *faults) {
    /* Shift right by 8 bits (24-16) with rounding */
    int32_t shift = CT_GRAD_FRAC_BITS - FIXED_FRAC_BITS;
    int32_t half = 1 << (shift - 1);
    int64_t rounded = (int64_t)grad + half;
    int64_t shifted = rounded >> shift;
    return dvm_clamp32(shifted, faults);
}

/* ============================================================================
 * Fixed-Point Square Root
 * ============================================================================ */

fixed_t ct_opt_sqrt(fixed_t x, ct_fault_flags_t *faults) {
    (void)faults;  /* sqrt doesn't overflow for valid inputs */
    
    if (x <= 0) {
        return 0;
    }
    
    /* For Q16.16 fixed-point sqrt:
     * Input x represents value V = x / 2^16
     * We want output representing sqrt(V) = sqrt(x / 2^16)
     * In Q16.16: result = sqrt(V) * 2^16 = sqrt(x) * 2^8
     * 
     * Better approach: work with scaled value
     * Let y = x * 2^16 (scale up to get more precision)
     * sqrt(y) = sqrt(x) * 2^8 = result in Q16.16
     */
    
    /* Scale up by 16 bits for precision */
    uint64_t scaled = (uint64_t)x << 16;
    
    /* Newton-Raphson on scaled value */
    uint64_t guess = scaled;
    
    /* Better initial guess using bit position */
    int leading = 0;
    uint64_t temp = scaled;
    while (temp > 1) {
        temp >>= 1;
        leading++;
    }
    guess = (uint64_t)1 << ((leading + 1) / 2);
    
    /* Fixed 8 iterations per CT-MATH-001 §13 */
    for (int i = 0; i < CT_OPT_SQRT_ITERATIONS; i++) {
        if (guess == 0) break;
        uint64_t div = scaled / guess;
        uint64_t new_guess = (guess + div) >> 1;
        if (new_guess >= guess) break;  /* Converged */
        guess = new_guess;
    }
    
    /* Result is sqrt(x * 2^16) = sqrt(x) * 2^8, which is Q16.16 */
    if (guess > (uint64_t)INT32_MAX) {
        return INT32_MAX;
    }
    return (fixed_t)guess;
}

/* ============================================================================
 * SGD Optimizer
 * ============================================================================ */

ct_sgd_config_t ct_sgd_config_default(void) {
    ct_sgd_config_t config;
    config.learning_rate = CT_OPT_DEFAULT_LR;
    config.weight_decay = 0;
    return config;
}

ct_error_t ct_sgd_init(ct_sgd_t *opt, const ct_sgd_config_t *config) {
    if (!opt) {
        return CT_ERR_NULL;
    }
    
    if (config) {
        opt->config = *config;
    } else {
        opt->config = ct_sgd_config_default();
    }
    
    opt->step = 0;
    return CT_OK;
}

ct_error_t ct_sgd_step(ct_sgd_t *opt,
                       ct_tensor_t *params,
                       const ct_grad_tensor_t *grads,
                       ct_fault_flags_t *faults) {
    if (!opt || !params || !grads) {
        return CT_ERR_NULL;
    }
    if (params->total_size != grads->total_size) {
        return CT_ERR_DIMENSION;
    }
    
    fixed_t lr = opt->config.learning_rate;
    fixed_t wd = opt->config.weight_decay;
    uint32_t n = params->total_size;
    
    for (uint32_t i = 0; i < n; i++) {
        fixed_t theta = params->data[i];
        fixed_hp_t g_hp = grads->data[i];
        
        /* Convert gradient to Q16.16 */
        fixed_t g = grad_to_param(g_hp, faults);
        
        /* Apply weight decay: g = g + λ * θ */
        if (wd != 0) {
            fixed_t decay = dvm_mul(wd, theta, faults);
            g = dvm_add(g, decay, faults);
        }
        
        /* Update: θ = θ - η * g */
        fixed_t update = dvm_mul(lr, g, faults);
        params->data[i] = dvm_sub(theta, update, faults);
    }
    
    opt->step++;
    return CT_OK;
}

void ct_sgd_reset(ct_sgd_t *opt) {
    if (opt) {
        opt->step = 0;
    }
}

/* ============================================================================
 * SGD with Momentum
 * ============================================================================ */

ct_sgd_momentum_config_t ct_sgd_momentum_config_default(void) {
    ct_sgd_momentum_config_t config;
    config.learning_rate = CT_OPT_DEFAULT_LR;
    config.momentum = CT_OPT_DEFAULT_MOMENTUM;
    config.weight_decay = 0;
    return config;
}

ct_error_t ct_sgd_momentum_init(ct_sgd_momentum_t *opt,
                                const ct_sgd_momentum_config_t *config,
                                fixed_t *velocity_buffer,
                                uint32_t num_params) {
    if (!opt || !velocity_buffer || num_params == 0) {
        return CT_ERR_NULL;
    }
    
    if (config) {
        opt->config = *config;
    } else {
        opt->config = ct_sgd_momentum_config_default();
    }
    
    /* Initialize velocity tensor */
    ct_tensor_init_1d(&opt->velocity, velocity_buffer, num_params);
    
    /* Zero velocity */
    memset(velocity_buffer, 0, num_params * sizeof(fixed_t));
    
    opt->num_params = num_params;
    opt->step = 0;
    opt->initialized = true;
    
    return CT_OK;
}

ct_error_t ct_sgd_momentum_step(ct_sgd_momentum_t *opt,
                                ct_tensor_t *params,
                                const ct_grad_tensor_t *grads,
                                ct_fault_flags_t *faults) {
    if (!opt || !params || !grads || !opt->initialized) {
        return CT_ERR_NULL;
    }
    if (params->total_size != grads->total_size ||
        params->total_size != opt->num_params) {
        return CT_ERR_DIMENSION;
    }
    
    fixed_t lr = opt->config.learning_rate;
    fixed_t beta = opt->config.momentum;
    fixed_t wd = opt->config.weight_decay;
    uint32_t n = params->total_size;
    
    for (uint32_t i = 0; i < n; i++) {
        fixed_t theta = params->data[i];
        fixed_t v = opt->velocity.data[i];
        fixed_hp_t g_hp = grads->data[i];
        
        /* Convert gradient to Q16.16 */
        fixed_t g = grad_to_param(g_hp, faults);
        
        /* Update velocity: v = β * v + g */
        fixed_t v_scaled = dvm_mul(beta, v, faults);
        v = dvm_add(v_scaled, g, faults);
        opt->velocity.data[i] = v;
        
        /* Apply weight decay to effective gradient */
        fixed_t effective_g = v;
        if (wd != 0) {
            fixed_t decay = dvm_mul(wd, theta, faults);
            effective_g = dvm_add(v, decay, faults);
        }
        
        /* Update: θ = θ - η * effective_g */
        fixed_t update = dvm_mul(lr, effective_g, faults);
        params->data[i] = dvm_sub(theta, update, faults);
    }
    
    opt->step++;
    return CT_OK;
}

void ct_sgd_momentum_reset(ct_sgd_momentum_t *opt) {
    if (opt && opt->initialized) {
        memset(opt->velocity.data, 0, opt->num_params * sizeof(fixed_t));
        opt->step = 0;
    }
}

/* ============================================================================
 * Adam Optimizer
 * ============================================================================ */

ct_adam_config_t ct_adam_config_default(void) {
    ct_adam_config_t config;
    config.learning_rate = CT_OPT_DEFAULT_LR;
    config.beta1 = CT_OPT_ADAM_BETA1;
    config.beta2 = CT_OPT_ADAM_BETA2;
    config.epsilon = CT_OPT_ADAM_EPSILON;
    config.weight_decay = 0;
    return config;
}

ct_error_t ct_adam_init(ct_adam_t *opt,
                        const ct_adam_config_t *config,
                        fixed_t *m_buffer,
                        fixed_t *v_buffer,
                        uint32_t num_params) {
    if (!opt || !m_buffer || !v_buffer || num_params == 0) {
        return CT_ERR_NULL;
    }
    
    if (config) {
        opt->config = *config;
    } else {
        opt->config = ct_adam_config_default();
    }
    
    /* Initialize moment tensors */
    ct_tensor_init_1d(&opt->m, m_buffer, num_params);
    ct_tensor_init_1d(&opt->v, v_buffer, num_params);
    
    /* Zero moments */
    memset(m_buffer, 0, num_params * sizeof(fixed_t));
    memset(v_buffer, 0, num_params * sizeof(fixed_t));
    
    /* Initialize bias correction terms to 1.0 */
    opt->beta1_power = FIXED_ONE;
    opt->beta2_power = FIXED_ONE;
    
    opt->num_params = num_params;
    opt->step = 0;
    opt->initialized = true;
    
    return CT_OK;
}

ct_error_t ct_adam_step(ct_adam_t *opt,
                        ct_tensor_t *params,
                        const ct_grad_tensor_t *grads,
                        ct_fault_flags_t *faults) {
    if (!opt || !params || !grads || !opt->initialized) {
        return CT_ERR_NULL;
    }
    if (params->total_size != grads->total_size ||
        params->total_size != opt->num_params) {
        return CT_ERR_DIMENSION;
    }
    
    fixed_t lr = opt->config.learning_rate;
    fixed_t beta1 = opt->config.beta1;
    fixed_t beta2 = opt->config.beta2;
    fixed_t eps = opt->config.epsilon;
    fixed_t wd = opt->config.weight_decay;
    uint32_t n = params->total_size;
    
    /* Update bias correction: β^t = β^(t-1) * β */
    opt->beta1_power = dvm_mul(opt->beta1_power, beta1, faults);
    opt->beta2_power = dvm_mul(opt->beta2_power, beta2, faults);
    
    /* Compute 1 - β^t for bias correction */
    fixed_t one_minus_beta1_t = dvm_sub(FIXED_ONE, opt->beta1_power, faults);
    fixed_t one_minus_beta2_t = dvm_sub(FIXED_ONE, opt->beta2_power, faults);
    
    /* Precompute (1 - β) for moment updates */
    fixed_t one_minus_beta1 = dvm_sub(FIXED_ONE, beta1, faults);
    fixed_t one_minus_beta2 = dvm_sub(FIXED_ONE, beta2, faults);
    
    for (uint32_t i = 0; i < n; i++) {
        fixed_t theta = params->data[i];
        fixed_t m_i = opt->m.data[i];
        fixed_t v_i = opt->v.data[i];
        fixed_hp_t g_hp = grads->data[i];
        
        /* Convert gradient to Q16.16 */
        fixed_t g = grad_to_param(g_hp, faults);
        
        /* Update first moment: m = β₁ * m + (1-β₁) * g */
        fixed_t m_scaled = dvm_mul(beta1, m_i, faults);
        fixed_t g_scaled = dvm_mul(one_minus_beta1, g, faults);
        m_i = dvm_add(m_scaled, g_scaled, faults);
        opt->m.data[i] = m_i;
        
        /* Update second moment: v = β₂ * v + (1-β₂) * g² */
        fixed_t v_scaled = dvm_mul(beta2, v_i, faults);
        fixed_t g_sq = dvm_mul(g, g, faults);
        fixed_t g_sq_scaled = dvm_mul(one_minus_beta2, g_sq, faults);
        v_i = dvm_add(v_scaled, g_sq_scaled, faults);
        opt->v.data[i] = v_i;
        
        /* Bias-corrected estimates */
        /* m̂ = m / (1 - β₁^t) */
        fixed_t m_hat;
        if (one_minus_beta1_t > 0) {
            m_hat = dvm_div_q(m_i, one_minus_beta1_t, FIXED_FRAC_BITS, faults);
        } else {
            m_hat = m_i;
        }
        
        /* v̂ = v / (1 - β₂^t) */
        fixed_t v_hat;
        if (one_minus_beta2_t > 0) {
            v_hat = dvm_div_q(v_i, one_minus_beta2_t, FIXED_FRAC_BITS, faults);
        } else {
            v_hat = v_i;
        }
        
        /* Compute update: η * m̂ / (√v̂ + ε) */
        fixed_t sqrt_v = ct_opt_sqrt(v_hat, faults);
        fixed_t denom = dvm_add(sqrt_v, eps, faults);
        
        fixed_t update;
        if (denom > 0) {
            fixed_t ratio = dvm_div_q(m_hat, denom, FIXED_FRAC_BITS, faults);
            update = dvm_mul(lr, ratio, faults);
        } else {
            update = 0;
        }
        
        /* Apply weight decay (AdamW style: decay applied to params directly) */
        if (wd != 0) {
            fixed_t decay = dvm_mul(dvm_mul(lr, wd, faults), theta, faults);
            theta = dvm_sub(theta, decay, faults);
        }
        
        /* Final update: θ = θ - update */
        params->data[i] = dvm_sub(theta, update, faults);
    }
    
    opt->step++;
    return CT_OK;
}

void ct_adam_reset(ct_adam_t *opt) {
    if (opt && opt->initialized) {
        memset(opt->m.data, 0, opt->num_params * sizeof(fixed_t));
        memset(opt->v.data, 0, opt->num_params * sizeof(fixed_t));
        opt->beta1_power = FIXED_ONE;
        opt->beta2_power = FIXED_ONE;
        opt->step = 0;
    }
}
