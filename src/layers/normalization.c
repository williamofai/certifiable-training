/**
 * @file normalization.c
 * @project Certifiable Training
 * @brief Deterministic batch and layer normalization
 *
 * @details Implements normalization layers for neural networks:
 *          - Batch Normalization (training and inference modes)
 *          - Layer Normalization
 *
 *          All operations use fixed-point arithmetic with DVM primitives.
 *
 *          BatchNorm formula (inference):
 *            y = γ * (x - μ) / √(σ² + ε) + β
 *
 *          For determinism, we use:
 *            - Pre-computed inverse sqrt as multiplication
 *            - Running mean/variance updated with exponential moving average
 *
 * @traceability CT-MATH-001 §7.4
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304, DO-178C
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 * @license GPL-3.0 or Commercial License (william@fstopify.com)
 */

#include "ct_types.h"
#include "forward.h"
#include "dvm.h"
#include "compensated.h"
#include "optimizer.h"  /* For ct_opt_sqrt */
#include <string.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

/** Default epsilon for numerical stability (1e-5 in Q16.16 ≈ 1) */
#define CT_NORM_EPSILON_DEFAULT     ((fixed_t)1)

/** Default momentum for running stats (0.1 in Q16.16) */
#define CT_NORM_MOMENTUM_DEFAULT    ((fixed_t)6554)

/* ============================================================================
 * Batch Normalization Structures
 * ============================================================================ */

/**
 * @brief Batch normalization configuration
 */
typedef struct {
    uint32_t num_features;      /**< Number of features (channels) */
    fixed_t epsilon;            /**< Small constant for stability */
    fixed_t momentum;           /**< Momentum for running stats (Q16.16) */
    bool track_running_stats;   /**< Maintain running mean/variance */
} ct_batchnorm_config_t;

/**
 * @brief Batch normalization layer
 */
typedef struct {
    ct_batchnorm_config_t config;
    fixed_t *gamma;             /**< Scale parameter γ [num_features] */
    fixed_t *beta;              /**< Shift parameter β [num_features] */
    fixed_t *running_mean;      /**< Running mean [num_features] */
    fixed_t *running_var;       /**< Running variance [num_features] */
    fixed_t *inv_std_cache;     /**< Cached 1/√(var + ε) for backward */
    fixed_t *mean_cache;        /**< Cached batch mean for backward */
    uint64_t num_batches;       /**< Number of batches seen */
    bool training;              /**< Training mode flag */
} ct_batchnorm_t;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Get default batch normalization configuration
 */
ct_batchnorm_config_t ct_batchnorm_config_default(uint32_t num_features)
{
    ct_batchnorm_config_t cfg = {
        .num_features = num_features,
        .epsilon = CT_NORM_EPSILON_DEFAULT,
        .momentum = CT_NORM_MOMENTUM_DEFAULT,
        .affine = true,
        .track_running_stats = true
    };
    return cfg;
}

/* ============================================================================
 * Initialization
 * ============================================================================ */

/**
 * @brief Initialize batch normalization layer
 *
 * @param bn Layer to initialize
 * @param cfg Configuration
 * @param gamma_buf Buffer for scale parameters (NULL if !affine)
 * @param beta_buf Buffer for shift parameters (NULL if !affine)
 * @param running_mean_buf Buffer for running mean
 * @param running_var_buf Buffer for running variance
 * @param inv_std_buf Buffer for inverse std cache (for training backward)
 * @param mean_buf Buffer for mean cache (for training backward)
 * @return CT_OK on success
 */
ct_error_t ct_batchnorm_init(ct_batchnorm_t *bn,
                             const ct_batchnorm_config_t *cfg,
                             fixed_t *gamma_buf,
                             fixed_t *beta_buf,
                             fixed_t *running_mean_buf,
                             fixed_t *running_var_buf,
                             fixed_t *inv_std_buf,
                             fixed_t *mean_buf)
{
    if (bn == NULL || cfg == NULL) {
        return CT_ERR_NULL;
    }

    if (cfg->num_features == 0) {
        return CT_ERR_CONFIG;
    }

    bn->config = *cfg;
    bn->gamma = gamma_buf;
    bn->beta = beta_buf;
    bn->running_mean = running_mean_buf;
    bn->running_var = running_var_buf;
    bn->inv_std_cache = inv_std_buf;
    bn->mean_cache = mean_buf;
    bn->num_batches = 0;
    bn->training = true;

    /* Initialize gamma to 1, beta to 0 */
    if (bn->gamma != NULL) {
        for (uint32_t i = 0; i < cfg->num_features; i++) {
            bn->gamma[i] = FIXED_ONE;
        }
    }
    if (bn->beta != NULL) {
        memset(bn->beta, 0, cfg->num_features * sizeof(fixed_t));
    }

    /* Initialize running stats: mean=0, var=1 */
    if (bn->running_mean != NULL) {
        memset(bn->running_mean, 0, cfg->num_features * sizeof(fixed_t));
    }
    if (bn->running_var != NULL) {
        for (uint32_t i = 0; i < cfg->num_features; i++) {
            bn->running_var[i] = FIXED_ONE;
        }
    }

    return CT_OK;
}

/**
 * @brief Set training mode
 *
 * @param bn Batch normalization layer
 * @param training true for training, false for inference
 */
void ct_batchnorm_train(ct_batchnorm_t *bn, bool training)
{
    if (bn != NULL) {
        bn->training = training;
    }
}

/* ============================================================================
 * Forward Pass
 * ============================================================================ */

/**
 * @brief Batch normalization forward pass
 *
 * @param bn Batch normalization layer
 * @param input Input tensor: [batch_size, num_features]
 * @param output Output tensor: [batch_size, num_features]
 * @param batch_size Number of samples in batch
 * @param faults Fault accumulator
 * @return CT_OK on success
 *
 * @details In training mode: computes batch statistics and updates running stats
 *          In inference mode: uses running statistics
 */
ct_error_t ct_batchnorm_forward(ct_batchnorm_t *bn,
                                const fixed_t *input,
                                fixed_t *output,
                                uint32_t batch_size,
                                ct_fault_flags_t *faults)
{
    if (bn == NULL || input == NULL || output == NULL) {
        return CT_ERR_NULL;
    }

    const uint32_t nf = bn->config.num_features;

    if (bn->training) {
        /* Training mode: compute batch statistics */
        for (uint32_t f = 0; f < nf; f++) {
            /* Compute mean */
            ct_comp_accum_t sum_acc;
            ct_comp_init(&sum_acc);

            for (uint32_t b = 0; b < batch_size; b++) {
                ct_comp_add(&sum_acc, (int64_t)input[b * nf + f] << FIXED_FRAC_BITS, faults);
            }
            int64_t sum = ct_comp_finalize(&sum_acc, faults);
            fixed_t mean = dvm_round_shift_rne(sum / (int64_t)batch_size, FIXED_FRAC_BITS, faults);

            /* Cache mean for backward */
            if (bn->mean_cache != NULL) {
                bn->mean_cache[f] = mean;
            }

            /* Compute variance: E[(x - μ)²] */
            ct_comp_accum_t var_acc;
            ct_comp_init(&var_acc);

            for (uint32_t b = 0; b < batch_size; b++) {
                fixed_t centered = dvm_sub(input[b * nf + f], mean, faults);
                int64_t sq = (int64_t)centered * (int64_t)centered;
                ct_comp_add(&var_acc, sq, faults);
            }
            int64_t var_sum = ct_comp_finalize(&var_acc, faults);
            fixed_t variance = dvm_round_shift_rne(var_sum / (int64_t)batch_size, FIXED_FRAC_BITS, faults);

            /* Compute inverse std: 1 / √(var + ε) */
            fixed_t var_plus_eps = dvm_add(variance, bn->config.epsilon, faults);
            fixed_t std = ct_opt_sqrt(var_plus_eps, faults);
            fixed_t inv_std = (std > 0) ? dvm_div_q(FIXED_ONE, std, FIXED_FRAC_BITS, faults) : FIXED_ONE;

            /* Cache inverse std for backward */
            if (bn->inv_std_cache != NULL) {
                bn->inv_std_cache[f] = inv_std;
            }

            /* Normalize and apply affine transformation */
            fixed_t gamma = (bn->gamma != NULL) ? bn->gamma[f] : FIXED_ONE;
            fixed_t beta = (bn->beta != NULL) ? bn->beta[f] : 0;

            for (uint32_t b = 0; b < batch_size; b++) {
                fixed_t x = input[b * nf + f];
                fixed_t centered = dvm_sub(x, mean, faults);
                int64_t normalized = (int64_t)centered * (int64_t)inv_std;
                fixed_t norm = dvm_round_shift_rne(normalized, FIXED_FRAC_BITS, faults);

                /* y = γ * norm + β */
                int64_t scaled = (int64_t)gamma * (int64_t)norm;
                fixed_t y = dvm_add(dvm_round_shift_rne(scaled, FIXED_FRAC_BITS, faults), beta, faults);
                output[b * nf + f] = y;
            }

            /* Update running statistics */
            if (bn->config.track_running_stats) {
                /* running_mean = (1 - momentum) * running_mean + momentum * mean */
                fixed_t one_minus_mom = dvm_sub(FIXED_ONE, bn->config.momentum, faults);
                int64_t rm1 = (int64_t)one_minus_mom * (int64_t)bn->running_mean[f];
                int64_t rm2 = (int64_t)bn->config.momentum * (int64_t)mean;
                bn->running_mean[f] = dvm_round_shift_rne(rm1 + rm2, FIXED_FRAC_BITS, faults);

                /* running_var = (1 - momentum) * running_var + momentum * variance */
                int64_t rv1 = (int64_t)one_minus_mom * (int64_t)bn->running_var[f];
                int64_t rv2 = (int64_t)bn->config.momentum * (int64_t)variance;
                bn->running_var[f] = dvm_round_shift_rne(rv1 + rv2, FIXED_FRAC_BITS, faults);
            }
        }

        bn->num_batches++;

    } else {
        /* Inference mode: use running statistics */
        for (uint32_t f = 0; f < nf; f++) {
            fixed_t mean = (bn->running_mean != NULL) ? bn->running_mean[f] : 0;
            fixed_t variance = (bn->running_var != NULL) ? bn->running_var[f] : FIXED_ONE;

            fixed_t var_plus_eps = dvm_add(variance, bn->config.epsilon, faults);
            fixed_t std = ct_opt_sqrt(var_plus_eps, faults);
            fixed_t inv_std = (std > 0) ? dvm_div_q(FIXED_ONE, std, FIXED_FRAC_BITS, faults) : FIXED_ONE;

            fixed_t gamma = (bn->gamma != NULL) ? bn->gamma[f] : FIXED_ONE;
            fixed_t beta = (bn->beta != NULL) ? bn->beta[f] : 0;

            for (uint32_t b = 0; b < batch_size; b++) {
                fixed_t x = input[b * nf + f];
                fixed_t centered = dvm_sub(x, mean, faults);
                int64_t normalized = (int64_t)centered * (int64_t)inv_std;
                fixed_t norm = dvm_round_shift_rne(normalized, FIXED_FRAC_BITS, faults);

                int64_t scaled = (int64_t)gamma * (int64_t)norm;
                fixed_t y = dvm_add(dvm_round_shift_rne(scaled, FIXED_FRAC_BITS, faults), beta, faults);
                output[b * nf + f] = y;
            }
        }
    }

    return CT_OK;
}

/* ============================================================================
 * Layer Normalization
 * ============================================================================ */

/**
 * @brief Layer normalization configuration
 */
typedef struct {
    uint32_t normalized_shape;  /**< Size of normalization dimension */
    fixed_t epsilon;            /**< Numerical stability constant */
} ct_layernorm_config_t;

/**
 * @brief Layer normalization layer
 */
typedef struct {
    ct_layernorm_config_t config;
    fixed_t *gamma;             /**< Scale parameter [normalized_shape] */
    fixed_t *beta;              /**< Shift parameter [normalized_shape] */
} ct_layernorm_t;

/**
 * @brief Get default layer normalization configuration
 */
ct_layernorm_config_t ct_layernorm_config_default(uint32_t normalized_shape)
{
    ct_layernorm_config_t cfg = {
        .normalized_shape = normalized_shape,
        .epsilon = CT_NORM_EPSILON_DEFAULT,
        .affine = true
    };
    return cfg;
}

/**
 * @brief Initialize layer normalization
 *
 * @param ln Layer to initialize
 * @param cfg Configuration
 * @param gamma_buf Buffer for scale parameters
 * @param beta_buf Buffer for shift parameters
 * @return CT_OK on success
 */
ct_error_t ct_layernorm_init(ct_layernorm_t *ln,
                             const ct_layernorm_config_t *cfg,
                             fixed_t *gamma_buf,
                             fixed_t *beta_buf)
{
    if (ln == NULL || cfg == NULL) {
        return CT_ERR_NULL;
    }

    if (cfg->normalized_shape == 0) {
        return CT_ERR_CONFIG;
    }

    ln->config = *cfg;
    ln->gamma = gamma_buf;
    ln->beta = beta_buf;

    /* Initialize gamma to 1, beta to 0 */
    if (ln->gamma != NULL) {
        for (uint32_t i = 0; i < cfg->normalized_shape; i++) {
            ln->gamma[i] = FIXED_ONE;
        }
    }
    if (ln->beta != NULL) {
        memset(ln->beta, 0, cfg->normalized_shape * sizeof(fixed_t));
    }

    return CT_OK;
}

/**
 * @brief Layer normalization forward pass
 *
 * @param ln Layer normalization layer
 * @param input Input tensor: [batch_size, normalized_shape]
 * @param output Output tensor: [batch_size, normalized_shape]
 * @param batch_size Number of samples
 * @param faults Fault accumulator
 * @return CT_OK on success
 *
 * @details Normalizes across the normalized_shape dimension for each sample.
 *          y = γ * (x - E[x]) / √(Var[x] + ε) + β
 */
ct_error_t ct_layernorm_forward(const ct_layernorm_t *ln,
                                const fixed_t *input,
                                fixed_t *output,
                                uint32_t batch_size,
                                ct_fault_flags_t *faults)
{
    if (ln == NULL || input == NULL || output == NULL) {
        return CT_ERR_NULL;
    }

    const uint32_t ns = ln->config.normalized_shape;

    /* Process each sample independently */
    for (uint32_t b = 0; b < batch_size; b++) {
        const fixed_t *x = &input[b * ns];
        fixed_t *y = &output[b * ns];

        /* Compute mean */
        ct_comp_accum_t sum_acc;
        ct_comp_init(&sum_acc);

        for (uint32_t i = 0; i < ns; i++) {
            ct_comp_add(&sum_acc, (int64_t)x[i] << FIXED_FRAC_BITS, faults);
        }
        int64_t sum = ct_comp_finalize(&sum_acc, faults);
        fixed_t mean = dvm_round_shift_rne(sum / (int64_t)ns, FIXED_FRAC_BITS, faults);

        /* Compute variance */
        ct_comp_accum_t var_acc;
        ct_comp_init(&var_acc);

        for (uint32_t i = 0; i < ns; i++) {
            fixed_t centered = dvm_sub(x[i], mean, faults);
            int64_t sq = (int64_t)centered * (int64_t)centered;
            ct_comp_add(&var_acc, sq, faults);
        }
        int64_t var_sum = ct_comp_finalize(&var_acc, faults);
        fixed_t variance = dvm_round_shift_rne(var_sum / (int64_t)ns, FIXED_FRAC_BITS, faults);

        /* Compute inverse std */
        fixed_t var_plus_eps = dvm_add(variance, ln->config.epsilon, faults);
        fixed_t std = ct_opt_sqrt(var_plus_eps, faults);
        fixed_t inv_std = (std > 0) ? dvm_div_q(FIXED_ONE, std, FIXED_FRAC_BITS, faults) : FIXED_ONE;

        /* Normalize and apply affine transformation */
        for (uint32_t i = 0; i < ns; i++) {
            fixed_t centered = dvm_sub(x[i], mean, faults);
            int64_t normalized = (int64_t)centered * (int64_t)inv_std;
            fixed_t norm = dvm_round_shift_rne(normalized, FIXED_FRAC_BITS, faults);

            fixed_t gamma = (ln->gamma != NULL) ? ln->gamma[i] : FIXED_ONE;
            fixed_t beta = (ln->beta != NULL) ? ln->beta[i] : 0;

            int64_t scaled = (int64_t)gamma * (int64_t)norm;
            y[i] = dvm_add(dvm_round_shift_rne(scaled, FIXED_FRAC_BITS, faults), beta, faults);
        }
    }

    return CT_OK;
}
