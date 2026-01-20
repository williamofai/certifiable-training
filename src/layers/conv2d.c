/**
 * @file conv2d.c
 * @project Certifiable Training
 * @brief Deterministic 2D convolution layer
 *
 * @details Implements 2D convolution for CNNs using fixed-point arithmetic:
 *          - Forward: y[n,c,h,w] = Σ Σ Σ x[n,ci,h+kh,w+kw] * W[c,ci,kh,kw] + b[c]
 *          - Backward: gradient computation for backpropagation
 *
 *          Supports:
 *          - Arbitrary kernel sizes (typically 3x3, 5x5)
 *          - Padding (same, valid)
 *          - Stride
 *          - No dilation (for simplicity in safety-critical systems)
 *
 * @traceability CT-MATH-001 §7.3
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
#include "compensated.h"
#include <string.h>

/* ============================================================================
 * Conv2D Configuration
 * ============================================================================ */

/**
 * @brief Padding mode
 */
typedef enum {
    CT_PAD_VALID = 0,   /**< No padding - output smaller than input */
    CT_PAD_SAME  = 1    /**< Pad to maintain input size (with stride=1) */
} ct_padding_mode_t;

/**
 * @brief Conv2D layer configuration
 */
typedef struct {
    uint32_t in_channels;       /**< Number of input channels */
    uint32_t out_channels;      /**< Number of output channels (filters) */
    uint32_t kernel_h;          /**< Kernel height */
    uint32_t kernel_w;          /**< Kernel width */
    uint32_t stride_h;          /**< Vertical stride */
    uint32_t stride_w;          /**< Horizontal stride */
    uint32_t padding_h;         /**< Vertical padding (per side) */
    uint32_t padding_w;         /**< Horizontal padding (per side) */
} ct_conv2d_config_t;

/**
 * @brief Conv2D layer
 */
typedef struct {
    ct_conv2d_config_t config;
    fixed_t *weights;           /**< W: [out_ch, in_ch, kh, kw] */
    fixed_t *bias;              /**< b: [out_ch] */
    uint32_t weight_size;       /**< Total weight elements */
} ct_conv2d_t;

/**
 * @brief Conv2D gradient cache for backward pass
 */
typedef struct {
    fixed_hp_t *grad_weights;   /**< ∂L/∂W: [out_ch, in_ch, kh, kw] */
    fixed_hp_t *grad_bias;      /**< ∂L/∂b: [out_ch] */
    fixed_t *input_cache;       /**< Cached input for backward */
    uint32_t cache_size;        /**< Input cache size */
} ct_conv2d_grad_t;

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Compute output dimension for convolution
 */
static uint32_t conv_output_dim(uint32_t input_dim, uint32_t kernel_dim,
                                uint32_t stride, uint32_t padding)
{
    return (input_dim + 2 * padding - kernel_dim) / stride + 1;
}

/**
 * @brief Get weight index (4D -> linear)
 */
static uint32_t weight_idx(const ct_conv2d_config_t *cfg,
                           uint32_t oc, uint32_t ic, uint32_t kh, uint32_t kw)
{
    return ((oc * cfg->in_channels + ic) * cfg->kernel_h + kh) * cfg->kernel_w + kw;
}

/* ============================================================================
 * Initialization
 * ============================================================================ */

/**
 * @brief Get default Conv2D configuration for 3x3 kernel
 */
ct_conv2d_config_t ct_conv2d_config_default(uint32_t in_ch, uint32_t out_ch)
{
    ct_conv2d_config_t cfg = {
        .in_channels = in_ch,
        .out_channels = out_ch,
        .kernel_h = 3,
        .kernel_w = 3,
        .stride_h = 1,
        .stride_w = 1,
        .padding_h = 1,  /* Same padding for 3x3 */
        .padding_w = 1
    };
    return cfg;
}

/**
 * @brief Compute required weight buffer size
 */
uint32_t ct_conv2d_weight_size(const ct_conv2d_config_t *cfg)
{
    if (cfg == NULL) return 0;
    return cfg->out_channels * cfg->in_channels * cfg->kernel_h * cfg->kernel_w;
}

/**
 * @brief Initialize Conv2D layer
 *
 * @param layer Layer to initialize
 * @param cfg Configuration
 * @param weights_buf Pre-allocated weight buffer
 * @param bias_buf Pre-allocated bias buffer [out_channels]
 * @return CT_OK on success
 */
ct_error_t ct_conv2d_init(ct_conv2d_t *layer,
                          const ct_conv2d_config_t *cfg,
                          fixed_t *weights_buf,
                          fixed_t *bias_buf)
{
    if (layer == NULL || cfg == NULL || weights_buf == NULL || bias_buf == NULL) {
        return CT_ERR_NULL;
    }

    if (cfg->kernel_h == 0 || cfg->kernel_w == 0) {
        return CT_ERR_CONFIG;
    }
    if (cfg->stride_h == 0 || cfg->stride_w == 0) {
        return CT_ERR_CONFIG;
    }

    layer->config = *cfg;
    layer->weights = weights_buf;
    layer->bias = bias_buf;
    layer->weight_size = ct_conv2d_weight_size(cfg);

    return CT_OK;
}

/* ============================================================================
 * Forward Pass
 * ============================================================================ */

/**
 * @brief Conv2D forward pass
 *
 * @param layer Initialized Conv2D layer
 * @param input Input tensor: [in_channels, height, width]
 * @param output Output tensor: [out_channels, out_height, out_width]
 * @param in_h Input height
 * @param in_w Input width
 * @param faults Fault accumulator
 * @return CT_OK on success
 *
 * @note Input and output are flat arrays in channel-major order (CHW).
 *       Caller must ensure output buffer is correctly sized.
 */
ct_error_t ct_conv2d_forward(const ct_conv2d_t *layer,
                             const fixed_t *input,
                             fixed_t *output,
                             uint32_t in_h,
                             uint32_t in_w,
                             ct_fault_flags_t *faults)
{
    if (layer == NULL || input == NULL || output == NULL) {
        return CT_ERR_NULL;
    }

    const ct_conv2d_config_t *cfg = &layer->config;

    /* Compute output dimensions */
    uint32_t out_h = conv_output_dim(in_h, cfg->kernel_h, cfg->stride_h, cfg->padding_h);
    uint32_t out_w = conv_output_dim(in_w, cfg->kernel_w, cfg->stride_w, cfg->padding_w);

    /* For each output channel */
    for (uint32_t oc = 0; oc < cfg->out_channels; oc++) {
        /* For each output spatial position */
        for (uint32_t oh = 0; oh < out_h; oh++) {
            for (uint32_t ow = 0; ow < out_w; ow++) {
                /* Accumulate convolution sum */
                ct_comp_accum_t accum;
                ct_comp_init(&accum);

                /* For each input channel */
                for (uint32_t ic = 0; ic < cfg->in_channels; ic++) {
                    /* For each kernel position */
                    for (uint32_t kh = 0; kh < cfg->kernel_h; kh++) {
                        for (uint32_t kw = 0; kw < cfg->kernel_w; kw++) {
                            /* Compute input position */
                            int32_t ih = (int32_t)(oh * cfg->stride_h + kh) - (int32_t)cfg->padding_h;
                            int32_t iw = (int32_t)(ow * cfg->stride_w + kw) - (int32_t)cfg->padding_w;

                            /* Check bounds (zero-padding) */
                            if (ih >= 0 && ih < (int32_t)in_h &&
                                iw >= 0 && iw < (int32_t)in_w) {
                                /* Input index: [ic, ih, iw] */
                                uint32_t in_idx = (ic * in_h + (uint32_t)ih) * in_w + (uint32_t)iw;
                                fixed_t in_val = input[in_idx];

                                /* Weight index: [oc, ic, kh, kw] */
                                uint32_t w_idx = weight_idx(cfg, oc, ic, kh, kw);
                                fixed_t w_val = layer->weights[w_idx];

                                /* Accumulate: in_val * w_val */
                                int64_t prod = (int64_t)in_val * (int64_t)w_val;
                                ct_comp_add(&accum, prod, faults);
                            }
                        }
                    }
                }

                /* Finalize accumulator and add bias */
                int64_t sum = ct_comp_finalize(&accum, faults);
                fixed_t conv_result = dvm_round_shift_rne(sum, FIXED_FRAC_BITS, faults);
                fixed_t with_bias = dvm_add(conv_result, layer->bias[oc], faults);

                /* Store output: [oc, oh, ow] */
                uint32_t out_idx = (oc * out_h + oh) * out_w + ow;
                output[out_idx] = with_bias;
            }
        }
    }

    return CT_OK;
}

/**
 * @brief Compute output size for given input size
 *
 * @param layer Conv2D layer
 * @param in_h Input height
 * @param in_w Input width
 * @param out_h Output: height
 * @param out_w Output: width
 * @return CT_OK on success
 */
ct_error_t ct_conv2d_output_size(const ct_conv2d_t *layer,
                                 uint32_t in_h, uint32_t in_w,
                                 uint32_t *out_h, uint32_t *out_w)
{
    if (layer == NULL || out_h == NULL || out_w == NULL) {
        return CT_ERR_NULL;
    }

    const ct_conv2d_config_t *cfg = &layer->config;
    *out_h = conv_output_dim(in_h, cfg->kernel_h, cfg->stride_h, cfg->padding_h);
    *out_w = conv_output_dim(in_w, cfg->kernel_w, cfg->stride_w, cfg->padding_w);

    return CT_OK;
}

/* ============================================================================
 * Backward Pass (Gradient Computation)
 * ============================================================================ */

/**
 * @brief Initialize Conv2D gradient cache
 *
 * @param grad Gradient cache to initialize
 * @param cfg Layer configuration
 * @param grad_weights_buf Buffer for weight gradients
 * @param grad_bias_buf Buffer for bias gradients
 * @param input_cache_buf Buffer for input cache
 * @param input_cache_size Size of input cache
 * @return CT_OK on success
 */
ct_error_t ct_conv2d_grad_init(ct_conv2d_grad_t *grad,
                               const ct_conv2d_config_t *cfg,
                               fixed_hp_t *grad_weights_buf,
                               fixed_hp_t *grad_bias_buf,
                               fixed_t *input_cache_buf,
                               uint32_t input_cache_size)
{
    if (grad == NULL || cfg == NULL) {
        return CT_ERR_NULL;
    }

    grad->grad_weights = grad_weights_buf;
    grad->grad_bias = grad_bias_buf;
    grad->input_cache = input_cache_buf;
    grad->cache_size = input_cache_size;

    return CT_OK;
}

/**
 * @brief Zero gradient buffers
 *
 * @param grad Gradient cache
 * @param cfg Layer configuration
 */
void ct_conv2d_grad_zero(ct_conv2d_grad_t *grad, const ct_conv2d_config_t *cfg)
{
    if (grad == NULL || cfg == NULL) return;

    uint32_t weight_size = ct_conv2d_weight_size(cfg);

    if (grad->grad_weights != NULL) {
        memset(grad->grad_weights, 0, weight_size * sizeof(fixed_hp_t));
    }
    if (grad->grad_bias != NULL) {
        memset(grad->grad_bias, 0, cfg->out_channels * sizeof(fixed_hp_t));
    }
}

/**
 * @brief Conv2D backward pass
 *
 * @param layer Conv2D layer
 * @param grad Gradient cache (must have input cached)
 * @param grad_output Upstream gradient [out_ch, out_h, out_w] (Q8.24)
 * @param grad_input Output gradient [in_ch, in_h, in_w] (Q8.24), can be NULL
 * @param in_h Input height
 * @param in_w Input width
 * @param faults Fault accumulator
 * @return CT_OK on success
 *
 * @note This is a simplified backward pass for demonstration.
 *       Production implementation would need careful optimization.
 */
ct_error_t ct_conv2d_backward(const ct_conv2d_t *layer,
                              ct_conv2d_grad_t *grad,
                              const fixed_hp_t *grad_output,
                              fixed_hp_t *grad_input,
                              uint32_t in_h,
                              uint32_t in_w,
                              ct_fault_flags_t *faults)
{
    if (layer == NULL || grad == NULL || grad_output == NULL) {
        return CT_ERR_NULL;
    }
    if (grad->input_cache == NULL) {
        return CT_ERR_STATE;
    }

    const ct_conv2d_config_t *cfg = &layer->config;
    const fixed_t *input = grad->input_cache;

    uint32_t out_h = conv_output_dim(in_h, cfg->kernel_h, cfg->stride_h, cfg->padding_h);
    uint32_t out_w = conv_output_dim(in_w, cfg->kernel_w, cfg->stride_w, cfg->padding_w);

    /* Zero grad_input if provided */
    if (grad_input != NULL) {
        memset(grad_input, 0, cfg->in_channels * in_h * in_w * sizeof(fixed_hp_t));
    }

    /* Compute gradients */
    for (uint32_t oc = 0; oc < cfg->out_channels; oc++) {
        for (uint32_t oh = 0; oh < out_h; oh++) {
            for (uint32_t ow = 0; ow < out_w; ow++) {
                uint32_t out_idx = (oc * out_h + oh) * out_w + ow;
                fixed_hp_t grad_out_val = grad_output[out_idx];

                /* Accumulate bias gradient */
                if (grad->grad_bias != NULL) {
                    grad->grad_bias[oc] += grad_out_val;
                }

                /* Accumulate weight gradients and input gradients */
                for (uint32_t ic = 0; ic < cfg->in_channels; ic++) {
                    for (uint32_t kh = 0; kh < cfg->kernel_h; kh++) {
                        for (uint32_t kw = 0; kw < cfg->kernel_w; kw++) {
                            int32_t ih = (int32_t)(oh * cfg->stride_h + kh) - (int32_t)cfg->padding_h;
                            int32_t iw = (int32_t)(ow * cfg->stride_w + kw) - (int32_t)cfg->padding_w;

                            if (ih >= 0 && ih < (int32_t)in_h &&
                                iw >= 0 && iw < (int32_t)in_w) {
                                uint32_t in_idx = (ic * in_h + (uint32_t)ih) * in_w + (uint32_t)iw;
                                uint32_t w_idx = weight_idx(cfg, oc, ic, kh, kw);

                                /* grad_weights += grad_output * input */
                                if (grad->grad_weights != NULL) {
                                    /* Convert input to Q8.24, multiply, accumulate */
                                    int64_t prod = (int64_t)grad_out_val *
                                                   ((int64_t)input[in_idx] << (CT_GRAD_FRAC_BITS - FIXED_FRAC_BITS));
                                    grad->grad_weights[w_idx] += dvm_round_shift_rne(prod, CT_GRAD_FRAC_BITS, faults);
                                }

                                /* grad_input += grad_output * weight */
                                if (grad_input != NULL) {
                                    int64_t prod = (int64_t)grad_out_val *
                                                   ((int64_t)layer->weights[w_idx] << (CT_GRAD_FRAC_BITS - FIXED_FRAC_BITS));
                                    grad_input[in_idx] += dvm_round_shift_rne(prod, CT_GRAD_FRAC_BITS, faults);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return CT_OK;
}
