/**
 * @file forward.c
 * @project Certifiable Training
 * @brief Forward pass layers: linear, activations
 *
 * @details Implements neural network forward pass using fixed-point arithmetic.
 *          All operations use DVM primitives for determinism.
 *
 * @traceability CT-MATH-001 §7.1, §12
 * @compliance MISRA-C:2012, DO-178C, IEC 62304, ISO 26262
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 */

#include "forward.h"
#include "dvm.h"
#include "compensated.h"
#include <stddef.h>
#include <math.h>  /* Only for LUT initialization - not used at runtime */

/* ============================================================================
 * Tensor Operations
 * ============================================================================ */

void ct_tensor_init_1d(ct_tensor_t *tensor, fixed_t *data, uint32_t size)
{
    if (tensor == NULL) return;
    
    tensor->data = data;
    tensor->dims[0] = size;
    tensor->dims[1] = 1;
    tensor->dims[2] = 1;
    tensor->dims[3] = 1;
    tensor->strides[0] = 1;
    tensor->strides[1] = size;
    tensor->strides[2] = size;
    tensor->strides[3] = size;
    tensor->ndims = 1;
    tensor->total_size = size;
}

void ct_tensor_init_2d(ct_tensor_t *tensor, fixed_t *data,
                       uint32_t rows, uint32_t cols)
{
    if (tensor == NULL) return;
    
    tensor->data = data;
    tensor->dims[0] = rows;
    tensor->dims[1] = cols;
    tensor->dims[2] = 1;
    tensor->dims[3] = 1;
    tensor->strides[0] = cols;  /* Row stride */
    tensor->strides[1] = 1;     /* Column stride */
    tensor->strides[2] = rows * cols;
    tensor->strides[3] = rows * cols;
    tensor->ndims = 2;
    tensor->total_size = rows * cols;
}

fixed_t ct_tensor_get_1d(const ct_tensor_t *tensor, uint32_t i)
{
    if (tensor == NULL || tensor->data == NULL) return 0;
    if (i >= tensor->total_size) return 0;
    return tensor->data[i];
}

void ct_tensor_set_1d(ct_tensor_t *tensor, uint32_t i, fixed_t value)
{
    if (tensor == NULL || tensor->data == NULL) return;
    if (i >= tensor->total_size) return;
    tensor->data[i] = value;
}

fixed_t ct_tensor_get_2d(const ct_tensor_t *tensor, uint32_t row, uint32_t col)
{
    if (tensor == NULL || tensor->data == NULL) return 0;
    if (row >= tensor->dims[0] || col >= tensor->dims[1]) return 0;
    return tensor->data[row * tensor->strides[0] + col * tensor->strides[1]];
}

void ct_tensor_set_2d(ct_tensor_t *tensor, uint32_t row, uint32_t col,
                      fixed_t value)
{
    if (tensor == NULL || tensor->data == NULL) return;
    if (row >= tensor->dims[0] || col >= tensor->dims[1]) return;
    tensor->data[row * tensor->strides[0] + col * tensor->strides[1]] = value;
}

void ct_tensor_fill(ct_tensor_t *tensor, fixed_t value)
{
    if (tensor == NULL || tensor->data == NULL) return;
    
    for (uint32_t i = 0; i < tensor->total_size; i++) {
        tensor->data[i] = value;
    }
}

void ct_tensor_zero(ct_tensor_t *tensor)
{
    ct_tensor_fill(tensor, 0);
}

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

void ct_matvec_mul(const fixed_t *A, const fixed_t *x, fixed_t *y,
                   uint32_t rows, uint32_t cols,
                   ct_fault_flags_t *faults)
{
    if (A == NULL || x == NULL || y == NULL) return;
    
    for (uint32_t i = 0; i < rows; i++) {
        /* Compute dot product of row i with x */
        ct_comp_accum_t accum;
        ct_comp_init(&accum);
        
        for (uint32_t j = 0; j < cols; j++) {
            /* A[i,j] * x[j] using 64-bit intermediate */
            int64_t prod = (int64_t)A[i * cols + j] * (int64_t)x[j];
            ct_comp_add(&accum, prod, faults);
        }
        
        /* Round the accumulated sum back to Q16.16 */
        int64_t sum = ct_comp_finalize(&accum, faults);
        y[i] = dvm_round_shift_rne(sum, FIXED_FRAC_BITS, faults);
    }
}

void ct_vec_add(const fixed_t *a, const fixed_t *b, fixed_t *y,
                uint32_t size, ct_fault_flags_t *faults)
{
    if (a == NULL || b == NULL || y == NULL) return;
    
    for (uint32_t i = 0; i < size; i++) {
        y[i] = dvm_add(a[i], b[i], faults);
    }
}

fixed_t ct_dot_product(const fixed_t *a, const fixed_t *b,
                       uint32_t size, ct_fault_flags_t *faults)
{
    if (a == NULL || b == NULL || size == 0) return 0;
    
    ct_comp_accum_t accum;
    ct_comp_init(&accum);
    
    for (uint32_t i = 0; i < size; i++) {
        int64_t prod = (int64_t)a[i] * (int64_t)b[i];
        ct_comp_add(&accum, prod, faults);
    }
    
    int64_t sum = ct_comp_finalize(&accum, faults);
    return dvm_round_shift_rne(sum, FIXED_FRAC_BITS, faults);
}

/* ============================================================================
 * Linear Layer
 * ============================================================================ */

ct_error_t ct_linear_init(ct_linear_t *layer,
                          fixed_t *weights_buf,
                          fixed_t *bias_buf,
                          uint32_t input_size,
                          uint32_t output_size)
{
    if (layer == NULL) return CT_ERR_NULL;
    if (weights_buf == NULL || bias_buf == NULL) return CT_ERR_NULL;
    if (input_size == 0 || output_size == 0) return CT_ERR_CONFIG;
    
    ct_tensor_init_2d(&layer->weights, weights_buf, output_size, input_size);
    ct_tensor_init_1d(&layer->bias, bias_buf, output_size);
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    return CT_OK;
}

ct_error_t ct_linear_forward(const ct_linear_t *layer,
                             const ct_tensor_t *input,
                             ct_tensor_t *output,
                             ct_fault_flags_t *faults)
{
    if (layer == NULL || input == NULL || output == NULL) {
        return CT_ERR_NULL;
    }
    
    if (input->total_size != layer->input_size) {
        return CT_ERR_DIMENSION;
    }
    
    if (output->total_size != layer->output_size) {
        return CT_ERR_DIMENSION;
    }
    
    /* y = W * x */
    ct_matvec_mul(layer->weights.data, input->data, output->data,
                  layer->output_size, layer->input_size, faults);
    
    /* y = y + b */
    ct_vec_add(output->data, layer->bias.data, output->data,
               layer->output_size, faults);
    
    return CT_OK;
}

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

/**
 * @brief Convert float to Q16.16 fixed-point (initialization only)
 *
 * @note This uses floating-point and is ONLY for LUT initialization.
 *       Never called at inference/training runtime.
 */
static fixed_t float_to_fixed(double f)
{
    return (fixed_t)(f * (double)FIXED_ONE + (f >= 0 ? 0.5 : -0.5));
}

void ct_activation_init_sigmoid_lut(ct_activation_lut_t *lut)
{
    if (lut == NULL) return;
    
    lut->domain_min = float_to_fixed(-8.0);
    lut->domain_max = float_to_fixed(8.0);
    lut->step_size = float_to_fixed(16.0 / 256.0);  /* 0.0625 */
    
    /* Fill table with sigmoid values */
    for (int i = 0; i < CT_ACTIVATION_LUT_SIZE; i++) {
        double x = -8.0 + (16.0 * i) / 256.0;
        double sigmoid = 1.0 / (1.0 + exp(-x));
        lut->table[i] = float_to_fixed(sigmoid);
    }
}

void ct_activation_init_tanh_lut(ct_activation_lut_t *lut)
{
    if (lut == NULL) return;
    
    lut->domain_min = float_to_fixed(-8.0);
    lut->domain_max = float_to_fixed(8.0);
    lut->step_size = float_to_fixed(16.0 / 256.0);
    
    /* Fill table with tanh values */
    for (int i = 0; i < CT_ACTIVATION_LUT_SIZE; i++) {
        double x = -8.0 + (16.0 * i) / 256.0;
        double tanh_val = tanh(x);
        lut->table[i] = float_to_fixed(tanh_val);
    }
}

void ct_activation_init(ct_activation_t *act,
                        ct_activation_type_t type,
                        const ct_activation_lut_t *lut)
{
    if (act == NULL) return;
    
    act->type = type;
    act->lut = lut;
}

fixed_t ct_relu(fixed_t x)
{
    return (x > 0) ? x : 0;
}

fixed_t ct_relu_derivative(fixed_t x)
{
    return (x > 0) ? FIXED_ONE : 0;
}

/**
 * @brief Lookup with linear interpolation
 *
 * @details Implements CT-MATH-001 §12.3 exactly:
 *          1. Saturate inputs outside [-8, +8]
 *          2. Map to table index
 *          3. Linear interpolation between entries
 */
fixed_t ct_sigmoid(fixed_t x, const ct_activation_lut_t *lut)
{
    if (lut == NULL) return 0;
    
    /* Saturation */
    if (x <= lut->domain_min) return 0;
    if (x >= lut->domain_max) return FIXED_ONE;
    
    /* Shift to [0, 16] range in Q16.16 */
    int64_t x_shifted = (int64_t)x - (int64_t)lut->domain_min;
    
    /* 
     * Map to table index.
     * x_shifted is in [0, 16] in Q16.16.
     * We want index in [0, 256].
     * index = (x_shifted * 256) / 16 = x_shifted * 16
     * 
     * In Q16.16: x_shifted * 16 gives us a Q16.16 value.
     * Divide by FIXED_ONE to get the integer index.
     */
    int64_t scaled = x_shifted * 16;  /* Now in range [0, 256] in Q16.16 */
    uint32_t index = (uint32_t)(scaled >> FIXED_FRAC_BITS);
    
    /* Clamp index to valid range */
    if (index >= CT_ACTIVATION_LUT_SIZE - 1) {
        index = CT_ACTIVATION_LUT_SIZE - 2;
    }
    
    /* Fractional part for interpolation (in Q16.16) */
    int64_t frac = scaled & ((1LL << FIXED_FRAC_BITS) - 1);
    
    /* Linear interpolation: y0 + frac * (y1 - y0) */
    fixed_t y0 = lut->table[index];
    fixed_t y1 = lut->table[index + 1];
    int64_t diff = (int64_t)y1 - (int64_t)y0;
    int64_t interp = (diff * frac) >> FIXED_FRAC_BITS;
    
    return (fixed_t)(y0 + interp);
}

fixed_t ct_sigmoid_derivative(fixed_t sigmoid_x, ct_fault_flags_t *faults)
{
    /* σ'(x) = σ(x) * (1 - σ(x)) */
    fixed_t one_minus = dvm_sub(FIXED_ONE, sigmoid_x, faults);
    int64_t product = (int64_t)sigmoid_x * (int64_t)one_minus;
    return dvm_round_shift_rne(product, FIXED_FRAC_BITS, faults);
}

fixed_t ct_tanh_act(fixed_t x, const ct_activation_lut_t *lut)
{
    if (lut == NULL) return 0;
    
    /* Saturation */
    if (x <= lut->domain_min) return -FIXED_ONE;
    if (x >= lut->domain_max) return FIXED_ONE;
    
    /* Same lookup logic as sigmoid */
    int64_t x_shifted = (int64_t)x - (int64_t)lut->domain_min;
    int64_t scaled = x_shifted * 16;
    uint32_t index = (uint32_t)(scaled >> FIXED_FRAC_BITS);
    
    if (index >= CT_ACTIVATION_LUT_SIZE - 1) {
        index = CT_ACTIVATION_LUT_SIZE - 2;
    }
    
    int64_t frac = scaled & ((1LL << FIXED_FRAC_BITS) - 1);
    
    fixed_t y0 = lut->table[index];
    fixed_t y1 = lut->table[index + 1];
    int64_t diff = (int64_t)y1 - (int64_t)y0;
    int64_t interp = (diff * frac) >> FIXED_FRAC_BITS;
    
    return (fixed_t)(y0 + interp);
}

fixed_t ct_tanh_derivative(fixed_t tanh_x, ct_fault_flags_t *faults)
{
    /* tanh'(x) = 1 - tanh²(x) */
    int64_t squared = (int64_t)tanh_x * (int64_t)tanh_x;
    fixed_t tanh_sq = dvm_round_shift_rne(squared, FIXED_FRAC_BITS, faults);
    return dvm_sub(FIXED_ONE, tanh_sq, faults);
}

fixed_t ct_activation_apply(const ct_activation_t *act, fixed_t x,
                            ct_fault_flags_t *faults)
{
    if (act == NULL) return x;
    
    switch (act->type) {
        case CT_ACT_NONE:
            return x;
        
        case CT_ACT_RELU:
            return ct_relu(x);
        
        case CT_ACT_SIGMOID:
            return ct_sigmoid(x, act->lut);
        
        case CT_ACT_TANH:
            return ct_tanh_act(x, act->lut);
        
        default:
            return x;
    }
    
    (void)faults;  /* Not used for basic activations */
}

ct_error_t ct_activation_forward(const ct_activation_t *act,
                                 const ct_tensor_t *input,
                                 ct_tensor_t *output,
                                 ct_fault_flags_t *faults)
{
    if (act == NULL || input == NULL || output == NULL) {
        return CT_ERR_NULL;
    }
    
    if (input->total_size != output->total_size) {
        return CT_ERR_DIMENSION;
    }
    
    for (uint32_t i = 0; i < input->total_size; i++) {
        output->data[i] = ct_activation_apply(act, input->data[i], faults);
    }
    
    return CT_OK;
}
