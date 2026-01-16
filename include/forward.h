/**
 * @file forward.h
 * @project Certifiable Training
 * @brief Forward pass layers: linear, activations, convolution
 *
 * @details Provides neural network layer implementations using fixed-point
 *          arithmetic. All operations are deterministic and use DVM primitives.
 *
 *          Layers:
 *          - Linear (dense): y = Wx + b
 *          - Activations: ReLU, Sigmoid (LUT), Tanh (LUT)
 *
 * @traceability CT-MATH-001 §7.1, §12; CT-STRUCT-001 §6
 * @compliance MISRA-C:2012, DO-178C, IEC 62304, ISO 26262
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 */

#ifndef CT_FORWARD_H
#define CT_FORWARD_H

#include "ct_types.h"

/* ============================================================================
 * Tensor Structure
 * ============================================================================ */

/** Maximum tensor dimensions */
#define CT_MAX_DIMS 4

/**
 * @brief Tensor descriptor for fixed-point data
 *
 * @details Describes a multi-dimensional array of fixed-point values.
 *          Data buffer is caller-provided (no dynamic allocation).
 *
 * @ref CT-STRUCT-001 §5.1
 */
typedef struct {
    fixed_t *data;                  /**< Data buffer (caller-provided) */
    uint32_t dims[CT_MAX_DIMS];     /**< Dimension sizes */
    uint32_t strides[CT_MAX_DIMS];  /**< Element strides (for views) */
    uint32_t ndims;                 /**< Number of dimensions */
    uint32_t total_size;            /**< Total elements */
} ct_tensor_t;

/**
 * @brief Initialize a 1D tensor (vector)
 *
 * @param tensor Tensor to initialize
 * @param data   Data buffer
 * @param size   Number of elements
 */
void ct_tensor_init_1d(ct_tensor_t *tensor, fixed_t *data, uint32_t size);

/**
 * @brief Initialize a 2D tensor (matrix)
 *
 * @param tensor Tensor to initialize
 * @param data   Data buffer (row-major)
 * @param rows   Number of rows
 * @param cols   Number of columns
 */
void ct_tensor_init_2d(ct_tensor_t *tensor, fixed_t *data,
                       uint32_t rows, uint32_t cols);

/**
 * @brief Get element at index (1D)
 */
fixed_t ct_tensor_get_1d(const ct_tensor_t *tensor, uint32_t i);

/**
 * @brief Set element at index (1D)
 */
void ct_tensor_set_1d(ct_tensor_t *tensor, uint32_t i, fixed_t value);

/**
 * @brief Get element at (row, col) (2D)
 */
fixed_t ct_tensor_get_2d(const ct_tensor_t *tensor, uint32_t row, uint32_t col);

/**
 * @brief Set element at (row, col) (2D)
 */
void ct_tensor_set_2d(ct_tensor_t *tensor, uint32_t row, uint32_t col,
                      fixed_t value);

/**
 * @brief Fill tensor with a value
 */
void ct_tensor_fill(ct_tensor_t *tensor, fixed_t value);

/**
 * @brief Fill tensor with zeros
 */
void ct_tensor_zero(ct_tensor_t *tensor);

/* ============================================================================
 * Linear Layer
 * ============================================================================ */

/**
 * @brief Linear (dense/fully-connected) layer
 *
 * @details Computes y = Wx + b where:
 *          - W is [output_size x input_size]
 *          - x is [input_size]
 *          - b is [output_size]
 *          - y is [output_size]
 *
 * @ref CT-STRUCT-001 §6.1
 */
typedef struct {
    ct_tensor_t weights;        /**< W: [output_size, input_size] */
    ct_tensor_t bias;           /**< b: [output_size] */
    uint32_t input_size;        /**< Input dimension */
    uint32_t output_size;       /**< Output dimension */
} ct_linear_t;

/**
 * @brief Initialize a linear layer
 *
 * @param layer       Layer to initialize
 * @param weights_buf Buffer for weights [output_size * input_size]
 * @param bias_buf    Buffer for bias [output_size]
 * @param input_size  Input dimension
 * @param output_size Output dimension
 * @return CT_OK on success
 */
ct_error_t ct_linear_init(ct_linear_t *layer,
                          fixed_t *weights_buf,
                          fixed_t *bias_buf,
                          uint32_t input_size,
                          uint32_t output_size);

/**
 * @brief Forward pass through linear layer
 *
 * @param layer  Initialized layer
 * @param input  Input tensor [input_size]
 * @param output Output tensor [output_size] (caller-provided)
 * @param faults Fault flags
 * @return CT_OK on success
 *
 * @details Computes y = Wx + b using fixed-point matrix-vector multiply.
 *          All multiplications use 64-bit intermediate with rounding.
 *
 * Complexity: O(input_size * output_size)
 * Determinism: Bit-perfect
 *
 * @ref CT-MATH-001 §7.1
 */
ct_error_t ct_linear_forward(const ct_linear_t *layer,
                             const ct_tensor_t *input,
                             ct_tensor_t *output,
                             ct_fault_flags_t *faults);

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

/**
 * @brief Activation function types
 *
 * @ref CT-STRUCT-001 §6.2
 */
typedef enum {
    CT_ACT_NONE     = 0,    /**< No activation (identity) */
    CT_ACT_RELU     = 1,    /**< ReLU: max(0, x) */
    CT_ACT_SIGMOID  = 2,    /**< Sigmoid: LUT-based */
    CT_ACT_TANH     = 3     /**< Tanh: LUT-based */
} ct_activation_type_t;

/** Size of activation LUT (257 entries for linear interpolation) */
#define CT_ACTIVATION_LUT_SIZE 257

/**
 * @brief Activation lookup table
 *
 * @details Pre-computed table for sigmoid/tanh with linear interpolation.
 *          Domain: [-8, +8] mapped to 257 entries.
 *
 * @ref CT-MATH-001 §12.3
 */
typedef struct {
    fixed_t table[CT_ACTIVATION_LUT_SIZE];  /**< LUT values */
    fixed_t domain_min;                      /**< -8 in Q16.16 */
    fixed_t domain_max;                      /**< +8 in Q16.16 */
    fixed_t step_size;                       /**< (max-min)/(size-1) */
} ct_activation_lut_t;

/**
 * @brief Activation layer
 */
typedef struct {
    ct_activation_type_t type;          /**< Activation type */
    const ct_activation_lut_t *lut;     /**< LUT for sigmoid/tanh (NULL for ReLU) */
} ct_activation_t;

/**
 * @brief Initialize the global sigmoid LUT
 *
 * @param lut LUT structure to initialize
 *
 * @details Fills the table with pre-computed sigmoid values.
 *          Call once at startup, then share across all sigmoid layers.
 */
void ct_activation_init_sigmoid_lut(ct_activation_lut_t *lut);

/**
 * @brief Initialize the global tanh LUT
 *
 * @param lut LUT structure to initialize
 */
void ct_activation_init_tanh_lut(ct_activation_lut_t *lut);

/**
 * @brief Initialize an activation layer
 *
 * @param act  Activation layer to initialize
 * @param type Activation type
 * @param lut  LUT for sigmoid/tanh (NULL for ReLU/none)
 */
void ct_activation_init(ct_activation_t *act,
                        ct_activation_type_t type,
                        const ct_activation_lut_t *lut);

/**
 * @brief Apply ReLU activation: y = max(0, x)
 *
 * @param x Input value (Q16.16)
 * @return ReLU(x)
 *
 * Complexity: O(1)
 * Determinism: Bit-perfect (trivial)
 *
 * @ref CT-MATH-001 §12.2
 */
fixed_t ct_relu(fixed_t x);

/**
 * @brief Apply ReLU derivative: 1 if x > 0, else 0
 *
 * @param x Input value
 * @return ReLU'(x) in Q16.16
 */
fixed_t ct_relu_derivative(fixed_t x);

/**
 * @brief Apply sigmoid activation using LUT
 *
 * @param x   Input value (Q16.16)
 * @param lut Initialized sigmoid LUT
 * @return sigmoid(x) in Q16.16
 *
 * @details Uses linear interpolation between table entries.
 *          Saturates to 0 or 1 outside [-8, +8].
 *
 * Complexity: O(1)
 * Determinism: Bit-perfect
 *
 * @ref CT-MATH-001 §12.3
 */
fixed_t ct_sigmoid(fixed_t x, const ct_activation_lut_t *lut);

/**
 * @brief Apply sigmoid derivative: σ(x) * (1 - σ(x))
 *
 * @param sigmoid_x Pre-computed sigmoid(x)
 * @param faults    Fault flags
 * @return σ'(x) in Q16.16
 *
 * @ref CT-MATH-001 §12.5
 */
fixed_t ct_sigmoid_derivative(fixed_t sigmoid_x, ct_fault_flags_t *faults);

/**
 * @brief Apply tanh activation using LUT
 *
 * @param x   Input value (Q16.16)
 * @param lut Initialized tanh LUT
 * @return tanh(x) in Q16.16
 */
fixed_t ct_tanh_act(fixed_t x, const ct_activation_lut_t *lut);

/**
 * @brief Apply tanh derivative: 1 - tanh²(x)
 *
 * @param tanh_x Pre-computed tanh(x)
 * @param faults Fault flags
 * @return tanh'(x) in Q16.16
 */
fixed_t ct_tanh_derivative(fixed_t tanh_x, ct_fault_flags_t *faults);

/**
 * @brief Apply activation to single value
 *
 * @param act   Activation layer
 * @param x     Input value
 * @param faults Fault flags
 * @return Activated value
 */
fixed_t ct_activation_apply(const ct_activation_t *act, fixed_t x,
                            ct_fault_flags_t *faults);

/**
 * @brief Forward pass through activation layer (tensor)
 *
 * @param act    Activation layer
 * @param input  Input tensor
 * @param output Output tensor (can be same as input for in-place)
 * @param faults Fault flags
 * @return CT_OK on success
 *
 * @details Applies activation element-wise.
 *
 * Complexity: O(n)
 * Determinism: Bit-perfect
 */
ct_error_t ct_activation_forward(const ct_activation_t *act,
                                 const ct_tensor_t *input,
                                 ct_tensor_t *output,
                                 ct_fault_flags_t *faults);

/* ============================================================================
 * Matrix Operations (for linear layer)
 * ============================================================================ */

/**
 * @brief Matrix-vector multiply: y = A * x
 *
 * @param A      Matrix [rows x cols]
 * @param x      Vector [cols]
 * @param y      Output vector [rows] (caller-provided)
 * @param rows   Number of rows in A
 * @param cols   Number of columns in A
 * @param faults Fault flags
 *
 * @details Uses 64-bit intermediate for each dot product element,
 *          then rounds back to 32-bit.
 *
 * Complexity: O(rows * cols)
 * Determinism: Bit-perfect
 */
void ct_matvec_mul(const fixed_t *A, const fixed_t *x, fixed_t *y,
                   uint32_t rows, uint32_t cols,
                   ct_fault_flags_t *faults);

/**
 * @brief Vector addition: y = a + b
 *
 * @param a      First vector
 * @param b      Second vector
 * @param y      Output vector (can be same as a or b)
 * @param size   Vector size
 * @param faults Fault flags
 */
void ct_vec_add(const fixed_t *a, const fixed_t *b, fixed_t *y,
                uint32_t size, ct_fault_flags_t *faults);

/**
 * @brief Dot product: result = a · b
 *
 * @param a      First vector
 * @param b      Second vector
 * @param size   Vector size
 * @param faults Fault flags
 * @return Dot product in Q16.16
 *
 * @details Uses compensated summation internally for precision.
 */
fixed_t ct_dot_product(const fixed_t *a, const fixed_t *b,
                       uint32_t size, ct_fault_flags_t *faults);

#endif /* CT_FORWARD_H */
