/**
 * @file test_forward.c
 * @project Certifiable Training
 * @brief Unit tests for forward pass layers
 *
 * @traceability SRS-005, CT-MATH-001 §7.1, §12
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "ct_types.h"
#include "forward.h"
#include "dvm.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(fn) do { \
    printf("  %-50s ", #fn); \
    tests_run++; \
    if (fn()) { printf("PASS\n"); tests_passed++; } \
    else { printf("FAIL\n"); } \
} while(0)

/* Helper: convert float to fixed-point for test setup */
static fixed_t to_fixed(double f)
{
    return (fixed_t)(f * (double)FIXED_ONE + (f >= 0 ? 0.5 : -0.5));
}

/* Helper: convert fixed-point to float for verification */
static double to_float(fixed_t f)
{
    return (double)f / (double)FIXED_ONE;
}

/* ============================================================================
 * Test: Tensor Operations
 * ============================================================================ */

static int test_tensor_init_1d(void)
{
    fixed_t data[10];
    ct_tensor_t tensor;
    
    ct_tensor_init_1d(&tensor, data, 10);
    
    if (tensor.data != data) return 0;
    if (tensor.dims[0] != 10) return 0;
    if (tensor.ndims != 1) return 0;
    if (tensor.total_size != 10) return 0;
    
    return 1;
}

static int test_tensor_init_2d(void)
{
    fixed_t data[12];
    ct_tensor_t tensor;
    
    ct_tensor_init_2d(&tensor, data, 3, 4);
    
    if (tensor.dims[0] != 3) return 0;  /* rows */
    if (tensor.dims[1] != 4) return 0;  /* cols */
    if (tensor.ndims != 2) return 0;
    if (tensor.total_size != 12) return 0;
    if (tensor.strides[0] != 4) return 0;  /* row stride = cols */
    
    return 1;
}

static int test_tensor_get_set_1d(void)
{
    fixed_t data[5] = {0};
    ct_tensor_t tensor;
    
    ct_tensor_init_1d(&tensor, data, 5);
    
    ct_tensor_set_1d(&tensor, 2, to_fixed(3.5));
    fixed_t val = ct_tensor_get_1d(&tensor, 2);
    
    return val == to_fixed(3.5);
}

static int test_tensor_get_set_2d(void)
{
    fixed_t data[6] = {0};
    ct_tensor_t tensor;
    
    ct_tensor_init_2d(&tensor, data, 2, 3);
    
    /* Set element at row 1, col 2 */
    ct_tensor_set_2d(&tensor, 1, 2, to_fixed(7.25));
    fixed_t val = ct_tensor_get_2d(&tensor, 1, 2);
    
    return val == to_fixed(7.25);
}

static int test_tensor_fill(void)
{
    fixed_t data[5];
    ct_tensor_t tensor;
    
    ct_tensor_init_1d(&tensor, data, 5);
    ct_tensor_fill(&tensor, to_fixed(2.0));
    
    for (int i = 0; i < 5; i++) {
        if (data[i] != to_fixed(2.0)) return 0;
    }
    
    return 1;
}

static int test_tensor_zero(void)
{
    fixed_t data[5] = {1, 2, 3, 4, 5};
    ct_tensor_t tensor;
    
    ct_tensor_init_1d(&tensor, data, 5);
    ct_tensor_zero(&tensor);
    
    for (int i = 0; i < 5; i++) {
        if (data[i] != 0) return 0;
    }
    
    return 1;
}

/* ============================================================================
 * Test: Matrix Operations
 * ============================================================================ */

static int test_vec_add(void)
{
    fixed_t a[3] = {to_fixed(1.0), to_fixed(2.0), to_fixed(3.0)};
    fixed_t b[3] = {to_fixed(0.5), to_fixed(1.5), to_fixed(2.5)};
    fixed_t y[3];
    ct_fault_flags_t faults = {0};
    
    ct_vec_add(a, b, y, 3, &faults);
    
    if (y[0] != to_fixed(1.5)) return 0;
    if (y[1] != to_fixed(3.5)) return 0;
    if (y[2] != to_fixed(5.5)) return 0;
    
    return 1;
}

static int test_dot_product(void)
{
    fixed_t a[3] = {to_fixed(1.0), to_fixed(2.0), to_fixed(3.0)};
    fixed_t b[3] = {to_fixed(4.0), to_fixed(5.0), to_fixed(6.0)};
    ct_fault_flags_t faults = {0};
    
    /* 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32 */
    fixed_t result = ct_dot_product(a, b, 3, &faults);
    
    /* Allow small rounding error */
    double result_f = to_float(result);
    return (result_f > 31.99 && result_f < 32.01);
}

static int test_matvec_mul(void)
{
    /* 2x3 matrix times 3-vector */
    fixed_t A[6] = {
        to_fixed(1.0), to_fixed(2.0), to_fixed(3.0),
        to_fixed(4.0), to_fixed(5.0), to_fixed(6.0)
    };
    fixed_t x[3] = {to_fixed(1.0), to_fixed(1.0), to_fixed(1.0)};
    fixed_t y[2];
    ct_fault_flags_t faults = {0};
    
    ct_matvec_mul(A, x, y, 2, 3, &faults);
    
    /* Row 0: 1+2+3 = 6, Row 1: 4+5+6 = 15 */
    double y0 = to_float(y[0]);
    double y1 = to_float(y[1]);
    
    if (y0 < 5.99 || y0 > 6.01) return 0;
    if (y1 < 14.99 || y1 > 15.01) return 0;
    
    return 1;
}

/* ============================================================================
 * Test: Linear Layer
 * ============================================================================ */

static int test_linear_init(void)
{
    ct_linear_t layer;
    fixed_t weights[6];
    fixed_t bias[2];
    
    ct_error_t err = ct_linear_init(&layer, weights, bias, 3, 2);
    
    if (err != CT_OK) return 0;
    if (layer.input_size != 3) return 0;
    if (layer.output_size != 2) return 0;
    
    return 1;
}

static int test_linear_init_null_safe(void)
{
    ct_linear_t layer;
    fixed_t buf[10];
    
    if (ct_linear_init(NULL, buf, buf, 3, 2) != CT_ERR_NULL) return 0;
    if (ct_linear_init(&layer, NULL, buf, 3, 2) != CT_ERR_NULL) return 0;
    if (ct_linear_init(&layer, buf, NULL, 3, 2) != CT_ERR_NULL) return 0;
    
    return 1;
}

static int test_linear_forward_identity(void)
{
    /* Test with identity-ish weights */
    ct_linear_t layer;
    fixed_t weights[4] = {
        to_fixed(1.0), to_fixed(0.0),
        to_fixed(0.0), to_fixed(1.0)
    };
    fixed_t bias[2] = {0, 0};
    
    ct_linear_init(&layer, weights, bias, 2, 2);
    
    fixed_t input_data[2] = {to_fixed(3.0), to_fixed(5.0)};
    fixed_t output_data[2];
    ct_tensor_t input, output;
    ct_fault_flags_t faults = {0};
    
    ct_tensor_init_1d(&input, input_data, 2);
    ct_tensor_init_1d(&output, output_data, 2);
    
    ct_linear_forward(&layer, &input, &output, &faults);
    
    double y0 = to_float(output_data[0]);
    double y1 = to_float(output_data[1]);
    
    if (y0 < 2.99 || y0 > 3.01) return 0;
    if (y1 < 4.99 || y1 > 5.01) return 0;
    
    return 1;
}

static int test_linear_forward_with_bias(void)
{
    ct_linear_t layer;
    fixed_t weights[2] = {to_fixed(2.0), to_fixed(3.0)};  /* 1x2 */
    fixed_t bias[1] = {to_fixed(10.0)};
    
    ct_linear_init(&layer, weights, bias, 2, 1);
    
    fixed_t input_data[2] = {to_fixed(1.0), to_fixed(1.0)};
    fixed_t output_data[1];
    ct_tensor_t input, output;
    ct_fault_flags_t faults = {0};
    
    ct_tensor_init_1d(&input, input_data, 2);
    ct_tensor_init_1d(&output, output_data, 1);
    
    ct_linear_forward(&layer, &input, &output, &faults);
    
    /* 2*1 + 3*1 + 10 = 15 */
    double y = to_float(output_data[0]);
    
    return (y > 14.99 && y < 15.01);
}

static int test_linear_forward_determinism(void)
{
    ct_linear_t layer;
    fixed_t weights[6] = {
        to_fixed(0.5), to_fixed(-0.3), to_fixed(0.8),
        to_fixed(-0.2), to_fixed(0.6), to_fixed(-0.4)
    };
    fixed_t bias[2] = {to_fixed(0.1), to_fixed(-0.1)};
    
    ct_linear_init(&layer, weights, bias, 3, 2);
    
    fixed_t input_data[3] = {to_fixed(1.0), to_fixed(2.0), to_fixed(3.0)};
    fixed_t output1[2], output2[2];
    ct_tensor_t input, out1, out2;
    ct_fault_flags_t faults = {0};
    
    ct_tensor_init_1d(&input, input_data, 3);
    ct_tensor_init_1d(&out1, output1, 2);
    ct_tensor_init_1d(&out2, output2, 2);
    
    ct_linear_forward(&layer, &input, &out1, &faults);
    ct_linear_forward(&layer, &input, &out2, &faults);
    
    return (output1[0] == output2[0]) && (output1[1] == output2[1]);
}

/* ============================================================================
 * Test: ReLU Activation
 * ============================================================================ */

static int test_relu_positive(void)
{
    fixed_t x = to_fixed(5.0);
    fixed_t y = ct_relu(x);
    return y == x;
}

static int test_relu_negative(void)
{
    fixed_t x = to_fixed(-5.0);
    fixed_t y = ct_relu(x);
    return y == 0;
}

static int test_relu_zero(void)
{
    fixed_t y = ct_relu(0);
    return y == 0;
}

static int test_relu_derivative(void)
{
    if (ct_relu_derivative(to_fixed(5.0)) != FIXED_ONE) return 0;
    if (ct_relu_derivative(to_fixed(-5.0)) != 0) return 0;
    if (ct_relu_derivative(0) != 0) return 0;
    
    return 1;
}

/* ============================================================================
 * Test: Sigmoid Activation
 * ============================================================================ */

static ct_activation_lut_t sigmoid_lut;
static int sigmoid_lut_initialized = 0;

static void ensure_sigmoid_lut(void)
{
    if (!sigmoid_lut_initialized) {
        ct_activation_init_sigmoid_lut(&sigmoid_lut);
        sigmoid_lut_initialized = 1;
    }
}

static int test_sigmoid_zero(void)
{
    ensure_sigmoid_lut();
    
    fixed_t y = ct_sigmoid(0, &sigmoid_lut);
    double yf = to_float(y);
    
    /* sigmoid(0) = 0.5 */
    return (yf > 0.49 && yf < 0.51);
}

static int test_sigmoid_large_positive(void)
{
    ensure_sigmoid_lut();
    
    fixed_t y = ct_sigmoid(to_fixed(10.0), &sigmoid_lut);
    double yf = to_float(y);
    
    /* sigmoid(10) ≈ 1 */
    return (yf > 0.99);
}

static int test_sigmoid_large_negative(void)
{
    ensure_sigmoid_lut();
    
    fixed_t y = ct_sigmoid(to_fixed(-10.0), &sigmoid_lut);
    double yf = to_float(y);
    
    /* sigmoid(-10) ≈ 0 */
    return (yf < 0.01);
}

static int test_sigmoid_monotonic(void)
{
    ensure_sigmoid_lut();
    
    fixed_t prev = ct_sigmoid(to_fixed(-8.0), &sigmoid_lut);
    
    for (int i = -70; i <= 80; i++) {
        double x = i * 0.1;
        fixed_t y = ct_sigmoid(to_fixed(x), &sigmoid_lut);
        if (y < prev) return 0;  /* Must be monotonically increasing */
        prev = y;
    }
    
    return 1;
}

static int test_sigmoid_accuracy(void)
{
    ensure_sigmoid_lut();
    
    /* Test at several points and check error < 0.002 */
    double test_points[] = {-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0};
    
    for (int i = 0; i < 7; i++) {
        double x = test_points[i];
        double expected = 1.0 / (1.0 + exp(-x));
        fixed_t y = ct_sigmoid(to_fixed(x), &sigmoid_lut);
        double actual = to_float(y);
        double error = fabs(actual - expected);
        
        if (error > 0.002) return 0;
    }
    
    return 1;
}

static int test_sigmoid_derivative_calc(void)
{
    ensure_sigmoid_lut();
    ct_fault_flags_t faults = {0};
    
    /* At x=0, sigmoid=0.5, derivative = 0.5 * 0.5 = 0.25 */
    fixed_t sig = ct_sigmoid(0, &sigmoid_lut);
    fixed_t deriv = ct_sigmoid_derivative(sig, &faults);
    double d = to_float(deriv);
    
    return (d > 0.24 && d < 0.26);
}

/* ============================================================================
 * Test: Tanh Activation
 * ============================================================================ */

static ct_activation_lut_t tanh_lut;
static int tanh_lut_initialized = 0;

static void ensure_tanh_lut(void)
{
    if (!tanh_lut_initialized) {
        ct_activation_init_tanh_lut(&tanh_lut);
        tanh_lut_initialized = 1;
    }
}

static int test_tanh_zero(void)
{
    ensure_tanh_lut();
    
    fixed_t y = ct_tanh_act(0, &tanh_lut);
    double yf = to_float(y);
    
    /* tanh(0) = 0 */
    return (yf > -0.01 && yf < 0.01);
}

static int test_tanh_saturation(void)
{
    ensure_tanh_lut();
    
    fixed_t yp = ct_tanh_act(to_fixed(10.0), &tanh_lut);
    fixed_t yn = ct_tanh_act(to_fixed(-10.0), &tanh_lut);
    
    double yp_f = to_float(yp);
    double yn_f = to_float(yn);
    
    return (yp_f > 0.99) && (yn_f < -0.99);
}

/* ============================================================================
 * Test: Activation Layer
 * ============================================================================ */

static int test_activation_forward_relu(void)
{
    ct_activation_t act;
    ct_activation_init(&act, CT_ACT_RELU, NULL);
    
    fixed_t input_data[4] = {
        to_fixed(-2.0), to_fixed(-1.0), to_fixed(1.0), to_fixed(2.0)
    };
    fixed_t output_data[4];
    ct_tensor_t input, output;
    ct_fault_flags_t faults = {0};
    
    ct_tensor_init_1d(&input, input_data, 4);
    ct_tensor_init_1d(&output, output_data, 4);
    
    ct_activation_forward(&act, &input, &output, &faults);
    
    if (output_data[0] != 0) return 0;
    if (output_data[1] != 0) return 0;
    if (output_data[2] != to_fixed(1.0)) return 0;
    if (output_data[3] != to_fixed(2.0)) return 0;
    
    return 1;
}

static int test_activation_forward_sigmoid(void)
{
    ensure_sigmoid_lut();
    
    ct_activation_t act;
    ct_activation_init(&act, CT_ACT_SIGMOID, &sigmoid_lut);
    
    fixed_t input_data[3] = {to_fixed(-5.0), to_fixed(0.0), to_fixed(5.0)};
    fixed_t output_data[3];
    ct_tensor_t input, output;
    ct_fault_flags_t faults = {0};
    
    ct_tensor_init_1d(&input, input_data, 3);
    ct_tensor_init_1d(&output, output_data, 3);
    
    ct_activation_forward(&act, &input, &output, &faults);
    
    double y0 = to_float(output_data[0]);
    double y1 = to_float(output_data[1]);
    double y2 = to_float(output_data[2]);
    
    if (y0 > 0.1) return 0;      /* Should be close to 0 */
    if (y1 < 0.45 || y1 > 0.55) return 0;  /* Should be 0.5 */
    if (y2 < 0.9) return 0;      /* Should be close to 1 */
    
    return 1;
}

static int test_activation_determinism(void)
{
    ensure_sigmoid_lut();
    
    ct_activation_t act;
    ct_activation_init(&act, CT_ACT_SIGMOID, &sigmoid_lut);
    
    fixed_t input_data[5];
    for (int i = 0; i < 5; i++) {
        input_data[i] = to_fixed((i - 2) * 1.5);
    }
    
    fixed_t out1[5], out2[5];
    ct_tensor_t input, o1, o2;
    ct_fault_flags_t faults = {0};
    
    ct_tensor_init_1d(&input, input_data, 5);
    ct_tensor_init_1d(&o1, out1, 5);
    ct_tensor_init_1d(&o2, out2, 5);
    
    ct_activation_forward(&act, &input, &o1, &faults);
    ct_activation_forward(&act, &input, &o2, &faults);
    
    for (int i = 0; i < 5; i++) {
        if (out1[i] != out2[i]) return 0;
    }
    
    return 1;
}

/* ============================================================================
 * Test: End-to-End
 * ============================================================================ */

static int test_linear_relu_pipeline(void)
{
    /* Linear layer followed by ReLU */
    ct_linear_t layer;
    fixed_t weights[4] = {
        to_fixed(1.0), to_fixed(-1.0),
        to_fixed(-1.0), to_fixed(1.0)
    };
    fixed_t bias[2] = {to_fixed(-0.5), to_fixed(-0.5)};
    
    ct_linear_init(&layer, weights, bias, 2, 2);
    
    ct_activation_t relu;
    ct_activation_init(&relu, CT_ACT_RELU, NULL);
    
    fixed_t input_data[2] = {to_fixed(1.0), to_fixed(0.0)};
    fixed_t linear_out[2], relu_out[2];
    ct_tensor_t input, lin_tensor, relu_tensor;
    ct_fault_flags_t faults = {0};
    
    ct_tensor_init_1d(&input, input_data, 2);
    ct_tensor_init_1d(&lin_tensor, linear_out, 2);
    ct_tensor_init_1d(&relu_tensor, relu_out, 2);
    
    ct_linear_forward(&layer, &input, &lin_tensor, &faults);
    ct_activation_forward(&relu, &lin_tensor, &relu_tensor, &faults);
    
    /* linear[0] = 1*1 + (-1)*0 - 0.5 = 0.5 -> relu = 0.5 */
    /* linear[1] = (-1)*1 + 1*0 - 0.5 = -1.5 -> relu = 0 */
    double y0 = to_float(relu_out[0]);
    double y1 = to_float(relu_out[1]);
    
    if (y0 < 0.49 || y0 > 0.51) return 0;
    if (y1 != 0.0) return 0;
    
    return 1;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void)
{
    printf("==============================================\n");
    printf("Certifiable Training - Forward Pass Tests\n");
    printf("Traceability: SRS-005, CT-MATH-001 §7.1, §12\n");
    printf("==============================================\n\n");
    
    printf("Tensor operations:\n");
    RUN_TEST(test_tensor_init_1d);
    RUN_TEST(test_tensor_init_2d);
    RUN_TEST(test_tensor_get_set_1d);
    RUN_TEST(test_tensor_get_set_2d);
    RUN_TEST(test_tensor_fill);
    RUN_TEST(test_tensor_zero);
    
    printf("\nMatrix operations:\n");
    RUN_TEST(test_vec_add);
    RUN_TEST(test_dot_product);
    RUN_TEST(test_matvec_mul);
    
    printf("\nLinear layer:\n");
    RUN_TEST(test_linear_init);
    RUN_TEST(test_linear_init_null_safe);
    RUN_TEST(test_linear_forward_identity);
    RUN_TEST(test_linear_forward_with_bias);
    RUN_TEST(test_linear_forward_determinism);
    
    printf("\nReLU activation:\n");
    RUN_TEST(test_relu_positive);
    RUN_TEST(test_relu_negative);
    RUN_TEST(test_relu_zero);
    RUN_TEST(test_relu_derivative);
    
    printf("\nSigmoid activation (LUT):\n");
    RUN_TEST(test_sigmoid_zero);
    RUN_TEST(test_sigmoid_large_positive);
    RUN_TEST(test_sigmoid_large_negative);
    RUN_TEST(test_sigmoid_monotonic);
    RUN_TEST(test_sigmoid_accuracy);
    RUN_TEST(test_sigmoid_derivative_calc);
    
    printf("\nTanh activation (LUT):\n");
    RUN_TEST(test_tanh_zero);
    RUN_TEST(test_tanh_saturation);
    
    printf("\nActivation layer:\n");
    RUN_TEST(test_activation_forward_relu);
    RUN_TEST(test_activation_forward_sigmoid);
    RUN_TEST(test_activation_determinism);
    
    printf("\nEnd-to-end:\n");
    RUN_TEST(test_linear_relu_pipeline);
    
    printf("\n==============================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("==============================================\n");
    
    return (tests_passed == tests_run) ? 0 : 1;
}
