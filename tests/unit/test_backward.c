/**
 * @file test_backward.c
 * @project Certifiable Training
 * @brief Unit tests for backward pass (backpropagation).
 *
 * @traceability SRS-006-BACKWARD
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust. All rights reserved.
 * @license Licensed under the GPL-3.0 (Open Source) or Commercial License.
 *          For commercial licensing: william@fstopify.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "backward.h"
#include "forward.h"
#include "dvm.h"

/* ============================================================================
 * Test Framework
 * ============================================================================ */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN_TEST(name) do { \
    printf("  Testing %s... ", #name); \
    test_##name(); \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAIL\n    Assertion failed: %s\n    at %s:%d\n", \
               #cond, __FILE__, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_GT(a, b) ASSERT((a) > (b))
#define ASSERT_LT(a, b) ASSERT((a) < (b))
#define ASSERT_GE(a, b) ASSERT((a) >= (b))
#define ASSERT_LE(a, b) ASSERT((a) <= (b))

/* Allow ~1% error for floating point comparisons */
#define ASSERT_NEAR_Q16(actual, expected, tolerance) do { \
    int32_t diff = ((actual) > (expected)) ? ((actual) - (expected)) : ((expected) - (actual)); \
    if (diff > (tolerance)) { \
        printf("FAIL\n    Expected: %d, Actual: %d, Diff: %d > %d\n", \
               (expected), (actual), diff, (tolerance)); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ============================================================================
 * Test: Gradient Tensor Initialization
 * ============================================================================ */

TEST(grad_tensor_init_1d) {
    fixed_hp_t buffer[10];
    ct_grad_tensor_t grad;
    
    ct_error_t err = ct_grad_tensor_init(&grad, buffer, 10, 0);
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(grad.ndims, 1);
    ASSERT_EQ(grad.dims[0], 10);
    ASSERT_EQ(grad.total_size, 10);
    ASSERT_EQ(grad.strides[0], 1);
}

TEST(grad_tensor_init_2d) {
    fixed_hp_t buffer[12];
    ct_grad_tensor_t grad;
    
    ct_error_t err = ct_grad_tensor_init(&grad, buffer, 3, 4);
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(grad.ndims, 2);
    ASSERT_EQ(grad.dims[0], 3);
    ASSERT_EQ(grad.dims[1], 4);
    ASSERT_EQ(grad.total_size, 12);
    ASSERT_EQ(grad.strides[0], 4);
    ASSERT_EQ(grad.strides[1], 1);
}

TEST(grad_tensor_zero) {
    fixed_hp_t buffer[5] = {1, 2, 3, 4, 5};
    ct_grad_tensor_t grad;
    
    ct_grad_tensor_init(&grad, buffer, 5, 0);
    ct_grad_tensor_zero(&grad);
    
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(buffer[i], 0);
    }
}

TEST(grad_tensor_accessors) {
    fixed_hp_t buffer[6];
    ct_grad_tensor_t grad;
    
    ct_grad_tensor_init(&grad, buffer, 2, 3);
    
    /* Set values */
    ct_grad_set_2d(&grad, 0, 0, 100);
    ct_grad_set_2d(&grad, 0, 1, 200);
    ct_grad_set_2d(&grad, 0, 2, 300);
    ct_grad_set_2d(&grad, 1, 0, 400);
    ct_grad_set_2d(&grad, 1, 1, 500);
    ct_grad_set_2d(&grad, 1, 2, 600);
    
    /* Verify */
    ASSERT_EQ(ct_grad_get_2d(&grad, 0, 0), 100);
    ASSERT_EQ(ct_grad_get_2d(&grad, 0, 1), 200);
    ASSERT_EQ(ct_grad_get_2d(&grad, 0, 2), 300);
    ASSERT_EQ(ct_grad_get_2d(&grad, 1, 0), 400);
    ASSERT_EQ(ct_grad_get_2d(&grad, 1, 1), 500);
    ASSERT_EQ(ct_grad_get_2d(&grad, 1, 2), 600);
}

/* ============================================================================
 * Test: Format Conversion
 * ============================================================================ */

TEST(fixed_to_grad_conversion) {
    /* Q16.16 value 1.0 should become Q8.24 value 1.0 */
    fixed_t one_q16 = FIXED_ONE;  /* 65536 */
    fixed_hp_t one_q24 = ct_fixed_to_grad(one_q16);
    
    /* Q8.24 one = 2^24 = 16777216 */
    ASSERT_EQ(one_q24, CT_GRAD_ONE);
}

TEST(grad_to_fixed_conversion) {
    ct_fault_flags_t faults = {0};
    
    /* Q8.24 value 1.0 should become Q16.16 value 1.0 */
    fixed_hp_t one_q24 = CT_GRAD_ONE;
    fixed_t one_q16 = ct_grad_to_fixed(one_q24, &faults);
    
    ASSERT_EQ(one_q16, FIXED_ONE);
    ASSERT_EQ(faults.overflow, 0);
    ASSERT_EQ(faults.underflow, 0);
}

TEST(conversion_roundtrip) {
    ct_fault_flags_t faults = {0};
    
    /* Test various values */
    fixed_t test_values[] = {0, FIXED_ONE, FIXED_HALF, -FIXED_ONE, 12345};
    
    for (int i = 0; i < 5; i++) {
        fixed_t original = test_values[i];
        fixed_hp_t widened = ct_fixed_to_grad(original);
        fixed_t restored = ct_grad_to_fixed(widened, &faults);
        
        ASSERT_EQ(original, restored);
    }
}

/* ============================================================================
 * Test: MSE Loss
 * ============================================================================ */

TEST(mse_loss_forward_zero) {
    /* When output == target, loss should be 0 */
    fixed_t out_buf[4] = {FIXED_ONE, FIXED_ONE, FIXED_ONE, FIXED_ONE};
    fixed_t tgt_buf[4] = {FIXED_ONE, FIXED_ONE, FIXED_ONE, FIXED_ONE};
    
    ct_tensor_t output, target;
    ct_tensor_init_1d(&output, out_buf, 4);
    ct_tensor_init_1d(&target, tgt_buf, 4);
    
    fixed_t loss;
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_loss_mse_forward(&output, &target, &loss, &faults);
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(loss, 0);
}

TEST(mse_loss_forward_nonzero) {
    /* output = [1, 1], target = [0, 0] */
    /* MSE = (1/2) * ((1-0)^2 + (1-0)^2) = (1/2) * 2 = 1 */
    fixed_t out_buf[2] = {FIXED_ONE, FIXED_ONE};
    fixed_t tgt_buf[2] = {0, 0};
    
    ct_tensor_t output, target;
    ct_tensor_init_1d(&output, out_buf, 2);
    ct_tensor_init_1d(&target, tgt_buf, 2);
    
    fixed_t loss;
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_loss_mse_forward(&output, &target, &loss, &faults);
    ASSERT_EQ(err, CT_OK);
    
    /* Loss should be approximately 1.0 */
    ASSERT_NEAR_Q16(loss, FIXED_ONE, FIXED_ONE / 100);
}

TEST(mse_loss_backward) {
    /* output = [2, 0], target = [1, 1], N=2 */
    /* Gradient = (2/N) * (output - target) = (2/2) * [1, -1] = [1, -1] */
    fixed_t out_buf[2] = {2 * FIXED_ONE, 0};
    fixed_t tgt_buf[2] = {FIXED_ONE, FIXED_ONE};
    fixed_hp_t grad_buf[2];
    
    ct_tensor_t output, target;
    ct_grad_tensor_t grad_output;
    
    ct_tensor_init_1d(&output, out_buf, 2);
    ct_tensor_init_1d(&target, tgt_buf, 2);
    ct_grad_tensor_init(&grad_output, grad_buf, 2, 0);
    
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_loss_mse_backward(&output, &target, &grad_output, &faults);
    ASSERT_EQ(err, CT_OK);
    
    /* grad[0] should be ~1.0, grad[1] should be ~-1.0 */
    /* In Q8.24: 1.0 = 16777216 */
    int32_t tolerance = CT_GRAD_ONE / 100;  /* 1% tolerance */
    ASSERT_NEAR_Q16(grad_buf[0], CT_GRAD_ONE, tolerance);
    ASSERT_NEAR_Q16(grad_buf[1], -CT_GRAD_ONE, tolerance);
}

/* ============================================================================
 * Test: Activation Derivatives
 * ============================================================================ */

TEST(relu_backward_positive) {
    /* Pre-activation positive: gradient passes through */
    fixed_t pre_buf[3] = {FIXED_ONE, FIXED_HALF, 100};
    fixed_hp_t grad_out_buf[3] = {CT_GRAD_ONE, CT_GRAD_HALF, 12345};
    fixed_hp_t grad_in_buf[3];
    
    ct_tensor_t pre_act;
    ct_grad_tensor_t grad_out, grad_in;
    
    ct_tensor_init_1d(&pre_act, pre_buf, 3);
    ct_grad_tensor_init(&grad_out, grad_out_buf, 3, 0);
    ct_grad_tensor_init(&grad_in, grad_in_buf, 3, 0);
    
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_activation_relu_backward(&grad_out, &pre_act, &grad_in, &faults);
    ASSERT_EQ(err, CT_OK);
    
    ASSERT_EQ(grad_in_buf[0], CT_GRAD_ONE);
    ASSERT_EQ(grad_in_buf[1], CT_GRAD_HALF);
    ASSERT_EQ(grad_in_buf[2], 12345);
}

TEST(relu_backward_negative) {
    /* Pre-activation negative: gradient blocked */
    fixed_t pre_buf[3] = {-FIXED_ONE, -1, 0};
    fixed_hp_t grad_out_buf[3] = {CT_GRAD_ONE, CT_GRAD_HALF, 12345};
    fixed_hp_t grad_in_buf[3] = {999, 999, 999};
    
    ct_tensor_t pre_act;
    ct_grad_tensor_t grad_out, grad_in;
    
    ct_tensor_init_1d(&pre_act, pre_buf, 3);
    ct_grad_tensor_init(&grad_out, grad_out_buf, 3, 0);
    ct_grad_tensor_init(&grad_in, grad_in_buf, 3, 0);
    
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_activation_relu_backward(&grad_out, &pre_act, &grad_in, &faults);
    ASSERT_EQ(err, CT_OK);
    
    ASSERT_EQ(grad_in_buf[0], 0);
    ASSERT_EQ(grad_in_buf[1], 0);
    ASSERT_EQ(grad_in_buf[2], 0);  /* Zero is treated as non-positive for ReLU */
}

TEST(sigmoid_backward) {
    /* σ'(x) = σ(x) * (1 - σ(x)) */
    /* At σ(x) = 0.5: σ'(x) = 0.5 * 0.5 = 0.25 */
    
    fixed_t act_buf[1] = {FIXED_HALF};  /* σ(x) = 0.5 */
    fixed_hp_t grad_out_buf[1] = {CT_GRAD_ONE};  /* upstream = 1.0 */
    fixed_hp_t grad_in_buf[1];
    
    ct_tensor_t activation;
    ct_grad_tensor_t grad_out, grad_in;
    
    ct_tensor_init_1d(&activation, act_buf, 1);
    ct_grad_tensor_init(&grad_out, grad_out_buf, 1, 0);
    ct_grad_tensor_init(&grad_in, grad_in_buf, 1, 0);
    
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_activation_sigmoid_backward(&grad_out, &activation, &grad_in, &faults);
    ASSERT_EQ(err, CT_OK);
    
    /* Expected: 1.0 * 0.25 = 0.25 in Q8.24 = 4194304 */
    int32_t expected = CT_GRAD_ONE / 4;
    int32_t tolerance = CT_GRAD_ONE / 100;
    ASSERT_NEAR_Q16(grad_in_buf[0], expected, tolerance);
}

TEST(tanh_backward) {
    /* tanh'(x) = 1 - tanh²(x) */
    /* At tanh(x) = 0: tanh'(x) = 1 - 0 = 1 */
    
    fixed_t act_buf[1] = {0};  /* tanh(x) = 0 */
    fixed_hp_t grad_out_buf[1] = {CT_GRAD_ONE};  /* upstream = 1.0 */
    fixed_hp_t grad_in_buf[1];
    
    ct_tensor_t activation;
    ct_grad_tensor_t grad_out, grad_in;
    
    ct_tensor_init_1d(&activation, act_buf, 1);
    ct_grad_tensor_init(&grad_out, grad_out_buf, 1, 0);
    ct_grad_tensor_init(&grad_in, grad_in_buf, 1, 0);
    
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_activation_tanh_backward(&grad_out, &activation, &grad_in, &faults);
    ASSERT_EQ(err, CT_OK);
    
    /* Expected: 1.0 * 1.0 = 1.0 in Q8.24 */
    int32_t tolerance = CT_GRAD_ONE / 100;
    ASSERT_NEAR_Q16(grad_in_buf[0], CT_GRAD_ONE, tolerance);
}

/* ============================================================================
 * Test: Linear Layer Backward
 * ============================================================================ */

TEST(linear_grad_init) {
    fixed_hp_t weight_buf[6];
    fixed_hp_t bias_buf[2];
    ct_linear_grad_t grad;
    
    ct_error_t err = ct_linear_grad_init(&grad, weight_buf, bias_buf, NULL, 3, 2);
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(grad.input_size, 3);
    ASSERT_EQ(grad.output_size, 2);
    ASSERT_EQ(grad.grad_weights.total_size, 6);
    ASSERT_EQ(grad.grad_bias.total_size, 2);
}

TEST(linear_backward_bias_gradient) {
    /* Simple test: bias gradient should equal output gradient */
    fixed_t weight_buf[4] = {FIXED_ONE, 0, 0, FIXED_ONE};  /* Identity */
    fixed_t bias_buf[2] = {0, 0};
    fixed_t input_buf[2] = {FIXED_ONE, FIXED_ONE};
    
    ct_linear_t layer;
    ct_tensor_init_2d(&layer.weights, weight_buf, 2, 2);
    ct_tensor_init_1d(&layer.bias, bias_buf, 2);
    layer.input_size = 2;
    layer.output_size = 2;
    
    ct_tensor_t input_cache;
    ct_tensor_init_1d(&input_cache, input_buf, 2);
    
    fixed_hp_t grad_weight_buf[4];
    fixed_hp_t grad_bias_buf[2];
    ct_linear_grad_t grad;
    ct_linear_grad_init(&grad, grad_weight_buf, grad_bias_buf, &input_cache, 2, 2);
    
    fixed_hp_t grad_out_buf[2] = {CT_GRAD_ONE, CT_GRAD_HALF};
    ct_grad_tensor_t grad_output;
    ct_grad_tensor_init(&grad_output, grad_out_buf, 2, 0);
    
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_linear_backward(&layer, &grad, &grad_output, NULL, &faults);
    ASSERT_EQ(err, CT_OK);
    
    /* Bias gradients should match output gradients */
    ASSERT_EQ(ct_grad_get_1d(&grad.grad_bias, 0), CT_GRAD_ONE);
    ASSERT_EQ(ct_grad_get_1d(&grad.grad_bias, 1), CT_GRAD_HALF);
}

/* ============================================================================
 * Test: Gradient Processing
 * ============================================================================ */

TEST(grad_clip) {
    fixed_hp_t buffer[5] = {-1000000000, -100, 0, 100, 1000000000};
    ct_grad_tensor_t grad;
    ct_grad_tensor_init(&grad, buffer, 5, 0);
    
    fixed_hp_t min_val = -500;
    fixed_hp_t max_val = 500;
    ct_fault_flags_t faults = {0};
    
    uint32_t clipped = ct_grad_clip(&grad, min_val, max_val, &faults);
    
    ASSERT_EQ(clipped, 2);  /* Two values should be clipped */
    ASSERT_EQ(buffer[0], -500);
    ASSERT_EQ(buffer[1], -100);
    ASSERT_EQ(buffer[2], 0);
    ASSERT_EQ(buffer[3], 100);
    ASSERT_EQ(buffer[4], 500);
}

TEST(grad_scale) {
    fixed_hp_t buffer[3] = {CT_GRAD_ONE, CT_GRAD_HALF, 0};
    ct_grad_tensor_t grad;
    ct_grad_tensor_init(&grad, buffer, 3, 0);
    
    ct_fault_flags_t faults = {0};
    
    /* Scale by 0.5 (GRAD_HALF) */
    ct_grad_scale(&grad, CT_GRAD_HALF, &faults);
    
    /* 1.0 * 0.5 = 0.5 */
    int32_t tolerance = CT_GRAD_ONE / 100;
    ASSERT_NEAR_Q16(buffer[0], CT_GRAD_HALF, tolerance);
    /* 0.5 * 0.5 = 0.25 */
    ASSERT_NEAR_Q16(buffer[1], CT_GRAD_ONE / 4, tolerance);
    ASSERT_EQ(buffer[2], 0);
}

TEST(grad_norm) {
    /* Vector [3, 4]: norm = sqrt(9 + 16) = 5 */
    fixed_hp_t buffer[2] = {3 * CT_GRAD_ONE, 4 * CT_GRAD_ONE};
    ct_grad_tensor_t grad;
    ct_grad_tensor_init(&grad, buffer, 2, 0);
    
    fixed_hp_t norm;
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_grad_norm(&grad, &norm, &faults);
    ASSERT_EQ(err, CT_OK);
    
    /* Expected: 5.0 in Q8.24 = 5 * 16777216 = 83886080 */
    int32_t expected = 5 * CT_GRAD_ONE;
    int32_t tolerance = CT_GRAD_ONE / 10;  /* 10% tolerance for sqrt approximation */
    ASSERT_NEAR_Q16(norm, expected, tolerance);
}

/* ============================================================================
 * Test: Gradient Health Monitoring
 * ============================================================================ */

TEST(grad_health_init) {
    ct_grad_health_t health;
    ct_grad_health_init(&health);
    
    ASSERT_EQ(health.zero_grad_count, 0);
    ASSERT_EQ(health.total_grad_count, 0);
    ASSERT_EQ(health.min_nonzero_grad, INT32_MAX);
    ASSERT_EQ(health.max_grad, 0);
}

TEST(grad_health_update) {
    ct_grad_health_t health;
    ct_grad_health_init(&health);
    
    fixed_hp_t buffer[5] = {0, 100, 200, 0, 50};
    ct_grad_tensor_t grad;
    ct_grad_tensor_init(&grad, buffer, 5, 0);
    
    ct_grad_health_update(&health, &grad);
    
    ASSERT_EQ(health.total_grad_count, 5);
    ASSERT_EQ(health.zero_grad_count, 2);
    ASSERT_EQ(health.min_nonzero_grad, 50);
    ASSERT_EQ(health.max_grad, 200);
}

TEST(grad_health_vanishing_detection) {
    ct_grad_health_t health;
    ct_grad_health_init(&health);
    
    /* 10% zeros should trigger warning (threshold is 5%) */
    health.total_grad_count = 100;
    health.zero_grad_count = 10;
    
    ASSERT(ct_grad_health_is_vanishing(&health));
    
    /* 3% zeros should be fine */
    health.zero_grad_count = 3;
    ASSERT(!ct_grad_health_is_vanishing(&health));
}

TEST(grad_health_zero_ratio) {
    ct_grad_health_t health;
    ct_grad_health_init(&health);
    
    health.total_grad_count = 4;
    health.zero_grad_count = 1;
    
    /* 25% = 0.25 in Q16.16 = 16384 */
    fixed_t ratio = ct_grad_health_zero_ratio(&health);
    int32_t expected = FIXED_ONE / 4;
    int32_t tolerance = FIXED_ONE / 100;
    
    ASSERT_NEAR_Q16(ratio, expected, tolerance);
}

/* ============================================================================
 * Test: Backward Context
 * ============================================================================ */

TEST(backward_config_default) {
    ct_backward_config_t config = ct_backward_config_default();
    
    ASSERT_EQ(config.grad_clip_max, CT_GRAD_CLIP_DEFAULT);
    ASSERT_EQ(config.grad_clip_min, -CT_GRAD_CLIP_DEFAULT);
    ASSERT(config.enable_grad_health);
    ASSERT_EQ(config.batch_size, 1);
}

TEST(backward_ctx_init) {
    ct_backward_ctx_t ctx;
    ct_backward_config_t config = ct_backward_config_default();
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_backward_ctx_init(&ctx, &config, &faults);
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(ctx.faults, &faults);
    ASSERT_EQ(ctx.health.total_grad_count, 0);
}

/* ============================================================================
 * Test: Error Handling
 * ============================================================================ */

TEST(null_pointer_handling) {
    ct_fault_flags_t faults = {0};
    
    ASSERT_EQ(ct_grad_tensor_init(NULL, NULL, 1, 0), CT_ERR_NULL);
    ASSERT_EQ(ct_loss_mse_forward(NULL, NULL, NULL, &faults), CT_ERR_NULL);
    ASSERT_EQ(ct_loss_mse_backward(NULL, NULL, NULL, &faults), CT_ERR_NULL);
    ASSERT_EQ(ct_activation_relu_backward(NULL, NULL, NULL, &faults), CT_ERR_NULL);
    ASSERT_EQ(ct_linear_grad_init(NULL, NULL, NULL, NULL, 1, 1), CT_ERR_NULL);
    ASSERT_EQ(ct_grad_norm(NULL, NULL, &faults), CT_ERR_NULL);
    ASSERT_EQ(ct_backward_ctx_init(NULL, NULL, NULL), CT_ERR_NULL);
}

TEST(dimension_mismatch) {
    fixed_t out_buf[2] = {0, 0};
    fixed_t tgt_buf[3] = {0, 0, 0};
    
    ct_tensor_t output, target;
    ct_tensor_init_1d(&output, out_buf, 2);
    ct_tensor_init_1d(&target, tgt_buf, 3);
    
    fixed_t loss;
    ct_fault_flags_t faults = {0};
    
    ASSERT_EQ(ct_loss_mse_forward(&output, &target, &loss, &faults), CT_ERR_DIMENSION);
}

/* ============================================================================
 * Test: Determinism
 * ============================================================================ */

TEST(backward_determinism) {
    /* Run same computation twice, verify identical results */
    fixed_t out_buf[4] = {FIXED_ONE, FIXED_HALF, 0, -FIXED_HALF};
    fixed_t tgt_buf[4] = {FIXED_HALF, 0, FIXED_HALF, 0};
    fixed_hp_t grad_buf1[4], grad_buf2[4];
    
    ct_tensor_t output, target;
    ct_grad_tensor_t grad1, grad2;
    
    ct_tensor_init_1d(&output, out_buf, 4);
    ct_tensor_init_1d(&target, tgt_buf, 4);
    ct_grad_tensor_init(&grad1, grad_buf1, 4, 0);
    ct_grad_tensor_init(&grad2, grad_buf2, 4, 0);
    
    ct_fault_flags_t faults1 = {0}, faults2 = {0};
    
    /* First run */
    ct_loss_mse_backward(&output, &target, &grad1, &faults1);
    
    /* Second run */
    ct_loss_mse_backward(&output, &target, &grad2, &faults2);
    
    /* Must be bit-identical */
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(grad_buf1[i], grad_buf2[i]);
    }
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== SRS-006: Backward Pass Tests ===\n\n");
    
    printf("Gradient Tensor Tests:\n");
    RUN_TEST(grad_tensor_init_1d);
    RUN_TEST(grad_tensor_init_2d);
    RUN_TEST(grad_tensor_zero);
    RUN_TEST(grad_tensor_accessors);
    
    printf("\nFormat Conversion Tests:\n");
    RUN_TEST(fixed_to_grad_conversion);
    RUN_TEST(grad_to_fixed_conversion);
    RUN_TEST(conversion_roundtrip);
    
    printf("\nMSE Loss Tests:\n");
    RUN_TEST(mse_loss_forward_zero);
    RUN_TEST(mse_loss_forward_nonzero);
    RUN_TEST(mse_loss_backward);
    
    printf("\nActivation Derivative Tests:\n");
    RUN_TEST(relu_backward_positive);
    RUN_TEST(relu_backward_negative);
    RUN_TEST(sigmoid_backward);
    RUN_TEST(tanh_backward);
    
    printf("\nLinear Layer Backward Tests:\n");
    RUN_TEST(linear_grad_init);
    RUN_TEST(linear_backward_bias_gradient);
    
    printf("\nGradient Processing Tests:\n");
    RUN_TEST(grad_clip);
    RUN_TEST(grad_scale);
    RUN_TEST(grad_norm);
    
    printf("\nGradient Health Tests:\n");
    RUN_TEST(grad_health_init);
    RUN_TEST(grad_health_update);
    RUN_TEST(grad_health_vanishing_detection);
    RUN_TEST(grad_health_zero_ratio);
    
    printf("\nBackward Context Tests:\n");
    RUN_TEST(backward_config_default);
    RUN_TEST(backward_ctx_init);
    
    printf("\nError Handling Tests:\n");
    RUN_TEST(null_pointer_handling);
    RUN_TEST(dimension_mismatch);
    
    printf("\nDeterminism Tests:\n");
    RUN_TEST(backward_determinism);
    
    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    
    return tests_failed > 0 ? 1 : 0;
}
