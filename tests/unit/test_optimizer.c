/**
 * @file test_optimizer.c
 * @project Certifiable Training
 * @brief Unit tests for optimizers (SGD, Momentum, Adam).
 *
 * @traceability SRS-007-OPTIMIZER
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
#include "optimizer.h"
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

#define ASSERT_NEAR(actual, expected, tolerance) do { \
    int32_t diff = ((actual) > (expected)) ? ((actual) - (expected)) : ((expected) - (actual)); \
    if (diff > (tolerance)) { \
        printf("FAIL\n    Expected: %d, Actual: %d, Diff: %d > %d\n", \
               (expected), (actual), diff, (tolerance)); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ============================================================================
 * Test: Square Root
 * ============================================================================ */

TEST(sqrt_zero) {
    ct_fault_flags_t faults = {0};
    fixed_t result = ct_opt_sqrt(0, &faults);
    ASSERT_EQ(result, 0);
}

TEST(sqrt_one) {
    ct_fault_flags_t faults = {0};
    /* sqrt(1.0) = 1.0 */
    fixed_t result = ct_opt_sqrt(FIXED_ONE, &faults);
    /* Allow small error */
    ASSERT_NEAR(result, FIXED_ONE, FIXED_ONE / 100);
}

TEST(sqrt_four) {
    ct_fault_flags_t faults = {0};
    /* sqrt(4.0) = 2.0 */
    fixed_t four = 4 * FIXED_ONE;
    fixed_t result = ct_opt_sqrt(four, &faults);
    fixed_t expected = 2 * FIXED_ONE;
    ASSERT_NEAR(result, expected, FIXED_ONE / 50);
}

TEST(sqrt_quarter) {
    ct_fault_flags_t faults = {0};
    /* sqrt(0.25) = 0.5 */
    fixed_t quarter = FIXED_ONE / 4;
    fixed_t result = ct_opt_sqrt(quarter, &faults);
    fixed_t expected = FIXED_HALF;
    ASSERT_NEAR(result, expected, FIXED_ONE / 50);
}

TEST(sqrt_negative) {
    ct_fault_flags_t faults = {0};
    /* sqrt(-1) should return 0 */
    fixed_t result = ct_opt_sqrt(-FIXED_ONE, &faults);
    ASSERT_EQ(result, 0);
}

/* ============================================================================
 * Test: SGD Configuration
 * ============================================================================ */

TEST(sgd_config_default) {
    ct_sgd_config_t config = ct_sgd_config_default();
    ASSERT_EQ(config.learning_rate, CT_OPT_DEFAULT_LR);
    ASSERT_EQ(config.weight_decay, 0);
}

TEST(sgd_init) {
    ct_sgd_t opt;
    ct_error_t err = ct_sgd_init(&opt, NULL);
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(opt.config.learning_rate, CT_OPT_DEFAULT_LR);
    ASSERT_EQ(opt.step, 0);
}

TEST(sgd_init_custom) {
    ct_sgd_t opt;
    ct_sgd_config_t config = {
        .learning_rate = FIXED_ONE / 100,  /* 0.01 */
        .weight_decay = FIXED_ONE / 1000   /* 0.001 */
    };
    ct_error_t err = ct_sgd_init(&opt, &config);
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(opt.config.learning_rate, config.learning_rate);
    ASSERT_EQ(opt.config.weight_decay, config.weight_decay);
}

/* ============================================================================
 * Test: SGD Step
 * ============================================================================ */

TEST(sgd_step_basic) {
    ct_sgd_t opt;
    ct_sgd_config_t config = {
        .learning_rate = FIXED_ONE / 10,  /* 0.1 */
        .weight_decay = 0
    };
    ct_sgd_init(&opt, &config);
    
    /* Parameter: 1.0 */
    fixed_t param_buf[1] = {FIXED_ONE};
    ct_tensor_t params;
    ct_tensor_init_1d(&params, param_buf, 1);
    
    /* Gradient: 1.0 (in Q8.24) */
    fixed_hp_t grad_buf[1] = {CT_GRAD_ONE};
    ct_grad_tensor_t grads;
    ct_grad_tensor_init(&grads, grad_buf, 1, 0);
    
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_sgd_step(&opt, &params, &grads, &faults);
    ASSERT_EQ(err, CT_OK);
    
    /* Expected: 1.0 - 0.1 * 1.0 = 0.9 */
    fixed_t expected = FIXED_ONE - FIXED_ONE / 10;
    ASSERT_NEAR(param_buf[0], expected, FIXED_ONE / 100);
    ASSERT_EQ(opt.step, 1);
}

TEST(sgd_step_with_weight_decay) {
    ct_sgd_t opt;
    ct_sgd_config_t config = {
        .learning_rate = FIXED_ONE / 10,  /* 0.1 */
        .weight_decay = FIXED_ONE / 10    /* 0.1 */
    };
    ct_sgd_init(&opt, &config);
    
    /* Parameter: 1.0 */
    fixed_t param_buf[1] = {FIXED_ONE};
    ct_tensor_t params;
    ct_tensor_init_1d(&params, param_buf, 1);
    
    /* Gradient: 0.0 (test weight decay only) */
    fixed_hp_t grad_buf[1] = {0};
    ct_grad_tensor_t grads;
    ct_grad_tensor_init(&grads, grad_buf, 1, 0);
    
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_sgd_step(&opt, &params, &grads, &faults);
    ASSERT_EQ(err, CT_OK);
    
    /* Expected: 1.0 - 0.1 * (0 + 0.1 * 1.0) = 1.0 - 0.01 = 0.99 */
    fixed_t expected = FIXED_ONE - FIXED_ONE / 100;
    ASSERT_NEAR(param_buf[0], expected, FIXED_ONE / 100);
}

TEST(sgd_multiple_steps) {
    ct_sgd_t opt;
    ct_sgd_config_t config = {
        .learning_rate = FIXED_ONE / 10,
        .weight_decay = 0
    };
    ct_sgd_init(&opt, &config);
    
    fixed_t param_buf[1] = {FIXED_ONE};
    ct_tensor_t params;
    ct_tensor_init_1d(&params, param_buf, 1);
    
    fixed_hp_t grad_buf[1] = {CT_GRAD_ONE / 10};  /* 0.1 gradient */
    ct_grad_tensor_t grads;
    ct_grad_tensor_init(&grads, grad_buf, 1, 0);
    
    ct_fault_flags_t faults = {0};
    
    /* Run 3 steps */
    for (int i = 0; i < 3; i++) {
        ct_sgd_step(&opt, &params, &grads, &faults);
    }
    
    ASSERT_EQ(opt.step, 3);
    /* Should have decreased by 3 * 0.1 * 0.1 = 0.03 */
    ASSERT_LT(param_buf[0], FIXED_ONE);
}

/* ============================================================================
 * Test: SGD with Momentum
 * ============================================================================ */

TEST(sgd_momentum_config_default) {
    ct_sgd_momentum_config_t config = ct_sgd_momentum_config_default();
    ASSERT_EQ(config.learning_rate, CT_OPT_DEFAULT_LR);
    ASSERT_EQ(config.momentum, CT_OPT_DEFAULT_MOMENTUM);
    ASSERT_EQ(config.weight_decay, 0);
}

TEST(sgd_momentum_init) {
    ct_sgd_momentum_t opt;
    fixed_t vel_buf[4] = {0};
    
    ct_error_t err = ct_sgd_momentum_init(&opt, NULL, vel_buf, 4);
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(opt.num_params, 4);
    ASSERT_EQ(opt.step, 0);
    ASSERT(opt.initialized);
}

TEST(sgd_momentum_step) {
    ct_sgd_momentum_t opt;
    fixed_t vel_buf[1] = {0};
    ct_sgd_momentum_config_t config = {
        .learning_rate = FIXED_ONE / 10,
        .momentum = FIXED_HALF,  /* 0.5 */
        .weight_decay = 0
    };
    ct_sgd_momentum_init(&opt, &config, vel_buf, 1);
    
    fixed_t param_buf[1] = {FIXED_ONE};
    ct_tensor_t params;
    ct_tensor_init_1d(&params, param_buf, 1);
    
    fixed_hp_t grad_buf[1] = {CT_GRAD_ONE};
    ct_grad_tensor_t grads;
    ct_grad_tensor_init(&grads, grad_buf, 1, 0);
    
    ct_fault_flags_t faults = {0};
    
    /* First step: v = 0*0.5 + 1.0 = 1.0 */
    ct_error_t err = ct_sgd_momentum_step(&opt, &params, &grads, &faults);
    ASSERT_EQ(err, CT_OK);
    
    /* Velocity should be ~1.0 */
    ASSERT_NEAR(vel_buf[0], FIXED_ONE, FIXED_ONE / 50);
    
    /* Second step: v = 1.0*0.5 + 1.0 = 1.5 */
    ct_sgd_momentum_step(&opt, &params, &grads, &faults);
    
    /* Velocity should be ~1.5 */
    fixed_t expected_vel = FIXED_ONE + FIXED_HALF;
    ASSERT_NEAR(vel_buf[0], expected_vel, FIXED_ONE / 50);
}

TEST(sgd_momentum_accumulates) {
    ct_sgd_momentum_t opt;
    fixed_t vel_buf[1] = {0};
    ct_sgd_momentum_config_t config = {
        .learning_rate = FIXED_ONE / 100,
        .momentum = CT_OPT_DEFAULT_MOMENTUM,  /* 0.9 */
        .weight_decay = 0
    };
    ct_sgd_momentum_init(&opt, &config, vel_buf, 1);
    
    fixed_t param_buf[1] = {FIXED_ONE};
    ct_tensor_t params;
    ct_tensor_init_1d(&params, param_buf, 1);
    
    fixed_hp_t grad_buf[1] = {CT_GRAD_ONE / 10};  /* Constant gradient */
    ct_grad_tensor_t grads;
    ct_grad_tensor_init(&grads, grad_buf, 1, 0);
    
    ct_fault_flags_t faults = {0};
    
    fixed_t initial_param = param_buf[0];
    
    /* Run several steps */
    for (int i = 0; i < 10; i++) {
        ct_sgd_momentum_step(&opt, &params, &grads, &faults);
    }
    
    /* Parameter should have decreased */
    ASSERT_LT(param_buf[0], initial_param);
    
    /* Velocity should have accumulated */
    ASSERT_GT(vel_buf[0], FIXED_ONE / 10);
}

/* ============================================================================
 * Test: Adam
 * ============================================================================ */

TEST(adam_config_default) {
    ct_adam_config_t config = ct_adam_config_default();
    ASSERT_EQ(config.learning_rate, CT_OPT_DEFAULT_LR);
    ASSERT_EQ(config.beta1, CT_OPT_ADAM_BETA1);
    ASSERT_EQ(config.beta2, CT_OPT_ADAM_BETA2);
    ASSERT_EQ(config.epsilon, CT_OPT_ADAM_EPSILON);
}

TEST(adam_init) {
    ct_adam_t opt;
    fixed_t m_buf[4] = {0};
    fixed_t v_buf[4] = {0};
    
    ct_error_t err = ct_adam_init(&opt, NULL, m_buf, v_buf, 4);
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(opt.num_params, 4);
    ASSERT_EQ(opt.step, 0);
    ASSERT_EQ(opt.beta1_power, FIXED_ONE);
    ASSERT_EQ(opt.beta2_power, FIXED_ONE);
    ASSERT(opt.initialized);
}

TEST(adam_step_basic) {
    ct_adam_t opt;
    fixed_t m_buf[1] = {0};
    fixed_t v_buf[1] = {0};
    ct_adam_config_t config = {
        .learning_rate = FIXED_ONE / 100,  /* 0.01 */
        .beta1 = CT_OPT_ADAM_BETA1,
        .beta2 = CT_OPT_ADAM_BETA2,
        .epsilon = CT_OPT_ADAM_EPSILON,
        .weight_decay = 0
    };
    ct_adam_init(&opt, &config, m_buf, v_buf, 1);
    
    fixed_t param_buf[1] = {FIXED_ONE};
    ct_tensor_t params;
    ct_tensor_init_1d(&params, param_buf, 1);
    
    fixed_hp_t grad_buf[1] = {CT_GRAD_ONE};
    ct_grad_tensor_t grads;
    ct_grad_tensor_init(&grads, grad_buf, 1, 0);
    
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_adam_step(&opt, &params, &grads, &faults);
    ASSERT_EQ(err, CT_OK);
    ASSERT_EQ(opt.step, 1);
    
    /* First moment should have updated */
    ASSERT_NE(m_buf[0], 0);
    
    /* Second moment should have updated */
    ASSERT_NE(v_buf[0], 0);
    
    /* Parameter should have decreased (positive gradient) */
    ASSERT_LT(param_buf[0], FIXED_ONE);
}

TEST(adam_bias_correction) {
    ct_adam_t opt;
    fixed_t m_buf[1] = {0};
    fixed_t v_buf[1] = {0};
    ct_adam_init(&opt, NULL, m_buf, v_buf, 1);
    
    ct_fault_flags_t faults = {0};
    
    fixed_t param_buf[1] = {FIXED_ONE};
    ct_tensor_t params;
    ct_tensor_init_1d(&params, param_buf, 1);
    
    fixed_hp_t grad_buf[1] = {CT_GRAD_ONE / 10};
    ct_grad_tensor_t grads;
    ct_grad_tensor_init(&grads, grad_buf, 1, 0);
    
    /* Initial beta powers should be 1.0 */
    ASSERT_EQ(opt.beta1_power, FIXED_ONE);
    ASSERT_EQ(opt.beta2_power, FIXED_ONE);
    
    ct_adam_step(&opt, &params, &grads, &faults);
    
    /* After step, beta powers should have decreased */
    ASSERT_LT(opt.beta1_power, FIXED_ONE);
    ASSERT_LT(opt.beta2_power, FIXED_ONE);
}

TEST(adam_multiple_steps) {
    ct_adam_t opt;
    fixed_t m_buf[1] = {0};
    fixed_t v_buf[1] = {0};
    ct_adam_init(&opt, NULL, m_buf, v_buf, 1);
    
    fixed_t param_buf[1] = {FIXED_ONE};
    ct_tensor_t params;
    ct_tensor_init_1d(&params, param_buf, 1);
    
    fixed_hp_t grad_buf[1] = {CT_GRAD_ONE / 10};
    ct_grad_tensor_t grads;
    ct_grad_tensor_init(&grads, grad_buf, 1, 0);
    
    ct_fault_flags_t faults = {0};
    
    fixed_t initial = param_buf[0];
    
    for (int i = 0; i < 10; i++) {
        ct_adam_step(&opt, &params, &grads, &faults);
    }
    
    ASSERT_EQ(opt.step, 10);
    ASSERT_LT(param_buf[0], initial);
}

/* ============================================================================
 * Test: Reset Functions
 * ============================================================================ */

TEST(sgd_reset) {
    ct_sgd_t opt;
    ct_sgd_init(&opt, NULL);
    opt.step = 100;
    
    ct_sgd_reset(&opt);
    ASSERT_EQ(opt.step, 0);
}

TEST(sgd_momentum_reset) {
    ct_sgd_momentum_t opt;
    fixed_t vel_buf[2] = {FIXED_ONE, FIXED_ONE};
    ct_sgd_momentum_init(&opt, NULL, vel_buf, 2);
    opt.step = 50;
    
    ct_sgd_momentum_reset(&opt);
    ASSERT_EQ(opt.step, 0);
    ASSERT_EQ(vel_buf[0], 0);
    ASSERT_EQ(vel_buf[1], 0);
}

TEST(adam_reset) {
    ct_adam_t opt;
    fixed_t m_buf[2] = {FIXED_ONE, FIXED_ONE};
    fixed_t v_buf[2] = {FIXED_HALF, FIXED_HALF};
    ct_adam_init(&opt, NULL, m_buf, v_buf, 2);
    opt.step = 100;
    opt.beta1_power = FIXED_HALF;
    
    ct_adam_reset(&opt);
    ASSERT_EQ(opt.step, 0);
    ASSERT_EQ(opt.beta1_power, FIXED_ONE);
    ASSERT_EQ(opt.beta2_power, FIXED_ONE);
    ASSERT_EQ(m_buf[0], 0);
    ASSERT_EQ(v_buf[0], 0);
}

/* ============================================================================
 * Test: Error Handling
 * ============================================================================ */

TEST(null_pointer_handling) {
    ct_fault_flags_t faults = {0};
    
    ASSERT_EQ(ct_sgd_init(NULL, NULL), CT_ERR_NULL);
    ASSERT_EQ(ct_sgd_step(NULL, NULL, NULL, &faults), CT_ERR_NULL);
    ASSERT_EQ(ct_sgd_momentum_init(NULL, NULL, NULL, 0), CT_ERR_NULL);
    ASSERT_EQ(ct_sgd_momentum_step(NULL, NULL, NULL, &faults), CT_ERR_NULL);
    ASSERT_EQ(ct_adam_init(NULL, NULL, NULL, NULL, 0), CT_ERR_NULL);
    ASSERT_EQ(ct_adam_step(NULL, NULL, NULL, &faults), CT_ERR_NULL);
}

TEST(dimension_mismatch) {
    ct_sgd_t opt;
    ct_sgd_init(&opt, NULL);
    
    fixed_t param_buf[2] = {0, 0};
    ct_tensor_t params;
    ct_tensor_init_1d(&params, param_buf, 2);
    
    fixed_hp_t grad_buf[3] = {0, 0, 0};
    ct_grad_tensor_t grads;
    ct_grad_tensor_init(&grads, grad_buf, 3, 0);
    
    ct_fault_flags_t faults = {0};
    
    ASSERT_EQ(ct_sgd_step(&opt, &params, &grads, &faults), CT_ERR_DIMENSION);
}

/* ============================================================================
 * Test: Determinism
 * ============================================================================ */

TEST(sgd_determinism) {
    ct_sgd_t opt1, opt2;
    ct_sgd_config_t config = {
        .learning_rate = FIXED_ONE / 10,
        .weight_decay = FIXED_ONE / 100
    };
    ct_sgd_init(&opt1, &config);
    ct_sgd_init(&opt2, &config);
    
    fixed_t param1[2] = {FIXED_ONE, FIXED_HALF};
    fixed_t param2[2] = {FIXED_ONE, FIXED_HALF};
    ct_tensor_t params1, params2;
    ct_tensor_init_1d(&params1, param1, 2);
    ct_tensor_init_1d(&params2, param2, 2);
    
    fixed_hp_t grad_buf[2] = {CT_GRAD_ONE / 5, -CT_GRAD_ONE / 10};
    ct_grad_tensor_t grads;
    ct_grad_tensor_init(&grads, grad_buf, 2, 0);
    
    ct_fault_flags_t faults1 = {0}, faults2 = {0};
    
    /* Run same update on both */
    ct_sgd_step(&opt1, &params1, &grads, &faults1);
    ct_sgd_step(&opt2, &params2, &grads, &faults2);
    
    /* Must be bit-identical */
    ASSERT_EQ(param1[0], param2[0]);
    ASSERT_EQ(param1[1], param2[1]);
}

TEST(adam_determinism) {
    ct_adam_t opt1, opt2;
    fixed_t m1[2] = {0}, v1[2] = {0};
    fixed_t m2[2] = {0}, v2[2] = {0};
    
    ct_adam_init(&opt1, NULL, m1, v1, 2);
    ct_adam_init(&opt2, NULL, m2, v2, 2);
    
    fixed_t param1[2] = {FIXED_ONE, -FIXED_HALF};
    fixed_t param2[2] = {FIXED_ONE, -FIXED_HALF};
    ct_tensor_t params1, params2;
    ct_tensor_init_1d(&params1, param1, 2);
    ct_tensor_init_1d(&params2, param2, 2);
    
    fixed_hp_t grad_buf[2] = {CT_GRAD_ONE / 3, CT_GRAD_ONE / 7};
    ct_grad_tensor_t grads;
    ct_grad_tensor_init(&grads, grad_buf, 2, 0);
    
    ct_fault_flags_t faults1 = {0}, faults2 = {0};
    
    /* Run 5 steps on both */
    for (int i = 0; i < 5; i++) {
        ct_adam_step(&opt1, &params1, &grads, &faults1);
        ct_adam_step(&opt2, &params2, &grads, &faults2);
    }
    
    /* Must be bit-identical */
    ASSERT_EQ(param1[0], param2[0]);
    ASSERT_EQ(param1[1], param2[1]);
    ASSERT_EQ(m1[0], m2[0]);
    ASSERT_EQ(v1[0], v2[0]);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== SRS-007: Optimizer Tests ===\n\n");
    
    printf("Square Root Tests:\n");
    RUN_TEST(sqrt_zero);
    RUN_TEST(sqrt_one);
    RUN_TEST(sqrt_four);
    RUN_TEST(sqrt_quarter);
    RUN_TEST(sqrt_negative);
    
    printf("\nSGD Configuration Tests:\n");
    RUN_TEST(sgd_config_default);
    RUN_TEST(sgd_init);
    RUN_TEST(sgd_init_custom);
    
    printf("\nSGD Step Tests:\n");
    RUN_TEST(sgd_step_basic);
    RUN_TEST(sgd_step_with_weight_decay);
    RUN_TEST(sgd_multiple_steps);
    
    printf("\nSGD Momentum Tests:\n");
    RUN_TEST(sgd_momentum_config_default);
    RUN_TEST(sgd_momentum_init);
    RUN_TEST(sgd_momentum_step);
    RUN_TEST(sgd_momentum_accumulates);
    
    printf("\nAdam Tests:\n");
    RUN_TEST(adam_config_default);
    RUN_TEST(adam_init);
    RUN_TEST(adam_step_basic);
    RUN_TEST(adam_bias_correction);
    RUN_TEST(adam_multiple_steps);
    
    printf("\nReset Tests:\n");
    RUN_TEST(sgd_reset);
    RUN_TEST(sgd_momentum_reset);
    RUN_TEST(adam_reset);
    
    printf("\nError Handling Tests:\n");
    RUN_TEST(null_pointer_handling);
    RUN_TEST(dimension_mismatch);
    
    printf("\nDeterminism Tests:\n");
    RUN_TEST(sgd_determinism);
    RUN_TEST(adam_determinism);
    
    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    
    return tests_failed > 0 ? 1 : 0;
}
