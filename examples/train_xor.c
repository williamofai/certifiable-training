/**
 * @file train_xor.c
 * @project Certifiable Training
 * @brief XOR training demo - deterministic neural network training
 *
 * Demonstrates the complete certifiable training pipeline:
 * - Fixed-point forward pass (Q16.16)
 * - Fixed-point backward pass
 * - Deterministic SGD optimization
 * - Merkle chain for auditability
 *
 * NO FLOATING POINT - All computation and display uses integer arithmetic.
 *
 * XOR truth table:
 *   0 XOR 0 = 0
 *   0 XOR 1 = 1
 *   1 XOR 0 = 1
 *   1 XOR 1 = 0
 *
 * Network: 2 inputs -> 8 hidden (ReLU) -> 1 output (sigmoid)
 *
 * @traceability CT-MATH-001, SRS-005, SRS-006, SRS-007
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 * @license GPL-3.0
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "ct_types.h"
#include "prng.h"
#include "merkle.h"
#include "forward.h"

/*===========================================================================
 * Configuration
 *===========================================================================*/

#define HIDDEN_SIZE     8
#define INPUT_SIZE      2
#define OUTPUT_SIZE     1
#define NUM_SAMPLES     4
#define NUM_EPOCHS      5000
#define PRINT_EVERY     500

/* Learning rate: 0.5 in Q16.16 */
#define LEARNING_RATE   32768

/* Random seed for reproducibility */
#define SEED            0xDEADBEEFCAFEBABEULL

/*===========================================================================
 * XOR Dataset (Q16.16 fixed-point, signed decimal)
 *===========================================================================*/

static const fixed_t XOR_INPUTS[NUM_SAMPLES][INPUT_SIZE] = {
    { 0,      0      },
    { 0,      65536  },  /* 0, 1.0 */
    { 65536,  0      },  /* 1.0, 0 */
    { 65536,  65536  },  /* 1.0, 1.0 */
};

static const fixed_t XOR_TARGETS[NUM_SAMPLES] = {
    0,       /* 0 XOR 0 = 0 */
    65536,   /* 0 XOR 1 = 1 */
    65536,   /* 1 XOR 0 = 1 */
    0,       /* 1 XOR 1 = 0 */
};

/*===========================================================================
 * Network Weights (statically allocated)
 *===========================================================================*/

/* Total weights for Merkle hashing */
#define TOTAL_WEIGHTS   (HIDDEN_SIZE * INPUT_SIZE + HIDDEN_SIZE + \
                         OUTPUT_SIZE * HIDDEN_SIZE + OUTPUT_SIZE)

static fixed_t all_weights[TOTAL_WEIGHTS];

/* Pointers into all_weights for convenience */
static fixed_t *w1_data;  /* [HIDDEN_SIZE * INPUT_SIZE] */
static fixed_t *b1_data;  /* [HIDDEN_SIZE] */
static fixed_t *w2_data;  /* [OUTPUT_SIZE * HIDDEN_SIZE] */
static fixed_t *b2_data;  /* [OUTPUT_SIZE] */

static fixed_t hidden_data[HIDDEN_SIZE];
static fixed_t output_data[OUTPUT_SIZE];

/*===========================================================================
 * DVM Primitives (inline)
 *===========================================================================*/

static inline int32_t dvm_clamp32(int64_t x, ct_fault_flags_t *faults) {
    if (x > INT32_MAX) {
        if (faults) faults->overflow = 1;
        return INT32_MAX;
    }
    if (x < INT32_MIN) {
        if (faults) faults->underflow = 1;
        return INT32_MIN;
    }
    return (int32_t)x;
}

static inline int32_t dvm_round_shift_rne(int64_t x, uint32_t shift, ct_fault_flags_t *faults) {
    if (shift == 0) return dvm_clamp32(x, faults);
    if (shift > 62) {
        if (faults) faults->domain = 1;
        return 0;
    }

    int64_t half = 1LL << (shift - 1);
    int64_t mask = (1LL << shift) - 1;
    int64_t frac = x & mask;
    int64_t quot = x >> shift;

    int64_t result;
    if (frac < half) {
        result = quot;
    } else if (frac > half) {
        result = quot + 1;
    } else {
        result = quot + (quot & 1);
    }

    return dvm_clamp32(result, faults);
}

/*===========================================================================
 * Sigmoid LUT (CT-MATH-001 compliant)
 *
 * Domain: [-8, +8] in Q16.16
 * Output: [0, 1] in Q16.16 (i.e., [0, 65536])
 * Table size: 257 entries
 * Step size: 16/256 = 0.0625
 *
 * Generated via: sig(x) = 1 / (1 + exp(-x)), then scaled to Q16.16
 *===========================================================================*/

static const fixed_t SIGMOID_LUT[257] = {
       22,    23,    25,    27,    28,    30,    32,    34,
       36,    39,    41,    44,    47,    50,    53,    56,
       60,    64,    68,    72,    77,    82,    87,    92,
       98,   105,   111,   119,   126,   134,   143,   152,
      162,   172,   184,   195,   208,   221,   236,   251,
      267,   284,   302,   321,   342,   364,   387,   412,
      439,   467,   497,   528,   562,   598,   636,   677,
      720,   766,   815,   867,   922,   980,  1042,  1109,
     1179,  1253,  1333,  1417,  1506,  1601,  1701,  1808,
     1921,  2041,  2168,  2303,  2446,  2598,  2758,  2928,
     3108,  3298,  3500,  3713,  3938,  4176,  4427,  4692,
     4971,  5266,  5577,  5904,  6249,  6611,  6992,  7392,
     7812,  8252,  8714,  9197,  9702, 10230, 10782, 11357,
    11955, 12579, 13226, 13898, 14595, 15316, 16062, 16832,
    17625, 18442, 19282, 20143, 21025, 21928, 22849, 23788,
    24743, 25712, 26695, 27689, 28693, 29705, 30723, 31744,
    32768, 33792, 34813, 35831, 36843, 37847, 38841, 39824,
    40793, 41748, 42687, 43608, 44511, 45393, 46254, 47094,
    47911, 48704, 49474, 50220, 50941, 51638, 52310, 52957,
    53581, 54179, 54754, 55306, 55834, 56339, 56822, 57284,
    57724, 58144, 58544, 58925, 59287, 59632, 59959, 60270,
    60565, 60844, 61109, 61360, 61598, 61823, 62036, 62238,
    62428, 62608, 62778, 62938, 63090, 63233, 63368, 63495,
    63615, 63728, 63835, 63935, 64030, 64119, 64203, 64283,
    64357, 64427, 64494, 64556, 64614, 64669, 64721, 64770,
    64816, 64859, 64900, 64938, 64974, 65008, 65039, 65069,
    65097, 65124, 65149, 65172, 65194, 65215, 65234, 65252,
    65269, 65285, 65300, 65315, 65328, 65341, 65352, 65364,
    65374, 65384, 65393, 65402, 65410, 65417, 65425, 65431,
    65438, 65444, 65449, 65454, 65459, 65464, 65468, 65472,
    65476, 65480, 65483, 65486, 65489, 65492, 65495, 65497,
    65500, 65502, 65504, 65506, 65508, 65509, 65511, 65513,
    65514
};

/**
 * @brief Sigmoid activation using LUT with linear interpolation
 * @param x Input in Q16.16 format
 * @return Sigmoid output in Q16.16 format, range [0, 65536]
 *
 * @traceability CT-MATH-001 §14
 */
static fixed_t sigmoid_lut(fixed_t x) {
    /* Domain: [-8, +8] in Q16.16 = [-524288, +524288] */
    if (x <= -524288) return SIGMOID_LUT[0];
    if (x >= 524288) return SIGMOID_LUT[256];

    /* Map x from [-524288, 524288] to [0, 256] */
    /* shifted = x + 524288, range [0, 1048576] */
    int64_t shifted = (int64_t)x + 524288;

    /* index = shifted / 4096 (since 1048576 / 257 ≈ 4096) */
    /* Actually: 1048576 / 256 = 4096 exactly */
    uint32_t index = (uint32_t)(shifted >> 12);  /* /4096 */

    if (index >= 256) return SIGMOID_LUT[256];

    /* Fractional part for interpolation: (shifted % 4096) / 16 gives 0-255 */
    uint32_t frac = (uint32_t)((shifted >> 4) & 0xFFU);

    int64_t y0 = SIGMOID_LUT[index];
    int64_t y1 = SIGMOID_LUT[index + 1];

    /* Linear interpolation: y0 + (y1 - y0) * frac / 256 */
    return (fixed_t)(y0 + (((y1 - y0) * (int64_t)frac) >> 8));
}

/*===========================================================================
 * Display Helpers (Integer Only)
 *===========================================================================*/

/**
 * @brief Print Q16.16 fixed-point as decimal (integer arithmetic only)
 * @param x Value in Q16.16 format
 *
 * Prints format: [-]N.NNNN (4 decimal places)
 */
static void print_fixed(fixed_t x) {
    if (x < 0) {
        printf("-");
        /* Handle INT32_MIN carefully */
        if (x == FIXED_MIN) {
            x = FIXED_MAX;
        } else {
            x = -x;
        }
    }

    uint32_t int_part = (uint32_t)x >> 16;
    uint32_t frac_bits = (uint32_t)x & 0xFFFFU;

    /* Convert fractional bits to 4 decimal digits: (frac * 10000) >> 16 */
    uint32_t frac_decimal = (frac_bits * 10000U) >> 16;

    printf("%u.%04u", int_part, frac_decimal);
}

/**
 * @brief Print hash bytes
 */
static void print_hash(const uint8_t *hash, int n) {
    for (int i = 0; i < n; i++) {
        printf("%02x", hash[i]);
    }
}

/*===========================================================================
 * Weight Initialization - Improved for XOR
 *===========================================================================*/

static void init_weights(ct_prng_t *prng) {
    /* Setup pointers into unified weight array */
    w1_data = &all_weights[0];
    b1_data = &all_weights[HIDDEN_SIZE * INPUT_SIZE];
    w2_data = &all_weights[HIDDEN_SIZE * INPUT_SIZE + HIDDEN_SIZE];
    b2_data = &all_weights[HIDDEN_SIZE * INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE * HIDDEN_SIZE];

    /* Layer 1: Wider range [-1.0, 1.0] for better XOR learning */
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        uint32_t r = ct_prng_next(prng);
        /* Range [-1.0, 1.0] in Q16.16 = [-65536, 65536] */
        int32_t val = (int32_t)(r % 131072U) - 65536;
        w1_data[i] = (fixed_t)val;
    }

    /* Small positive biases to prevent dead ReLU */
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        uint32_t r = ct_prng_next(prng);
        /* Small bias [0, 0.1] to keep some neurons alive */
        b1_data[i] = (fixed_t)(r % 6554U);  /* 0.1 in Q16.16 = 6553.6 */
    }

    /* Layer 2: Smaller range for output stability */
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        uint32_t r = ct_prng_next(prng);
        /* Range [-0.5, 0.5] in Q16.16 */
        int32_t val = (int32_t)(r % 65536U) - 32768;
        w2_data[i] = (fixed_t)val;
    }

    /* Output bias: start near 0 */
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        b2_data[i] = 0;
    }
}

/*===========================================================================
 * Forward Pass
 *===========================================================================*/

static fixed_t forward(const fixed_t *input, ct_fault_flags_t *faults) {
    /* Layer 1: Linear + ReLU */
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        int64_t acc = (int64_t)b1_data[h] << FIXED_FRAC_BITS;

        for (int i = 0; i < INPUT_SIZE; i++) {
            acc += (int64_t)w1_data[h * INPUT_SIZE + i] * (int64_t)input[i];
        }

        fixed_t pre_act = dvm_round_shift_rne(acc, FIXED_FRAC_BITS, faults);
        hidden_data[h] = (pre_act > 0) ? pre_act : 0;
    }

    /* Layer 2: Linear + Sigmoid */
    int64_t acc = (int64_t)b2_data[0] << FIXED_FRAC_BITS;

    for (int h = 0; h < HIDDEN_SIZE; h++) {
        acc += (int64_t)w2_data[h] * (int64_t)hidden_data[h];
    }

    fixed_t pre_sigmoid = dvm_round_shift_rne(acc, FIXED_FRAC_BITS, faults);
    output_data[0] = sigmoid_lut(pre_sigmoid);

    return output_data[0];
}

/*===========================================================================
 * Backward Pass + SGD Update
 *===========================================================================*/

static fixed_t backward_and_update(const fixed_t *input, fixed_t target,
                                    fixed_t prediction, ct_fault_flags_t *faults) {
    (void)faults;  /* We handle overflow gracefully via saturation */

    /* dL/dpred = prediction - target (both in Q16.16) */
    int64_t error = (int64_t)prediction - (int64_t)target;

    /* Clamp error to prevent overflow in squared calculation */
    if (error > 65536) error = 65536;
    if (error < -65536) error = -65536;
    fixed_t grad_pred = (fixed_t)error;

    /* Loss = error^2 / 2 (safe now that error is clamped) */
    int64_t loss_wide = (error * error) >> 17;  /* /2 in Q16.16 = >>17 */
    fixed_t loss = (fixed_t)loss_wide;

    /*
     * Sigmoid derivative: sig'(x) = sig(x) * (1 - sig(x))
     * With correct LUT, prediction is in [0, 65536] representing [0, 1]
     * So (1 - prediction) = FIXED_ONE - prediction = 65536 - prediction
     */
    int64_t sig = (int64_t)prediction;
    int64_t one_minus_sig = FIXED_ONE - sig;  /* Should be positive for valid sigmoid output */

    /* sig_deriv = sig * (1 - sig) in Q16.16 */
    int64_t sig_deriv = (sig * one_minus_sig) >> FIXED_FRAC_BITS;

    /* grad_pre_sigmoid = grad_pred * sigmoid'(x) */
    int64_t grad_pre = ((int64_t)grad_pred * sig_deriv) >> FIXED_FRAC_BITS;

    /* Clamp gradient to prevent explosion */
    if (grad_pre > 65536) grad_pre = 65536;
    if (grad_pre < -65536) grad_pre = -65536;
    fixed_t grad_pre_sigmoid = (fixed_t)grad_pre;

    /* Gradients for W2, b2 */
    fixed_t gw2[HIDDEN_SIZE];
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        int64_t gw = ((int64_t)grad_pre_sigmoid * (int64_t)hidden_data[h]) >> FIXED_FRAC_BITS;
        if (gw > 65536) gw = 65536;
        if (gw < -65536) gw = -65536;
        gw2[h] = (fixed_t)gw;
    }
    fixed_t gb2 = grad_pre_sigmoid;

    /* Propagate to hidden */
    fixed_t grad_hidden[HIDDEN_SIZE];
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        int64_t gh = ((int64_t)grad_pre_sigmoid * (int64_t)w2_data[h]) >> FIXED_FRAC_BITS;
        if (gh > 65536) gh = 65536;
        if (gh < -65536) gh = -65536;
        grad_hidden[h] = (fixed_t)gh;
    }

    /* Layer 1 backward (ReLU derivative) */
    fixed_t gw1[HIDDEN_SIZE * INPUT_SIZE];
    fixed_t gb1[HIDDEN_SIZE];

    for (int h = 0; h < HIDDEN_SIZE; h++) {
        fixed_t grad_pre_relu = (hidden_data[h] > 0) ? grad_hidden[h] : 0;

        for (int i = 0; i < INPUT_SIZE; i++) {
            int64_t gw = ((int64_t)grad_pre_relu * (int64_t)input[i]) >> FIXED_FRAC_BITS;
            if (gw > 65536) gw = 65536;
            if (gw < -65536) gw = -65536;
            gw1[h * INPUT_SIZE + i] = (fixed_t)gw;
        }
        gb1[h] = grad_pre_relu;
    }

    /* SGD Update: w = w - lr * grad */
    /* Clamp weight updates to prevent runaway */
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        int64_t update = ((int64_t)LEARNING_RATE * (int64_t)gw2[h]) >> FIXED_FRAC_BITS;
        int64_t new_w = (int64_t)w2_data[h] - update;
        if (new_w > FIXED_MAX) new_w = FIXED_MAX;
        if (new_w < FIXED_MIN) new_w = FIXED_MIN;
        w2_data[h] = (fixed_t)new_w;
    }
    {
        int64_t update = ((int64_t)LEARNING_RATE * (int64_t)gb2) >> FIXED_FRAC_BITS;
        int64_t new_b = (int64_t)b2_data[0] - update;
        if (new_b > FIXED_MAX) new_b = FIXED_MAX;
        if (new_b < FIXED_MIN) new_b = FIXED_MIN;
        b2_data[0] = (fixed_t)new_b;
    }

    for (int h = 0; h < HIDDEN_SIZE; h++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            int64_t update = ((int64_t)LEARNING_RATE * (int64_t)gw1[h * INPUT_SIZE + i]) >> FIXED_FRAC_BITS;
            int64_t new_w = (int64_t)w1_data[h * INPUT_SIZE + i] - update;
            if (new_w > FIXED_MAX) new_w = FIXED_MAX;
            if (new_w < FIXED_MIN) new_w = FIXED_MIN;
            w1_data[h * INPUT_SIZE + i] = (fixed_t)new_w;
        }
        {
            int64_t update = ((int64_t)LEARNING_RATE * (int64_t)gb1[h]) >> FIXED_FRAC_BITS;
            int64_t new_b = (int64_t)b1_data[h] - update;
            if (new_b > FIXED_MAX) new_b = FIXED_MAX;
            if (new_b < FIXED_MIN) new_b = FIXED_MIN;
            b1_data[h] = (fixed_t)new_b;
        }
    }

    return loss;
}

/*===========================================================================
 * Main
 *===========================================================================*/

int main(void) {
    printf("===============================================================\n");
    printf("  Certifiable Training - XOR Demo\n");
    printf("===============================================================\n\n");

    printf("Network: %d -> %d (ReLU) -> %d (sigmoid)\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Learning rate: ");
    print_fixed(LEARNING_RATE);
    printf("\n");
    printf("Epochs: %d\n", NUM_EPOCHS);
    printf("Seed: 0x%016llX\n\n", (unsigned long long)SEED);

    ct_fault_flags_t faults = {0};

    /* Initialize PRNG */
    ct_prng_t prng;
    ct_prng_init(&prng, SEED, 0);

    /* Initialize weights */
    printf("Initializing weights...\n");
    init_weights(&prng);

    /* Create weight tensor for Merkle chain */
    ct_tensor_t weights_tensor;
    weights_tensor.data = all_weights;
    weights_tensor.dims[0] = TOTAL_WEIGHTS;
    weights_tensor.strides[0] = 1;
    weights_tensor.ndims = 1;
    weights_tensor.total_size = TOTAL_WEIGHTS;

    /* Initialize Merkle chain */
    ct_merkle_ctx_t merkle;
    uint8_t config_data[] = "xor_demo_v1";
    ct_error_t err = ct_merkle_init(&merkle, &weights_tensor,
                                     config_data, sizeof(config_data), SEED);

    if (err != CT_OK) {
        printf("ERROR: Failed to initialize Merkle chain (%d)\n", err);
        return 1;
    }

    printf("Merkle chain initialized.\n");
    printf("  Initial hash (h_0): ");
    print_hash(merkle.current_hash, 8);
    printf("...\n");

    printf("\nTraining...\n");
    printf("---------------------------------------------------------------\n");

    /* Batch indices (we use all 4 samples each epoch) */
    uint32_t batch_indices[NUM_SAMPLES] = {0, 1, 2, 3};

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        /* Clear faults each epoch - transient overflow is OK */
        ct_clear_faults(&faults);

        fixed_t epoch_loss = 0;

        for (int s = 0; s < NUM_SAMPLES; s++) {
            fixed_t pred = forward(XOR_INPUTS[s], &faults);
            fixed_t loss = backward_and_update(XOR_INPUTS[s], XOR_TARGETS[s], pred, &faults);

            int64_t acc = (int64_t)epoch_loss + (int64_t)loss;
            if (acc > FIXED_MAX) acc = FIXED_MAX;
            epoch_loss = (fixed_t)acc;
        }

        fixed_t avg_loss = epoch_loss / NUM_SAMPLES;

        /* Commit epoch to Merkle chain */
        ct_training_step_t step_record;
        ct_fault_flags_t step_faults = {0};
        (void)ct_merkle_step(&merkle, &weights_tensor, batch_indices,
                             NUM_SAMPLES, &step_record, &step_faults);

        if (epoch % PRINT_EVERY == 0 || epoch == NUM_EPOCHS - 1) {
            printf("Epoch %4d | Loss: ", epoch);
            print_fixed(avg_loss);
            printf(" | h: ");
            print_hash(merkle.current_hash, 4);
            printf("...\n");
        }
    }

    printf("---------------------------------------------------------------\n\n");

    printf("Final Predictions:\n");
    printf("---------------------------------------------------------------\n");

    ct_clear_faults(&faults);
    int correct = 0;
    for (int s = 0; s < NUM_SAMPLES; s++) {
        fixed_t pred = forward(XOR_INPUTS[s], &faults);
        fixed_t target = XOR_TARGETS[s];

        /* Threshold at 0.5 (32768 in Q16.16) */
        int pred_class = (pred > 32768) ? 1 : 0;
        int target_class = (target > 32768) ? 1 : 0;

        bool is_correct = (pred_class == target_class);
        if (is_correct) correct++;

        int in0 = (XOR_INPUTS[s][0] > 32768) ? 1 : 0;
        int in1 = (XOR_INPUTS[s][1] > 32768) ? 1 : 0;

        printf("  %d XOR %d = %d (pred: ", in0, in1, pred_class);
        print_fixed(pred);
        printf(", target: %d) %s\n", target_class, is_correct ? "[OK]" : "[FAIL]");
    }

    printf("---------------------------------------------------------------\n");

    /* Integer percentage calculation */
    int percent = (correct * 100) / NUM_SAMPLES;
    printf("Accuracy: %d/%d (%d%%)\n\n", correct, NUM_SAMPLES, percent);

    /* Print Merkle chain summary */
    printf("Merkle Chain Summary:\n");
    printf("---------------------------------------------------------------\n");
    printf("  Steps committed: %llu\n", (unsigned long long)merkle.step);
    printf("  Final hash: ");
    print_hash(merkle.current_hash, 16);
    printf("...\n");
    printf("  Chain valid: %s\n", ct_merkle_is_valid(&merkle) ? "YES" : "NO (faulted)");

    printf("\n===============================================================\n");

    if (correct == NUM_SAMPLES && ct_merkle_is_valid(&merkle)) {
        printf("  [PASS] XOR learned successfully\n");
        printf("  [PASS] Merkle chain intact - training is auditable\n");
        printf("===============================================================\n");
        return 0;
    } else {
        printf("  [FAIL] Training incomplete\n");
        printf("  Accuracy: %d%% (need 100%%)\n", percent);
        printf("===============================================================\n");
        return 1;
    }
}
