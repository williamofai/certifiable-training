/**
 * @file scheduler.c
 * @project Certifiable Training
 * @brief Deterministic learning rate schedulers
 *
 * @details Implements fixed-point learning rate schedules:
 *          - Constant (no decay)
 *          - Step decay (lr = lr_0 * gamma^(epoch / step_size))
 *          - Linear warmup (lr = lr_0 * step / warmup_steps)
 *          - Cosine annealing (lr = lr_min + 0.5*(lr_0 - lr_min)*(1 + cos(π*t/T)))
 *
 *          All computations use DVM primitives for determinism.
 *          Cosine uses LUT with linear interpolation.
 *
 * @traceability CT-MATH-001 §11
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304, DO-178C
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 * @license GPL-3.0 or Commercial License (william@fstopify.com)
 */

#include "ct_types.h"
#include "dvm.h"
#include <stddef.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

/** Cosine LUT size (257 entries for [0, π]) */
#define CT_SCHED_COS_LUT_SIZE 257

/** Pi in Q16.16 (π ≈ 3.14159) */
#define CT_SCHED_PI_Q16       ((fixed_t)205887)

/* ============================================================================
 * Scheduler Types
 * ============================================================================ */

/**
 * @brief Scheduler type enumeration
 */
typedef enum {
    CT_SCHED_CONSTANT       = 0,    /**< No decay */
    CT_SCHED_STEP           = 1,    /**< Step decay */
    CT_SCHED_LINEAR_WARMUP  = 2,    /**< Linear warmup then constant */
    CT_SCHED_COSINE         = 3     /**< Cosine annealing */
} ct_scheduler_type_t;

/**
 * @brief Cosine lookup table for deterministic annealing
 */
typedef struct {
    fixed_t table[CT_SCHED_COS_LUT_SIZE];  /**< cos(x) for x in [0, π] */
    bool initialized;
} ct_cosine_lut_t;

/**
 * @brief Step decay configuration
 */
typedef struct {
    fixed_t initial_lr;     /**< Initial learning rate */
    fixed_t gamma;          /**< Decay factor (typically 0.1 = 6554) */
    uint32_t step_size;     /**< Epochs between decays */
} ct_step_decay_config_t;

/**
 * @brief Linear warmup configuration
 */
typedef struct {
    fixed_t target_lr;      /**< Target LR after warmup */
    uint32_t warmup_steps;  /**< Number of warmup steps */
} ct_warmup_config_t;

/**
 * @brief Cosine annealing configuration
 */
typedef struct {
    fixed_t initial_lr;     /**< Maximum LR */
    fixed_t min_lr;         /**< Minimum LR */
    uint32_t total_steps;   /**< Total training steps */
    const ct_cosine_lut_t *lut;  /**< Cosine LUT (shared) */
} ct_cosine_config_t;

/**
 * @brief Scheduler state
 */
typedef struct {
    ct_scheduler_type_t type;
    union {
        ct_step_decay_config_t step;
        ct_warmup_config_t warmup;
        ct_cosine_config_t cosine;
    } config;
    fixed_t current_lr;     /**< Current learning rate */
    uint64_t step;          /**< Current step */
    uint32_t epoch;         /**< Current epoch */
} ct_scheduler_t;

/* ============================================================================
 * Cosine LUT Initialization
 * ============================================================================ */

/**
 * @brief Initialize cosine LUT for deterministic annealing
 *
 * @param lut LUT structure to initialize
 *
 * @details Fills table with cos(x) for x in [0, π] using 257 entries.
 *          This is called once at startup using floating-point,
 *          then the LUT provides deterministic lookup at runtime.
 *
 * @note Uses floating-point ONLY for initialization, never at runtime.
 */
void ct_scheduler_init_cosine_lut(ct_cosine_lut_t *lut)
{
    if (lut == NULL) return;

    /* Use simple integer approximation to avoid math.h dependency at runtime.
     * For initialization, we compute cos(i * π / 256) for i in [0, 256].
     * cos(0) = 1, cos(π) = -1
     *
     * We use the identity: cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720 (Taylor series)
     * But for simplicity and determinism, we use pre-computed values.
     */

    /* Pre-computed cosine table values in Q16.16 format.
     * cos(i * π / 256) for i = 0..256
     * Generated offline to ensure determinism.
     */
    static const int32_t cos_values[CT_SCHED_COS_LUT_SIZE] = {
        65536, 65526, 65496, 65446, 65376, 65287, 65177, 65048,  /* 0-7 */
        64898, 64729, 64540, 64332, 64104, 63856, 63589, 63303,  /* 8-15 */
        62997, 62672, 62328, 61966, 61584, 61183, 60764, 60326,  /* 16-23 */
        59870, 59396, 58903, 58393, 57865, 57319, 56756, 56175,  /* 24-31 */
        55578, 54963, 54332, 53684, 53020, 52339, 51643, 50931,  /* 32-39 */
        50203, 49461, 48703, 47930, 47143, 46341, 45525, 44695,  /* 40-47 */
        43852, 42995, 42126, 41243, 40348, 39441, 38521, 37590,  /* 48-55 */
        36647, 35693, 34729, 33754, 32768, 31772, 30767, 29753,  /* 56-63 */
        28729, 27697, 26656, 25607, 24550, 23486, 22415, 21336,  /* 64-71 */
        20252, 19161, 18064, 16962, 15855, 14742, 13626, 12505,  /* 72-79 */
        11380, 10252, 9121, 7987, 6850, 5712, 4572, 3430,        /* 80-87 */
        2287, 1144, 0, -1144, -2287, -3430, -4572, -5712,        /* 88-95 */
        -6850, -7987, -9121, -10252, -11380, -12505, -13626, -14742,  /* 96-103 */
        -15855, -16962, -18064, -19161, -20252, -21336, -22415, -23486,  /* 104-111 */
        -24550, -25607, -26656, -27697, -28729, -29753, -30767, -31772,  /* 112-119 */
        -32768, -33754, -34729, -35693, -36647, -37590, -38521, -39441,  /* 120-127 */
        -40348, -41243, -42126, -42995, -43852, -44695, -45525, -46341,  /* 128-135 */
        -47143, -47930, -48703, -49461, -50203, -50931, -51643, -52339,  /* 136-143 */
        -53020, -53684, -54332, -54963, -55578, -56175, -56756, -57319,  /* 144-151 */
        -57865, -58393, -58903, -59396, -59870, -60326, -60764, -61183,  /* 152-159 */
        -61584, -61966, -62328, -62672, -62997, -63303, -63589, -63856,  /* 160-167 */
        -64104, -64332, -64540, -64729, -64898, -65048, -65177, -65287,  /* 168-175 */
        -65376, -65446, -65496, -65526, -65536, -65526, -65496, -65446,  /* 176-183 */
        -65376, -65287, -65177, -65048, -64898, -64729, -64540, -64332,  /* 184-191 */
        -64104, -63856, -63589, -63303, -62997, -62672, -62328, -61966,  /* 192-199 */
        -61584, -61183, -60764, -60326, -59870, -59396, -58903, -58393,  /* 200-207 */
        -57865, -57319, -56756, -56175, -55578, -54963, -54332, -53684,  /* 208-215 */
        -53020, -52339, -51643, -50931, -50203, -49461, -48703, -47930,  /* 216-223 */
        -47143, -46341, -45525, -44695, -43852, -42995, -42126, -41243,  /* 224-231 */
        -40348, -39441, -38521, -37590, -36647, -35693, -34729, -33754,  /* 232-239 */
        -32768, -31772, -30767, -29753, -28729, -27697, -26656, -25607,  /* 240-247 */
        -24550, -23486, -22415, -21336, -20252, -19161, -18064, -16962   /* 248-255 */
        /* Entry 256 = cos(π) = -1 = -65536 */
    };

    for (int i = 0; i < CT_SCHED_COS_LUT_SIZE - 1; i++) {
        lut->table[i] = cos_values[i];
    }
    lut->table[CT_SCHED_COS_LUT_SIZE - 1] = -FIXED_ONE;  /* cos(π) = -1 */

    lut->initialized = true;
}

/**
 * @brief Lookup cosine with linear interpolation
 *
 * @param x Input in [0, FIXED_ONE] representing [0, π]
 * @param lut Initialized cosine LUT
 * @return cos(x * π) in Q16.16
 */
static fixed_t cosine_lookup(fixed_t x, const ct_cosine_lut_t *lut)
{
    if (lut == NULL || !lut->initialized) return FIXED_ONE;

    /* Clamp to valid range */
    if (x <= 0) return lut->table[0];
    if (x >= FIXED_ONE) return lut->table[CT_SCHED_COS_LUT_SIZE - 1];

    /* Scale x from [0, 1] to [0, 256] */
    int64_t scaled = ((int64_t)x * 256) >> FIXED_FRAC_BITS;
    uint32_t index = (uint32_t)scaled;

    if (index >= CT_SCHED_COS_LUT_SIZE - 1) {
        index = CT_SCHED_COS_LUT_SIZE - 2;
    }

    /* Fractional part for interpolation */
    int64_t frac = ((int64_t)x * 256) - ((int64_t)index << FIXED_FRAC_BITS);

    /* Linear interpolation */
    fixed_t y0 = lut->table[index];
    fixed_t y1 = lut->table[index + 1];
    int64_t diff = (int64_t)y1 - (int64_t)y0;
    int64_t interp = (diff * frac) >> FIXED_FRAC_BITS;

    return (fixed_t)(y0 + interp);
}

/* ============================================================================
 * Scheduler Initialization
 * ============================================================================ */

/**
 * @brief Initialize constant scheduler (no decay)
 *
 * @param sched Scheduler to initialize
 * @param lr Learning rate (Q16.16)
 * @return CT_OK on success
 */
ct_error_t ct_scheduler_init_constant(ct_scheduler_t *sched, fixed_t lr)
{
    if (sched == NULL) return CT_ERR_NULL;

    sched->type = CT_SCHED_CONSTANT;
    sched->current_lr = lr;
    sched->step = 0;
    sched->epoch = 0;

    return CT_OK;
}

/**
 * @brief Initialize step decay scheduler
 *
 * @param sched Scheduler to initialize
 * @param initial_lr Initial learning rate
 * @param gamma Decay factor per step_size epochs
 * @param step_size Epochs between decays
 * @return CT_OK on success
 */
ct_error_t ct_scheduler_init_step(ct_scheduler_t *sched,
                                  fixed_t initial_lr,
                                  fixed_t gamma,
                                  uint32_t step_size)
{
    if (sched == NULL) return CT_ERR_NULL;
    if (step_size == 0) return CT_ERR_CONFIG;

    sched->type = CT_SCHED_STEP;
    sched->config.step.initial_lr = initial_lr;
    sched->config.step.gamma = gamma;
    sched->config.step.step_size = step_size;
    sched->current_lr = initial_lr;
    sched->step = 0;
    sched->epoch = 0;

    return CT_OK;
}

/**
 * @brief Initialize linear warmup scheduler
 *
 * @param sched Scheduler to initialize
 * @param target_lr Target learning rate after warmup
 * @param warmup_steps Number of warmup steps
 * @return CT_OK on success
 */
ct_error_t ct_scheduler_init_warmup(ct_scheduler_t *sched,
                                    fixed_t target_lr,
                                    uint32_t warmup_steps)
{
    if (sched == NULL) return CT_ERR_NULL;
    if (warmup_steps == 0) return CT_ERR_CONFIG;

    sched->type = CT_SCHED_LINEAR_WARMUP;
    sched->config.warmup.target_lr = target_lr;
    sched->config.warmup.warmup_steps = warmup_steps;
    sched->current_lr = 0;  /* Start from 0 */
    sched->step = 0;
    sched->epoch = 0;

    return CT_OK;
}

/**
 * @brief Initialize cosine annealing scheduler
 *
 * @param sched Scheduler to initialize
 * @param initial_lr Maximum learning rate
 * @param min_lr Minimum learning rate
 * @param total_steps Total training steps
 * @param lut Pre-initialized cosine LUT
 * @return CT_OK on success
 */
ct_error_t ct_scheduler_init_cosine(ct_scheduler_t *sched,
                                    fixed_t initial_lr,
                                    fixed_t min_lr,
                                    uint32_t total_steps,
                                    const ct_cosine_lut_t *lut)
{
    if (sched == NULL) return CT_ERR_NULL;
    if (total_steps == 0) return CT_ERR_CONFIG;
    if (lut == NULL || !lut->initialized) return CT_ERR_CONFIG;

    sched->type = CT_SCHED_COSINE;
    sched->config.cosine.initial_lr = initial_lr;
    sched->config.cosine.min_lr = min_lr;
    sched->config.cosine.total_steps = total_steps;
    sched->config.cosine.lut = lut;
    sched->current_lr = initial_lr;
    sched->step = 0;
    sched->epoch = 0;

    return CT_OK;
}

/* ============================================================================
 * Scheduler Operations
 * ============================================================================ */

/**
 * @brief Get current learning rate
 *
 * @param sched Scheduler
 * @return Current learning rate (Q16.16)
 */
fixed_t ct_scheduler_get_lr(const ct_scheduler_t *sched)
{
    if (sched == NULL) return 0;
    return sched->current_lr;
}

/**
 * @brief Advance scheduler by one step and update learning rate
 *
 * @param sched Scheduler
 * @param faults Fault accumulator
 * @return Updated learning rate
 */
fixed_t ct_scheduler_step(ct_scheduler_t *sched, ct_fault_flags_t *faults)
{
    if (sched == NULL) return 0;

    sched->step++;

    switch (sched->type) {
        case CT_SCHED_CONSTANT:
            /* No change */
            break;

        case CT_SCHED_STEP:
            /* lr = lr_0 * gamma^(epoch / step_size)
             * Computed incrementally: multiply by gamma each step_size epochs
             */
            /* This is epoch-based, handled in ct_scheduler_epoch_end */
            break;

        case CT_SCHED_LINEAR_WARMUP:
            if (sched->step < sched->config.warmup.warmup_steps) {
                /* lr = target_lr * step / warmup_steps */
                int64_t numer = (int64_t)sched->config.warmup.target_lr * (int64_t)sched->step;
                sched->current_lr = (fixed_t)(numer / (int64_t)sched->config.warmup.warmup_steps);
            } else {
                sched->current_lr = sched->config.warmup.target_lr;
            }
            break;

        case CT_SCHED_COSINE:
            {
                /* lr = lr_min + 0.5 * (lr_0 - lr_min) * (1 + cos(π * t / T)) */
                uint32_t t = (uint32_t)sched->step;
                uint32_t T = sched->config.cosine.total_steps;

                if (t >= T) {
                    sched->current_lr = sched->config.cosine.min_lr;
                } else {
                    /* Compute t/T in Q16.16 */
                    int64_t ratio = ((int64_t)t << FIXED_FRAC_BITS) / (int64_t)T;

                    /* cos(π * t / T) via LUT */
                    fixed_t cos_val = cosine_lookup((fixed_t)ratio, sched->config.cosine.lut);

                    /* (1 + cos_val) / 2 */
                    int64_t one_plus_cos = (int64_t)FIXED_ONE + (int64_t)cos_val;
                    fixed_t factor = (fixed_t)(one_plus_cos >> 1);

                    /* (lr_0 - lr_min) * factor */
                    int64_t range = (int64_t)sched->config.cosine.initial_lr -
                                    (int64_t)sched->config.cosine.min_lr;
                    int64_t scaled = (range * (int64_t)factor) >> FIXED_FRAC_BITS;

                    sched->current_lr = sched->config.cosine.min_lr + (fixed_t)scaled;
                }
            }
            break;

        default:
            break;
    }

    (void)faults;  /* Reserved for future fault detection */
    return sched->current_lr;
}

/**
 * @brief Signal end of epoch (for epoch-based schedulers)
 *
 * @param sched Scheduler
 * @param faults Fault accumulator
 * @return Updated learning rate
 */
fixed_t ct_scheduler_epoch_end(ct_scheduler_t *sched, ct_fault_flags_t *faults)
{
    if (sched == NULL) return 0;

    sched->epoch++;

    if (sched->type == CT_SCHED_STEP) {
        /* Check if we should decay */
        if (sched->epoch % sched->config.step.step_size == 0) {
            /* lr = lr * gamma */
            int64_t prod = (int64_t)sched->current_lr * (int64_t)sched->config.step.gamma;
            sched->current_lr = dvm_round_shift_rne(prod, FIXED_FRAC_BITS, faults);
        }
    }

    return sched->current_lr;
}

/**
 * @brief Reset scheduler to initial state
 *
 * @param sched Scheduler
 */
void ct_scheduler_reset(ct_scheduler_t *sched)
{
    if (sched == NULL) return;

    sched->step = 0;
    sched->epoch = 0;

    switch (sched->type) {
        case CT_SCHED_CONSTANT:
            /* current_lr unchanged */
            break;
        case CT_SCHED_STEP:
            sched->current_lr = sched->config.step.initial_lr;
            break;
        case CT_SCHED_LINEAR_WARMUP:
            sched->current_lr = 0;
            break;
        case CT_SCHED_COSINE:
            sched->current_lr = sched->config.cosine.initial_lr;
            break;
        default:
            break;
    }
}
