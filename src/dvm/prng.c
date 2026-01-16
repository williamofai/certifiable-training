/**
 * @file prng.c
 * @project Certifiable Training
 * @brief Counter-based PRNG for deterministic randomness
 *
 * @details Implements a Philox-style counter-based PRNG that produces
 *          deterministic pseudo-random bits as a pure function of
 *          (seed, op_id, step). No internal state that varies between calls.
 *
 * @traceability CT-MATH-001 §6
 * @compliance MISRA-C:2012, DO-178C, IEC 62304, ISO 26262
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 */

#include "prng.h"
#include "dvm.h"

/* Philox-style mixing constants */
#define PRNG_MUL_CTR    0xD2511F53ULL
#define PRNG_MUL_KEY    0xCD9E8D57ULL
#define PRNG_ADD_KEY    0x9E3779B9ULL  /* Golden ratio fractional part */
#define PRNG_ROUNDS     10

/**
 * @brief Initialize PRNG state
 *
 * @param prng  Pointer to PRNG state structure
 * @param seed  Master seed (determines entire sequence)
 * @param op_id Operation identifier (unique per operation context)
 *
 * @pre prng != NULL
 * @post prng initialized with step = 0
 *
 * @traceability CT-MATH-001 §6.1
 */
void ct_prng_init(ct_prng_t *prng, uint64_t seed, uint64_t op_id)
{
    if (prng == NULL) {
        return;
    }
    
    prng->seed = seed;
    prng->op_id = op_id;
    prng->step = 0;
}

/**
 * @brief Core PRNG function - pure function of inputs
 *
 * @param seed  Master seed
 * @param op_id Operation identifier
 * @param step  Step counter
 * @return Deterministic 32-bit pseudo-random value
 *
 * @details Philox-style counter-based RNG:
 *          - Counter formed from (op_id, step) with proper mixing
 *          - Key derived from seed XOR'd with op_id for differentiation
 *          - 10 rounds of mixing
 *          - Output is low 32 bits
 *
 * @note This is a PURE FUNCTION - same inputs always produce same output
 *
 * @traceability CT-MATH-001 §6.2
 */
static uint32_t prng_core(uint64_t seed, uint64_t op_id, uint64_t step)
{
    /*
     * Form counter from op_id and step.
     * We need both op_id and step to influence the output strongly.
     * 
     * Original: ctr = (op_id << 32) | (step & 0xFFFFFFFF)
     * Problem: Only lower 32 bits of op_id used, and when step=0,
     *          similar op_ids produce similar counters.
     *
     * Fix: XOR op_id into the key as well, ensuring different op_ids
     *      produce completely different sequences even at step=0.
     */
    uint64_t ctr = (op_id << 32) | (step & 0xFFFFFFFFULL);
    
    /* 
     * Key incorporates both seed AND op_id.
     * This ensures different op_ids produce different sequences
     * even when the counter portion is similar.
     */
    uint64_t key = seed ^ (op_id * 0x9E3779B97F4A7C15ULL);
    
    /* 10 rounds of Philox-style mixing */
    for (int r = 0; r < PRNG_ROUNDS; r++) {
        ctr = ((ctr * PRNG_MUL_CTR) & 0xFFFFFFFFFFFFFFFFULL) ^ key;
        key = ((key * PRNG_MUL_KEY) + PRNG_ADD_KEY) & 0xFFFFFFFFFFFFFFFFULL;
    }
    
    /* Return low 32 bits */
    return (uint32_t)(ctr & 0xFFFFFFFFULL);
}

/**
 * @brief Generate next pseudo-random value and advance state
 *
 * @param prng Pointer to PRNG state
 * @return 32-bit pseudo-random value
 *
 * @pre prng != NULL and initialized
 * @post prng->step incremented by 1
 *
 * @traceability CT-MATH-001 §6.2
 */
uint32_t ct_prng_next(ct_prng_t *prng)
{
    if (prng == NULL) {
        return 0;
    }
    
    uint32_t result = prng_core(prng->seed, prng->op_id, prng->step);
    prng->step++;
    
    return result;
}

/**
 * @brief Generate value at specific step without modifying state
 *
 * @param prng Pointer to PRNG state (const - not modified)
 * @param step Step to query
 * @return 32-bit pseudo-random value at that step
 *
 * @note Allows random access to any point in the sequence
 *
 * @traceability CT-MATH-001 §6.2
 */
uint32_t ct_prng_peek(const ct_prng_t *prng, uint64_t step)
{
    if (prng == NULL) {
        return 0;
    }
    
    return prng_core(prng->seed, prng->op_id, step);
}

/**
 * @brief Deterministic stochastic rounding
 *
 * @param x      Value to round (64-bit intermediate)
 * @param shift  Number of fractional bits to remove
 * @param prng   PRNG state (will be advanced)
 * @param faults Fault flags (set on domain error)
 * @return Stochastically rounded 32-bit result
 *
 * @details Uses PRNG output as threshold for probabilistic rounding.
 *          The probability of rounding up equals the fractional part.
 *          This provides regularization benefits while remaining
 *          fully deterministic (same seed = same sequence).
 *
 * @pre prng != NULL and initialized
 * @pre 0 <= shift <= 62
 * @post prng->step advanced by 1
 *
 * @traceability CT-MATH-001 §8.4
 */
int32_t ct_stochastic_round(int64_t x, uint32_t shift, ct_prng_t *prng,
                            ct_fault_flags_t *faults)
{
    /* Shift bounds check */
    if (shift > CT_MAX_SHIFT) {
        if (faults) faults->domain = 1;
        return 0;
    }
    
    /* Zero shift - just clamp */
    if (shift == 0) {
        return dvm_clamp32(x, faults);
    }
    
    /* NULL prng - fall back to truncation */
    if (prng == NULL) {
        return dvm_clamp32(x >> shift, faults);
    }
    
    /* Get random threshold */
    uint32_t rand = ct_prng_next(prng);
    
    /* Extract fractional part */
    int64_t mask = (1LL << shift) - 1;
    int64_t fraction = x & mask;
    
    /* Scale random to match fraction range */
    /* threshold is in [0, 2^shift - 1] */
    uint32_t threshold = rand >> (32 - shift);
    
    /* Compute quotient (truncated) */
    int64_t quotient = x >> shift;
    
    /* Probabilistic rounding: round up if fraction > threshold */
    int64_t result;
    if ((uint64_t)fraction > (uint64_t)threshold) {
        result = quotient + 1;
    } else {
        result = quotient;
    }
    
    return dvm_clamp32(result, faults);
}

/**
 * @brief Compute op_id from context (helper function)
 *
 * @param layer_id      Layer index in model
 * @param tensor_id     Tensor index within layer
 * @param element_idx   Element index within tensor
 * @return 64-bit operation identifier
 *
 * @details Combines multiple indices into a single unique identifier.
 *          Uses multiplication and XOR for mixing.
 *
 * @note For 128-bit op_id (recommended for large models), use
 *       ct_prng_make_op_id_128() instead.
 *
 * @traceability CT-MATH-001 §6.3
 */
uint64_t ct_prng_make_op_id(uint32_t layer_id, uint32_t tensor_id,
                            uint32_t element_idx)
{
    /* Mix the three IDs into 64 bits */
    uint64_t id = (uint64_t)layer_id;
    id = id * 0x9E3779B97F4A7C15ULL + tensor_id;
    id = id * 0xBF58476D1CE4E5B9ULL + element_idx;
    id = id ^ (id >> 30);
    id = id * 0x94D049BB133111EBULL;
    id = id ^ (id >> 31);
    
    return id;
}
