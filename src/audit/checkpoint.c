/**
 * @file checkpoint.c
 * @project Certifiable Training
 * @brief Training checkpoint serialization and verification
 *
 * @details Implements checkpoint operations for resumable training:
 *          - Checkpoint creation with cryptographic binding
 *          - Checkpoint serialization to byte stream
 *          - Checkpoint deserialization and verification
 *          - Integrity checking via hash comparison
 *
 *          Checkpoint format (CT-STRUCT-001 §10.2):
 *          - Fixed-size header with version and step
 *          - Merkle hash binding to training state
 *          - PRNG state for resumption
 *          - Fault flags for safety
 *
 * @traceability SRS-008-MERKLE, CT-STRUCT-001 §10.2
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304, DO-178C
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 * @license GPL-3.0 or Commercial License (william@fstopify.com)
 */

#include "merkle.h"
#include <string.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

/** Checkpoint magic number: "CTCK" in little-endian */
#define CT_CHECKPOINT_MAGIC     0x4B435443

/** Checkpoint serialization buffer size */
#define CT_CHECKPOINT_SERIAL_SIZE \
    (4 +                          /* magic */ \
     4 +                          /* version */ \
     8 +                          /* step */ \
     4 +                          /* epoch */ \
     CT_HASH_SIZE +               /* merkle_hash */ \
     CT_HASH_SIZE +               /* weights_hash */ \
     CT_HASH_SIZE +               /* config_hash */ \
     8 + 8 + 8 +                  /* prng_state: seed, op_id, step */ \
     4 +                          /* fault_flags */ \
     8)                           /* timestamp (not committed) */

/* ============================================================================
 * Little-Endian Serialization Helpers
 * ============================================================================ */

/**
 * @brief Write 32-bit value in little-endian format
 */
static void write_le32(uint8_t *buf, uint32_t val)
{
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
    buf[2] = (uint8_t)((val >> 16) & 0xFF);
    buf[3] = (uint8_t)((val >> 24) & 0xFF);
}

/**
 * @brief Write 64-bit value in little-endian format
 */
static void write_le64(uint8_t *buf, uint64_t val)
{
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
    buf[2] = (uint8_t)((val >> 16) & 0xFF);
    buf[3] = (uint8_t)((val >> 24) & 0xFF);
    buf[4] = (uint8_t)((val >> 32) & 0xFF);
    buf[5] = (uint8_t)((val >> 40) & 0xFF);
    buf[6] = (uint8_t)((val >> 48) & 0xFF);
    buf[7] = (uint8_t)((val >> 56) & 0xFF);
}

/**
 * @brief Read 32-bit value from little-endian format
 */
static uint32_t read_le32(const uint8_t *buf)
{
    return (uint32_t)buf[0] |
           ((uint32_t)buf[1] << 8) |
           ((uint32_t)buf[2] << 16) |
           ((uint32_t)buf[3] << 24);
}

/**
 * @brief Read 64-bit value from little-endian format
 */
static uint64_t read_le64(const uint8_t *buf)
{
    return (uint64_t)buf[0] |
           ((uint64_t)buf[1] << 8) |
           ((uint64_t)buf[2] << 16) |
           ((uint64_t)buf[3] << 24) |
           ((uint64_t)buf[4] << 32) |
           ((uint64_t)buf[5] << 40) |
           ((uint64_t)buf[6] << 48) |
           ((uint64_t)buf[7] << 56);
}

/* ============================================================================
 * Checkpoint Serialization
 * ============================================================================ */

/**
 * @brief Get required buffer size for checkpoint serialization
 *
 * @return Size in bytes
 */
size_t ct_checkpoint_serial_size(void)
{
    return CT_CHECKPOINT_SERIAL_SIZE;
}

/**
 * @brief Serialize checkpoint to byte buffer
 *
 * @param checkpoint Checkpoint to serialize
 * @param buffer Output buffer (must be at least CT_CHECKPOINT_SERIAL_SIZE bytes)
 * @param buffer_size Available buffer size
 * @return Bytes written, or negative error code
 *
 * @details Format (little-endian):
 *          - magic (4 bytes)
 *          - version (4 bytes)
 *          - step (8 bytes)
 *          - epoch (4 bytes)
 *          - merkle_hash (32 bytes)
 *          - weights_hash (32 bytes)
 *          - config_hash (32 bytes)
 *          - prng.seed (8 bytes)
 *          - prng.op_id (8 bytes)
 *          - prng.step (8 bytes)
 *          - fault_flags (4 bytes)
 *          - timestamp (8 bytes) - NOT included in hash commitment
 */
int32_t ct_checkpoint_serialize(const ct_checkpoint_t *checkpoint,
                                uint8_t *buffer,
                                size_t buffer_size)
{
    if (checkpoint == NULL || buffer == NULL) {
        return CT_ERR_NULL;
    }

    if (buffer_size < CT_CHECKPOINT_SERIAL_SIZE) {
        return CT_ERR_MEMORY;
    }

    uint8_t *p = buffer;

    /* Magic number */
    write_le32(p, CT_CHECKPOINT_MAGIC);
    p += 4;

    /* Version */
    write_le32(p, checkpoint->version);
    p += 4;

    /* Step */
    write_le64(p, checkpoint->step);
    p += 8;

    /* Epoch */
    write_le32(p, checkpoint->epoch);
    p += 4;

    /* Merkle hash */
    memcpy(p, checkpoint->merkle_hash, CT_HASH_SIZE);
    p += CT_HASH_SIZE;

    /* Weights hash */
    memcpy(p, checkpoint->weights_hash, CT_HASH_SIZE);
    p += CT_HASH_SIZE;

    /* Config hash */
    memcpy(p, checkpoint->config_hash, CT_HASH_SIZE);
    p += CT_HASH_SIZE;

    /* PRNG state */
    write_le64(p, checkpoint->prng_state.seed);
    p += 8;
    write_le64(p, checkpoint->prng_state.op_id);
    p += 8;
    write_le64(p, checkpoint->prng_state.step);
    p += 8;

    /* Fault flags (packed into 32 bits) */
    uint32_t flags = 0;
    flags |= (checkpoint->fault_flags.overflow ? 1 : 0);
    flags |= (checkpoint->fault_flags.underflow ? 2 : 0);
    flags |= (checkpoint->fault_flags.div_zero ? 4 : 0);
    flags |= (checkpoint->fault_flags.domain ? 8 : 0);
    flags |= (checkpoint->fault_flags.grad_floor ? 16 : 0);
    write_le32(p, flags);
    p += 4;

    /* Timestamp (not committed to hash, but stored) */
    write_le64(p, checkpoint->timestamp);
    p += 8;

    return (int32_t)(p - buffer);
}

/**
 * @brief Deserialize checkpoint from byte buffer
 *
 * @param buffer Input buffer
 * @param buffer_size Buffer size
 * @param checkpoint Output checkpoint structure
 * @return CT_OK on success, CT_ERR_HASH if format invalid
 */
ct_error_t ct_checkpoint_deserialize(const uint8_t *buffer,
                                     size_t buffer_size,
                                     ct_checkpoint_t *checkpoint)
{
    if (buffer == NULL || checkpoint == NULL) {
        return CT_ERR_NULL;
    }

    if (buffer_size < CT_CHECKPOINT_SERIAL_SIZE) {
        return CT_ERR_MEMORY;
    }

    const uint8_t *p = buffer;

    /* Verify magic number */
    uint32_t magic = read_le32(p);
    p += 4;
    if (magic != CT_CHECKPOINT_MAGIC) {
        return CT_ERR_HASH;  /* Invalid format */
    }

    /* Version */
    checkpoint->version = read_le32(p);
    p += 4;
    if (checkpoint->version > CT_CHECKPOINT_VERSION) {
        return CT_ERR_CONFIG;  /* Unsupported version */
    }

    /* Step */
    checkpoint->step = read_le64(p);
    p += 8;

    /* Epoch */
    checkpoint->epoch = read_le32(p);
    p += 4;

    /* Merkle hash */
    memcpy(checkpoint->merkle_hash, p, CT_HASH_SIZE);
    p += CT_HASH_SIZE;

    /* Weights hash */
    memcpy(checkpoint->weights_hash, p, CT_HASH_SIZE);
    p += CT_HASH_SIZE;

    /* Config hash */
    memcpy(checkpoint->config_hash, p, CT_HASH_SIZE);
    p += CT_HASH_SIZE;

    /* PRNG state */
    checkpoint->prng_state.seed = read_le64(p);
    p += 8;
    checkpoint->prng_state.op_id = read_le64(p);
    p += 8;
    checkpoint->prng_state.step = read_le64(p);
    p += 8;

    /* Fault flags */
    uint32_t flags = read_le32(p);
    p += 4;
    checkpoint->fault_flags.overflow = (uint32_t)((flags & 1U) ? 1U : 0U);
    checkpoint->fault_flags.underflow = (uint32_t)((flags & 2U) ? 1U : 0U);
    checkpoint->fault_flags.div_zero = (uint32_t)((flags & 4U) ? 1U : 0U);
    checkpoint->fault_flags.domain = (uint32_t)((flags & 8U) ? 1U : 0U);
    checkpoint->fault_flags.grad_floor = (uint32_t)((flags & 16U) ? 1U : 0U);

    /* Timestamp */
    checkpoint->timestamp = read_le64(p);
    p += 8;

    (void)p;  /* Suppress unused warning */
    return CT_OK;
}

/* ============================================================================
 * Checkpoint Hash Computation
 * ============================================================================ */

/**
 * @brief Compute hash of checkpoint content (excluding timestamp)
 *
 * @param checkpoint Checkpoint to hash
 * @param hash_out Output hash [32 bytes]
 * @return CT_OK on success
 *
 * @details The hash covers all committed fields:
 *          version, step, epoch, merkle_hash, weights_hash,
 *          config_hash, prng_state, fault_flags
 *
 *          Timestamp is EXCLUDED per CT-STRUCT-001 §10.2
 */
ct_error_t ct_checkpoint_compute_hash(const ct_checkpoint_t *checkpoint,
                                      uint8_t hash_out[CT_HASH_SIZE])
{
    if (checkpoint == NULL || hash_out == NULL) {
        return CT_ERR_NULL;
    }

    /* Serialize committed fields only (exclude timestamp) */
    uint8_t buf[CT_CHECKPOINT_SERIAL_SIZE - 8];  /* Exclude timestamp (8 bytes) */
    uint8_t *p = buf;

    /* Version */
    write_le32(p, checkpoint->version);
    p += 4;

    /* Step */
    write_le64(p, checkpoint->step);
    p += 8;

    /* Epoch */
    write_le32(p, checkpoint->epoch);
    p += 4;

    /* Merkle hash */
    memcpy(p, checkpoint->merkle_hash, CT_HASH_SIZE);
    p += CT_HASH_SIZE;

    /* Weights hash */
    memcpy(p, checkpoint->weights_hash, CT_HASH_SIZE);
    p += CT_HASH_SIZE;

    /* Config hash */
    memcpy(p, checkpoint->config_hash, CT_HASH_SIZE);
    p += CT_HASH_SIZE;

    /* PRNG state */
    write_le64(p, checkpoint->prng_state.seed);
    p += 8;
    write_le64(p, checkpoint->prng_state.op_id);
    p += 8;
    write_le64(p, checkpoint->prng_state.step);
    p += 8;

    /* Fault flags */
    uint32_t flags = 0;
    flags |= (checkpoint->fault_flags.overflow ? 1 : 0);
    flags |= (checkpoint->fault_flags.underflow ? 2 : 0);
    flags |= (checkpoint->fault_flags.div_zero ? 4 : 0);
    flags |= (checkpoint->fault_flags.domain ? 8 : 0);
    flags |= (checkpoint->fault_flags.grad_floor ? 16 : 0);
    write_le32(p, flags);
    p += 4;

    /* Compute SHA256 */
    ct_sha256(buf, (size_t)(p - buf), hash_out);

    return CT_OK;
}

/**
 * @brief Compare two checkpoints for equality (hash-based)
 *
 * @param a First checkpoint
 * @param b Second checkpoint
 * @return true if committed content is identical
 */
bool ct_checkpoint_equal(const ct_checkpoint_t *a, const ct_checkpoint_t *b)
{
    if (a == NULL || b == NULL) return false;

    uint8_t hash_a[CT_HASH_SIZE];
    uint8_t hash_b[CT_HASH_SIZE];

    if (ct_checkpoint_compute_hash(a, hash_a) != CT_OK) return false;
    if (ct_checkpoint_compute_hash(b, hash_b) != CT_OK) return false;

    return ct_hash_equal(hash_a, hash_b);
}

/**
 * @brief Verify checkpoint against current weights
 *
 * @param checkpoint Checkpoint to verify
 * @param weights Current weights tensor
 * @return CT_OK if weights hash matches, CT_ERR_HASH otherwise
 *
 * @note Defined in merkle.h, implemented here for completeness
 */
ct_error_t ct_checkpoint_verify_weights(const ct_checkpoint_t *checkpoint,
                                        const ct_tensor_t *weights)
{
    if (checkpoint == NULL || weights == NULL) {
        return CT_ERR_NULL;
    }

    uint8_t computed_hash[CT_HASH_SIZE];
    ct_error_t err = ct_tensor_hash(weights, computed_hash);
    if (err != CT_OK) {
        return err;
    }

    if (!ct_hash_equal(checkpoint->weights_hash, computed_hash)) {
        return CT_ERR_HASH;
    }

    return CT_OK;
}

/**
 * @brief Initialize checkpoint from current training state
 *
 * @param checkpoint Output checkpoint
 * @param merkle_ctx Current Merkle chain context
 * @param prng Current PRNG state
 * @param epoch Current epoch
 * @param weights Current weights tensor
 * @param config_hash Pre-computed config hash
 * @return CT_OK on success
 */
ct_error_t ct_checkpoint_init(ct_checkpoint_t *checkpoint,
                              const ct_merkle_ctx_t *merkle_ctx,
                              const ct_prng_t *prng,
                              uint32_t epoch,
                              const ct_tensor_t *weights,
                              const uint8_t config_hash[CT_HASH_SIZE])
{
    if (checkpoint == NULL || merkle_ctx == NULL || prng == NULL ||
        weights == NULL || config_hash == NULL) {
        return CT_ERR_NULL;
    }

    /* Use ct_checkpoint_create from merkle.h */
    return ct_checkpoint_create(merkle_ctx, prng, epoch, weights,
                                config_hash, checkpoint);
}
