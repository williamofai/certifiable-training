/**
 * @file merkle.h
 * @project Certifiable Training
 * @brief Merkle training chain for auditable ML.
 *
 * @details Implements cryptographic audit trail:
 *          - Step hashes: h_t = SHA256(h_{t-1} || H(θ_t) || H(B_t) || t)
 *          - Canonical tensor serialization
 *          - Checkpoint creation and verification
 *          - Fault invalidation
 *
 * @traceability SRS-008-MERKLE, CT-MATH-001 §16-17
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust. All rights reserved.
 * @license Licensed under the GPL-3.0 (Open Source) or Commercial License.
 *          For commercial licensing: william@fstopify.com
 */

#ifndef CERTIFIABLE_TRAINING_MERKLE_H
#define CERTIFIABLE_TRAINING_MERKLE_H

#include "ct_types.h"
#include "forward.h"
#include "prng.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

/** SHA256 hash size in bytes */
#define CT_HASH_SIZE            32

/** Serialization format version */
#define CT_SERIALIZE_VERSION    1

/** Checkpoint format version */
#define CT_CHECKPOINT_VERSION   2

/** Data type identifiers for serialization */
#define CT_DTYPE_Q16_16         0
#define CT_DTYPE_Q8_24          1
#define CT_DTYPE_Q32_32         2

/* ============================================================================
 * Serialization Structures
 * ============================================================================ */

/**
 * @brief Header for canonical tensor serialization
 * @ref CT-STRUCT-001 §12.2
 */
typedef struct {
    uint32_t version;               /**< Format version (1) */
    uint32_t dtype;                 /**< 0=Q16.16, 1=Q8.24, 2=Q32.32 */
    uint32_t ndims;                 /**< Number of dimensions */
    uint32_t dims[CT_MAX_DIMS];     /**< Dimension sizes */
    uint64_t total_size;            /**< Total elements */
} ct_serialize_header_t;

/* ============================================================================
 * Merkle Chain Structures
 * ============================================================================ */

/**
 * @brief Training step record for Merkle chain
 * @ref CT-STRUCT-001 §10.1
 */
typedef struct {
    uint8_t prev_hash[CT_HASH_SIZE];    /**< h_{t-1} */
    uint8_t weights_hash[CT_HASH_SIZE]; /**< H(θ_t) */
    uint8_t batch_hash[CT_HASH_SIZE];   /**< H(B_t) */
    uint64_t step;                      /**< Step number t */
    uint8_t step_hash[CT_HASH_SIZE];    /**< h_t = result */
} ct_training_step_t;

/**
 * @brief Training checkpoint
 * @ref CT-STRUCT-001 §10.2
 */
typedef struct {
    uint64_t step;                      /**< Training step */
    uint32_t epoch;                     /**< Current epoch */
    uint8_t merkle_hash[CT_HASH_SIZE];  /**< Current chain hash */
    uint8_t weights_hash[CT_HASH_SIZE]; /**< Weights hash at checkpoint */
    uint8_t config_hash[CT_HASH_SIZE];  /**< Config hash */
    ct_prng_t prng_state;               /**< PRNG state for resumption */
    uint64_t timestamp;                 /**< EXCLUDED from hash commitment */
    uint32_t version;                   /**< Checkpoint format version */
    ct_fault_flags_t fault_flags;       /**< Fault state */
} ct_checkpoint_t;

/**
 * @brief Merkle chain context
 */
typedef struct {
    uint8_t current_hash[CT_HASH_SIZE]; /**< Current chain head */
    uint8_t initial_hash[CT_HASH_SIZE]; /**< h_0 for verification */
    uint64_t step;                      /**< Current step */
    uint32_t epoch;                     /**< Current epoch */
    bool initialized;                   /**< Chain initialized flag */
    bool faulted;                       /**< Chain invalidated by fault */
} ct_merkle_ctx_t;

/* ============================================================================
 * SHA256 Interface (Simple Implementation)
 * ============================================================================ */

/**
 * @brief SHA256 context for incremental hashing
 */
typedef struct {
    uint32_t state[8];      /**< Hash state */
    uint64_t count;         /**< Bytes processed */
    uint8_t buffer[64];     /**< Input buffer */
} ct_sha256_ctx_t;

/**
 * @brief Initialize SHA256 context
 */
void ct_sha256_init(ct_sha256_ctx_t *ctx);

/**
 * @brief Update SHA256 with data
 */
void ct_sha256_update(ct_sha256_ctx_t *ctx, const void *data, size_t len);

/**
 * @brief Finalize SHA256 and output hash
 */
void ct_sha256_final(ct_sha256_ctx_t *ctx, uint8_t hash[CT_HASH_SIZE]);

/**
 * @brief One-shot SHA256 hash
 */
void ct_sha256(const void *data, size_t len, uint8_t hash[CT_HASH_SIZE]);

/* ============================================================================
 * Canonical Serialization
 * ============================================================================ */

/**
 * @brief Check if tensor is contiguous (required for hashing)
 * @param tensor Tensor to check
 * @return true if contiguous layout
 */
bool ct_tensor_is_contiguous(const ct_tensor_t *tensor);

/**
 * @brief Get serialization buffer size needed for tensor
 * @param tensor Tensor to serialize
 * @return Required buffer size in bytes
 */
size_t ct_tensor_serial_size(const ct_tensor_t *tensor);

/**
 * @brief Serialize tensor to canonical byte stream
 * @param tensor Source tensor (must be contiguous)
 * @param buffer Output buffer
 * @param buffer_size Available buffer size
 * @return Bytes written, or negative error
 *
 * @ref CT-MATH-001 §17.2
 */
int32_t ct_tensor_serialize(const ct_tensor_t *tensor,
                            uint8_t *buffer,
                            size_t buffer_size);

/**
 * @brief Compute SHA256 hash of tensor in canonical form
 * @param tensor Tensor to hash (must be contiguous)
 * @param hash_out Output hash [32 bytes]
 * @return CT_OK on success
 *
 * @note This is what gets committed to Merkle chain
 */
ct_error_t ct_tensor_hash(const ct_tensor_t *tensor,
                          uint8_t hash_out[CT_HASH_SIZE]);

/* ============================================================================
 * Merkle Chain Operations
 * ============================================================================ */

/**
 * @brief Initialize Merkle chain with initial state
 * @param ctx Chain context
 * @param initial_weights Initial weights tensor
 * @param config_data Configuration data
 * @param config_size Configuration data size
 * @param seed Training seed
 * @return CT_OK on success
 *
 * @details Computes h_0 = SHA256(H(θ_0) || H(config) || seed)
 * @ref CT-MATH-001 §16.2
 */
ct_error_t ct_merkle_init(ct_merkle_ctx_t *ctx,
                          const ct_tensor_t *initial_weights,
                          const void *config_data,
                          size_t config_size,
                          uint64_t seed);

/**
 * @brief Compute step hash and advance chain
 * @param ctx Chain context
 * @param weights Current weights tensor
 * @param batch_indices Batch sample indices
 * @param batch_size Number of samples in batch
 * @param step_out Optional output for step record
 * @param faults Fault flags (chain invalidated if faulted)
 * @return CT_OK on success
 *
 * @details Computes h_t = SHA256(h_{t-1} || H(θ_t) || H(B_t) || t)
 * @ref CT-MATH-001 §16.1
 */
ct_error_t ct_merkle_step(ct_merkle_ctx_t *ctx,
                          const ct_tensor_t *weights,
                          const uint32_t *batch_indices,
                          uint32_t batch_size,
                          ct_training_step_t *step_out,
                          const ct_fault_flags_t *faults);

/**
 * @brief Get current chain hash
 * @param ctx Chain context
 * @param hash_out Output hash [32 bytes]
 */
void ct_merkle_get_hash(const ct_merkle_ctx_t *ctx,
                        uint8_t hash_out[CT_HASH_SIZE]);

/**
 * @brief Check if chain is valid (not faulted)
 */
bool ct_merkle_is_valid(const ct_merkle_ctx_t *ctx);

/**
 * @brief Invalidate chain due to fault
 */
void ct_merkle_invalidate(ct_merkle_ctx_t *ctx);

/* ============================================================================
 * Checkpoint Operations
 * ============================================================================ */

/**
 * @brief Create checkpoint from current state
 * @param ctx Merkle chain context
 * @param prng PRNG state for resumption
 * @param epoch Current epoch
 * @param weights Current weights
 * @param config_hash Pre-computed config hash
 * @param checkpoint Output checkpoint
 * @return CT_OK on success
 */
ct_error_t ct_checkpoint_create(const ct_merkle_ctx_t *ctx,
                                const ct_prng_t *prng,
                                uint32_t epoch,
                                const ct_tensor_t *weights,
                                const uint8_t config_hash[CT_HASH_SIZE],
                                ct_checkpoint_t *checkpoint);

/**
 * @brief Verify checkpoint integrity
 * @param checkpoint Checkpoint to verify
 * @param weights Weights tensor to verify against
 * @return CT_OK if valid, CT_ERR_HASH if mismatch
 */
ct_error_t ct_checkpoint_verify(const ct_checkpoint_t *checkpoint,
                                const ct_tensor_t *weights);

/**
 * @brief Restore Merkle context from checkpoint
 * @param ctx Context to restore
 * @param checkpoint Checkpoint to restore from
 * @return CT_OK on success
 */
ct_error_t ct_merkle_restore(ct_merkle_ctx_t *ctx,
                             const ct_checkpoint_t *checkpoint);

/* ============================================================================
 * Verification Utilities
 * ============================================================================ */

/**
 * @brief Verify a single step in the chain
 * @param step Step record to verify
 * @param prev_hash Expected previous hash
 * @param weights Weights at this step
 * @param batch_indices Batch indices at this step
 * @param batch_size Batch size
 * @return CT_OK if valid
 */
ct_error_t ct_merkle_verify_step(const ct_training_step_t *step,
                                 const uint8_t prev_hash[CT_HASH_SIZE],
                                 const ct_tensor_t *weights,
                                 const uint32_t *batch_indices,
                                 uint32_t batch_size);

/**
 * @brief Compare two hashes for equality
 */
bool ct_hash_equal(const uint8_t a[CT_HASH_SIZE],
                   const uint8_t b[CT_HASH_SIZE]);

/**
 * @brief Copy hash
 */
void ct_hash_copy(uint8_t dst[CT_HASH_SIZE],
                  const uint8_t src[CT_HASH_SIZE]);

/**
 * @brief Zero hash
 */
void ct_hash_zero(uint8_t hash[CT_HASH_SIZE]);

#ifdef __cplusplus
}
#endif

#endif /* CERTIFIABLE_TRAINING_MERKLE_H */
