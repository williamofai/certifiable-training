/**
 * @file merkle.c
 * @project Certifiable Training
 * @brief Merkle training chain implementation.
 *
 * @details Includes embedded SHA256 implementation for portability.
 *
 * @traceability SRS-008-MERKLE, CT-MATH-001 §16-17
 * @compliance MISRA-C:2012, ISO 26262, IEC 62304
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust. All rights reserved.
 * @license Licensed under the GPL-3.0 (Open Source) or Commercial License.
 *          For commercial licensing: william@fstopify.com
 */

#include "merkle.h"
#include <string.h>
#include <time.h>

/* ============================================================================
 * SHA256 Implementation (FIPS 180-4)
 * ============================================================================ */

static const uint32_t sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

static void sha256_transform(ct_sha256_ctx_t *ctx, const uint8_t *data) {
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t w[64];
    uint32_t t1, t2;
    
    /* Prepare message schedule */
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)data[i * 4] << 24) |
               ((uint32_t)data[i * 4 + 1] << 16) |
               ((uint32_t)data[i * 4 + 2] << 8) |
               ((uint32_t)data[i * 4 + 3]);
    }
    for (int i = 16; i < 64; i++) {
        w[i] = SIG1(w[i - 2]) + w[i - 7] + SIG0(w[i - 15]) + w[i - 16];
    }
    
    /* Initialize working variables */
    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];
    
    /* 64 rounds */
    for (int i = 0; i < 64; i++) {
        t1 = h + EP1(e) + CH(e, f, g) + sha256_k[i] + w[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    /* Update state */
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

void ct_sha256_init(ct_sha256_ctx_t *ctx) {
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
}

void ct_sha256_update(ct_sha256_ctx_t *ctx, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    size_t buf_idx = (size_t)(ctx->count & 63);
    
    ctx->count += len;
    
    /* Fill buffer and transform if full */
    if (buf_idx > 0) {
        size_t to_copy = 64 - buf_idx;
        if (to_copy > len) to_copy = len;
        memcpy(ctx->buffer + buf_idx, p, to_copy);
        p += to_copy;
        len -= to_copy;
        buf_idx += to_copy;
        if (buf_idx == 64) {
            sha256_transform(ctx, ctx->buffer);
            buf_idx = 0;
        }
    }
    
    /* Process full blocks */
    while (len >= 64) {
        sha256_transform(ctx, p);
        p += 64;
        len -= 64;
    }
    
    /* Buffer remaining */
    if (len > 0) {
        memcpy(ctx->buffer, p, len);
    }
}

void ct_sha256_final(ct_sha256_ctx_t *ctx, uint8_t hash[CT_HASH_SIZE]) {
    size_t buf_idx = (size_t)(ctx->count & 63);
    uint64_t bit_count = ctx->count * 8;
    
    /* Padding */
    ctx->buffer[buf_idx++] = 0x80;
    
    if (buf_idx > 56) {
        memset(ctx->buffer + buf_idx, 0, 64 - buf_idx);
        sha256_transform(ctx, ctx->buffer);
        buf_idx = 0;
    }
    
    memset(ctx->buffer + buf_idx, 0, 56 - buf_idx);
    
    /* Append length (big-endian) */
    ctx->buffer[56] = (uint8_t)(bit_count >> 56);
    ctx->buffer[57] = (uint8_t)(bit_count >> 48);
    ctx->buffer[58] = (uint8_t)(bit_count >> 40);
    ctx->buffer[59] = (uint8_t)(bit_count >> 32);
    ctx->buffer[60] = (uint8_t)(bit_count >> 24);
    ctx->buffer[61] = (uint8_t)(bit_count >> 16);
    ctx->buffer[62] = (uint8_t)(bit_count >> 8);
    ctx->buffer[63] = (uint8_t)(bit_count);
    
    sha256_transform(ctx, ctx->buffer);
    
    /* Output hash (big-endian) */
    for (int i = 0; i < 8; i++) {
        hash[i * 4] = (uint8_t)(ctx->state[i] >> 24);
        hash[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        hash[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        hash[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

void ct_sha256(const void *data, size_t len, uint8_t hash[CT_HASH_SIZE]) {
    ct_sha256_ctx_t ctx;
    ct_sha256_init(&ctx);
    ct_sha256_update(&ctx, data, len);
    ct_sha256_final(&ctx, hash);
}

/* ============================================================================
 * Hash Utilities
 * ============================================================================ */

bool ct_hash_equal(const uint8_t a[CT_HASH_SIZE],
                   const uint8_t b[CT_HASH_SIZE]) {
    /* Constant-time comparison */
    uint8_t diff = 0;
    for (int i = 0; i < CT_HASH_SIZE; i++) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

void ct_hash_copy(uint8_t dst[CT_HASH_SIZE],
                  const uint8_t src[CT_HASH_SIZE]) {
    memcpy(dst, src, CT_HASH_SIZE);
}

void ct_hash_zero(uint8_t hash[CT_HASH_SIZE]) {
    memset(hash, 0, CT_HASH_SIZE);
}

/* ============================================================================
 * Canonical Serialization
 * ============================================================================ */

bool ct_tensor_is_contiguous(const ct_tensor_t *tensor) {
    if (!tensor || tensor->ndims == 0) {
        return true;
    }
    
    /* Check if strides match natural row-major layout */
    uint32_t expected_stride = 1;
    for (int i = (int)tensor->ndims - 1; i >= 0; i--) {
        if (tensor->strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= tensor->dims[i];
    }
    return true;
}

size_t ct_tensor_serial_size(const ct_tensor_t *tensor) {
    if (!tensor) return 0;
    
    /* Header: version(4) + dtype(4) + ndims(4) + dims(4*MAX_DIMS) + total_size(8) */
    size_t header_size = 4 + 4 + 4 + (4 * CT_MAX_DIMS) + 8;
    
    /* Data: total_size * sizeof(fixed_t) */
    size_t data_size = tensor->total_size * sizeof(fixed_t);
    
    return header_size + data_size;
}

/**
 * @brief Write uint32 in little-endian
 */
static void write_u32_le(uint8_t *buf, uint32_t val) {
    buf[0] = (uint8_t)(val);
    buf[1] = (uint8_t)(val >> 8);
    buf[2] = (uint8_t)(val >> 16);
    buf[3] = (uint8_t)(val >> 24);
}

/**
 * @brief Write uint64 in little-endian
 */
static void write_u64_le(uint8_t *buf, uint64_t val) {
    buf[0] = (uint8_t)(val);
    buf[1] = (uint8_t)(val >> 8);
    buf[2] = (uint8_t)(val >> 16);
    buf[3] = (uint8_t)(val >> 24);
    buf[4] = (uint8_t)(val >> 32);
    buf[5] = (uint8_t)(val >> 40);
    buf[6] = (uint8_t)(val >> 48);
    buf[7] = (uint8_t)(val >> 56);
}

/**
 * @brief Write int32 in little-endian (two's complement)
 */
static void write_i32_le(uint8_t *buf, int32_t val) {
    write_u32_le(buf, (uint32_t)val);
}

int32_t ct_tensor_serialize(const ct_tensor_t *tensor,
                            uint8_t *buffer,
                            size_t buffer_size) {
    if (!tensor || !buffer) {
        return CT_ERR_NULL;
    }
    
    if (!ct_tensor_is_contiguous(tensor)) {
        return CT_ERR_STATE;  /* Must be contiguous */
    }
    
    size_t needed = ct_tensor_serial_size(tensor);
    if (buffer_size < needed) {
        return CT_ERR_MEMORY;
    }
    
    uint8_t *p = buffer;
    
    /* Header */
    write_u32_le(p, CT_SERIALIZE_VERSION); p += 4;
    write_u32_le(p, CT_DTYPE_Q16_16); p += 4;
    write_u32_le(p, tensor->ndims); p += 4;
    
    for (uint32_t i = 0; i < CT_MAX_DIMS; i++) {
        write_u32_le(p, tensor->dims[i]); p += 4;
    }
    
    write_u64_le(p, tensor->total_size); p += 8;
    
    /* Data (little-endian) */
    for (uint32_t i = 0; i < tensor->total_size; i++) {
        write_i32_le(p, tensor->data[i]); p += 4;
    }
    
    return (int32_t)(p - buffer);
}

ct_error_t ct_tensor_hash(const ct_tensor_t *tensor,
                          uint8_t hash_out[CT_HASH_SIZE]) {
    if (!tensor || !hash_out) {
        return CT_ERR_NULL;
    }
    
    if (!ct_tensor_is_contiguous(tensor)) {
        return CT_ERR_STATE;
    }
    
    ct_sha256_ctx_t ctx;
    ct_sha256_init(&ctx);
    
    /* Hash header */
    uint8_t header[4 + 4 + 4 + (4 * CT_MAX_DIMS) + 8];
    uint8_t *p = header;
    
    write_u32_le(p, CT_SERIALIZE_VERSION); p += 4;
    write_u32_le(p, CT_DTYPE_Q16_16); p += 4;
    write_u32_le(p, tensor->ndims); p += 4;
    
    for (uint32_t i = 0; i < CT_MAX_DIMS; i++) {
        write_u32_le(p, tensor->dims[i]); p += 4;
    }
    
    write_u64_le(p, tensor->total_size); p += 8;
    
    ct_sha256_update(&ctx, header, sizeof(header));
    
    /* Hash data element by element (little-endian) */
    uint8_t elem[4];
    for (uint32_t i = 0; i < tensor->total_size; i++) {
        write_i32_le(elem, tensor->data[i]);
        ct_sha256_update(&ctx, elem, 4);
    }
    
    ct_sha256_final(&ctx, hash_out);
    
    return CT_OK;
}

/* ============================================================================
 * Merkle Chain Operations
 * ============================================================================ */

ct_error_t ct_merkle_init(ct_merkle_ctx_t *ctx,
                          const ct_tensor_t *initial_weights,
                          const void *config_data,
                          size_t config_size,
                          uint64_t seed) {
    if (!ctx || !initial_weights) {
        return CT_ERR_NULL;
    }
    
    /* h_0 = SHA256(H(θ_0) || H(config) || seed) */
    ct_sha256_ctx_t sha;
    ct_sha256_init(&sha);
    
    /* Hash initial weights */
    uint8_t weights_hash[CT_HASH_SIZE];
    ct_error_t err = ct_tensor_hash(initial_weights, weights_hash);
    if (err != CT_OK) return err;
    ct_sha256_update(&sha, weights_hash, CT_HASH_SIZE);
    
    /* Hash config */
    if (config_data && config_size > 0) {
        uint8_t config_hash[CT_HASH_SIZE];
        ct_sha256(config_data, config_size, config_hash);
        ct_sha256_update(&sha, config_hash, CT_HASH_SIZE);
    } else {
        /* Empty config hash */
        uint8_t zero_hash[CT_HASH_SIZE] = {0};
        ct_sha256_update(&sha, zero_hash, CT_HASH_SIZE);
    }
    
    /* Hash seed (little-endian) */
    uint8_t seed_bytes[8];
    write_u64_le(seed_bytes, seed);
    ct_sha256_update(&sha, seed_bytes, 8);
    
    ct_sha256_final(&sha, ctx->current_hash);
    ct_hash_copy(ctx->initial_hash, ctx->current_hash);
    
    ctx->step = 0;
    ctx->epoch = 0;
    ctx->initialized = true;
    ctx->faulted = false;
    
    return CT_OK;
}

/**
 * @brief Compute batch hash: H(B_t) = SHA256(indices...)
 */
static void compute_batch_hash(const uint32_t *indices,
                               uint32_t count,
                               uint8_t hash_out[CT_HASH_SIZE]) {
    ct_sha256_ctx_t ctx;
    ct_sha256_init(&ctx);
    
    uint8_t idx_bytes[4];
    for (uint32_t i = 0; i < count; i++) {
        write_u32_le(idx_bytes, indices[i]);
        ct_sha256_update(&ctx, idx_bytes, 4);
    }
    
    ct_sha256_final(&ctx, hash_out);
}

ct_error_t ct_merkle_step(ct_merkle_ctx_t *ctx,
                          const ct_tensor_t *weights,
                          const uint32_t *batch_indices,
                          uint32_t batch_size,
                          ct_training_step_t *step_out,
                          const ct_fault_flags_t *faults) {
    if (!ctx || !weights || !batch_indices) {
        return CT_ERR_NULL;
    }
    
    if (!ctx->initialized) {
        return CT_ERR_STATE;
    }
    
    /* Check for faults - invalidate chain if any */
    if (faults && ct_has_fault(faults)) {
        ctx->faulted = true;
        return CT_ERR_FAULT;
    }
    
    if (ctx->faulted) {
        return CT_ERR_FAULT;
    }
    
    /* h_t = SHA256(h_{t-1} || H(θ_t) || H(B_t) || t) */
    ct_sha256_ctx_t sha;
    ct_sha256_init(&sha);
    
    /* Previous hash */
    ct_sha256_update(&sha, ctx->current_hash, CT_HASH_SIZE);
    
    /* Weights hash */
    uint8_t weights_hash[CT_HASH_SIZE];
    ct_error_t err = ct_tensor_hash(weights, weights_hash);
    if (err != CT_OK) return err;
    ct_sha256_update(&sha, weights_hash, CT_HASH_SIZE);
    
    /* Batch hash */
    uint8_t batch_hash[CT_HASH_SIZE];
    compute_batch_hash(batch_indices, batch_size, batch_hash);
    ct_sha256_update(&sha, batch_hash, CT_HASH_SIZE);
    
    /* Step number (little-endian) */
    uint8_t step_bytes[8];
    write_u64_le(step_bytes, ctx->step);
    ct_sha256_update(&sha, step_bytes, 8);
    
    /* Compute new hash */
    uint8_t new_hash[CT_HASH_SIZE];
    ct_sha256_final(&sha, new_hash);
    
    /* Fill step record if requested */
    if (step_out) {
        ct_hash_copy(step_out->prev_hash, ctx->current_hash);
        ct_hash_copy(step_out->weights_hash, weights_hash);
        ct_hash_copy(step_out->batch_hash, batch_hash);
        step_out->step = ctx->step;
        ct_hash_copy(step_out->step_hash, new_hash);
    }
    
    /* Update context */
    ct_hash_copy(ctx->current_hash, new_hash);
    ctx->step++;
    
    return CT_OK;
}

void ct_merkle_get_hash(const ct_merkle_ctx_t *ctx,
                        uint8_t hash_out[CT_HASH_SIZE]) {
    if (ctx && hash_out) {
        ct_hash_copy(hash_out, ctx->current_hash);
    }
}

bool ct_merkle_is_valid(const ct_merkle_ctx_t *ctx) {
    return ctx && ctx->initialized && !ctx->faulted;
}

void ct_merkle_invalidate(ct_merkle_ctx_t *ctx) {
    if (ctx) {
        ctx->faulted = true;
    }
}

/* ============================================================================
 * Checkpoint Operations
 * ============================================================================ */

ct_error_t ct_checkpoint_create(const ct_merkle_ctx_t *ctx,
                                const ct_prng_t *prng,
                                uint32_t epoch,
                                const ct_tensor_t *weights,
                                const uint8_t config_hash[CT_HASH_SIZE],
                                ct_checkpoint_t *checkpoint) {
    if (!ctx || !prng || !weights || !config_hash || !checkpoint) {
        return CT_ERR_NULL;
    }
    
    checkpoint->step = ctx->step;
    checkpoint->epoch = epoch;
    ct_hash_copy(checkpoint->merkle_hash, ctx->current_hash);
    
    ct_error_t err = ct_tensor_hash(weights, checkpoint->weights_hash);
    if (err != CT_OK) return err;
    
    ct_hash_copy(checkpoint->config_hash, config_hash);
    checkpoint->prng_state = *prng;
    checkpoint->timestamp = (uint64_t)time(NULL);
    checkpoint->version = CT_CHECKPOINT_VERSION;
    ct_clear_faults(&checkpoint->fault_flags);
    
    if (ctx->faulted) {
        checkpoint->fault_flags.overflow = 1;  /* Mark as faulted */
    }
    
    return CT_OK;
}

ct_error_t ct_checkpoint_verify(const ct_checkpoint_t *checkpoint,
                                const ct_tensor_t *weights) {
    if (!checkpoint || !weights) {
        return CT_ERR_NULL;
    }
    
    uint8_t computed_hash[CT_HASH_SIZE];
    ct_error_t err = ct_tensor_hash(weights, computed_hash);
    if (err != CT_OK) return err;
    
    if (!ct_hash_equal(computed_hash, checkpoint->weights_hash)) {
        return CT_ERR_HASH;
    }
    
    return CT_OK;
}

ct_error_t ct_merkle_restore(ct_merkle_ctx_t *ctx,
                             const ct_checkpoint_t *checkpoint) {
    if (!ctx || !checkpoint) {
        return CT_ERR_NULL;
    }
    
    ct_hash_copy(ctx->current_hash, checkpoint->merkle_hash);
    ctx->step = checkpoint->step;
    ctx->epoch = checkpoint->epoch;
    ctx->initialized = true;
    ctx->faulted = ct_has_fault(&checkpoint->fault_flags);
    
    return CT_OK;
}

/* ============================================================================
 * Verification
 * ============================================================================ */

ct_error_t ct_merkle_verify_step(const ct_training_step_t *step,
                                 const uint8_t prev_hash[CT_HASH_SIZE],
                                 const ct_tensor_t *weights,
                                 const uint32_t *batch_indices,
                                 uint32_t batch_size) {
    if (!step || !prev_hash || !weights || !batch_indices) {
        return CT_ERR_NULL;
    }
    
    /* Verify prev_hash matches */
    if (!ct_hash_equal(step->prev_hash, prev_hash)) {
        return CT_ERR_HASH;
    }
    
    /* Verify weights hash */
    uint8_t computed_weights[CT_HASH_SIZE];
    ct_error_t err = ct_tensor_hash(weights, computed_weights);
    if (err != CT_OK) return err;
    
    if (!ct_hash_equal(step->weights_hash, computed_weights)) {
        return CT_ERR_HASH;
    }
    
    /* Verify batch hash */
    uint8_t computed_batch[CT_HASH_SIZE];
    compute_batch_hash(batch_indices, batch_size, computed_batch);
    
    if (!ct_hash_equal(step->batch_hash, computed_batch)) {
        return CT_ERR_HASH;
    }
    
    /* Recompute step hash */
    ct_sha256_ctx_t sha;
    ct_sha256_init(&sha);
    ct_sha256_update(&sha, step->prev_hash, CT_HASH_SIZE);
    ct_sha256_update(&sha, step->weights_hash, CT_HASH_SIZE);
    ct_sha256_update(&sha, step->batch_hash, CT_HASH_SIZE);
    
    uint8_t step_bytes[8];
    write_u64_le(step_bytes, step->step);
    ct_sha256_update(&sha, step_bytes, 8);
    
    uint8_t computed_step[CT_HASH_SIZE];
    ct_sha256_final(&sha, computed_step);
    
    if (!ct_hash_equal(step->step_hash, computed_step)) {
        return CT_ERR_HASH;
    }
    
    return CT_OK;
}
