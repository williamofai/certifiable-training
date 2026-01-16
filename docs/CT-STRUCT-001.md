# CT-STRUCT-001: Data Structure Specification

**Certifiable Training — Data Dictionary: Structs**

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | CT-STRUCT-001 |
| Version | 2.1.0 |
| Author | William Murray |
| Date | January 2026 |
| Status | Production Draft |
| Classification | Public |
| Copyright | © 2026 The Murray Family Innovation Trust. All rights reserved. |

---

## 1. Purpose

This document specifies every data structure in Certifiable Training. Each struct is derived directly from CT-MATH-001. Fields map to mathematical objects. Nothing is stored that cannot be justified mathematically.

**Principle**: Structs encode state. State is minimal. If it can be computed, don't store it.

---

## 2. Design Principles

| Principle | Requirement |
|-----------|-------------|
| **Minimal State** | Only store what cannot be recomputed |
| **Fixed Size** | No pointers to variable-length data within structs |
| **Alignment** | All structs naturally aligned for target platforms |
| **Determinism** | Struct layout is compiler-independent where possible |
| **Traceability** | Every field traces to CT-MATH-001 |
| **Hash Safety** | No struct is hashed directly; only canonical serialization |

### 2.1 Compiler Hardening

**Required compiler flags**:

```makefile
CFLAGS += -std=c99 -Wall -Wextra -Werror
CFLAGS += -ffp-contract=off      # No FMA fusion
CFLAGS += -fno-fast-math         # Strict semantics  
CFLAGS += -fno-associative-math  # Preserve operation order
```

### 2.2 Toolchain Semantics (Mandatory)

**Policy**: Explicit widening to 64-bit before all 32-bit arithmetic. No reliance on `-fwrapv`.

**Rationale**: `-fwrapv` behaviour varies subtly across compilers and versions. Explicit widening is portable and auditable.

**Required pattern**:

```c
/* CORRECT: Explicit widening */
int32_t safe_add(int32_t a, int32_t b) {
    int64_t wide = (int64_t)a + (int64_t)b;
    return dvm_clamp32(wide);
}

/* FORBIDDEN: Relying on -fwrapv */
int32_t unsafe_add(int32_t a, int32_t b) {
    return a + b;  /* UB without -fwrapv, non-portable with it */
}
```

**Enforcement**: Code review must reject any direct `+`, `-`, `*` on `int32_t` operands without widening.

### 2.2 Toolchain Semantics (Mandatory)

**Policy**: Explicit widening to 64-bit for all 32-bit arithmetic. No reliance on `-fwrapv`.

**Rationale**: `-fwrapv` behaviour varies across compilers and versions. Explicit widening is portable and auditable.

**Required pattern** (all add/sub/mul on `int32_t`):

```c
/* CORRECT: Explicit widening */
int64_t wide = (int64_t)a + (int64_t)b;
int32_t result = ct_clamp32(wide);

/* FORBIDDEN: Relying on -fwrapv */
int32_t result = a + b;  /* UB if overflow */
```

**Code review rule**: Any direct `+`, `-`, `*` on `fixed_t` without widening is a rejection.

**Toolchain constraints**:
- GCC ≥ 7.0 or Clang ≥ 6.0
- Must pass `-fsanitize=undefined` tests without overflow errors
- `-fwrapv` is NOT required and must NOT be relied upon

---

## 3. Primitive Types

### 3.1 Fixed-Point Types

```c
typedef int32_t fixed_t;      /* Q16.16: weights, activations */
typedef int32_t fixed_hp_t;   /* Q8.24: gradients */
typedef int64_t fixed_acc_t;  /* Q32.32: accumulators */
```

### 3.2 Constants

```c
#define FIXED_FRAC_BITS     16
#define FIXED_ONE           ((fixed_t)(1 << FIXED_FRAC_BITS))
#define FIXED_HALF          ((fixed_t)(1 << (FIXED_FRAC_BITS-1)))
#define FIXED_MAX           ((fixed_t)INT32_MAX)
#define FIXED_MIN           ((fixed_t)INT32_MIN)

#define FIXED_HP_FRAC_BITS  24
#define FIXED_ACC_FRAC_BITS 32

#define CT_MAX_BATCH_SIZE   65536  /* Accumulator exhaustion limit */
```

---

## 4. DVM Primitives

### 4.1 PRNG State

```c
/**
 * @brief Counter-based PRNG state
 * @math DVM_PRNG(seed, op_id, step) → uint32
 * @ref CT-MATH-001 §6
 * 
 * @note op_id is 64-bit to ensure global uniqueness
 */
typedef struct {
    uint64_t seed;      /**< Master seed (immutable after init) */
    uint64_t op_id;     /**< Operation identifier (64-bit minimum) */
    uint64_t step;      /**< Current step counter */
} ct_prng_t;
```

### 4.2 Compensated Accumulator

```c
/**
 * @brief Neumaier compensated accumulator
 * @math (sum, err) pair for high-precision reduction
 * @ref CT-MATH-001 §9.3
 */
typedef struct {
    fixed_acc_t sum;    /**< Running sum */
    fixed_acc_t err;    /**< Compensation term */
} ct_comp_accum_t;
```

### 4.3 Reduction Node

```c
/**
 * @brief Node in fixed-topology reduction tree
 * @ref CT-MATH-001 §9.1
 */
typedef struct {
    uint32_t left_child;    /**< Index or CT_LEAF_MARKER */
    uint32_t right_child;   /**< Index or CT_LEAF_MARKER */
    uint64_t op_id;         /**< 64-bit unique operation ID */
    uint32_t parent;        /**< Parent index or CT_ROOT_MARKER */
    uint32_t _pad;          /**< Alignment padding */
} ct_reduction_node_t;

#define CT_LEAF_MARKER   UINT32_MAX
#define CT_ROOT_MARKER   UINT32_MAX
```

### 4.4 Reduction Tree

```c
typedef struct {
    ct_reduction_node_t *nodes;  /**< Array of nodes (caller-provided) */
    uint32_t num_leaves;         /**< Batch size */
    uint32_t num_internal;       /**< num_leaves - 1 */
    uint32_t root_index;         /**< Root node index */
    uint32_t depth;              /**< Tree depth ⌈log₂(num_leaves)⌉ */
} ct_reduction_tree_t;
```

---

## 5. Tensor Structures

### 5.1 Tensor Descriptor

```c
#define CT_MAX_DIMS 4

typedef struct {
    fixed_t *data;                  /**< Data buffer (caller-provided) */
    uint32_t dims[CT_MAX_DIMS];     /**< Dimension sizes */
    uint32_t strides[CT_MAX_DIMS];  /**< Element strides */
    uint32_t ndims;                 /**< Number of dimensions */
    uint32_t total_size;            /**< Total elements */
} ct_tensor_t;
```

### 5.2 Gradient Tensor

```c
typedef struct {
    fixed_hp_t *data;               /**< High-precision data */
    uint32_t dims[CT_MAX_DIMS];
    uint32_t strides[CT_MAX_DIMS];
    uint32_t ndims;
    uint32_t total_size;
} ct_grad_tensor_t;
```

### 5.3 Gradient Health Monitor

```c
/**
 * @brief Vanishing gradient detection
 * @ref CT-MATH-001 §7.4
 */
typedef struct {
    uint64_t zero_grad_count;       /**< Gradients hitting floor */
    uint64_t total_grad_count;      /**< Total updates */
    fixed_hp_t min_nonzero_grad;    /**< Smallest non-zero seen */
    fixed_hp_t max_grad;            /**< Largest gradient seen */
} ct_grad_health_t;

#define CT_GRAD_HEALTH_THRESHOLD  3277  /* 0.05 in Q16.16 */
```

---

## 6. Layer Structures

### 6.1 Linear Layer

```c
typedef struct {
    ct_tensor_t weights;            /**< W: [output, input] */
    ct_tensor_t bias;               /**< b: [output] */
    ct_grad_tensor_t grad_weights;  /**< ∂L/∂W */
    ct_grad_tensor_t grad_bias;     /**< ∂L/∂b */
    uint32_t input_size;
    uint32_t output_size;
    ct_tensor_t *input_cache;       /**< Cached input for backward */
} ct_linear_t;
```

### 6.2 Activation Layer

```c
typedef enum {
    CT_ACT_NONE     = 0,
    CT_ACT_RELU     = 1,
    CT_ACT_SIGMOID  = 2,    /**< LUT-based, no exp() */
    CT_ACT_TANH     = 3,    /**< LUT-based */
} ct_activation_type_t;

#define CT_ACTIVATION_LUT_SIZE  257

typedef struct {
    fixed_t table[CT_ACTIVATION_LUT_SIZE];
    fixed_t domain_min;     /**< -8 in Q16.16 */
    fixed_t domain_max;     /**< +8 in Q16.16 */
} ct_activation_lut_t;

typedef struct {
    ct_activation_type_t type;
    const ct_activation_lut_t *lut;  /**< NULL for ReLU */
    ct_tensor_t *pre_act_cache;
} ct_activation_t;
```

---

## 7. Optimizer Structures

### 7.1 SGD

```c
typedef struct {
    fixed_t learning_rate;
    fixed_t weight_decay;
} ct_sgd_config_t;
```

### 7.2 SGD with Momentum

```c
typedef struct {
    fixed_t learning_rate;
    fixed_t momentum;           /**< β, typically 0.9 */
    fixed_t weight_decay;
    ct_tensor_t *velocity;      /**< Caller-provided */
    uint32_t num_params;
} ct_sgd_momentum_t;
```

### 7.3 Adam

```c
typedef struct {
    fixed_t learning_rate;
    fixed_t beta1;              /**< 0.9 */
    fixed_t beta2;              /**< 0.999 */
    fixed_t epsilon;
    fixed_t weight_decay;
    ct_tensor_t *m;             /**< First moment */
    ct_tensor_t *v;             /**< Second moment */
    fixed_t beta1_power;        /**< β₁^t */
    fixed_t beta2_power;        /**< β₂^t */
    uint32_t num_params;
    uint32_t step;
} ct_adam_t;
```

---

## 8. Data Permutation

```c
/**
 * @brief Cycle-walking Feistel permutation state
 * @ref CT-MATH-001 §5
 */
typedef struct {
    uint64_t seed;
    uint32_t epoch;
    uint32_t dataset_size;      /**< N */
} ct_permutation_t;

/* Permutation is computed on-demand, not stored */
```

---

## 9. Training State

### 9.1 Configuration

```c
typedef struct {
    uint64_t seed;
    uint32_t batch_size;
    uint32_t num_epochs;
    uint32_t dataset_size;
    fixed_t grad_clip_max;
    uint32_t checkpoint_interval;
} ct_train_config_t;
```

### 9.2 Training Context

```c
typedef struct {
    ct_train_config_t config;
    uint64_t step;
    uint32_t epoch;
    uint32_t step_in_epoch;
    ct_prng_t prng;
    ct_reduction_tree_t *reduction_tree;
    uint8_t prev_hash[32];
    uint8_t current_hash[32];
    uint32_t steps_per_epoch;
    uint32_t total_steps;
    ct_fault_flags_t fault_flags;   /**< Sticky fault state */
} ct_train_ctx_t;
```

---

## 10. Merkle Audit Structures

### 10.1 Training Step Record

```c
/**
 * @brief Merkle chain step
 * @ref CT-MATH-001 §16
 */
typedef struct {
    uint8_t prev_hash[32];
    uint8_t weights_hash[32];
    uint8_t batch_hash[32];
    uint64_t step;
    uint8_t step_hash[32];
} ct_training_step_t;
```

### 10.2 Checkpoint

```c
/**
 * @note timestamp is NOT included in hash
 */
typedef struct {
    uint64_t step;
    uint32_t epoch;
    uint8_t merkle_hash[32];
    uint8_t weights_hash[32];
    uint8_t config_hash[32];
    ct_prng_t prng_state;
    uint64_t timestamp;         /**< EXCLUDED from commitments */
    uint32_t version;
    ct_fault_flags_t fault_flags;
} ct_checkpoint_t;

#define CT_CHECKPOINT_VERSION 2
```

---

## 11. Fault Model

### 11.1 Fault Flags

```c
/**
 * @brief Sticky fault flags
 * @ref CT-MATH-001 §3.7
 */
typedef struct {
    uint32_t overflow    : 1;   /**< Saturated high */
    uint32_t underflow   : 1;   /**< Saturated low */
    uint32_t div_zero    : 1;   /**< Division by zero */
    uint32_t domain      : 1;   /**< Out-of-range input (e.g., shift > 62) */
    uint32_t grad_floor  : 1;   /**< Excessive zero gradients */
    uint32_t _reserved   : 27;
} ct_fault_flags_t;

/**
 * @brief Check if any fault is set
 */
static inline bool ct_has_fault(const ct_fault_flags_t *f) {
    return f->overflow || f->underflow || f->div_zero || f->domain;
}

/**
 * @brief Clear all faults (explicit action required)
 */
static inline void ct_clear_faults(ct_fault_flags_t *f) {
    f->overflow = 0;
    f->underflow = 0;
    f->div_zero = 0;
    f->domain = 0;
    f->grad_floor = 0;
}
```

### 11.2 Error Codes

```c
typedef enum {
    CT_OK               =  0,
    CT_ERR_NULL         = -1,
    CT_ERR_DIMENSION    = -2,
    CT_ERR_OVERFLOW     = -3,
    CT_ERR_UNDERFLOW    = -4,
    CT_ERR_DIV_ZERO     = -5,
    CT_ERR_CONFIG       = -6,
    CT_ERR_STATE        = -7,
    CT_ERR_MEMORY       = -8,
    CT_ERR_HASH         = -9,
    CT_ERR_FAULT        = -10,  /**< Operation halted due to fault */
} ct_error_t;
```

### 11.3 Step Result

```c
typedef struct {
    ct_error_t error;
    fixed_t loss;
    fixed_t grad_norm;
    uint64_t step;
    uint8_t step_hash[32];
    ct_fault_flags_t faults;    /**< Faults during this step */
} ct_step_result_t;
```

---

## 12. Canonical Serialization

### 12.1 Contiguous Tensor Requirement

**Committed tensors must be contiguous.** Strides are for transient/working memory only.

Strides exist in `ct_tensor_t` for computational convenience (views, transposes), but:
- Before hashing: tensor must be contiguous
- Before checkpointing: tensor must be contiguous
- Strides are NOT serialized or hashed

**Rationale**: Strided tensors can represent identical mathematical data with different memory layouts. Canonical serialization requires a single, unambiguous representation.

### 12.2 Serialization Header

```c
/**
 * @brief Header for canonical tensor serialization
 * @ref CT-MATH-001 §17.2
 */
typedef struct {
    uint32_t version;           /**< Format version (1) */
    uint32_t dtype;             /**< 0=Q16.16, 1=Q8.24, 2=Q32.32 */
    uint32_t ndims;
    uint32_t dims[CT_MAX_DIMS];
    uint64_t total_size;
} ct_serialize_header_t;

/* All fields written as little-endian */
```

### 12.2 Serialization Functions

```c
/**
 * @brief Serialize tensor to canonical byte stream
 * @param tensor Source tensor
 * @param buffer Output buffer (must be large enough)
 * @param buffer_size Available buffer size
 * @return Bytes written, or negative error
 * 
 * @note Pointers are NEVER serialized
 * @note Padding bytes are NEVER included
 */
int32_t ct_tensor_serialize(const ct_tensor_t *tensor,
                            uint8_t *buffer,
                            size_t buffer_size);

/**
 * @brief Compute SHA256 of canonical tensor representation
 * @note This is what gets committed to Merkle chain
 */
void ct_tensor_hash(const ct_tensor_t *tensor, uint8_t hash_out[32]);
```

### 12.3 Hashing Rules

**NEVER hash**:
- Pointers (`data`, `input_cache`, etc.)
- Padding bytes between struct fields
- Timestamps (except in metadata, excluded from commitment)
- Transient state

**ALWAYS hash**:
- Tensor data in little-endian format
- Tensor metadata (dims, dtype)
- Configuration values
- Step numbers

---

## 13. Memory Layout

### 13.1 Alignment

```c
#define CT_ALIGN_DEFAULT    8
#define CT_ALIGN_SIMD       32

#define CT_ALIGNED(type, name, count) \
    type name[count] __attribute__((aligned(CT_ALIGN_DEFAULT)))
```

### 13.2 Static Assertions

```c
#define CT_STATIC_ASSERT(cond, msg) \
    typedef char ct_static_assert_##__LINE__[(cond) ? 1 : -1]

CT_STATIC_ASSERT(CT_MAX_BATCH_SIZE <= 65536,
                 "Batch size exceeds accumulator safety margin");

CT_STATIC_ASSERT(sizeof(ct_reduction_node_t) == 24,
                 "Reduction node size changed");
```

---

## 14. What is NOT Stored

| Not Stored | Reason |
|------------|--------|
| Previous PRNG outputs | Regenerable from (seed, op_id, step) |
| Permuted index arrays | Computed via cycle-walking Feistel |
| Intermediate activations | Recomputed or cached externally |
| Node values in reduction | Computed during reduction pass |
| Full training history | Only Merkle hashes retained |
| Struct pointers in hashes | Never part of committed state |

---

## 15. API Patterns

### 15.1 Init-Update-Status

```c
ct_error_t ct_xxx_init(ct_xxx_t *ctx, const ct_xxx_config_t *config);
ct_error_t ct_xxx_update(ct_xxx_t *ctx, /* inputs */);
ct_xxx_status_t ct_xxx_status(const ct_xxx_t *ctx);
void ct_xxx_reset(ct_xxx_t *ctx);
```

### 15.2 Forward-Backward

```c
ct_error_t ct_layer_forward(ct_layer_t *layer,
                            const ct_tensor_t *input,
                            ct_tensor_t *output,
                            ct_fault_flags_t *faults);

ct_error_t ct_layer_backward(ct_layer_t *layer,
                             const ct_grad_tensor_t *grad_output,
                             ct_grad_tensor_t *grad_input,
                             ct_fault_flags_t *faults);
```

---

## 16. Document Alignment Matrix

| Topic | CT-SPEC-001 | CT-MATH-001 | CT-STRUCT-001 |
|-------|-------------|-------------|---------------|
| Rounding | §4 | §8 | §3 (types) |
| Permutation | §5 | §5 | §8 (ct_permutation_t) |
| Batch indexing | §5.5 | §5.6 | N/A (computed) |
| CompAdd | §6.3 | §9.3 | §4.2 (ct_comp_accum_t) |
| Fault model | §3.3-3.4 | §3.6 | §11 (this doc) |
| Serialization | §10 | §17 | §12 (this doc) |
| op_id | §6.2 | §6.3 | §4.1 (64-bit) |

---

## 17. References

1. CT-SPEC-001 — Certifiable Training Specification
2. CT-MATH-001 — Mathematical Foundations
3. CT-DVM-001 — Deterministic Virtual Machine Specification
4. c-from-scratch — Module patterns and testing methodology

---

*End of CT-STRUCT-001 v2.0.0*
