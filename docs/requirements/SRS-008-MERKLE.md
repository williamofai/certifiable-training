# SRS-008: Merkle Training Chain

**Certifiable Training — Software Requirements Specification**

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | SRS-008-MERKLE |
| Version | 1.0.0 |
| Author | William Murray |
| Date | January 2026 |
| Status | Implementation Complete |

---

## 1. Purpose

This document specifies the Merkle training chain for auditable ML. Every training step produces a cryptographic hash linking to the previous step, creating an immutable audit trail.

---

## 2. References

| Document | Section |
|----------|---------|
| CT-MATH-001 | §16 Merkle Training Chains, §17 Canonical Serialization |
| CT-STRUCT-001 | §10 Merkle Audit Structures, §12 Serialization |

---

## 3. Requirements

### 3.1 SHA256 (REQ-MRK-001)
Embedded FIPS 180-4 compliant SHA256 implementation.

### 3.2 Initial Hash (REQ-MRK-002)
h_0 = SHA256(H(θ_0) || H(config) || seed)

### 3.3 Step Hash (REQ-MRK-003)
h_t = SHA256(h_{t-1} || H(θ_t) || H(B_t) || t)

### 3.4 Canonical Serialization (REQ-MRK-004)
- Little-endian byte order
- Contiguous tensors only
- No pointers in serialized data
- Per CT-MATH-001 §17

### 3.5 Fault Invalidation (REQ-MRK-005)
If any DVM fault flag is set, chain becomes invalid.

### 3.6 Checkpoint (REQ-MRK-006)
Save/restore chain state with verification.

### 3.7 Step Verification (REQ-MRK-007)
Ability to verify any step in the chain independently.

---

## 4. Data Structures

```c
typedef struct {
    uint8_t prev_hash[32];
    uint8_t weights_hash[32];
    uint8_t batch_hash[32];
    uint64_t step;
    uint8_t step_hash[32];
} ct_training_step_t;

typedef struct {
    uint64_t step;
    uint32_t epoch;
    uint8_t merkle_hash[32];
    uint8_t weights_hash[32];
    uint8_t config_hash[32];
    ct_prng_t prng_state;
    uint64_t timestamp;  /* EXCLUDED from hash */
    uint32_t version;
    ct_fault_flags_t fault_flags;
} ct_checkpoint_t;
```

---

## 5. Test Coverage

28 unit tests covering:
- SHA256 (empty, "abc", incremental)
- Hash utilities (equal, copy, zero)
- Tensor serialization
- Merkle init and determinism
- Step chaining
- Fault invalidation
- Checkpoint create/verify/restore
- Step verification
- Error handling

---

## 6. Files

| File | Description |
|------|-------------|
| merkle.h | Public API |
| merkle.c | SHA256 + Merkle chain implementation |
| test_merkle.c | 28 unit tests |
| SRS-008-MERKLE.md | This document |

---

*End of SRS-008-MERKLE v1.0.0*
