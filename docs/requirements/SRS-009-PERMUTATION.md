# SRS-009: Data Permutation

**Certifiable Training — Software Requirements Specification**

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | SRS-009-PERMUTATION |
| Version | 1.0.0 |
| Author | William Murray |
| Date | January 2026 |
| Status | Implementation Complete |

---

## 1. Purpose

This document specifies the Cycle-Walking Feistel permutation for deterministic dataset shuffling. Every epoch uses a different but reproducible permutation, enabling bit-identical training across platforms.

---

## 2. References

| Document | Section |
|----------|---------|
| CT-MATH-001 | §5 Data Ordering: Cycle-Walking Feistel |
| CT-STRUCT-001 | §8 Data Permutation |

---

## 3. Requirements

### 3.1 Bijection (REQ-PERM-001)
π: [0,N-1] → [0,N-1] is a true bijection for any N.

### 3.2 Cycle-Walking Feistel (REQ-PERM-002)
4-round Feistel network with cycle-walking for non-power-of-two N.
Per CT-MATH-001 §5.3.

### 3.3 Feistel Hash (REQ-PERM-003)
Hash function per CT-MATH-001 §5.4:
```
h = seed
h = (h × 0x9E3779B9 + epoch)
h = (h × 0x85EBCA6B + round)
h = (h × 0xC2B2AE35 + value)
h = h ⊕ (h >> 16)
h = (h × 0x85EBCA6B)
h = h ⊕ (h >> 13)
```

### 3.4 Batch Construction (REQ-PERM-004)
B_t = { d_{π(t*B + j)} : j ∈ [0, B-1] }
Per CT-MATH-001 §5.6.

### 3.5 Bounded Iterations (REQ-PERM-005)
Max iterations = 2^k (where 2^k ≥ N).
Expected iterations < 2 (never triggered in practice).

### 3.6 Inverse (REQ-PERM-006)
π⁻¹ available for verification via reversed Feistel rounds.

---

## 4. Data Structures

```c
typedef struct {
    uint64_t seed;
    uint32_t epoch;
    uint32_t dataset_size;
    uint32_t half_bits;
    uint32_t half_mask;
    uint32_t range;
    bool initialized;
} ct_permutation_t;

typedef struct {
    ct_permutation_t perm;
    uint32_t batch_size;
    uint32_t steps_per_epoch;
} ct_batch_ctx_t;
```

---

## 5. Properties

| Property | Guarantee |
|----------|-----------|
| Bijection | Every i maps to unique j |
| Determinism | Same (seed, epoch, i) → same j |
| Termination | Bounded by 2^k iterations |
| Expected iterations | < 2 |

---

## 6. Test Coverage

39 unit tests covering:
- Initialization (basic, power of 2, non-power of 2, size 1)
- Feistel hash (determinism, sensitivity to inputs)
- Permutation apply (range, determinism, shuffling)
- Bijection verification (various sizes including primes)
- Inverse roundtrip
- Epoch changes permutation
- Batch operations (indices, size, epoch/step tracking)
- Large dataset (10,000 elements)
- Error handling
- Full determinism

---

## 7. Files

| File | Description |
|------|-------------|
| permutation.h | Public API |
| permutation.c | Cycle-Walking Feistel implementation |
| test_permutation.c | 39 unit tests |
| SRS-009-PERMUTATION.md | This document |

---

*End of SRS-009-PERMUTATION v1.0.0*
