# SRS-006: Backward Pass (Backpropagation)

**Certifiable Training — Software Requirements Specification**

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | SRS-006-BACKWARD |
| Version | 1.0.0 |
| Author | William Murray |
| Date | January 2026 |
| Status | Implementation Complete |

---

## 1. Purpose

This document specifies the backward pass (backpropagation) module for Certifiable Training. The backward pass computes gradients for all trainable parameters using Q8.24 high-precision fixed-point arithmetic.

---

## 2. References

| Document | Section |
|----------|---------|
| CT-MATH-001 | §7 Gradient Computation, §12 Activations, §14 Loss Functions |
| CT-STRUCT-001 | §5 Tensor Structures, §6 Layer Structures |
| SRS-001 | DVM Primitives |
| SRS-003 | Compensated Summation |
| SRS-005 | Forward Pass |

---

## 3. Requirements

### 3.1 Gradient Format (REQ-BACK-001)
Gradients use Q8.24 fixed-point for higher precision on small values.

### 3.2 Format Conversion (REQ-BACK-002)
- `ct_fixed_to_grad()`: Q16.16 → Q8.24 (lossless)
- `ct_grad_to_fixed()`: Q8.24 → Q16.16 (rounded)

### 3.3 Loss Functions
- **REQ-BACK-003**: MSE Forward: L = (1/N) Σᵢ (ŷᵢ - yᵢ)²
- **REQ-BACK-004**: MSE Backward: ∂L/∂ŷᵢ = (2/N)(ŷᵢ - yᵢ)

### 3.4 Activation Derivatives
- **REQ-BACK-005**: ReLU: 1 if x > 0, else 0
- **REQ-BACK-006**: Sigmoid: σ'(x) = σ(x)(1 - σ(x))
- **REQ-BACK-007**: Tanh: 1 - tanh²(x)

### 3.5 Linear Layer Backward (REQ-BACK-008)
- grad_input = Wᵀ · grad_output
- grad_weights = grad_output · inputᵀ  
- grad_bias = grad_output

### 3.6 Gradient Processing
- **REQ-BACK-009**: Clipping to range
- **REQ-BACK-010**: Scaling by constant
- **REQ-BACK-011**: L2 norm computation

### 3.7 Health Monitoring (REQ-BACK-020)
Vanishing gradient detection: warn if >5% gradients are zero.

### 3.8 Non-Functional
- **REQ-BACK-030**: Bit-identical determinism
- **REQ-BACK-031**: No dynamic allocation
- **REQ-BACK-032**: Compensated accumulation

---

## 4. Data Structures

```c
typedef struct {
    fixed_hp_t *data;
    uint32_t dims[CT_MAX_DIMS];
    uint32_t strides[CT_MAX_DIMS];
    uint32_t ndims;
    uint32_t total_size;
} ct_grad_tensor_t;

typedef struct {
    uint64_t zero_grad_count;
    uint64_t total_grad_count;
    fixed_hp_t min_nonzero_grad;
    fixed_hp_t max_grad;
} ct_grad_health_t;
```

---

## 5. Constants

| Constant | Value | Description |
|----------|-------|-------------|
| CT_GRAD_FRAC_BITS | 24 | Q8.24 fractional bits |
| CT_GRAD_ONE | 16777216 | 1.0 in Q8.24 |
| CT_GRAD_FLOOR_THRESHOLD_PERCENT | 5 | Vanishing warning |

---

## 6. Test Coverage

28 unit tests covering:
- Tensor initialization and accessors
- Format conversion roundtrip
- MSE loss forward/backward
- All activation derivatives
- Linear layer gradients
- Gradient clipping/scaling/norm
- Health monitoring
- Error handling
- Determinism verification

---

## 7. Files

| File | Description |
|------|-------------|
| backward.h | Public API |
| backward.c | Implementation |
| test_backward.c | 28 unit tests |
| SRS-006-BACKWARD.md | This document |

---

*End of SRS-006-BACKWARD v1.0.0*
