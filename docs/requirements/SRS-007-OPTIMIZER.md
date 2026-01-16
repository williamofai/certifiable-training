# SRS-007: Optimizers

**Certifiable Training — Software Requirements Specification**

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | SRS-007-OPTIMIZER |
| Version | 1.0.0 |
| Author | William Murray |
| Date | January 2026 |
| Status | Implementation Complete |

---

## 1. Purpose

This document specifies deterministic optimizers for Certifiable Training: SGD, SGD with Momentum, and Adam. All use DVM primitives for bit-identical results across platforms.

---

## 2. References

| Document | Section |
|----------|---------|
| CT-MATH-001 | §10 Training Operator, §13 Square Root |
| CT-STRUCT-001 | §7 Optimizer Structures |
| SRS-001 | DVM Primitives |
| SRS-006 | Backward Pass |

---

## 3. Requirements

### 3.1 SGD (REQ-OPT-001)
Update rule: θ = θ - η * (g + λ * θ)
- Learning rate η
- Optional weight decay λ

### 3.2 SGD with Momentum (REQ-OPT-002)
Update rule:
- v = β * v + g
- θ = θ - η * (v + λ * θ)
- Momentum coefficient β (typically 0.9)
- Caller-provided velocity buffer

### 3.3 Adam (REQ-OPT-003)
Update rule per CT-MATH-001 §10.4:
- m = β₁ * m + (1-β₁) * g
- v = β₂ * v + (1-β₂) * g²
- m̂ = m / (1 - β₁^t)
- v̂ = v / (1 - β₂^t)
- θ = θ - η * m̂ / (√v̂ + ε)

### 3.4 Fixed-Point Square Root (REQ-OPT-004)
- Newton-Raphson with fixed 8 iterations
- No data-dependent branching
- Per CT-MATH-001 §13

### 3.5 Non-Functional Requirements
- **REQ-OPT-010**: Bit-identical determinism
- **REQ-OPT-011**: No dynamic allocation
- **REQ-OPT-012**: Fault propagation via flags

---

## 4. Data Structures

```c
typedef struct {
    fixed_t learning_rate;
    fixed_t weight_decay;
} ct_sgd_config_t;

typedef struct {
    fixed_t learning_rate;
    fixed_t momentum;
    fixed_t weight_decay;
} ct_sgd_momentum_config_t;

typedef struct {
    fixed_t learning_rate;
    fixed_t beta1;
    fixed_t beta2;
    fixed_t epsilon;
    fixed_t weight_decay;
} ct_adam_config_t;
```

---

## 5. Constants

| Constant | Value | Description |
|----------|-------|-------------|
| CT_OPT_DEFAULT_LR | 655 | 0.01 in Q16.16 |
| CT_OPT_DEFAULT_MOMENTUM | 58982 | 0.9 in Q16.16 |
| CT_OPT_ADAM_BETA1 | 58982 | 0.9 in Q16.16 |
| CT_OPT_ADAM_BETA2 | 65471 | 0.999 in Q16.16 |
| CT_OPT_ADAM_EPSILON | 1 | ~1.5e-5 in Q16.16 |
| CT_OPT_SQRT_ITERATIONS | 8 | Fixed iteration count |

---

## 6. Test Coverage

31 unit tests covering:
- Square root (zero, one, four, quarter, negative)
- SGD configuration and steps
- SGD with Momentum accumulation
- Adam bias correction and moment updates
- Reset functions
- Error handling
- Determinism verification

---

## 7. Files

| File | Description |
|------|-------------|
| optimizer.h | Public API |
| optimizer.c | Implementation |
| test_optimizer.c | 31 unit tests |
| SRS-007-OPTIMIZER.md | This document |

---

*End of SRS-007-OPTIMIZER v1.0.0*
