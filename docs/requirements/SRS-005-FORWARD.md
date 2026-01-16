# SRS-005: Forward Pass

**Software Requirements Specification — Neural Network Layers**

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | SRS-005 |
| Version | 1.0.0 |
| Author | William Murray |
| Date | January 2026 |
| Status | Draft |
| Classification | Public |

---

## 1. Purpose

This document specifies the requirements for the Forward Pass module, which implements neural network layers using fixed-point arithmetic.

**Traceability**: CT-MATH-001 §7.1, §12; CT-STRUCT-001 §5, §6

---

## 2. Overview

### 2.1 Scope

The forward pass module provides:
- **Tensor operations**: Storage and manipulation of fixed-point arrays
- **Linear layer**: y = Wx + b
- **Activation functions**: ReLU, Sigmoid (LUT), Tanh (LUT)

### 2.2 Design Principles

1. **No floating-point at runtime**: All inference uses integer arithmetic
2. **LUT-based transcendentals**: Sigmoid/tanh use pre-computed tables
3. **Compensated accumulation**: Dot products use Neumaier summation
4. **Static allocation**: All buffers provided by caller

---

## 3. Requirements

### SRS-005.1: Tensor Structure

**Requirement**: The tensor structure shall contain:
- `data`: Pointer to fixed-point data (caller-provided)
- `dims[4]`: Dimension sizes (up to 4D)
- `strides[4]`: Element strides for views
- `ndims`: Number of dimensions
- `total_size`: Total element count

**Rationale**: CT-STRUCT-001 §5.1 specifies tensor layout.

**Verification**: Inspect `ct_tensor_t` definition.

### SRS-005.2: Tensor Initialization

**Requirement**: Functions shall initialize 1D and 2D tensors:
- `ct_tensor_init_1d()`: Vector initialization
- `ct_tensor_init_2d()`: Matrix initialization (row-major)

**Rationale**: Common shapes need convenient initialization.

**Verification**: Unit tests for 1D and 2D tensors.

### SRS-005.3: Linear Layer Structure

**Requirement**: The linear layer shall contain:
- `weights`: Weight tensor [output_size x input_size]
- `bias`: Bias tensor [output_size]
- `input_size`: Input dimension
- `output_size`: Output dimension

**Rationale**: CT-STRUCT-001 §6.1 specifies linear layer.

**Verification**: Inspect `ct_linear_t` definition.

### SRS-005.4: Linear Forward Pass

**Requirement**: `ct_linear_forward()` shall compute y = Wx + b where:
1. Matrix-vector multiply uses 64-bit intermediate accumulation
2. Accumulation uses compensated summation (Neumaier)
3. Result is rounded to Q16.16 using DVM_RoundShiftR_RNE
4. Bias is added using saturating addition

**Rationale**: CT-MATH-001 §7.1 specifies forward computation.

**Verification**: Unit tests for identity, bias, and general cases.

### SRS-005.5: ReLU Activation

**Requirement**: ReLU shall compute max(0, x) using:
- Integer comparison
- Conditional select
- No branching on value magnitude

**Rationale**: CT-MATH-001 §12.2 specifies ReLU.

**Verification**: Unit tests for positive, negative, zero inputs.

### SRS-005.6: ReLU Derivative

**Requirement**: ReLU derivative shall return:
- FIXED_ONE (1.0) if x > 0
- 0 if x ≤ 0

**Rationale**: Required for backward pass.

**Verification**: Derivative unit tests.

### SRS-005.7: Sigmoid LUT Specification

**Requirement**: The sigmoid lookup table shall have:
- Domain: [-8, +8]
- Table size: 257 entries
- Step size: 16/256 = 0.0625
- Format: Q16.16
- Saturation: Return 0 below -8, return FIXED_ONE above +8

**Rationale**: CT-MATH-001 §12.3 specifies sigmoid LUT.

**Verification**: Domain and saturation tests.

### SRS-005.8: Sigmoid Linear Interpolation

**Requirement**: `ct_sigmoid()` shall:
1. Saturate inputs outside [-8, +8]
2. Map input to table index
3. Compute fractional part for interpolation
4. Return y0 + frac * (y1 - y0) using integer ops

**Rationale**: CT-MATH-001 §12.3 specifies interpolation algorithm.

**Verification**: Monotonicity and accuracy tests.

### SRS-005.9: Sigmoid Accuracy

**Requirement**: Maximum error vs true sigmoid shall be < 0.002 within [-8, +8].

**Rationale**: CT-MATH-001 §12.4 specifies error bound.

**Verification**: Accuracy unit tests against math.h reference.

### SRS-005.10: Sigmoid Derivative

**Requirement**: Sigmoid derivative shall compute σ(x) * (1 - σ(x)) using:
- Pre-computed σ(x) value
- Fixed-point multiply with rounding

**Rationale**: CT-MATH-001 §12.5 specifies derivative computation.

**Verification**: Derivative calculation test.

### SRS-005.11: Tanh Activation

**Requirement**: Tanh shall use the same LUT structure as sigmoid with:
- Domain: [-8, +8]
- Saturation: Return -FIXED_ONE below -8, FIXED_ONE above +8
- Linear interpolation

**Rationale**: Same approach as sigmoid for consistency.

**Verification**: Zero point and saturation tests.

### SRS-005.12: Activation Layer Forward

**Requirement**: `ct_activation_forward()` shall:
- Apply activation element-wise
- Support in-place operation (input == output)
- Handle all activation types via type enum

**Rationale**: Layer abstraction for building networks.

**Verification**: Forward tests for ReLU and sigmoid layers.

### SRS-005.13: Determinism

**Requirement**: All forward pass operations shall produce bit-identical results:
- Across multiple executions
- Across different compilers
- Across different architectures

**Rationale**: Core requirement of Certifiable Training.

**Verification**: Determinism unit tests.

### SRS-005.14: Matrix-Vector Multiply

**Requirement**: `ct_matvec_mul()` shall:
1. For each row, compute dot product with input vector
2. Use 64-bit intermediate products
3. Accumulate using compensated summation
4. Round final sum to Q16.16

**Rationale**: Core operation for linear layers.

**Verification**: Matrix-vector multiply tests.

### SRS-005.15: Dot Product

**Requirement**: `ct_dot_product()` shall:
- Use compensated summation (Neumaier)
- Return result in Q16.16

**Rationale**: Precision critical for gradient computation.

**Verification**: Dot product accuracy tests.

---

## 4. Test Vectors

### 4.1 Linear Layer

| Input | Weights | Bias | Expected Output |
|-------|---------|------|-----------------|
| [1, 1, 1] | [[1,2,3],[4,5,6]] | [0,0] | [6, 15] |
| [1, 0] | [[1,0],[0,1]] | [0, 0] | [1, 0] |
| [1, 1] | [[2, 3]] | [10] | [15] |

### 4.2 ReLU

| Input | Output |
|-------|--------|
| 5.0 | 5.0 |
| -5.0 | 0 |
| 0 | 0 |

### 4.3 Sigmoid

| Input | Expected (approx) |
|-------|-------------------|
| 0 | 0.5 |
| -8 | ≈ 0 |
| +8 | ≈ 1 |
| -10 | 0 (saturated) |
| +10 | 1 (saturated) |

---

## 5. Implementation Mapping

### 5.1 Source Files

| File | Purpose |
|------|---------|
| `include/forward.h` | Public interface |
| `src/training/forward.c` | Implementation |
| `tests/unit/test_forward.c` | Unit tests |

### 5.2 Functions

| Function | Requirement |
|----------|-------------|
| `ct_tensor_init_1d()` | SRS-005.2 |
| `ct_tensor_init_2d()` | SRS-005.2 |
| `ct_linear_init()` | SRS-005.3 |
| `ct_linear_forward()` | SRS-005.4 |
| `ct_relu()` | SRS-005.5 |
| `ct_relu_derivative()` | SRS-005.6 |
| `ct_activation_init_sigmoid_lut()` | SRS-005.7 |
| `ct_sigmoid()` | SRS-005.8, SRS-005.9 |
| `ct_sigmoid_derivative()` | SRS-005.10 |
| `ct_tanh_act()` | SRS-005.11 |
| `ct_activation_forward()` | SRS-005.12 |
| `ct_matvec_mul()` | SRS-005.14 |
| `ct_dot_product()` | SRS-005.15 |

---

## 6. Traceability

### 6.1 Upstream (Math Specification)

| Requirement | CT-MATH-001 Section |
|-------------|---------------------|
| SRS-005.4 | §7.1 Forward Pass |
| SRS-005.5 | §12.2 ReLU |
| SRS-005.7-9 | §12.3 Sigmoid LUT |
| SRS-005.10 | §12.5 Sigmoid Derivative |

### 6.2 Dependencies

| Module | Relationship |
|--------|--------------|
| SRS-001 DVM | Uses dvm_add, dvm_round_shift_rne |
| SRS-003 Compensated | Uses ct_comp_add for dot products |

### 6.3 Downstream

| Module | Relationship |
|--------|--------------|
| SRS-006 Backward | Uses forward outputs and caches |

### 6.4 Compliance Mapping

| Standard | Relevance |
|----------|-----------|
| DO-178C | No floating-point at runtime |
| IEC 62304 | Deterministic output |
| ISO 26262 | Bounded error in activations |
| MISRA-C:2012 | Coding standard compliance |

---

## 7. Design Notes

### 7.1 LUT Initialization

The sigmoid and tanh LUTs are initialized using `math.h` functions (exp, tanh). This is acceptable because:
- Initialization happens once at startup
- The LUT values become fixed constants
- Runtime never uses floating-point

The LUT values could alternatively be pre-computed and stored as C arrays.

### 7.2 Compensated Dot Products

Using Neumaier summation for dot products is critical because:
- Many small products are accumulated
- Fixed-point rounding would otherwise accumulate error
- Gradient computation magnifies any forward pass errors

### 7.3 In-Place Activation

Activation functions support in-place operation (output = input) because:
- Reduces memory requirements
- Common pattern in inference
- Safe for element-wise operations

---

## 8. Quality Criteria

### 8.1 Test Coverage

- [ ] All functions have unit tests
- [ ] Tensor operations tested
- [ ] Linear layer tested
- [ ] All activations tested
- [ ] Determinism verified
- [ ] Accuracy bounds verified

### 8.2 Code Quality

- [ ] Zero warnings with strict flags
- [ ] No floating-point at runtime (except LUT init)
- [ ] All functions documented
- [ ] Traceability tags present

### 8.3 Performance

- [ ] O(n*m) for linear layer (n outputs, m inputs)
- [ ] O(1) for activation functions
- [ ] No dynamic allocation

---

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | Jan 2026 | William Murray | Initial release |

---

*End of SRS-005*
