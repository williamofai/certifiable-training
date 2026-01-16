# SRS-003: Compensated Summation

**Software Requirements Specification — Neumaier Compensated Arithmetic**

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | SRS-003 |
| Version | 1.0.0 |
| Author | William Murray |
| Date | January 2026 |
| Status | Draft |
| Classification | Public |

---

## 1. Purpose

This document specifies the requirements for the Compensated Summation module, which implements the Neumaier algorithm for high-precision accumulation using only integer operations.

**Traceability**: CT-MATH-001 §9, CT-STRUCT-001 §4.2

---

## 2. Overview

### 2.1 Problem Statement

When summing many values in fixed-point arithmetic, rounding errors accumulate. In gradient aggregation, a batch of 64 samples might produce 64 gradient values that must be summed. Naive summation loses precision when:

1. Large and small values are mixed
2. Many small values are summed
3. Values of opposite sign partially cancel

### 2.2 Solution

The Neumaier algorithm tracks rounding errors in a separate compensation term. After all additions, the error term is added back to recover lost precision.

**Key insight**: In integer arithmetic, the "lost bits" from each addition can be computed exactly and accumulated.

### 2.3 Why Neumaier over Kahan

Kahan summation assumes the new value is smaller than the running sum. Neumaier handles the general case where the new value may be larger, making it more robust for arbitrary input sequences.

---

## 3. Requirements

### SRS-003.1: Accumulator Structure

**Requirement**: The compensated accumulator shall consist of two 64-bit signed integers: `sum` and `err`.

**Rationale**: CT-MATH-001 §9.2 specifies the (sum, error) pair structure. 64-bit provides sufficient range for Q32.32 extended precision.

**Verification**: Inspect `ct_comp_accum_t` structure definition.

### SRS-003.2: Initialization

**Requirement**: The module shall provide initialization functions that:
- Set both `sum` and `err` to zero (default init)
- Set `sum` to a specified value and `err` to zero (value init)
- Handle NULL pointers without crashing

**Rationale**: Safe initialization is required for deterministic behavior.

**Verification**: Unit tests `test_init_zeros`, `test_init_value`, `test_init_null_safe`.

### SRS-003.3: Neumaier Compensated Addition

**Requirement**: The `ct_comp_add()` function shall implement the Neumaier algorithm exactly as specified in CT-MATH-001 §9.3:

```
CompAdd(accum, v):
    t = accum.sum + v
    
    if |accum.sum| >= |v|:
        e = (accum.sum - t) + v
    else:
        e = (v - t) + accum.sum
    
    accum.sum = t
    accum.err = accum.err + e
```

**Rationale**: The Neumaier variant handles cases where `v` > `sum`, which Kahan does not.

**Verification**: Unit tests for large+small, small+large, and alternating patterns.

### SRS-003.4: Safe Absolute Value

**Requirement**: Magnitude comparison shall use `DVM_Abs64_Sat()` which:
- Returns `|x|` for all values except `INT64_MIN`
- Returns `INT64_MAX` for `INT64_MIN` (saturate rather than undefined behavior)
- Optionally sets overflow fault flag for `INT64_MIN` case

**Rationale**: CT-MATH-001 §9.3 specifies safe absolute value. `abs(INT64_MIN)` is undefined in C.

**Verification**: Overflow handling tests.

### SRS-003.5: Accumulator Merge

**Requirement**: The `ct_comp_merge()` function shall combine two accumulators by:
1. Adding `src.sum` to `dest` using compensated addition
2. Adding `src.err` to `dest.err`

**Rationale**: Required for tree-based parallel reduction where child nodes must be merged.

**Verification**: Unit tests `test_merge_basic`, `test_merge_preserves_error`.

### SRS-003.6: Finalization

**Requirement**: The `ct_comp_finalize()` function shall return `sum + err` as the final high-precision result.

**Rationale**: The true accumulated value is the sum of both terms.

**Verification**: All tests that check final values use finalization.

### SRS-003.7: Array Summation

**Requirement**: Convenience functions shall sum arrays using compensated arithmetic:
- `ct_comp_sum_array()`: Sum 64-bit values
- `ct_comp_sum_array_32()`: Sum 32-bit values (widened internally)
- `ct_comp_mean_array()`: Compute mean

**Rationale**: Common operations should be provided for ease of use.

**Verification**: Array operation unit tests.

### SRS-003.8: Batch Size Limit

**Requirement**: Array functions shall check for batch sizes exceeding `CT_MAX_BATCH_SIZE` (65536) and set the domain fault flag if exceeded.

**Rationale**: CT-MATH-001 §9.5 specifies batch size limits to prevent accumulator exhaustion.

**Verification**: Test with large batch size.

### SRS-003.9: Determinism

**Requirement**: For identical input sequences, compensated summation shall produce bit-identical results across:
- Multiple executions on the same platform
- Different compilers (GCC, Clang)
- Different architectures (x86, ARM, RISC-V)

**Rationale**: Core requirement of Certifiable Training.

**Verification**: Determinism unit tests, cross-platform verification.

### SRS-003.10: Overflow Handling

**Requirement**: Overflow conditions shall:
- Saturate to `INT64_MAX` or `INT64_MIN`
- Set appropriate fault flag (`overflow` or `underflow`)
- Not cause undefined behavior

**Rationale**: Fail-closed design per CT-SPEC-001 §3.3.

**Verification**: Edge case tests for `INT64_MAX` and `INT64_MIN`.

---

## 4. Test Vectors

### 4.1 Basic Summation

| Values | Expected Sum |
|--------|--------------|
| [10, 20, 30, 40, 50] | 150 |
| [100, 200, 300, 400, 500] | 1500 |
| [1, 2, ..., 1000] | 500500 |
| [0, 1, 2, ..., 9999] | 49995000 |

### 4.2 Compensation Tests

| Operation | Expected |
|-----------|----------|
| (2^40) + 1000×1 | 2^40 + 1000 |
| 1000×1 + (2^40) | 2^40 + 1000 |
| 100×[(2^30) + 1] | 100×2^30 + 100 |

### 4.3 Edge Cases

| Input | Expected Behavior |
|-------|-------------------|
| INT64_MAX + 1 | Saturate, set overflow |
| INT64_MIN - 1 | Saturate, set underflow |
| 1000×(+1) + 1000×(-1) | 0 |

---

## 5. Implementation Mapping

### 5.1 Source Files

| File | Purpose |
|------|---------|
| `include/compensated.h` | Public interface |
| `src/dvm/compensated.c` | Implementation |
| `tests/unit/test_compensated.c` | Unit tests |

### 5.2 Functions

| Function | Requirement |
|----------|-------------|
| `ct_comp_init()` | SRS-003.2 |
| `ct_comp_init_value()` | SRS-003.2 |
| `ct_comp_add()` | SRS-003.3, SRS-003.4 |
| `ct_comp_merge()` | SRS-003.5 |
| `ct_comp_finalize()` | SRS-003.6 |
| `ct_comp_sum_array()` | SRS-003.7, SRS-003.8 |
| `ct_comp_sum_array_32()` | SRS-003.7, SRS-003.8 |
| `ct_comp_mean_array()` | SRS-003.7 |

---

## 6. Traceability

### 6.1 Upstream (Math Specification)

| Requirement | CT-MATH-001 Section |
|-------------|---------------------|
| SRS-003.1 | §9.2 Compensated Accumulator |
| SRS-003.3 | §9.3 Neumaier Summation |
| SRS-003.4 | §9.3 DVM_Abs64_Sat |
| SRS-003.8 | §9.5 Accumulator Exhaustion |

### 6.2 Downstream (Dependencies)

| Module | Dependency |
|--------|------------|
| SRS-004 Reduction Tree | Uses compensated merge |
| SRS-006 Backward Pass | Uses gradient aggregation |
| SRS-007 Optimizer | Uses mean computation |

### 6.3 Compliance Mapping

| Standard | Relevance |
|----------|-----------|
| DO-178C | Traceability, WCET analysis |
| IEC 62304 | Software unit verification |
| ISO 26262 | Fault detection |
| MISRA-C:2012 | Coding standard compliance |

---

## 7. Quality Criteria

### 7.1 Test Coverage

- [ ] All functions have unit tests
- [ ] All error paths tested
- [ ] Boundary conditions tested
- [ ] Determinism verified

### 7.2 Code Quality

- [ ] Zero warnings with strict flags
- [ ] MISRA-C compliant
- [ ] All functions documented
- [ ] Traceability tags present

### 7.3 Performance

- [ ] O(1) per addition
- [ ] O(n) for array summation
- [ ] No dynamic allocation

---

## 8. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | Jan 2026 | William Murray | Initial release |

---

*End of SRS-003*
