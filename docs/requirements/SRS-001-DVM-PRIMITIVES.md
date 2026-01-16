# SRS-001: DVM Arithmetic Primitives

**Requirement ID:** SRS-001  
**Type:** System Requirement  
**Priority:** Critical  
**Status:** Implemented

## Description

The Deterministic Virtual Machine (DVM) shall provide arithmetic primitives that produce identical outputs for identical inputs across all supported platforms and builds, with mandatory fault signaling on overflow, underflow, and division by zero.

## Rationale

Safety-critical ML training requires:
- **Bit-identical results** across x86, ARM, and RISC-V
- **No undefined behavior** from integer overflow
- **Fail-closed semantics** with explicit fault detection
- **Bounded error** through deterministic rounding

Non-deterministic arithmetic prevents certification under DO-178C, IEC 62304, and ISO 26262.

## Requirements

### SRS-001.1: Widening Arithmetic

All 32-bit arithmetic shall use 64-bit intermediates to avoid undefined behavior.

**Rationale:** C standard defines signed overflow as undefined behavior. Explicit widening ensures deterministic saturation.

```c
// Required pattern
int64_t wide = (int64_t)a + (int64_t)b;
return dvm_clamp32(wide, faults);

// Forbidden pattern
return a + b;  // UB on overflow
```

### SRS-001.2: Saturation with Fault Signaling

When a 64-bit result exceeds 32-bit range, the system shall:
1. Return the saturated value (INT32_MAX or INT32_MIN)
2. Set the appropriate fault flag (overflow or underflow)

**Rationale:** Silent saturation hides errors. Fault flags enable fail-closed behavior.

### SRS-001.3: Division by Zero Handling

Division by zero shall:
1. Return zero (deterministic value)
2. Set the div_zero fault flag

**Rationale:** Exceptions and traps vary across platforms. Deterministic return with fault flag enables portable error handling.

### SRS-001.4: Safe Absolute Value

`dvm_abs64_sat(INT64_MIN)` shall:
1. Return INT64_MAX (saturated)
2. Set the overflow fault flag

**Rationale:** `abs(INT64_MIN)` is undefined in C. Required for Neumaier summation magnitude comparisons.

### SRS-001.5: Unified Rounding (RNE)

All fixed-point scaling shall use Round-to-Nearest, Ties-to-Even:
- Shift domain: 0 ≤ shift ≤ 62
- Out-of-range shifts shall set domain fault flag

**Rationale:** Eliminates statistical bias. Single rounding primitive ensures consistency.

### SRS-001.6: Fixed-Point Multiplication

`dvm_mul(a, b)` shall compute `(a × b) >> FRAC_BITS` using:
1. 64-bit intermediate product
2. RNE rounding for the shift
3. 32-bit saturation on result

### SRS-001.7: Fixed-Point Division

`dvm_div_q(a, b, frac_bits)` shall compute `(a << frac_bits) / b` with:
1. Division by zero check (return 0, set fault)
2. Shift bounds check (frac_bits ≤ 62)
3. 32-bit saturation on result

## Verification

### V-001.1: Cross-Platform Identity
- Compile and run test suite on x86_64, ARM64, RISC-V
- Compare output checksums (must be identical)

### V-001.2: Fault Flag Coverage
- Test overflow on INT32_MAX + 1
- Test underflow on INT32_MIN - 1
- Test div_zero on a / 0
- Test domain on shift > 62
- Verify flags are sticky until cleared

### V-001.3: Rounding Test Vectors
Mandatory test vectors from CT-MATH-001 §8.3:

| Input (hex) | Shift | Expected | Description |
|-------------|-------|----------|-------------|
| 0x00018000 | 16 | 2 | 1.5 → 2 (even) |
| 0x00028000 | 16 | 2 | 2.5 → 2 (even) |
| 0x00038000 | 16 | 4 | 3.5 → 4 (even) |
| 0xFFFE8000 | 16 | -2 | -1.5 → -2 (even) |
| 0xFFFD8000 | 16 | -2 | -2.5 → -2 (even) |

### V-001.4: Boundary Conditions
- INT32_MAX + INT32_MAX → saturate, set overflow
- INT32_MIN - INT32_MAX → saturate, set underflow
- INT64_MIN absolute value → INT64_MAX, set overflow

## Implementation

### Source Files
- `src/dvm/primitives.c` — All DVM arithmetic primitives

### Header Files
- `include/ct_types.h` — Fixed-point types, fault flags, error codes
- `include/dvm.h` — DVM function declarations

### Functions

| Function | Description | Ref |
|----------|-------------|-----|
| `dvm_add()` | Widening addition with saturation | CT-MATH-001 §3.1 |
| `dvm_sub()` | Widening subtraction with saturation | CT-MATH-001 §3.2 |
| `dvm_mul()` | Fixed-point multiplication | CT-MATH-001 §3.3 |
| `dvm_div_int32()` | Integer division | CT-MATH-001 §3.4 |
| `dvm_div_q()` | Fixed-point division | CT-MATH-001 §3.5 |
| `dvm_clamp32()` | 64→32 saturation with fault | CT-MATH-001 §3.6 |
| `dvm_abs64_sat()` | Safe absolute value | CT-MATH-001 §3.7 |
| `dvm_round_shift_rne()` | RNE rounding shift | CT-MATH-001 §8.1 |

## Tests

- `tests/unit/test_primitives.c` — Unit tests for all primitives
- `tests/unit/test_bit_identity.c` — Cross-platform identity verification

## Traceability

### Upstream (Mathematical Specification)
- CT-MATH-001 §3 — DVM Arithmetic Primitives
- CT-MATH-001 §8 — Deterministic Rounding
- CT-STRUCT-001 §11 — Fault Model

### Downstream (Dependent Modules)
- SRS-002 (PRNG) — Uses dvm_mul for mixing
- SRS-003 (Compensated Summation) — Uses dvm_abs64_sat
- SRS-004 (Reduction Tree) — Uses all arithmetic primitives
- SRS-005 (Forward Pass) — Uses dvm_mul, dvm_add
- SRS-006 (Backward Pass) — Uses all arithmetic primitives

### Compliance Mapping

| Standard | Requirement | How Addressed |
|----------|-------------|---------------|
| DO-178C | Deterministic execution | Bit-identical across platforms |
| DO-178C | Bounded behavior | Saturation, no UB |
| IEC 62304 | Fault handling | Sticky fault flags |
| ISO 26262 | Fail-safe | Deterministic return on fault |
| MISRA-C | No UB | Explicit widening |

---

**Author:** William Murray  
**Date:** January 2026  
**Version:** 1.0  
**Copyright:** © 2026 The Murray Family Innovation Trust. All rights reserved.
