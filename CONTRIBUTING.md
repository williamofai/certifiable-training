# Contributing to Certifiable Training

Thank you for your interest! We are building the world's first deterministic ML training system for safety-critical applications.

## 1. The Legal Bit (CLA)

All contributors must sign our **Contributor License Agreement (CLA)**.

**Why?** It allows SpeyTech to provide commercial licenses to companies that cannot use GPL code while keeping the project open source.

**How?** Our [CLA Assistant](https://cla-assistant.io/) will prompt you when you open your first Pull Request.

## 2. Coding Standards

All code must adhere to our **High-Integrity C Style Guide** (see `docs/style-guide.md`):

- **No Dynamic Allocation:** Do not use `malloc`, `free`, or `realloc` after initialization
- **MISRA-C Compliance:** Follow MISRA-C:2012 guidelines
- **Explicit Types:** Use `int32_t`, `uint32_t`, not `int` or `long`
- **Explicit Widening:** All 32-bit arithmetic via 64-bit intermediates
- **No Floating-Point:** Training path must be pure integer arithmetic
- **Bounded Loops:** All loops must have provable upper bounds

## 3. The Definition of Done

A PR is only merged when:

1. ✅ It is linked to a **Requirement ID** in `/docs/requirements/`
2. ✅ It has **100% Branch Coverage** in unit tests
3. ✅ It passes our **Bit-Perfect Test** (identical output on x86 and ARM)
4. ✅ It is **MISRA-C compliant**
5. ✅ It traces to **CT-MATH-001** or **CT-STRUCT-001**
6. ✅ It has been reviewed by the Project Lead

## 4. Documentation

Every function must document:
- Purpose
- Preconditions
- Postconditions
- Complexity (O(1), O(n), etc.)
- Determinism guarantee
- Traceability reference

Example:
```c
/**
 * @brief Neumaier compensated addition
 * 
 * @traceability CT-MATH-001 §9.3
 * 
 * Precondition: accum initialized via ct_comp_init()
 * Postcondition: accum.sum + accum.err approximates true sum
 * Complexity: O(1) time, O(1) space
 * Determinism: Bit-perfect across all platforms
 */
void ct_comp_add(ct_comp_accum_t *accum, fixed_acc_t v, ct_fault_flags_t *faults);
```

## 5. DVM Compliance

All training-path code must use DVM primitives (see `include/dvm.h`):

- `dvm_add()`, `dvm_sub()`, `dvm_mul()` — Widening arithmetic
- `dvm_div_int32()`, `dvm_div_q()` — Safe division
- `dvm_round_shift_rne()` — Unified rounding
- `dvm_clamp32()` — Saturation with fault flags
- `dvm_abs64_sat()` — Safe absolute value

**Never** use raw `+`, `-`, `*` on `int32_t` without widening.

## 6. Fault Handling

All arithmetic operations must:
1. Accept a `ct_fault_flags_t *faults` parameter
2. Set appropriate flags on overflow/underflow/div-zero
3. Return a deterministic value (even on fault)

## 7. Getting Started

Look for issues labeled `good-first-issue` or `dvm-layer`.

We recommend starting with:
- DVM primitive tests (test vectors from CT-MATH-001 §8.3)
- PRNG implementation
- Compensated summation

## Questions?

- **Technical questions:** Open an issue
- **General inquiries:** william@fstopify.com
- **Security issues:** Email william@fstopify.com (do not open public issues)

Thank you for helping make deterministic ML training a reality! 🎯
