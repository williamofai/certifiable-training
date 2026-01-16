# SRS-002: Counter-Based PRNG

**Requirement ID:** SRS-002  
**Type:** System Requirement  
**Priority:** Critical  
**Status:** Implemented

## Description

The system shall provide a counter-based pseudo-random number generator (PRNG) that produces identical output sequences for identical inputs (seed, op_id, step) across all platforms.

## Rationale

Deterministic "randomness" is required for:
- **Stochastic rounding** — Regularization without execution variance
- **Data shuffling** — Reproducible batch construction via Feistel permutation
- **Dropout** — If implemented, must be reproducible

Traditional PRNGs maintain internal state that varies with execution timing. Counter-based PRNGs are pure functions, enabling bit-identical replay.

## Requirements

### SRS-002.1: Pure Function Semantics

`ct_prng_peek(prng, step)` shall be a pure function of (seed, op_id, step):
- Same inputs always produce same 32-bit output
- No dependency on execution history
- No side effects

### SRS-002.2: Sequence Advancement

`ct_prng_next(prng)` shall:
1. Return `ct_prng_peek(prng, prng->step)`
2. Increment `prng->step` by 1

### SRS-002.3: Statistical Quality

The output shall:
- Cover all 32 bits (each bit set/clear at least once in reasonable samples)
- Not be constant, incrementing, or trivially patterned
- Pass basic randomness sanity checks

**Note:** Cryptographic quality is NOT required. Statistical quality sufficient for ML training is the goal.

### SRS-002.4: Operation Identifier Uniqueness

`ct_prng_make_op_id(layer_id, tensor_id, element_idx)` shall produce:
- Different outputs for different inputs
- Identical outputs for identical inputs

### SRS-002.5: Stochastic Rounding

`ct_stochastic_round(x, shift, prng, faults)` shall:
1. Check shift bounds (0 ≤ shift ≤ 62), fault if invalid
2. Extract fractional part of x
3. Generate threshold from PRNG
4. Round up if fraction > threshold, else round down
5. Return clamped 32-bit result

**Property:** Probability of rounding up equals fractional value.

### SRS-002.6: Null Safety

All functions shall handle NULL pointers gracefully:
- `ct_prng_init(NULL, ...)` — No-op
- `ct_prng_next(NULL)` — Return 0
- `ct_prng_peek(NULL, ...)` — Return 0
- `ct_stochastic_round(..., NULL, ...)` — Fall back to truncation

## Verification

### V-002.1: Determinism Test
```
prng1 = init(seed=S, op_id=O)
prng2 = init(seed=S, op_id=O)
for i in 1..1000:
    assert next(prng1) == next(prng2)
```

### V-002.2: Different Seed Produces Different Sequence
```
prng1 = init(seed=1, op_id=0)
prng2 = init(seed=2, op_id=0)
different_count = sum(next(prng1) != next(prng2) for i in 1..100)
assert different_count > 90
```

### V-002.3: Peek Matches Next
```
prng = init(seed=42, op_id=0)
for i in 0..9:
    peeked[i] = peek(prng, i)
for i in 0..9:
    assert next(prng) == peeked[i]
```

### V-002.4: Stochastic Rounding Probability
```
prng = init(seed=42, op_id=0)
half = 0x8000  // 0.5 in Q16.16
round_up_count = 0
for i in 1..10000:
    if stochastic_round(half, 16, prng) == 1:
        round_up_count++
assert 4500 <= round_up_count <= 5500  // ~50% within tolerance
```

### V-002.5: Cross-Platform Verification
Record PRNG output on reference platform (x86_64 Linux GCC):
```
prng = init(seed=0, op_id=0)
v0 = next(prng)  // Record hex value
v1 = next(prng)
v2 = next(prng)
v3 = next(prng)
v4 = next(prng)
```
Verify identical values on ARM64, RISC-V.

## Implementation

### Source Files
- `src/dvm/prng.c` — PRNG implementation

### Header Files
- `include/prng.h` — PRNG declarations
- `include/ct_types.h` — ct_prng_t structure

### Functions

| Function | Description | Ref |
|----------|-------------|-----|
| `ct_prng_init()` | Initialize PRNG state | CT-MATH-001 §6.1 |
| `ct_prng_next()` | Generate and advance | CT-MATH-001 §6.2 |
| `ct_prng_peek()` | Generate without advancing | CT-MATH-001 §6.2 |
| `ct_stochastic_round()` | Deterministic stochastic rounding | CT-MATH-001 §8.4 |
| `ct_prng_make_op_id()` | Compute operation identifier | CT-MATH-001 §6.3 |

### Algorithm

Philox-style counter-based PRNG:
```
PRNG(seed, op_id, step):
    ctr = (op_id << 32) | (step & 0xFFFFFFFF)
    key = seed
    for r in 0..9:
        ctr = (ctr * 0xD2511F53) ^ key
        key = (key * 0xCD9E8D57) + 0x9E3779B9
    return ctr & 0xFFFFFFFF
```

## Tests

- `tests/unit/test_prng.c` — 18 test cases covering all requirements

## Traceability

### Upstream (Mathematical Specification)
- CT-MATH-001 §6 — Counter-Based PRNG
- CT-MATH-001 §8.4 — Stochastic Rounding
- CT-STRUCT-001 §4.1 — ct_prng_t structure

### Downstream (Dependent Modules)
- SRS-004 (Reduction Tree) — May use PRNG for stochastic rounding
- SRS-009 (Permutation) — Uses PRNG via Feistel construction

### Compliance Mapping

| Standard | Requirement | How Addressed |
|----------|-------------|---------------|
| DO-178C | Deterministic execution | Pure function, no hidden state |
| DO-178C | Reproducibility | Same inputs → same outputs |
| IEC 62304 | Traceability | Each function traces to CT-MATH-001 |
| ISO 26262 | Testability | All requirements have verification methods |

---

**Author:** William Murray  
**Date:** January 2026  
**Version:** 1.0  
**Copyright:** © 2026 The Murray Family Innovation Trust. All rights reserved.
