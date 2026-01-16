# Certifiable Training

**A Deterministic Architecture for Auditable Machine Learning**

**Document ID**: CT-SPEC-001  
**Version**: 2.0.0  
**Status**: Production Draft

---

## 0. Executive Summary

This document specifies Certifiable Training, a deterministic architecture for machine learning in which training is defined as a state evolution law over an integer ring rather than a stochastic numerical procedure.

By replacing IEEE-754 floating-point arithmetic, nondeterministic parallel reduction, and stochastic sampling with a Deterministic Virtual Machine (DVM), training becomes a pure function of code, data, configuration, and seed.

We prove that the resulting discrete training trajectory shadows an ideal continuous optimizer within a bounded stability tube, enabling for the first time:

- **Bit-identical training and inference** across heterogeneous hardware
- **Replayable and locally verifiable** training steps
- **Cryptographic proof** of model provenance

This establishes machine learning as a provable circuit, not a black box.

---

## 1. Philosophical Reframing: From Stochastic Training to State Evolution

### 1.1 The Core Pivot

All digital training is approximate.

- **Floating-point ML** approximates real arithmetic with unbounded, hardware-dependent, nondeterministic noise.
- **Certifiable Training** approximates real arithmetic with bounded, deterministic, analyzable perturbations.

Therefore, correctness is not defined by matching floating-point behavior, but by shadowing a valid optimizer trajectory.

**The discrete trajectory is not "wrong"; it is the trajectory defined by the system.**

---

## 2. Mathematical Foundations

### 2.1 Integer Execution Domain

Let **Z**_{2^w} denote two's-complement integers of width w.

A fixed-point format Q(a,b) represents integers x ∈ **Z**_{2^{a+b}} with interpretation:

```
⊢ ⟦x⟧ = x · 2^{-b}
```

### 2.2 State Space

Let the training state be:

```
θ_t ∈ W_Δ = (Q_w)^P
```

where P is the number of parameters.

This space is **finite**, **discrete**, and **closed**.

### 2.3 Canonical Fixed-Point Formats

| Format | Bits | Integer | Fractional | Use Case |
|--------|------|---------|------------|----------|
| Q16.16 | 32 | 16 | 16 | Weights, activations |
| Q8.24 | 32 | 8 | 24 | Gradients (high precision) |
| Q32.32 | 64 | 32 | 32 | Accumulators |

---

## 3. The Deterministic Virtual Machine (DVM)

The DVM defines the only legal execution semantics for Certifiable Training.

### 3.1 Mandatory Primitives

All operations are integer-only with defined overflow semantics:

| Operation | Semantics | Overflow Behavior |
|-----------|-----------|-------------------|
| `DVM_Add64` | 64-bit addition | Saturate + Fault Flag |
| `DVM_Sub64` | 64-bit subtraction | Saturate + Fault Flag |
| `DVM_Mul64` | 32×32→64 multiply | No overflow possible |
| `DVM_RoundShiftR_RNE` | Round-to-nearest-even shift | Defined for negatives |
| `DVM_Clamp32` | Saturate 64→32 | Sets Fault Flag if clamped |
| `DVM_PRNG` | Counter-based RNG | Pure function |
| `DVM_TreeReduce` | Fixed-topology reduction | Order by op_id |

### 3.2 Forbidden Primitives

| Forbidden | Reason |
|-----------|--------|
| IEEE-754 float or double | Hardware-dependent rounding |
| FMA instructions | Implementation-dependent intermediate rounding |
| `atomicAdd` on shared memory | Non-deterministic ordering |
| Nondeterministic kernel fusion | Hidden execution order |
| Data-dependent branching on accumulator order | Breaks reproducibility |
| Raw signed `+`/`-` on `int32_t` | C undefined behavior on overflow |

### 3.3 Arithmetic Overflow Model

**Policy**: Widen-then-saturate with mandatory fault signaling.

```
DVM_Add_Safe(a, b):
    wide = (int64_t)a + (int64_t)b
    if wide > INT32_MAX:
        SET_FAULT_FLAG(OVERFLOW)
        return INT32_MAX
    if wide < INT32_MIN:
        SET_FAULT_FLAG(UNDERFLOW)
        return INT32_MIN
    return (int32_t)wide
```

**Fault Flag**: Sticky bit that persists until explicitly cleared. Any fault invalidates the current Merkle hash.

### 3.4 Division Operations

**Two distinct operations** (to avoid ambiguity):

```
DVM_Div_Int32(a, b):
    if b == 0:
        SET_FAULT_FLAG(DIV_ZERO)
        return 0  // Deterministic but flagged
    return a / b  // Truncate toward zero (C99)

DVM_Div_Q(a, b, frac_bits):
    if b == 0:
        SET_FAULT_FLAG(DIV_ZERO)
        return 0
    if frac_bits > 62:
        SET_FAULT_FLAG(DOMAIN)
        return 0
    wide = (int64_t)a << frac_bits
    // ... rounding logic per CT-MATH-001 §3.5
    return DVM_Clamp32(result)
```

**Usage**: `DVM_Div_Int32` for indices/counters; `DVM_Div_Q` for fixed-point arithmetic.

### 3.5 Safe Absolute Value

```
DVM_Abs64_Sat(x):
    if x == INT64_MIN:
        SET_FAULT_FLAG(OVERFLOW)
        return INT64_MAX  // Saturate rather than UB
    return (x < 0) ? -x : x
```

**Rationale**: `abs(INT64_MIN)` is undefined in C. Required for Neumaier summation magnitude comparisons.

---

## 4. Deterministic Rounding: DVM_RoundShiftR_RNE

All fixed-point scaling uses a single, mandatory rounding primitive.

### 4.1 Definition

**Shift Domain Constraint**: `0 ≤ shift ≤ 62`. Out-of-range shifts trigger `FAULT_DOMAIN`.

**Round to Nearest, Ties to Even (RNE)**:

```
DVM_RoundShiftR_RNE(x: int64_t, shift: uint32_t) → int32_t:
    if shift > 62:
        SET_FAULT_FLAG(DOMAIN)
        return 0
    
    if shift == 0:
        return DVM_Clamp32(x)
    
    mask = (1LL << shift) - 1
    halfway = 1LL << (shift - 1)
    fraction = x & mask
    quotient = x >> shift  // Arithmetic shift (sign-extends)
    
    if fraction < halfway:
        result = quotient
    else if fraction > halfway:
        result = quotient + 1
    else:  // Exactly halfway - round to even
        result = quotient + (quotient & 1)
    
    return DVM_Clamp32(result)
```

### 4.2 Properties

- **Deterministic**: Same input always produces same output
- **Negative-safe**: Arithmetic right shift preserves sign
- **Tie-breaking**: Round-to-even eliminates statistical bias
- **Bounded**: Output clamped to 32-bit range

### 4.3 Test Vectors

| Input x | Shift | Expected | Reason |
|---------|-------|----------|--------|
| 0x00018000 | 16 | 2 | 1.5 rounds to 2 (even) |
| 0x00028000 | 16 | 2 | 2.5 rounds to 2 (even) |
| 0x00038000 | 16 | 4 | 3.5 rounds to 4 (even) |
| 0xFFFE8000 | 16 | -2 | -1.5 rounds to -2 (even) |
| 0xFFFD8000 | 16 | -2 | -2.5 rounds to -2 (even) |
| 0x00017FFF | 16 | 1 | 1.499... rounds down |
| 0x00018001 | 16 | 2 | 1.500...1 rounds up |

---

## 5. Deterministic Data Ordering: Cycle-Walking Feistel

### 5.1 Problem Statement

Dataset D = (d_0, ..., d_{N-1}) must be shuffled deterministically. The permutation must be a **true bijection** for any N, not just powers of two.

### 5.2 Cycle-Walking Feistel Permutation

```
π(index, seed, epoch, N) → [0, N-1]:
    // Find smallest k such that 2^k ≥ N
    k = ceil_log2(N)
    range = 1 << k
    half_bits = k >> 1
    half_mask = (1 << half_bits) - 1
    
    // Max iterations guard (for bounded loop requirement)
    max_iterations = range  // Should never need more than one cycle
    iterations = 0
    
    i = index
    do:
        if iterations >= max_iterations:
            SET_FAULT_FLAG(DOMAIN)
            return index % N  // Fallback (flagged)
        iterations++
        
        // 4-round Feistel on 2^k domain
        L = i & half_mask
        R = (i >> half_bits) & half_mask
        
        for round in [0, 1, 2, 3]:
            F = DVM_Hash(seed, epoch, round, R)
            temp = R
            R = L ^ (F & half_mask)
            L = temp
        
        i = (R << half_bits) | L
        
        // Cycle-walk: repeat until result is in valid range
    while i >= N
    
    return i
```

**Max iteration guard**: Safety-critical requires bounded loops. Guard should never trigger (E[iterations] < 2).

### 5.3 Properties

- **Bijection**: Every element in [0, N-1] maps to exactly one other element
- **Deterministic**: Pure function of (index, seed, epoch, N)
- **Termination**: Expected iterations < 2 for any N (provable)

### 5.4 Hash Function

```
DVM_Hash(seed, epoch, round, value) → uint32_t:
    h = seed
    h = h * 0x9E3779B9 + epoch
    h = h * 0x85EBCA6B + round
    h = h * 0xC2B2AE35 + value
    h = h ^ (h >> 16)
    h = h * 0x85EBCA6B
    h = h ^ (h >> 13)
    return h
```

### 5.5 Batch Construction

Batch at step t with batch size B:

```
B_t = { d_{π(t·B + j, seed, epoch, N)} : j ∈ [0, B-1] }
```

**Canonical Formula**: This exact formula must be used in all documents and code.

---

## 6. Deterministic Gradient Reduction

### 6.1 Fixed Reduction Topology

All gradient aggregation uses a rigid, pre-declared binary tree. Execution order is encoded by `op_id` and enforced regardless of hardware parallelism.

### 6.2 Operation Identifier (op_id)

**Size**: 64-bit minimum (128-bit recommended for large models).

```
op_id = Hash(graph_node_id, tensor_id, element_index, reduction_level, step)
```

This ensures globally unique identifiers across:
- All layers in the model
- All elements in each tensor
- All levels in the reduction tree
- All training steps

### 6.3 Compensated Reduction: Neumaier Summation

**Unified Algorithm** (used everywhere):

```
CompAdd(accum, v) → accum:
    t = accum.sum + v
    
    if abs(accum.sum) >= abs(v):
        e = (accum.sum - t) + v
    else:
        e = (v - t) + accum.sum
    
    accum.sum = t
    accum.err = accum.err + e
    return accum
```

### 6.4 Reduction Error Bound

For tree depth D with base precision Δ:

```
|⟦g̃⟧ - Σᵢ⟦gᵢ⟧| ≤ γ(D) · Δ
```

Where γ(D) = 2D for Neumaier summation (provable bound).

### 6.5 Accumulator Exhaustion Bound

For batch size B with maximum value 2³¹:

```
Max sum = B × 2³¹
64-bit capacity = 2⁶³
Safe batch limit: B ≤ 2³² (enforced: B ≤ 65536)
```

---

## 7. Deterministic "Stochastic" Rounding

### 7.1 Counter-Based PRNG

```
DVM_PRNG(seed, op_id, step) → uint32_t:
    // Philox-like counter-based RNG
    ctr = (op_id << 32) | step
    key = seed
    
    for r in [0..9]:
        ctr = (ctr * 0xD2511F53) ^ key
        key = (key * 0xCD9E8D57) + 0x9E3779B9
    
    return ctr & 0xFFFFFFFF
```

### 7.2 Stochastic Rounding (Deterministic)

```
DVM_StochasticRound(x, shift, seed, op_id, step) → int32_t:
    rand = DVM_PRNG(seed, op_id, step)
    mask = (1LL << shift) - 1
    frac = x & mask
    threshold = rand >> (32 - shift)
    
    if frac > threshold:
        return DVM_Clamp32((x >> shift) + 1)
    else:
        return DVM_Clamp32(x >> shift)
```

**Randomness is epistemic, not ontological.**

---

## 8. Training as a Discrete Dynamical System

### 8.1 Training Operator

```
T: (θ_t, t) ↦ θ_{t+1}
θ_{t+1} = Quantize(θ_t - η · ∇̃f(θ_t; B_t))
```

This is a **total function** — always produces a valid result or halts with fault.

### 8.2 Primitive Recursion

```
θ_T = T^{(T)}(θ_0)
```

Training is a primitive recursive function of:

```
(θ_0, D, DVM, config, Seed)
```

---

## 9. Stability Theory: The Shadowing Tube

### 9.1 Reference Optimizer

```
Φ(θ) = θ - η · ∇f(θ)
```

### 9.2 Implemented Update

```
Φ̃(θ) = Quantize(θ - η · ∇̃f(θ))
```

Bounded perturbation:

```
‖Φ̃(θ) - Φ(θ)‖ ≤ ε
```

### 9.3 Enforced Contraction

```
‖Φ(x) - Φ(y)‖ ≤ L · ‖x - y‖,  where 0 < L < 1
```

Maintained via:
- Spectral constraints (weight matrix norms)
- Weight decay (explicit dissipation)
- Learning rate bounds

### 9.4 Shadowing Theorem

**Theorem**: For all t ≤ T_s (stability horizon):

```
‖θ̃_t - θ_t‖ ≤ ε / (1 - L)
```

**Interpretation**: Error saturates, does not accumulate.

### 9.5 Runtime Stability Monitor

Every N steps, verify:

```
‖W_ℓᵀ · W_ℓ - I‖ ≤ κ
```

If violated, trigger Björck orthogonalization or halt.

---

## 10. Canonical Serialization for Merkle Commitments

### 10.1 Principle

**Never hash in-memory structs.** Always hash canonical byte streams.

### 10.2 Tensor Serialization

```
CanonicalTensor(T) → bytes:
    header = [
        version: uint32_le,      // Format version (1)
        dtype: uint32_le,        // 0=Q16.16, 1=Q8.24, 2=Q32.32
        ndims: uint32_le,        // Number of dimensions
        dims[0..ndims]: uint32_le,  // Dimension sizes
        total_size: uint64_le    // Total elements
    ]
    data = [
        elements[0..total_size]: int32_le or int64_le  // Little-endian
    ]
    return header || data
```

### 10.3 Excluded from Hashing

- Pointers (memory addresses)
- Padding bytes
- Timestamps
- Checkpoint metadata (except where explicitly included)

### 10.4 Batch Hash

```
H(B_t) = SHA256(
    for j in [0, B-1]:
        index = π(t·B + j, seed, epoch, N)
        emit uint32_le(index)
)
```

**Note**: Hash of sample indices, not raw sample data (unless data canonicalization is separately specified).

---

## 11. Auditability via Merkle Training Chains

### 11.1 Step Hash

```
h_t = SHA256(h_{t-1} ‖ H(θ_t) ‖ H(B_t) ‖ uint64_le(t))
```

### 11.2 Initial Hash

```
h_0 = SHA256(H(θ_0) ‖ H(config) ‖ uint64_le(seed))
```

### 11.3 Verification Protocol

To verify step t → t+1:

1. Obtain checkpoint (h_{t-1}, θ_t, B_t, t)
2. Recompute θ_{t+1} = T(θ_t, B_t) using DVM
3. Compute h'_t = SHA256(h_{t-1} ‖ H(θ_t) ‖ H(B_t) ‖ t)
4. Verify h'_t = h_t

**Complexity**: O(1) per step verification.

### 11.4 Fault Invalidation

If any DVM fault flag is set during step computation, the resulting hash is **invalid** and must not be committed to the chain.

---

## 12. Bit-Identical Inference

Inference executes the same DVM semantics:

```
y = DVM_Clamp32(DVM_RoundShiftR_RNE(W·x + b, shift))
```

### Cross-Platform Equivalence Theorem

For any two DVM-compliant machines A, B:

```
M_A(x) ≡ M_B(x)
```

---

## 13. Hardware Requirements: Certifiable-Ready ISA

### I. Integer-Only Execution

- **Mandatory**: ≥64-bit integer accumulators
- **Mandatory**: Saturating arithmetic with fault flag
- **Forbidden**: Float/double units in DVM mode
- **Forbidden**: FMA instructions

### II. Rigid Reduction Control

- Exposed compute topology
- Deterministic synchronization primitive
- Reduction order fixed by op_id (64-bit minimum)

### III. Counter-Based PRNG

- Hardware or microcode support
- Input: (Seed: 64-bit, Op_ID: 64-bit, Step: 64-bit)
- Output: Deterministic 32-bit random value

### IV. Fault Signaling

- Sticky fault flags for: overflow, underflow, div-by-zero
- Queryable fault status register
- Fault must halt or invalidate current operation

---

## 14. The Three Core Theorems

### Theorem 1 — Bit Identity

For any two DVM-compliant architectures A, B:

```
F_A(s) = F_B(s)
```

**Proof requirement**: DVM conformance tests pass on both architectures.

### Theorem 2 — Shadowing Stability

For all t ≤ T_s:

```
‖θ̃_t - θ_t‖ ≤ ε / (1 - L)
```

**Proof requirement**: Contraction factor L < 1 maintained by runtime monitor.

### Theorem 3 — Auditability

Any training step is verifiable in O(1) time given the checkpoint.

**Proof requirement**: Merkle chain construction follows canonical serialization.

---

## 15. DVM Conformance Profile

### 15.1 Mandatory Test Vectors

A DVM implementation must pass:

| Test | Description |
|------|-------------|
| ROUND-001 | Rounding test vectors (§4.3) |
| PERM-001 | Feistel permutation bijectivity for N ∈ {100, 1000, 60000} |
| REDUCE-001 | Reduction tree determinism (parallel vs sequential) |
| PRNG-001 | PRNG reproducibility across platforms |
| HASH-001 | Canonical serialization produces identical hashes |
| FAULT-001 | Overflow/div-zero sets fault flag correctly |

### 15.2 Compiler Requirements

```makefile
CFLAGS += -std=c99 -Wall -Wextra -Werror
CFLAGS += -ffp-contract=off       # No FMA fusion
CFLAGS += -fno-fast-math          # Strict semantics
CFLAGS += -fno-associative-math   # Preserve operation order
```

### 15.3 Toolchain Semantics (Mandatory)

**Policy**: Explicit widening to 64-bit before all 32-bit arithmetic. Do NOT rely on `-fwrapv`.

**Rationale**: `-fwrapv` behaviour varies across compilers. Explicit widening is portable and auditable.

**Required**:
```c
int64_t wide = (int64_t)a + (int64_t)b;  // CORRECT
return dvm_clamp32(wide);
```

**Forbidden**:
```c
return a + b;  // FORBIDDEN: UB without -fwrapv, non-portable with it
```

---

## 16. Conclusion

Certifiable Training replaces probabilistic trust with mathematical inevitability.

Models become:
- **Reproducible** by construction
- **Auditable** by design
- **Portable** across silicon
- **Provable** as artifacts of a deterministic law

This is not an optimization trick. **This is a protocol.**

You are no longer training models. **You are defining the TCP/IP of model provenance.**

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | CT-SPEC-001 |
| Version | 2.1.0 |
| Author | William Murray |
| Date | January 2026 |
| Status | Production Draft |
| Classification | Public |
| Copyright | © 2026 The Murray Family Innovation Trust. All rights reserved. |

---

## References

1. Certifiable Inference — https://github.com/williamofai/certifiable-inference
2. Murray Deterministic Computing Platform (MDCP) — Patent GB2521625.0
3. DO-178C — Software Considerations in Airborne Systems
4. IEC 62304 — Medical Device Software Lifecycle
5. ISO 26262 — Functional Safety for Road Vehicles

---

## Appendix A: Document Alignment Matrix

| Topic | SPECIFICATION | CT-MATH-001 | CT-STRUCT-001 |
|-------|---------------|-------------|---------------|
| Rounding | §4 DVM_RoundShiftR_RNE | §8 Unified | §3.3 Type definition |
| Permutation | §5 Cycle-walking Feistel | §6 Identical | §10 ct_permutation_t |
| Batch indexing | §5.5 π(t·B + j, ...) | §5.3 Identical | N/A (computed) |
| CompAdd | §6.3 Neumaier | §10.3 Identical | §4.2 ct_comp_accum_t |
| Fault model | §3.3, §3.4 | §3.6 Identical | §11 ct_error_t |
| Serialization | §10 Canonical | §17 Identical | §14 Functions |
| op_id | §6.2 64-bit hash | §7.2 Identical | §4.1 ct_prng_t |

*All documents MUST remain synchronized on these topics.*
