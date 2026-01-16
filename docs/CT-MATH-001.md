# CT-MATH-001: Mathematical Foundations

**Certifiable Training — Data Dictionary: Mathematics**

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | CT-MATH-001 |
| Version | 2.1.0 |
| Author | William Murray |
| Date | January 2026 |
| Status | Production Draft |
| Classification | Public |
| Copyright | © 2026 The Murray Family Innovation Trust. All rights reserved. |

---

## 1. Purpose

This document defines the complete mathematical foundation for Certifiable Training. Every symbol, operator, and bound is specified here. Code is a literal transcription of this mathematics.

**Principle**: The math is the specification. The code is the implementation. They must be identical.

**Alignment**: This document is synchronized with CT-SPEC-001 and CT-STRUCT-001. See Appendix B for alignment matrix.

---

## 2. Number Systems

### 2.1 Integer Ring

Let **Z** denote the integers. Let **Z**_{2^w} denote two's-complement integers of width w bits.

| Symbol | Definition | Range |
|--------|------------|-------|
| **Z**_{2^32} | 32-bit signed integers | [-2³¹, 2³¹ - 1] |
| **Z**_{2^64} | 64-bit signed integers | [-2⁶³, 2⁶³ - 1] |

**Overflow Model**: All arithmetic uses 64-bit intermediates with deterministic saturation to 32-bit. See §3.6.

### 2.2 Fixed-Point Format Q(a,b)

A fixed-point number x ∈ **Z**_{2^{a+b}} has interpretation:

```
⟦x⟧ = x · 2⁻ᵇ
```

| Format | Total Bits | Integer Bits | Fractional Bits | Range | Precision |
|--------|------------|--------------|-----------------|-------|-----------|
| Q16.16 | 32 | 16 | 16 | [-32768, 32767.99998] | 1.5 × 10⁻⁵ |
| Q8.24 | 32 | 8 | 24 | [-128, 127.99999994] | 5.9 × 10⁻⁸ |
| Q32.32 | 64 | 32 | 32 | [-2³¹, 2³¹ - 2⁻³²] | 2.3 × 10⁻¹⁰ |

**Primary Format**: Q16.16 for weights and activations.  
**Gradient Format**: Q8.24 for gradients (higher precision for small values).  
**Accumulator Format**: Q32.32 for intermediate sums.

### 2.3 Fixed-Point Constants

For Q16.16:

| Constant | Symbol | Value | Hexadecimal |
|----------|--------|-------|-------------|
| One | FIXED_ONE | 65536 | 0x00010000 |
| Half | FIXED_HALF | 32768 | 0x00008000 |
| Zero | FIXED_ZERO | 0 | 0x00000000 |
| Max | FIXED_MAX | 2147483647 | 0x7FFFFFFF |
| Min | FIXED_MIN | -2147483648 | 0x80000000 |
| Epsilon | FIXED_EPS | 1 | 0x00000001 |

---

## 3. DVM Arithmetic Primitives

### 3.1 Widening Addition

All addition uses 64-bit intermediates to avoid undefined behavior:

```
DVM_Add(a: int32, b: int32) → int32:
    wide = (int64)a + (int64)b
    return DVM_Clamp32(wide)
```

### 3.2 Widening Subtraction

```
DVM_Sub(a: int32, b: int32) → int32:
    wide = (int64)a - (int64)b
    return DVM_Clamp32(wide)
```

### 3.3 Multiplication

For a, b ∈ Q(a₁,b₁):

```
DVM_Mul(a: int32, b: int32) → int32:
    wide = (int64)a × (int64)b
    return DVM_RoundShiftR_RNE(wide, FRAC_BITS)
```

**Intermediate**: 64-bit product before rounding.  
**Result**: Back to original Q format via unified rounding.

### 3.4 Integer Division

```
DVM_Div_Int32(a: int32, b: int32) → int32:
    if b == 0:
        SET_FAULT_FLAG(DIV_ZERO)
        return 0  // Deterministic but flagged
    return a / b  // Truncate toward zero (C99 semantics)
```

### 3.5 Fixed-Point Division

```
DVM_Div_Q(a: int32, b: int32, frac_bits: uint32) → int32:
    if b == 0:
        SET_FAULT_FLAG(DIV_ZERO)
        return 0
    if frac_bits > 62:
        SET_FAULT_FLAG(DOMAIN)
        return 0
    
    // Scale numerator to preserve precision
    wide = (int64)a << frac_bits
    
    // Divide with rounding
    quotient = wide / b
    remainder = wide % b
    
    // Round to nearest (using safe abs)
    if DVM_Abs64_Sat(2 * remainder) >= DVM_Abs64_Sat(b):
        quotient = quotient + sign(a) * sign(b)
    
    return DVM_Clamp32(quotient)
```

**Usage**:
- `DVM_Div_Int32`: Integer division (loop counters, indices)
- `DVM_Div_Q`: Fixed-point division (weights, gradients, Adam updates)

### 3.6 Saturating Clamp

```
DVM_Clamp32(x: int64) → int32:
    if x > INT32_MAX:
        SET_FAULT_FLAG(OVERFLOW)
        return INT32_MAX
    if x < INT32_MIN:
        SET_FAULT_FLAG(UNDERFLOW)
        return INT32_MIN
    return (int32)x
```

### 3.7 Safe Absolute Value

```
DVM_Abs64_Sat(x: int64) → int64:
    if x == INT64_MIN:
        SET_FAULT_FLAG(OVERFLOW)
        return INT64_MAX  // -INT64_MIN overflows; saturate
    return (x < 0) ? -x : x
```

**Rationale**: `abs(INT64_MIN)` is undefined/overflow in C. This function handles that edge case deterministically.

**Usage**: Required in Neumaier summation magnitude comparisons and fixed-point division rounding.

### 3.8 Fault Model

**Policy**: Fail-closed with mandatory fault signaling.

| Fault | Flag | Behavior |
|-------|------|----------|
| Overflow | FAULT_OVERFLOW | Saturate to MAX, set flag |
| Underflow | FAULT_UNDERFLOW | Saturate to MIN, set flag |
| Divide-by-zero | FAULT_DIV_ZERO | Return 0, set flag |

**Fault Flags**: Sticky bits that persist until explicitly cleared.

**System Response**:
1. Any fault invalidates current Merkle hash
2. Training must halt or checkpoint before fault
3. No silent continuation with corrupted state

---

## 4. State Space

### 4.1 Parameter Space

Let P be the number of trainable parameters. The parameter space is:

```
W_Δ = (Q_w)^P
```

Where Q_w is the chosen fixed-point format (typically Q16.16).

**Property**: W_Δ is finite, discrete, and closed.

### 4.2 Training State

At step t, the training state is:

```
θ_t ∈ W_Δ
```

This is a P-dimensional vector of fixed-point values.

### 4.3 Gradient Space

Gradients live in:

```
G_Δ = (Q_g)^P
```

Where Q_g = Q8.24 for higher precision on small gradient values.

---

## 5. Data Ordering: Cycle-Walking Feistel

### 5.1 Dataset

Let D = (d₀, d₁, ..., d_{N-1}) be the training dataset with N samples.

### 5.2 Bijective Permutation

A permutation π must be a **true bijection** on [0, N-1]:

```
π: {0, ..., N-1} → {0, ..., N-1}  (one-to-one and onto)
```

**Critical**: Simple `mod N` on a power-of-two Feistel output is NOT a bijection for arbitrary N.

### 5.3 Cycle-Walking Feistel Construction

```
π(index, seed, epoch, N) → [0, N-1]:
    // Step 1: Find smallest k such that 2^k ≥ N
    k = ⌈log₂(N)⌉
    range = 2^k
    half_bits = ⌊k/2⌋
    half_mask = 2^{half_bits} - 1
    
    // Max iterations guard (should never trigger; theoretical E[iter] < 2)
    max_iterations = 2^k  // At most one full cycle
    iterations = 0
    
    // Step 2: Cycle-walk until result is in valid range
    i = index
    do:
        if iterations >= max_iterations:
            SET_FAULT_FLAG(DOMAIN)
            return index % N  // Fallback (deterministic but flagged)
        iterations = iterations + 1
        
        // 4-round Feistel network
        L = i & half_mask
        R = (i >> half_bits) & half_mask
        
        for r in [0, 1, 2, 3]:
            F = DVM_Hash(seed, epoch, r, R)
            temp = R
            R = L ⊕ (F & half_mask)
            L = temp
        
        i = (R << half_bits) | L
    while i ≥ N
    
    return i
```

**Max iteration guard**: Safety-critical systems require bounded loops. The guard should never trigger (theoretical expected iterations < 2), but its presence satisfies DO-178C bounded loop requirements.

### 5.4 Hash Function for Feistel

```
DVM_Hash(seed, epoch, round, value) → uint32:
    h = seed
    h = (h × 0x9E3779B9 + epoch) & 0xFFFFFFFF
    h = (h × 0x85EBCA6B + round) & 0xFFFFFFFF
    h = (h × 0xC2B2AE35 + value) & 0xFFFFFFFF
    h = h ⊕ (h >> 16)
    h = (h × 0x85EBCA6B) & 0xFFFFFFFF
    h = h ⊕ (h >> 13)
    return h
```

### 5.5 Properties

| Property | Guarantee | Proof |
|----------|-----------|-------|
| Bijection | Every i ∈ [0,N-1] maps to unique j ∈ [0,N-1] | Feistel is bijection; cycle-walk preserves it |
| Deterministic | π(i, s, e, N) is pure function | No external state |
| Termination | Expected iterations < 2 | Range ≤ 2N, each iteration has ≥50% success |

### 5.6 Batch Construction (Canonical Formula)

Batch at step t with batch size B:

```
B_t = { d_{π(t·B + j, seed, epoch, N)} : j ∈ [0, B-1] }
```

**This exact formula must be used everywhere.** No variations.

---

## 6. Counter-Based PRNG

### 6.1 Definition

```
DVM_PRNG(seed: uint64, op_id: uint64, step: uint64) → uint32
```

A pure function producing deterministic pseudo-random bits.

### 6.2 Implementation (Philox-style)

```
DVM_PRNG(seed, op_id, step):
    ctr = (op_id << 32) | (step & 0xFFFFFFFF)
    key = seed
    
    for r in [0..9]:
        ctr = ((ctr × 0xD2511F53) & 0xFFFFFFFFFFFFFFFF) ⊕ key
        key = ((key × 0xCD9E8D57) + 0x9E3779B9) & 0xFFFFFFFFFFFFFFFF
    
    return ctr & 0xFFFFFFFF
```

### 6.3 Operation Identifier (op_id)

**Size**: 64-bit minimum (128-bit recommended).

**Construction**: Hash of operation context to ensure global uniqueness:

```
op_id = Hash(graph_node_id, tensor_id, element_index, reduction_level, step_offset)
```

**Uniqueness Requirement**: op_id must be unique across:
- All layers in the model
- All elements in each tensor  
- All levels in reduction trees
- All training steps

### 6.4 Properties

| Property | Guarantee |
|----------|-----------|
| Deterministic | Same (seed, op_id, step) → same output |
| Uniform | Output uniformly distributed in [0, 2³²-1] |
| Independent | Different op_id produces uncorrelated sequences |

---

## 7. Gradient Computation

### 7.1 Forward Pass

For layer ℓ with weights W_ℓ and input x:

```
z_ℓ = W_ℓ · x_ℓ + b_ℓ        (pre-activation)
a_ℓ = σ(z_ℓ)                  (activation)
```

All operations in fixed-point using DVM primitives.

### 7.2 Backward Pass

Loss gradient with respect to output:

```
δ_L = ∂L/∂a_L
```

Backpropagation through layer ℓ:

```
δ_ℓ = (W_{ℓ+1}ᵀ · δ_{ℓ+1}) ⊙ σ'(z_ℓ)
```

Weight gradients:

```
∂L/∂W_ℓ = δ_ℓ · x_ℓᵀ
∂L/∂b_ℓ = δ_ℓ
```

### 7.3 Gradient Scaling

Gradients use Q8.24 format with scaling:

```
g_scaled = g · GRAD_SCALE
```

Where GRAD_SCALE = 2¹⁶ (configurable).

### 7.4 Vanishing Gradient Monitoring

**Problem**: Q8.24 minimum non-zero is 2⁻²⁴ ≈ 5.9 × 10⁻⁸. Smaller gradients become zero.

**Monitoring**:

```
zero_grad_ratio = zero_grad_count / total_grad_count
```

**Contract**: If zero_grad_ratio > 0.05, raise warning CT_WARN_GRAD_FLOOR.

---

## 8. Deterministic Rounding: DVM_RoundShiftR_RNE

### 8.1 Unified Rounding Primitive

**All** fixed-point scaling uses this single primitive. No exceptions.

**Shift bounds**: `0 ≤ shift ≤ 62`. Shift values outside this range trigger `FAULT_DOMAIN`.

**Rationale**: `1LL << 63` overflows signed int64; `1LL << 64` is UB. Cap at 62 for safety.

**Round to Nearest, Ties to Even (RNE)**:

```
DVM_RoundShiftR_RNE(x: int64, shift: uint32) → int32:
    // Shift bounds check (MANDATORY)
    if shift > 62:
        SET_FAULT_FLAG(DOMAIN)
        return 0
    
    if shift == 0:
        return DVM_Clamp32(x)
    
    mask = (1LL << shift) - 1
    halfway = 1LL << (shift - 1)
    
    // Extract fraction (works for negative x with arithmetic shift)
    fraction = x & mask
    quotient = x >> shift  // Arithmetic right shift
    
    if fraction < halfway:
        result = quotient
    else if fraction > halfway:
        result = quotient + 1
    else:  // Exactly halfway
        // Round to nearest even (eliminates statistical bias)
        result = quotient + (quotient & 1)
    
    return DVM_Clamp32(result)
```

### 8.2 Properties

| Property | Guarantee |
|----------|-----------|
| Deterministic | Same input → same output |
| Negative-safe | Arithmetic shift preserves sign |
| Unbiased | Round-to-even eliminates drift |
| Bounded | Output clamped to int32 range |

### 8.3 Test Vectors (Mandatory)

| Input x (hex) | Shift | Expected | Interpretation |
|---------------|-------|----------|----------------|
| 0x0001_8000 | 16 | 2 | 1.5 → 2 (round to even) |
| 0x0002_8000 | 16 | 2 | 2.5 → 2 (round to even) |
| 0x0003_8000 | 16 | 4 | 3.5 → 4 (round to even) |
| 0x0004_8000 | 16 | 4 | 4.5 → 4 (round to even) |
| 0xFFFF_8000 | 16 | 0 | -0.5 → 0 (round to even) |
| 0xFFFE_8000 | 16 | -2 | -1.5 → -2 (round to even) |
| 0xFFFD_8000 | 16 | -2 | -2.5 → -2 (round to even) |
| 0x0001_7FFF | 16 | 1 | 1.499... → 1 (round down) |
| 0x0001_8001 | 16 | 2 | 1.500...1 → 2 (round up) |
| 0xFFFE_7FFF | 16 | -1 | -1.500...1 → -1 (round up toward zero) |

### 8.4 Stochastic Rounding Variant

For regularization benefits with zero execution variance:

```
DVM_StochasticRound(x: int64, shift: uint32, seed: uint64, op_id: uint64, step: uint64) → int32:
    if shift == 0:
        return DVM_Clamp32(x)
    
    rand = DVM_PRNG(seed, op_id, step)
    mask = (1 << shift) - 1
    fraction = x & mask
    threshold = rand >> (32 - shift)
    
    quotient = x >> shift
    if fraction > threshold:
        result = quotient + 1
    else:
        result = quotient
    
    return DVM_Clamp32(result)
```

---

## 9. Gradient Reduction: Neumaier Summation

### 9.1 Fixed Reduction Topology

All gradient aggregation uses a rigid binary tree with pre-declared structure.

For B samples:
- Tree depth: D = ⌈log₂(B)⌉
- Internal nodes: B - 1
- Leaf nodes: B

### 9.2 Compensated Accumulator

Each node carries (sum, error) pair in Q32.32:

```
Accumulator = (sum: int64, err: int64)
```

### 9.3 Neumaier Summation (Unified Algorithm)

**This exact algorithm must be used everywhere.** No variations.

**Safe absolute value** (required for magnitude comparison):

```
DVM_Abs64_Sat(x: int64) → int64:
    if x == INT64_MIN:
        SET_FAULT_FLAG(OVERFLOW)  // Optional: may choose not to fault
        return INT64_MAX          // Saturate rather than UB
    if x < 0:
        return -x
    return x
```

**Neumaier compensated addition**:

```
CompAdd(accum: Accumulator, v: int64) → Accumulator:
    t = accum.sum + v
    
    if DVM_Abs64_Sat(accum.sum) >= DVM_Abs64_Sat(v):
        e = (accum.sum - t) + v
    else:
        e = (v - t) + accum.sum
    
    return (sum: t, err: accum.err + e)
```

**Why Neumaier over Kahan**: More robust when v may be larger than running sum.

### 9.4 Reduction Error Bound

For tree depth D with base precision Δ:

```
|⟦g̃⟧ - Σᵢ⟦gᵢ⟧| ≤ γ(D) · Δ
```

For Neumaier summation: γ(D) = 2D (tight bound, provable).

### 9.5 Accumulator Exhaustion Analysis

**Worst-case**: Batch size B with maximum value per sample.

```
Max single value: 2³¹ - 1
Max sum: B × (2³¹ - 1)
64-bit capacity: 2⁶³ - 1
```

**Safe batch size**:

```
B_max = (2⁶³ - 1) / (2³¹ - 1) ≈ 2³²
```

**Enforced limit**: B ≤ 65536 (provides 65536× safety margin).

---

## 10. Training Operator

### 10.1 Update Function

```
T: (θ_t, t) ↦ θ_{t+1}
θ_{t+1} = Quantize(θ_t - η · ∇̃f(θ_t; B_t))
```

This is a **total function** — always produces valid result or halts with fault.

### 10.2 SGD Update

```
θ_{t+1} = DVM_Sub(θ_t, DVM_Mul(η, g_t))
```

### 10.3 SGD with Momentum

```
v_t = DVM_Add(DVM_Mul(β, v_{t-1}), g_t)
θ_{t+1} = DVM_Sub(θ_t, DVM_Mul(η, v_t))
```

### 10.4 Adam (Fixed-Point)

```
m_t = DVM_Add(DVM_Mul(β₁, m_{t-1}), DVM_Mul(1-β₁, g_t))
v_t = DVM_Add(DVM_Mul(β₂, v_{t-1}), DVM_Mul(1-β₂, DVM_Mul(g_t, g_t)))
m̂_t = DVM_Div(m_t, 1 - β₁ᵗ)
v̂_t = DVM_Div(v_t, 1 - β₂ᵗ)
θ_{t+1} = DVM_Sub(θ_t, DVM_Div(DVM_Mul(η, m̂_t), DVM_Add(DVM_Sqrt(v̂_t), ε)))
```

**Note**: DVM_Sqrt requires full specification. See §13.

---

## 11. Stability Theory

### 11.1 Reference Optimizer

```
Φ(θ) = θ - η · ∇f(θ)
```

### 11.2 Implemented Update

```
Φ̃(θ) = Quantize(θ - η · ∇̃f(θ))
```

### 11.3 Perturbation Bound

```
‖Φ̃(θ) - Φ(θ)‖ ≤ ε
```

Where ε depends on: quantization step Δ, reduction error γ(D)·Δ, rounding.

### 11.4 Contraction Requirement

```
‖Φ(x) - Φ(y)‖ ≤ L · ‖x - y‖,  where 0 < L < 1
```

Enforced via: learning rate bounds, weight decay, spectral normalization.

### 11.5 Shadowing Theorem

**Theorem**: For all t ≤ T_s (stability horizon):

```
‖θ̃_t - θ_t‖ ≤ ε / (1 - L)
```

**Proof**: See CT-STABILITY-001.

**Interpretation**: Error saturates, does not accumulate.

### 11.6 Runtime Stability Monitor

Every N steps, verify:

```
‖W_ℓᵀ · W_ℓ - I‖ ≤ κ
```

If violated: trigger Björck orthogonalization or halt training.

---

## 12. Activation Functions (Deterministic LUT)

### 12.1 Forbidden

**No** library functions: `exp()`, `expf()`, `tanh()`, `log()`, etc.

### 12.2 ReLU

```
ReLU(x) = max(0, x)
```

Trivial in fixed-point (comparison and select).

### 12.3 Sigmoid (LUT with Linear Interpolation)

**LUT Specification**:

| Parameter | Value |
|-----------|-------|
| Domain | [-8, +8] |
| Table size | 257 entries |
| Step size | 16/256 = 0.0625 |
| Format | Q16.16 |

**Lookup**:

```
DVM_Sigmoid(x: int32) → int32:
    // Saturation
    if x <= FIXED_FROM_FLOAT(-8.0): return 0
    if x >= FIXED_FROM_FLOAT(+8.0): return FIXED_ONE
    
    // Map to table index
    x_shifted = x + FIXED_FROM_FLOAT(8.0)  // Now in [0, 16]
    index = (x_shifted * 256) >> (FRAC_BITS + 4)  // /16 via shift
    frac = (x_shifted * 256) & ((1 << (FRAC_BITS + 4)) - 1)
    
    // Linear interpolation (all integer ops)
    y0 = SIGMOID_LUT[index]
    y1 = SIGMOID_LUT[index + 1]
    result = y0 + DVM_RoundShiftR_RNE((int64)(y1 - y0) * frac, FRAC_BITS + 4)
    
    return result
```

### 12.4 LUT Properties

| Property | Guarantee |
|----------|-----------|
| Bit-identical | Same input → same output on all platforms |
| Monotonic | x₁ < x₂ ⟹ σ(x₁) ≤ σ(x₂) |
| Max error | < 0.002 vs true sigmoid |
| Complexity | O(1), two lookups + one multiply |

### 12.5 Sigmoid Derivative

```
σ'(x) = σ(x) · (1 - σ(x))
```

Computed from LUT output.

---

## 13. Square Root (For Adam)

### 13.1 Newton-Raphson in Fixed-Point

```
DVM_Sqrt(x: int32) → int32:
    if x <= 0: return 0
    
    // Initial guess: x/2 (or better estimate via leading zeros)
    guess = x >> 1
    if guess == 0: guess = 1
    
    // Fixed iteration count (no data-dependent termination)
    for i in [0..7]:
        // Newton-Raphson: guess = (guess + x/guess) / 2
        div = DVM_Div(x, guess)
        sum = DVM_Add(guess, div)
        guess = sum >> 1
    
    return guess
```

### 13.2 Properties

| Property | Value |
|----------|-------|
| Iterations | Fixed 8 (no data-dependent branching) |
| Convergence | Quadratic, 8 iterations sufficient for 32-bit |
| Max error | < 1 LSB for Q16.16 inputs |

**Alternative**: Remove Adam from safety profile; use SGD/momentum only.

---

## 14. Loss Functions

### 14.1 Mean Squared Error

```
L_MSE = (1/N) · Σᵢ (yᵢ - ŷᵢ)²
```

Gradient:

```
∂L/∂ŷᵢ = (2/N) · (ŷᵢ - yᵢ)
```

### 14.2 Cross-Entropy

Requires fixed-point log approximation (specify LUT similar to sigmoid).

---

## 15. Numerical Bounds

### 15.1 Weight Bounds

```
|w_ij| ≤ W_MAX
```

W_MAX chosen to prevent overflow in forward/backward passes.

### 15.2 Gradient Clipping

```
g_i = clamp(g_i, -G_MAX, G_MAX)
```

### 15.3 Learning Rate Bounds

For stability (L < 1):

```
η < 2 / L_f
```

Where L_f is Lipschitz constant of ∇f.

---

## 16. Merkle Training Chains

### 16.1 Step Hash

```
h_t = SHA256(h_{t-1} ‖ H(θ_t) ‖ H(B_t) ‖ uint64_le(t))
```

### 16.2 Initial Hash

```
h_0 = SHA256(H(θ_0) ‖ H(config) ‖ uint64_le(seed))
```

### 16.3 Fault Invalidation

If any DVM fault flag is set during step t computation, h_t is **invalid** and must not be committed.

---

## 17. Canonical Serialization

### 17.1 Principle

**Never hash in-memory structs.** Always hash canonical byte streams.

### 17.2 Tensor Serialization

**Committed tensors must be contiguous.** Strides are for transient/working memory only.

**Rationale**: Strided tensors can represent identical data differently. Canonical serialization requires a single representation.

**Rule**: Before hashing or checkpointing, tensors must be in canonical contiguous layout (row-major, strides = natural).

```
CanonicalTensor(T) → bytes:
    // Precondition: T must be contiguous (strides == natural)
    ASSERT(is_contiguous(T))
    
    header = [
        version: uint32_le,          // Format version (1)
        dtype: uint32_le,            // 0=Q16.16, 1=Q8.24, 2=Q32.32
        ndims: uint32_le,            // Number of dimensions
        dims[0..ndims]: uint32_le,   // Dimension sizes
        total_size: uint64_le        // Total elements
        // NOTE: strides are NOT serialized - implied by contiguous layout
    ]
    data = [
        elements[0..total_size]: int32_le or int64_le  // Little-endian, row-major
    ]
    return header ‖ data
```

**Signed integer encoding**: Two's complement (standard C representation), serialized as little-endian.

### 17.3 Batch Hash

```
H(B_t) = SHA256(
    for j in [0, B-1]:
        index = π(t·B + j, seed, epoch, N)
        emit uint32_le(index)
)
```

### 17.4 Excluded from Hashing

- Pointers
- Padding bytes
- Timestamps
- Transient metadata

---

## 18. Summary of Symbols

| Symbol | Meaning | Domain |
|--------|---------|--------|
| θ_t | Parameters at step t | W_Δ = (Q_w)^P |
| g_t | Gradient at step t | G_Δ = (Q_g)^P |
| η | Learning rate | Q_w |
| B_t | Batch at step t | Set of B samples |
| T | Training operator | W_Δ × ℕ → W_Δ |
| Φ | Reference optimizer | W → W |
| Φ̃ | Quantized optimizer | W_Δ → W_Δ |
| L | Contraction factor | (0, 1) |
| ε | Per-step error bound | ℝ⁺ |
| h_t | Merkle hash at step t | {0,1}²⁵⁶ |
| π | Data permutation (bijection) | [N] → [N] |
| Δ | Quantization step | 2⁻ᵇ |
| γ(D) | Reduction error factor | 2D for Neumaier |

---

## Appendix A: Test Vector Summary

| Test ID | Description | Section |
|---------|-------------|---------|
| ROUND-001 | DVM_RoundShiftR_RNE vectors | §8.3 |
| PERM-001 | Feistel bijectivity | §5.5 |
| PRNG-001 | DVM_PRNG reproducibility | §6.4 |
| COMP-001 | CompAdd accuracy | §9.4 |
| SQRT-001 | DVM_Sqrt convergence | §13.2 |
| SIGMOID-001 | LUT monotonicity | §12.4 |

---

## Appendix B: Document Alignment Matrix

| Topic | CT-SPEC-001 | CT-MATH-001 | CT-STRUCT-001 |
|-------|-------------|-------------|---------------|
| Rounding | §4 | §8 (this doc) | §3.3 |
| Permutation | §5 | §5 (this doc) | §10 |
| Batch indexing | §5.5 | §5.6 (this doc) | N/A |
| CompAdd | §6.3 | §9.3 (this doc) | §4.2 |
| Fault model | §3.3-3.4 | §3.6 (this doc) | §11 |
| Serialization | §10 | §17 (this doc) | §14 |
| op_id | §6.2 | §6.3 (this doc) | §4.1 |

---

## References

1. CT-SPEC-001 — Certifiable Training Specification
2. CT-STRUCT-001 — Data Structure Specification
3. CT-DVM-001 — Deterministic Virtual Machine Specification
4. CT-STABILITY-001 — Stability Analysis and Proofs

---

*End of CT-MATH-001 v2.0.0*
