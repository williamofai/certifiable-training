# certifiable-training

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/williamofai/certifiable-training)
[![Tests](https://img.shields.io/badge/tests-10%2F10%20passing-brightgreen)](https://github.com/williamofai/certifiable-training)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)
[![MISRA Compliance](https://img.shields.io/badge/MISRA--C-2012-blue)](docs/misra-compliance.md)

**Deterministic, bit-perfect ML training for safety-critical systems.**

Pure C99. Zero dynamic allocation. Certifiable for DO-178C, IEC 62304, and ISO 26262.

🔴 **Live Demo:** [training.speytech.com](https://training.speytech.com)

---

## The Problem

Training machine learning models is inherently non-deterministic:
- Floating-point operations vary across platforms
- Parallel reductions produce different results each run
- Stochastic gradient descent introduces randomness
- Data shuffling depends on random number generators

For safety-critical systems, you cannot certify what you cannot reproduce.

**Read more:** [Why Floating Point Is Dangerous](https://speytech.com/ai-architecture/floating-point-danger/)

## The Solution

`certifiable-training` defines training as a **deterministic state evolution**:

### 1. Fixed-Point Arithmetic
Q16.16 weights, Q8.24 gradients, Q32.32 accumulators. Same math, same result, every platform.

### 2. Deterministic Reduction
Fixed tree topology with Neumaier compensated summation. Parallel execution, deterministic result.

### 3. Reproducible "Randomness"
Counter-based PRNG: `PRNG(seed, op_id, step) → deterministic bits`. Same seed = same sequence.

### 4. Merkle Audit Trail
Every training step cryptographically committed. Any step verifiable in O(1) time.

**Result:** `θ_T = T^(T)(θ_0, D, seed)` — Training is a pure function.

## Status

**All core modules complete — 10/10 test suites passing.**

| Module | Description | Status |
|--------|-------------|--------|
| DVM Primitives | Fixed-point arithmetic with fault detection | ✅ |
| Counter-based PRNG | Deterministic pseudo-random generation | ✅ |
| Compensated Summation | Neumaier algorithm for precision | ✅ |
| Reduction Tree | Fixed-topology parallel reduction | ✅ |
| Forward Pass | Q16.16 activations (ReLU, sigmoid, tanh) | ✅ |
| Backward Pass | Q8.24 gradient computation | ✅ |
| Optimizers | SGD, Momentum, Adam | ✅ |
| Merkle Chain | SHA256 audit trail with checkpoints | ✅ |
| Data Permutation | Cycle-Walking Feistel bijection | ✅ |
| Bit Identity | Cross-platform reproducibility tests | ✅ |

## Quick Start

### Build

```bash
mkdir build && cd build
cmake ..
make
make test  # Run all 10 test suites
```

### Expected Output

```
100% tests passed, 0 tests failed out of 10
Total Test time (real) = 0.04 sec
```

### Basic Training Step

```c
#include "ct_types.h"
#include "dvm.h"
#include "forward.h"
#include "backward.h"
#include "optimizer.h"
#include "merkle.h"
#include "permutation.h"

// All buffers pre-allocated (no malloc)
fixed_t weights[784 * 128];
grad_t gradients[784 * 128];
ct_fault_flags_t faults = {0};

// Initialize Merkle chain
ct_merkle_ctx_t merkle;
ct_merkle_init(&merkle, &weights_tensor, config, config_size, seed);

// Get batch via deterministic permutation
ct_batch_ctx_t batch;
ct_batch_init(&batch, seed, epoch, dataset_size, batch_size);
ct_batch_get_indices(&batch, step, indices, &faults);

// Forward pass
ct_forward_linear(&layer, input, output, &faults);
ct_forward_relu(output, activated, size, &faults);

// Backward pass
ct_backward_relu(grad_out, activated, grad_in, size, &faults);
ct_backward_linear(&layer, grad_in, &faults);

// Optimizer step
ct_sgd_step(&sgd, weights, gradients, size, &faults);

// Commit to Merkle chain
ct_merkle_step(&merkle, &weights_tensor, indices, batch_size, &step_record, &faults);

if (ct_has_fault(&faults)) {
    // Chain invalidated - do not proceed
}
```

## Architecture

### Deterministic Virtual Machine (DVM)

All arithmetic operations use widening and saturation:

```c
// CORRECT: Explicit widening
int64_t wide = (int64_t)a + (int64_t)b;
return dvm_clamp32(wide, &faults);

// FORBIDDEN: Raw overflow
return a + b;  // Undefined behavior
```

### Fixed-Point Formats

| Format | Use Case | Range | Precision |
|--------|----------|-------|-----------|
| Q16.16 | Weights, activations | ±32768 | 1.5×10⁻⁵ |
| Q8.24 | Gradients | ±128 | 5.9×10⁻⁸ |
| Q32.32 | Accumulators | ±2³¹ | 2.3×10⁻¹⁰ |

### Fault Model

Every operation signals faults without silent failure:

```c
typedef struct {
    uint32_t overflow    : 1;  // Saturated high
    uint32_t underflow   : 1;  // Saturated low
    uint32_t div_zero    : 1;  // Division by zero
    uint32_t domain      : 1;  // Invalid input
    uint32_t precision   : 1;  // Precision loss detected
} ct_fault_flags_t;
```

### Merkle Audit Trail

Every training step produces a cryptographic commitment:

```
h_t = SHA256(h_{t-1} || H(θ_t) || H(B_t) || t)
```

Any step can be independently verified. If faults occur, the chain is invalidated.

### Data Permutation

Cycle-Walking Feistel provides true bijection for any dataset size N:

```
π: [0, N-1] → [0, N-1]  (one-to-one and onto)
```

Same seed + epoch = same shuffle, every time, every platform.

## Documentation

- **CT-MATH-001.md** — Mathematical foundations
- **CT-STRUCT-001.md** — Data structure specifications  
- **docs/requirements/** — SRS documents with full traceability

## Related Projects

| Project | Description | Demo |
|---------|-------------|------|
| [certifiable-inference](https://github.com/williamofai/certifiable-inference) | Deterministic inference engine | [inference.speytech.com](https://inference.speytech.com) |
| **certifiable-training** | Deterministic training engine | [training.speytech.com](https://training.speytech.com) |
| [certifiable-data](https://github.com/williamofai/certifiable-data) | Deterministic data pipeline | — |

Together, `certifiable-data` + `certifiable-training` + `certifiable-inference` provide a complete deterministic ML pipeline for safety-critical systems.

## Why This Matters

### Medical Devices
IEC 62304 Class C requires traceable, reproducible software. Non-deterministic training cannot be validated.

### Autonomous Vehicles
ISO 26262 ASIL-D demands provable behavior. Training must be auditable.

### Aerospace
DO-178C Level A requires complete requirements traceability. "We trained it and it works" is not certifiable.

This is the first ML training system designed from the ground up for safety-critical certification.

## Compliance Support

This implementation is designed to support certification under:
- **DO-178C** (Aerospace software)
- **IEC 62304** (Medical device software)
- **ISO 26262** (Automotive functional safety)
- **IEC 61508** (Industrial safety systems)

For compliance packages and certification assistance, contact below.

## Contributing

We welcome contributions from systems engineers working in safety-critical domains. See [CONTRIBUTING.md](CONTRIBUTING.md).

**Important:** All contributors must sign a [Contributor License Agreement](CONTRIBUTOR-LICENSE-AGREEMENT.md).

## License

**Dual Licensed:**
- **Open Source:** GNU General Public License v3.0 (GPLv3)
- **Commercial:** Available for proprietary use in safety-critical systems

For commercial licensing and compliance documentation packages, contact below.

## Patent Protection

This implementation is built on the **Murray Deterministic Computing Platform (MDCP)**,
protected by UK Patent **GB2521625.0**.

MDCP defines a deterministic computing architecture for safety-critical systems,
providing:
- Provable execution bounds
- Resource-deterministic operation
- Certification-ready patterns
- Platform-independent behavior

For commercial licensing inquiries: william@fstopify.com

## About

Built by **SpeyTech** in the Scottish Highlands.

30 years of UNIX infrastructure experience applied to deterministic computing for safety-critical systems.

Patent: UK GB2521625.0 - Murray Deterministic Computing Platform (MDCP)

**Contact:**
William Murray  
william@fstopify.com  
[speytech.com](https://speytech.com)

---

*Building deterministic AI systems for when lives depend on the answer.*

Copyright © 2026 The Murray Family Innovation Trust. All rights reserved.
