# SRS-004: Reduction Tree

**Software Requirements Specification — Fixed-Topology Gradient Reduction**

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | SRS-004 |
| Version | 1.0.0 |
| Author | William Murray |
| Date | January 2026 |
| Status | Draft |
| Classification | Public |

---

## 1. Purpose

This document specifies the requirements for the Reduction Tree module, which provides deterministic gradient aggregation using a fixed binary tree topology.

**Traceability**: CT-MATH-001 §9.1, CT-STRUCT-001 §4.3-4.4

---

## 2. Overview

### 2.1 Problem Statement

Gradient aggregation in training requires summing many values (one per sample in a batch). Standard parallel reduction using `atomicAdd` is non-deterministic because the order of atomic operations depends on thread scheduling.

This non-determinism breaks:
- Bit-identical reproducibility
- Certification arguments (cannot prove identical behavior)
- Debugging (different runs produce different results)

### 2.2 Solution

A **fixed-topology binary tree** where:
- The tree structure is declared at initialization
- Each node has a unique `op_id` determining execution order
- Reduction proceeds bottom-up with deterministic merge sequence
- Compensated arithmetic prevents precision loss

### 2.3 Tree Structure

For B leaves (batch size B):
- **Leaf nodes**: B (indices 0 to B-1)
- **Internal nodes**: B-1 (indices B to 2B-2)
- **Total nodes**: 2B-1
- **Root**: index 2B-2
- **Depth**: ceil(log2(B))

Example for 4 leaves:
```
            [6] root
           /    \
        [4]      [5]
       /   \    /   \
     [0]  [1] [2]  [3]  <- leaves
```

---

## 3. Requirements

### SRS-004.1: Node Structure

**Requirement**: Each reduction node shall contain:
- `left_child`: Index of left child, or `CT_LEAF_MARKER` for leaves
- `right_child`: Index of right child, or `CT_LEAF_MARKER` for leaves
- `op_id`: 64-bit unique operation identifier
- `parent`: Index of parent, or `CT_ROOT_MARKER` for root
- `_pad`: Explicit padding for alignment

**Rationale**: CT-STRUCT-001 §4.3 specifies this structure. Explicit padding ensures consistent struct size across compilers.

**Verification**: Inspect `ct_reduction_node_t` definition.

### SRS-004.2: Tree Structure

**Requirement**: The tree structure shall contain:
- `nodes`: Pointer to caller-provided node array
- `num_leaves`: Batch size
- `num_internal`: Number of internal nodes (num_leaves - 1)
- `num_nodes`: Total nodes (2 * num_leaves - 1)
- `root_index`: Index of root node
- `depth`: Tree depth (ceil(log2(num_leaves)))
- `base_op_id`: Base for generating node op_ids

**Rationale**: CT-STRUCT-001 §4.4 specifies tree metadata.

**Verification**: Inspect `ct_reduction_tree_t` definition.

### SRS-004.3: Initialization

**Requirement**: The `ct_reduction_init()` function shall:
- Accept num_leaves from 1 to CT_MAX_LEAVES (65536)
- Build a complete binary tree structure
- Assign op_id = base_op_id + node_index to each node
- Set child/parent pointers correctly
- Return CT_OK on success, error code on failure
- Handle NULL pointers without crashing

**Rationale**: Fixed structure must be established before reduction.

**Verification**: Unit tests for various leaf counts.

### SRS-004.4: Node Count Calculation

**Requirement**: The `ct_reduction_node_count()` function shall return:
- 2n-1 for n leaves (where 1 ≤ n ≤ CT_MAX_LEAVES)
- 0 for invalid inputs (n=0 or n > CT_MAX_LEAVES)

**Rationale**: Caller needs to allocate node array before initialization.

**Verification**: Unit tests for boundary values.

### SRS-004.5: Leaf Identification

**Requirement**: The `ct_reduction_is_leaf()` function shall return:
- true for indices [0, num_leaves-1]
- false for indices [num_leaves, num_nodes-1]

**Rationale**: Different handling for leaves vs internal nodes.

**Verification**: Unit tests verify leaf/internal classification.

### SRS-004.6: Reduction Algorithm

**Requirement**: The `ct_reduction_reduce_64()` function shall:
1. Load input values into leaf node accumulators
2. Process internal nodes in index order (children before parents)
3. At each internal node, merge left and right children using compensated addition
4. Return the finalized root accumulator value

**Rationale**: CT-MATH-001 §9.1 requires fixed topology. Index-order processing guarantees deterministic sequence.

**Verification**: Unit tests verify correct sums and determinism.

### SRS-004.7: Compensated Merge

**Requirement**: Internal node merging shall use `ct_comp_merge()` from the compensated summation module.

**Rationale**: Preserves precision through the reduction tree per CT-MATH-001 §9.3.

**Verification**: Test with large/small value mixtures.

### SRS-004.8: 32-bit Reduction

**Requirement**: The `ct_reduction_reduce_32()` function shall:
- Accept 32-bit input values
- Widen to 64-bit before accumulation
- Return 64-bit result

**Rationale**: Gradient values are typically Q16.16 (32-bit), but sum may exceed 32-bit range.

**Verification**: Unit tests with 32-bit inputs.

### SRS-004.9: Traced Reduction

**Requirement**: The `ct_reduction_reduce_traced()` function shall:
- Accept an optional callback function
- Call the callback after processing each node
- Pass node index, accumulator state, and user context to callback
- Produce identical results to non-traced reduction

**Rationale**: Enables debugging, verification, and Merkle hash computation.

**Verification**: Test callback invocation count and result consistency.

### SRS-004.10: Determinism

**Requirement**: For identical inputs, reduction shall produce bit-identical results:
- Across multiple executions
- Across different compilers
- Across different architectures

**Rationale**: Core requirement of Certifiable Training.

**Verification**: Determinism unit tests, cross-platform verification.

### SRS-004.11: Batch Size Limit

**Requirement**: The module shall enforce CT_MAX_LEAVES (65536) as the maximum batch size.

**Rationale**: CT-MATH-001 §9.5 specifies this limit for accumulator safety.

**Verification**: Initialization rejects larger batch sizes.

---

## 4. Test Vectors

### 4.1 Simple Reductions

| Leaves | Values | Expected Sum |
|--------|--------|--------------|
| 1 | [12345] | 12345 |
| 2 | [100, 200] | 300 |
| 4 | [10, 20, 30, 40] | 100 |
| 64 | [1, 2, ..., 64] | 2080 |

### 4.2 Tree Parameters

| Leaves | Nodes | Internal | Root Index | Depth |
|--------|-------|----------|------------|-------|
| 1 | 1 | 0 | 0 | 0 |
| 2 | 3 | 1 | 2 | 1 |
| 4 | 7 | 3 | 6 | 2 |
| 8 | 15 | 7 | 14 | 3 |
| 64 | 127 | 63 | 126 | 6 |

### 4.3 Compensated Accuracy

| Values | Expected |
|--------|----------|
| [2^40, 1, 2^40, 2, 2^40, 3, 2^40, 4] | 4×2^40 + 10 |

---

## 5. Implementation Mapping

### 5.1 Source Files

| File | Purpose |
|------|---------|
| `include/reduction.h` | Public interface |
| `src/dvm/reduction.c` | Implementation |
| `tests/unit/test_reduction.c` | Unit tests |

### 5.2 Functions

| Function | Requirement |
|----------|-------------|
| `ct_reduction_node_count()` | SRS-004.4 |
| `ct_reduction_buffer_size()` | SRS-004.4 |
| `ct_reduction_init()` | SRS-004.3 |
| `ct_reduction_is_leaf()` | SRS-004.5 |
| `ct_reduction_depth()` | SRS-004.2 |
| `ct_reduction_reduce_64()` | SRS-004.6, SRS-004.7 |
| `ct_reduction_reduce_32()` | SRS-004.8 |
| `ct_reduction_reduce_traced()` | SRS-004.9 |

---

## 6. Traceability

### 6.1 Upstream (Math Specification)

| Requirement | CT-MATH-001 Section |
|-------------|---------------------|
| SRS-004.1 | §9.1 Fixed Reduction Topology |
| SRS-004.6 | §9.1 Binary tree structure |
| SRS-004.7 | §9.3 Neumaier Summation |
| SRS-004.11 | §9.5 Batch size limit |

### 6.2 Dependencies

| Module | Relationship |
|--------|--------------|
| SRS-003 Compensated | Uses ct_comp_merge() |
| SRS-006 Backward Pass | Uses tree for gradient aggregation |

### 6.3 Compliance Mapping

| Standard | Relevance |
|----------|-----------|
| DO-178C | Fixed execution order, WCET analysis |
| IEC 62304 | Deterministic behavior verification |
| ISO 26262 | Reproducible results |
| MISRA-C:2012 | Coding standard compliance |

---

## 7. Design Notes

### 7.1 Why Fixed Topology?

Parallel reduction algorithms (e.g., GPU atomicAdd) have non-deterministic execution order. Even with the same inputs, different runs produce different intermediate sums due to thread scheduling.

A fixed tree topology forces identical merge sequences:
- Node 4 always merges nodes 0 and 1
- Node 5 always merges nodes 2 and 3
- Node 6 always merges nodes 4 and 5

This is mathematically equivalent to parallel reduction but deterministically ordered.

### 7.2 Operation IDs

Each node has a unique `op_id` computed as `base_op_id + node_index`. This supports:
- Deterministic PRNG if stochastic rounding is used
- Merkle hash computation for auditability
- Debugging and tracing

### 7.3 Memory Model

The caller provides the node array, enabling:
- Static allocation for safety-critical systems
- Stack allocation for small batches
- No hidden heap usage

### 7.4 Stack Limit

The implementation uses stack-allocated accumulators for trees up to 256 nodes. Larger trees trigger a fault flag and fall back to sequential reduction.

For production use with large batches, a caller-provided accumulator workspace should be added.

---

## 8. Quality Criteria

### 8.1 Test Coverage

- [ ] All functions have unit tests
- [ ] All tree sizes from 1 to 64 tested
- [ ] Determinism verified
- [ ] Compensated accuracy verified

### 8.2 Code Quality

- [ ] Zero warnings with strict flags
- [ ] MISRA-C compliant
- [ ] All functions documented
- [ ] Traceability tags present

### 8.3 Performance

- [ ] O(n) initialization
- [ ] O(n) reduction
- [ ] No dynamic allocation

---

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | Jan 2026 | William Murray | Initial release |

---

*End of SRS-004*
