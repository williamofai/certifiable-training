/**
 * @file reduction.h
 * @project Certifiable Training
 * @brief Fixed-topology reduction tree for deterministic gradient aggregation
 *
 * @details Provides a binary tree structure for reducing (summing) values
 *          in a deterministic order. Unlike parallel reduction with atomics,
 *          this tree has a fixed topology declared at initialization time,
 *          guaranteeing identical execution order across all platforms.
 *
 *          Used for gradient aggregation where batch gradients must be
 *          summed reproducibly.
 *
 * @traceability CT-MATH-001 §9.1, CT-STRUCT-001 §4.3-4.4
 * @compliance MISRA-C:2012, DO-178C, IEC 62304, ISO 26262
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 */

#ifndef CT_REDUCTION_H
#define CT_REDUCTION_H

#include "ct_types.h"
#include "compensated.h"

/** Marker for leaf nodes (no children) */
#define CT_LEAF_MARKER   UINT32_MAX

/** Marker for root node (no parent) */
#define CT_ROOT_MARKER   UINT32_MAX

/** Maximum supported batch size */
#define CT_MAX_LEAVES    65536

/**
 * @brief Node in fixed-topology reduction tree
 *
 * @details Each node represents either:
 *          - A leaf: holds input value, left_child = right_child = CT_LEAF_MARKER
 *          - An internal node: has two children to merge
 *
 *          The op_id ensures deterministic PRNG sequences if stochastic
 *          rounding is used during reduction.
 *
 * @ref CT-STRUCT-001 §4.3
 */
typedef struct {
    uint32_t left_child;    /**< Index of left child, or CT_LEAF_MARKER */
    uint32_t right_child;   /**< Index of right child, or CT_LEAF_MARKER */
    uint64_t op_id;         /**< Unique operation ID for this node */
    uint32_t parent;        /**< Index of parent, or CT_ROOT_MARKER */
    uint32_t _pad;          /**< Alignment padding (explicit) */
} ct_reduction_node_t;

/**
 * @brief Reduction tree structure
 *
 * @details A complete binary tree for B leaves has:
 *          - B leaf nodes (indices 0 to B-1)
 *          - B-1 internal nodes (indices B to 2B-2)
 *          - Total: 2B-1 nodes
 *          - Root at index 2B-2
 *          - Depth: ceil(log2(B))
 *
 * @ref CT-STRUCT-001 §4.4
 */
typedef struct {
    ct_reduction_node_t *nodes;  /**< Array of nodes (caller-provided) */
    uint32_t num_leaves;         /**< Number of leaf nodes (batch size) */
    uint32_t num_internal;       /**< Number of internal nodes (num_leaves - 1) */
    uint32_t num_nodes;          /**< Total nodes (2 * num_leaves - 1) */
    uint32_t root_index;         /**< Index of root node */
    uint32_t depth;              /**< Tree depth: ceil(log2(num_leaves)) */
    uint64_t base_op_id;         /**< Base op_id for this tree */
} ct_reduction_tree_t;

/**
 * @brief Calculate required node count for a given batch size
 *
 * @param num_leaves Number of leaf nodes (batch size)
 * @return Total nodes needed (2 * num_leaves - 1), or 0 if invalid
 *
 * @note Use this to allocate the node array before calling ct_reduction_init()
 */
uint32_t ct_reduction_node_count(uint32_t num_leaves);

/**
 * @brief Calculate required buffer size in bytes
 *
 * @param num_leaves Number of leaf nodes
 * @return Size in bytes for node array, or 0 if invalid
 */
size_t ct_reduction_buffer_size(uint32_t num_leaves);

/**
 * @brief Initialize a reduction tree
 *
 * @param tree       Pointer to tree structure
 * @param nodes      Caller-provided node array (must hold 2*num_leaves-1 nodes)
 * @param num_leaves Number of leaf nodes (batch size, 1 to CT_MAX_LEAVES)
 * @param base_op_id Base operation ID (nodes get sequential IDs from this)
 * @param faults     Fault flags
 * @return CT_OK on success, error code on failure
 *
 * @details Builds a complete binary tree structure:
 *          - Leaves at indices [0, num_leaves-1]
 *          - Internal nodes at indices [num_leaves, 2*num_leaves-2]
 *          - Root at index 2*num_leaves-2
 *
 *          Each node receives op_id = base_op_id + node_index.
 *
 * @pre tree != NULL, nodes != NULL
 * @pre 1 <= num_leaves <= CT_MAX_LEAVES
 * @post Tree structure fully initialized and ready for reduction
 *
 * Complexity: O(num_leaves)
 *
 * @ref CT-MATH-001 §9.1
 */
ct_error_t ct_reduction_init(ct_reduction_tree_t *tree,
                             ct_reduction_node_t *nodes,
                             uint32_t num_leaves,
                             uint64_t base_op_id,
                             ct_fault_flags_t *faults);

/**
 * @brief Check if a node is a leaf
 *
 * @param tree Pointer to tree
 * @param index Node index
 * @return true if leaf, false if internal node
 */
bool ct_reduction_is_leaf(const ct_reduction_tree_t *tree, uint32_t index);

/**
 * @brief Get tree depth
 *
 * @param tree Pointer to tree
 * @return Tree depth (ceil(log2(num_leaves)))
 */
uint32_t ct_reduction_depth(const ct_reduction_tree_t *tree);

/**
 * @brief Reduce an array of 64-bit values using the tree
 *
 * @param tree   Initialized reduction tree
 * @param values Input values (must have tree->num_leaves elements)
 * @param faults Fault flags
 * @return Reduced (summed) value using compensated arithmetic
 *
 * @details Performs bottom-up reduction:
 *          1. Load values into leaf nodes
 *          2. For each level from bottom to top:
 *             - Merge left and right children using compensated addition
 *          3. Return finalized root accumulator
 *
 *          The tree structure guarantees identical merge order on every
 *          execution, regardless of hardware parallelism.
 *
 * @pre tree initialized, values != NULL
 * @pre values has exactly tree->num_leaves elements
 *
 * Complexity: O(num_leaves)
 * Determinism: Bit-perfect - same inputs always produce same output
 *
 * @ref CT-MATH-001 §9.3
 */
int64_t ct_reduction_reduce_64(const ct_reduction_tree_t *tree,
                               const int64_t *values,
                               ct_fault_flags_t *faults);

/**
 * @brief Reduce an array of 32-bit values using the tree
 *
 * @param tree   Initialized reduction tree
 * @param values Input values (must have tree->num_leaves elements)
 * @param faults Fault flags
 * @return Reduced (summed) value (64-bit to prevent overflow)
 *
 * @details Same as ct_reduction_reduce_64 but widens 32-bit inputs.
 *          Result is 64-bit to accommodate sum of many 32-bit values.
 */
int64_t ct_reduction_reduce_32(const ct_reduction_tree_t *tree,
                               const int32_t *values,
                               ct_fault_flags_t *faults);

/**
 * @brief Reduce with per-node callback (for debugging/tracing)
 *
 * @param tree     Initialized reduction tree
 * @param values   Input values
 * @param callback Function called after each node reduction (may be NULL)
 * @param context  User context passed to callback
 * @param faults   Fault flags
 * @return Reduced value
 *
 * @details The callback receives:
 *          - node_index: which node was just processed
 *          - accum: the accumulator value at that node
 *          - context: user-provided pointer
 *
 *          Useful for debugging, verification, and Merkle hashing.
 */
typedef void (*ct_reduction_callback_t)(uint32_t node_index,
                                        const ct_comp_accum_t *accum,
                                        void *context);

int64_t ct_reduction_reduce_traced(const ct_reduction_tree_t *tree,
                                   const int64_t *values,
                                   ct_reduction_callback_t callback,
                                   void *context,
                                   ct_fault_flags_t *faults);

/**
 * @brief Get the parent index of a node
 *
 * @param tree Pointer to tree
 * @param index Node index
 * @return Parent index, or CT_ROOT_MARKER if root
 */
uint32_t ct_reduction_parent(const ct_reduction_tree_t *tree, uint32_t index);

/**
 * @brief Get the left child index of a node
 *
 * @param tree Pointer to tree
 * @param index Node index
 * @return Left child index, or CT_LEAF_MARKER if leaf
 */
uint32_t ct_reduction_left_child(const ct_reduction_tree_t *tree, uint32_t index);

/**
 * @brief Get the right child index of a node
 *
 * @param tree Pointer to tree
 * @param index Node index
 * @return Right child index, or CT_LEAF_MARKER if leaf
 */
uint32_t ct_reduction_right_child(const ct_reduction_tree_t *tree, uint32_t index);

/**
 * @brief Get the op_id of a node
 *
 * @param tree Pointer to tree
 * @param index Node index
 * @return Operation ID for that node
 */
uint64_t ct_reduction_op_id(const ct_reduction_tree_t *tree, uint32_t index);

#endif /* CT_REDUCTION_H */
