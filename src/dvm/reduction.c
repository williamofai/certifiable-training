/**
 * @file reduction.c
 * @project Certifiable Training
 * @brief Fixed-topology reduction tree for deterministic gradient aggregation
 *
 * @details Implements a binary tree structure for reducing values in a
 *          deterministic order. The tree topology is fixed at initialization,
 *          guaranteeing identical merge sequences across all executions.
 *
 *          This replaces non-deterministic parallel reduction (atomicAdd)
 *          with a provably reproducible algorithm.
 *
 * @traceability CT-MATH-001 §9.1
 * @compliance MISRA-C:2012, DO-178C, IEC 62304, ISO 26262
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 *            All rights reserved.
 */

#include "reduction.h"
#include "compensated.h"
#include <stddef.h>

/**
 * @brief Compute ceil(log2(n)) for tree depth calculation
 *
 * @param n Input value (must be > 0)
 * @return Ceiling of log base 2
 */
static uint32_t ceil_log2(uint32_t n)
{
    if (n == 0) return 0;
    if (n == 1) return 0;
    
    uint32_t log = 0;
    uint32_t val = n - 1;
    
    while (val > 0) {
        val >>= 1;
        log++;
    }
    
    return log;
}

/**
 * @brief Calculate required node count for a given batch size
 */
uint32_t ct_reduction_node_count(uint32_t num_leaves)
{
    if (num_leaves == 0 || num_leaves > CT_MAX_LEAVES) {
        return 0;
    }
    
    /* Complete binary tree: 2n - 1 nodes for n leaves */
    return 2 * num_leaves - 1;
}

/**
 * @brief Calculate required buffer size in bytes
 */
size_t ct_reduction_buffer_size(uint32_t num_leaves)
{
    uint32_t count = ct_reduction_node_count(num_leaves);
    if (count == 0) {
        return 0;
    }
    return (size_t)count * sizeof(ct_reduction_node_t);
}

/**
 * @brief Initialize a reduction tree
 *
 * @details Builds a complete binary tree where:
 *          - Indices [0, num_leaves-1] are leaf nodes
 *          - Indices [num_leaves, 2*num_leaves-2] are internal nodes
 *          - Root is at index 2*num_leaves-2
 *
 *          Tree structure (example for 4 leaves):
 *
 *                    [6] root
 *                   /    \
 *                [4]      [5]
 *               /   \    /   \
 *             [0]  [1] [2]  [3]  <- leaves
 *
 *          Internal node i has children at:
 *          - left:  2 * (i - num_leaves)
 *          - right: 2 * (i - num_leaves) + 1
 */
ct_error_t ct_reduction_init(ct_reduction_tree_t *tree,
                             ct_reduction_node_t *nodes,
                             uint32_t num_leaves,
                             uint64_t base_op_id,
                             ct_fault_flags_t *faults)
{
    /* Validate inputs */
    if (tree == NULL || nodes == NULL) {
        return CT_ERR_NULL;
    }
    
    if (num_leaves == 0) {
        if (faults) faults->domain = 1;
        return CT_ERR_CONFIG;
    }
    
    if (num_leaves > CT_MAX_LEAVES) {
        if (faults) faults->domain = 1;
        return CT_ERR_CONFIG;
    }
    
    /* Calculate tree parameters */
    uint32_t num_internal = (num_leaves > 1) ? (num_leaves - 1) : 0;
    uint32_t num_nodes = num_leaves + num_internal;
    uint32_t root_index = (num_leaves > 1) ? (num_nodes - 1) : 0;
    uint32_t depth = ceil_log2(num_leaves);
    
    /* Initialize tree structure */
    tree->nodes = nodes;
    tree->num_leaves = num_leaves;
    tree->num_internal = num_internal;
    tree->num_nodes = num_nodes;
    tree->root_index = root_index;
    tree->depth = depth;
    tree->base_op_id = base_op_id;
    
    /* Initialize leaf nodes */
    for (uint32_t i = 0; i < num_leaves; i++) {
        nodes[i].left_child = CT_LEAF_MARKER;
        nodes[i].right_child = CT_LEAF_MARKER;
        nodes[i].op_id = base_op_id + i;
        nodes[i].parent = CT_ROOT_MARKER;  /* Set below for non-trivial trees */
        nodes[i]._pad = 0;
    }
    
    /* Initialize internal nodes and set parent pointers */
    if (num_leaves > 1) {
        for (uint32_t i = 0; i < num_internal; i++) {
            uint32_t node_idx = num_leaves + i;
            
            /*
             * For a complete binary tree built bottom-up:
             * Internal node i (0-indexed among internals) has children
             * at indices 2*i and 2*i+1 (among all nodes below this level).
             *
             * Simpler approach: pair leaves, then pair those results, etc.
             * We use a level-by-level construction.
             */
            uint32_t left_child = 2 * i;
            uint32_t right_child = 2 * i + 1;
            
            /* Clamp to valid indices */
            if (left_child >= node_idx) {
                left_child = CT_LEAF_MARKER;
            }
            if (right_child >= node_idx) {
                right_child = CT_LEAF_MARKER;
            }
            
            nodes[node_idx].left_child = left_child;
            nodes[node_idx].right_child = right_child;
            nodes[node_idx].op_id = base_op_id + node_idx;
            nodes[node_idx].parent = CT_ROOT_MARKER;  /* Updated below */
            nodes[node_idx]._pad = 0;
            
            /* Set parent pointers for children */
            if (left_child != CT_LEAF_MARKER && left_child < num_nodes) {
                nodes[left_child].parent = node_idx;
            }
            if (right_child != CT_LEAF_MARKER && right_child < num_nodes) {
                nodes[right_child].parent = node_idx;
            }
        }
    }
    
    return CT_OK;
}

/**
 * @brief Check if a node is a leaf
 */
bool ct_reduction_is_leaf(const ct_reduction_tree_t *tree, uint32_t index)
{
    if (tree == NULL || index >= tree->num_nodes) {
        return false;
    }
    return index < tree->num_leaves;
}

/**
 * @brief Get tree depth
 */
uint32_t ct_reduction_depth(const ct_reduction_tree_t *tree)
{
    if (tree == NULL) {
        return 0;
    }
    return tree->depth;
}

/**
 * @brief Get parent index
 */
uint32_t ct_reduction_parent(const ct_reduction_tree_t *tree, uint32_t index)
{
    if (tree == NULL || tree->nodes == NULL || index >= tree->num_nodes) {
        return CT_ROOT_MARKER;
    }
    return tree->nodes[index].parent;
}

/**
 * @brief Get left child index
 */
uint32_t ct_reduction_left_child(const ct_reduction_tree_t *tree, uint32_t index)
{
    if (tree == NULL || tree->nodes == NULL || index >= tree->num_nodes) {
        return CT_LEAF_MARKER;
    }
    return tree->nodes[index].left_child;
}

/**
 * @brief Get right child index
 */
uint32_t ct_reduction_right_child(const ct_reduction_tree_t *tree, uint32_t index)
{
    if (tree == NULL || tree->nodes == NULL || index >= tree->num_nodes) {
        return CT_LEAF_MARKER;
    }
    return tree->nodes[index].right_child;
}

/**
 * @brief Get op_id of a node
 */
uint64_t ct_reduction_op_id(const ct_reduction_tree_t *tree, uint32_t index)
{
    if (tree == NULL || tree->nodes == NULL || index >= tree->num_nodes) {
        return 0;
    }
    return tree->nodes[index].op_id;
}

/**
 * @brief Reduce an array of 64-bit values using the tree
 *
 * @details Bottom-up reduction using compensated arithmetic:
 *          1. Initialize accumulators for all nodes
 *          2. Load values into leaf accumulators
 *          3. Process internal nodes in order (children before parents)
 *          4. Merge left and right children at each internal node
 *          5. Finalize and return root accumulator
 *
 *          The fixed node ordering guarantees deterministic results.
 */
int64_t ct_reduction_reduce_64(const ct_reduction_tree_t *tree,
                               const int64_t *values,
                               ct_fault_flags_t *faults)
{
    if (tree == NULL || values == NULL || tree->nodes == NULL) {
        return 0;
    }
    
    if (tree->num_leaves == 0) {
        return 0;
    }
    
    /* Special case: single value */
    if (tree->num_leaves == 1) {
        return values[0];
    }
    
    /*
     * Allocate accumulators on stack for small trees,
     * but we need to handle larger trees too.
     * 
     * For safety-critical: caller should provide workspace.
     * For now, use a reasonable stack limit.
     */
    #define MAX_STACK_NODES 256
    
    ct_comp_accum_t stack_accum[MAX_STACK_NODES];
    ct_comp_accum_t *accum;
    
    if (tree->num_nodes <= MAX_STACK_NODES) {
        accum = stack_accum;
    } else {
        /* 
         * For larger trees, we'd need caller-provided workspace.
         * For now, return error via fault flag.
         */
        if (faults) faults->domain = 1;
        
        /* Fall back to simple sequential sum */
        ct_comp_accum_t simple;
        ct_comp_init(&simple);
        for (uint32_t i = 0; i < tree->num_leaves; i++) {
            ct_comp_add(&simple, values[i], faults);
        }
        return ct_comp_finalize(&simple, faults);
    }
    
    /* Initialize all accumulators */
    for (uint32_t i = 0; i < tree->num_nodes; i++) {
        ct_comp_init(&accum[i]);
    }
    
    /* Load values into leaf accumulators */
    for (uint32_t i = 0; i < tree->num_leaves; i++) {
        ct_comp_init_value(&accum[i], values[i]);
    }
    
    /* 
     * Process internal nodes in index order.
     * Since children have lower indices than parents in our layout,
     * processing in order guarantees children are ready before parents.
     */
    for (uint32_t i = tree->num_leaves; i < tree->num_nodes; i++) {
        uint32_t left = tree->nodes[i].left_child;
        uint32_t right = tree->nodes[i].right_child;
        
        /* Merge left child */
        if (left != CT_LEAF_MARKER && left < tree->num_nodes) {
            ct_comp_merge(&accum[i], &accum[left], faults);
        }
        
        /* Merge right child */
        if (right != CT_LEAF_MARKER && right < tree->num_nodes) {
            ct_comp_merge(&accum[i], &accum[right], faults);
        }
    }
    
    /* Finalize root */
    return ct_comp_finalize(&accum[tree->root_index], faults);
    
    #undef MAX_STACK_NODES
}

/**
 * @brief Reduce an array of 32-bit values
 */
int64_t ct_reduction_reduce_32(const ct_reduction_tree_t *tree,
                               const int32_t *values,
                               ct_fault_flags_t *faults)
{
    if (tree == NULL || values == NULL || tree->nodes == NULL) {
        return 0;
    }
    
    if (tree->num_leaves == 0) {
        return 0;
    }
    
    /* Special case: single value */
    if (tree->num_leaves == 1) {
        return (int64_t)values[0];
    }
    
    #define MAX_STACK_NODES 256
    
    ct_comp_accum_t stack_accum[MAX_STACK_NODES];
    ct_comp_accum_t *accum;
    
    if (tree->num_nodes <= MAX_STACK_NODES) {
        accum = stack_accum;
    } else {
        if (faults) faults->domain = 1;
        
        /* Fall back to simple sum */
        ct_comp_accum_t simple;
        ct_comp_init(&simple);
        for (uint32_t i = 0; i < tree->num_leaves; i++) {
            ct_comp_add(&simple, (int64_t)values[i], faults);
        }
        return ct_comp_finalize(&simple, faults);
    }
    
    /* Initialize all accumulators */
    for (uint32_t i = 0; i < tree->num_nodes; i++) {
        ct_comp_init(&accum[i]);
    }
    
    /* Load values into leaf accumulators (widen to 64-bit) */
    for (uint32_t i = 0; i < tree->num_leaves; i++) {
        ct_comp_init_value(&accum[i], (int64_t)values[i]);
    }
    
    /* Process internal nodes */
    for (uint32_t i = tree->num_leaves; i < tree->num_nodes; i++) {
        uint32_t left = tree->nodes[i].left_child;
        uint32_t right = tree->nodes[i].right_child;
        
        if (left != CT_LEAF_MARKER && left < tree->num_nodes) {
            ct_comp_merge(&accum[i], &accum[left], faults);
        }
        
        if (right != CT_LEAF_MARKER && right < tree->num_nodes) {
            ct_comp_merge(&accum[i], &accum[right], faults);
        }
    }
    
    return ct_comp_finalize(&accum[tree->root_index], faults);
    
    #undef MAX_STACK_NODES
}

/**
 * @brief Reduce with callback for tracing/debugging
 */
int64_t ct_reduction_reduce_traced(const ct_reduction_tree_t *tree,
                                   const int64_t *values,
                                   ct_reduction_callback_t callback,
                                   void *context,
                                   ct_fault_flags_t *faults)
{
    if (tree == NULL || values == NULL || tree->nodes == NULL) {
        return 0;
    }
    
    if (tree->num_leaves == 0) {
        return 0;
    }
    
    if (tree->num_leaves == 1) {
        if (callback != NULL) {
            ct_comp_accum_t single;
            ct_comp_init_value(&single, values[0]);
            callback(0, &single, context);
        }
        return values[0];
    }
    
    #define MAX_STACK_NODES 256
    
    ct_comp_accum_t stack_accum[MAX_STACK_NODES];
    ct_comp_accum_t *accum;
    
    if (tree->num_nodes <= MAX_STACK_NODES) {
        accum = stack_accum;
    } else {
        if (faults) faults->domain = 1;
        return 0;  /* Can't trace without full accumulator array */
    }
    
    /* Initialize all accumulators */
    for (uint32_t i = 0; i < tree->num_nodes; i++) {
        ct_comp_init(&accum[i]);
    }
    
    /* Load values and trace leaves */
    for (uint32_t i = 0; i < tree->num_leaves; i++) {
        ct_comp_init_value(&accum[i], values[i]);
        if (callback != NULL) {
            callback(i, &accum[i], context);
        }
    }
    
    /* Process internal nodes with tracing */
    for (uint32_t i = tree->num_leaves; i < tree->num_nodes; i++) {
        uint32_t left = tree->nodes[i].left_child;
        uint32_t right = tree->nodes[i].right_child;
        
        if (left != CT_LEAF_MARKER && left < tree->num_nodes) {
            ct_comp_merge(&accum[i], &accum[left], faults);
        }
        
        if (right != CT_LEAF_MARKER && right < tree->num_nodes) {
            ct_comp_merge(&accum[i], &accum[right], faults);
        }
        
        if (callback != NULL) {
            callback(i, &accum[i], context);
        }
    }
    
    return ct_comp_finalize(&accum[tree->root_index], faults);
    
    #undef MAX_STACK_NODES
}
