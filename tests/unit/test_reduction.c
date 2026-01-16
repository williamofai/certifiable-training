/**
 * @file test_reduction.c
 * @project Certifiable Training
 * @brief Unit tests for fixed-topology reduction tree
 *
 * @traceability SRS-004, CT-MATH-001 §9.1
 *
 * @author William Murray
 * @copyright Copyright (c) 2026 The Murray Family Innovation Trust.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "ct_types.h"
#include "reduction.h"
#include "compensated.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(fn) do { \
    printf("  %-50s ", #fn); \
    tests_run++; \
    if (fn()) { printf("PASS\n"); tests_passed++; } \
    else { printf("FAIL\n"); } \
} while(0)

/* ============================================================================
 * Test: Helper Functions
 * ============================================================================ */

static int test_node_count_basic(void)
{
    /* n leaves -> 2n-1 nodes */
    if (ct_reduction_node_count(1) != 1) return 0;
    if (ct_reduction_node_count(2) != 3) return 0;
    if (ct_reduction_node_count(4) != 7) return 0;
    if (ct_reduction_node_count(8) != 15) return 0;
    if (ct_reduction_node_count(64) != 127) return 0;
    
    return 1;
}

static int test_node_count_invalid(void)
{
    /* Zero and over-limit should return 0 */
    if (ct_reduction_node_count(0) != 0) return 0;
    if (ct_reduction_node_count(CT_MAX_LEAVES + 1) != 0) return 0;
    
    return 1;
}

static int test_buffer_size(void)
{
    size_t size = ct_reduction_buffer_size(4);
    size_t expected = 7 * sizeof(ct_reduction_node_t);
    
    return size == expected;
}

/* ============================================================================
 * Test: Initialization
 * ============================================================================ */

static int test_init_single_leaf(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[1];
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_reduction_init(&tree, nodes, 1, 1000, &faults);
    
    if (err != CT_OK) return 0;
    if (tree.num_leaves != 1) return 0;
    if (tree.num_internal != 0) return 0;
    if (tree.num_nodes != 1) return 0;
    if (tree.root_index != 0) return 0;
    if (tree.depth != 0) return 0;
    
    return 1;
}

static int test_init_two_leaves(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[3];
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_reduction_init(&tree, nodes, 2, 1000, &faults);
    
    if (err != CT_OK) return 0;
    if (tree.num_leaves != 2) return 0;
    if (tree.num_internal != 1) return 0;
    if (tree.num_nodes != 3) return 0;
    if (tree.root_index != 2) return 0;
    if (tree.depth != 1) return 0;
    
    return 1;
}

static int test_init_four_leaves(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_reduction_init(&tree, nodes, 4, 1000, &faults);
    
    if (err != CT_OK) return 0;
    if (tree.num_leaves != 4) return 0;
    if (tree.num_internal != 3) return 0;
    if (tree.num_nodes != 7) return 0;
    if (tree.root_index != 6) return 0;
    if (tree.depth != 2) return 0;
    
    return 1;
}

static int test_init_null_safe(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    
    /* NULL tree */
    if (ct_reduction_init(NULL, nodes, 4, 0, &faults) != CT_ERR_NULL) return 0;
    
    /* NULL nodes */
    if (ct_reduction_init(&tree, NULL, 4, 0, &faults) != CT_ERR_NULL) return 0;
    
    return 1;
}

static int test_init_zero_leaves_fails(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[1];
    ct_fault_flags_t faults = {0};
    
    ct_error_t err = ct_reduction_init(&tree, nodes, 0, 0, &faults);
    
    return (err == CT_ERR_CONFIG) && (faults.domain == 1);
}

static int test_init_op_ids_assigned(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 4, 1000, &faults);
    
    /* Each node should have op_id = base + index */
    for (uint32_t i = 0; i < 7; i++) {
        if (ct_reduction_op_id(&tree, i) != 1000 + i) return 0;
    }
    
    return 1;
}

/* ============================================================================
 * Test: Tree Structure
 * ============================================================================ */

static int test_is_leaf(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 4, 0, &faults);
    
    /* Nodes 0-3 are leaves */
    if (!ct_reduction_is_leaf(&tree, 0)) return 0;
    if (!ct_reduction_is_leaf(&tree, 1)) return 0;
    if (!ct_reduction_is_leaf(&tree, 2)) return 0;
    if (!ct_reduction_is_leaf(&tree, 3)) return 0;
    
    /* Nodes 4-6 are internal */
    if (ct_reduction_is_leaf(&tree, 4)) return 0;
    if (ct_reduction_is_leaf(&tree, 5)) return 0;
    if (ct_reduction_is_leaf(&tree, 6)) return 0;
    
    return 1;
}

static int test_leaf_children_are_markers(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 4, 0, &faults);
    
    /* All leaves should have CT_LEAF_MARKER children */
    for (uint32_t i = 0; i < 4; i++) {
        if (ct_reduction_left_child(&tree, i) != CT_LEAF_MARKER) return 0;
        if (ct_reduction_right_child(&tree, i) != CT_LEAF_MARKER) return 0;
    }
    
    return 1;
}

static int test_internal_nodes_have_children(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 4, 0, &faults);
    
    /* Internal nodes should have valid children */
    for (uint32_t i = 4; i < 7; i++) {
        uint32_t left = ct_reduction_left_child(&tree, i);
        uint32_t right = ct_reduction_right_child(&tree, i);
        
        if (left == CT_LEAF_MARKER && right == CT_LEAF_MARKER) {
            return 0;  /* Internal node must have at least one child */
        }
    }
    
    return 1;
}

static int test_depth_calculation(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[127];
    ct_fault_flags_t faults = {0};
    
    /* 1 leaf -> depth 0 */
    ct_reduction_init(&tree, nodes, 1, 0, &faults);
    if (ct_reduction_depth(&tree) != 0) return 0;
    
    /* 2 leaves -> depth 1 */
    ct_reduction_init(&tree, nodes, 2, 0, &faults);
    if (ct_reduction_depth(&tree) != 1) return 0;
    
    /* 4 leaves -> depth 2 */
    ct_reduction_init(&tree, nodes, 4, 0, &faults);
    if (ct_reduction_depth(&tree) != 2) return 0;
    
    /* 8 leaves -> depth 3 */
    ct_reduction_init(&tree, nodes, 8, 0, &faults);
    if (ct_reduction_depth(&tree) != 3) return 0;
    
    /* 64 leaves -> depth 6 */
    ct_reduction_init(&tree, nodes, 64, 0, &faults);
    if (ct_reduction_depth(&tree) != 6) return 0;
    
    return 1;
}

/* ============================================================================
 * Test: Reduction Operations
 * ============================================================================ */

static int test_reduce_single_value(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[1];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 1, 0, &faults);
    
    int64_t values[] = {12345};
    int64_t result = ct_reduction_reduce_64(&tree, values, &faults);
    
    return result == 12345;
}

static int test_reduce_two_values(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[3];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 2, 0, &faults);
    
    int64_t values[] = {100, 200};
    int64_t result = ct_reduction_reduce_64(&tree, values, &faults);
    
    return result == 300;
}

static int test_reduce_four_values(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 4, 0, &faults);
    
    int64_t values[] = {10, 20, 30, 40};
    int64_t result = ct_reduction_reduce_64(&tree, values, &faults);
    
    return result == 100;
}

static int test_reduce_power_of_two(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[127];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 64, 0, &faults);
    
    /* Sum 1 to 64 = 64*65/2 = 2080 */
    int64_t values[64];
    for (int i = 0; i < 64; i++) {
        values[i] = i + 1;
    }
    
    int64_t result = ct_reduction_reduce_64(&tree, values, &faults);
    
    return result == 2080;
}

static int test_reduce_32_basic(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 4, 0, &faults);
    
    int32_t values[] = {100, 200, 300, 400};
    int64_t result = ct_reduction_reduce_32(&tree, values, &faults);
    
    return result == 1000;
}

static int test_reduce_negative_values(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 4, 0, &faults);
    
    int64_t values[] = {100, -50, 200, -100};
    int64_t result = ct_reduction_reduce_64(&tree, values, &faults);
    
    return result == 150;
}

static int test_reduce_zeros(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 4, 0, &faults);
    
    int64_t values[] = {0, 0, 0, 0};
    int64_t result = ct_reduction_reduce_64(&tree, values, &faults);
    
    return result == 0;
}

/* ============================================================================
 * Test: Determinism
 * ============================================================================ */

static int test_reduce_deterministic(void)
{
    ct_reduction_tree_t tree1, tree2;
    ct_reduction_node_t nodes1[127], nodes2[127];
    ct_fault_flags_t faults1 = {0}, faults2 = {0};
    
    ct_reduction_init(&tree1, nodes1, 64, 0, &faults1);
    ct_reduction_init(&tree2, nodes2, 64, 0, &faults2);
    
    int64_t values[64];
    for (int i = 0; i < 64; i++) {
        values[i] = (int64_t)i * 12345 - 400000;
    }
    
    int64_t result1 = ct_reduction_reduce_64(&tree1, values, &faults1);
    int64_t result2 = ct_reduction_reduce_64(&tree2, values, &faults2);
    
    return result1 == result2;
}

static int test_reduce_matches_sequential(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[127];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 64, 0, &faults);
    
    int64_t values[64];
    for (int i = 0; i < 64; i++) {
        values[i] = i + 1;
    }
    
    /* Tree reduction */
    int64_t tree_result = ct_reduction_reduce_64(&tree, values, &faults);
    
    /* Sequential sum (for comparison) */
    int64_t seq_result = ct_comp_sum_array(values, 64, &faults);
    
    return tree_result == seq_result;
}

/* ============================================================================
 * Test: Traced Reduction
 * ============================================================================ */

static int trace_count = 0;

static void trace_callback(uint32_t node_index, 
                          const ct_comp_accum_t *accum,
                          void *context)
{
    (void)node_index;
    (void)accum;
    (void)context;
    trace_count++;
}

static int test_traced_reduction_calls_callback(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 4, 0, &faults);
    
    int64_t values[] = {10, 20, 30, 40};
    trace_count = 0;
    
    int64_t result = ct_reduction_reduce_traced(&tree, values, 
                                                 trace_callback, NULL, &faults);
    
    /* Should call callback for all 7 nodes */
    if (trace_count != 7) return 0;
    
    /* Result should still be correct */
    return result == 100;
}

static int test_traced_matches_untraced(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[127];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 64, 0, &faults);
    
    int64_t values[64];
    for (int i = 0; i < 64; i++) {
        values[i] = i * 100;
    }
    
    int64_t traced = ct_reduction_reduce_traced(&tree, values, NULL, NULL, &faults);
    int64_t untraced = ct_reduction_reduce_64(&tree, values, &faults);
    
    return traced == untraced;
}

/* ============================================================================
 * Test: Compensated Accuracy
 * ============================================================================ */

static int test_reduce_large_small_values(void)
{
    /*
     * Mix of large and small values tests compensated arithmetic
     * through the reduction tree.
     */
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[15];
    ct_fault_flags_t faults = {0};
    
    ct_reduction_init(&tree, nodes, 8, 0, &faults);
    
    int64_t values[] = {
        (int64_t)1 << 40,   /* Large */
        1,                   /* Small */
        (int64_t)1 << 40,   /* Large */
        2,                   /* Small */
        (int64_t)1 << 40,   /* Large */
        3,                   /* Small */
        (int64_t)1 << 40,   /* Large */
        4                    /* Small */
    };
    
    int64_t result = ct_reduction_reduce_64(&tree, values, &faults);
    int64_t expected = 4 * ((int64_t)1 << 40) + 1 + 2 + 3 + 4;
    
    return result == expected;
}

/* ============================================================================
 * Test: Edge Cases
 * ============================================================================ */

static int test_reduce_null_safe(void)
{
    ct_reduction_tree_t tree;
    ct_reduction_node_t nodes[7];
    ct_fault_flags_t faults = {0};
    int64_t values[] = {1, 2, 3, 4};
    
    ct_reduction_init(&tree, nodes, 4, 0, &faults);
    
    /* NULL tree */
    if (ct_reduction_reduce_64(NULL, values, &faults) != 0) return 0;
    
    /* NULL values */
    if (ct_reduction_reduce_64(&tree, NULL, &faults) != 0) return 0;
    
    return 1;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void)
{
    printf("==============================================\n");
    printf("Certifiable Training - Reduction Tree Tests\n");
    printf("Traceability: SRS-004, CT-MATH-001 §9.1\n");
    printf("==============================================\n\n");
    
    printf("Helper functions:\n");
    RUN_TEST(test_node_count_basic);
    RUN_TEST(test_node_count_invalid);
    RUN_TEST(test_buffer_size);
    
    printf("\nInitialization:\n");
    RUN_TEST(test_init_single_leaf);
    RUN_TEST(test_init_two_leaves);
    RUN_TEST(test_init_four_leaves);
    RUN_TEST(test_init_null_safe);
    RUN_TEST(test_init_zero_leaves_fails);
    RUN_TEST(test_init_op_ids_assigned);
    
    printf("\nTree structure:\n");
    RUN_TEST(test_is_leaf);
    RUN_TEST(test_leaf_children_are_markers);
    RUN_TEST(test_internal_nodes_have_children);
    RUN_TEST(test_depth_calculation);
    
    printf("\nReduction operations:\n");
    RUN_TEST(test_reduce_single_value);
    RUN_TEST(test_reduce_two_values);
    RUN_TEST(test_reduce_four_values);
    RUN_TEST(test_reduce_power_of_two);
    RUN_TEST(test_reduce_32_basic);
    RUN_TEST(test_reduce_negative_values);
    RUN_TEST(test_reduce_zeros);
    
    printf("\nDeterminism:\n");
    RUN_TEST(test_reduce_deterministic);
    RUN_TEST(test_reduce_matches_sequential);
    
    printf("\nTraced reduction:\n");
    RUN_TEST(test_traced_reduction_calls_callback);
    RUN_TEST(test_traced_matches_untraced);
    
    printf("\nCompensated accuracy:\n");
    RUN_TEST(test_reduce_large_small_values);
    
    printf("\nEdge cases:\n");
    RUN_TEST(test_reduce_null_safe);
    
    printf("\n==============================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("==============================================\n");
    
    return (tests_passed == tests_run) ? 0 : 1;
}
