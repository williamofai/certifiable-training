cmake_minimum_required(VERSION 3.16)
project(certifiable_training C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Strict compiler flags for safety-critical code
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -pedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wshadow -Wconversion -Wstrict-prototypes")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-common")

# DVM-required flags: no FMA, no fast-math, preserve operation order
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffp-contract=off")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-fast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-associative-math")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(DVM_SOURCES
    src/dvm/primitives.c
    src/dvm/prng.c
    src/dvm/compensated.c
    src/dvm/reduction.c
)

set(TRAINING_SOURCES
    src/training/forward.c
    src/training/backward.c
    src/training/optimizer.c
    src/training/scheduler.c
    src/training/permutation.c
)

set(LAYER_SOURCES
    src/layers/linear.c
    src/layers/conv2d.c
    src/layers/activation.c
    src/layers/normalization.c
)

set(AUDIT_SOURCES
    src/audit/merkle.c
    src/audit/checkpoint.c
)

# Build static library
add_library(certifiable_training STATIC
    ${DVM_SOURCES}
    ${TRAINING_SOURCES}
    ${LAYER_SOURCES}
    ${AUDIT_SOURCES}
)

# Link math library to main library (for LUT initialization)
target_link_libraries(certifiable_training m)

# Enable testing
enable_testing()

# Unit tests
add_executable(test_primitives tests/unit/test_primitives.c)
target_link_libraries(test_primitives certifiable_training m)
add_test(NAME test_primitives COMMAND test_primitives)

add_executable(test_prng tests/unit/test_prng.c)
target_link_libraries(test_prng certifiable_training m)
add_test(NAME test_prng COMMAND test_prng)

add_executable(test_compensated tests/unit/test_compensated.c)
target_link_libraries(test_compensated certifiable_training m)
add_test(NAME test_compensated COMMAND test_compensated)

add_executable(test_reduction tests/unit/test_reduction.c)
target_link_libraries(test_reduction certifiable_training m)
add_test(NAME test_reduction COMMAND test_reduction)

add_executable(test_forward tests/unit/test_forward.c)
target_link_libraries(test_forward certifiable_training m)
add_test(NAME test_forward COMMAND test_forward)

add_executable(test_backward tests/unit/test_backward.c)
target_link_libraries(test_backward certifiable_training m)
add_test(NAME test_backward COMMAND test_backward)

add_executable(test_optimizer tests/unit/test_optimizer.c)
target_link_libraries(test_optimizer certifiable_training m)
add_test(NAME test_optimizer COMMAND test_optimizer)

add_executable(test_bit_identity tests/unit/test_bit_identity.c)
target_link_libraries(test_bit_identity certifiable_training m)
add_test(NAME test_bit_identity COMMAND test_bit_identity)

add_executable(test_merkle tests/unit/test_merkle.c)
target_link_libraries(test_merkle certifiable_training m)
add_test(NAME test_merkle COMMAND test_merkle)

# Examples
add_executable(train_xor examples/train_xor.c)
target_link_libraries(train_xor certifiable_training m)

add_executable(verify_step examples/verify_step.c)
target_link_libraries(verify_step certifiable_training m)

# Custom targets
add_custom_target(test-all
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_primitives test_prng test_compensated test_reduction
            test_forward test_backward test_optimizer test_bit_identity test_merkle
)

add_executable(test_permutation tests/unit/test_permutation.c)
target_link_libraries(test_permutation certifiable_training m)
add_test(NAME test_permutation COMMAND test_permutation)
