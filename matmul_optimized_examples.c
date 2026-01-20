/*
 * Optimized matmul examples - demonstrating various optimization techniques
 * These are examples showing how production libraries optimize matrix multiplication
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Your Original Implementation (for comparison)
// ============================================================================

void matmul_naive(double** A, double** B, int m, int n, int p, double** result) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            result[i][j] = 0.0;
            for(int k = 0; k < n; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// ============================================================================
// Optimization 1: Loop Reordering (Better Cache Performance)
// ============================================================================

/**
 * Reorder loops to improve cache locality.
 * Instead of: i -> j -> k
 * We do: i -> k -> j
 *
 * This accesses B[k][j] sequentially (better cache behavior)
 * and accumulates into result[i][j] which stays in cache.
 */
void matmul_reordered(double** A, double** B, int m, int n, int p, double** result) {
    // Initialize result to zero
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            result[i][j] = 0.0;
        }
    }

    // Reordered: i -> k -> j
    for(int i = 0; i < m; i++) {
        for(int k = 0; k < n; k++) {
            double a_ik = A[i][k];  // Load once, use multiple times
            for(int j = 0; j < p; j++) {
                result[i][j] += a_ik * B[k][j];  // Sequential access to B
            }
        }
    }
}

// ============================================================================
// Optimization 2: Blocking/Tiling (Cache-Aware)
// ============================================================================

/**
 * Blocked matrix multiplication for better cache utilization.
 * Processes the matrix in blocks that fit in cache.
 *
 * Block size (BLOCK_SIZE) should be chosen based on cache size.
 * Typical values: 32-128 for L1 cache
 */
#define BLOCK_SIZE 64

void matmul_blocked(double** A, double** B, int m, int n, int p, double** result) {
    // Initialize result
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            result[i][j] = 0.0;
        }
    }

    // Process in blocks
    for(int ii = 0; ii < m; ii += BLOCK_SIZE) {
        for(int jj = 0; jj < p; jj += BLOCK_SIZE) {
            for(int kk = 0; kk < n; kk += BLOCK_SIZE) {
                // Process block
                int i_end = (ii + BLOCK_SIZE < m) ? ii + BLOCK_SIZE : m;
                int j_end = (jj + BLOCK_SIZE < p) ? jj + BLOCK_SIZE : p;
                int k_end = (kk + BLOCK_SIZE < n) ? kk + BLOCK_SIZE : n;

                for(int i = ii; i < i_end; i++) {
                    for(int j = jj; j < j_end; j++) {
                        double sum = result[i][j];
                        for(int k = kk; k < k_end; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        result[i][j] = sum;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Optimization 3: SIMD (Single Instruction Multiple Data)
// ============================================================================

#ifdef __AVX2__
#include <immintrin.h>

/**
 * SIMD-optimized version using AVX2 (256-bit vectors = 4 doubles at once).
 * This requires AVX2 support and compiler flags: -mavx2 -mfma
 *
 * Processes 4 elements of the inner loop in parallel.
 */
void matmul_simd(double** A, double** B, int m, int n, int p, double** result) {
    // Initialize result
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            result[i][j] = 0.0;
        }
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            __m256d sum_vec = _mm256_setzero_pd();  // Vector of 4 zeros
            int k;

            // Process 4 elements at a time
            for(k = 0; k < n - 3; k += 4) {
                // Load 4 elements from A[i][k:k+4]
                __m256d a_vec = _mm256_loadu_pd(&A[i][k]);

                // Load 4 elements from B[k:k+4][j] (not contiguous, need to gather)
                // For simplicity, we'll do this element-wise (real SIMD would use gather)
                __m256d b_vec = _mm256_set_pd(B[k+3][j], B[k+2][j], B[k+1][j], B[k][j]);

                // Multiply and accumulate: sum_vec += a_vec * b_vec
                sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
            }

            // Horizontal sum of the 4 elements in sum_vec
            double sum_array[4];
            _mm256_storeu_pd(sum_array, sum_vec);
            double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

            // Handle remaining elements
            for(; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }

            result[i][j] = sum;
        }
    }
}
#endif

// ============================================================================
// Optimization 4: Combined (Reordered + Blocking)
// ============================================================================

/**
 * Best practical optimization: combine loop reordering with blocking.
 * This gives good cache performance without requiring SIMD.
 */
void matmul_optimized(double** A, double** B, int m, int n, int p, double** result) {
    // Initialize result
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            result[i][j] = 0.0;
        }
    }

    // Blocked + reordered
    for(int ii = 0; ii < m; ii += BLOCK_SIZE) {
        int i_end = (ii + BLOCK_SIZE < m) ? ii + BLOCK_SIZE : m;

        for(int jj = 0; jj < p; jj += BLOCK_SIZE) {
            int j_end = (jj + BLOCK_SIZE < p) ? jj + BLOCK_SIZE : p;

            for(int kk = 0; kk < n; kk += BLOCK_SIZE) {
                int k_end = (kk + BLOCK_SIZE < n) ? kk + BLOCK_SIZE : n;

                // Reordered inner loops: i -> k -> j
                for(int i = ii; i < i_end; i++) {
                    for(int k = kk; k < k_end; k++) {
                        double a_ik = A[i][k];
                        for(int j = jj; j < j_end; j++) {
                            result[i][j] += a_ik * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Example Usage and Performance Comparison
// ============================================================================

/*
 * To compile and test:
 *
 * Basic: gcc -O2 matmul_optimized_examples.c -o matmul_test
 * SIMD:  gcc -O2 -mavx2 -mfma matmul_optimized_examples.c -o matmul_test
 *
 * Expected speedups (rough estimates):
 * - Reordered: 1.5-2x faster
 * - Blocked: 2-4x faster
 * - SIMD: 3-5x faster
 * - Combined: 4-8x faster
 *
 * Note: Actual speedups depend on:
 * - Matrix size
 * - Cache size
 * - CPU architecture
 * - Compiler optimizations
 */

int main() {
    // Example: 100x100 matrices
    int m = 100, n = 100, p = 100;

    // Allocate matrices (simplified - you'd use your zeros() function)
    double** A = (double**)malloc(m * sizeof(double*));
    double** B = (double**)malloc(n * sizeof(double*));
    double** result1 = (double**)malloc(m * sizeof(double*));
    double** result2 = (double**)malloc(m * sizeof(double*));

    for(int i = 0; i < m; i++) {
        A[i] = (double*)malloc(n * sizeof(double));
        result1[i] = (double*)calloc(p, sizeof(double));
        result2[i] = (double*)calloc(p, sizeof(double));
    }
    for(int i = 0; i < n; i++) {
        B[i] = (double*)malloc(p * sizeof(double));
    }

    // Initialize with test data
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            A[i][j] = 1.0;
        }
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < p; j++) {
            B[i][j] = 1.0;
        }
    }

    printf("Testing matrix multiplication optimizations...\n");

    // Test naive version
    matmul_naive(A, B, m, n, p, result1);
    printf("Naive version completed.\n");

    // Test optimized version
    matmul_optimized(A, B, m, n, p, result2);
    printf("Optimized version completed.\n");

    // Verify results are similar (allowing for floating point differences)
    int matches = 1;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            double diff = result1[i][j] - result2[i][j];
            if(diff > 0.0001 || diff < -0.0001) {
                matches = 0;
                break;
            }
        }
    }
    printf("Results match: %s\n", matches ? "Yes" : "No");

    // Cleanup
    for(int i = 0; i < m; i++) {
        free(A[i]);
        free(result1[i]);
        free(result2[i]);
    }
    for(int i = 0; i < n; i++) {
        free(B[i]);
    }
    free(A);
    free(B);
    free(result1);
    free(result2);

    return 0;
}
