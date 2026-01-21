/*
 * Optimized matmul examples - demonstrating various optimization techniques
 * These are examples showing how production libraries optimize matrix multiplication
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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
    // Reordered: i -> k -> j
    for(int i = 0; i < m; i++) {
        // Initialize this row right before we use it (better cache locality)
        for(int j = 0; j < p; j++) {
            result[i][j] = 0.0;
        }

        // Then accumulate into it
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
// Timing Utilities
// ============================================================================

/**
 * Get current time in seconds (high resolution)
 */
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/**
 * Time a matrix multiplication function
 */
double time_matmul(void (*func)(double**, double**, int, int, int, double**),
                   double** A, double** B, int m, int n, int p, double** result,
                   int iterations) {
    // Warmup
    func(A, B, m, n, p, result);

    double start = get_time();
    for(int i = 0; i < iterations; i++) {
        func(A, B, m, n, p, result);
    }
    double end = get_time();

    return (end - start) / iterations;
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
    // Large matrix test for meaningful performance differences
    int m = 1000, n = 1000, p = 1000;
    int iterations = 2;  // Number of runs to average

    printf("=================================================================\n");
    printf("Matrix Multiplication Performance Benchmark\n");
    printf("=================================================================\n");
    printf("Matrix size: %d x %d x %d\n", m, n, p);
    printf("Iterations per test: %d\n", iterations);
    printf("=================================================================\n\n");

    // Allocate matrices
    double** A = (double**)malloc(m * sizeof(double*));
    double** B = (double**)malloc(n * sizeof(double*));
    double** result_naive = (double**)malloc(m * sizeof(double*));
    double** result_reordered = (double**)malloc(m * sizeof(double*));
    double** result_blocked = (double**)malloc(m * sizeof(double*));
    double** result_optimized = (double**)malloc(m * sizeof(double*));
    #ifdef __AVX2__
    double** result_simd = (double**)malloc(m * sizeof(double*));
    #endif

    for(int i = 0; i < m; i++) {
        A[i] = (double*)malloc(n * sizeof(double));
        result_naive[i] = (double*)calloc(p, sizeof(double));
        result_reordered[i] = (double*)calloc(p, sizeof(double));
        result_blocked[i] = (double*)calloc(p, sizeof(double));
        result_optimized[i] = (double*)calloc(p, sizeof(double));
        #ifdef __AVX2__
        result_simd[i] = (double*)calloc(p, sizeof(double));
        #endif
    }
    for(int i = 0; i < n; i++) {
        B[i] = (double*)malloc(p * sizeof(double));
    }

    // Initialize with random test data
    srand(42);  // Fixed seed for reproducibility
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
        }
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < p; j++) {
            B[i][j] = (double)rand() / RAND_MAX;
        }
    }

    printf("Running benchmarks...\n\n");

    // Time each implementation
    double time_naive = time_matmul(matmul_naive, A, B, m, n, p, result_naive, iterations);
    printf("1. Naive:                    %.4f seconds\n", time_naive);

    double time_reordered = time_matmul(matmul_reordered, A, B, m, n, p, result_reordered, iterations);
    printf("2. Reordered:                %.4f seconds (%.2fx speedup)\n",
           time_reordered, time_naive / time_reordered);

    double time_blocked = time_matmul(matmul_blocked, A, B, m, n, p, result_blocked, iterations);
    printf("3. Blocked:                  %.4f seconds (%.2fx speedup)\n",
           time_blocked, time_naive / time_blocked);

    double time_optimized = time_matmul(matmul_optimized, A, B, m, n, p, result_optimized, iterations);
    printf("4. Optimized (Blocked+Reord): %.4f seconds (%.2fx speedup)\n",
           time_optimized, time_naive / time_optimized);

    #ifdef __AVX2__
    double time_simd = time_matmul(matmul_simd, A, B, m, n, p, result_simd, iterations);
    printf("5. SIMD (AVX2):              %.4f seconds (%.2fx speedup)\n",
           time_simd, time_naive / time_simd);
    #else
    printf("5. SIMD (AVX2):              [Not compiled - use -mavx2 -mfma flags]\n");
    #endif

    printf("\n=================================================================\n");
    printf("Fastest implementation: ");

    double fastest_time = time_naive;
    const char* fastest_name = "Naive";

    if(time_reordered < fastest_time) {
        fastest_time = time_reordered;
        fastest_name = "Reordered";
    }
    if(time_blocked < fastest_time) {
        fastest_time = time_blocked;
        fastest_name = "Blocked";
    }
    if(time_optimized < fastest_time) {
        fastest_time = time_optimized;
        fastest_name = "Optimized (Blocked+Reordered)";
    }
    #ifdef __AVX2__
    if(time_simd < fastest_time) {
        fastest_time = time_simd;
        fastest_name = "SIMD (AVX2)";
    }
    #endif

    printf("%s (%.4f seconds)\n", fastest_name, fastest_time);
    printf("=================================================================\n\n");

    // Verify results are similar (allowing for floating point differences)
    printf("Verifying correctness...\n");
    int all_match = 1;
    double max_diff = 0.0;

    // Compare all results against naive
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            double ref = result_naive[i][j];

            double diff_reordered = fabs(result_reordered[i][j] - ref);
            double diff_blocked = fabs(result_blocked[i][j] - ref);
            double diff_optimized = fabs(result_optimized[i][j] - ref);

            if(diff_reordered > max_diff) max_diff = diff_reordered;
            if(diff_blocked > max_diff) max_diff = diff_blocked;
            if(diff_optimized > max_diff) max_diff = diff_optimized;

            if(diff_reordered > 0.001 || diff_blocked > 0.001 || diff_optimized > 0.001) {
                all_match = 0;
            }

            #ifdef __AVX2__
            double diff_simd = fabs(result_simd[i][j] - ref);
            if(diff_simd > max_diff) max_diff = diff_simd;
            if(diff_simd > 0.001) all_match = 0;
            #endif
        }
    }

    printf("Results match: %s (max difference: %.6f)\n",
           all_match ? "Yes" : "No", max_diff);

    // Cleanup
    for(int i = 0; i < m; i++) {
        free(A[i]);
        free(result_naive[i]);
        free(result_reordered[i]);
        free(result_blocked[i]);
        free(result_optimized[i]);
        #ifdef __AVX2__
        free(result_simd[i]);
        #endif
    }
    for(int i = 0; i < n; i++) {
        free(B[i]);
    }
    free(A);
    free(B);
    free(result_naive);
    free(result_reordered);
    free(result_blocked);
    free(result_optimized);
    #ifdef __AVX2__
    free(result_simd);
    #endif

    return 0;
}
