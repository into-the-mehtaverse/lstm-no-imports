/*
 * LSTM implementation worksheet in C - no external dependencies.
 * Implement TODOs to complete the forward pass.
 *
 * Matrix conventions:
 * - Matrices are represented as double** (double pointer to double)
 * - Row-major order: matrix[row][col]
 * - x is an array of T matrices where each x[t] is (n_x x m)
 * - a and c are (n_a x m)
 * - Params:
 *   - Wf, Wi, Wc, Wo: (n_a x (n_a+n_x))
 *   - bf, bi, bc, bo: (n_a x 1) broadcast across batch
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define E 2.718281828459045  // Euler's number

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Create a matrix of zeros with given dimensions.
 * Caller must free the returned matrix using free_matrix().
 *
 * Args:
 *   rows: number of rows
 *   cols: number of columns
 *
 * Returns:
 *   double**: allocated matrix filled with zeros, NULL on failure
 */
double** zeros(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (!matrix) return NULL;

    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)calloc(cols, sizeof(double));
        if (!matrix[i]) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

/**
 * Free a dynamically allocated matrix.
 *
 * Args:
 *   matrix: matrix to free
 *   rows: number of rows in the matrix
 */
void free_matrix(double** matrix, int rows) {
    if (!matrix) return;
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/**
 * Matrix multiply A @ B.
 *
 * Args:
 *   A: matrix shape (m, n)
 *   B: matrix shape (n, p)
 *   m, n, p: dimensions
 *   result: output matrix shape (m, p) - must be pre-allocated
 */
void matmul(double** A, double** B, int m, int n, int p, double** result) {
    // TODO: Implement matrix multiplication
    // result[i][j] = sum over k of A[i][k] * B[k][j]
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            result[i][j] = 0.0;
            for(int k = 0; k < n; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

/**
 * Element-wise addition A + B (broadcasting if needed).
 *
 * Args:
 *   A: matrix shape (m, n)
 *   B: matrix shape (m, n) or broadcastable with input like [[scalar]]
 *   m, n: dimensions
 *   broadcast: integer value of either 0 or 1
 *   result: output matrix shape (m, n) - must be pre-allocated
 */
void add(double** A, double** B, int m, int n, double** result, int broadcast) {

    if(broadcast) {
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                result[i][j] = A[i][j] + B[0][0];
            }
        }
    } else {
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                result[i][j] = A[i][j] + B[i][j];
            }
        }
    }

}

/**
 * Add bias vector b to each column of Z (broadcasting).
 *
 * Args:
 *   Z: matrix shape (n_a, m)
 *   b: matrix shape (n_a, 1)
 *   n_a, m: dimensions
 *   result: output matrix shape (n_a, m) - must be pre-allocated
 */
void add_bias(double** Z, double** b, int n_a, int m, double** result) {

    for(int i = 0; i < n_a; i++) {
        for(int j = 0; j < m; j++) {
            result[i][j] = Z[i][j] + b[i][0];
        }
    }
}

/**
 * Element-wise (Hadamard) product A * B.
 *
 * Args:
 *   A: matrix shape (m, n)
 *   B: matrix shape (m, n)
 *   m, n: dimensions
 *   result: output matrix shape (m, n) - must be pre-allocated
 */
void hadamard(double** A, double** B, int m, int n, double** result) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            result[i][j] = A[i][j] * B[i][j];
        }
    }
}

/**
 * Apply function fn element-wise to matrix A.
 *
 * Args:
 *   A: input matrix
 *   rows, cols: dimensions
 *   fn: function pointer to apply element-wise
 *   result: output matrix - must be pre-allocated
 *
 */
void apply_fn(double** A, int rows, int cols, double (*fn)(double), double** result) {

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            result[i][j] = fn(A[i][j]);
        }
    }
}

/**
 * Concatenate matrices A and B along columns (horizontal concat).
 *
 * Args:
 *   A: matrix shape (m, n)
 *   B: matrix shape (m, p)
 *   m, n, p: dimensions
 *   result: output matrix shape (m, n+p) - must be pre-allocated
 *
 */
void concat_rows(double** A, double** B, int m, int n, int p, double** result) {

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < (n + p); j++) {
            if (j < n) {
                result[i][j] = A[i][j];
            } else {
                result[i][j] = B[i][j - n];
            }
        }
    }
}

/**
 * Concatenate matrices A and B along rows (vertical concat).
 * Stacks A on top of B.
 *
 * Args:
 *   A: matrix shape (m, n)
 *   B: matrix shape (p, n)  // same number of columns
 *   m, n, p: dimensions
 *   result: output matrix shape (m+p, n) - must be pre-allocated
 *
 * TODO: Implement vertical concatenation (stack A on top of B)
 */
void concat_cols(double** A, double** B, int m, int n, int p, double** result) {

    for (int i = 0; i < (m + p); i++) {
        for (int j = 0; j < n; j++) {
            if (i < m) {
                result[i][j] = A[i][j];
            } else {
                result[i][j] = B[i - m][j];
            }
        }
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * Clamp value to range [min_val, max_val].
 */
double clamp(double x, double min_val, double max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

/**
 * Approximate exp(x) using E ** x.
 * Clamp input to [-30, 30] to avoid overflow.
 *
 * Args:
 *   x: input value
 *
 * Returns:
 *   double: E ** clamp(x, -30, 30)
 *
 * TODO: Implement exp approximation with clamping
 */

double exp_approx(double x) {

    double clamped = clamp(x, -30, 30);


    return pow(E, clamped);
}

/**
 * Sigmoid activation: 1 / (1 + exp(-x)).
 *
 * Args:
 *   x: input value
 *
 * Returns:
 *   double: sigmoid(x)
 */
double sigmoid(double x) {

    return 1 / (1 + exp_approx(-x));
}

/**
 * Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
 *
 * Args:
 *   x: input value
 *
 * Returns:
 *   double: tanh(x)
 */
double tanh_activation(double x) {
    return (exp_approx(x) - exp_approx(-x)) / (exp_approx(x) + exp_approx(-x));
}

// ============================================================================
// LSTM Forward Pass
// ============================================================================

/**
 * Forward pass for single LSTM cell.
 *
 * Args:
 *   xt: input at time t, shape (n_x, m)
 *   a_prev: previous hidden state, shape (n_a, m)
 *   c_prev: previous cell state, shape (n_a, m)
 *   params: LSTM parameters (weights and biases)
 *   n_x, n_a, m: dimensions
 *   a_next: output hidden state, shape (n_a, m) - must be pre-allocated
 *   c_next: output cell state, shape (n_a, m) - must be pre-allocated
 *
 * Equations to implement:
 *   concat = [a_prev; xt]  # shape (n_a+n_x, m)
 *   ft = sigmoid(Wf @ concat + bf)  # forget gate
 *   it = sigmoid(Wi @ concat + bi)  # input gate
 *   cct = tanh(Wc @ concat + bc)    # candidate cell
 *   c_next = ft * c_prev + it * cct # new cell state
 *   ot = sigmoid(Wo @ concat + bo)  # output gate
 *   a_next = ot * tanh(c_next)      # new hidden state
 *
 * TODO: Implement LSTM cell forward equations
 */

typedef struct {
    double** Wf, **Wi, **Wc, **Wo;
    double** bf, **bi, **bc, **bo;
} LSTM_Params;

void lstm_cell_forward(double** xt, double** a_prev, double** c_prev,
                        LSTM_Params* params,
                       int n_x, int n_a, int m,
                       double** a_next, double** c_next) {
    // TODO: Implement LSTM cell forward pass
    // Hint: You'll need temporary matrices for intermediate calculations
    // Remember to free them when done!

    double** concat = zeros(n_a + n_x, m);
    double** bias_temp = zeros(n_a, m);
    double** gate_temp = zeros(n_a, m);

    concat_cols(a_prev, xt, n_a, m, n_x, concat);

    // forget gate

    double** ft = zeros(n_a, m);

    matmul(params->Wf, concat, n_a, n_a+n_x, m, gate_temp);
    add_bias(gate_temp, params->bf, n_a, m, bias_temp);
    apply_fn(bias_temp, n_a, m, sigmoid, ft);

    // input gate
    double** it = zeros(n_a, m);

    matmul(params->Wi, concat, n_a, n_a+n_x, m, gate_temp);
    add_bias(gate_temp, params->bi, n_a, m, bias_temp);
    apply_fn(bias_temp, n_a, m, sigmoid, it);

    // candidate cell
    double** cct = zeros(n_a, m);

    matmul(params->Wc, concat, n_a, n_a+n_x, m, gate_temp);
    add_bias(gate_temp, params->bc, n_a, m, bias_temp);
    apply_fn(bias_temp, n_a, m, tanh_activation, cct);

    // new cell state

    double** temp_result1 = zeros(n_a, m);

    hadamard(ft, c_prev, n_a, m, temp_result1);
    hadamard(it, cct, n_a, m, gate_temp);

    add(temp_result1, gate_temp, n_a, m, c_next, 0);

    // output gate

    double** ot = zeros(n_a, m);
    matmul(params->Wo, concat, n_a, n_a+n_x, m, gate_temp);
    add_bias(gate_temp, params->bo, n_a, m, bias_temp);
    apply_fn(bias_temp, n_a, m, sigmoid, ot);

    // new hidden state

    apply_fn(c_next, n_a, m, tanh_activation, gate_temp);
    hadamard(ot, gate_temp, n_a, m, a_next);

    // free temp matrices
    free_matrix(concat, n_a + n_x);
    free_matrix(bias_temp, n_a);
    free_matrix(gate_temp, n_a);
    free_matrix(ft, n_a);
    free_matrix(it, n_a);
    free_matrix(cct, n_a);
    free_matrix(temp_result1, n_a);
    free_matrix(ot, n_a);


}

/**
 * Forward pass for LSTM over all timesteps.
 *
 * Args:
 *   x: array of T input matrices, each x[t] shape (n_x, m)
 *   a0: initial hidden state, shape (n_a, m)
 *   c0: initial cell state, shape (n_a, m)
 *   params: LSTM parameters (weights and biases)
 *   n_x, n_a, m, T: dimensions
 *   a_T: final hidden state, shape (n_a, m) - must be pre-allocated
 *   c_T: final cell state, shape (n_a, m) - must be pre-allocated
 *
 * TODO: Loop over timesteps t=0..T-1 and call lstm_cell_forward
 */
void lstm_forward(double*** x, double** a0, double** c0,
                  LSTM_Params* params,
                  int n_x, int n_a, int m, int T,
                  double** a_T, double** c_T) {
    // TODO: Implement forward pass loop over timesteps
    // Hint: Initialize a and c from a0 and c0
    // For each timestep, call lstm_cell_forward and update a, c
    // Copy final states to a_T and c_T


    // allocate a and c
    double** a = zeros(n_a, m);
    double** c = zeros(n_a, m);

    // copy a0 and c0 into a and c to start
    for (int i = 0; i < n_a; i++) {
        for (int j = 0; j < m; j++) {
            a[i][j] = a0[i][j];
            c[i][j] = c0[i][j];
        }
    }

    // loop over all time steps
    for (int t = 0; t < T; t++) {
        lstm_cell_forward(x[t], a, c, params, n_x, n_a, m, a, c);
    }

    // copy final a and c into a_T and c_T
    for (int i = 0; i < n_a; i++) {
        for (int j = 0; j < m; j++) {
            a_T[i][j] = a[i][j];
            c_T[i][j] = c[i][j];
        }
    }

    // free temps
    free_matrix(a, n_a);
    free_matrix(c, n_a);

}

// ============================================================================
// Self-Test Harness
// ============================================================================

int main() {
    // Dimensions
    int n_x = 2;  // input dimension
    int n_a = 3;  // hidden dimension
    int m = 1;    // batch size
    int T = 2;    // sequence length

    // Allocate input sequence x: array of T matrices, each (n_x, m)
    double*** x = (double***)malloc(T * sizeof(double**));
    x[0] = zeros(n_x, m);
    x[1] = zeros(n_x, m);
    x[0][0][0] = 0.1;
    x[0][1][0] = 0.2;
    x[1][0][0] = 0.3;
    x[1][1][0] = 0.4;

    // Initialize initial states
    double** a0 = zeros(n_a, m);
    double** c0 = zeros(n_a, m);

    // Initialize parameters (small constant weights, zero biases)
    double** Wf = zeros(n_a, n_a + n_x);
    double** Wi = zeros(n_a, n_a + n_x);
    double** Wc = zeros(n_a, n_a + n_x);
    double** Wo = zeros(n_a, n_a + n_x);
    double** bf = zeros(n_a, 1);
    double** bi = zeros(n_a, 1);
    double** bc = zeros(n_a, 1);
    double** bo = zeros(n_a, 1);

    // Fill weights with 0.1 (small constant)
    for (int i = 0; i < n_a; i++) {
        for (int j = 0; j < n_a + n_x; j++) {
            Wf[i][j] = 0.1;
            Wi[i][j] = 0.1;
            Wc[i][j] = 0.1;
            Wo[i][j] = 0.1;
        }
    }
    // Biases are already zeros from zeros() function

    // Allocate output matrices
    double** a_T = zeros(n_a, m);
    double** c_T = zeros(n_a, m);

    // Create params struct
    LSTM_Params params;
    params.Wf = Wf;
    params.Wi = Wi;
    params.Wc = Wc;
    params.Wo = Wo;
    params.bf = bf;
    params.bi = bi;
    params.bc = bc;
    params.bo = bo;

    // Run forward pass
    printf("Running LSTM forward pass...\n");
    printf("Input x: %d timesteps, each shape (%d, %d)\n", T, n_x, m);
    printf("Initial a0 shape: (%d, %d)\n", n_a, m);
    printf("Initial c0 shape: (%d, %d)\n", n_a, m);

    lstm_forward(x, a0, c0, &params, n_x, n_a, m, T, a_T, c_T);

    printf("\nFinal a_T shape: (%d, %d)\n", n_a, m);
    printf("Final c_T shape: (%d, %d)\n", n_a, m);
    printf("\na_T = \n");
    for (int i = 0; i < n_a; i++) {
        printf("  [%.10f]\n", a_T[i][0]);
    }
    printf("c_T = \n");
    for (int i = 0; i < n_a; i++) {
        printf("  [%.10f]\n", c_T[i][0]);
    }

    // Cleanup
    free_matrix(a_T, n_a);
    free_matrix(c_T, n_a);
    free_matrix(bf, n_a);
    free_matrix(bi, n_a);
    free_matrix(bc, n_a);
    free_matrix(bo, n_a);
    free_matrix(Wf, n_a);
    free_matrix(Wi, n_a);
    free_matrix(Wc, n_a);
    free_matrix(Wo, n_a);
    free_matrix(a0, n_a);
    free_matrix(c0, n_a);
    free_matrix(x[0], n_x);
    free_matrix(x[1], n_x);
    free(x);

    return 0;
}
