


Here I'm going to attempt to explain various memory and processing optimization techniques through the example of matrix multiplication. We'll start with a naive matmul operation in C, and incrementally improve it with various techniques until it hyper optimized.


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


The above algorithm algorithm has O(m + n + p) time. The first step we can take to optimize it is to change how we use the memory to access rows sequentially so we have less cache misses and less memory cycles. Take a look at the below diagram to understand how CPU memory is generally configured. To be clear, these are actual hardware configurations on a CPU.

1. Registers - these focus on the active operation, require 0 memory cycles, and typically hold 16-32 values each
2. L1 cache (1 - 3 cycles, ~32 kb) <- this is the fastest and smallest cache.
     if this misses, go to L2 cache
3. L2 cache (10 cycles, ~256 kb) <- medium speed / size cache
     if this misses, go to L3 cache
4. L3 cache (40 cycles, ~8 mb) <- slower, larger cache
     if this misses, go to RAM
5. RAM (100-300 cycles, ~16 GB) <- Slowest, largest cache

Some questions you might have after seeing this:
Why can't we have unlimited L1 cache on a CPU?
    Because fast memory is expensive and takes more spaces, whereas slower memory is cheaper and denser. Our goal is to reduce the amount of wasted memory and reduce trips to more expensive cache.
What is a cycle?
    A cycle is one tick of the CPU clock. The CPU clock executes instructions in sync with this clock. For example, if you have 3.0GHz (gigahertz) cpu, then that's 3,000,000,000 cycles per second, and if you do the math, one second is 0.33 nanoseconds. So if you look at the above diagram, a trip to L1 cache takes 1-3 nanoseconds versus a trip to RAM takes 100-300 nanoseconds. That's 100 - 300 times slower than using L1 cache more efficiently!

So going back to our naive matmul example, if we have assume m=100, n=100, and p=100, then the current way the code is written will have the following memory pattern:

For result[0][0]:
  Access B[0][0] → cache miss (200 cycles)
  Access B[1][0] → cache miss (200 cycles)
  Access B[2][0] → cache miss (200 cycles)
  ... 100 cache misses = 20,000 cycles

For result[0][1]:
  Access B[0][1] → might be in cache (1 cycle)
  Access B[1][1] → cache miss (200 cycles)
  ... ~100 cache misses = 20,000 cycles

Total for one row: ~2,000,000 cycles

That's alot of cycles for a single a cycle row!

We'll go ahead and implement our first optimization which is reordering the loop.

for(int i = 0; i < m; i++) {
    for(int k = 0; k < n; k++) {
        double a_ik = A[i][k];  # Load once, reuse
        for(int j = 0; j < p; j++) {
            result[i][j] += a_ik * B[k][j];  # Sequential access!
        }
    }
}

So what we've done here is now we've optimized the B[k][j] operation. How? Before, when we looped over j first, we looped over the k rows in B, meaning we'd have a cache miss everytime and have to go to RAM. Instead, by accessing K first, we're now holding the row fixed and accessing the column values sequentially since the row values will be stored together. Now, our data pattern looks like the following:

For i=0, k=0:
  Access B[0][0] → cache miss (200 cycles)
  Access B[0][1] → cache hit (1 cycle)
  Access B[0][2] → cache hit (1 cycle)
  ... B[0][7] → cache hit (1 cycle)
  Access B[0][8] → cache miss (200 cycles)
  ... ~12 cache misses for 100 elements = 2,400 cycles

For i=0, k=1:
  Access B[1][0] → cache miss (200 cycles)
  ... ~12 cache misses = 2,400 cycles

Total for one row: ~240,000 cycles

Compared to the naive implementation, we've cut down from ~2 million cycles down to ~240K cycles for a 100x100 matrix. A 10x improvement! Remarkable. You can imagine how much time is saved for huge matrices with millions of operations just by this little change.

Again, now let's remind ourselves of the goal: How can reduce the amount of wasted cache and reduce trips to more expensive memory?
