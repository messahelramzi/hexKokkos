## *hexKokkos* is a collection of examples leveraging Kokkos for portability across CPU and GPU architectures.

Test 2 CPU and GPU implementations were fused thanks to Kokkos portability.

Same for Test 3 fused with test 4 that covers serial implementation with CPU and GPU implementations.

**Completed tests**

- [ ] Test 1: Condition number of a matrix (CPU OpenMP + GPU CUDA) using power iteration method.
- [x] Test 2: Dense MatVec (CPU OpenMP + GPU CUDA)
- [x] Test 3 & 4: PCG with Jacobi preconditioner (CPU OpenMP + GPU CUDA)
