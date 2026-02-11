## *hexKokkos* is a collection of examples leveraging Kokkos for portability across CPU and GPU architectures.

Test 2 CPU and GPU implementations were fused thanks to Kokkos portability.

Same for Test 3 fused with test 4 that covers serial implementation with CPU and GPU implementations.

The codes should run on theory on any architecture supported by Kokkos (also Intel/AMD GPUs), but they were tested on CPU with OpenMP and GPU with CUDA.

**Completed tests**

- [x] Test 1: Condition number of a matrix (CPU OpenMP + GPU CUDA) using power iteration method.
- [x] Test 2: Dense MatVec (CPU OpenMP + GPU CUDA)
- [x] Test 3 & 4: PCG with Jacobi preconditioner (CPU OpenMP + GPU CUDA)

The following Kokkos configuration was considered for the tests:

```console
  Kokkos Version: 4.7.1
Compiler:
  KOKKOS_COMPILER_GNU: 1430
  KOKKOS_COMPILER_NVCC: 1300
Architecture:
  CPU architecture: none
  Default Device: Cuda
  GPU architecture: AMPERE86
  platform: 64bit
Atomics:
Vectorization:
  KOKKOS_ENABLE_PRAGMA_IVDEP: no
  KOKKOS_ENABLE_PRAGMA_LOOPCOUNT: no
  KOKKOS_ENABLE_PRAGMA_UNROLL: no
  KOKKOS_ENABLE_PRAGMA_VECTOR: no
Memory:
Options:
  KOKKOS_ENABLE_ASM: yes
  KOKKOS_ENABLE_CXX17: no
  KOKKOS_ENABLE_CXX20: yes
  KOKKOS_ENABLE_CXX23: no
  KOKKOS_ENABLE_CXX26: no
  KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK: no
  KOKKOS_ENABLE_HWLOC: no
  KOKKOS_ENABLE_LIBDL: yes
Host Parallel Execution Space:
  KOKKOS_ENABLE_OPENMP: yes

OpenMP Runtime Configuration:
Kokkos::OpenMP thread_pool_topology[ 1 x 8 x 1 ]
Device Execution Space:
  KOKKOS_ENABLE_CUDA: yes
Cuda Options:
  KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE: no
  KOKKOS_ENABLE_CUDA_UVM: no
  KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC: no

Cuda Runtime Configuration:
macro  KOKKOS_ENABLE_CUDA      : defined
macro  CUDA_VERSION          = 13000 = version 13.0
Kokkos::Cuda[ 0 ] NVIDIA GeForce RTX 3080 Laptop GPU : Selected
  Capability: 8.6
  Total Global Memory: 7.658 GiB
  Shared Memory per Block: 48 KiB
  Can access system allocated memory: 0
    via Address Translation Service: 0
```
