
# Test 3 — PCG for 2‑D Poisson (Jacobi Preconditioner)

This assignment evaluates your understanding of iterative methods,
preconditioning, sparse storage formats, and HPC performance concepts.
You will implement a **Preconditioned Conjugate Gradient (PCG)** solver for a system:
\[ A x = b \]
where `A` is an SPD matrix derived from a 5‑point finite‑difference Laplacian.

## 1. Problem Description
The matrix represents a 2‑D Poisson operator on an `nx × ny` grid.
Each interior point contributes:
- Central coefficient: **4**
- Neighbour coefficients (left/right/up/down): **−1**
Boundary points have fewer neighbours (Dirichlet = 0).

`A` is stored in CSR format using `rowptr.csv`, `col.csv`, and `val.csv`.
`b` is the all‑ones vector.

## 2. Tasks
### 2.1 Implement PCG
- Use Jacobi preconditioning: \(M^{-1} = 1/	ext{diag}(A)\)
- Stopping condition: \( ||r_k|| / ||b|| < 10^{-8} \)

### 2.2 Implement Required Sparse Operations
- CSR sparse matrix-vector multiply (SpMV)
- Dot products
- Preconditioner application
- Vector updates (AXPY-style operations)

### 2.3 Convergence and Numerical Behaviour
Report:
- Iteration count
- Residual norms
- Optional: plot residual vs. iteration
- Effect of grid resolution (compare 64×64 to 128×128 if generated)

### 2.4 Performance Evaluation
- Discuss memory access patterns
- Explain why Poisson SpMV is bandwidth‑bound

## 3. How to Generate Data
Run:
```
python3 data/generate_poisson.py --nx 64 --ny 64 --outdir poisson_64x64
```
This generates CSR files you must load.

## 4. Deliverables
- Source code for your PCG solver
- A report including:
  - Iteration counts
  - Residual history
  - Performance discussion
  - Grid‑size comparison

## 5. Additional Requirements
- Provide tables summarizing convergence
- Document hardware and libraries used
