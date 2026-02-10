
# Test 3 (PCG Solver)

This test evaluates your ability to implement a Preconditioned Conjugate Gradient (PCG) solver
for a sparse SPD matrix arising from a 2‑D Poisson operator.

## Generate Input Data
Run this command from the **root** of the package:
```
python3 data/generate_poisson.py --nx 64 --ny 64 --outdir poisson_64x64
```
This produces `rowptr.csv`, `col.csv`, and `val.csv` inside `poisson_64x64/`.

## How the Matrix is Structured (5‑Point Poisson)
The 2‑D grid uses a **5‑point finite-difference stencil**:
```
      (i-1,j)
         |
(i,j-1) — (i,j) — (i,j+1)
         |
      (i+1,j)
```
Matrix coefficients:
- Center: **4**
- Nearest neighbours: **−1**
- Missing neighbours at boundaries → homogeneous Dirichlet (treated as zero)

## CSR Format Explanation
The matrix is stored using Compressed Sparse Row (CSR):
- `rowptr.csv`: row pointer (length N+1)
- `col.csv`: column indices for nonzeros
- `val.csv`: values of nonzeros

Vector `b` = ones(N), initial guess `x0` = zeros(N).
