
# Estimating the Condition Number of an SPD Matrix

This exercise evaluates your numerical methods skills and HPC preparedness by requiring you to estimate the spectral condition number \(\kappa(A) = \lambda_{\max}/\lambda_{\min}\) of a generated SPD matrix.

## 1. Problem Description
You must use numerical methods to estimate the extremal eigenvalues of an SPD matrix constructed via:
\[
A = Q^T D Q,
\]
where \(Q\) is orthogonal and \(D\) has eigenvalues geometrically spaced between 1 and \(\kappa\).
The right-hand side vector is defined by:
\[
b = A \cdot \mathbf{1}.
\]

## 2. Tasks
### 2.1 Estimate Eigenvalues
- Implement **power iteration** to approximate \(\lambda_{\max}\).
- Implement **inverse power iteration**, or a CG-based variant, to estimate \(\lambda_{\min}\).
- Report:
  - Estimated \(\lambda_{\max}\)
  - Estimated \(\lambda_{\min}\)
  - Estimated condition number \(\hat{\kappa}\)

### 2.2 Validation
- Compute **Rayleigh quotient** samples.
- Provide **Gershgorin bounds**.
- For small matrices (e.g., n = 64), you may compute full eigenvalues using a library routine for validation.

### 2.3 Numerical Stability Documentation
Explain:
- Convergence criteria
- Normalization strategy
- Handling loss of orthogonality
- Reproducibility practices (fixed seeds)

### 2.4 Performance Notes
- Benchmark performance on the n=256 dataset.
- Comment on floating‑point intensity, memory access, and BLAS‑2/3 usage.

## 3. How to Generate Data
Use the provided generator:
```
python3 data/generate_matrix.py --n 64 --kappa 1e3 --outA A.csv --outb b.csv
```
Generate the larger dataset using:
```
python3 data/generate_matrix.py --n 256 --kappa 1e4 --outA A2.csv --outb b2.csv
```

## 4. Deliverables
- Source code
- A report describing:
  - Numerical results and tables
  - \(\lambda_{\max}\), \(\lambda_{\min}\), \(\hat{\kappa}\)
  - Stability discussion
  - Performance measurements

## 5. Additional Requirements
- Include summary tables of results.
- Include plots if possible (e.g., convergence curves).
- Document hardware/software environment, compiler flags, and libraries used.
