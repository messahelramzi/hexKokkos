import numpy as np
from scipy.linalg import eigvalsh
# Load matrix A from CSV
A = np.loadtxt("data/A.csv", delimiter=",")

eigs = eigvalsh(A)

lam_min = eigs[0]
lam_max = eigs[-1]

kappa = lam_max / lam_min

print(f"Estimated lam_min: {lam_min:.2e}")
print(f"Estimated lam_max: {lam_max:.2e}")
print(f"Estimated condition number: {kappa:.2e}")