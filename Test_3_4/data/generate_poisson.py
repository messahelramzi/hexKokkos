
#!/usr/bin/env python3
import numpy as np, argparse, os
p=argparse.ArgumentParser()
p.add_argument('--nx', type=int, default=64)
p.add_argument('--ny', type=int, default=64)
p.add_argument('--outdir', default='poisson_64x64')
args=p.parse_args()
rowptr=[0]; cols=[]; vals=[]
idx=lambda i,j: i*args.ny + j
for i in range(args.nx):
    for j in range(args.ny):
        r=idx(i,j)
        cols.append(r); vals.append(4.0)
        if i>0:   cols.append(idx(i-1,j)); vals.append(-1.0)
        if i<args.nx-1: cols.append(idx(i+1,j)); vals.append(-1.0)
        if j>0:   cols.append(idx(i,j-1)); vals.append(-1.0)
        if j<args.ny-1: cols.append(idx(i,j+1)); vals.append(-1.0)
        rowptr.append(len(cols))
os.makedirs(args.outdir, exist_ok=True)
np.savetxt(f"{args.outdir}/rowptr.csv", np.array(rowptr,dtype=int), fmt='%d', delimiter=',')
np.savetxt(f"{args.outdir}/col.csv", np.array(cols,dtype=int), fmt='%d', delimiter=',')
np.savetxt(f"{args.outdir}/val.csv", np.array(vals,dtype=float), delimiter=',')
print('Wrote', args.outdir)
