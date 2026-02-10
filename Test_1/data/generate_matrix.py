
#!/usr/bin/env python3
import numpy as np, argparse
p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=64)
p.add_argument('--kappa', type=float, default=1e3)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--outA', type=str, default='A.csv')
p.add_argument('--outb', type=str, default='b.csv')
args = p.parse_args()
rng = np.random.default_rng(args.seed)
Q,_ = np.linalg.qr(rng.standard_normal((args.n, args.n)))
D = np.diag(np.geomspace(1.0, float(args.kappa), num=args.n))
A = Q.T @ D @ Q
x_true = np.ones(args.n)
b = A @ x_true
np.savetxt(args.outA, A, delimiter=',')
np.savetxt(args.outb, b, delimiter=',')
print(f'Wrote {args.outA}, {args.outb}')
