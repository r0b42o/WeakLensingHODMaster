"""MCMC ueber alle Lens-Source-Kombinationen (Kosmologie + HOD je Lens-Bin).

Aufruf (Env MA):
    python run_mcmc.py --out chain_A.npz --nwalkers 96 --nsteps 3000 --nproc 16
Fortsetzen: gleicher Befehl, vorhandene --out wird weitergefuehrt.
"""
import argparse, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2l_nn as g
import emcee

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="chain.npz")
ap.add_argument("--nwalkers", type=int, default=96)
ap.add_argument("--nsteps", type=int, default=3000)
ap.add_argument("--nproc", type=int, default=16)
ap.add_argument("--checkpoint", type=int, default=50)
ap.add_argument("--theta-min", type=float, default=None)
ap.add_argument("--theta-max", type=float, default=None)
ap.add_argument("--no-opt", action="store_true")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

data, theory, space, post = g.build(args.theta_min, args.theta_max)


def lnprob(x):
    return post(x)

nd = space.ndim
rng = np.random.default_rng(args.seed)
print(f"{len(data.combos)} Kombinationen, {data.n} Datenpunkte, {nd} Parameter", flush=True)


def neg(x):
    lp = post(x)
    return 1e30 if not np.isfinite(lp) else -lp


def init_walkers():
    x0 = space.fiducial()
    if not args.no_opt:
        t = time.time()
        res = minimize(neg, x0, method="Powell", options=dict(maxiter=20000, xtol=1e-3, ftol=1e-6))
        x0 = res.x
        print(f"Optimierung: chi2={2*res.fun:.1f}, {time.time()-t:.0f}s", flush=True)
    scale = 1e-3 * (space.hi - space.lo)
    p0 = x0 + scale * rng.standard_normal((args.nwalkers, nd))
    return np.clip(p0, space.lo + 1e-9, space.hi - 1e-9)


def main():
    done = 0
    chain = np.zeros((0, args.nwalkers, nd))
    lnp = np.zeros((0, args.nwalkers))
    if os.path.exists(args.out):
        old = np.load(args.out, allow_pickle=True)
        chain, lnp = old["chain"], old["lnprob"]
        done = chain.shape[0]
        p0 = chain[-1]
        print(f"Fortsetzen ab Schritt {done}", flush=True)
    else:
        p0 = init_walkers()

    with Pool(args.nproc) as pool:
        sampler = emcee.EnsembleSampler(args.nwalkers, nd, lnprob, pool=pool,
                                        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
        state = p0
        t0 = time.time()
        while done < args.nsteps:
            n = min(args.checkpoint, args.nsteps - done)
            state = sampler.run_mcmc(state, n, progress=False)
            chain = np.concatenate([chain, sampler.get_chain()])
            lnp = np.concatenate([lnp, sampler.get_log_prob()])
            sampler.reset()
            done += n
            np.savez(args.out, chain=chain, lnprob=lnp, names=np.array(space.names),
                     labels=np.array(space.labels), lo=space.lo, hi=space.hi,
                     combos=np.array([f"{a}|{b}" for a, b in data.combos]))
            best = lnp.max()
            print(f"Schritt {done}/{args.nsteps}  max lnP={best:.1f}  "
                  f"{(time.time()-t0)/(n and 1):.0f}s seit Start", flush=True)


if __name__ == "__main__":
    main()
