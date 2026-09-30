# -*- coding: utf-8 -*-
"""
07_permutation_pka: permutation robustness of the optimized 9-pKa vector (GBMS).

SI Table S15 / Figure S4. In each of 1000 iterations, 581 randomly chosen
proteins are removed from the 1743-protein training set and replaced by all
581 test proteins; the nine pKa values are then re-optimized with the exact
GBMS protocol (13-start trust-region least squares, box [0,14], no
regularization), and the lowest-train-RMSE solution is retained.

Optimizer / solver details are identical to the main-text GBMS fit:
  - 3 published starting scales: Thurlkill 2006, IPC1 2016, IPC2 2021
  - 10 Latin hypercube samples in [0,14]^9
  - scipy least_squares (trf), analytic Jacobian via the implicit-function
    theorem, xtol=ftol=gtol=1e-8
  - bisection-protected Newton pI solver (charge balance)

Outputs:
  results/bootstrap/perm_pka_samples.csv   per-iteration pKa + train RMSE
  results/bootstrap/perm_pka_summary.csv   mean/median/std/95% CI per group

A frozen reference copy (1000 iterations) ships in
data/reference_results/; this script regenerates it (~70 min, 4 workers).
Set NITER=10 NPROC=1 for a quick smoke test.
"""
import sys, os, time, pathlib
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from esmpi import dataio

# ---------------- constants ----------------
N_ITER = 1000
SEED = 20260927
BOX0 = np.zeros(9)
BOX1 = np.full(9, 14.0)
NAMES = ['N_TER', 'C_TER', 'asp', 'glu', 'his', 'lys', 'cys', 'tyr', 'arg']
THURL9 = np.array([8.00, 3.67, 3.67, 4.25, 6.54, 10.40, 8.55, 9.84, 12.00])
IPC1 = np.array([9.094, 2.869, 3.872, 4.412, 5.637, 9.052, 7.555, 10.85, 11.84])
from esmpi.pka_engine import PKA_IPC2_PAPER as IPC2

RESULTS = ROOT / "results"
BOOT = RESULTS / "bootstrap"
os.makedirs(BOOT, exist_ok=True)

# ---------------- pI solver (Newton with bisection guard, GBMS delivery protocol) ----
LN10 = np.log(10.0)
# acidic groups (deprotonation -> negative): C_TER, D, E, C, Y
ACID_IDX = [1, 2, 3, 6, 7]
BASE_IDX = [0, 4, 5, 8]


def _charge_slope(pH, p, ctr):
    s = 1.0 / (1.0 + 10.0 ** (p[None, :] - pH[:, None]))
    q = -(s[:, ACID_IDX] * ctr[:, ACID_IDX]).sum(1)
    q += ((1.0 - s[:, BASE_IDX]) * ctr[:, BASE_IDX]).sum(1)
    slope = -LN10 * (ctr * s * (1.0 - s)).sum(1)
    return q, slope


def pi9_fast(p, ctr, iters=20):
    n = ctr.shape[0]
    lo, hi = np.zeros(n), np.full(n, 14.0)
    pH = np.full(n, 7.0)
    frozen = np.zeros(n, bool)
    eps = 1e-9
    for _ in range(iters):
        q, sl = _charge_slope(pH, p, ctr)
        go_up = q > 0
        lo = np.where(go_up, pH, lo)
        hi = np.where(go_up, hi, pH)
        move = (~frozen) & (np.abs(q) > 1e-12)
        step = np.where(np.abs(sl) > 1e-14,
                        q / np.where(np.abs(sl) > 1e-14, sl, -1.0), 0.0)
        newton = pH - step
        width = np.maximum(hi - lo, 1e-12)
        interior = (newton > lo + eps) & (newton < hi - eps) & \
                   (np.abs(newton - pH) < 0.45 * width)
        cand = np.where(interior, newton, (lo + hi) / 2.0)
        pH = np.where(move, cand, pH)
        frozen |= np.abs(q) <= 1e-12
    return np.where(frozen, pH, (lo + hi) / 2.0)


def _residuals(p, ctr, y):
    return pi9_fast(p, ctr) - y


def _jacobian(p, ctr, y):
    pi = pi9_fast(p, ctr)
    s = 1.0 / (1.0 + 10.0 ** (p[None, :] - pi[:, None]))
    w = ctr * s * (1.0 - s)
    return w / np.clip(w.sum(1, keepdims=True), 1e-12, None)


def _fit(p0, ctr, y):
    return least_squares(_residuals, p0, jac=_jacobian, args=(ctr, y),
                         bounds=(BOX0, BOX1), method='trf',
                         xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=200).x


def optimize_pka(ctr, y, seed):
    """13-start GBMS fit; return best vector and its train RMSE."""
    rng = np.random.default_rng(seed)
    lhs = rng.uniform(BOX0, BOX1, size=(10, 9))
    best_x, best_r = None, np.inf
    for s0 in [THURL9, IPC1, IPC2] + [lhs[i] for i in range(10)]:
        x = _fit(s0.copy(), ctr, y)
        r = float(np.sqrt(np.mean((pi9_fast(x, ctr) - y) ** 2)))
        if r < best_r:
            best_x, best_r = x, r
    return best_x, best_r


# ---------------- multiprocessing workers ----------------
_G = {}


def _init(ctr_all, y_all, n_tr, n_te):
    _G['ctr'] = ctr_all
    _G['y'] = y_all
    _G['n_tr'] = n_tr
    _G['n_te'] = n_te


def _run_one(args):
    i, seed = args
    ctr_all, y_all = _G['ctr'], _G['y']
    n_tr, n_te = _G['n_tr'], _G['n_te']
    rng = np.random.default_rng(seed)
    drop = rng.choice(n_tr, size=n_te, replace=False)
    keep = np.ones(n_tr, dtype=bool)
    keep[drop] = False
    ctr_new = np.vstack([ctr_all[:n_tr][keep], ctr_all[n_tr:]])
    y_new = np.concatenate([y_all[:n_tr][keep], y_all[n_tr:]])
    x, r = optimize_pka(ctr_new, y_new, seed)
    return i, x, r


def _run_one_mp(args):
    # Windows spawn: re-unpack globals passed via initializer
    return _run_one(args)


def _dump(results, out_csv):
    rows = []
    for i, item in enumerate(results):
        if item is None:
            continue
        x, r = item
        row = {'iter': i, 'rmse': r}
        for j, nm in enumerate(NAMES):
            row[nm] = x[j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8-sig')


def _summary(samples_csv):
    df = pd.read_csv(samples_csv)
    rows = []
    for nm in NAMES:
        col = df[nm].values
        rows.append(dict(dim=nm, mean=col.mean(), std=col.std(),
                         ci_lo=np.percentile(col, 2.5),
                         ci_hi=np.percentile(col, 97.5),
                         median=np.median(col)))
    pd.DataFrame(rows).to_csv(BOOT / 'perm_pka_summary.csv',
                              index=False, encoding='utf-8-sig')
    return pd.DataFrame(rows)


def main(n_iter=N_ITER, n_proc=4):
    print('Loading train/test sequences ...', flush=True)
    d = dataio.load_data()
    n_tr, n_te = len(d['ytr']), len(d['yte'])
    ctr_all = np.vstack([d['ctr'], d['cte']])
    y_all = np.concatenate([d['ytr'], d['yte']])
    print(f'train={n_tr}, test={n_te}, iterations={n_iter}', flush=True)

    master = np.random.default_rng(SEED)
    seeds = master.integers(0, 2 ** 31 - 1, n_iter)
    out_csv = BOOT / 'perm_pka_samples.csv'
    results = [None] * n_iter
    t0 = time.time()

    if n_proc <= 1:
        _init(ctr_all, y_all, n_tr, n_te)
        for i in range(n_iter):
            results[i] = _run_one((i, int(seeds[i])))[1:]
            if (i + 1) % 10 == 0 or i < 5:
                print(f'[{i+1}/{n_iter}] rmse={results[i][1]:.4f} '
                      f'elapsed={time.time()-t0:.0f}s', flush=True)
            if (i + 1) % 10 == 0 or i + 1 == n_iter:
                _dump(results, out_csv)
    else:
        from multiprocessing import Pool
        args_iter = [(i, int(seeds[i])) for i in range(n_iter)]
        with Pool(n_proc, initializer=_init,
                  initargs=(ctr_all, y_all, n_tr, n_te)) as pool:
            done = 0
            for idx, x, r in pool.imap_unordered(_run_one_mp, args_iter):
                results[idx] = (x, r)
                done += 1
                if done % 10 == 0 or done <= 5:
                    print(f'[{done}/{n_iter}] iter={idx} rmse={r:.4f} '
                          f'elapsed={time.time()-t0:.0f}s', flush=True)
                if done % 10 == 0 or done == n_iter:
                    _dump(results, out_csv)

    _dump(results, out_csv)
    summ = _summary(out_csv)
    print(summ.round(3).to_string(index=False))
    print(f'DONE total={time.time()-t0:.0f}s -> {out_csv}', flush=True)


if __name__ == '__main__':
    main(n_iter=int(os.environ.get('NITER', str(N_ITER))),
         n_proc=int(os.environ.get('NPROC', '4')))
