# -*- coding: utf-8 -*-
"""
03_bootstrap_pka: bootstrap of 9-pKa parameters on the training set.

Migrated from z168_bootstrap.py. Performs B=1000 bootstrap resampling of
the 9-pKa least-squares fit (no regularization, box [0,14]) on the 1743-
protein training set, then evaluates 5 selected pKa sets with three
protocols: (a) raw 9-param test RMSE, (b) + ESM residual SVR OOF/TEST,
(c) IPC2.SVR(F19) with swapped pI_IPC2_protein column OOF/TEST.

Outputs:
  results/bootstrap/z168_bootstrap_pKa.csv      bootstrap pKa samples
  results/bootstrap/z168_bootstrap_rmse.csv    in-bag / OOB RMSE
  results/bootstrap/z168_bootstrap_summary.csv  CI summary per dimension
  results/bootstrap/z168_selected5_pKa.csv      5 selected pKa sets
  results/bootstrap/z168_selected5_eval.csv       3-protocol evaluation
  results/bootstrap/z168_selected5_agg.csv      aggregate stats
  results/figures/fig_z168_bootstrap_pKa.png    3x3 distribution figure
"""
import sys, os, time, math, pathlib
sys.stdout.reconfigure(encoding='utf-8')
# Make the piesm package importable (project root is one level up)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from piesm import pka_engine as C, dataio, pooling, models, metrics

# Output directories
RESULTS = ROOT / "results"
BOOTSTRAP = RESULTS / "bootstrap"
FIGURES = RESULTS / "figures"
os.makedirs(BOOTSTRAP, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

B_BOOT = 1000
SEED = 20240917
BOX0 = np.zeros(9)
BOX1 = np.full(9, 14.0)

t0 = time.time()
d = dataio.load_data()
ytr, yte, sw = d['ytr'], d['yte'], d['sw']
ctr, cte = d['ctr'], d['cte']
ftr, fte = d['ftr'], d['fte']
n = len(ytr)


def fit_free(pka0, counts, exp_pI):
    """Unregularized TRF + analytical Jacobian, box [0,14]."""
    def residuals(pka):
        pi, _ = C.compute_pI_and_jacobian(pka, counts)
        return pi - exp_pI
    def jac(pka):
        _, J = C.compute_pI_and_jacobian(pka, counts)
        return J
    res = least_squares(residuals, pka0, jac=jac, bounds=(BOX0, BOX1), method='trf',
                        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
    return res.x


# ---------- 1) Bootstrap (skip if output exists, replay sample to restore rng state) ----------
rng = np.random.default_rng(SEED)
samples = np.zeros((B_BOOT, 9))
oob_rmse = np.zeros(B_BOOT)
inb_rmse = np.zeros(B_BOOT)
boot_pka_csv = BOOTSTRAP / 'z168_bootstrap_pKa.csv'
boot_rmse_csv = BOOTSTRAP / 'z168_bootstrap_rmse.csv'
if boot_pka_csv.exists() and boot_rmse_csv.exists():
    print('Bootstrap output exists, skip resampling (delete z168_bootstrap_*.csv to force rerun)',
          flush=True)
    samples = pd.read_csv(str(boot_pka_csv))[C.DIMS].values
    _prev = pd.read_csv(str(boot_rmse_csv))
    inb_rmse, oob_rmse = _prev['inb_rmse'].values, _prev['oob_rmse'].values
    for _b in range(B_BOOT):
        rng.integers(0, n, n)          # Replay sample sequence to ensure reproducible picks
else:
    print(f'Bootstrap B={B_BOOT}, train n={n}, box=[0,14], lam=0 ...', flush=True)
    start = C.PKA_IPC2_PAPER.copy()
    for b in range(B_BOOT):
        idx = rng.integers(0, n, n)
        pka = fit_free(start, ctr[idx], ytr[idx])
        samples[b] = pka
        inb_rmse[b] = math.sqrt(float(np.mean((C.compute_pI(pka, ctr[idx]) - ytr[idx]) ** 2)))
        oob_mask = np.ones(n, bool); oob_mask[np.unique(idx)] = False
        if oob_mask.any():
            oob_rmse[b] = math.sqrt(float(np.mean(
                (C.compute_pI(pka, ctr[oob_mask]) - ytr[oob_mask]) ** 2)))
        else:
            oob_rmse[b] = np.nan
        if (b + 1) % 100 == 0:
            print(f'  {b+1}/{B_BOOT}  median OOB RMSE={np.nanmedian(oob_rmse[:b+1]):.4f} '
                  f'[{time.time()-t0:.0f}s]', flush=True)

pd.DataFrame(samples, columns=C.DIMS).to_csv(
    str(BOOTSTRAP / 'z168_bootstrap_pKa.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(dict(boot=np.arange(B_BOOT), inb_rmse=inb_rmse, oob_rmse=oob_rmse)).to_csv(
    str(BOOTSTRAP / 'z168_bootstrap_rmse.csv'), index=False, encoding='utf-8-sig')

# Summary
summ = []
for j, dim in enumerate(C.DIMS):
    col = samples[:, j]
    lo, hi = np.percentile(col, [2.5, 97.5])
    summ.append(dict(dim=dim, mean=col.mean(), std=col.std(ddof=1), ci_lo=lo, ci_hi=hi,
                     pile0=float(np.mean(np.abs(col - 0.0) < 1e-6)),
                     pile14=float(np.mean(np.abs(col - 14.0) < 1e-6)),
                     ipc2=C.PKA_IPC2_PAPER[j], bjellqvist=C.PKA_BJELLQVIST[j]))
summ_df = pd.DataFrame(summ)
summ_df.to_csv(str(BOOTSTRAP / 'z168_bootstrap_summary.csv'),
               index=False, encoding='utf-8-sig')
print('\n', summ_df.round(3).to_string(index=False), flush=True)

# ---------- Plot: 3x3 distributions ----------
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for j, dim in enumerate(C.DIMS):
    ax = axes[j // 3][j % 3]
    col = samples[:, j]
    ax.hist(np.clip(col, 0.02, 13.98), bins=50, color='steelblue', alpha=0.75,
            edgecolor='white')
    ax.axvline(col.mean(), color='red', lw=2, label=f'mean={col.mean():.2f}')
    ax.axvline(np.percentile(col, 2.5), color='red', ls='--', lw=1.3)
    ax.axvline(np.percentile(col, 97.5), color='red', ls='--', lw=1.3,
               label='95% CI')
    ax.axvline(C.PKA_BJELLQVIST[j], color='blue', ls=':', lw=1.8, label='Bjellqvist')
    ax.axvline(C.PKA_IPC2_PAPER[j], color='green', ls=':', lw=1.8, label='IPC2')
    ax.set_title(f'{dim}  std={col.std(ddof=1):.3f}  '
                 f'pile@0={summ_df.pile0[j]:.2f} pile@14={summ_df.pile14[j]:.2f}',
                 fontsize=10)
    if j == 0: ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(str(FIGURES / 'fig_z168_bootstrap_pKa.png'), dpi=160)
plt.close()

# ---------- 2) Select 5 groups ----------
valid = ~np.isnan(oob_rmse)
order = np.argsort(oob_rmse[valid])
top50 = np.where(valid)[0][order[:50]]
pick = rng.choice(top50, 5, replace=False)
pick = sorted(pick.tolist())
print('\n5 pKa groups (bootstrap indices):', pick, flush=True)
print('Their OOB RMSE:', np.round(oob_rmse[pick], 4), flush=True)
pd.DataFrame(samples[pick], columns=C.DIMS).to_csv(
    str(BOOTSTRAP / 'z168_selected5_pKa.csv'), index=False, encoding='utf-8-sig')

# ---------- 3) Three-protocol evaluation ----------
Xtr, Xte = pooling.load_emb_pool('ESM2-150M', pooling.W_ELEGANT)
Ftr0, Fte0 = ftr[models.F19].values.astype(np.float64), fte[models.F19].values.astype(np.float64)
COL_IPC = models.F19.index('pI_IPC2_protein')

# Reference: IPC2 vector
ref_raw = metrics.metrics(yte, fte['pI_IPC2_protein'].values)
ref_cvs, _ = models.cv5(models.make_esm_svr, Xtr, ytr - ftr['pI_IPC2_protein'].values,
                   ftr['pI_IPC2_protein'].values, ytr, sw)
ref_pt, _ = models.full_test(models.make_esm_svr, Xtr, ytr - ftr['pI_IPC2_protein'].values,
                        sw, Xte, fte['pI_IPC2_protein'].values)
ref_esm = metrics.metrics(yte, ref_pt)
# F19 reference (i.e. standard IPC2.SVR)
ref_f19_oof_l, ref_f19_oof = [], None
for sd in dataio.SEEDS:
    kf_ = KFold(5, shuffle=True, random_state=sd); o = np.zeros(n)
    for a, c_ in kf_.split(Ftr0):
        e = models.make_f19_svr(); e.fit(Ftr0[a], ytr[a]); o[c_] = e.predict(Ftr0[c_])
    ref_f19_oof_l.append(float(np.sqrt(np.mean((o-ytr)**2))))
    if sd == 71923: ref_f19_oof = o.copy()
ref_f19 = metrics.metrics(yte, models.make_f19_svr().fit(Ftr0, ytr).predict(Fte0))

rows = []
print(f'\n{"combo":>8s} {"raw_tr":>7s} {"raw_te":>7s} | {"ESM_OOF":>8s} {"ESM_TEST":>8s} '
      f'| {"F19_OOF":>8s} {"F19_TEST":>8s}', flush=True)
for k, bi in enumerate(pick):
    pka = samples[bi]
    b0 = C.compute_pI(pka, ctr); b1 = C.compute_pI(pka, cte)
    raw_tr = math.sqrt(float(np.mean((b0-ytr)**2)))
    raw_te = math.sqrt(float(np.mean((b1-yte)**2)))
    # (b) ESM residual
    cvs, _ = models.cv5(models.make_esm_svr, Xtr, ytr-b0, b0, ytr, sw)
    pt, _ = models.full_test(models.make_esm_svr, Xtr, ytr-b0, sw, Xte, b1)
    m_esm = metrics.metrics(yte, pt)
    # (c) Retrain F19 with pI_IPC2_protein column swapped
    Gtr = Ftr0.copy(); Gte = Fte0.copy()
    Gtr[:, COL_IPC] = b0; Gte[:, COL_IPC] = b1
    cvsF_l = []
    for sd in dataio.SEEDS:
        kf_ = KFold(5, shuffle=True, random_state=sd); o = np.zeros(n)
        for a, c_ in kf_.split(Gtr):
            e = models.make_f19_svr(); e.fit(Gtr[a], ytr[a]); o[c_] = e.predict(Gtr[c_])
        cvsF_l.append(float(np.sqrt(np.mean((o-ytr)**2))))
    eG = models.make_f19_svr(); eG.fit(Gtr, ytr)
    m_f19 = metrics.metrics(yte, eG.predict(Gte))
    rows.append(dict(combo=f'B{k+1}', boot_idx=bi, raw_trRMSE=raw_tr, raw_teRMSE=raw_te,
                     ESM_OOF=np.mean(cvs), ESM_OOF_std=np.std(cvs), ESM_TEST=m_esm['RMSE'],
                     F19_OOF=np.mean(cvsF_l), F19_TEST=m_f19['RMSE'],
                     oob_rmse=oob_rmse[bi]))
    print(f'B{k+1:>7d} {raw_tr:7.4f} {raw_te:7.4f} | {np.mean(cvs):8.4f} {m_esm["RMSE"]:8.4f} '
          f'| {np.mean(cvsF_l):8.4f} {m_f19["RMSE"]:8.4f}', flush=True)

# Reference row
rows.append(dict(combo='IPC2_ref', boot_idx=-1,
                 raw_trRMSE=math.sqrt(float(np.mean((ftr["pI_IPC2_protein"].values-ytr)**2))),
                 raw_teRMSE=ref_raw['RMSE'], ESM_OOF=np.mean(ref_cvs),
                 ESM_OOF_std=np.std(ref_cvs), ESM_TEST=ref_esm['RMSE'],
                 F19_OOF=np.mean(ref_f19_oof_l), F19_TEST=ref_f19['RMSE'], oob_rmse=np.nan))
df = pd.DataFrame(rows)
# 5-group mean ± std
agg = (df[df.combo.str.startswith('B')]
       .drop(columns=['combo', 'boot_idx', 'oob_rmse'])
       .agg(['mean', 'std']))
df.to_csv(str(BOOTSTRAP / 'z168_selected5_eval.csv'), index=False, encoding='utf-8-sig')
print('\nReference IPC2:', df[df.combo=='IPC2_ref'].to_string(index=False), flush=True)
print('\n5-group aggregate:\n', agg.round(4).to_string(), flush=True)
agg.round(4).to_csv(str(BOOTSTRAP / 'z168_selected5_agg.csv'), encoding='utf-8-sig')
print(f'\nTotal time {time.time()-t0:.0f}s', flush=True)
