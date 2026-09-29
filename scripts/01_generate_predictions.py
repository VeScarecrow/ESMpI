# -*- coding: utf-8 -*-
"""
01_generate_predictions: pI-ESM predictions and comparison analysis.

Migrated from z166_comparisons.py. Produces the core prediction file
z166_preds.npz (pI-ESM + IPC2-SVR predictions on the 581-protein test set)
plus pairwise comparison figures, pI-distribution tables, and physical
baseline / encoder swap ablations.

Outputs:
  data/predictions/z166_preds.npz    pI-ESM + F19 predictions
  results/tables/z166_*.csv          comparison tables
  results/figures/fig_z166_*.png     comparison figures

NOTE: The ESM3-1.4B comparison section from the original z166 is omitted
because the ESM3 embeddings (1.4 GB) are not published. Only ESM2-150M
(the published encoder) is retained in the encoder swap.
"""
import sys, os, time, pathlib
sys.stdout.reconfigure(encoding='utf-8')
# Make the piesm package importable (project root is one level up)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from piesm import dataio, pooling, models, metrics

# Output directories
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PREDICTIONS = ROOT / "data" / "predictions"
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(PREDICTIONS, exist_ok=True)

t0 = time.time()
d = dataio.load_data()
ytr, yte, sw = d['ytr'], d['yte'], d['sw']
ftr, fte = d['ftr'], d['fte']
bo = ftr['pI_IPC2_protein'].values.astype(np.float64)
bt = fte['pI_IPC2_protein'].values.astype(np.float64)
rtr = ytr - bo

# ---------- Main model predictions ----------
Xtr, Xte = pooling.load_emb_pool('ESM2-150M', pooling.W_ELEGANT)
cvs_ours, oof_ours = models.cv5(models.make_esm_svr, Xtr, rtr, bo, ytr, sw)
pt_ours, _ = models.full_test(models.make_esm_svr, Xtr, rtr, sw, Xte, bt)
Ftr, Fte = ftr[models.F19].values.astype(np.float64), fte[models.F19].values.astype(np.float64)
eF = models.make_f19_svr(); eF.fit(Ftr, ytr)   # official F19, no sample weights
pt_f19 = eF.predict(Fte)
np.savez(str(PREDICTIONS / 'z166_preds.npz'),
         pt_ours=pt_ours, pt_f19=pt_f19, bt_ipc2=bt, yte=yte,
         oof_ours=bo + oof_ours, ytr=ytr, bo=bo)

# ---------- Figure 1: 4x4 pairplot ----------
labels = ['Experimental pI', 'IPC2 9-pKa', 'IPC2.SVR (F19)', 'ESM-SVR (ours)']
keys = ['exp', 'ipc', 'f19', 'ours']
P = np.column_stack([yte, bt, pt_f19, pt_ours])
fig, axes = plt.subplots(4, 4, figsize=(13, 12))
for i in range(4):
    for j in range(4):
        ax = axes[i][j]
        if i == j:
            ax.hist(P[:, i], bins=30, color='#4A90D9', alpha=0.75, edgecolor='white')
            ax.set_title(labels[i], fontsize=10)
        elif i > j:
            ax.scatter(P[:, j], P[:, i], s=5, alpha=0.35, color='#34495E')
            lim = [min(P[:, j].min(), P[:, i].min()) - 0.3,
                   max(P[:, j].max(), P[:, i].max()) + 0.3]
            ax.plot(lim, lim, 'r--', lw=1)
            ax.set_xlim(lim); ax.set_ylim(lim)
            r = pearsonr(P[:, j], P[:, i])[0]
            rm = np.sqrt(np.mean((P[:, i] - P[:, j]) ** 2))
            ax.text(0.05, 0.92, f'r={r:.3f}\nRMSE={rm:.3f}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7, ec='none'))
        else:
            r = pearsonr(P[:, j], P[:, i])[0]
            rm = np.sqrt(np.mean((P[:, i] - P[:, j]) ** 2))
            ax.axis('off')
            ax.text(0.5, 0.55, f'r = {r:.4f}\nRMSE = {rm:.4f}',
                    ha='center', va='center', fontsize=13)
        if i == 3 and i > j: ax.set_xlabel(labels[j], fontsize=9)
        if j == 0 and i > j: ax.set_ylabel(labels[i], fontsize=9)
plt.suptitle('Pairwise comparison on independent test set (n=581)', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(str(FIGURES / 'fig_z166_pairwise.png'), dpi=180)
plt.close()
print('pairwise figure saved', flush=True)

# ---------- Figure 2: train/test pI distribution ----------
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(1, 12.5, 46)
ax.hist(ytr, bins=bins, alpha=0.55, label=f'Train (n={len(ytr)})', color='#4A90D9', edgecolor='white')
ax.hist(yte, bins=bins, alpha=0.55, label=f'Test (n={len(yte)})', color='#E67E22', edgecolor='white')
ax.axvline(ytr.mean(), color='#4A90D9', ls='--', lw=1.5)
ax.axvline(yte.mean(), color='#E67E22', ls='--', lw=1.5)
ax.set_xlabel('Experimental pI'); ax.set_ylabel('Count'); ax.legend()
ax.set_title(f'Distribution of experimental pI  '
             f'(train mean {ytr.mean():.2f}±{ytr.std():.2f}, test mean {yte.mean():.2f}±{yte.std():.2f})')
plt.tight_layout()
plt.savefig(str(FIGURES / 'fig_z166_pI_distribution.png'), dpi=180)
plt.close()
dist_rows = []
for nm, y in [('train', ytr), ('test', yte)]:
    dist_rows.append(dict(set=nm, n=len(y), mean=y.mean(), std=y.std(),
                          median=np.median(y), p25=np.percentile(y,25),
                          p75=np.percentile(y,75), min=y.min(), max=y.max(),
                          n_lt6=(y<6).sum(), n_6_8=((y>=6)&(y<8)).sum(),
                          n_8_9=((y>=8)&(y<9)).sum(), n_gt9=(y>9).sum()))
pd.DataFrame(dist_rows).to_csv(str(TABLES / 'z166_pI_distribution_stats.csv'),
                               index=False, encoding='utf-8-sig')
print('distribution figure saved', flush=True)

# ---------- 3) Physical baseline swap (20 pKa sets) ----------
print('\nSwap physical baseline:', flush=True)
pka_rows = []
for col in [c for c in ftr.columns if c.startswith('pI_')]:
    b0 = ftr[col].values.astype(np.float64)
    b1 = fte[col].values.astype(np.float64)
    r0 = ytr - b0
    cvs, _ = models.cv5(models.make_esm_svr, Xtr, r0, b0, ytr, sw)
    pt, _ = models.full_test(models.make_esm_svr, Xtr, r0, sw, Xte, b1)
    m = metrics.metrics(yte, pt)
    raw = metrics.metrics(yte, b1)
    pka_rows.append(dict(base=col, raw_TEST=raw['RMSE'],
                         ESM_OOF=np.mean(cvs), ESM_TEST=m['RMSE'],
                         gain=raw['RMSE']-m['RMSE']))
    print(f'  {col:20s} raw={raw["RMSE"]:.4f}  OOF={np.mean(cvs):.4f}  '
          f"TEST={m['RMSE']:.4f} gain={raw['RMSE']-m['RMSE']:+.4f}", flush=True)
pka_df = pd.DataFrame(pka_rows).sort_values('ESM_TEST')
pka_df.to_csv(str(TABLES / 'z166_swap_pKa_baseline.csv'), index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(10, 7))
pp = pka_df.sort_values('raw_TEST')
y = np.arange(len(pp))
ax.barh(y-0.2, pp['raw_TEST'], 0.4, label='Raw 9-pKa', color='#95A5A6')
ax.barh(y+0.2, pp['ESM_TEST'], 0.4, label='+ ESM residual SVR', color='#2ECC71')
ax.set_yticks(y); ax.set_yticklabels(pp['base'].str.replace('pI_',''), fontsize=8)
ax.set_xlabel('Test RMSE'); ax.legend(); ax.set_title('Physical baseline swap: raw vs ESM-corrected')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(str(FIGURES / 'fig_z166_pka_swap.png'), dpi=180)
plt.close()

# ---------- 4) ESM encoder swap ----------
# NOTE: Only ESM2-150M is retained (the published encoder used by pI-ESM).
# ESMC-300M, ESMC-600M, ESM2-650M, ESM3-1.4B, and ESM2-35M comparisons are
# omitted because their embedding files are not published.
print('\nSwap ESM encoder:', flush=True)
emb_rows = []
for tag in ['ESM2-150M']:
    xa, xb = pooling.load_emb_pool(tag, pooling.W_ELEGANT)
    cvs, _ = models.cv5(models.make_esm_svr, xa, rtr, bo, ytr, sw)
    pt, _ = models.full_test(models.make_esm_svr, xa, rtr, sw, xb, bt)
    m = metrics.metrics(yte, pt)
    emb_rows.append(dict(encoder=tag, dim=xa.shape[1], OOF=np.mean(cvs),
                         OOF_std=np.std(cvs), TEST=m['RMSE']))
    print(f'  {tag:11s} dim={xa.shape[1]:4d} OOF={np.mean(cvs):.4f} TEST={m["RMSE"]:.4f}', flush=True)
pd.DataFrame(emb_rows).to_csv(str(TABLES / 'z166_swap_ESM_encoder.csv'),
                              index=False, encoding='utf-8-sig')

print(f'\nTotal time {time.time()-t0:.0f}s', flush=True)
