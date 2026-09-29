# -*- coding: utf-8 -*-
"""
04_bootstrap_intermethod: paired bootstrap pI-ESM vs IPC2-SVR.

Migrated from z176_ours_vs_f19.py. Compares two methods on the 581-protein
test set with stratified metrics and 2000-iteration paired bootstrap CIs:
  ours : ESM2-150M residual SVR (pI-ESM, C=0.5, eps=0.05, sw-weighted)
  F19  : IPC2_protein.SVR 19-dim pI features (C=1.0, eps=0.12)

Stratification dimensions: data source (SWISS-2DPAGE / PIP-DB), sequence
length (4 bins), experimental pI region (acidic / basic).

Outputs:
  results/tables/z176_ours_vs_f19_metrics.csv   stratified metrics
  results/tables/z176_ours_vs_f19_paired_CI.csv     paired bootstrap CIs
  results/figures/fig_z176_strata.png           stratified bar chart
  results/figures/fig_z176_scatter.png           prediction scatter
  results/figures/fig_z176_error_cdf.png         error CDF
  results/figures/fig_z176_error_vs_pi.png       error vs exp pI
"""
import os, sys, pathlib
sys.stdout.reconfigure(encoding='utf-8')
# Make the piesm package importable (project root is one level up)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from piesm import dataio

# Output directories
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PREDICTIONS = ROOT / "data" / "predictions"
MAPPING = ROOT / "data" / "mapping"
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BOOT_SEED, N_BOOT = 20260919, 2000

d = dataio.load_data()
yte = d['yte']
keys_te = d['keys_te']
preds = np.load(str(PREDICTIONS / 'z166_preds.npz'))
ours, f19 = preds['pt_ours'], preds['pt_f19']
assert len(yte) == 581

# ---- source labels (zidx aligned to z81 row order) ----
lab = pd.read_csv(str(MAPPING / 'z46d_source_labels.csv'))
mp = pd.read_csv(str(MAPPING / 'z164_seqno2zidx.csv'))
src_df = mp.merge(lab[['seq_no_full', 'source']], left_on='seq_no', right_on='seq_no_full',
                  how='left').sort_values('zidx').reset_index(drop=True)
assert len(src_df) == 581 and src_df['source'].notna().all()
src = src_df['source'].values
lens = np.array([len(k) for k in keys_te], dtype=float)

METHODS = [('ours', ours), ('F19', f19)]
C_OURS, C_F19 = '#E67E22', '#5DADE2'


# ---------- Metrics ----------
def metr(p, y, mk):
    e = p[mk] - y[mk]
    return dict(n=int(mk.sum()),
                RMSE=float(np.sqrt(np.mean(e ** 2))),
                MAE=float(np.mean(np.abs(e))),
                MedianAE=float(np.median(np.abs(e))),
                bias=float(np.mean(e)),
                median_e=float(np.median(e)),
                R2=float(1 - np.sum(e ** 2) / np.sum((y[mk] - y[mk].mean()) ** 2)),
                Pearson_r=float(pearsonr(y[mk], p[mk])[0]),
                Spearman_rho=float(spearmanr(y[mk], p[mk])[0]),
                outl_05=float(np.mean(np.abs(e) > 0.5)),
                outl_10=float(np.mean(np.abs(e) > 1.0)))


def paired(y, po, pf, mk):
    """Paired bootstrap CI + win rate. Delta = F19 - ours; positive = ours better."""
    eo, ef = po[mk] - y[mk], pf[mk] - y[mk]
    ao, af = np.abs(eo), np.abs(ef)
    n = len(eo)
    rng = np.random.RandomState(BOOT_SEED)
    dr, dm = np.zeros(N_BOOT), np.zeros(N_BOOT)
    for b in range(N_BOOT):
        ix = rng.randint(0, n, n)
        dr[b] = np.sqrt(np.mean(ef[ix] ** 2)) - np.sqrt(np.mean(eo[ix] ** 2))
        dm[b] = af[ix].mean() - ao[ix].mean()
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    dr_lo, dr_hi = ci(dr); dm_lo, dm_hi = ci(dm)
    return dict(n=n,
                dRMSE=float(np.sqrt(np.mean(ef ** 2)) - np.sqrt(np.mean(eo ** 2))),
                dRMSE_lo=dr_lo, dRMSE_hi=dr_hi, dRMSE_sig=bool(dr_lo > 0 or dr_hi < 0),
                dMAE=float(af.mean() - ao.mean()),
                dMAE_lo=dm_lo, dMAE_hi=dm_hi, dMAE_sig=bool(dm_lo > 0 or dm_hi < 0),
                win_ours=float(np.mean(ao < af)), win_F19=float(np.mean(af < ao)),
                tie=float(np.mean(ao == af)),
                MAE_reduction_pct=float(100 * (af.mean() - ao.mean()) / af.mean()))


# ---------- Stratification ----------
len_defs = [(0, 200, 'L≤200'), (200, 350, '200<L≤350'),
            (350, 550, '350<L≤550'), (550, 10 ** 9, 'L>550')]
dims = []
dims.append(('Data_source', 'source',
             [('SWISS-2DPAGE', src == 'SWISS-2DPAGE'), ('PIP-DB', src == 'PIP-DB')]))
dims.append(('Sequence_length', 'length',
             [(nm, (lens > lo) & (lens <= hi)) for lo, hi, nm in len_defs]))
dims.append(('Exp_pI_region', 'pI', [('pI<7_acidic', yte < 7), ('pI>=7_basic', yte >= 7)]))

rows, pairs = [], []
# Overall
for nm, p in METHODS:
    r = metr(p, yte, np.ones(581, bool)); rows.append(dict(dim='Overall', stratum='ALL', method=nm, **r))
pr = paired(yte, ours, f19, np.ones(581, bool)); pairs.append(dict(dim='Overall', stratum='ALL', **pr))
# Each stratum
for dim_cn, dim_key, strata in dims:
    for nm, mk in strata:
        for mname, p in METHODS:
            rows.append(dict(dim=dim_cn, stratum=nm, method=mname, **metr(p, yte, mk)))
        pairs.append(dict(dim=dim_cn, stratum=nm, **paired(yte, ours, f19, mk)))

long = pd.DataFrame(rows)
pci = pd.DataFrame(pairs)
long.to_csv(str(TABLES / 'z176_ours_vs_f19_metrics.csv'),
            index=False, encoding='utf-8-sig')
pci.to_csv(str(TABLES / 'z176_ours_vs_f19_paired_CI.csv'),
           index=False, encoding='utf-8-sig')

# ---------- Console summary ----------
show_cols = ['n', 'RMSE', 'MAE', 'MedianAE', 'bias', 'R2', 'Pearson_r',
             'Spearman_rho', 'outl_05', 'outl_10']
for dim_cn in ['Overall', 'Data_source', 'Sequence_length', 'Exp_pI_region']:
    sub = long[long.dim == dim_cn]
    print('\n=== %s ===' % dim_cn)
    piv = sub.pivot(index='stratum', columns='method', values='RMSE')
    print(sub.pivot_table(index='stratum', columns='method', values=show_cols).round(4).to_string())
print('\n=== Paired stats (Delta=F19-ours, positive=ours better) ===')
print(pci[['dim', 'stratum', 'n', 'dRMSE', 'dRMSE_lo', 'dRMSE_hi', 'dRMSE_sig',
           'dMAE', 'dMAE_lo', 'dMAE_hi', 'win_ours', 'win_F19', 'tie',
           'MAE_reduction_pct']].round(4).to_string())

# ---------- Figure 1: stratified RMSE/MAE bars ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (dim_cn, _, strata) in zip(axes, dims):
    names = [nm for nm, _ in strata]
    sub = long[long.dim == dim_cn].set_index(['stratum', 'method'])
    x = np.arange(len(names)); w = 0.36
    for k, (mname, c) in enumerate([('ours', C_OURS), ('F19', C_F19)]):
        vals = [sub.loc[(nm, mname), 'RMSE'] for nm in names]
        ax.bar(x + (k - 0.5) * w, vals, w, label=mname, color=c)
        for xx, vv in zip(x + (k - 0.5) * w, vals):
            ax.text(xx, vv + 0.01, '%.3f' % vv, ha='center', fontsize=8)
    ns = [int(mk.sum()) for _, mk in strata]
    ax.set_xticks(x); ax.set_xticklabels(['%s\n(n=%d)' % (nm, n) for nm, n in zip(names, ns)], fontsize=9)
    ax.set_ylabel('RMSE'); ax.set_ylim(0, max([sub.loc[(nm, m), 'RMSE']
                                          for nm in names for m in ('ours', 'F19')]) * 1.18)
    ax.set_title({'Data_source': 'By data source', 'Sequence_length': 'By sequence length', 'Exp_pI_region': 'By experimental pI region'}[dim_cn])
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(str(FIGURES / 'fig_z176_strata.png'), dpi=170)
plt.close()

# ---------- Figure 2: predicted vs experimental scatter ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
for ax, (mname, p, c) in zip(axes, [('ours', ours, C_OURS), ('F19', f19, C_F19)]):
    for lab_reg, mk, cc, mk_ in [('pI<7', yte < 7, '#3B7DDD', 'o'),
                                ('pI≥7', yte >= 7, '#E76F51', '^')]:
        ax.scatter(yte[mk], p[mk], s=16, alpha=0.55, c=cc, marker=mk_, label=lab_reg, edgecolors='none')
    lim = [3.2, 12.6]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.7)
    r = metr(p, yte, np.ones(581, bool))
    ax.set_title('%s  RMSE=%.3f  MAE=%.3f  R²=%.3f' % (mname, r['RMSE'], r['MAE'], r['R2']), fontsize=11)
    ax.set_xlabel('Exp. pI'); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(fontsize=9, loc='upper left')
axes[0].set_ylabel('Predicted pI')
plt.tight_layout()
plt.savefig(str(FIGURES / 'fig_z176_scatter.png'), dpi=170)
plt.close()

# ---------- Figure 3: absolute error CDF ----------
fig, ax = plt.subplots(figsize=(7.5, 5))
for mname, p, c in [('ours', ours, C_OURS), ('F19', f19, C_F19)]:
    ae = np.sort(np.abs(p - yte))
    ax.plot(ae, np.linspace(0, 1, len(ae)), lw=2, color=c, label=mname)
for thr in [0.5, 1.0]:
    ax.axvline(thr, color='gray', ls=':', lw=1)
    ax.text(thr, 0.05, '|e|=%.1f' % thr, rotation=90, fontsize=8, color='gray', va='bottom')
ax.set_xlabel('Absolute error |pred − exp|'); ax.set_ylabel('Cumulative fraction')
ax.set_xlim(0, 3.5); ax.legend()
ax.set_title('Empirical CDF of absolute error (leftward = better)')
plt.tight_layout()
plt.savefig(str(FIGURES / 'fig_z176_error_cdf.png'), dpi=170)
plt.close()

# ---------- Figure 4: error vs experimental pI (scatter + binned mean) --------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
bins = np.arange(3.5, 12.5, 0.5)
for ax, (mname, p, c) in zip(axes, [('ours', ours, C_OURS), ('F19', f19, C_F19)]):
    e = p - yte
    ax.scatter(yte, e, s=12, alpha=0.35, color=c, edgecolors='none')
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.axvline(7, color='#888', ls=':', lw=1.2)
    bc = 0.5 * (bins[:-1] + bins[1:])
    bm = np.array([e[(yte >= lo) & (yte < hi)].mean() if ((yte >= lo) & (yte < hi)).sum() >= 5
                   else np.nan for lo, hi in zip(bins[:-1], bins[1:])])
    ax.plot(bc, bm, color='red', lw=2, marker='o', ms=4, label='0.5-pH binned mean error')
    ax.set_title(mname); ax.set_xlabel('Exp. pI'); ax.legend(fontsize=9)
    ax.set_ylim(-2.5, 3.2)
axes[0].set_ylabel('Prediction error (pred − exp)')
plt.tight_layout()
plt.savefig(str(FIGURES / 'fig_z176_error_vs_pi.png'), dpi=170)
plt.close()

print('\nsaved tables + 4 figures ->', str(TABLES))
