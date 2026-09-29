# -*- coding: utf-8 -*-
"""
06_stratified_analysis: stratified RMSE analysis on the 581-protein test set.

Migrated from z172_strata.py. Evaluates 7 methods (9p-IPC2, 9p-Opt, F19,
ours, M1 DeepKa, M2 PropKa, M5 DeepKa-buried) plus M7 PypKa (covered subset)
across three stratification dimensions: data source, sequence length, and
experimental pI range.

Outputs:
  results/tables/z172_stratified_metrics_long.csv       long-format stratified metrics
  results/tables/z172_stratified_source.csv             source pivot (all581)
  results/tables/z172_stratified_length.csv             length pivot (all581)
  results/tables/z172_stratified_pI.csv                 pI pivot (all581)
  results/tables/z172_stratified_source_PypKa_covered.csv  PypKa-covered source
  results/tables/z172_stratified_length_PypKa_covered.csv  PypKa-covered length
  results/tables/z172_stratified_pI_PypKa_covered.csv      PypKa-covered pI
  results/figures/fig_z172_strata.png                   stratified bar chart
  results/figures/fig_z172_strata_pypka.png             PypKa-covered bar chart
"""
import sys, os, pathlib
sys.stdout.reconfigure(encoding='utf-8')
# Make the piesm package importable (project root is one level up)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from piesm import dataio

# Output directories
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PREDICTIONS = ROOT / "data" / "predictions"
MAPPING = ROOT / "data" / "mapping"
BENCHMARK = ROOT / "data" / "benchmark"
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

d = dataio.load_data()
fte, yte = d['fte'], d['yte']
keys_te = d['keys_te']

pi = pd.read_csv(str(BENCHMARK / 'pi_af.csv'))
py = pd.read_csv(str(BENCHMARK / 'pypka_exp_z44.csv'))
lab = pd.read_csv(str(MAPPING / 'z46d_source_labels.csv'))
mp = pd.read_csv(str(MAPPING / 'z164_seqno2zidx.csv'))
preds = np.load(str(PREDICTIONS / 'z166_preds.npz'))

m = pi.merge(mp, on='seq_no').merge(
    py[['seq_no', 'M7_PypKa_Thurl']], on='seq_no', how='left'
).merge(lab[['seq_no_full', 'source']].rename(columns={'source': 'data_source'}),
        left_on='seq_no', right_on='seq_no_full',
        how='left').sort_values('zidx').reset_index(drop=True)
zidx = m['zidx'].values.astype(int)
assert sorted(zidx.tolist()) == list(range(581))
assert np.abs(m['exp_pI'].values - yte).max() == 0.0

lens = np.array([len(k) for k in keys_te], dtype=float)   # z81 row order
# m is sorted by zidx => consistent with z81 row order
lens_m = lens
src = m['data_source'].values
cov = m['M7_PypKa_Thurl'].notna().values
N_PY = int(cov.sum())
PY_LABEL = f'PypKa{N_PY}'
print(f'PypKa covered subset n={N_PY}')

# Structure method columns consistent with 02 / Table 1 "AF_primary" convention (pi_af.csv _AF columns)
methods = {
    '9p-IPC2': fte['pI_IPC2_protein'].values,          # IPC2 original 9 parameters
    '9p-Opt': m['M3_our_9param_AF'].values,            # re-optimized 9 parameters (GBMS)
    'F19': preds['pt_f19'],                            # IPC2.SVR(19-dim features)
    'ours': preds['pt_ours'],                          # ESM residual SVR (this model)
    'M1 DeepKa': m['M1_DeepKa_Thurl_AF'].values,
    'M2 PropKa': m['M2_PropKa_Thurl_AF'].values,
    'M5 DeepKa-buried': m['M5_DeepKa_SA_ipc_AF'].values,
}
methods_py = {'M7 PypKa': m['M7_PypKa_Thurl'].values}

def metr(p, y, mk):
    e = p[mk] - y[mk]
    return dict(n=int(mk.sum()), RMSE=np.sqrt(np.mean(e**2)),
                MAE=np.mean(np.abs(e)), bias=np.mean(e))

# ---------- Bin definitions ----------
print('Length quantiles:', np.percentile(lens_m, [0,25,50,75,100]).round(0))
len_bins = [(0,200), (200,350), (350,550), (550,100000)]
pi_bins = [(0,5), (5,6), (6,7), (7,8), (8,9), (9,15)]

def len_bin(L):
    return [f'{lo}<L<={hi}' if hi<100000 else f'L>{lo}' for lo, hi in len_bins], \
        [((L>lo)&(L<=hi)) for lo, hi in len_bins]
def pi_bin(y):
    return [f'{lo}–{hi}' if hi<15 else f'>{lo}' for lo, hi in pi_bins], \
        [((y>=lo)&(y<hi)) for lo, hi in pi_bins]

src_names, src_masks = ['SWISS-2DPAGE', 'PIP-DB'], [src=='SWISS-2DPAGE', src=='PIP-DB']
len_names, len_masks = len_bin(lens_m)
pi_names, pi_masks = pi_bin(yte)

dims = [('source', src_names, src_masks), ('length', len_names, len_masks),
        ('pI', pi_names, pi_masks)]

rows = []
for dim, names, masks in dims:
    for nm, mk in zip(names, masks):
        for mname, p in methods.items():
            r = metr(p, yte, mk)
            rows.append(dict(dim=dim, stratum=nm, method=mname, coverage='all581', **r))
        # M7 only in mk∩cov
        for mname, p in methods_py.items():
            r = metr(p, yte, mk & cov)
            rows.append(dict(dim=dim, stratum=nm, method=mname, coverage=PY_LABEL, **r))
        # All methods in mk∩cov for same-convention reference
        for mname, p in methods.items():
            r = metr(p, yte, mk & cov)
            rows.append(dict(dim=dim, stratum=nm, method=mname, coverage=f'on{N_PY}', **r))

long = pd.DataFrame(rows)
long.to_csv(str(TABLES / 'z172_stratified_metrics_long.csv'), index=False, encoding='utf-8-sig')

# Pivot print
for dim, names, _ in dims:
    piv = long[(long.dim==dim)&(long.coverage=='all581')].pivot(
        index='stratum', columns='method', values='RMSE').loc[names]
    ns = long[(long.dim==dim)&(long.coverage=='all581')].pivot(
        index='stratum', columns='method', values='n').loc[names]
    print(f'\n=== {dim} RMSE (all581) ===')
    print('n:', ns.iloc[:,0].astype(int).to_dict())
    print(piv.round(4).to_string())
    piv.round(4).to_csv(str(TABLES / f'z172_stratified_{dim}.csv'), encoding='utf-8-sig')
    pivp = long[(long.dim==dim)&(long.coverage.isin([PY_LABEL, f'on{N_PY}']))].pivot_table(
        index='stratum', columns='method', values='RMSE').loc[names]
    print(f'--- {dim} RMSE (on PypKa-covered subset, n={N_PY}) ---')
    print(pivp.round(4).to_string())
    pivp.round(4).to_csv(str(TABLES / f'z172_stratified_{dim}_PypKa_covered.csv'),
                         encoding='utf-8-sig')

# ---------- Plot ----------
colors = {'9p-IPC2':'#95A5A6','9p-Opt':'#BDC3C7','F19':'#5DADE2','ours':'#E67E22',
          'M1 DeepKa':'#27AE60','M2 PropKa':'#E74C3C','M5 DeepKa-buried':'#8E44AD',
          'M7 PypKa':'#16A085'}
order = ['9p-IPC2','9p-Opt','F19','M2 PropKa','M1 DeepKa','M5 DeepKa-buried','ours']
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
for ax, (dim, names, _), ttl in zip(
        axes, dims, ['By data source', 'By sequence length (aa)', 'By experimental pI']):
    sub = long[(long.dim==dim)&(long.coverage=='all581')]
    xx = np.arange(len(names)); w = 0.13
    for k, mname in enumerate(order):
        vals = [sub[(sub.stratum==nm)&(sub.method==mname)]['RMSE'].iloc[0] for nm in names]
        ax.bar(xx+(k-2.5)*w, vals, w, label=mname, color=colors[mname])
    ns = [int(sub[(sub.stratum==nm)&(sub.method=='ours')]['n'].iloc[0]) for nm in names]
    ax.set_xticks(xx); ax.set_xticklabels([f'{nm}\n(n={n})' for nm,n in zip(names,ns)], fontsize=8.5)
    ax.set_ylabel('RMSE'); ax.set_title(ttl, fontsize=11)
    if dim=='pI': ax.set_ylim(0, 2.6)
axes[2].legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig(str(FIGURES / 'fig_z172_strata.png'), dpi=170)
plt.close()

# PypKa-convention plot (stratum∩178)
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
orderp = ['9p-IPC2','9p-Opt','F19','M2 PropKa','M1 DeepKa','M5 DeepKa-buried','M7 PypKa','ours']
for ax, (dim, names, _), ttl in zip(
        axes, dims, ['Source (PypKa-covered)', 'Length (PypKa-covered)', 'pI (PypKa-covered)']):
    sub = long[(long.dim==dim)&(long.coverage.isin([PY_LABEL, f'on{N_PY}']))]
    xx = np.arange(len(names)); w = 0.10
    for k, mname in enumerate(orderp):
        vals = [sub[(sub.stratum==nm)&(sub.method==mname)]['RMSE'].iloc[0] for nm in names]
        ax.bar(xx+(k-3.5)*w, vals, w, label=mname,
               color=colors[mname], hatch='//' if mname=='M7 PypKa' else None)
    ns = [int(sub[(sub.stratum==nm)&(sub.method=='ours')]['n'].iloc[0]) for nm in names]
    ax.set_xticks(xx); ax.set_xticklabels([f'{nm}\n(n={n})' for nm,n in zip(names,ns)], fontsize=8.5)
    ax.set_ylabel('RMSE'); ax.set_title(ttl, fontsize=11)
    if dim=='pI': ax.set_ylim(0, 2.6)
axes[2].legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig(str(FIGURES / 'fig_z172_strata_pypka.png'), dpi=170)
plt.close()
print('\nsaved figures.')
