# -*- coding: utf-8 -*-
"""
08_ipc2_protocol: reproduce the IPC2 paper Table 2 evaluation protocol.

SI Table S4. IPC2's printed Table 2 reports RMSE/MAE/R2 obtained by averaging
per-fold metrics of a random 10-fold split on the test-set predictions (not the
pooled full-set RMSE). Their official code uses RepeatedKFold; here we use
20 seeds x 10 folds = 200 folds to remove split randomness. The two protocols
differ systematically by ~0.005 pH (Jensen's inequality):
    E[sqrt(MSE_fold)] < sqrt(E[MSE_fold]).

For every raw pI method we report pooled metrics, 10-fold-CV metrics, and the
IPC2 paper printed value where available (plus our GBMS re-optimized scale).

Output:
  results/tables/table_ipc2_protocol.csv
Frozen reference: data/reference_results/table_ipc2_protocol.csv
"""
import sys, os, pathlib
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from piesm import dataio

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
os.makedirs(TABLES, exist_ok=True)

THR = 0.5

# IPC2 paper Table 2 printed values (protein, N=581, outlier threshold 0.5)
PAPER_T2 = {
    'IPC_protein':  (0.8677, 0.6109, 0.5760, 250),
    'IPC2_protein': (0.8608, 0.6052, 0.5748, 251),
    'ProMoST':      (0.9113, 0.6444, 0.5183, 263),
    'Toseland':     (0.9278, 0.6537, 0.5095, 250),
    'Dawson':       (0.9365, 0.6586, 0.4977, 263),
    'Bjellqvist':   (0.9369, 0.6536, 0.5005, 260),
    'Wikipedia':    (0.9484, 0.6795, 0.4860, 262),
    'Rodwell':      (0.9579, 0.6762, 0.4706, 262),
    'Grimsley':     (0.9588, 0.6953, 0.4779, 265),
    'Lehninger':    (0.9617, 0.6783, 0.4607, 266),
    'Solomon':      (0.9631, 0.6746, 0.4606, 272),
    'Nozaki':       (1.0164, 0.7219, 0.3980, 288),
    'Thurlkill':    (1.0250, 0.7573, 0.3948, 302),
    'DTASelect':    (1.0278, 0.7798, 0.3947, 319),
    'EMBOSS':       (1.0498, 0.7757, 0.3734, 308),
    'Sillero':      (1.0519, 0.7694, 0.3461, 308),
}

# feature-table column -> display name
FEAT_MAP = {
    'pI_Bjellqvist':  'Bjellqvist',
    'pI_DTASelect':   'DTASelect',
    'pI_Dawson':      'Dawson',
    'pI_EMBOSS':      'EMBOSS',
    'pI_Grimsley':    'Grimsley',
    'pI_IPC_protein': 'IPC_protein',
    'pI_Lehninger':   'Lehninger',
    'pI_Nozaki':      'Nozaki',
    'pI_Patrickios':  'Patrickios',
    'pI_Rodwell':     'Rodwell',
    'pI_Sillero':     'Sillero',
    'pI_Solomon':     'Solomon',
    'pI_Thurlkill':   'Thurlkill',
    'pI_Toseland':    'Toseland',
    'pI_Wikipedia':   'Wikipedia',
    'pI_ProMoST':     'ProMoST',
}


def pooled_metrics(p, e):
    r = p - e
    sst = np.sum((e - e.mean()) ** 2)
    return dict(rmse=float(np.sqrt(np.mean(r ** 2))),
                mae=float(np.mean(np.abs(r))),
                r2=float(1.0 - np.sum(r ** 2) / sst),
                outliers=int(np.sum(np.abs(r) > THR)))


def fold_metrics(p, e, n_seeds=20, n_splits=10):
    """IPC2 protocol: average per-fold RMSE/MAE/R2 over 200 random folds."""
    r = p - e
    folds = []
    for s in range(n_seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=s)
        for _, te in kf.split(p):
            folds.append(te)
    fr, fm, fr2 = [], [], []
    for te in folds:
        rr, ee = r[te], e[te]
        sst_f = np.sum((ee - ee.mean()) ** 2)
        fr.append(float(np.sqrt(np.mean(rr ** 2))))
        fm.append(float(np.mean(np.abs(rr))))
        fr2.append(1.0 - float(np.sum(rr ** 2)) / sst_f)
    return dict(rmse=float(np.mean(fr)), mae=float(np.mean(fm)),
                r2=float(np.mean(fr2)),
                outliers=int(np.sum(np.abs(r) > THR)))


def main():
    d = dataio.load_data()
    fte = d['fte']
    e = d['yte']
    preds = np.load(str(ROOT / 'data' / 'predictions' / 'z166_preds.npz'))
    assert len(e) == 581
    # npz rows and feature rows share the same zidx order
    assert np.allclose(preds['yte'], e)

    # sequence-model predictions (npz) + GBMS re-optimized baseline (features)
    seq_pred = {
        'IPC2_protein': preds['bt_ipc2'],
        'IPC2.svr.19':  preds['pt_f19'],
        'pI-ESM':       preds['pt_ours'],
        'GBMS':         fte['pI_our'].values,
    }

    rows = []
    def add(name, p):
        pm, fm = pooled_metrics(p, e), fold_metrics(p, e)
        paper = PAPER_T2.get(name, (None, None, None, None))
        rows.append(dict(
            method=name,
            pooled_rmse=round(pm['rmse'], 4), pooled_mae=round(pm['mae'], 4),
            pooled_r2=round(pm['r2'], 4), pooled_out=pm['outliers'],
            fold_rmse=round(fm['rmse'], 4), fold_mae=round(fm['mae'], 4),
            fold_r2=round(fm['r2'], 4), fold_out=fm['outliers'],
            paper_rmse=paper[0], paper_mae=paper[1],
            paper_r2=paper[2], paper_out=paper[3],
            diff_rmse=round(fm['rmse'] - paper[0], 4) if paper[0] is not None else None,
        ))

    for col, name in FEAT_MAP.items():
        if col in fte.columns:
            add(name, fte[col].values)
    for name, p in seq_pred.items():
        add(name, p)

    df = pd.DataFrame(rows).sort_values('fold_rmse').reset_index(drop=True)
    out = TABLES / 'table_ipc2_protocol.csv'
    df.to_csv(out, index=False, encoding='utf-8-sig')

    print('=' * 100)
    print('IPC2 paper protocol (10-fold CV on test_25, 200-fold expectation) '
          'vs pooled vs paper')
    print('=' * 100)
    print(df[['method', 'pooled_rmse', 'fold_rmse', 'paper_rmse', 'diff_rmse']]
          .to_string(index=False))
    print(f'\nsaved -> {out}')

    ipc = df[df.method == 'IPC_protein'].iloc[0]
    print(f"[check] IPC_protein fold RMSE={ipc.fold_rmse} vs paper=0.8677 "
          f"diff={ipc.diff_rmse}")
    assert abs(ipc['diff_rmse']) < 0.005

    # frozen-reference consistency
    ref_path = ROOT / 'data' / 'reference_results' / 'table_ipc2_protocol.csv'
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        m = df.merge(ref, on='method', suffixes=('', '_ref'))
        bad = m[(m.pooled_rmse - m.pooled_rmse_ref).abs() > 5e-4]
        assert len(bad) == 0, f'frozen reference mismatch: {bad.method.tolist()}'
        print('[check] matches frozen reference table_ipc2_protocol.csv')


if __name__ == '__main__':
    main()
