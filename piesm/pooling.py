# -*- coding: utf-8 -*-
"""pooling: ESM2 embedding loading and 7-type weighted pooling.

Extracted from z164_common.py. Only the ESM2-150M entry is retained
(the only encoder used by the published pI-ESM model).
"""
from __future__ import annotations

import numpy as np

from .dataio import load_data, DATA_DIR

# C3 weights (learned on IPC2 base; order D/E/H/K/C/Y/R)
W_ELEGANT = np.array([0.0156, 0.0156, 0.0062, 0.5, 0.05, 0.0062, 1.0])

# ESM2-150M per-type embedding files (train, test) and pooling key.
# Only ESM2-150M is retained; other encoders are not published.
EMB_FILES = {
    'ESM2-150M': (str(DATA_DIR / "embeddings" / "z82_150m_pertype_train.npz"),
                  str(DATA_DIR / "embeddings" / "z82_150m_pertype_test.npz"),
                  'site_uniform'),
}


def load_emb_pool(tag, w=W_ELEGANT):
    """Read per-type npz embeddings and apply 7-type weighted pooling.

    Verifies key alignment against the feature table via load_data().

    Returns
    -------
    (Xtr, Xte) : pooled feature matrices (float32).
    """
    ftr, fte, uk = EMB_FILES[tag]
    dtr = np.load(ftr, allow_pickle=True)
    dte = np.load(fte, allow_pickle=True)
    d = load_data()
    assert [str(x) for x in dtr['keys']] == d['keys_tr']
    assert [str(x) for x in dte['keys']] == d['keys_te']

    def pool(dd):
        pt = dd['site_per_type'].astype(np.float64)   # (n, 7, H)
        cn = dd['site_counts'].astype(np.float64)      # (n, 7)
        wc = cn * w[None, :]
        v = np.einsum('nk,nkh->nh', wc, pt) / np.clip(wc.sum(1, keepdims=True), 1e-9, None)
        return v.astype(np.float32)
    return pool(dtr), pool(dte)
