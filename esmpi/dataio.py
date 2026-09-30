# -*- coding: utf-8 -*-
"""dataio: data loading and sequence feature extraction for ESMpI.

Extracted from z164_common.py. Reads the z81 feature tables, builds 9-dim
ionizable-group count vectors, KFold splits, and sample weights.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .pka_engine import compute_pI

# Project root (one level above the esmpi/ package directory)
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Random seeds for multi-seed 5-fold CV (matches z164_common)
SEEDS = [71923, 42, 123, 777, 2024]
# The 7 ionizable side-chain / terminus types (excludes N_TER, C_TER)
SITE7 = ['D', 'E', 'H', 'K', 'C', 'Y', 'R']

_CACHE = {}


def load_data():
    """Load train/test feature tables and build derived arrays.

    Returns a dict with keys: ytr, yte, ftr, fte, ctr, cte, kf, sw,
    keys_tr, keys_te. Results are cached on first call.
    """
    if 'd' in _CACHE:
        return _CACHE['d']
    ftr = pd.read_csv(DATA_DIR / "features" / "z81_features_train.csv")
    fte = pd.read_csv(DATA_DIR / "features" / "z81_features_test.csv")
    ytr = ftr.exp_pI.values.astype(np.float64)
    yte = fte.exp_pI.values.astype(np.float64)
    ctr = np.array([seq_counts(s) for s in ftr['key'].astype(str)])
    cte = np.array([seq_counts(s) for s in fte['key'].astype(str)])
    kf = KFold(5, shuffle=True, random_state=71923)
    # sample weight: up-weight proteins with |pI|>9
    sw = (1.0 + np.maximum(np.abs(ytr) - 9.0, 0.0))
    d = dict(ytr=ytr, yte=yte, ftr=ftr, fte=fte, ctr=ctr, cte=cte,
             kf=kf, sw=sw,
             keys_tr=ftr['key'].astype(str).tolist(),
             keys_te=fte['key'].astype(str).tolist())
    _CACHE['d'] = d
    return d


def seq_counts(seq):
    """Count 9 ionizable groups in a sequence: [N_TER, C_TER, D, E, H, K, C, Y, R].

    N-terminus and C-terminus are always present (count=1). B/U/X/Z and
    other non-ionizable residues are ignored.
    """
    v = np.zeros(9)
    v[0] = 1.0  # N-terminus
    v[1] = 1.0  # C-terminus
    mp = {'D': 2, 'E': 3, 'H': 4, 'K': 5, 'C': 6, 'Y': 7, 'R': 8}
    for ch in seq:
        j = mp.get(ch)
        if j is not None:
            v[j] += 1.0
    return v


def base_pi(pka, counts):
    """9-parameter physical pI via the bisection engine."""
    return compute_pI(np.asarray(pka, np.float64), np.asarray(counts, np.float64))
