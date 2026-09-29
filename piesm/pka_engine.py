# -*- coding: utf-8 -*-
"""pka_engine: vectorized pI bisection solver with analytic Jacobian.

Extracted from pKa_env_shift_analysis/common.py. Self-contained pI engine
using the Henderson-Hasselbalch charge-balance bisection. The 9-pKa
dimension order matches the IPC2 paper:
[N_TER, C_TER, asp, glu, his, lys, cys, tyr, arg]
"""
from __future__ import annotations

import math

import numpy as np

# === Physical constants for the bisection solver ===
PH_LOW = 2.0       # lower pH bound of the search window
PH_HIGH = 12.0     # upper pH bound of the search window
TOL = 1e-8         # convergence tolerance for the standard bisection
MAX_ITERS = 60     # maximum bisection iterations (standard)

# Tighter convergence used by the Jacobian variant
TOL_TIGHT = 1e-7
MAX_ITERS_TIGHT = 60

# 9-pKa dimension order (matches IPC/IPC2 paper)
DIMS = ['N_TER', 'C_TER', 'asp', 'glu', 'his', 'lys', 'cys', 'tyr', 'arg']
# Indices of basic (protonatable-positive) groups in DIMS: N_TER, his, lys, arg
BASIC_IDX = [0, 4, 5, 8]

# Classic physical prior (Bjellqvist 1994)
PKA_BJELLQVIST = np.array([7.50, 3.55, 4.05, 4.45, 5.98, 10.00, 9.00, 10.00, 12.00])

# Paper-reported fixed pKa set (IPC2)
PKA_IPC2_PAPER = np.array([5.779, 6.065, 3.766, 4.497, 5.492, 9.247, 7.890, 11.491, 10.223])


def compute_pI(pka_vec, counts):
    """Vectorized bisection to find pI (isoelectric point).

    Parameters
    ----------
    pka_vec : array of shape (9,)
        Fixed pKa values for the 9 ionizable groups.
    counts : array of shape (N, 9)
        Per-protein counts of each ionizable group.

    Returns
    -------
    pI : array of shape (N,)
        Isoelectric point for each protein.
    """
    n = counts.shape[0]
    n_basic = counts[:, BASIC_IDX].sum(axis=1)
    low = np.full(n, PH_LOW, dtype=np.float64)
    high = np.full(n, PH_HIGH, dtype=np.float64)
    for _ in range(MAX_ITERS):
        mid = (low + high) * 0.5
        # deprotonation fraction at the midpoint pH
        s = 1.0 / (1.0 + np.power(10.0, pka_vec[None, :] - mid[:, None]))
        neg_sum = (counts * s).sum(axis=1)
        q_mid = n_basic - neg_sum   # net charge at midpoint
        go_low = q_mid > 0.0        # positive charge -> need higher pH
        low = np.where(go_low, mid, low)
        high = np.where(go_low, high, mid)
        if np.max(high - low) < TOL:
            break
    return (low + high) * 0.5


def compute_pI_and_jacobian(pka_vec, counts, tol=TOL_TIGHT, max_iters=MAX_ITERS_TIGHT):
    """Vectorized bisection for pI plus the analytic Jacobian w.r.t. pKa.

    J[i, j] = count_ij * sigma_ij * (1 - sigma_ij) /
              sum_k count_ik * sigma_ik * (1 - sigma_ik)

    Returns
    -------
    pI : array of shape (N,)
    J : array of shape (N, 9)
    """
    n = counts.shape[0]
    n_basic = counts[:, BASIC_IDX].sum(axis=1)
    low = np.full(n, PH_LOW, dtype=np.float64)
    high = np.full(n, PH_HIGH, dtype=np.float64)
    s_final = None
    for _ in range(max_iters):
        mid = (low + high) * 0.5
        s = 1.0 / (1.0 + np.power(10.0, pka_vec[None, :] - mid[:, None]))
        neg_sum = (counts * s).sum(axis=1)
        q_mid = n_basic - neg_sum
        go_low = q_mid > 0.0
        low = np.where(go_low, mid, low)
        high = np.where(go_low, high, mid)
        s_final = s
        if np.max(high - low) < tol:
            break
    pI = (low + high) * 0.5
    dsigma = s_final * (1.0 - s_final)
    weighted = counts * dsigma
    norm = weighted.sum(axis=1)
    safe_norm = np.where(norm > 0, norm, 1.0)
    J = weighted / safe_norm[:, None]
    return pI, J


def rmsd(pI_pred, exp_pI):
    """Root-mean-square deviation between predicted and experimental pI."""
    return math.sqrt(float(np.mean((pI_pred - exp_pI) ** 2)))
