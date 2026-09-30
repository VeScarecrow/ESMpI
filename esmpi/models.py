# -*- coding: utf-8 -*-
"""models: SVR model factories and cross-validation helpers.

Extracted from z164_common.py. Provides the ESMpI residual SVR and the
IPC2_protein F19-SVR, plus multi-seed 5-fold OOF evaluation.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .dataio import SEEDS

# F19: the 19 pKa-scale pI features used by the official IPC2_protein SVR
# (= z81's 20 pI_* columns minus pI_our)
F19 = ['pI_Bjellqvist', 'pI_DTASelect', 'pI_Dawson', 'pI_EMBOSS', 'pI_Grimsley',
       'pI_IPC2_peptide', 'pI_IPC2_protein', 'pI_IPC_peptide', 'pI_IPC_protein',
       'pI_Lehninger', 'pI_Nozaki', 'pI_Patrickios', 'pI_Rodwell', 'pI_Sillero',
       'pI_Solomon', 'pI_Thurlkill', 'pI_Toseland', 'pI_Wikipedia', 'pI_ProMoST']


def make_esm_svr():
    """ESMpI model: StandardScaler + RBF-SVR (C=0.5, eps=0.05)."""
    return make_pipeline(StandardScaler(),
                         SVR(kernel='rbf', C=0.5, gamma='scale', epsilon=0.05,
                             cache_size=2000))


def make_f19_svr():
    """IPC2_protein.SVR: C=1.0, eps=0.12 (official reproduction)."""
    return make_pipeline(StandardScaler(),
                         SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.12,
                             cache_size=2000))


def fit_weighted(est, X, y, sw):
    """Fit a pipeline estimator with per-sample SVR weights."""
    est.fit(X, y, svr__sample_weight=sw)
    return est


def cv5(make_est, Xtr, rtr, bo, ytr, sw, seeds=SEEDS):
    """Multi-seed 5-fold out-of-fold evaluation.

    Returns
    -------
    cvs : list of per-seed RMSE values.
    oof_main : OOF residual predictions for the main seed (71923).
    """
    cvs, oof_main = [], None
    for k, sd in enumerate(seeds):
        kf = KFold(5, shuffle=True, random_state=sd)
        oof = np.zeros(len(rtr))
        for a, c_ in kf.split(Xtr):
            est = make_est()
            fit_weighted(est, Xtr[a], rtr[a], sw[a])
            oof[c_] = est.predict(Xtr[c_])
        cvs.append(float(np.sqrt(np.mean((bo + oof - ytr) ** 2))))
        if sd == 71923:
            oof_main = oof.copy()
    return cvs, oof_main


def full_test(make_est, Xtr, rtr, sw, Xte, bt):
    """Full retrain on all training data + test prediction.

    Returns
    -------
    (predictions, fitted_estimator).
    """
    est = make_est()
    fit_weighted(est, Xtr, rtr, sw)
    return bt + est.predict(Xte), est
