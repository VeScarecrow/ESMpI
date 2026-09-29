# -*- coding: utf-8 -*-
"""metrics: evaluation metrics for pI prediction.

Extracted from z164_common.py.
"""
from __future__ import annotations

import numpy as np


def metrics(y, p):
    """Compute standard regression metrics.

    Parameters
    ----------
    y : array, experimental pI.
    p : array, predicted pI.

    Returns
    -------
    dict with RMSE, MAE, bias (mean error), and outlier rate (|e|>0.5).
    """
    e = p - y
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))),
                MAE=float(np.mean(np.abs(e))),
                bias=float(np.mean(e)),
                outl=float(np.mean(np.abs(e) > 0.5)))


def seg_rmse(y, p):
    """Segmented RMSE by experimental pI range.

    Returns dict keyed by range label: '<6', '6-8', '8-9', '>9'.
    """
    out = {}
    for lo, hi, lab in [(-20, 6, '<6'), (6, 8, '6-8'), (8, 9, '8-9'), (9, 20, '>9')]:
        ix = (y >= lo) & (y < hi)
        out[lab] = float(np.sqrt(np.mean((p[ix] - y[ix]) ** 2)))
    return out
