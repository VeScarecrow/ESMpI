# -*- coding: utf-8 -*-
"""esmpi: ESMpI prediction and reproduction package.

Self-contained package for reproducing the ESMpI isoelectric point
prediction results. Provides the pI bisection engine, data loading,
ESM2 embedding pooling, SVR models, and evaluation metrics.
"""
from __future__ import annotations

__version__ = "1.0.0"

from . import pka_engine, dataio, pooling, models, metrics, structure  # noqa: F401
