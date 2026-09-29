# -*- coding: utf-8 -*-
"""Structure-based pI calculation from per-residue pKa values.

Two protocols:
  FULL: use model-predicted pKa for all available sites (NTR, CTR, side chains);
        fall back to Thurlkill 2006 for missing sites.
  DEHK: use model-predicted pKa only for D/E/H/K side chains; everything else
        (including NTR/CTR) falls back to Thurlkill.

Self-contained bisection solver (per-protein, not vectorized) to handle
variable-length per-residue pKa lists that differ from protein to protein.
"""
from __future__ import annotations

import numpy as np

# === Thurlkill 2006 model-compound pKa values ===
THURL = {
    "NTR": 8.00, "CTR": 3.67,
    "ASP": 3.67, "GLU": 4.25, "HIS": 6.54,
    "LYS": 10.40, "CYS": 8.55, "TYR": 9.84, "ARG": 12.00,
}

AA3 = {"D": "ASP", "E": "GLU", "H": "HIS", "K": "LYS",
       "C": "CYS", "Y": "TYR", "R": "ARG"}

SIDE_GROUPS = ["ASP", "GLU", "HIS", "LYS", "CYS", "TYR", "ARG"]
DEHK = {"ASP", "GLU", "HIS", "LYS"}
ACID = {"ASP", "GLU", "CYS", "TYR", "CTR"}  # deprotonated → negative charge

# Bisection parameters (consistent with pka_engine.py)
PH_LOW = 2.0
PH_HIGH = 12.0
TOL = 1e-8
MAX_ITERS = 60


def pi_from_pairs(pairs):
    """Compute pI from a list of (pKa, is_acid) pairs via bisection.

    Parameters
    ----------
    pairs : list of (float, bool)
        Each tuple is (pKa_value, is_acid_group).
        Acid groups (ASP, GLU, CYS, TYR, CTR) carry 0 to -1 charge.
        Basic groups (NTR, HIS, LYS, ARG) carry +1 to 0 charge.

    Returns
    -------
    float : pH where net charge = 0.
    """
    pka_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    is_acid = np.array([p[1] for p in pairs], dtype=bool)
    n_basic = int((~is_acid).sum())

    low, high = PH_LOW, PH_HIGH
    for _ in range(MAX_ITERS):
        mid = (low + high) * 0.5
        s = 1.0 / (1.0 + 10.0 ** (pka_arr - mid))
        neg = s[is_acid].sum()
        pos = (1.0 - s[~is_acid]).sum()
        q = pos - neg
        if q > 0:
            low = mid
        else:
            high = mid
        if high - low < TOL:
            break
    return (low + high) * 0.5


def empty_sites():
    """Create an empty per-residue pKa container."""
    return {"side": {}, "ntr": None, "ctr": None}


def assemble(seq, sites, mode):
    """Assemble (pKa, is_acid) pairs for a protein under the given protocol.

    Parameters
    ----------
    seq : str : amino acid sequence (1-letter).
    sites : dict : per-residue pKa container with keys 'ntr', 'ctr', 'side'.
                  'side' is {position: {group: pKa}}.
    mode : str : 'FULL' or 'DEHK'.

    Returns
    -------
    list of (float, bool) : pairs for pI calculation.
    """
    L = len(seq)
    planned = [("NTR", 1), ("CTR", L)]
    for i, aa in enumerate(seq, 1):
        if aa in AA3:
            planned.append((AA3[aa], i))

    out = []
    for grp, pos in planned:
        pk = None
        if grp == "NTR":
            if mode == "FULL":
                pk = sites.get("ntr")
        elif grp == "CTR":
            if mode == "FULL":
                pk = sites.get("ctr")
        else:
            pk = sites.get("side", {}).get(pos, {}).get(grp)
            if mode == "DEHK" and grp not in DEHK:
                pk = None
        out.append((pk if pk is not None else THURL[grp], grp in ACID))
    return out


def structure_pi(seq, sites, mode):
    """Calculate structure-based pI for a single protein."""
    return pi_from_pairs(assemble(seq, sites, mode))


def thurlkill_pi(seq):
    """Pure Thurlkill baseline pI (no structure-based pKa)."""
    pairs = [(THURL["NTR"], False), (THURL["CTR"], True)]
    pairs += [(THURL[AA3[aa]], AA3[aa] in ACID) for aa in seq if aa in AA3]
    return pi_from_pairs(pairs)


def sites_to_rows(sn, track, method, sites, seq):
    """Flatten parsed sites into long-table rows for export."""
    rows = []
    if sites.get("ntr") is not None:
        rows.append(dict(seq_no=sn, track=track, method=method,
                         kind="TER", group="NTR", pos=1, pka=sites["ntr"]))
    if sites.get("ctr") is not None:
        rows.append(dict(seq_no=sn, track=track, method=method,
                         kind="TER", group="CTR", pos=len(seq), pka=sites["ctr"]))
    for pos, gd in sites.get("side", {}).items():
        for grp, pk in gd.items():
            rows.append(dict(seq_no=sn, track=track, method=method,
                             kind="SIDE", group=grp, pos=pos, pka=pk))
    return rows
