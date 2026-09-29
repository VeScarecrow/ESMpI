# -*- coding: utf-8 -*-
"""10: Calculate structure-based pI from per-residue pKa predictions.

Reads per-residue pKa output files from DeepKa, PropKa, and PypKa for each
protein structure, assembles (pKa, is_acid) pairs under FULL and DEHK
protocols, and computes pI via the Henderson-Hasselbalch bisection solver.

Prerequisites:
  - Per-residue pKa files in data/pka_af2/ and data/pka_exp/ (see below)
  - Experimental structures (from script 09) or AF2 structures (provided as zip)
  - Sequences from data/benchmark/IPC_protein_25.csv

Expected directory layout (if you have raw pKa outputs):
  data/pka_af2/deepka/{seq_no}_deepka.csv   — columns: ...pos, res, ..., pka
  data/pka_af2/propka/{seq_no}_propka.csv   — columns: res_name, res_number, pka
  data/pka_af2/pypka/{seq_no}_pKa.csv       — columns: res_name, res_number, pKa
  data/pka_exp/deepka/{seq_no}_deepka.csv   — same format
  data/pka_exp/propka/{seq_no}_propka.csv
  data/pka_exp/pypka/{seq_no}_pKa.csv

If raw pKa files are not available, this script will:
  1. Skip structure pI calculation
  2. Verify frozen results in pi_af.csv against pi_fallback.csv

Usage:
  python scripts/10_structure_pI.py
  python scripts/10_structure_pI.py --pka-dir data/pka_af2
"""
import argparse
import os
import sys
import pathlib

sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from piesm.structure import (
    THURL, AA3, ACID, SIDE_GROUPS, DEHK,
    empty_sites, assemble, structure_pi, thurlkill_pi, pi_from_pairs,
    sites_to_rows,
)
from piesm.dataio import DATA_DIR, seq_counts
from piesm.pka_engine import compute_pI, PKA_IPC2_PAPER

DATA = ROOT / "data"
SEQ_CSV = DATA / "benchmark" / "IPC_protein_25.csv"
REG_CSV = DATA / "benchmark" / "reg_581.csv"
PI_AF_CSV = DATA / "benchmark" / "pi_af.csv"
PI_FB_CSV = DATA / "benchmark" / "pi_fallback.csv"
OUT_DIR = ROOT / "results" / "tables"


def fnum(v):
    """Parse a numeric value, returning None for missing/invalid entries."""
    try:
        if pd.isna(v):
            return None
        s = str(v).strip()
        if s in ("", "NOTINRANGE", "NotInRange", "-", "nan", "na", "NA"):
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


def _put_side(d, pos, grp, pk):
    if pk is None or grp not in set(SIDE_GROUPS):
        return
    d["side"].setdefault(pos, {})[grp] = pk


def load_deepka(fp):
    """Parse DeepKa per-residue pKa CSV. Returns sites dict."""
    d = empty_sites()
    if not os.path.exists(fp):
        return d
    df = pd.read_csv(fp)
    for r in df.itertuples():
        # DeepKa columns: pos at index 3, res at index 4, pka at index 6
        pos = fnum(r[3]) if len(r) > 3 else None
        grp = str(r[4]).strip() if len(r) > 4 else ""
        pk = fnum(r[6]) if len(r) > 6 else None
        if pos is None or grp not in DEHK or pk is None:
            continue
        _put_side(d, int(pos), grp, pk)
    return d


def load_propka(fp, L):
    """Parse PROPKA per-residue pKa CSV. Returns sites dict."""
    d = empty_sites()
    if not os.path.exists(fp):
        return d
    df = pd.read_csv(fp)
    for r in df.itertuples():
        grp = str(r.res_name).strip()
        pos = fnum(r.res_number)
        pk = fnum(r.pka)
        if pos is None or pk is None:
            continue
        pos = int(pos)
        if grp == "NTR":
            if pos == 1 and d["ntr"] is None:
                d["ntr"] = pk
        elif grp == "CTR":
            if pos == L and d["ctr"] is None:
                d["ctr"] = pk
        elif grp in set(SIDE_GROUPS):
            _put_side(d, pos, grp, pk)
    return d


def load_pypka(fp, seq):
    """Parse PypKa per-residue pKa CSV. Returns sites dict."""
    d = empty_sites()
    L = len(seq)
    if not os.path.exists(fp):
        return d
    df = pd.read_csv(fp)
    for r in df.itertuples():
        grp = str(r.res_name).strip()
        pos = fnum(r.res_number)
        pk = fnum(r.pKa)
        if pos is None:
            continue
        pos = int(pos)
        if grp == "NTR":
            if pos == 1 and pk is not None:
                d["ntr"] = pk
            continue
        if grp == "CTR":
            if pos == L and pk is not None:
                d["ctr"] = pk
            continue
        if grp not in set(SIDE_GROUPS) or pos < 1 or pos > L:
            continue
        if AA3.get(seq[pos - 1]) != grp:
            continue
        if pk is not None:
            _put_side(d, pos, grp, pk)
    return d


def verify_frozen():
    """Report consistency between AF2 and experimental structure pI results."""
    if not PI_AF_CSV.exists():
        print("[SKIP] pi_af.csv not found")
        return
    if not PI_FB_CSV.exists():
        print("[SKIP] pi_fallback.csv not found")
        return

    af = pd.read_csv(PI_AF_CSV, encoding="utf-8-sig")
    fb = pd.read_csv(PI_FB_CSV, encoding="utf-8-sig")
    reg = pd.read_csv(REG_CSV, encoding="utf-8-sig")

    # A61 subset (proteins with high-confidence experimental structures)
    a61_mask = reg["conf_class"].astype(str).str.startswith("A")
    a61_ids = set(reg.loc[a61_mask, "seq_no"].astype(int).tolist())

    # Map AF2 columns (pi_af) to fallback columns (pi_fallback)
    col_map = {
        "M1_DeepKa_Thurl_AF": "M1_DeepKa_Thurl",
        "M2_PropKa_Thurl_AF": "M2_PropKa_Thurl",
        "M4_DeepKa_SA_Thurl_AF": "M4_DeepKa_SA_Thurl",
    }
    print(f"\n--- AF2 vs experimental structure pI (A61 subset, n={len(a61_ids)}) ---")
    for af_col, fb_col in col_map.items():
        if af_col not in af.columns or fb_col not in fb.columns:
            print(f"  [SKIP] {af_col} / {fb_col}: column not found")
            continue
        af_a61 = af[af["seq_no"].isin(a61_ids)][["seq_no", af_col]].set_index("seq_no")
        fb_a61 = fb[fb["seq_no"].isin(a61_ids)][["seq_no", fb_col]].set_index("seq_no")
        common = af_a61.index.intersection(fb_a61.index)
        a = af_a61.loc[common, af_col].values
        b = fb_a61.loc[common, fb_col].values
        mask = ~(np.isnan(a) & np.isnan(b))
        if mask.sum() == 0:
            print(f"  [SKIP] {af_col}: all NaN")
            continue
        diff = np.abs(a[mask] - b[mask])
        print(f"  {af_col:30s} vs {fb_col:25s} n={mask.sum():2d} max|diff|={diff.max():.3f}")

    # Full-set RMSE for AF2 structure methods
    print(f"\n--- AF2 structure pI RMSE (full 581 set) ---")
    for af_col, label in [("M1_DeepKa_Thurl_AF", "DeepKa+Thurl (AF2)"),
                          ("M2_PropKa_Thurl_AF", "PropKa+Thurl (AF2)"),
                          ("M8_IPC2_svr", "IPC2.svr.19")]:
        if af_col not in af.columns:
            continue
        vals = af[af_col].values
        exp = af["exp_pI"].values
        mask = ~np.isnan(vals)
        r = np.sqrt(np.mean((vals[mask] - exp[mask]) ** 2))
        print(f"  {label:25s} RMSE={r:.4f}  n={mask.sum()}")

    # A61 RMSE for experimental structure methods
    print(f"\n--- Experimental structure pI RMSE (A61 subset) ---")
    for fb_col, label in [("M1_DeepKa_Thurl", "DeepKa+Thurl (exp PDB)"),
                          ("M2_PropKa_Thurl", "PropKa+Thurl (exp PDB)")]:
        if fb_col not in fb.columns:
            continue
        fb_a61 = fb[fb["seq_no"].isin(a61_ids)]
        vals = fb_a61[fb_col].values
        exp = fb_a61["exp_pI"].values
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            continue
        r = np.sqrt(np.mean((vals[mask] - exp[mask]) ** 2))
        print(f"  {label:30s} RMSE={r:.4f}  n={mask.sum()}")

    print("\n[Frozen] Structure pI results are stored in pi_af.csv (AF2) and pi_fallback.csv (exp PDB).")
    print("  These use different structure sources; differences are expected.")
    print("  To recalculate from raw per-residue pKa, provide --pka-dir argument.")


def main():
    parser = argparse.ArgumentParser(description="Structure-based pI calculation")
    parser.add_argument("--pka-dir", default=None,
                        help="Directory containing per-residue pKa files")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify frozen results, skip calculation")
    args = parser.parse_args()

    if args.verify_only or args.pka_dir is None:
        print("=" * 60)
        print("Structure pI: verification mode (no raw pKa files provided)")
        print("=" * 60)
        verify_frozen()
        print("\nTo calculate structure pI from raw per-residue pKa files:")
        print("  python scripts/10_structure_pI.py --pka-dir data/pka_af2")
        print("\nExpected layout:")
        print("  data/pka_af2/deepka/{seq_no}_deepka.csv")
        print("  data/pka_af2/propka/{seq_no}_propka.csv")
        print("  data/pka_af2/pypka/{seq_no}_pKa.csv")
        return

    # Load sequences
    seq_df = pd.read_csv(SEQ_CSV, encoding="utf-8-sig")
    seqs = {i + 1: str(s).strip().upper() for i, s in enumerate(seq_df["sequence"])}
    exp = {i + 1: float(v) for i, v in enumerate(seq_df["exp_pI"])}
    reg = pd.read_csv(REG_CSV, encoding="utf-8-sig")

    pka_base = pathlib.Path(args.pka_dir)
    ids581 = sorted(seqs)

    # A61 subset
    a61_mask = reg["conf_class"].astype(str).str.startswith("A")
    ids61 = sorted(int(x) for x in reg.loc[a61_mask, "seq_no"])

    print("=" * 60)
    print(f"Structure pI calculation from {pka_base}")
    print(f"  Total sequences: {len(ids581)}")
    print(f"  A61 subset:      {len(ids61)}")
    print("=" * 60)

    # Load per-residue pKa for each protein
    sites = {"DeepKa": {}, "PropKa": {}, "PypKa": {}}
    found = {"DeepKa": 0, "PropKa": 0, "PypKa": 0}

    for sn in ids581:
        L = len(seqs[sn])
        dk = load_deepka(pka_base / "deepka" / f"{sn}_deepka.csv")
        pk = load_propka(pka_base / "propka" / f"{sn}_propka.csv", L)
        pp = load_pypka(pka_base / "pypka" / f"{sn}_pKa.csv", seqs[sn])
        sites["DeepKa"][sn] = dk
        sites["PropKa"][sn] = pk
        sites["PypKa"][sn] = pp
        if dk["side"] or dk["ntr"] or dk["ctr"]:
            found["DeepKa"] += 1
        if pk["side"] or pk["ntr"] or pk["ctr"]:
            found["PropKa"] += 1
        if pp["side"] or pp["ntr"] or pp["ctr"]:
            found["PypKa"] += 1

    print(f"\nPer-residue pKa files found:")
    for m, n in found.items():
        print(f"  {m}: {n}/{len(ids581)}")

    if sum(found.values()) == 0:
        print("\n[WARN] No per-residue pKa files found. Verifying frozen results only.")
        verify_frozen()
        return

    # Calculate structure pI
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for sn in ids581:
        rows.append(dict(seq_no=sn, exp_pI=exp[sn], length=len(seqs[sn]),
                         method="Thurlkill",
                         pI_FULL=round(thurlkill_pi(seqs[sn]), 6),
                         pI_DEHK=round(thurlkill_pi(seqs[sn]), 6)))
        for name in ("DeepKa", "PropKa", "PypKa"):
            s = sites[name][sn]
            rows.append(dict(seq_no=sn, exp_pI=exp[sn], length=len(seqs[sn]),
                             method=name,
                             pI_FULL=round(structure_pi(seqs[sn], s, "FULL"), 6),
                             pI_DEHK=round(structure_pi(seqs[sn], s, "DEHK"), 6)))

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "structure_pI_results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nResults saved: {out_csv}")
    print(f"  Shape: {df.shape}")

    # RMSE summary
    print("\n--- RMSE summary ---")
    for method in ("Thurlkill", "DeepKa", "PropKa", "PypKa"):
        sub = df[df["method"] == method]
        r = np.sqrt(np.mean((sub["pI_FULL"] - sub["exp_pI"]) ** 2))
        print(f"  {method:12s} FULL: RMSE={r:.4f}  (n={len(sub)})")

    # Verify against frozen results
    print("\n--- Verifying against frozen pi_af.csv ---")
    if PI_AF_CSV.exists():
        af = pd.read_csv(PI_AF_CSV, encoding="utf-8-sig")
        for method, col in [("DeepKa", "M1_DeepKa_Thurl_AF"), ("PropKa", "M2_PropKa_Thurl_AF")]:
            calc = df[df["method"] == method].set_index("seq_no")["pI_FULL"]
            if col in af.columns:
                fr = af.set_index("seq_no")[col]
                common = calc.index.intersection(fr.index)
                diff = (calc.loc[common] - fr.loc[common]).abs()
                n_bad = (diff > 0.01).sum()
                status = "OK" if n_bad == 0 else f"*** {n_bad} mismatches ***"
                print(f"  {method} vs {col}: max|diff|={diff.max():.4f}  {status}")
    else:
        print("  pi_af.csv not found; skipping verification")


if __name__ == "__main__":
    main()
