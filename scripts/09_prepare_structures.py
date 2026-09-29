# -*- coding: utf-8 -*-
"""09: Download and prepare experimental PDB structures.

Reads reg_581.csv to identify which proteins have experimental PDB entries,
downloads them from RCSB, selects the correct chain, repairs missing residues,
and classifies by confidence level (A/B/C/D/P).

Requires:
  - requests (pip install requests)
  - PDBFixer (pip install pdbfixer)  [optional; if absent, raw PDB is kept]

Usage:
  python scripts/09_prepare_structures.py
  python scripts/09_prepare_structures.py --skip-download  # use existing PDBs
"""
import argparse
import os
import sys
import pathlib

sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from piesm.structure import THURL, AA3, ACID, SIDE_GROUPS, DEHK

DATA = ROOT / "data"
REG_CSV = DATA / "benchmark" / "reg_581.csv"
OUT_DIR = DATA / "structures" / "experimental_pdb"

# RCSB PDB download URL (text format)
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download_pdb(pdb_id, out_path):
    """Download a single PDB file from RCSB."""
    import requests
    url = RCSB_URL.format(pdb_id=pdb_id)
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"  [WARN] {pdb_id}: HTTP {resp.status_code}")
        return False
    out_path.write_bytes(resp.content)
    return True


def select_chain(pdb_text, chain_id):
    """Extract a single chain from PDB text (ATOM/HETATM records only)."""
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM", "TER", "END")):
            rec_chain = line[21:22].strip()
            if rec_chain == chain_id or line.startswith(("END", "TER")):
                lines.append(line)
    return "\n".join(lines) + "\n"


def repair_pdb(pdb_path):
    """Repair missing residues using PDBFixer (if available)."""
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        return False  # PDBFixer not installed; skip repair

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    with open(pdb_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return True


def classify_resolution(resolution):
    """Classify PDB by X-ray resolution."""
    if pd.isna(resolution) or resolution is None:
        return "P-predicted_no_exp_PDB"
    r = float(resolution)
    if r < 3.0:
        return "A-high_confidence_exp"
    elif r < 3.5:
        return "B-trusted_exp_annotated"
    elif r < 4.0:
        return "C-limited_exp_caution"
    else:
        return "D-exp_structure_rejected"


def main():
    parser = argparse.ArgumentParser(description="Download and prepare experimental PDB structures")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only repair existing PDBs")
    args = parser.parse_args()

    reg = pd.read_csv(REG_CSV, encoding="utf-8-sig")
    exp_proteins = reg[reg["exp_pdb_id"].notna()].copy()
    n_exp = len(exp_proteins)
    n_total = len(reg)
    n_pred = n_total - n_exp

    print(f"Total proteins: {n_total}")
    print(f"  With experimental PDB: {n_exp}")
    print(f"  Predicted only (AF2):  {n_pred}")

    # Confidence class distribution
    print("\nConfidence class distribution:")
    for cls, count in reg["conf_class"].value_counts().sort_index().items():
        print(f"  {cls}: {count}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print(f"\nDownloading {n_exp} PDB files from RCSB...")
        try:
            import requests
        except ImportError:
            print("[ERROR] requests not installed. Run: pip install requests")
            return

        ok, fail = 0, 0
        for _, row in exp_proteins.iterrows():
            sn = int(row["seq_no"])
            pdb_id = str(row["exp_pdb_id"]).strip()
            chain = str(row["final_chain"]).strip() if pd.notna(row["final_chain"]) else "A"
            out_file = OUT_DIR / f"{sn}.pdb"

            if out_file.exists():
                ok += 1
                continue

            print(f"  seq_no={sn:3d}  PDB={pdb_id}  chain={chain}", end="  ")
            try:
                raw_text = requests.get(RCSB_URL.format(pdb_id=pdb_id), timeout=30).text
                if "<html" in raw_text[:100].lower():
                    print("[FAIL: not a PDB file]")
                    fail += 1
                    continue
                # Select chain
                repaired = select_chain(raw_text, chain)
                out_file.write_text(repaired, encoding="utf-8")
                print("[OK]")
                ok += 1
            except Exception as e:
                print(f"[ERROR: {e}]")
                fail += 1

        print(f"\nDownload complete: {ok} OK, {fail} failed")
    else:
        print("\n[SKIP] Download skipped (--skip-download)")

    # Repair pass
    print("\nRepairing missing residues (PDBFixer)...")
    try:
        from pdbfixer import PDBFixer  # noqa: F401
    except ImportError:
        print("[INFO] PDBFixer not installed; raw PDBs kept as-is.")
        print("       To enable repair: pip install pdbfixer")
    else:
        repaired, skipped = 0, 0
        for pdb_file in sorted(OUT_DIR.glob("*.pdb")):
            try:
                repair_pdb(pdb_file)
                repaired += 1
            except Exception as e:
                print(f"  [WARN] {pdb_file.name}: {e}")
                skipped += 1
        print(f"Repaired: {repaired}, skipped: {skipped}")

    # Summary
    print("\n--- Structure preparation summary ---")
    print(f"Output directory: {OUT_DIR}")
    n_files = len(list(OUT_DIR.glob("*.pdb")))
    print(f"PDB files: {n_files}")
    print(f"Missing (no exp PDB): {n_pred} — use AlphaFold2 structures instead")
    print(f"\nConfidence classes (see reg_581.csv 'conf_class' column):")
    print("  A — high confidence  : X-ray < 3.0 A, complete chain, no gaps")
    print("  B — trusted           : X-ray/NMR, minor gaps or lower resolution")
    print("  C — limited           : significant gaps, low resolution")
    print("  D — rejected          : severe quality issues")
    print("  P — predicted only    : no experimental PDB (AF2 used)")


if __name__ == "__main__":
    main()
