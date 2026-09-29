# -*- coding: utf-8 -*-
"""
predict: predict isoelectric point (pI) for user-provided protein sequences.

Two modes:
  1. Full mode (requires GPU + transformers + fair-esm): extracts ESM-2 150M
     embeddings, applies 7-type weighted pooling, and predicts with the
     published pI-ESM residual SVR model.
  2. Fallback mode (no GPU): computes only the IPC2 9-pKa physical baseline
     via the Henderson-Hasselbalch bisection engine. No ESM embedding needed.

Usage:
  python predict.py --input sequences.fasta [--output predictions.csv] [--device cuda|cpu]
  python predict.py --sequence "MKKFF..." [--device cpu]

Input:  FASTA file (one or more sequences) or a single sequence string.
Output: CSV with columns [id, sequence, length, pI_IPC2_baseline, pI_ESM (if available)].
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from piesm import dataio, pka_engine

# IPC2 published 9-pKa values (used as physical baseline)
PKA_IPC2 = pka_engine.PKA_IPC2_PAPER

# GBMS re-optimized 9-pKa (SI Table S1)
PKA_GBMS = np.array([5.570, 6.165, 3.804, 4.495, 5.481, 9.230, 7.904, 11.217, 10.658])


def parse_fasta(path):
    """Parse a FASTA file. Returns list of (id, sequence)."""
    records = []
    seq_id, seq_parts = None, []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq_id is not None:
                    records.append((seq_id, ''.join(seq_parts).upper()))
                seq_id = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if seq_id is not None:
        records.append((seq_id, ''.join(seq_parts).upper()))
    return records


def predict_baseline(sequences):
    """Compute IPC2 9-pKa baseline pI for a list of sequences."""
    counts = np.array([dataio.seq_counts(s) for s in sequences])
    pI_ipc2 = pka_engine.compute_pI(PKA_IPC2, counts)
    pI_gbms = pka_engine.compute_pI(PKA_GBMS, counts)
    return pI_ipc2, pI_gbms


def predict_esm(sequences, device='auto'):
    """Full pI-ESM prediction: ESM-2 embedding + SVR.

    Requires: torch, transformers. Falls back to baseline-only
    if any dependency is missing or the model cannot be downloaded.
    Set the ESM2_MODEL_PATH environment variable to a local ESM-2 150M
    directory to avoid network downloads.
    """
    try:
        import torch
        from transformers import AutoTokenizer, EsmModel
    except ImportError:
        print("WARNING: torch/transformers/fair-esm not found. "
              "Running in baseline-only mode (no ESM correction).")
        return None

    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load ESM-2 150M (try local cache first, then HuggingFace)
    model_name = os.environ.get(
        'ESM2_MODEL_PATH', 'facebook/esm2_t30_150M_UR50D')
    print(f"Loading {model_name} ...")
    # Fail fast when offline (avoid 5x retry delays)
    os.environ.setdefault('HF_HUB_OFFLINE', '0')
    try:
        import requests
        requests.head('https://huggingface.co', timeout=5)
    except Exception:
        print("WARNING: Cannot reach huggingface.co. "
              "Falling back to baseline-only mode. "
              "Set ESM2_MODEL_PATH to a local model directory to enable full prediction.")
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)
    except Exception as e:
        print(f"WARNING: Cannot load ESM-2 model ({e}). "
              "Falling back to baseline-only mode. "
              "Set ESM2_MODEL_PATH to a local model directory to enable full prediction.")
        return None
    model = model.to(device).eval()

    # Extract per-residue embeddings (last hidden state)
    embeddings = []
    for i, seq in enumerate(sequences):
        inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        # Remove BOS/EOS tokens, keep per-residue embeddings
        emb = out.last_hidden_state[0, 1:-1, :].cpu().numpy()  # (L, 640)
        embeddings.append(emb)
        if (i + 1) % 50 == 0:
            print(f"  embedded {i+1}/{len(sequences)} sequences", flush=True)

    # 7-type weighted pooling (matches piesm.pooling)
    from piesm.pooling import W_ELEGANT
    SITE7 = ['D', 'E', 'H', 'K', 'C', 'Y', 'R']
    H = embeddings[0].shape[1]  # 640

    pooled = np.zeros((len(sequences), H), dtype=np.float32)
    for i, (seq, emb) in enumerate(zip(sequences, embeddings)):
        # Per-type mean embedding
        pt = np.zeros((7, H), dtype=np.float64)
        cn = np.zeros(7, dtype=np.float64)
        for j, res in enumerate(SITE7):
            mask = np.array([c == res for c in seq])
            if mask.sum() > 0:
                pt[j] = emb[mask].mean(axis=0)
                cn[j] = mask.sum()
        wc = cn * W_ELEGANT[None, :]
        total = wc.sum()
        if total > 0:
            pooled[i] = (np.einsum('k,kh->h', wc[0], pt) / total).astype(np.float32)

    # Load pre-computed training embeddings and train SVR on the fly
    from piesm import pooling, models
    print("Loading training embeddings for SVR retraining ...")
    Xtr, _ = pooling.load_emb_pool('ESM2-150M', pooling.W_ELEGANT)
    d = dataio.load_data()
    ytr = d['ytr']
    sw = d['sw']
    bo = d['ftr']['pI_IPC2_protein'].values.astype(np.float64)
    rtr = ytr - bo

    est = models.make_esm_svr()
    models.fit_weighted(est, Xtr, rtr, sw)

    # Baseline for new sequences
    _, pI_gbms = predict_baseline(sequences)
    correction = est.predict(pooled)
    pI_esm = pI_gbms + correction
    return pI_esm


def main():
    parser = argparse.ArgumentParser(
        description='Predict isoelectric point (pI) for protein sequences.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=str,
                       help='Path to a FASTA file with one or more sequences.')
    group.add_argument('--sequence', '-s', type=str,
                       help='A single protein sequence (amino acids, no header).')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV path (default: stdout).')
    parser.add_argument('--device', '-d', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device for ESM-2 inference (default: auto).')
    parser.add_argument('--id', type=str, default='query',
                        help='Sequence ID when using --sequence (default: query).')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Skip ESM embedding; only compute IPC2/GBMS baseline pI.')
    args = parser.parse_args()

    # Parse input
    if args.input:
        records = parse_fasta(args.input)
        print(f"Loaded {len(records)} sequence(s) from {args.input}")
    else:
        records = [(args.id, args.sequence.upper())]

    sequences = [r[1] for r in records]
    ids = [r[0] for r in records]

    # Validate sequences
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    for sid, seq in records:
        invalid = set(seq) - valid_aa
        if invalid:
            print(f"WARNING: {sid} contains non-standard residues: {invalid}. "
                  "These are ignored in pKa counting.")

    # Baseline prediction (always available)
    t0 = time.time()
    pI_ipc2, pI_gbms = predict_baseline(sequences)
    print(f"Baseline pI computed ({time.time()-t0:.1f}s)")

    # Full pI-ESM prediction
    pI_esm = None
    if not args.baseline_only:
        pI_esm = predict_esm(sequences, device=args.device)

    # Build output
    out = pd.DataFrame({
        'id': ids,
        'sequence': sequences,
        'length': [len(s) for s in sequences],
        'pI_IPC2_baseline': np.round(pI_ipc2, 4),
        'pI_GBMS_baseline': np.round(pI_gbms, 4),
    })
    if pI_esm is not None:
        out['pI_ESM'] = np.round(pI_esm, 4)

    if args.output:
        out.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"Saved to {args.output}")
    else:
        print(out.to_string(index=False))


if __name__ == '__main__':
    main()
