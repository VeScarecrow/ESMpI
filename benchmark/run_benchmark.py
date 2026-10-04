# -*- coding: utf-8 -*-
"""
run_benchmark: end-to-end reproduction of the published ESMpI model metrics.

This is a self-contained Level-2 (full-retrain) benchmark. It reads only the
files shipped under benchmark/inputs/, rebuilds the published paper model from
scratch, and reproduces both headline numbers:

  * training set (n=1743): 5-fold out-of-fold (OOF) RMSE, main seed 71923
  * independent test set (n=581): RMSE / MAE / bias / R^2 / Pearson r /
    outlier count, evaluated exactly once

Published paper model (640-dim ESM-only configuration):
  * physical baseline: IPC2 published 9-pKa values, H-H bisection pI
  * encoder: ESM-2 150M (esm2_t30_150M_UR50D) frozen per-residue embeddings,
    pre-extracted per ionizable-site type (7 x 640, float16 on disk)
  * pooling: 7-type weighted mean with
    W = [0.0156, 0.0156, 0.0062, 0.5, 0.05, 0.0062, 1.0] (D/E/H/K/C/Y/R)
  * regressor: StandardScaler + RBF-SVR (C=0.5, gamma='scale', epsilon=0.05)
  * training target: residual r = exp_pI - IPC2_baseline_pI
  * sample weights: w = 1 + max(abs(y) - 9, 0)
  * CV: 5-fold, five seeds [71923, 42, 123, 777, 2024]; the frozen anchor
    stores the main-seed (71923) OOF predictions

Expected results (see expected_outputs/expected_metrics.csv):
  train OOF RMSE = 0.8002 ; test RMSE = 0.8104, MAE = 0.5573, R^2 = 0.6478

Usage:
  python benchmark/run_benchmark.py            # run from the repo root
  python run_benchmark.py                      # or from inside benchmark/

Runtime: ~1-2 min on a 6-core CPU (26 SVR fits on 640-dim features).
No GPU, network access, torch, or transformers are required.

Outputs (written to benchmark/outputs/):
  test_predictions.csv       per-protein test predictions (581 rows)
  train_oof_predictions.csv  per-protein training OOF predictions (1743 rows)
  benchmark_metrics.csv      reproduced metric table
  run_info.txt               versions, timing, and per-seed CV values

The script exits with code 1 if any frozen-anchor check fails.
"""
from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

# Make the esmpi package importable (repo root is one level above benchmark/)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from esmpi import models  # noqa: E402
from esmpi.pooling import W_ELEGANT  # noqa: E402  (7-type pooling weights)

INP = os.path.join(HERE, "inputs")
EXP = os.path.join(HERE, "expected_outputs")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")

# Tolerances for the frozen-anchor checks
PRED_ATOL = 1e-4          # max abs diff of per-protein pI predictions
METRIC_ATOL = 5e-4        # abs diff of rounded 4-decimal published metrics
N_TRAIN, N_TEST = 1743, 581

check_failures = []


def check(name, ok, detail=""):
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        check_failures.append(name)


def metric_row(y, pred):
    """Full metric set matching the paper tables (4-decimal rounding)."""
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    e = pred - y
    r = float(stats.pearsonr(pred, y)[0])
    return {
        "n": len(e),
        "RMSE": round(float(np.sqrt((e ** 2).mean())), 4),
        "MAE": round(float(np.abs(e).mean()), 4),
        "bias": round(float(e.mean()), 4),
        "R2_coef_det": round(float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4),
        "Pearson_r": round(r, 4),
        "r2": round(r ** 2, 4),
        "n_abs_err_gt_0p5": int((np.abs(e) > 0.5).sum()),
    }


def load_pooled(npz_path, keys_expected):
    """Load per-type ESM-2 embeddings and apply the published 7-type pooling.

    Mirrors esmpi.pooling.load_emb_pool but reads from benchmark/inputs/.
    """
    dd = np.load(npz_path, allow_pickle=True)
    keys = [str(k) for k in dd["keys"]]
    assert keys == keys_expected, (
        f"key alignment failed for {os.path.basename(npz_path)}: "
        "embedding row order does not match z81 feature table")
    pt = dd["site_per_type"].astype(np.float64)   # (n, 7, 640), float16 on disk
    cn = dd["site_counts"].astype(np.float64)     # (n, 7)
    wc = cn * W_ELEGANT[None, :]
    v = np.einsum("nk,nkh->nh", wc, pt) / np.clip(wc.sum(1, keepdims=True), 1e-9, None)
    return v.astype(np.float32)


def main():
    t0 = time.time()
    print("=" * 88)
    print("ESMpI Level-2 benchmark: full 5-fold OOF training + one-time test evaluation")
    print("=" * 88)

    # ---------- Load inputs (copied under benchmark/inputs/) ----------
    ftr = pd.read_csv(os.path.join(INP, "z81_features_train.csv"))
    fte = pd.read_csv(os.path.join(INP, "z81_features_test.csv"))
    ipc_tr = pd.read_csv(os.path.join(INP, "IPC_protein_75.csv"))
    ipc_te = pd.read_csv(os.path.join(INP, "IPC_protein_25.csv"))

    check("train sample count", len(ftr) == N_TRAIN, f"n={len(ftr)} (expected {N_TRAIN})")
    check("test sample count", len(fte) == N_TEST, f"n={len(fte)} (expected {N_TEST})")
    check("IPC_protein_75 row count", len(ipc_tr) == N_TRAIN, f"n={len(ipc_tr)}")
    check("IPC_protein_25 row count", len(ipc_te) == N_TEST, f"n={len(ipc_te)}")
    # z81 row order differs from the IPC CSV row order; compare as multisets
    check("train exp_pI consistent with IPC_protein_75",
          np.allclose(np.sort(ftr["exp_pI"].values), np.sort(ipc_tr["exp_pI"].values)))
    check("test exp_pI consistent with IPC_protein_25",
          np.allclose(np.sort(fte["exp_pI"].values), np.sort(ipc_te["exp_pI"].values)))

    ytr = ftr["exp_pI"].values.astype(np.float64)
    yte = fte["exp_pI"].values.astype(np.float64)
    bo = ftr["pI_IPC2_protein"].values.astype(np.float64)   # IPC2 baseline (train)
    bt = fte["pI_IPC2_protein"].values.astype(np.float64)   # IPC2 baseline (test)
    rtr = ytr - bo
    sw = 1.0 + np.maximum(np.abs(ytr) - 9.0, 0.0)           # tail-emphasis weights

    # ---------- ESM-2 150M 7-type weighted pooling ----------
    print("\nLoading pre-extracted ESM-2 150M embeddings ...")
    Xtr = load_pooled(os.path.join(INP, "z82_150m_pertype_train.npz"),
                      ftr["key"].astype(str).tolist())
    Xte = load_pooled(os.path.join(INP, "z82_150m_pertype_test.npz"),
                      fte["key"].astype(str).tolist())
    check("pooled train feature shape", Xtr.shape == (N_TRAIN, 640), str(Xtr.shape))
    check("pooled test feature shape", Xte.shape == (N_TEST, 640), str(Xte.shape))

    # ---------- 5-fold OOF (5 seeds) + full retrain ----------
    print("\nRunning 5-fold OOF CV across 5 seeds (25 SVR fits) ...")
    cvs, oof_main = models.cv5(models.make_esm_svr, Xtr, rtr, bo, ytr, sw)
    oof_pI = bo + oof_main
    print("Per-seed OOF RMSE: "
          + ", ".join(f"{v:.4f}" for v in cvs)
          + f"  (mean {np.mean(cvs):.4f})")

    print("Retraining on all 1743 proteins and predicting the 581 test proteins ...")
    pt_ours, _ = models.full_test(models.make_esm_svr, Xtr, rtr, sw, Xte, bt)

    # ---------- Reproduced metrics ----------
    m_oof = metric_row(ytr, oof_pI)
    m_test = metric_row(yte, pt_ours)
    print("\n" + "-" * 88)
    print(f"Train OOF (n={m_oof['n']}): RMSE={m_oof['RMSE']}")
    print(f"Test      (n={m_test['n']}): "
          f"RMSE={m_test['RMSE']}, MAE={m_test['MAE']}, bias={m_test['bias']}, "
          f"R2={m_test['R2_coef_det']}, Pearson r={m_test['Pearson_r']}, "
          f"outliers(|err|>0.5)={m_test['n_abs_err_gt_0p5']}")
    print("-" * 88)

    # ---------- Frozen-anchor verification ----------
    print("\nVerifying against frozen anchor expected_outputs/z166_preds.npz ...")
    anchor = np.load(os.path.join(EXP, "z166_preds.npz"))
    # Row-order proof: labels and baselines must agree exactly first
    check("anchor yte matches local test labels",
          np.allclose(anchor["yte"], yte, atol=0.0),
          f"maxdiff={np.max(np.abs(anchor['yte'] - yte)):.2e}")
    check("anchor ytr matches local train labels",
          np.allclose(anchor["ytr"], ytr, atol=0.0),
          f"maxdiff={np.max(np.abs(anchor['ytr'] - ytr)):.2e}")
    check("anchor baselines match local IPC2 pI",
          np.allclose(anchor["bo"], bo, atol=0.0)
          and np.allclose(anchor["bt_ipc2"], bt, atol=0.0))
    d_pt = float(np.max(np.abs(anchor["pt_ours"] - pt_ours)))
    d_oof = float(np.max(np.abs(anchor["oof_ours"] - oof_pI)))
    check("test predictions reproduce frozen pt_ours", d_pt <= PRED_ATOL,
          f"maxdiff={d_pt:.2e} (tol {PRED_ATOL:g})")
    check("OOF predictions reproduce frozen oof_ours", d_oof <= PRED_ATOL,
          f"maxdiff={d_oof:.2e} (tol {PRED_ATOL:g})")

    # Published rounded metric table
    exp = pd.read_csv(os.path.join(EXP, "expected_metrics.csv"))
    exp_oof = float(exp.loc[exp["split"] == "train_OOF_5fold", "RMSE"].iloc[0])
    exp_test = exp[exp["split"] == "test"].iloc[0]
    check("train OOF RMSE matches published value",
          abs(m_oof["RMSE"] - exp_oof) <= METRIC_ATOL,
          f"reproduced {m_oof['RMSE']} vs expected {exp_oof:.4f}")
    for col in ["RMSE", "MAE", "bias", "R2_coef_det", "Pearson_r", "r2",
                "n_abs_err_gt_0p5"]:
        got = m_test[col]
        want = float(exp_test[col])
        ok = abs(got - want) <= METRIC_ATOL
        check(f"test {col} matches published value", ok,
              f"reproduced {got} vs expected {want:g}")

    # ---------- Write outputs ----------
    pd.DataFrame([
        dict(split="train_OOF_5fold", **m_oof),
        dict(split="test", **m_test),
    ]).to_csv(os.path.join(OUT, "benchmark_metrics.csv"),
              index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "sequence": fte["key"].astype(str).values,
        "exp_pI": yte,
        "pI_IPC2_baseline": np.round(bt, 6),
        "pI_ESMpI": np.round(pt_ours, 6),
        "abs_error": np.round(np.abs(pt_ours - yte), 6),
    }).to_csv(os.path.join(OUT, "test_predictions.csv"),
              index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "sequence": ftr["key"].astype(str).values,
        "exp_pI": ytr,
        "pI_IPC2_baseline": np.round(bo, 6),
        "oof_pI": np.round(oof_pI, 6),
        "abs_error": np.round(np.abs(oof_pI - ytr), 6),
    }).to_csv(os.path.join(OUT, "train_oof_predictions.csv"),
              index=False, encoding="utf-8-sig")

    elapsed = time.time() - t0
    with open(os.path.join(OUT, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write("ESMpI benchmark run info\n")
        f.write(f"python: {platform.python_version()} ({sys.executable})\n")
        import scipy
        f.write(f"numpy: {np.__version__}, pandas: {pd.__version__}, "
                f"scipy: {scipy.__version__}\n")
        import sklearn
        f.write(f"scikit-learn: {sklearn.__version__}\n")
        f.write(f"platform: {platform.platform()}\n")
        f.write(f"elapsed_seconds: {elapsed:.1f}\n")
        f.write(f"cv_seeds: {models.SEEDS}\n")
        f.write("per_seed_oof_rmse: "
                + ",".join(f"{v:.6f}" for v in cvs) + "\n")
        f.write(f"main_seed_oof_rmse: {m_oof['RMSE']}\n")
        f.write(f"test_rmse: {m_test['RMSE']}\n")
        f.write(f"anchor_maxdiff_pt_ours: {d_pt:.3e}\n")
        f.write(f"anchor_maxdiff_oof_ours: {d_oof:.3e}\n")

    print(f"\nOutputs written to {OUT}")
    print(f"Total runtime: {elapsed:.1f} s")
    print("=" * 88)
    if check_failures:
        print(f"BENCHMARK FAILED: {len(check_failures)} check(s) did not pass:")
        for nm in check_failures:
            print(f"  - {nm}")
        print("=" * 88)
        sys.exit(1)
    print("ALL CHECKS PASSED -- reproduced metrics match the frozen paper anchor.")
    print("=" * 88)


if __name__ == "__main__":
    main()
