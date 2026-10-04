# ESMpI Benchmark

Self-contained, one-command reproduction of the **published ESMpI model metrics**
(paper version, 640-dim ESM-only configuration).

| Split | n | RMSE | MAE | R² |
|-------|---|------|-----|-----|
| Training, 5-fold OOF (seed 71923) | 1743 | 0.8002 | — | — |
| Independent test (evaluated once) | 581 | **0.8104** | 0.5573 | 0.6478 |

Additional test-set metrics: bias = −0.0282, Pearson r = 0.8052,
outliers (|error| > 0.5 pH) = 226/581.

The script rebuilds the model **from scratch** — five-seed 5-fold OOF on the
1743 training proteins, one full retrain, and a single evaluation on the 581
test proteins — and then verifies every reproduced prediction against a frozen
anchor. No GPU, network access, `torch`, or `transformers` are required; only
numpy, pandas, scipy, and scikit-learn (see the repository `requirements.txt`).

## Run

```bash
# from the repository root
python benchmark/run_benchmark.py
```

Runtime: ~20 s on a 6-core CPU (26 RBF-SVR fits on 640-dim features).
A successful run ends with:

```
ALL CHECKS PASSED -- reproduced metrics match the frozen paper anchor.
```

Any mismatch prints `[FAIL]` and the script exits with code 1.

## What is verified

1. Sample counts (1743 train / 581 test) and label consistency against the
   original IPC2 `IPC_protein_75.csv` / `IPC_protein_25.csv` tables.
2. Row-order alignment between embedding files and feature tables
   (sequence keys compared exactly).
3. Reproduced test and OOF predictions are bit-for-bit identical
   (tolerance 1e-4) to the frozen anchor `expected_outputs/z166_preds.npz`,
   which is the same file shipped as `data/predictions/z166_preds.npz`.
4. All published rounded metrics match `expected_outputs/expected_metrics.csv`
   within 5e-4.

## Model specification

| Component | Value |
|-----------|-------|
| Physical baseline | IPC2 published 9-pKa values, Henderson–Hasselbalch bisection |
| Encoder | ESM-2 150M (`esm2_t30_150M_UR50D`), frozen, 640-dim per-residue embeddings |
| Pooling | 7-type (D/E/H/K/C/Y/R) weighted mean, W = [0.0156, 0.0156, 0.0062, 0.5, 0.05, 0.0062, 1.0] |
| Regressor | StandardScaler + RBF-SVR (C = 0.5, gamma = 'scale', epsilon = 0.05) |
| Training target | residual r = exp_pI − IPC2 baseline pI |
| Sample weights | w = 1 + max(|y| − 9, 0) |
| CV protocol | 5-fold, seeds [71923, 42, 123, 777, 2024] |

Per-seed OOF RMSE on a reference machine:
0.8002, 0.7964, 0.8006, 0.8006, 0.8000 (mean 0.7996).

## Folder layout

```
benchmark/
├── run_benchmark.py          # Level-2 full-retrain benchmark (this is the entry point)
├── README.md                 # this file
├── inputs/                   # self-contained input copies (no dependency on ../data)
│   ├── IPC_protein_75.csv          # 1743 training proteins (original IPC2 split)
│   ├── IPC_protein_25.csv          # 581 test proteins (original IPC2 split)
│   ├── z81_features_train.csv      # train table: sequences, exp_pI, IPC2 baseline pI
│   ├── z81_features_test.csv       # test table (same columns)
│   ├── z82_150m_pertype_train.npz  # ESM-2 150M per-type embeddings, 1743×7×640 (Git LFS)
│   └── z82_150m_pertype_test.npz   # ESM-2 150M per-type embeddings, 581×7×640 (Git LFS)
├── expected_outputs/         # frozen references, committed
│   ├── z166_preds.npz              # frozen OOF/test predictions (regular Git blob)
│   └── expected_metrics.csv        # published headline metrics
└── outputs/                  # regenerated on every run (git-ignored)
    ├── benchmark_metrics.csv       # reproduced metric table
    ├── test_predictions.csv        # 581 per-protein test predictions
    ├── train_oof_predictions.csv   # 1743 per-protein OOF predictions
    └── run_info.txt                # package versions, timing, per-seed CV values
```

Embeddings are stored with Git LFS; clone with Git LFS enabled
(see the main repository README). If the `.npz` files are small text pointers,
run `git lfs pull`.

## Relation to the other reproduction scripts

* `scripts/01_generate_predictions.py` — wider analysis pipeline: also
  regenerates `z166_preds.npz` plus baseline-swap ablations and comparison
  figures (~12 min).
* `scripts/02_reproduce_benchmark.py` — Level-1 table reproduction that
  consumes precomputed predictions and adds comparisons against 11 other
  methods (structure-based and sequence) on the 581 and A61 sets.
* `benchmark/run_benchmark.py` — the minimal, self-contained, anchored
  confirmation that **our model's own numbers** reproduce from raw inputs.
