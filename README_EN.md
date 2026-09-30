# ESMpI

Protein language model-boosted prediction of isoelectric point (pI). Combines a
physics-based IPC2 nine-pKa baseline with an ESM-2 150M residual SVR correction.
Sequence-only — no 3D structure required.

**Headline results** (581 independent test proteins):

| Method | RMSE | MAE | R² |
|--------|------|-----|-----|
| ESMpI (this work) | **0.8104** | 0.5573 | 0.6478 |
| IPC2 SVR | 0.8552 | 0.5907 | 0.6077 |

On the 61-protein high-confidence experimental-structure subset, ESMpI
(RMSE 0.6064) ties with the best structure-based method DeepKa (0.6058).

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| **Encoder** | ESM-2 150M (`esm2_t30_150M_UR50D`), frozen weights, 640-dim per-residue embeddings |
| **Pooling** | 7-type weighted average (D/E/H/K/C/Y/R), weights = `[0.0156, 0.0156, 0.0062, 0.5, 0.05, 0.0062, 1.0]` |
| **Regressor** | RBF-SVR (C=0.5, epsilon=0.05, gamma=scale), with StandardScaler |
| **Training target** | Residual r = exp_pI − physical_baseline_pI |
| **Sample weighting** | w = 1 + max(abs(y) − 9, 0) to emphasize extreme-pI proteins |
| **Cross-validation** | 5-fold, 5 random seeds; OOF predictions for model selection |
| **Physical baseline** | IPC2 published 9-pKa values, H-H bisection (tol 1e-8) |

### GBMS pKa Optimization

A key contribution of this work is the re-optimization of the 9-pKa vector
on the 1743-protein training set. The protocol (termed **GBMS**) uses the
same objective function as IPC2 (sum of squared pI residuals) but replaces
the original optimizer with a modern local solver:

| Aspect | IPC2 (2021) | GBMS (this work) |
|--------|-------------|-------------------|
| Optimizer | Differential Evolution | Multi-start Trust Region Reflective (TRF) |
| Starts | Population-based (default 10×9=90 individuals/generation) | 13 starts (Thurlkill + IPC1 + IPC2 + 10 LHS) |
| Bounds | [0, 14] | [0, 14] |
| Train RMSE | 0.8255 | **0.8253** |
| Test RMSE | 0.8660 | **0.8674** |
| Measured speed (1743 proteins) | 50 generations × 90 individuals ≈ **95 s** | 13 starts ≈ **13 s**; single start ≈ **0.4 s** |

*Benchmark environment: Intel Core i5-9400F (6 cores, 2.90 GHz), 16 GB RAM, Windows 10, Python 3.13.*

The GBMS pKa values are nearly identical to IPC2's (max difference 0.274 for
tyrosine), confirming that the 9-parameter pI objective has a flat landscape
near the optimum. The practical advantage is **reproducibility and speed**:
TRF converges deterministically from any of the 13 starts in ~0.4 seconds per
start (13 s total on a single core), enabling large-scale bootstrap and
permutation analyses (1000 iterations in ~70 minutes on 4 cores).

See `scripts/07_permutation_pka.py` for the full optimization implementation.

### Per-protein Predictions

The 581 test-set predictions (experimental pI, IPC2 baseline, GBMS baseline,
IPC2.svr.19, and ESMpI) are available as a human-readable CSV:
[data/reference_results/per_protein_predictions.csv](data/reference_results/per_protein_predictions.csv)

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- Git (optional, for cloning)
- Conda (recommended) or any Python 3.10+ environment
- [Git LFS](https://git-lfs.com) — the embedding (`.npz`) and structure-archive (`.zip`) files are stored with Git LFS. It is bundled with modern Git for Windows and the official macOS/Linux Git installers; verify with `git lfs version` (Ubuntu/Debian: `sudo apt install git-lfs`). Cloning without LFS yields pointer files instead of data.

### Installation (Conda, recommended)

```bash
# 1. Clone the repository
git clone https://github.com/VeScarecrow/ESMpI.git
cd ESMpI

# 2. Create a conda environment
conda create -n esmpi python=3.11 -y
conda activate esmpi

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional) For full ESMpI prediction with ESM-2 embeddings
pip install torch transformers
```

### Installation (venv, alternative)

```bash
# 1. Clone and enter
git clone https://github.com/VeScarecrow/ESMpI.git
cd ESMpI

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional) For full ESMpI prediction with ESM-2 embeddings
pip install torch transformers
```

The core `requirements.txt` covers table reproduction and figure generation
(numpy, pandas, scipy, scikit-learn, matplotlib). `torch` and `transformers`
are only needed for predicting pI on new sequences with the full ESM-2 model.

### Predict pI for your own sequences

The most common use case — predict pI for new protein sequences:

```bash
# Baseline-only (no GPU needed, instant)
python predict.py -s "MKKFFDSRREQQKFLDAVAEHGRPDQVNPTQFIKVDSSAYNGLTEFLVFDRYLDGFNLDFEGTRTTAHQKLIEEAIDAFIKHGNTNLTIADALKDKGYRVEGYLKGYVDGNLSTTAQFNQAFKEKVNRLPDGQVVDHLAQGQPVVTAEQYAANEKRQAFDQVTGLPGYTHPQTAAPRTL" --baseline-only

# Full ESMpI prediction (requires torch + transformers, auto-downloads ESM-2 150M)
python predict.py -s "MKKFF..." --device cpu

# Batch from FASTA file
python predict.py -i my_proteins.fasta -o predictions.csv

# Use local ESM-2 model (avoid network download)
set ESM2_MODEL_PATH=C:\models\esm2_t30_150M_UR50D
python predict.py -i my_proteins.fasta
```

Output columns: `id`, `sequence`, `length`, `pI_IPC2_baseline`, `pI_GBMS_baseline`, `pI_ESMpI` (full mode only).

---

## Project Structure

```
ESMpI/
├── esmpi/                    # Importable Python package (shared engine + helpers)
│   ├── pka_engine.py         #   Henderson-Hasselbalch pI bisection + analytic Jacobian
│   ├── dataio.py             #   Data loading, path management (relative paths)
│   ├── pooling.py            #   ESM-2 per-type weighted pooling (W_ELEGANT)
│   ├── models.py             #   SVR factories, 5-fold CV, full retrain
│   ├── structure.py          #   Structure-based pI from per-residue pKa (FULL/DEHK)
│   └── metrics.py            #   RMSE, MAE, R², outlier rate, segmented RMSE
│
├── predict.py                # Predict pI for user-provided sequences (CLI)
│
├── scripts/                  # Pipeline scripts (run in order 01 → 10)
│   ├── 01_generate_predictions.py   # ESMpI predictions → data/predictions/z166_preds.npz
│   ├── 02_reproduce_benchmark.py    # Performance benchmark tables (581 full + 61 PDB subset)
│   ├── 03_bootstrap_pka.py          # B=1000 pKa bootstrap samples
│   ├── 04_bootstrap_intermethod.py  # B=2000 paired bootstrap
│   ├── 05_make_figures.py            # Scatter, effects, bootstrap, and SI figures
│   ├── 06_stratified_analysis.py     # Stratified RMSE by source/length/pI
│   ├── 07_permutation_pka.py         # 1000-iter permutation re-fit of 9-pKa
│   ├── 08_ipc2_protocol.py           # IPC2 evaluation protocol check
│   ├── 09_prepare_structures.py      # Download/repair experimental PDB structures
│   ├── 10_structure_pI.py            # Structure-based pI from per-residue pKa
│   └── 11_benchmark_inference_e2e.py  # End-to-end inference time benchmark (GPU + CPU)
│
├── data/                     # All input data (committed, ~28 MB total)
│   ├── embeddings/           #   ESM-2 150M per-type embeddings (22 MB)
│   ├── features/             #   Physical features (z81, 4 MB)
│   ├── predictions/          #   Precomputed ESMpI predictions (61 KB, Tier-1 anchor)
│   ├── mapping/              #   seq_no → npz index mapping + data-source labels
│   ├── benchmark/            #   Benchmark input CSVs (8 files)
│   ├── full_sources/         #   Two full-source FASTAs (SI Fig S3)
│   ├── structures/           #   PDB structure archives (see Data Inventory)
│   └── reference_results/    #   Frozen expected outputs for verification
│
├── figures/                   # Static figure assets
│   ├── figure1_framework.png  #   Framework diagram (hand-drawn, not code-generated)
│   └── figure1_framework.pptx
│
├── results/                   # Runtime outputs (gitignored)
│   ├── tables/                #   Benchmark metric CSVs
│   ├── figures/               #   Generated figures (PNG/PDF)
│   └── bootstrap/            #   Bootstrap results
│
├── requirements.txt
├── README_EN.md
├── README_CN.md
└── .gitignore
```

---

## Data Inventory

> Column-level documentation for every data file: [data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md).

| File | Location | Size | Description |
|------|----------|------|-------------|
| `z82_150m_pertype_train.npz` | `data/embeddings/` | 16.3 MB | ESM-2 150M per-type embeddings (train, 1743×7×640) |
| `z82_150m_pertype_test.npz` | `data/embeddings/` | 5.5 MB | ESM-2 150M per-type embeddings (test, 581×7×640) |
| `z81_features_train.csv` | `data/features/` | 3.1 MB | Physical features + 19 pKa-scale pI values (train) |
| `z81_features_test.csv` | `data/features/` | 1.0 MB | Physical features + 19 pKa-scale pI values (test) |
| `z166_preds.npz` | `data/predictions/` | 61 KB | Precomputed ESMpI + IPC2-SVR test predictions (Tier-1 anchor) |
| `z164_seqno2zidx.csv` | `data/mapping/` | 5 KB | seq_no → npz row index mapping (see Notes below) |
| `z46d_source_labels.csv` | `data/mapping/` | 81 KB | Data-source labels for the full set (PIP-DB=1380 / SWISS-2DPAGE=944, corrected version) |
| `pip_db_normal.fasta` | `data/full_sources/` | 1.0 MB | Full PIP-DB, 2427 entries (experimental pI in FASTA headers, SI Fig S3) |
| `ch2d19_2_1st_isoform.fasta` | `data/full_sources/` | 432 KB | Full SWISS-2DPAGE, 1054 first-isoform entries (SI Fig S3) |
| `IPC_protein_25.csv` | `data/benchmark/` | 225 KB | 581 test sequences + experimental pI |
| `reg_581.csv` | `data/benchmark/` | 155 KB | Registry: structure source, PDB paths, confidence class |
| `pi_fallback.csv` | `data/benchmark/` | 35 KB | Per-protein pI for 7 methods (experimental PDB input) |
| `pi_af.csv` | `data/benchmark/` | 132 KB | Per-protein pI for AF2-structure methods |
| `pkalm_piprott.csv` | `data/benchmark/` | 14 KB | pKALM official PIPROT predictions |
| `pypka_exp_z44.csv` | `data/benchmark/` | 18 KB | PypKa predictions on experimental structures |
| `perm_pka_samples.csv` / `perm_pka_summary.csv` | `data/reference_results/` | 184 KB | Frozen 1000-iteration permutation pKa samples and summary |
| `bench_inference_time_e2e.csv` | `data/reference_results/` | 52 KB | Frozen per-protein end-to-end inference times (IPC2, IPC2.svr.19, ESMpI GPU/CPU) |
| `per_protein_predictions.csv` | `data/reference_results/` | 254 KB | Human-readable per-protein predictions for all 581 test proteins |
| `int8_deployment_description.txt` | `data/reference_results/` | 0.6 KB | INT8 deployment inference description and benchmark environment |
| `alphafold2_structures.zip` | `data/structures/` | 28.9 MB | 581 AF2-predicted structures (seq_no-named, from AF2 pipeline) |
| `experimental_pdb_A_high_confidence.zip` | `data/structures/` | 2.4 MB | 61 high-confidence experimental PDBs (A61 subset, Table 2) |
| `experimental_pdb_B_trusted.zip` | `data/structures/` | 3.5 MB | 73 trusted experimental PDBs |
| `experimental_pdb_C_limited.zip` | `data/structures/` | 3.3 MB | 71 limited-confidence experimental PDBs |
| `experimental_pdb_D_rejected.zip` | `data/structures/` | 1.2 MB | 16 rejected experimental PDBs |
| `structure_patches.zip` | `data/structures/` | 3.4 MB | 33 patch structures (23 ESM-2/3 tail patches + 10 tail-10 patches) |
| `patch_seq111.zip` | `data/structures/` | 0.1 MB | 2 special-case structures for seq_no 111 |

**Experimental structure confidence classes** (see `reg_581.csv` for per-protein mapping):

| Class | Count | Criteria | Usage |
|-------|-------|----------|-------|
| A — high confidence | 61 | X-ray resolution < 3.0 Å, complete chain, no gaps | Table 2 (A61 subset) |
| B — trusted | 73 | X-ray/NMR, minor gaps or lower resolution | Supplementary analyses |
| C — limited | 71 | Significant gaps, low resolution, or partial chain | Not used for benchmarking |
| D — rejected | 16 | Severe quality issues | Excluded from analysis |
| P — predicted only | 360 | No experimental PDB available | AF2 structures used instead |

---

## Method Mapping (Table 1 & 2)

| Paper name | Type | CSV column / source |
|------------|------|---------------------|
| ESMpI | Sequence | `z166_preds.npz` → `pt_ours` (IPC2 baseline + ESM-2 150M SVR residual) |
| IPC2.protein.svr.19 | Sequence | F19 features → SVR(C=1.0, epsilon=0.12) |
| IPC2_protein | Sequence | IPC2 published 9-pKa H-H pI |
| pKALM | Sequence | `pkalm_piprott.csv` (official web server) |
| PropKa_AF2 | Structure (AF2) | PropKa per-residue pKa → H-H pI |
| PypKa_AF2 | Structure (AF2) | PypKa Poisson-Boltzmann → H-H pI |
| DeepKa_AF2 | Structure (AF2) | DeepKa ML pKa → H-H pI |
| PropKa_PDB | Structure (exp PDB) | PropKa on experimental PDB |
| PypKa_PDB | Structure (exp PDB) | PypKa on experimental PDB |
| DeepKa_PDB | Structure (exp PDB) | DeepKa on experimental PDB |

Structure-based methods use experimental PDB in Table 2 (A61 subset) and
AlphaFold2 predicted structures in Table 1 (full 581). Missing titratable
sites are backfilled with Thurlkill (2006) model-compound pKa values.

---

## Reproduction

### Tier 1 — Reproduce Tables (1 minute, no GPU)

```bash
python scripts/02_reproduce_benchmark.py
```

Produces Table 1 (581 full test set) and Table 2 (61-protein PDB subset) in
`results/tables/`. Compare with `data/reference_results/` for sanity check.

### Tier 2 — Full Pipeline (regenerate predictions, bootstrap, figures)

> Step 1 takes ~12 min on an i5-9400F (6 cores) — 19 pKa scales × 5-fold CV.
> Steps 3/4/7 run thousands of bootstrap/permutation iterations (hours on a
> single core); reduce `B` first if you just want to validate the pipeline.

```bash
# Step 1: Regenerate ESMpI predictions from ESM-2 150M embeddings (~12 min)
python scripts/01_generate_predictions.py

# Step 2: Reproduce benchmark tables (Tier 1 also works standalone)
python scripts/02_reproduce_benchmark.py

# Step 3: pKa parameter bootstrap (B=1000, for Figure 4)
python scripts/03_bootstrap_pka.py

# Step 4: Inter-method paired bootstrap (B=2000, ESMpI vs IPC2-SVR)
python scripts/04_bootstrap_intermethod.py

# Step 5: Generate paper figures (main Figures 2/3/4 + SI Figures S1–S4)
python scripts/05_make_figures.py

# Step 6 (optional): Stratified RMSE analysis
python scripts/06_stratified_analysis.py

# Step 7 (optional, ~70 min): permutation robustness experiment (SI Table S15 / Fig S4, 1000 iters)
#   Quick smoke test with env vars: NITER=3 NPROC=1 python scripts/07_permutation_pka.py
python scripts/07_permutation_pka.py

# Step 8 (optional): verify the IPC2 paper Table 2 evaluation protocol (SI Table S4)
python scripts/08_ipc2_protocol.py
```

> The 1000-iteration permutation samples for SI Fig S4 ship as a frozen copy in
> `data/reference_results/`; 05 falls back to them automatically if step 7 is not
> run. Figure 4 requires step 3 to generate the bootstrap samples first.

### Paper Figures and Additional Analyses

| Output | Script | Content |
|--------|--------|---------|
| `paper_fig2_scatter` | 05 | ESMpI vs experimental pI: black all-point fit and orange outlier-only fit (error>0.5) |
| `paper_fig3_effects` | 05 | Four-panel comparison — GBMS vs ESMpI density scatter, acidic/basic pI, data source, protein length |
| `paper_fig4_bootstrap` | 05 | B=1000 bootstrap distributions of the 9 pKa values |
| `si_figS1_pI_distribution` | 05 | Train/test experimental pI distributions |
| `si_figS2_pairwise` | 05 | Pairwise correlations exp / GBMS / IPC2.svr.19 / ESMpI (n=581) |
| `si_figS3_source_distribution` | 05 | Full-source pI distributions: PIP-DB (2427) and SWISS-2DPAGE (1054) |
| `si_figS4_perm_pka` | 05 | 9-pKa distributions from the 1000-iteration permutation experiment |
| `table_ipc2_protocol.csv` | 08 | Pooled / 10-fold CV / IPC2 printed values for 16 methods |
| `perm_pka_*.csv` | 07 | Per-iteration pKa with mean/median/95% CI |

**IPC2 evaluation protocol (08)**: IPC2's printed Table 2 reports
the average of per-fold metrics over a random 10-fold split, not the pooled
full-set RMSE; the two differ systematically by ~0.005 pH (Jensen's
inequality). Script 08 uses 20 seeds × 10 folds = 200 folds to remove split
randomness: the 200-fold RMSE for IPC_protein is 0.8675 vs the printed 0.8677
(difference 0.0002).

**Permutation robustness experiment (07)**: Each
iteration removes 581 randomly chosen proteins from the 1743 training set,
replaces them with all 581 test proteins, and re-optimizes the 9-pKa vector
under the GBMS protocol. 1000 iterations (seed 20260927, ~70 min on 4 workers,
Intel Core i5-9400F).
The ported implementation reproduces the frozen reference bit-for-bit; use the
`NITER`/`NPROC` environment variables for a quick smoke test.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Third-party components:
- [ESM-2](https://github.com/facebookresearch/esm) (Meta AI) — MIT License
- PIP-DB and SWISS-2DPAGE data sources — see original publications for terms of use

## Citation

If you use this code, please cite:

```bibtex
@article{luo2026esmpi,
  title={Protein language model-boosted prediction of isoelectric point},
  author={Luo, Fangfang and Lu, Xiangxiang and Cai, Zhitao and Wu, Riting and Su, Shubin and Huang, Yandong},
  journal={Journal of Molecular Biology},
  year={2026}
}
```
