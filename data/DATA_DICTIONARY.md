# Data Dictionary

Last updated: 2026-09-29

All paths are relative to this file's parent directory (`data/`).

---

## benchmark/ — Benchmark prediction results (structure + sequence methods on the 581 test set)

### IPC_protein_25.csv
| Column | Description |
|--------|-------------|
| `exp_pI` | Experimental isoelectric point (unit: pH) |
| `sequence` | Protein amino acid sequence (single-letter IUPAC) |

**Rows:** 581 (test set, IPC_protein_25).  
**Source:** Derived from IPC2 benchmark, filtered to sequences with experimentally measured pI.

---

### pi_af.csv
Per-protein pI predictions using **AlphaFold2-predicted structures** as the structural input.

| Column | Description |
|--------|-------------|
| `seq_no` | Protein ID (1–581), matches `z164_seqno2zidx.csv` |
| `exp_pI` | Experimental pI |
| `M1_DeepKa_Thurl_AF` | DeepKa pKa (AF2) → HH → pI, Thurlkill fallback |
| `M1_DeepKa_Thurl_ESM` | DeepKa pKa (ESMFold) → HH → pI, Thurlkill fallback |
| `M2_PropKa_Thurl_AF` | PropKa pKa (AF2) → HH → pI, Thurlkill fallback |
| `M2_PropKa_Thurl_ESM` | PropKa pKa (ESMFold) → HH → pI, Thurlkill fallback |
| `M3_our_9param_AF` | GBMS re-optimized 9-pKa (AF2) → HH → pI |
| `M3_our_9param_ESM` | GBMS re-optimized 9-pKa (ESMFold) → HH → pI |
| `M3_ipc_9param_AF` | IPC2 published 9-pKa (AF2) → HH → pI |
| `M3_ipc_9param_ESM` | IPC2 published 9-pKa (ESMFold) → HH → pI |
| `M4_DeepKa_SA_Thurl_AF` | DeepKa + solvent accessibility (AF2) → HH → pI, Thurlkill fallback |
| `M4_DeepKa_SA_Thurl_ESM` | DeepKa + SA (ESMFold) → HH → pI, Thurlkill fallback |
| `M5_DeepKa_SA_our_AF` | DeepKa + SA + GBMS 9-pKa (AF2) → HH → pI |
| `M5_DeepKa_SA_our_ESM` | DeepKa + SA + GBMS 9-pKa (ESMFold) → HH → pI |
| `M5_DeepKa_SA_ipc_AF` | DeepKa + SA + IPC2 9-pKa (AF2) → HH → pI |
| `M5_DeepKa_SA_ipc_ESM` | DeepKa + SA + IPC2 9-pKa (ESMFold) → HH → pI |
| `M8_IPC2_svr` | IPC2 SVR model prediction (sequence-only, no structure) |
| `source` | `esm_pred` = ESMFold structure used; `exp_fixed` = experimental PDB structure used |
| `conf_class` | Confidence class: `A-high_confidence_exp`, `B-trusted_exp_annotated`, `C-limited_exp_caution`, `D-exp_structure_rejected`, `P-predicted_no_exp_PDB` |
| `final_pdb` | Path to the PDB file used (`structures/final_581/` or `structures/recomputed_seq111/`) |
| `af_pdb_ref` | Path to AlphaFold2 reference PDB (`structures/af2_581/`; empty for experimental structures) |

**Rows:** 581.  
**Note:** "AF_primary" = for the 61 proteins with experimental PDBs, AF2 structures are used; for the rest, ESMFold or AF2 as available.

---

### pi_fallback.csv
Same predictions as `pi_af.csv` but using **experimental PDB structures** where available, ESMFold otherwise.

| Column | Description |
|--------|-------------|
| `seq_no` | Protein ID |
| `exp_pI` | Experimental pI |
| `n_sites` | Total ionizable sites in sequence |
| `n_deepka_used` | Sites where DeepKa pKa was used |
| `n_propka_used` | Sites where PropKa pKa was used |
| `M1_DeepKa_Thurl` | DeepKa pKa → HH → pI, Thurlkill fallback |
| `M2_PropKa_Thurl` | PropKa pKa → HH → pI, Thurlkill fallback |
| `M3_our_9param` | GBMS re-optimized 9-pKa → HH → pI |
| `M3_ipc_9param` | IPC2 published 9-pKa → HH → pI |
| `M4_DeepKa_SA_Thurl` | DeepKa + SA → HH → pI, Thurlkill fallback |
| `M5_DeepKa_SA_our` | DeepKa + SA + GBMS 9-pKa → HH → pI |
| `M5_DeepKa_SA_ipc` | DeepKa + SA + IPC2 9-pKa → HH → pI |

**Rows:** 581.

---

### pkalm_piprott.csv
pKALM (protein language model) predictions on the test set.

| Column | Description |
|--------|-------------|
| `seq_no` | Protein ID |
| `source` | Structure source (`esm_pred` / `exp_fixed`) |
| `exp_pI` | Experimental pI |
| `pI_pKALM` | pKALM predicted pI |

**Rows:** 581.

---

### pypka_exp_z44.csv
PypKa predictions on the **44 proteins** that have experimental PDB structures (subset of 581).

| Column | Description |
|--------|-------------|
| `seq_no` | Protein ID (subset of 1–581) |
| `exp_pI` | Experimental pI |
| `M7_PypKa_Thurl` | PypKa pKa → HH → pI, Thurlkill fallback |
| `PypKa_webserver_pI` | pI as reported by PypKa web server |
| `n_sites` | Total ionizable sites |
| `n_pypka_used` | Sites where PypKa pKa was used |
| `pypka_coverage` | Fraction of sites covered by PypKa |
| `NTR_used` | Whether N-terminus pKa was predicted by PypKa |
| `CTR_used` | Whether C-terminus pKa was predicted by PypKa |
| `M1_…` – `M5_…` | Same as `pi_fallback.csv` |
| `diff_from_webserver_pI` | Difference between our PypKa-based pI and PypKa web server pI |

**Rows:** 44.

---

### reg_581.csv
Structure registry for all 581 test proteins.

| Column | Description |
|--------|-------------|
| `seq_no` | Protein ID |
| `source` | `esm_pred` / `exp_fixed` / `exp_filled` |
| `exp_pdb_id` | PDB ID (if experimental structure exists) |
| `final_chain` | Chain used |
| `fill_status` | Gap-filling status: `no_gap`, `no_valid_segment(signal_peptide/propeptide/tag_not_filled)`, `not_filled_gt30_dual_track`, `OK`, `OK_close_contacts_present`, `no_identifiable_gap` |
| `src_path` | Original structure file path (`structures/esmfold_581/`, `structures/exp_pdb_repaired_215/`, `structures/exp_pdb_filled_pdbfixer/`) |
| `final_pdb` | Final PDB used for pKa calculation (`structures/final_581/` or `structures/recomputed_seq111/`) |
| `af_pdb_ref` | AlphaFold2 reference PDB path (`structures/af2_581/`) |
| `conf_class` | Confidence class: `A-high_confidence_exp`, `B-trusted_exp_annotated`, `C-limited_exp_caution`, `D-exp_structure_rejected`, `P-predicted_no_exp_PDB` |
| `genuine_maxgap` | Maximum unresolved gap length |
| `gap_special_action` | Special handling for gaps: `R0_no_gap`, `R1_small_gap_fill_atoms+fallback_model_pKa`, `R2_medium_gap(16-30) ...`, `R3_long_gap(31-100) ...`, `R4_very_long_gap(>100) ...` |
| `inv_titr_frac_test` | Inverse titration fraction test |
| `qc_status` | Quality control status |
| `deepka_status` | DeepKa prediction status |
| `pypka_status` | PypKa prediction status |
| `dssp_status` | DSSP secondary structure status |
| `pi_status` | pI calculation status |
| `n_chains` | Number of chains |
| `n_res` | Number of residues |
| `ca_only` | Whether only C-alpha atoms present |
| `propka_status` | PropKa prediction status |

**Rows:** 581.

---

### z164_seqno2zidx.csv
Mapping between `seq_no` (protein ID 1–581) and `zidx` (row index in z81 feature matrix, 0-based).

| Column | Description |
|--------|-------------|
| `seq_no` | Protein ID |
| `zidx` | Row index in z81 features / embeddings |

**Rows:** 581.

---

### z166_preds.npz
Frozen model predictions (NumPy compressed archive).

| Key | Shape | Description |
|-----|-------|-------------|
| `pt_ours` | (581,) | pI-ESM test-set predictions |
| `pt_f19` | (581,) | IPC2.svr.19 (F19) test-set predictions |
| `bt_ipc2` | (581,) | IPC2 published pKa → HH → pI (baseline) |
| `yte` | (581,) | Experimental pI (test set) |
| `oof_ours` | (1743,) | pI-ESM out-of-fold predictions (training set) |
| `ytr` | (1743,) | Experimental pI (training set) |
| `bo` | (1743,) | IPC2 published pKa → HH → pI (training set, out-of-fold) |

---

## embeddings/ — ESM-2 150M per-residue embeddings

### z82_150m_pertype_test.npz / z82_150m_pertype_train.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `site_per_type` | (N, 7, 640) | float16 | Per-residue ESM-2 embeddings, grouped by 7 ionizable types (N-term, C-term, D, E, H, K, C, Y, R → 7 groups: NTER, CTER, ASP, GLU, HIS, LYS, CYS+TYR+ARG merged) |
| `site_counts` | (N, 7) | int32 | Number of residues per type |
| `site_uniform` | (N, 640) | float16 | Uniform (mean) pooled embedding over all residues |
| `keys` | (N,) | str | Protein sequence |
| `exp_pI` | (N,) | float32 | Experimental pI |

**N:** test=581, train=1743.  
**Model:** ESM-2 150M (`esm2_t30_150M_UR50D`), layer 30, mean-pooled per residue type.

---

## features/ — Handcrafted feature matrix (z81)

### z81_features_test.csv / z81_features_train.csv
151 columns total. Groups:

| Group | Cols | Description |
|-------|------|-------------|
| `key` | 1 | Protein sequence |
| `exp_pI` | 1 | Experimental pI |
| `pI_*` | 20 | pI predicted by 20 different pKa scales (Bjellqvist, DTASelect, Dawson, EMBOSS, Grimsley, IPC2_peptide, IPC2_protein, IPC_peptide, IPC_protein, Lehninger, Nozaki, Patrickios, Rodwell, Sillero, Solomon, Thurlkill, Toseland, Wikipedia, ProMoST, `pI_our` = GBMS re-optimized) |
| `f19_*` | 6 | Summary statistics of the 19 scale pI values (std, min, max, range, mean, median) |
| `frac_*` | 21 | Amino acid composition fractions (20 AA + X) |
| `log10_len` | 1 | log10(sequence length) |
| `dens_*` | 7 | Density of ionizable residues (D, E, H, K, C, Y, R) |
| `N1_*` / `C1_*` | 21+21 | N-terminal / C-terminal residue identity (one-hot) |
| `Nwin3_*` / `Cwin3_*` | 21+21 | N-terminal / C-terminal 3-residue window composition |
| `Q_pH*` | 9 | Net charge at pH 3–11 (Henderson-Hasselbalch) |
| `beta_at_pI` | 1 | Buffer capacity at pI |

**Rows:** test=581, train=1743.

---

## full_sources/ — Full data source FASTA

### ch2d19_2_1st_isoform.fasta
**SWISS-2DPAGE** full dataset (1054 entries).

- Header format: `>UNIPROT_ID (pI1, pI2, ... / MW1, MW2, ...)`
- Sequence: single-letter IUPAC amino acid code
- Source: http://world-2dpage.expasy.org/swiss-2dpage/
- Preprocessed from: http://ipc.netmark.pl/datasets/ch2d19_2.dat

### pip_db_normal.fasta
**PIP-DB** full dataset (2427 entries).

- Header format: `>RANDOM_ID (pI1, pI2, ..., pI_avg)` where the last value is the average
- Sequence: single-letter IUPAC amino acid code
- Source: http://www.pip-db.org

---

## structures/ — PDB structure archives

### alphafold2_structures.zip
581 AlphaFold2-predicted structures (one per test protein).

- File naming: `{seq_no}.pdb` (e.g., `1.pdb`, `100.pdb`)
- Source: AlphaFold DB / ColabFold pipeline
- Used by: `pi_af.csv` (AF2-based structure methods)

### experimental_pdb_A_high_confidence.zip
61 high-confidence experimental PDB structures (A61 subset).

- File naming: `{seq_no}.pdb` (e.g., `1.pdb`, `100.pdb`)
- Source: RCSB PDB, X-ray resolution < 3.0 Å, complete chain, no gaps
- Processing: missing residues rebuilt, single chain selected
- Used by: Table 2 benchmark (A61 subset)

### experimental_pdb_B_trusted.zip
73 trusted experimental PDB structures.

- Criteria: X-ray/NMR, minor gaps or lower resolution
- Usage: supplementary analyses

### experimental_pdb_C_limited.zip
71 limited-confidence experimental PDB structures.

- Criteria: significant gaps, low resolution, or partial chain
- Usage: not used for benchmarking

### experimental_pdb_D_rejected.zip
16 rejected experimental PDB structures.

- Criteria: severe quality issues
- Usage: excluded from analysis

### structure_patches.zip
33 patch structures for special cases.

- `patches_esm23/` (23 files): ESM-2/3-predicted tail patches
- `patches_tail10/` (10 files): tail-10 patches
- Used for proteins where standard AF2/EBI structures had unresolved terminal regions

### patch_seq111.zip
2 special-case structures for `seq_no=111`.

---

## mapping/ — Index mapping and labels

### z164_seqno2zidx.csv
See [benchmark/z164_seqno2zidx.csv](#z164_seqno2zidxcsv) above.  
*(This copy in `mapping/` is the canonical version used by scripts 04, 05, 06.)*

### z46d_source_labels.csv
Data-source labels for the **full 2324-protein dataset** (train 1743 + test 581).

| Column | Description |
|--------|-------------|
| `seq_no_full` | Protein ID (1–2324) |
| `split` | `train_75` (1743) or `test_25` (581) |
| `uniprot_acc` | UniProt accession |
| `exp_pI` | Experimental pI |
| `source` | Data source: `PIP-DB` (1380) or `SWISS-2DPAGE` (944) |

**Rows:** 2324.  
**Important:** In an earlier version, PIP-DB and SWISS-2DPAGE labels were **swapped**. This file is the **corrected** version. Test subset: PIP-DB=350, SWISS-2DPAGE=231.

---

## predictions/ — Frozen predictions

### z166_preds.npz
See [benchmark/z166_preds.npz](#z166_predsnpz) above.  
*(This copy in `predictions/` is the canonical version used by scripts 01–06.)*

---

## reference_results/ — Frozen reference results (for reproduction validation)

### bench_A61_metrics.csv
Benchmark metrics on the **61-protein subset** with experimental PDB structures.

| Column | Description |
|--------|-------------|
| `Rank` | Rank |
| `Category` | Category (`Structure` / `Sequence` / `Seq_baseline`) |
| `Method` | Method name |
| `n` | Number of proteins |
| `RMSE` | Root mean squared error |
| `MAE` | Mean absolute error |
| `bias` | Mean signed error |
| `R2_coef_det` | Coefficient of determination |
| `Pearson_r` | Pearson correlation |
| `r2` | Squared Pearson correlation |
| `n_\|err\|>0.5` | Number of outliers (abs error > 0.5) |

**Rows:** 14 methods.

---

### bench_all581_metrics.csv
Same as above but on the **full 581 test set**.

**Rows:** 14 methods.

---

### perm_pka_samples.csv
Permutation experiment: 1000 iterations of 9-pKa re-optimization with randomly permuted train/test split.

| Column | Description |
|--------|-------------|
| `iter` | Iteration index (0–999) |
| `rmse` | Test-set RMSE for this iteration |
| `N_TER` – `arg` | Optimized pKa values (9 parameters) |

**Rows:** 1000.

---

### perm_pka_summary.csv
Summary statistics of the 1000 permutation iterations.

| Column | Description |
|--------|-------------|
| `dim` | pKa parameter name |
| `mean` | Mean across iterations |
| `std` | Standard deviation |
| `ci_lo` / `ci_hi` | 95% confidence interval |
| `median` | Median |

**Rows:** 9.

---

### table_ipc2_protocol.csv
IPC2 protocol comparison: pooled vs 10-fold CV vs paper-reported metrics.

| Column | Description |
|--------|-------------|
| `method` | Method name |
| `pooled_rmse` / `pooled_mae` / `pooled_r2` / `pooled_out` | Metrics on all 581 test proteins |
| `fold_rmse` / `fold_mae` / `fold_r2` / `fold_out` | Mean metrics across 200 folds (20 seeds × 10 folds) |
| `paper_rmse` / `paper_mae` / `paper_r2` / `paper_out` | Values reported in IPC2 paper Table 2 |
| `diff_rmse` | Difference between fold and paper RMSE |

**Rows:** 16 methods.

---

### z168_bootstrap_summary.csv
Bootstrap confidence intervals for the 9 GBMS pKa parameters.

| Column | Description |
|--------|-------------|
| `dim` | pKa parameter name |
| `mean` | Bootstrap mean |
| `std` | Bootstrap standard deviation |
| `ci_lo` / `ci_hi` | 95% CI |
| `pile0` | Fraction of samples hitting lower bound (0) |
| `pile14` | Fraction of samples hitting upper bound (14) |
| `ipc2` | IPC2 published value |
| `bjellqvist` | Bjellqvist scale value |

**Rows:** 9.

---

### bench_inference_time_e2e.csv
Per-protein end-to-end inference time on the 581-sequence test set (IPC_protein_25). Measured from raw amino-acid sequence to final pI value.

| Column | Description |
|--------|-------------|
| `seq_no` | Protein ID (1–581) |
| `length` | Sequence length (amino acids) |
| `t_ipc2_s` | IPC2 9-pKa bisection time (seconds), CPU |
| `t_svr19_s` | IPC2.svr.19 time (19 pKa baselines + SVR predict, seconds), CPU |
| `t_piesm_gpu_s` | pI-ESM time (ESM-2 150M FP32 forward + pool + SVR, seconds), GPU |
| `t_piesm_cpu_s` | pI-ESM time (ESM-2 150M FP32 forward + pool + SVR, seconds), CPU |

**Rows:** 581.

**Benchmark environment:**
- Local: Intel Core i5-9400F (6 cores, 2.90 GHz), 16 GB RAM, Windows 10, Python 3.10, PyTorch 2.5.0, NVIDIA GeForce GTX 1060 6GB (CUDA).
- Server (159.75.31.213:8000): Tencent Cloud Standard SA2 (AMD EPYC, 4 vCPUs, 8 GB RAM, Guangzhou Zone 3), CPU-only, INT8 quantized ESM-2 150M, 2 threads.

---

### int8_deployment_description.txt
Short English description of the INT8 accelerated deployment inference and benchmark environment.
