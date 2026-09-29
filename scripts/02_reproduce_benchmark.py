# -*- coding: utf-8 -*-
"""
02_reproduce_benchmark: A-class high-confidence benchmark reproduction.

Migrated from repro_A61_benchmark/code/reproduce_benchmark.py. Produces
Table 1 (581-protein test set) and Table 2 (A-class n=61 high-confidence
experimental-structure subset) method-level benchmark metrics.

The only change from the original is path remapping: input data is read
from data/benchmark/ and results are written to results/tables/. All
numerical logic is unchanged.

Outputs:
  results/tables/bench_A61_metrics.csv       A-class n=61 metrics
  results/tables/bench_A56_metrics.csv       exp_fixed n=56 metrics
  results/tables/bench_all581_metrics.csv    full set n=581 metrics
  results/tables/bench_A61_per_protein.csv   per-protein detail
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

# Project root is one level above the scripts/ directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(ROOT, "data", "benchmark")
OUT = os.path.join(ROOT, "results", "tables")
os.makedirs(OUT, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")

# ---------------- Thurlkill (2006) model pKa ----------------
THURL = {"N": 8.00, "C": 3.67, "D": 3.67, "E": 4.25, "H": 6.54,
         "K": 10.40, "C_cys": 8.55, "Y": 9.84, "R": 12.00}
# Group names consistent with z44 -> pKa; acidic groups (negatively charged after deprotonation)
ACID_AA = set("DECY")          # side-chain acidic: D E C Y
PH_GRID = np.arange(0, 14.0001, 0.01)


def thurlkill_pi(seq):
    """Pure Thurlkill baseline: fixed model pKa for all sites, H-H bisection for pI (sequence-only, no structure)."""
    # (pKa, is_acid): NTR basic, CTR acidic + all ionizable side chains
    sites = [(THURL["N"], False), (THURL["C"], True)]
    for aa in seq:
        if aa == "D": sites.append((THURL["D"], True))
        elif aa == "E": sites.append((THURL["E"], True))
        elif aa == "C": sites.append((THURL["C_cys"], True))
        elif aa == "Y": sites.append((THURL["Y"], True))
        elif aa == "H": sites.append((THURL["H"], False))
        elif aa == "K": sites.append((THURL["K"], False))
        elif aa == "R": sites.append((THURL["R"], False))
    q = np.zeros_like(PH_GRID)
    for pka, is_acid in sites:
        S = 1.0 / (1.0 + 10.0 ** (pka - PH_GRID))   # deprotonation fraction
        q -= S if is_acid else -(1.0 - S)
    for i in range(1, len(PH_GRID)):
        if q[i - 1] * q[i] <= 0 and q[i - 1] != q[i]:
            return PH_GRID[i - 1] - q[i - 1] * (PH_GRID[i] - PH_GRID[i - 1]) / (q[i] - q[i - 1])
    return float(PH_GRID[int(np.argmin(np.abs(q)))])


def metrics(pred, tru):
    pred = np.asarray(pred, float)
    tru = np.asarray(tru, float)
    if len(pred) < 2:
        return {"n": len(pred), "RMSE": np.nan, "MAE": np.nan, "bias": np.nan,
                "R2_coef_det": np.nan, "Pearson_r": np.nan, "r2": np.nan,
                "n_|err|>0.5": 0}
    e = pred - tru
    y = tru
    r = float(stats.pearsonr(pred, tru)[0])
    return {
        "n": len(e),
        "RMSE": round(float(np.sqrt((e ** 2).mean())), 4),
        "MAE": round(float(np.abs(e).mean()), 4),
        "bias": round(float(e.mean()), 4),
        "R2_coef_det": round(float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4),
        "Pearson_r": round(r, 4),
        "r2": round(r ** 2, 4),
        "n_|err|>0.5": int((np.abs(e) > 0.5).sum()),
    }


# ---------------- Load data ----------------
reg = pd.read_csv(os.path.join(INP, "reg_581.csv"))
fall = pd.read_csv(os.path.join(INP, "pi_fallback.csv"))     # M1-M5 with experimental-structure convention
afm = pd.read_csv(os.path.join(INP, "pi_af.csv"))             # M8 SVR
z44 = pd.read_csv(os.path.join(INP, "pypka_exp_z44.csv"))
pk = pd.read_csv(os.path.join(INP, "pkalm_piprott.csv"))
ipc = pd.read_csv(os.path.join(INP, "IPC_protein_25.csv"))

seqs = {i + 1: str(s).strip().upper() for i, s in enumerate(ipc["sequence"])}

# pI-ESM (this work): IPC2 pKa baseline + ESM-2 residual SVR.
# z166 array index != seq_no, must be mapped via z164_seqno2zidx.csv; yte order verified to match exp_pI.
_npz = np.load(os.path.join(INP, "z166_preds.npz"), allow_pickle=True)
_map = pd.read_csv(os.path.join(INP, "z164_seqno2zidx.csv"))
_pt_ours = _npz["pt_ours"]
pI_ESM = {int(sn): float(_pt_ours[int(zi)])
          for sn, zi in zip(_map["seq_no"], _map["zidx"])}

d = (fall
     .merge(afm[["seq_no", "M8_IPC2_svr"]], on="seq_no")
     .merge(z44[["seq_no", "M7_PypKa_Thurl"]], on="seq_no")
     .merge(pk[["seq_no", "pI_pKALM"]], on="seq_no")
     .merge(reg[["seq_no", "source", "conf_class"]], on="seq_no"))
d["pI_pureThurl"] = d.seq_no.map(lambda s: round(thurlkill_pi(seqs[int(s)]), 4))
d["pI_pIESM"] = d.seq_no.map(lambda s: round(pI_ESM.get(int(s)), 4))

# PypKa pure AF2 structure track (output of compute_pypka_af2_pi.py, unfinished seqs are NaN)
paf_path = os.path.join(OUT, "pypka_af2_pi.csv")
if os.path.exists(paf_path):
    paf = pd.read_csv(paf_path)[["seq_no", "pI_pypka_AF2"]]
    d = d.merge(paf, on="seq_no", how="left")
    print(f"Loaded PypKa AF2 pI: A61/A56 experimental-convention table coverage "
          f"{int(d.pI_pypka_AF2.notna().sum())}/{len(d)} (this column is reference only, not used in A61 metrics; see full-set coverage below)")
else:
    d["pI_pypka_AF2"] = np.nan
    print("pypka_af2_pi.csv not found, PypKa AF2 column all NaN (run compute_pypka_af2_pi.py first)")

# Method grouping: label, column name, category.
# A61/A56 tables only include structure models with [experimental PDB convention], not PypKa pure AF2 (that is the full-set unified AF2 convention, see bench_all581).
METHODS = [
    ("PropKa+Thurl (exp_PDB)",      "M2_PropKa_Thurl",     "Structure"),
    ("DeepKa+Thurl (exp_PDB)",      "M1_DeepKa_Thurl",     "Structure"),
    ("DeepKa+SA+Thurl (exp_PDB)",   "M4_DeepKa_SA_Thurl",  "Structure"),
    ("DeepKa+SA+our (exp_PDB)",     "M5_DeepKa_SA_our",    "Structure"),
    ("DeepKa+SA+IPC2 (exp_PDB)",    "M5_DeepKa_SA_ipc",    "Structure"),
    ("PypKa+Thurl (exp_PDB)",       "M7_PypKa_Thurl",      "Structure"),
    ("Thurlkill_baseline",          "pI_pureThurl",        "Seq_baseline"),
    ("pI-ESM (this_work)",          "pI_pIESM",            "Sequence"),
    ("Our 9-param",                 "M3_our_9param",       "Sequence"),
    ("IPC2 9-param",                "M3_ipc_9param",       "Sequence"),
    ("IPC2 SVR",                    "M8_IPC2_svr",         "Sequence"),
    ("pKALM PIPROT",                "pI_pKALM",            "Sequence"),
]


def bench(data, tag):
    rows = []
    for name, col, cat in METHODS:
        dd = data[data[col].notna()]
        m = metrics(dd[col], dd["exp_pI"])
        m.update(Method=name, Category=cat)
        rows.append(m)
    t = pd.DataFrame(rows)[["Category", "Method", "n", "RMSE", "MAE", "bias",
                            "R2_coef_det", "Pearson_r", "r2", "n_|err|>0.5"]]
    t = t.sort_values("RMSE").reset_index(drop=True)
    t.insert(0, "Rank", range(1, len(t) + 1))
    t.to_csv(os.path.join(OUT, f"bench_{tag}_metrics.csv"),
             index=False, encoding="utf-8-sig")
    return t


A_mask = d.conf_class.astype(str).str.startswith("A")
dA = d[A_mask].copy()
dA.to_csv(os.path.join(OUT, "bench_A61_per_protein.csv"),
          index=False, encoding="utf-8-sig")

t61 = bench(dA, "A61")
t56 = bench(dA[dA.source == "exp_fixed"], "A56")

print("=" * 108)
print("A-class high-confidence experimental structure subset n=61 (structure model input=exp PDB; sequence models do not use structure)")
print("=" * 108)
print(t61.to_string(index=False))
print()
print("=" * 108)
print("Strict exp_fixed n=56")
print("=" * 108)
print(t56.to_string(index=False))
print()
print("Per-protein detail:", os.path.join(OUT, "bench_A61_per_protein.csv"))

# ---------------- Full-set 581 reference table ----------------
# Two conventions for structure models side by side:
#   (a) AF-primary convention: A/B/C/D experimental proteins use experimental PDB, P-class predicted proteins use AF2/ESMFold;
#   (b) Pure AF2 convention: all 581 proteins uniformly use AlphaFold2 structures (PypKa batch run z180, unfinished are NaN).
da = (afm
      .merge(pk[["seq_no", "pI_pKALM"]], on="seq_no", how="left"))
# afm already contains source/conf_class, no need to merge reg again
da["pI_pureThurl"] = da.seq_no.map(lambda s: round(thurlkill_pi(seqs[int(s)]), 4))
da["pI_pIESM"] = da.seq_no.map(lambda s: round(pI_ESM.get(int(s)), 4))
if os.path.exists(paf_path):
    paf = pd.read_csv(paf_path)[["seq_no", "pI_pypka_AF2"]]
    da = da.merge(paf, on="seq_no", how="left")
else:
    da["pI_pypka_AF2"] = np.nan

# 581 per-protein full-method detail (including PypKa AF2 column, unfinished are NaN)
af_cols = [c for c in afm.columns if c != "seq_no"]
keep_cols = ["seq_no", "source", "conf_class", "exp_pI",
             "pI_pIESM", "pI_pureThurl", "pI_pKALM", "pI_pypka_AF2"] + af_cols
keep_cols = list(dict.fromkeys(keep_cols))   # af_cols contains source/conf_class/exp_pI, deduplicate preserving order
da[keep_cols].to_csv(os.path.join(OUT, "bench_all581_per_protein.csv"),
                     index=False, encoding="utf-8-sig")

METHODS_ALL = [
    ("pI-ESM (this_work)",        "pI_pIESM",               "Sequence"),
    ("IPC2 SVR",                 "M8_IPC2_svr",            "Sequence"),
    ("DeepKa+SA+IPC2 (AF_primary)",  "M5_DeepKa_SA_ipc_AF",    "Structure"),
    ("IPC2 9-param",             "M3_ipc_9param_AF",       "Sequence"),
    ("Our 9-param",              "M3_our_9param_AF",       "Sequence"),
    ("pKALM PIPROT",             "pI_pKALM",               "Sequence"),
    ("DeepKa+Thurl (AF_primary)",    "M1_DeepKa_Thurl_AF",     "Structure"),
    ("Thurlkill_baseline",       "pI_pureThurl",           "Seq_baseline"),
    ("PropKa+Thurl (AF_primary)",    "M2_PropKa_Thurl_AF",     "Structure"),
    ("PypKa+Thurl (pure_AF2)",      "pI_pypka_AF2",           "Structure"),
]
rows = []
for name, col, cat in METHODS_ALL:
    dd = da[da[col].notna()]
    m = metrics(dd[col], dd["exp_pI"])
    m.update(Method=name, Category=cat)
    rows.append(m)
t581 = pd.DataFrame(rows)[["Category", "Method", "n", "RMSE", "MAE", "bias",
                           "R2_coef_det", "Pearson_r", "r2", "n_|err|>0.5"]]
t581 = t581.sort_values("RMSE").reset_index(drop=True)
t581.insert(0, "Rank", range(1, len(t581) + 1))
t581.to_csv(os.path.join(OUT, "bench_all581_metrics.csv"),
            index=False, encoding="utf-8-sig")
print()
print("=" * 108)
print(f"Full set test_25 n=581 (PypKa pure AF2 completed {int(da.pI_pypka_AF2.notna().sum())}/581)")
print("=" * 108)
print(t581.to_string(index=False))
print()
print("581 per-protein detail:", os.path.join(OUT, "bench_all581_per_protein.csv"))
