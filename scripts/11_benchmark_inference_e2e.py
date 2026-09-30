# -*- coding: utf-8 -*-
"""
End-to-end inference time benchmark on the 581-sequence test set (IPC_protein_25).

Measures the TRUE per-protein latency from raw amino acid sequence -> pI value:
- IPC2:         seq -> seq_counts -> 9-pKa bisection (no ML, pure Python+numpy, CPU)
- IPC2.svr.19:  seq -> 19 pKa-scale pI features (bisection per scale) -> SVR predict (CPU)
- ESMpI (GPU): seq -> ESM-2 150M transformer forward (CUDA) -> 7-type pool -> SVR predict
- ESMpI (CPU): seq -> ESM-2 150M transformer forward (CPU) -> 7-type pool -> SVR predict

This is the latency a real user experiences when calling predict.py.
"""
import sys, pathlib, time, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from esmpi.dataio import load_data, seq_counts, DATA_DIR
from esmpi.pka_engine import compute_pI, PKA_IPC2_PAPER
from esmpi.pooling import W_ELEGANT
from esmpi.models import F19, make_esm_svr, make_f19_svr, fit_weighted

# ---------- Load test sequences ----------
d = load_data()
ftr, fte = d['ftr'], d['fte']
ytr, sw = d['ytr'], d['sw']
keys_te = d['keys_te']  # amino acid sequences as strings
lengths = np.array([len(k) for k in keys_te], dtype=int)
N = len(keys_te)
print(f'Test set: {N} sequences, length range {lengths.min()}-{lengths.max()}')

# ---------- Method 1: IPC2 (9-pKa bisection, per-protein) ----------
IPC2_PKA = PKA_IPC2_PAPER
t_ipc2 = np.zeros(N)
for i in range(N):
    seq = keys_te[i]
    t0 = time.perf_counter()
    c = seq_counts(seq).reshape(1, -1)
    _ = compute_pI(IPC2_PKA, c)
    t_ipc2[i] = time.perf_counter() - t0

# ---------- Method 2: IPC2.svr.19 ----------
Xtr_f19 = ftr[F19].values.astype(np.float64)
Xte_f19 = fte[F19].values.astype(np.float64)
svr19 = make_f19_svr()
fit_weighted(svr19, Xtr_f19, ytr, sw)

t_svr19 = np.zeros(N)
for i in range(N):
    seq = keys_te[i]
    t0 = time.perf_counter()
    c = seq_counts(seq).reshape(1, -1)
    for _ in range(19):
        _ = compute_pI(IPC2_PKA, c)
    _ = svr19.predict(Xte_f19[i:i+1])
    t_svr19[i] = time.perf_counter() - t0

# ---------- Method 3: ESMpI on GPU and CPU ----------
print('Loading ESM-2 150M ...')
os.environ['HF_HUB_OFFLINE'] = '1'  # use local cache, no network
import torch
from transformers import EsmTokenizer, EsmModel

model_name = 'facebook/esm2_t30_150M_UR50D'
tokenizer = EsmTokenizer.from_pretrained(model_name)

# Train SVR on precomputed train embeddings (do once)
from esmpi.pooling import load_emb_pool
Xtr_emb, Xte_emb = load_emb_pool('ESM2-150M', W_ELEGANT)
btr = ftr['pI_IPC2_protein'].values.astype(np.float64)
bte = fte['pI_IPC2_protein'].values.astype(np.float64)
rtr = ytr - btr
svr_esm = make_esm_svr()
fit_weighted(svr_esm, Xtr_emb, rtr, sw)

SITE7 = ['D', 'E', 'H', 'K', 'C', 'Y', 'R']
H = 640


def run_esm_bench(device_name):
    """Time ESM-2 forward + pool + SVR predict on the given device."""
    device = torch.device(device_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()
    print(f'ESM-2 loaded on {device_name.upper()}')

    # Warmup (3 forward passes on first sequence to stabilize timing)
    warm_seq = keys_te[0]
    inp = tokenizer(warm_seq, return_tensors='pt', add_special_tokens=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inp)
    if device_name == 'cuda':
        torch.cuda.synchronize()

    t_arr = np.zeros(N)
    print(f'Timing {N} sequences on {device_name.upper()} ...')
    for i in range(N):
        seq = keys_te[i]
        if device_name == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # --- ESM-2 forward ---
        inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        emb = out.last_hidden_state[0, 1:-1, :].cpu().numpy()  # (L, 640)
        # --- 7-type pool ---
        pt = np.zeros((7, H), dtype=np.float64)
        cn = np.zeros(7, dtype=np.float64)
        for j, res in enumerate(SITE7):
            mask = np.array([c == res for c in seq])
            if mask.sum() > 0:
                pt[j] = emb[mask].mean(axis=0)
                cn[j] = mask.sum()
        wc = cn * W_ELEGANT
        total = wc.sum()
        pooled = (np.einsum('k,kh->h', wc, pt) / total if total > 0
                 else np.zeros(H, dtype=np.float32)).astype(np.float32).reshape(1, -1)
        # --- SVR predict ---
        _ = bte[i] + svr_esm.predict(pooled)
        if device_name == 'cuda':
            torch.cuda.synchronize()
        t_arr[i] = time.perf_counter() - t0
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{N}', flush=True)
    return t_arr


t_esmpi_gpu = run_esm_bench('cuda')
t_esmpi_cpu = run_esm_bench('cpu')

# ---------- Save CSV ----------
out_csv = ROOT / 'results' / 'tables' / 'bench_inference_time_e2e.csv'
out_csv.parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame({
    'seq_no': np.arange(1, N+1),
    'length': lengths,
    't_ipc2_s': t_ipc2,
    't_svr19_s': t_svr19,
    't_esmpi_gpu_s': t_esmpi_gpu,
    't_esmpi_cpu_s': t_esmpi_cpu,
})
df.to_csv(str(out_csv), index=False, encoding='utf-8-sig')
print(f'\nSaved {out_csv}')
print(f'IPC2          median {np.median(t_ipc2)*1e3:.3f} ms  mean {np.mean(t_ipc2)*1e3:.3f} ms')
print(f'IPC2.svr.19   median {np.median(t_svr19)*1e3:.3f} ms  mean {np.mean(t_svr19)*1e3:.3f} ms')
print(f'ESMpI (GPU)  median {np.median(t_esmpi_gpu)*1e3:.3f} ms  mean {np.mean(t_esmpi_gpu)*1e3:.3f} ms')
print(f'ESMpI (CPU)  median {np.median(t_esmpi_cpu)*1e3:.3f} ms  mean {np.mean(t_esmpi_cpu)*1e3:.3f} ms')

# ---------- Plot ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

# Original colors
C_GRAY = '#95A5A6'   # IPC2
C_BLUE = 'steelblue'  # IPC2.svr.19
C_BROWN = '#dd8452'  # ESMpI GPU
C_RED = '#c0392b'    # ESMpI CPU

# Font/line +30%
F_TICK = 11 * 1.3   # = 14.3
F_LABEL = 12 * 1.3  # = 15.6
F_LEGEND = 8 * 1.3  # = 10.4
# Spine/tick linewidth +50%
LW_SPINE = 1.0 * 1.5  # = 1.5
LW_TICK = 0.8 * 1.5   # = 1.2

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.scatter(lengths, t_ipc2, s=25, alpha=0.6, color=C_GRAY,
           edgecolor='white', linewidth=0.2, marker='s',
           label='IPC2')
ax.scatter(lengths, t_svr19, s=25, alpha=0.6, color=C_BLUE,
           edgecolor='white', linewidth=0.2, marker='o',
           label='IPC2.svr.19')
ax.scatter(lengths, t_esmpi_gpu, s=25, alpha=0.6, color=C_BROWN,
           edgecolor='white', linewidth=0.2, marker='^',
           label='ESMpI GPU')
ax.scatter(lengths, t_esmpi_cpu, s=25, alpha=0.5, color=C_RED,
           edgecolor='white', linewidth=0.2, marker='D',
           label='ESMpI CPU')
ax.set_xlabel('Sequence length', fontsize=F_LABEL)
ax.set_ylabel('Time (s)', fontsize=F_LABEL)
ax.set_ylim(-0.5, 6.5)
ax.set_yticks([0, 2, 4, 6])
ax.tick_params(axis='both', labelsize=F_TICK, width=LW_TICK, length=5)
for sp in ('top', 'bottom', 'left', 'right'):
    ax.spines[sp].set_linewidth(LW_SPINE)
ax.legend(frameon=False, loc='upper left', fontsize=F_TICK)
fig.tight_layout()
out_png = ROOT / 'results' / 'figures' / 'fig_inference_time_e2e.png'
fig.savefig(str(out_png), dpi=300)
plt.close(fig)
print(f'Saved {out_png}')

si_dir = pathlib.Path(r'd:\projects\aiProject1\z102_reproduce\paper_work\Draft\figure\si')
shutil.copy(str(out_png), str(si_dir / 'SI_figure_inference_time.png'))
print(f'Copied to {si_dir / "SI_figure_inference_time.png"}')
