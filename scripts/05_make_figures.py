# -*- coding: utf-8 -*-
"""
05_make_figures: generate paper Figures 2, 3, 4 and SI Figures S1-S4.

Outputs (results/figures/):
  paper_figures/paper_fig2_scatter.{png,pdf}    main Fig 2 (All + Outlier fits)
  paper_figures/paper_fig3_effects.{png,pdf}    main Fig 3 (4 panels, heat A)
  paper_figures/paper_fig4_bootstrap.{png,pdf}  main Fig 4 (pKa bootstrap)
  si/si_figS1_pI_distribution.{png,pdf}         SI Fig S1 train/test pI hist
  si/si_figS2_pairwise.{png,pdf}                SI Fig S2 pairwise correlation
  si/si_figS3_source_distribution.{png,pdf}     SI Fig S3 PIP-DB / SWISS full hist
  si/si_figS4_perm_pka.{png,pdf}                SI Fig S4 permutation pKa dist

Data notes:
  - Source labels come from data/mapping/z46d_source_labels.csv (CORRECTED labels:
    full set SWISS-2DPAGE=944, PIP-DB=1380; test subset 350 / 231).
  - Fig 3 panel A uses the GBMS re-optimized physical baseline (pI_our).
  - SI Fig S4 reads results bootstrap samples if present, otherwise falls
    back to the frozen 1000-iteration copy in data/reference_results/.

NOTE: The letters_at_ylabel / _panel_letters pixel-alignment hacks are kept
UNCHANGED — fragile but they produce the published figures.
"""
import sys, os, re, ast, pathlib
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from piesm import dataio

# ---------------- global journal style ----------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 0.9,
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
    'xtick.major.size': 3.2, 'ytick.major.size': 3.2,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
# ---- low-saturation palette ----
C_REF_RED  = '#B32E1B'   # Fig 4 IPC2 dashed line (published figure)
C_OURS     = '#E06A6F'   # pI-ESM bars / SI S4 IPC2 line
C_CI       = '#DBD7B0'   # 95% CI shade
C_SVR      = '#D9D9D9'   # IPC2.svr bars
C_PT       = '#6E97B3'   # scatter fill
C_PT_ED    = '#35556B'   # scatter edge
C_BR_ED    = '#9A5442'   # outlier edge
C_GRAY     = '#9A9A9A'   # diagonal
C_OUTLIER  = '#E67E22'   # outlier OLS line
DASH       = (0, (3.0, 3.2))

# ---------------- paths ----------------
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PREDICTIONS = ROOT / "data" / "predictions"
MAPPING = ROOT / "data" / "mapping"
FULLSRC = ROOT / "data" / "full_sources"
REFERENCE = ROOT / "data" / "reference_results"
BOOTSTRAP = RESULTS / "bootstrap"
PFDIR = FIGURES / "paper_figures"
SIDIR = FIGURES / "si"
for _p in (PFDIR, SIDIR):
    os.makedirs(_p, exist_ok=True)
DPI = 600

# ---------------- data ----------------
d = dataio.load_data()
yte = d['yte']; keys_te = d['keys_te']; ytr = d['ytr']
fte = d['fte']
preds = np.load(str(PREDICTIONS / 'z166_preds.npz'))
ours = preds['pt_ours']; f19 = preds['pt_f19']; bt_ipc2 = preds['bt_ipc2']
bt = fte['pI_our'].values.astype(float)          # GBMS re-optimized baseline
lens = np.array([len(k) for k in keys_te], dtype=float)

lab = pd.read_csv(str(MAPPING / 'z46d_source_labels.csv'))
mp = pd.read_csv(str(MAPPING / 'z164_seqno2zidx.csv'))
src_df = mp.merge(lab[['seq_no_full', 'source']], left_on='seq_no',
                  right_on='seq_no_full', how='left').sort_values('zidx')
src = src_df['source'].values
assert len(src) == 581 and pd.Series(src).notna().all()

rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

LIM = (3, 11.5)
TICKS2 = np.arange(3, 12, 2)   # 3,5,7,9,11


def yticks_in(ax, labelsize=10):
    ax.tick_params(axis='y', direction='in', length=3.4, labelsize=labelsize)


def letters_at_ylabel(fig, axes_grid, fontsize=14, dy=0.004,
                      spine_no_ylabel=False):
    """Panel letters centred on the ylabel ink (two-pass renderer calibration)."""
    renderer = fig.canvas.get_renderer()

    def buf_gray():
        return np.asarray(fig.canvas.buffer_rgba())[:, :, 0]

    def ink_clusters(x_lo, x_hi, y_lo, y_hi, gray, thresh=160, gap_px=14):
        r_hi = max(0, int(ch - y_hi)); r_lo = min(gray.shape[0], int(ch - y_lo))
        strip = gray[r_hi:r_lo, int(x_lo):int(x_hi)]
        cols = np.where((strip < thresh).any(axis=0))[0]
        if not len(cols):
            return [], strip
        clusters, s, p = [], cols[0], cols[0]
        for c in cols[1:]:
            if c - p > gap_px:
                clusters.append((s, p)); s = c
            p = c
        clusters.append((s, p))
        return clusters, strip

    def label_center(ax):
        lab_obj = ax.yaxis.get_label()
        if not lab_obj.get_text():
            return None
        bb = lab_obj.get_window_extent(renderer=renderer)
        x_lo = bb.x0 - 15
        x_hi = min(bb.x1 + 15,
                   ax.get_position().x0 * cw - 3.4 * fig.dpi / 72.0 - 3)
        cls, strip = ink_clusters(x_lo, x_hi, bb.y0 - 3, bb.y1 + 3, buf)
        best, best_span = None, 0
        for a, b in cls:
            ys = np.where((strip[:, a:b + 1] < 160).any(axis=1))[0]
            span = (ys.max() - ys.min()) if len(ys) else 0
            if span > best_span:
                best, best_span = (a, b), span
        return x_lo + (best[0] + best[1]) / 2 if best else None

    fig.canvas.draw()
    buf = buf_gray(); ch, cw = buf.shape
    nrows = len(axes_grid); ncols = len(axes_grid[0])
    col_x0, col_tgt = [], []
    for c in range(ncols):
        pos = axes_grid[0][c].get_position()
        col_x0.append(pos.x0)
        tgt = None
        for r in range(nrows):
            tgt = label_center(axes_grid[r][c])
            if tgt is not None:
                break
        col_tgt.append(tgt)
    gap = col_x0[0] * cw - col_tgt[0]

    txts = {}
    for r in range(nrows):
        for c in range(ncols):
            ax = axes_grid[r][c]
            pos = ax.get_position()
            y1 = pos.y1
            if spine_no_ylabel and not ax.yaxis.get_label().get_text():
                sx = pos.x0 * cw
                cls_t, _ = ink_clusters(sx - 205, sx - 5,
                                        pos.y0 * ch + 2, pos.y1 * ch - 2, buf)
                tgt = sx - 205 + cls_t[0][0] if cls_t else sx
                ha = 'left'
            else:
                tgt = col_tgt[c] if col_tgt[c] is not None else col_x0[c] * cw - gap
                ha = 'center'
            t = fig.text(tgt / cw, y1 + dy, chr(65 + r * ncols + c),
                         fontsize=fontsize, fontweight='bold',
                         ha=ha, va='bottom')
            txts[(r, c)] = (t, tgt, ha)
    fig.canvas.draw(); buf = buf_gray()
    fpx = fontsize * fig.dpi / 72.0
    for (r, c), (t, tgt, ha) in txts.items():
        pos = axes_grid[r][c].get_position()
        y_lo = (pos.y1 + dy) * ch
        y_hi = (pos.y1 + dy) * ch + fpx * 1.15
        if ha == 'left':
            x_lo, x_hi = tgt - 6, tgt + fpx * 2.2
        else:
            x_lo, x_hi = tgt - fpx, min(tgt + fpx, pos.x0 * cw - 6)
        cls, _ = ink_clusters(x_lo, x_hi, y_lo, y_hi, buf, gap_px=8)
        if cls:
            if ha == 'left':
                got = x_lo + cls[0][0]
            else:
                got = tgt - fpx + (cls[0][0] + cls[0][1]) / 2
            t.set_position(((2 * tgt - got) / cw, t.get_position()[1]))


def _panel_letters(fig, axes_grid, labels=None, fontsize=12, dy=0.004,
                   left_shift_pts=0.0):
    """Fig 3 panel letters: left-column letters align with the ylabel ink LEFT
    edge, right-column letters with the ink CENTER. Extra left shift in pt."""
    renderer = fig.canvas.get_renderer()

    def buf_gray():
        return np.asarray(fig.canvas.buffer_rgba())[:, :, 0]

    def ink_clusters(x_lo, x_hi, y_lo, y_hi, gray, thresh=160, gap_px=14):
        r_hi = max(0, int(ch - y_hi)); r_lo = min(gray.shape[0], int(ch - y_lo))
        strip = gray[r_hi:r_lo, int(x_lo):int(x_hi)]
        cols = np.where((strip < thresh).any(axis=0))[0]
        if not len(cols):
            return [], strip
        clusters, s, p = [], cols[0], cols[0]
        for c in cols[1:]:
            if c - p > gap_px:
                clusters.append((s, p)); s = c
            p = c
        clusters.append((s, p))
        return clusters, strip

    def ylabel_left_edge(ax):
        lab = ax.yaxis.get_label()
        if not lab.get_text():
            return None
        bb = lab.get_window_extent(renderer=renderer)
        x_lo = bb.x0 - 15
        x_hi = min(bb.x1 + 15, ax.get_position().x0 * cw
                   - 3.4 * fig.dpi / 72.0 - 3)
        cls, strip = ink_clusters(x_lo, x_hi, bb.y0 - 3, bb.y1 + 3, buf)
        best, best_span = None, 0
        for a, b in cls:
            ys = np.where((strip[:, a:b + 1] < 160).any(axis=1))[0]
            span = (ys.max() - ys.min()) if len(ys) else 0
            if span > best_span:
                best, best_span = (a, b), span
        return (x_lo + best[0]) if best else None

    def ylabel_center(ax):
        lab = ax.yaxis.get_label()
        if not lab.get_text():
            return None
        bb = lab.get_window_extent(renderer=renderer)
        x_lo = bb.x0 - 15
        x_hi = min(bb.x1 + 15, ax.get_position().x0 * cw
                   - 3.4 * fig.dpi / 72.0 - 3)
        cls, strip = ink_clusters(x_lo, x_hi, bb.y0 - 3, bb.y1 + 3, buf)
        best, best_span = None, 0
        for a, b in cls:
            ys = np.where((strip[:, a:b + 1] < 160).any(axis=1))[0]
            span = (ys.max() - ys.min()) if len(ys) else 0
            if span > best_span:
                best, best_span = (a, b), span
        return x_lo + (best[0] + best[1]) / 2 if best else None

    fig.canvas.draw()
    buf = buf_gray(); ch, cw = buf.shape
    nrows = len(axes_grid); ncols = len(axes_grid[0])

    col_tgt = []
    for c in range(ncols):
        edges = []
        for r in range(nrows):
            fn = ylabel_left_edge if c == 0 else ylabel_center
            v = fn(axes_grid[r][c])
            if v is not None:
                edges.append(v)
        col_tgt.append(min(edges) if edges else None)

    if np.isscalar(left_shift_pts):
        row_shift = [left_shift_pts] * nrows
    else:
        row_shift = list(left_shift_pts)

    txts = {}
    for r in range(nrows):
        for c in range(ncols):
            ax = axes_grid[r][c]
            y1 = ax.get_position().y1
            if c == 0:
                ha = 'left'; tgt = col_tgt[c]
                if tgt is not None:
                    tgt = tgt - row_shift[r] * fig.dpi / 72.0
            else:
                ha = 'center'; tgt = col_tgt[c]
            t = fig.text(tgt / cw, y1 + dy,
                         labels[r][c] if labels is not None
                         else chr(65 + r * ncols + c),
                         fontsize=fontsize, fontweight='bold',
                         ha=ha, va='bottom')
            txts[(r, c)] = (t, tgt, ha)
    fig.canvas.draw(); buf = buf_gray()
    fpx = fontsize * fig.dpi / 72.0
    for (r, c), (t, tgt, ha) in txts.items():
        pos = axes_grid[r][c].get_position()
        y_lo = (pos.y1 + dy) * ch
        y_hi = (pos.y1 + dy) * ch + fpx * 1.15
        x_lo, x_hi = tgt - fpx, min(tgt + fpx, pos.x0 * cw - 6)
        cls, _ = ink_clusters(x_lo, x_hi, y_lo, y_hi, buf, gap_px=8)
        if cls:
            got = tgt - fpx + (cls[0][0] + cls[0][1]) / 2
            t.set_position(((2 * tgt - got) / cw, t.get_position()[1]))


# ============================================================
# Figure 2: pI-ESM vs Expt pI, All + Outlier OLS lines
# ============================================================
def fig2():
    k, b = np.polyfit(yte, ours, 1)
    out = np.abs(ours - yte) > 0.5
    k_out, b_out = np.polyfit(yte[out], ours[out], 1)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.scatter(yte[~out], ours[~out], s=15, facecolor=C_PT, edgecolor=C_PT_ED,
               linewidths=0.45, alpha=0.85)
    ax.scatter(yte[out], ours[out], s=17, facecolor='none', edgecolor=C_BR_ED,
               linewidths=0.9)
    ax.plot(LIM, LIM, ls=DASH, color=C_GRAY, lw=1.1)
    xx = np.linspace(3.5, 11.0, 50)
    ax.plot(xx, k * xx + b, '-', color='black', lw=1.7, label='All')
    ax.plot(xx, k_out * xx + b_out, '-', color=C_OUTLIER, lw=1.7, label='Outlier')
    ax.set_xlim(LIM); ax.set_ylim(LIM)
    ax.set_xticks(TICKS2); ax.set_yticks(TICKS2)
    ax.set_xlabel('Expt. pI')
    ax.set_ylabel('pI-ESM')
    ax.legend(loc='upper left', frameon=False, handlelength=1.6)
    for ext in ('png', 'pdf'):
        fig.savefig(str(PFDIR / f'paper_fig2_scatter.{ext}'), dpi=DPI,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('saved Figure 2')


# ============================================================
# Figure 3: A GBMS-vs-pI-ESM heat scatter; B pI acid/base;
#           C data source; D length (panels A/C top, B/D bottom)
# ============================================================
def fig3():
    fig, axes = plt.subplots(2, 2, figsize=(3.46, 3.46), dpi=DPI)
    plt.subplots_adjust(left=0.205, right=0.985, bottom=0.155, top=0.90,
                        wspace=0.46, hspace=0.58)

    # ---- A: GBMS physical pI vs pI-ESM (density heat scatter) ----
    axa = axes[0][0]
    for _sp in axa.spines.values():
        _sp.set_zorder(10)
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    kde = gaussian_kde(np.vstack([bt, ours]), bw_method=0.16)
    dens = kde(np.vstack([bt, ours]))
    cmap = LinearSegmentedColormap.from_list(
        'dens', ['#0A1E3F', '#2C6BAB', '#8FB8DC', '#F08242', '#FF2400'])
    order = np.argsort(dens)
    norm_ = PowerNorm(0.45, vmin=0, vmax=dens.max())
    face = cmap(norm_(dens[order]))
    axa.scatter(np.asarray(bt)[order], np.asarray(ours)[order], s=7.5,
                facecolors=face, edgecolors=(0.5, 0.5, 0.5, 0.55),
                linewidths=0.15, zorder=3)
    axa.plot(LIM, LIM, linestyle=(0, (4.0, 3.0)), color='black', lw=1.0,
             zorder=2)
    axa.set_xlim(LIM); axa.set_ylim(LIM)
    axa.set_xticks(TICKS2); axa.set_yticks(TICKS2)
    axa.set_xlabel('IPC2 pI', fontsize=9)
    axa.set_ylabel('pI-ESM pI', fontsize=9)
    axa.tick_params(axis='x', labelsize=8, length=2.2)
    yticks_in(axa, 8)
    axa.tick_params(axis='y', length=2.2)

    def grouped_bars(ax, groups, ylabel=False, legend=False):
        xx = np.arange(len(groups)); w = 0.38
        rmse_m = lambda p, mk: float(np.sqrt(np.mean((p[mk] - yte[mk]) ** 2)))
        vo = [rmse_m(ours, mk) for _, mk in groups]
        vf = [rmse_m(f19, mk) for _, mk in groups]
        ax.bar(xx - w / 2, vo, w, color=C_OURS, edgecolor='none',
               label='pI-ESM', zorder=3)
        ax.bar(xx + w / 2, vf, w, facecolor=C_SVR, edgecolor='none',
               label='IPC2.svr.19', zorder=3)
        ax.set_xticks(xx)
        ax.set_xticklabels([n for n, _ in groups], fontsize=7.6,
                           linespacing=0.95)
        ax.set_ylim(0, 1.5)
        ax.set_yticks([0, 0.5, 1.0, 1.5])
        ax.tick_params(axis='x', labelsize=8, length=2.2)
        yticks_in(ax, 8)
        ax.tick_params(axis='y', length=2.2)
        if ylabel:
            ax.set_ylabel('RMSE', fontsize=9)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for name in ('left', 'bottom'):
            sp = ax.spines[name]
            sp.set_color('black'); sp.set_alpha(1); sp.set_linewidth(0.9)
            sp.set_zorder(10)
        ax.tick_params(axis='both', width=0.9, color='black')
        if legend:
            ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.12),
                      frameon=False, fontsize=6.8, borderaxespad=0.0,
                      borderpad=0.25, labelspacing=0.32,
                      handletextpad=0.4, handlelength=1.0)

    b_groups = [('pI < 7', yte < 7), ('pI \u2265 7', yte >= 7)]
    c_groups = [('SWISS-\n2DPAGE', src == 'SWISS-2DPAGE'),
                ('PIP-DB', src == 'PIP-DB')]
    d_groups = [('S', lens <= 200),
                ('M', (lens > 200) & (lens <= 350)),
                ('L', lens > 350)]
    grouped_bars(axes[1][0], b_groups, ylabel=True)              # B
    grouped_bars(axes[0][1], c_groups, ylabel=True, legend=True)  # C
    for _t in axes[0][1].get_xticklabels():
        _t.set_linespacing(1.25)
    grouped_bars(axes[1][1], d_groups)                            # D

    axes[0][0].yaxis.labelpad = 8.5 - 2.25
    _panel_letters(fig, axes, labels=[['A', 'C'], ['B', 'D']], fontsize=12,
                   left_shift_pts=[-4.0, -3.5])

    for ext in ('png', 'pdf'):
        fig.savefig(str(PFDIR / f'paper_fig3_effects.{ext}'), dpi=DPI,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('saved Figure 3')


# ============================================================
# Figure 4: 3x3 bootstrap pKa distributions (published style)
# ============================================================
DIMS9 = ['N_TER', 'C_TER', 'asp', 'glu', 'his', 'lys', 'cys', 'tyr', 'arg']
TITLES9 = {'N_TER': 'N-terminus', 'C_TER': 'C-terminus', 'asp': 'Asp',
           'glu': 'Glu', 'his': 'His', 'lys': 'Lys', 'cys': 'Cys',
           'tyr': 'Tyr', 'arg': 'Arg'}
FBINS = np.linspace(3, 14, 41)
IPC2_PKA = {'N_TER': 5.779, 'C_TER': 6.065, 'asp': 3.766, 'glu': 4.497,
            'his': 5.492, 'lys': 9.247, 'cys': 7.890, 'tyr': 11.491,
            'arg': 10.223}


def _pka_grid(samp, summ, vline_color, vline_label, outbase,
              y10_dims=('glu',), legend_loc_panel=(0, 2)):
    """Shared 3x3 pKa-distribution panel (main Fig 4 and SI Fig S4)."""
    fig, axes = plt.subplots(3, 3, figsize=(7.09, 6.9), dpi=DPI)
    plt.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.925,
                        wspace=0.19, hspace=0.36)
    for j, dim in enumerate(DIMS9):
        ax = axes[j // 3][j % 3]
        col = samp[dim].values
        ax.axvspan(float(summ.loc[dim].ci_lo), float(summ.loc[dim].ci_hi),
                   color=C_CI, alpha=0.35, lw=0, zorder=1,
                   label='95% confidence interval')
        ax.hist(np.clip(col, 3.01, 13.99), bins=FBINS,
                weights=np.ones(len(col)) / len(col), facecolor='white',
                edgecolor='black', linewidth=1.2, zorder=2)
        ax.axvline(float(summ.loc[dim].ipc2) if 'ipc2' in summ.columns
                   else IPC2_PKA[dim],
                   color=vline_color, lw=1.6, ls='--', zorder=6,
                   label=vline_label)
        ax.set_xlim(3, 14)
        ax.set_xticks(np.arange(4, 15, 2))
        if dim in y10_dims:
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_yticks([0.25, 0.75], minor=True)
        else:
            ax.set_ylim(0, 0.6)
            ax.set_yticks([0, 0.3, 0.6])
            ax.set_yticks([0.15, 0.45], minor=True)
        if j // 3 == 2:
            ax.text(0.04, 0.90, TITLES9[dim], transform=ax.transAxes,
                    ha='left', va='top', fontsize=11.5)
        else:
            ax.text(0.96, 0.90, TITLES9[dim], transform=ax.transAxes,
                    ha='right', va='top', fontsize=11.5)
        ax.tick_params(axis='x', labelsize=9)
        yticks_in(ax, 9)
        ax.tick_params(axis='y', which='minor', direction='in', length=1.7,
                       width=0.9)
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)
        if j % 3 == 0:
            ax.set_ylabel('Probability', fontsize=10.5)
        if j // 3 == 2:
            ax.set_xlabel('$\\mathrm{p}K_{\\mathrm{a}}$', fontsize=10.5)
    r, c = legend_loc_panel
    axes[r][c].legend(loc='center right', fontsize=8.4, frameon=False,
                      borderpad=0.3, labelspacing=0.3, handletextpad=0.4)
    letters_at_ylabel(fig, axes, fontsize=15, dy=0.012,
                      spine_no_ylabel=True)
    for ext in ('png', 'pdf'):
        fig.savefig(str(PFDIR / f'{outbase}.{ext}'), dpi=DPI,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fig4():
    samp = pd.read_csv(str(BOOTSTRAP / 'z168_bootstrap_pKa.csv'))
    summ = pd.read_csv(str(BOOTSTRAP / 'z168_bootstrap_summary.csv')).set_index('dim')
    _pka_grid(samp, summ, vline_color=C_REF_RED,
              vline_label='IPC2 $\\mathrm{p}K_{\\mathrm{a}}$',
              outbase='paper_fig4_bootstrap', y10_dims=('glu',))
    print('saved Figure 4')


# ============================================================
# SI Figure S1: train/test experimental pI distribution
# ============================================================
def fig_s1():
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(1, 12.5, 46)
    ax.hist(ytr, bins=bins, alpha=0.55, label=f'Train (n={len(ytr)})',
            color='#4A90D9', edgecolor='white')
    ax.hist(yte, bins=bins, alpha=0.55, label=f'Test (n={len(yte)})',
            color=C_OUTLIER, edgecolor='white')
    ax.set_xlabel('Experimental pI'); ax.set_ylabel('Count'); ax.legend()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(str(SIDIR / f'si_figS1_pI_distribution.{ext}', ), dpi=DPI)
    plt.close(fig)
    print('saved SI Figure S1')


# ============================================================
# SI Figure S2: pairwise correlation
# ============================================================
def fig_s2():
    from scipy.stats import pearsonr
    cols = [('Experimental pI', yte), ('IPC2 9-pKa', bt),
            ('IPC2.SVR (F19)', f19), ('pI-ESM', ours)]
    P = np.column_stack([c[1] for c in cols])
    fig, axes = plt.subplots(4, 4, figsize=(11, 10))
    for i in range(4):
        for j in range(4):
            ax = axes[i][j]
            if i == j:
                ax.hist(P[:, i], bins=30, color='#4A90D9', alpha=0.75,
                        edgecolor='white')
            elif i > j:
                ax.scatter(P[:, j], P[:, i], s=5, alpha=0.35, color='#34495E')
                lim = [P[:, [i, j]].min() - .3, P[:, [i, j]].max() + .3]
                ax.plot(lim, lim, 'r--', lw=1)
                ax.set_xlim(lim); ax.set_ylim(lim)
                r = pearsonr(P[:, j], P[:, i])[0]
                ax.text(.05, .92, f'r={r:.3f}', transform=ax.transAxes,
                        fontsize=9, va='top')
            if i == 3:
                ax.set_xlabel(cols[j][0], fontsize=8)
            if j == 0 and i > 0:
                ax.set_ylabel(cols[i][0], fontsize=8)
    fig.suptitle('Pairwise comparison on independent test set (n=581)')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ('png', 'pdf'):
        fig.savefig(str(SIDIR / f'si_figS2_pairwise.{ext}'), dpi=DPI)
    plt.close(fig)
    print('saved SI Figure S2')


# ---------------- full-source FASTA parsers (isoelectric.org) ----
def _parse_swiss_pi(fasta_path):
    """SWISS-2DPAGE header: >P31947-1|['4.60/26759', ...] -> mean gel pI."""
    pis = []
    with open(fasta_path, encoding='utf-8') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            m = re.search(r'\|(\[.*\])\s*$', line.strip())
            if not m:
                continue
            try:
                lst = ast.literal_eval(m.group(1))
                vals = [float(x.split('/')[0]) for x in lst]
                pis.append(sum(vals) / len(vals))
            except Exception:
                continue
    return np.array(pis, dtype=float)


def _parse_pip_pi(fasta_path):
    """PIP-DB header: >id|acc pI pI -> last value (mean literature pI)."""
    pis = []
    with open(fasta_path, encoding='utf-8') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            parts = line[1:].strip().split()
            try:
                pis.append(float(parts[-1]))
            except Exception:
                continue
    return np.array(pis, dtype=float)


# ============================================================
# SI Figure S3: PIP-DB / SWISS-2DPAGE full-source distributions
# ============================================================
def fig_s3_source():
    swiss = _parse_swiss_pi(FULLSRC / 'ch2d19_2_1st_isoform.fasta')
    pip = _parse_pip_pi(FULLSRC / 'pip_db_normal.fasta')
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(3, 12, 37)
    ax.hist(pip, bins=bins, alpha=0.85, label=f'PIP-DB (n={len(pip)})',
            color='#89A6C7', edgecolor='white')
    ax.hist(swiss, bins=bins, alpha=0.75,
            label=f'SWISS-2DPAGE (n={len(swiss)})',
            color='#C97B5A', edgecolor='white')
    ax.set_xlabel('Experimental pI')
    ax.set_ylabel('Count')
    ax.legend()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(str(SIDIR / f'si_figS3_source_distribution.{ext}'),
                    dpi=DPI)
    plt.close(fig)
    print(f'saved SI Figure S3 (SWISS={len(swiss)}, PIP-DB={len(pip)})')


# ============================================================
# SI Figure S4: permutation-experiment pKa distributions
# ============================================================
def fig_s4_perm():
    samp_csv = BOOTSTRAP / 'perm_pka_samples.csv'
    summ_csv = BOOTSTRAP / 'perm_pka_summary.csv'
    if not samp_csv.exists():           # frozen reference fallback
        samp_csv = REFERENCE / 'perm_pka_samples.csv'
        summ_csv = REFERENCE / 'perm_pka_summary.csv'
    samp = pd.read_csv(samp_csv)
    summ = pd.read_csv(summ_csv).set_index('dim')
    fig, axes = plt.subplots(3, 3, figsize=(7.09, 6.9), dpi=DPI)
    plt.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.925,
                        wspace=0.19, hspace=0.36)
    for j, dim in enumerate(DIMS9):
        ax = axes[j // 3][j % 3]
        col = samp[dim].values
        ax.axvspan(float(summ.loc[dim].ci_lo), float(summ.loc[dim].ci_hi),
                   color=C_CI, alpha=0.35, lw=0, zorder=1,
                   label='95% confidence interval')
        ax.hist(np.clip(col, 3.01, 13.99), bins=FBINS,
                weights=np.ones(len(col)) / len(col), facecolor='white',
                edgecolor='black', linewidth=1.2, zorder=2)
        ax.axvline(IPC2_PKA[dim], color=C_OURS, lw=1.6, ls='--', zorder=6,
                   label='IPC2 $\\mathrm{p}K_{\\mathrm{a}}$')
        ax.set_xlim(3, 14)
        ax.set_xticks(np.arange(4, 15, 2))
        if dim in ('glu', 'his'):
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_yticks([0.25, 0.75], minor=True)
        else:
            ax.set_ylim(0, 0.6)
            ax.set_yticks([0, 0.3, 0.6])
            ax.set_yticks([0.15, 0.45], minor=True)
        if j // 3 == 2:
            ax.text(0.04, 0.90, TITLES9[dim], transform=ax.transAxes,
                    ha='left', va='top', fontsize=11.5)
        else:
            ax.text(0.96, 0.90, TITLES9[dim], transform=ax.transAxes,
                    ha='right', va='top', fontsize=11.5)
        ax.tick_params(axis='x', labelsize=9)
        yticks_in(ax, 9)
        ax.tick_params(axis='y', which='minor', direction='in', length=1.7,
                       width=0.9)
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)
        if j % 3 == 0:
            ax.set_ylabel('Probability', fontsize=10.5)
        if j // 3 == 2:
            ax.set_xlabel('$\\mathrm{p}K_{\\mathrm{a}}$', fontsize=10.5)
    axes[0][2].legend(loc='center right', fontsize=8.4, frameon=False,
                      borderpad=0.3, labelspacing=0.3, handletextpad=0.4)
    letters_at_ylabel(fig, axes, fontsize=15, dy=0.012,
                      spine_no_ylabel=True)
    for ext in ('png', 'pdf'):
        fig.savefig(str(SIDIR / f'si_figS4_perm_pka.{ext}'), dpi=DPI,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('saved SI Figure S4')


if __name__ == '__main__':
    fig2(); fig3()
    fig_s1(); fig_s2(); fig_s3_source(); fig_s4_perm()
    if (BOOTSTRAP / 'z168_bootstrap_pKa.csv').exists() and \
       (BOOTSTRAP / 'z168_bootstrap_summary.csv').exists():
        fig4()
    else:
        print('skip Figure 4: run 03_bootstrap_pka.py first')
    print('ALL DONE ->', str(FIGURES))
