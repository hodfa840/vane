"""
VANE paper-quality plotting functions.

Reusable plotting utilities for generating EMNLP-formatted figures
from saved hidden states and checkpoint data.

All figures are sized for ACL/EMNLP double-column format:
  - single column: ~3.25 inches
  - double column: ~6.75 inches

Usage as library:
    from vane.plotting import plot_pca_trajectory, plot_metric_dotplot, plot_orthogonality

Usage as script:
    PYTHONPATH=. python3 core/plotting.py --output_dir emnlp
    PYTHONPATH=. python3 core/plotting.py --output_dir emnlp --hidden_states path/to/hidden_states.npz
"""

import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from pathlib import Path

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE       = '#2166ac'
RED        = '#b2182b'
GREEN      = '#1b7837'
LIGHT_BLUE = '#aec7e8'
LIGHT_RED  = '#f4a582'
GREY       = '#bababa'

# ── EMNLP sizing ─────────────────────────────────────────────────────────────
DOUBLE_COL_W = 6.75
SINGLE_COL_W = 3.25

RCPARAMS = {
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':          9,
    'axes.labelsize':     9,
    'axes.titlesize':     10,
    'legend.fontsize':    8,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'text.usetex':        False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
}


def apply_style():
    """Apply EMNLP paper style to matplotlib."""
    plt.rcParams.update(RCPARAMS)


# ── Sample selection ─────────────────────────────────────────────────────────

def compute_straightness(hidden):
    """
    Compute trajectory straightness for each sample.
    Straightness = chord_length / path_length (1.0 = perfectly straight).
    """
    N = hidden.shape[0]
    st = np.zeros(N)
    for i in range(N):
        hs = hidden[i].astype(np.float32)
        chord = np.linalg.norm(hs[-1] - hs[0])
        path = sum(np.linalg.norm(hs[l+1] - hs[l]) for l in range(len(hs) - 1))
        st[i] = chord / (path + 1e-9)
    return st


def pick_examples_by_straightness(hidden, labels):
    """
    Pick the straightest correct sample and most winding incorrect sample.
    Returns (correct_idx, incorrect_idx).
    """
    st = compute_straightness(hidden)
    correct_mask   = labels == 1
    incorrect_mask = labels == 0
    correct_idxs   = np.where(correct_mask)[0]
    incorrect_idxs = np.where(incorrect_mask)[0]

    best_correct    = correct_idxs[np.argmax(st[correct_mask])]
    worst_incorrect = incorrect_idxs[np.argmin(st[incorrect_mask])]
    return best_correct, worst_incorrect


# ── Figure: 3D PCA Trajectory ────────────────────────────────────────────────

def plot_pca_trajectory(hidden, labels, out_path,
                        model_name='Llama-3-8B', dataset='GSM8K'):
    """
    3D PCA trajectory of hidden states through transformer layers.
    Single-column width, clean style with transparent panes.
    """
    apply_style()
    ci, ii = pick_examples_by_straightness(hidden, labels)

    hs_correct   = hidden[ci].astype(np.float32)
    hs_incorrect = hidden[ii].astype(np.float32)
    n_layers = hs_correct.shape[0]

    combined = np.vstack([hs_correct, hs_incorrect])
    pca = PCA(n_components=3)
    projected = pca.fit_transform(combined)

    pc_c = projected[:n_layers]
    pc_i = projected[n_layers:]
    var_exp = pca.explained_variance_ratio_ * 100

    fig = plt.figure(figsize=(SINGLE_COL_W, 3.0))
    ax = fig.add_subplot(111, projection='3d')

    cmap_c = LinearSegmentedColormap.from_list('c', [LIGHT_BLUE, BLUE])
    cmap_i = LinearSegmentedColormap.from_list('i', [LIGHT_RED,  RED])

    for pc, cmap, base_color, label in [
        (pc_c, cmap_c, BLUE, 'Correct'),
        (pc_i, cmap_i, RED,  'Incorrect'),
    ]:
        for l in range(n_layers - 1):
            t = l / (n_layers - 1)
            ax.plot(pc[l:l+2, 0], pc[l:l+2, 1], pc[l:l+2, 2],
                    color=cmap(t), linewidth=2.2, alpha=0.9,
                    solid_capstyle='round')
        ax.scatter(*pc[0],  color=cmap(0.0), s=50, marker='o', zorder=7,
                   edgecolors='white', linewidths=0.5)
        ax.scatter(*pc[-1], color=cmap(1.0), s=80, marker='*', zorder=7,
                   edgecolors='white', linewidths=0.3)
        ax.plot([], [], [], color=base_color, linewidth=2.0, label=label)

    ax.set_xlabel(f'PC1 ({var_exp[0]:.0f}%)', fontsize=7, labelpad=3)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.0f}%)', fontsize=7, labelpad=3)
    ax.set_zlabel(f'PC3 ({var_exp[2]:.0f}%)', fontsize=7, labelpad=3)

    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='#cccccc', fontsize=7, framealpha=0.9,
              handlelength=1.2, handletextpad=0.4)
    ax.view_init(elev=22, azim=225)
    ax.tick_params(axis='both', labelsize=6, pad=1)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#dddddd')
    ax.yaxis.pane.set_edgecolor('#dddddd')
    ax.zaxis.pane.set_edgecolor('#dddddd')
    ax.grid(True, alpha=0.15)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Figure: Per-Metric AUROC Dot Plot (Cleveland dot plot) ───────────────────

def plot_metric_dotplot(out_path, results=None):
    """
    Horizontal dot plot showing per-metric AUROC across three models.
    Double-column width. Each row = one metric, dots connected by range bars.

    Args:
        out_path: output file path
        results: dict of {model_name: {metric_name: auroc}}, or None for defaults
    """
    apply_style()

    metrics = ['VANE (combined)', 'Velocity', 'Curvature', 'Geodesic Dev.',
               'Token Coherence', 'Jerk', 'Log-Probability', 'Static Rep.']

    if results is None:
        llama     = [0.769, 0.748, 0.748, 0.687, 0.722, 0.742, 0.627, 0.590]
        gemma     = [0.793, 0.765, 0.747, 0.747, 0.744, 0.746, 0.541, 0.559]
        ministral = [0.711, 0.696, 0.698, 0.652, 0.706, 0.691, 0.585, 0.601]
    else:
        llama     = [results['Llama-3-8B'][m] for m in metrics]
        gemma     = [results['Gemma-3-12B'][m] for m in metrics]
        ministral = [results['Ministral-8B'][m] for m in metrics]

    n = len(metrics)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.0))

    ax.axvline(x=0.5, color='#aaaaaa', linewidth=0.7, linestyle=':', zorder=0)
    ax.text(0.502, n - 0.3, 'chance', fontsize=6.5, color='#999999',
            va='top', style='italic')
    ax.axhline(y=5.5, color='#cccccc', linewidth=0.6, linestyle='--')

    for i in range(n):
        vals = [llama[i], gemma[i], ministral[i]]
        ax.plot([min(vals), max(vals)], [y[i], y[i]],
                color='#dddddd', linewidth=4, solid_capstyle='round', zorder=1)

    for vals, color, label, marker in [
        (llama,     BLUE,  'Llama-3-8B',    'o'),
        (gemma,     RED,   'Gemma-3-12B',   'D'),
        (ministral, GREEN, 'Ministral-8B',  's'),
    ]:
        ax.scatter(vals, y, c=color, s=45, marker=marker, zorder=3,
                   label=label, edgecolors='white', linewidths=0.5)

    for val, color, dy in [(0.769, BLUE, 0.30), (0.793, RED, -0.30),
                           (0.711, GREEN, 0.0)]:
        ax.annotate(f'{val:.3f}', xy=(val, 0), xytext=(val, -0.55 + dy * 0.3),
                    fontsize=6.5, ha='center', va='top', color=color,
                    fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=8)
    ax.set_xlabel('AUROC (5-fold CV)')
    ax.set_xlim(0.48, 0.83)
    ax.invert_yaxis()

    ax.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='#cccccc', fontsize=7, markerscale=0.9,
              handletextpad=0.3)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Figure: Orthogonality (scatter + correlation heatmap) ────────────────────

def plot_orthogonality(ckpt, out_path, model_name='Llama-3-8B'):
    """
    Pairwise Pearson correlation heatmap across all signals.
    Single-column width. Shows baselines are redundant while
    geometric metrics (especially Geodesic Dev.) are orthogonal.
    """
    apply_style()
    import matplotlib.patches as mpatches

    geo_keys = ['curv_max','curv_mean','curv_ans',
                'jerk_max','jerk_mean','jerk_ans',
                'vel_max','vel_mean','vel_ans',
                'geodev_max','geodev_mean','geodev_ans',
                'tokc_max','tokc_mean','tokc_ans']

    def _agg(arr):
        return float(np.nanmean(arr)) if isinstance(arr, np.ndarray) else float(arr)

    X_geo = np.array([[_agg(r[k]) for k in geo_keys] for r in ckpt])
    X_geo = np.nan_to_num(X_geo.astype(np.float64))
    logprobs = np.array([r['mean_log_prob'] for r in ckpt])
    static_reps = np.array([
        float(r['static_rep']) if not isinstance(r['static_rep'], np.ndarray)
        else float(np.mean(r['static_rep'])) for r in ckpt
    ])

    all_signals = {
        'Log-Prob':      logprobs,
        'Static Rep.':   static_reps,
        'Curvature':     X_geo[:, [0,1,2]].mean(axis=1),
        'Velocity':      X_geo[:, [6,7,8]].mean(axis=1),
        'Jerk':          X_geo[:, [3,4,5]].mean(axis=1),
        'Geodesic Dev.': X_geo[:, [9,10,11]].mean(axis=1),
        'Token Coh.':    X_geo[:, [12,13,14]].mean(axis=1),
    }

    names = list(all_signals.keys())
    n_sig = len(names)
    corr_matrix = np.zeros((n_sig, n_sig))
    for i in range(n_sig):
        for j in range(n_sig):
            corr_matrix[i, j], _ = stats.pearsonr(
                all_signals[names[i]], all_signals[names[j]])

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 3.0))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(n_sig))
    ax.set_yticks(range(n_sig))
    ax.set_xticklabels(names, fontsize=7.5, rotation=45, ha='right')
    ax.set_yticklabels(names, fontsize=7.5)

    for i in range(n_sig):
        for j in range(n_sig):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.55 else 'black'
            weight = 'bold' if (i < 2 and j >= 2) or (j < 2 and i >= 2) else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight=weight)

    rect = mpatches.Rectangle((-0.5, -0.5), 2, 2, linewidth=1.5,
                               edgecolor='#333333', facecolor='none')
    ax.add_patch(rect)
    ax.text(0.5, -0.75, 'redundant (r=0.86)', ha='center', va='top',
            fontsize=6, color='#333333', style='italic')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Pearson r', fontsize=8)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Figure: 3D Orthogonality Scatter ─────────────────────────────────────────

def plot_3d_orthogonality(ckpt, clf_path, out_path, model_name='Llama-3-8B'):
    """
    3D scatter of log-probability, static representation distance, and VANE score.
    Requires the trained classifier to compute VANE scores.
    Single-column width, styled for appendix.
    """
    apply_style()
    import matplotlib.patches as mpatches
    from vane.metrics import build_features_full

    with open(clf_path, 'rb') as f:
        meta = pickle.load(f)

    clf = meta['clf']
    sc  = meta['scaler']
    windows = meta.get('windows', ['max', 'mean'])
    ls = meta.get('layer_start_frac', 0.0)
    le = meta.get('layer_end_frac', 1.0)

    X = np.array([build_features_full(r, windows, ls, le) for r in ckpt])
    X = np.nan_to_num(X.astype(np.float64))
    vane_scores = clf.predict_proba(sc.transform(X))[:, 1]

    logprobs = np.array([r['mean_log_prob'] for r in ckpt])
    labels   = np.array([r['is_correct'] for r in ckpt])
    static_reps = np.array([
        float(r['static_rep']) if not isinstance(r['static_rep'], np.ndarray)
        else float(np.mean(r['static_rep'])) for r in ckpt
    ])

    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    lp_norm = norm01(logprobs)
    sr_norm = norm01(static_reps)

    r_lp_sr, _   = stats.pearsonr(logprobs, static_reps)
    r_lp_vane, _ = stats.pearsonr(logprobs, vane_scores)
    r_sr_vane, _ = stats.pearsonr(static_reps, vane_scores)

    correct   = labels == 1
    incorrect = labels == 0

    BLUE_3D = '#3a7ebf'
    RED_3D  = '#d44a4a'

    fig = plt.figure(figsize=(4.0, 4.2))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(sr_norm[correct], lp_norm[correct], vane_scores[correct],
               c=BLUE_3D, alpha=0.28, s=14, depthshade=True,
               rasterized=True, edgecolors='none', zorder=1)
    ax.scatter(sr_norm[incorrect], lp_norm[incorrect], vane_scores[incorrect],
               c=RED_3D, alpha=0.50, s=16, depthshade=True,
               rasterized=True, edgecolors='none', zorder=2)

    ax.set_xlabel('Static Rep (distance)', fontsize=8.5, labelpad=6)
    ax.set_ylabel('Log-Prob (normalized)', fontsize=8.5, labelpad=6)
    ax.set_zlabel('VANE Score', fontsize=8.5, labelpad=6, rotation=90)

    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_zticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.tick_params(axis='x', labelsize=7, pad=0)
    ax.tick_params(axis='y', labelsize=7, pad=0)
    ax.tick_params(axis='z', labelsize=7, pad=2)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#cccccc')
    ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(True, alpha=0.15, linewidth=0.4)

    ax.view_init(elev=22, azim=225)

    corr_text = ('r(LP, SR)   =  {:.2f}\n'
                 'r(LP, VANE) = {:.2f}\n'
                 'r(SR, VANE) = {:.2f}').format(r_lp_sr, r_lp_vane, r_sr_vane)

    ax.text2D(0.02, 0.02, corr_text, transform=ax.transAxes, fontsize=7.5,
              va='bottom', family='monospace',
              bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#999999',
                        alpha=0.95, linewidth=0.6))

    leg_correct   = mpatches.Patch(color=BLUE_3D, alpha=0.7, label='Correct')
    leg_incorrect = mpatches.Patch(color=RED_3D, alpha=0.7, label='Incorrect')
    ax.legend(handles=[leg_correct, leg_incorrect],
              loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2,
              frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.2)

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.08)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Figure: Selective Prediction ─────────────────────────────────────────────

def plot_selective_prediction(sel_data, out_path, model_name='Llama-3-8B'):
    """
    Selective prediction accuracy vs coverage curve. Single-column width.
    """
    apply_style()
    coverages_raw = sorted([float(k) for k in sel_data.keys()], reverse=True)
    vane_acc = [sel_data[str(c)]['geo_acc'] * 100 for c in coverages_raw]
    logp_acc = [sel_data[str(c)]['logp_acc'] * 100 for c in coverages_raw]
    coverages = [c * 100 for c in coverages_raw]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.5))

    ax.plot(coverages, vane_acc, color=BLUE, linewidth=1.8, marker='o',
            markersize=3, label='VANE (ours)')
    ax.plot(coverages, logp_acc, color=RED, linewidth=1.8, marker='s',
            markersize=3, linestyle='--', label='Log-Prob')
    ax.axhline(sel_data['1.0']['baseline'] * 100,
               color='grey', linewidth=0.8, linestyle=':', alpha=0.6,
               label='Baseline (answer all)')

    ax.set_xlabel('Coverage (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=7)
    ax.set_xlim(50, 100)
    ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Figure: Single-layer Probe Profile ───────────────────────────────────────

def plot_probe_profile(probe_data, out_path, vane_auroc=0.769,
                       model_name='Llama-3-8B'):
    """
    Per-layer probe AUROC with VANE reference line. Single-column width.
    """
    apply_style()
    layers = np.arange(1, len(probe_data['mean']) + 1)
    mean_auroc = np.array(probe_data['mean'])
    std_auroc  = np.array(probe_data['std'])

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.5))

    ax.fill_between(layers, mean_auroc - std_auroc, mean_auroc + std_auroc,
                    alpha=0.15, color=BLUE)
    ax.plot(layers, mean_auroc, color=BLUE, linewidth=1.4,
            label='Single-layer probe')
    ax.axhline(vane_auroc, color=RED, linewidth=1.2, linestyle='--',
               label=f'VANE ({vane_auroc:.3f})')

    best_layer = np.argmax(mean_auroc) + 1
    ax.scatter([best_layer], [mean_auroc[best_layer-1]], color=BLUE, s=30,
               zorder=5, edgecolors='white', linewidths=0.5)
    ax.annotate(f'L{best_layer}: {mean_auroc[best_layer-1]:.3f}',
                xy=(best_layer, mean_auroc[best_layer-1]),
                xytext=(best_layer + 2, mean_auroc[best_layer-1] + 0.01),
                fontsize=7, ha='left')

    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('AUROC')
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=7)
    ax.set_xlim(1, len(layers))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Figure: Metric vs Layer (correct vs incorrect) ───────────────────────────

def plot_metric_vs_layer(ckpt, out_path, model_name='Llama-3-8B'):
    """
    3-panel figure showing per-layer metric profiles (velocity, curvature, jerk)
    averaged over correct vs incorrect samples with shaded std bands.
    Double-column width.
    """
    apply_style()
    correct   = [r for r in ckpt if r['is_correct'] == 1]
    incorrect = [r for r in ckpt if r['is_correct'] == 0]

    def mean_profile(records, key):
        profiles = [r[key].astype(np.float32) for r in records]
        return np.nanmean(profiles, axis=0), np.nanstd(profiles, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 2.2), sharey=False)
    metrics = [('vel_mean', 'Velocity'), ('curv_mean', 'Curvature'), ('jerk_mean', 'Jerk')]

    for ax, (key, title) in zip(axes, metrics):
        mc, sc = mean_profile(correct, key)
        mi, si = mean_profile(incorrect, key)
        layers = np.arange(len(mc))
        ax.plot(layers, mc, color=BLUE, linewidth=1.5, label='Correct')
        ax.fill_between(layers, mc - sc, mc + sc, alpha=0.15, color=BLUE)
        ax.plot(layers, mi, color=RED, linewidth=1.5, label='Incorrect')
        ax.fill_between(layers, mi - si, mi + si, alpha=0.15, color=RED)
        ax.set_xlabel('Layer')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlim(0, len(mc) - 1)

    axes[0].set_ylabel('Mean metric value')
    axes[0].legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Figure: Distribution Histograms ──────────────────────────────────────────

def plot_metric_distributions(ckpt, out_path, model_name='Llama-3-8B'):
    """
    3-panel density histograms of per-sample mean velocity, curvature, jerk,
    split by correct/incorrect. Double-column width.
    """
    apply_style()
    correct   = [r for r in ckpt if r['is_correct'] == 1]
    incorrect = [r for r in ckpt if r['is_correct'] == 0]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 2.2))
    dist_metrics = [
        ('vel_mean', 'Velocity (mean over layers)'),
        ('curv_mean', 'Curvature (mean over layers)'),
        ('jerk_mean', 'Jerk (mean over layers)'),
    ]

    for ax, (key, title) in zip(axes, dist_metrics):
        vals_c = np.array([np.nanmean(r[key].astype(np.float32)) for r in correct])
        vals_i = np.array([np.nanmean(r[key].astype(np.float32)) for r in incorrect])
        bins = np.linspace(min(vals_c.min(), vals_i.min()),
                           max(np.percentile(vals_c, 99), np.percentile(vals_i, 99)), 40)
        ax.hist(vals_c, bins=bins, alpha=0.5, color=BLUE, label='Correct', density=True)
        ax.hist(vals_i, bins=bins, alpha=0.5, color=RED, label='Incorrect', density=True)
        ax.set_xlabel(title, fontsize=7)
        ax.set_yticks([])

    axes[0].set_ylabel('Density')
    axes[0].legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Figure: Multi-Trajectory PCA ─────────────────────────────────────────────

def plot_multi_trajectory_pca(hidden, labels, out_path, n_traj=30,
                               model_name='Llama-3-8B'):
    """
    3D PCA with multiple randomly sampled trajectories (faint lines)
    and bold mean trajectories per class. Double-column width.
    Removes cherry-picking concern of single-example PCA.
    """
    apply_style()
    N, L, D = hidden.shape
    all_vecs = hidden.reshape(-1, D).astype(np.float32)
    pca = PCA(n_components=3, random_state=42)
    pca.fit(all_vecs)

    np.random.seed(42)
    correct_idx = np.where(labels == 1)[0]
    incorrect_idx = np.where(labels == 0)[0]
    sel_c = np.random.choice(correct_idx, min(n_traj, len(correct_idx)), replace=False)
    sel_i = np.random.choice(incorrect_idx, min(n_traj, len(incorrect_idx)), replace=False)

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    for idx in sel_c:
        traj = pca.transform(hidden[idx].astype(np.float32))
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=BLUE, alpha=0.15, linewidth=0.6)
    for idx in sel_i:
        traj = pca.transform(hidden[idx].astype(np.float32))
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=RED, alpha=0.25, linewidth=0.6)

    mean_c = np.mean(hidden[sel_c].astype(np.float32), axis=0)
    mean_i = np.mean(hidden[sel_i].astype(np.float32), axis=0)
    mc_pca = pca.transform(mean_c)
    mi_pca = pca.transform(mean_i)

    ax.plot(mc_pca[:, 0], mc_pca[:, 1], mc_pca[:, 2], color=BLUE, linewidth=2.5,
            alpha=0.9, label=f'Correct (n={len(sel_c)})')
    ax.plot(mi_pca[:, 0], mi_pca[:, 1], mi_pca[:, 2], color=RED, linewidth=2.5,
            alpha=0.9, label=f'Incorrect (n={len(sel_i)})')
    ax.scatter(*mc_pca[0], color=BLUE, s=40, marker='o', zorder=5)
    ax.scatter(*mc_pca[-1], color=BLUE, s=60, marker='*', zorder=5)
    ax.scatter(*mi_pca[0], color=RED, s=40, marker='o', zorder=5)
    ax.scatter(*mi_pca[-1], color=RED, s=60, marker='*', zorder=5)

    ax.set_xlabel('PC1', labelpad=8)
    ax.set_ylabel('PC2', labelpad=8)
    fig.text(0.02, 0.55, 'PC3', fontsize=10, rotation=90, va='center',
             ha='center', fontfamily='serif')
    ax.view_init(elev=20, azim=225)
    fig.legend(loc='upper center', ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.98))
    fig.subplots_adjust(bottom=0.06, left=0.08, right=0.96, top=0.92)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Figure: No-Classifier ROC ────────────────────────────────────────────────

def plot_no_classifier_roc(ckpt, out_path, model_name='Llama-3-8B'):
    """
    ROC curves from raw metric thresholding (no trained classifier).
    Proves the geometric signal is intrinsic, not a classifier artifact.
    Single-column width.
    """
    apply_style()
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = np.array([r['is_correct'] for r in ckpt])
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.8))

    colors_roc = ['#2166ac', '#d6604d', '#4dac26', '#984ea3', '#ff7f00']
    metric_keys = [
        ('vel_mean', 'Velocity'), ('curv_mean', 'Curvature'),
        ('jerk_mean', 'Jerk'), ('geodev_mean', 'Geodesic Dev.'),
        ('tokc_mean', 'Token Coh.'),
    ]

    for (key, name), color in zip(metric_keys, colors_roc):
        scores = np.array([-np.nanmean(r[key].astype(np.float32)) for r in ckpt])
        auroc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, color=color, linewidth=1.2, label=f'{name} ({auroc:.3f})')

    logprobs = np.array([r['mean_log_prob'] for r in ckpt])
    auroc_lp = roc_auc_score(labels, logprobs)
    fpr_lp, tpr_lp, _ = roc_curve(labels, logprobs)
    ax.plot(fpr_lp, tpr_lp, color='grey', linewidth=1.0, linestyle='--',
            label=f'Log-Prob ({auroc_lp:.3f})')

    ax.plot([0, 1], [0, 1], 'k:', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=6.5,
              loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate VANE paper figures from saved data')
    parser.add_argument('--output_dir', default='emnlp')
    parser.add_argument('--hidden_states',
        default='new_results_v2/llama3-8b-instruct_gsm8k/hidden_states.npz')
    parser.add_argument('--checkpoint',
        default='new_results_v2/llama3-8b-instruct_gsm8k/checkpoint.pkl')
    parser.add_argument('--selective',
        default='new_results_v2/llama3-8b-instruct_gsm8k/selective_prediction.json')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('Loading checkpoint...')
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)

    n_correct   = sum(r['is_correct'] for r in ckpt)
    n_incorrect = len(ckpt) - n_correct
    print(f'Samples: {len(ckpt)}, Correct: {n_correct}, '
          f'Incorrect: {n_incorrect}')

    print('\n--- Figure: Metric Dot Plot ---')
    plot_metric_dotplot(out / 'fig_metric_comparison.pdf')

    if Path(args.hidden_states).exists():
        print('\n--- Figure: PCA Trajectory ---')
        data   = np.load(args.hidden_states)
        hidden = data['hidden']
        labels = data['labels']
        plot_pca_trajectory(hidden, labels, out / 'fig1_pca_trajectory.pdf')

    print('\n--- Figure: Orthogonality ---')
    plot_orthogonality(ckpt, out / 'fig4_orthogonality.pdf')

    if Path(args.selective).exists():
        print('\n--- Figure: Selective Prediction ---')
        with open(args.selective) as f:
            sel_data = json.load(f)
        plot_selective_prediction(sel_data,
                                 out / 'fig3_selective_prediction.pdf')

    print(f'\nAll figures saved to {out}')


if __name__ == '__main__':
    main()
