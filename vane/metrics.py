"""
Geometric Instability Metrics — Clean Implementation
=====================================================
Five geometric trajectory metrics for LLM reasoning failure detection.

Core metrics (VANE — Velocity, Acceleration, and Nonlinearity Estimation):
  1. Curvature     — Frenet-Serret curvature: rate of change of unit tangent vector
                     (arc-length normalised, not confounded with velocity magnitude)
  2. Jerk          — acceleration magnitude (2nd derivative of hidden state)
  3. Velocity      — step-size magnitude (1st derivative of hidden state)
  4. Geodesic Dev  — deviation of hidden-state path from straight chord between
                     first and last layer (global path efficiency, scale-invariant)
  5. Token Coh     — per-layer directional incoherence across generated tokens:
                     how much do token velocity directions disagree at each layer?
                     (orthogonal axis — token dimension rather than layer dimension)

Note: FM Divergence is included in the Hybrid classifier alongside the other 5 metrics.

Each metric is aggregated at three token-window granularities:
  - max:  max over all generated tokens (worst-case instability)
  - mean: mean over all generated tokens (average instability)
  - ans:  max over final `answer_window` tokens (answer-region signal)

Also computes:
  - static_rep: mean cosine similarity between mid-layer and final-layer
                hidden states (static representation probe, used in Hybrid)

Feature extraction supports:
  - Layer windowing: use only a fraction of layers (e.g. middle 37%-98%)
  - Window selection: choose which token-windowing variants to include
  - Both controlled via build_features() parameters or Optuna search
"""

import torch
import torch.nn.functional as F
import numpy as np


# ─── Metric computation ───────────────────────────────────────────────────────

def compute_metrics(hidden_states, prompt_len, answer_window=20):
    """
    Compute geometric instability metrics from transformer hidden states.

    Args:
        hidden_states: tuple of tensors (L+1 layers), each shape (1, seq, dim).
                       Index 0 is the embedding layer and is skipped.
        prompt_len:    Number of prompt tokens to exclude from analysis.
        answer_window: Number of final tokens treated as the answer region.

    Returns:
        dict with per-layer numpy arrays and scalar values.
        Array keys: {metric}_{window} where metric in
        {curv, fm, jerk, vel, geodev, tokc} and window in {max, mean, ans}.
    """
    H = torch.stack(hidden_states[1:], dim=0)  # (L, 1, seq, dim) — skip embedding
    L, B, S, D = H.shape

    if S <= prompt_len or L < 4:
        return _empty_result()

    H_gen = H[:, 0, prompt_len:, :]  # (L, gen_seq, dim)
    gen_len = H_gen.shape[1]

    if gen_len < 3:
        return _empty_result()

    # Layer-wise velocity vectors
    Delta = H_gen[1:] - H_gen[:-1]  # (L-1, gen_seq, dim)

    # Unit tangent vectors (used by Curvature and Token Coherence)
    T = Delta / (torch.linalg.norm(Delta, dim=-1, keepdim=True) + 1e-8)  # (L-1, gen_seq, dim)

    # ── 1. Curvature (Frenet-Serret) ──────────────────────────────────────────
    # Rate of change of unit tangent — arc-length normalised, not confounded
    # with velocity magnitude unlike the raw cosine-angle formulation.
    dT       = T[1:] - T[:-1]                                # (L-2, gen_seq, dim)
    curvature = torch.linalg.norm(dT, dim=-1)                # (L-2, gen_seq)

    # ── 2. FM Divergence (kept for ablation, excluded from Hybrid) ────────────
    target = H_gen[-1:] - H_gen[:-1]                         # (L-1, gen_seq, dim)
    cos_fm = F.cosine_similarity(Delta, target + 1e-9, dim=-1)
    fm_div = 1.0 - cos_fm                                    # (L-1, gen_seq)

    # ── 3. Jerk (Acceleration magnitude) ──────────────────────────────────────
    accel = Delta[1:] - Delta[:-1]                           # (L-2, gen_seq, dim)
    jerk  = torch.linalg.norm(accel, dim=-1)                 # (L-2, gen_seq)

    # ── 4. Velocity (Step-size magnitude) ─────────────────────────────────────
    velocity = torch.linalg.norm(Delta, dim=-1)              # (L-1, gen_seq)

    # ── 5. Geodesic Deviation ─────────────────────────────────────────────────
    # Distance from each layer's hidden state to the straight-line chord
    # between the first and last layer, normalised by chord length.
    first  = H_gen[0]                                        # (gen_seq, dim)
    last   = H_gen[-1]                                       # (gen_seq, dim)
    chord  = last - first                                    # (gen_seq, dim)
    chord_len = torch.linalg.norm(chord, dim=-1) + 1e-8     # (gen_seq,)

    t_vals    = torch.arange(L, device=H_gen.device).float() / max(L - 1, 1)
    t_vals    = t_vals.view(L, 1, 1)                         # (L, 1, 1)
    geodesic  = first.unsqueeze(0) + t_vals * chord.unsqueeze(0)  # (L, gen_seq, dim)
    raw_dev   = torch.linalg.norm(H_gen - geodesic, dim=-1) # (L, gen_seq)
    geodev    = raw_dev / chord_len.unsqueeze(0)             # (L, gen_seq) normalised

    # ── 6. Token Coherence (directional incoherence across tokens) ────────────
    # At each layer: how much do individual token velocity directions deviate
    # from the mean direction across all tokens?
    # High value = tokens moving inconsistently = reasoning instability.
    mean_T  = T.mean(dim=1, keepdim=True)                    # (L-1, 1, dim)
    mean_T  = mean_T / (torch.linalg.norm(mean_T, dim=-1, keepdim=True) + 1e-8)
    tok_coh = 1.0 - F.cosine_similarity(T, mean_T.expand_as(T), dim=-1)  # (L-1, gen_seq)

    # ── Static representation probe ───────────────────────────────────────────
    mid        = L // 2
    sim        = F.cosine_similarity(H_gen[mid], H_gen[-1], dim=-1)  # (gen_seq,)
    static_rep = float(sim.mean().item())

    # ── Windowing helper ──────────────────────────────────────────────────────
    def _stats(t2d):
        """t2d: (layers, gen_seq) → (max_profile, mean_profile, ans_profile)"""
        t2d = t2d.float()
        mx  = t2d.max(dim=-1).values.cpu().numpy()
        mn  = t2d.mean(dim=-1).cpu().numpy()
        ans = t2d[:, -answer_window:].max(dim=-1).values.cpu().numpy() \
              if gen_len > answer_window else mx
        return mx, mn, ans

    curv_max,   curv_mean,   curv_ans   = _stats(curvature)
    fm_max,     fm_mean,     fm_ans     = _stats(fm_div)
    jerk_max,   jerk_mean,   jerk_ans   = _stats(jerk)
    vel_max,    vel_mean,    vel_ans    = _stats(velocity)
    geodev_max, geodev_mean, geodev_ans = _stats(geodev)
    tokc_max,   tokc_mean,   tokc_ans   = _stats(tok_coh)

    return {
        'curv_max':   curv_max,   'curv_mean':   curv_mean,   'curv_ans':   curv_ans,
        'fm_max':     fm_max,     'fm_mean':      fm_mean,     'fm_ans':     fm_ans,
        'jerk_max':   jerk_max,   'jerk_mean':   jerk_mean,   'jerk_ans':   jerk_ans,
        'vel_max':    vel_max,    'vel_mean':    vel_mean,    'vel_ans':    vel_ans,
        'geodev_max': geodev_max, 'geodev_mean': geodev_mean, 'geodev_ans': geodev_ans,
        'tokc_max':   tokc_max,   'tokc_mean':   tokc_mean,   'tokc_ans':   tokc_ans,
        'static_rep': static_rep,
        'gen_length': gen_len,
        'num_layers': L,
    }


# ─── Feature extraction ───────────────────────────────────────────────────────

METRIC_GROUPS = {
    'curvature':       ['curv_max',   'curv_mean',   'curv_ans'],
    'fm_divergence':   ['fm_max',     'fm_mean',     'fm_ans'],
    'jerk':            ['jerk_max',   'jerk_mean',   'jerk_ans'],
    'velocity':        ['vel_max',    'vel_mean',    'vel_ans'],
    'geodesic_dev':    ['geodev_max', 'geodev_mean', 'geodev_ans'],
    'token_coherence': ['tokc_max',   'tokc_mean',   'tokc_ans'],
}

ALL_WINDOWS = ['max', 'mean', 'ans']

# VANE uses 5 core metrics (FM Divergence excluded — architecture-dependent signal, reported in ablation only)
ALL_GROUPS  = ['curvature', 'jerk', 'velocity', 'geodesic_dev', 'token_coherence']

# Maps group name → short key prefix used in result dict
_PREFIX_MAP = {
    'curvature':       'curv',
    'fm_divergence':   'fm',
    'jerk':            'jerk',
    'velocity':        'vel',
    'geodesic_dev':    'geodev',
    'token_coherence': 'tokc',
}


def build_features(result,
                   groups=None,
                   windows=None,
                   layer_start_frac=0.0,
                   layer_end_frac=1.0):
    """
    Flatten per-layer metric profiles into a 1D feature vector.

    Args:
        result:           dict from compute_metrics()
        groups:           metric groups to include (default: ALL_GROUPS)
        windows:          windowing variants — subset of ['max','mean','ans']
                          (default: all three)
        layer_start_frac: start of layer window as fraction of total layers [0, 1)
        layer_end_frac:   end of layer window as fraction of total layers (0, 1]

    Returns:
        np.ndarray of shape (n_features,)
    """
    if groups  is None: groups  = ALL_GROUPS
    if windows is None: windows = ALL_WINDOWS

    feats = []
    for g in groups:
        prefix = _PREFIX_MAP[g]
        for w in windows:
            arr = np.asarray(result.get(f"{prefix}_{w}", [0.0]),
                             dtype=np.float64).ravel()
            n = len(arr)
            if n > 1:
                i0 = int(np.floor(layer_start_frac * n))
                i1 = int(np.ceil(layer_end_frac   * n))
                i1 = max(i1, i0 + 1)
                arr = arr[i0:i1]
            feats.append(arr)

    return np.concatenate(feats)


def build_features_full(result,
                        windows=None,
                        layer_start_frac=0.0,
                        layer_end_frac=1.0):
    """All VANE geometric features + static_rep (for Hybrid classifier)."""
    geo    = build_features(result,
                            groups=ALL_GROUPS,
                            windows=windows,
                            layer_start_frac=layer_start_frac,
                            layer_end_frac=layer_end_frac).ravel()
    static = np.array([result.get('static_rep', 0.0)], dtype=np.float64)
    return np.concatenate([geo, static])


# ─── Ablation configuration ───────────────────────────────────────────────────

ABLATION_CONFIGS = {
    # Baselines
    'Log-Prob':          'logprob',
    'Static Rep':        'static',
    # Individual metrics
    'Curvature':         ['curvature'],
    'FM Divergence':     ['fm_divergence'],
    'Jerk':              ['jerk'],
    'Velocity':          ['velocity'],
    'Geodesic Dev':      ['geodesic_dev'],
    'Token Coherence':   ['token_coherence'],
    # Combined
    'All Geometric':     ALL_GROUPS,
    'Hybrid':            'full',
}


def get_ablation_features(results, config_name,
                          windows=None,
                          layer_start_frac=0.0,
                          layer_end_frac=1.0):
    """
    Extract feature matrix X and label vector y for an ablation config.

    Returns:
        X: np.ndarray (n_samples, n_features)
        y: np.ndarray (n_samples,) binary int
    """
    y   = np.array([int(r['is_correct']) for r in results])
    cfg = ABLATION_CONFIGS[config_name]

    if cfg == 'logprob':
        X = np.array([[float(r.get('mean_log_prob', 0.0))] for r in results])
    elif cfg == 'static':
        X = np.array([[_scalar_static(r)] for r in results])
    elif cfg == 'full':
        X = np.array([build_features_full(r, windows, layer_start_frac, layer_end_frac)
                      for r in results])
    else:
        X = np.array([build_features(r, cfg, windows, layer_start_frac, layer_end_frac)
                      for r in results])

    return np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=1e6, neginf=-1e6), y


def _scalar_static(r):
    v = r.get('static_rep', 0.0)
    return float(np.mean(v)) if hasattr(v, '__len__') else float(v)


# ─── Empty result fallback ────────────────────────────────────────────────────

def _empty_result():
    z = np.array([0.0])
    return {
        'curv_max':   z, 'curv_mean':   z, 'curv_ans':   z,
        'fm_max':     z, 'fm_mean':     z, 'fm_ans':     z,
        'jerk_max':   z, 'jerk_mean':   z, 'jerk_ans':   z,
        'vel_max':    z, 'vel_mean':    z, 'vel_ans':    z,
        'geodev_max': z, 'geodev_mean': z, 'geodev_ans': z,
        'tokc_max':   z, 'tokc_mean':   z, 'tokc_ans':   z,
        'static_rep': 0.0, 'gen_length': 0, 'num_layers': 0,
    }
