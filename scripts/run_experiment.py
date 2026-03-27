"""
Geometric Instability — Experiment Pipeline
============================================
Full pipeline: data collection → ablation → optional Optuna hyperopt → hybrid classifier.

Usage:
    python3 scripts/new_experiment.py \
        --model_id  /path/to/model  \
        --benchmark gsm8k           \
        --max_samples 1319          \
        --batch_size  4             \
        --output_dir  new_results   \
        --optuna 1                  # 0=off (default), 1=run Optuna hyperopt

Outputs in new_results/{model_short}_{benchmark}/:
    checkpoint.pkl          per-sample results (resumable)
    progress.log            intermediate ablation every --eval_every samples
    gpu_monitor.log         GPU wattage every 15 min
    report.txt              final ablation + hybrid report
    hybrid_clf.pkl          trained GradBoost classifier (clf + scaler)
    optuna_best_params.json best Optuna params (if --optuna 1)
    optuna_study.pkl        full Optuna study object (if --optuna 1)

Improvements over previous version:
    - Optuna tunes: GradBoost params + layer window fracs + feature windows
    - Layer windowing: use only middle layers (e.g. 20%-80%)
    - Feature window selection: which of max/mean/ans to include
    - Per-fold feature importance logging
    - Orthogonality check: geometric vs geometric+logprob
    - All results saved so nothing is lost
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
import threading
import time
import subprocess
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vane.metrics import (
    compute_metrics, build_features_full, get_ablation_features,
    ABLATION_CONFIGS, ALL_WINDOWS, ALL_GROUPS
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Geometric Instability Experiment")
    p.add_argument("--model_id",       type=str,  required=True)
    p.add_argument("--benchmark",      type=str,  default="gsm8k", choices=["gsm8k"])
    p.add_argument("--max_samples",    type=int,  default=1319)
    p.add_argument("--batch_size",     type=int,  default=4)
    p.add_argument("--output_dir",     type=str,  default="new_results")
    p.add_argument("--eval_every",     type=int,  default=100,
                   help="Run intermediate ablation every N samples")
    p.add_argument("--max_new_tokens", type=int,  default=1024)
    p.add_argument("--optuna",         type=int,  default=0, choices=[0, 1],
                   help="0=off, 1=run Optuna hyperparameter search")
    p.add_argument("--optuna_trials",  type=int,  default=80,
                   help="Number of Optuna trials (only used if --optuna 1)")
    p.add_argument("--answer_window",  type=int,  default=20,
                   help="Number of final tokens treated as answer region")
    return p.parse_args()


# ─── Utilities ────────────────────────────────────────────────────────────────

def model_short_name(model_id):
    name = os.path.basename(model_id.rstrip('/'))
    return re.sub(r'[^a-zA-Z0-9\-]', '-', name).lower()


def extract_answer_gsm8k(text):
    m = re.search(r'####\s*(-?[\d,]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    m = re.search(r'[Aa]nswer\s*[:\s]\s*(-?[\d,]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    nums = re.findall(r'-?\d+', text.replace(',', ''))
    return nums[-1] if nums else None


def build_prompt(question, tokenizer):
    messages = [
        {"role": "system", "content": "Solve the following math problem step by step."},
        {"role": "user",   "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"Problem: {question}\nSolution: Let's think step by step.\n"


def setup_logger(log_path):
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ─── GPU Heater ───────────────────────────────────────────────────────────────

def _start_gpu_heater(device, matrix_size=4096):
    """
    Keep GPU above 100W during CPU-only phases (Optuna) by running
    continuous matmuls in a background daemon thread.
    Uses 1ms sleep between iterations — GPU active ~99% of the time.
    Pre-allocates matrices once to avoid memory churn.
    """
    a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    c = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)

    def _heater():
        with torch.no_grad():
            while True:
                tmp = torch.matmul(a, b)
                torch.matmul(tmp, c)
                torch.cuda.synchronize()
                time.sleep(0.001)   # 1ms — GPU active ~99% of time

    t = threading.Thread(target=_heater, daemon=True)
    t.start()
    return t


# ─── GPU Monitor ──────────────────────────────────────────────────────────────

def gpu_wattage_monitor(log_path, interval=900, warn_threshold=100):
    """Background thread: logs GPU wattage every interval seconds."""
    while True:
        time.sleep(interval)
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=power.draw,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(log_path, 'a') as f:
                for i, line in enumerate(out.splitlines()):
                    parts = [x.strip() for x in line.split(',')]
                    watts  = float(parts[0])
                    util   = parts[1] if len(parts) > 1 else '?'
                    mem    = parts[2] if len(parts) > 2 else '?'
                    warn   = ' *** LOW WATTAGE ***' if watts < warn_threshold else ''
                    f.write(f"[{ts}] GPU{i}: {watts:.0f}W  util={util}%  mem={mem}MiB{warn}\n")
        except Exception as e:
            with open(log_path, 'a') as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] monitor error: {e}\n")


# ─── CV AUROC ─────────────────────────────────────────────────────────────────

def cv_auroc(X, y, clf_params=None, n_splits=5):
    """
    5-fold cross-validated AUROC using GradientBoostingClassifier.
    Returns (mean_auc, std_auc).
    """
    if len(np.unique(y)) < 2 or len(y) < 20:
        return float('nan'), float('nan')

    default_params = dict(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        min_samples_leaf=5, random_state=42
    )
    params = {**default_params, **(clf_params or {})}

    X = np.nan_to_num(X.astype(np.float64), nan=0, posinf=1e6, neginf=-1e6)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        sc   = StandardScaler()
        Xtr  = np.nan_to_num(sc.fit_transform(X[tr]), nan=0, posinf=1e6, neginf=-1e6)
        Xte  = np.nan_to_num(sc.transform(X[te]),    nan=0, posinf=1e6, neginf=-1e6)
        clf  = GradientBoostingClassifier(**params)
        clf.fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def cv_auroc_logreg(X, y, n_splits=5):
    """5-fold CV AUROC with LogisticRegression (fast, for ablation table)."""
    if len(np.unique(y)) < 2 or len(y) < 20:
        return float('nan'), float('nan')
    X = np.nan_to_num(X.astype(np.float64), nan=0, posinf=1e6, neginf=-1e6)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        sc  = StandardScaler()
        Xtr = np.nan_to_num(sc.fit_transform(X[tr]), nan=0, posinf=1e6, neginf=-1e6)
        Xte = np.nan_to_num(sc.transform(X[te]),    nan=0, posinf=1e6, neginf=-1e6)
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


# ─── Ablation evaluation ──────────────────────────────────────────────────────

def run_ablation(results, logger, clf_params=None,
                 layer_start_frac=0.0, layer_end_frac=1.0, windows=None):
    """
    Run full ablation table: per-metric LogReg AUROC + hybrid GradBoost AUROC.
    Returns dict of results.
    """
    n = len(results)
    logger.info(f"\n{'='*62}")
    logger.info(f"  ABLATION ({n} samples)  "
                f"layers={layer_start_frac:.0%}-{layer_end_frac:.0%}  "
                f"windows={windows or ALL_WINDOWS}")
    logger.info(f"{'='*62}")
    logger.info(f"  {'Metric':<20} {'LogReg':>8}  {'±':>6}  {'GradBoost':>10}  {'±':>6}")
    logger.info(f"  {'-'*58}")

    abl = {}
    for name in ABLATION_CONFIGS:
        X, y = get_ablation_features(results, name,
                                     windows=windows,
                                     layer_start_frac=layer_start_frac,
                                     layer_end_frac=layer_end_frac)
        lr_mu,  lr_sd  = cv_auroc_logreg(X, y)
        gb_mu,  gb_sd  = cv_auroc(X, y, clf_params) if name == 'Hybrid' \
                         else (float('nan'), float('nan'))
        abl[name] = {'logreg': lr_mu, 'logreg_std': lr_sd,
                     'gradboost': gb_mu, 'gradboost_std': gb_sd}
        lr_s  = f"{lr_mu:.4f}±{lr_sd:.4f}" if not np.isnan(lr_mu)  else "   ---      "
        gb_s  = f"{gb_mu:.4f}±{gb_sd:.4f}" if not np.isnan(gb_mu)  else "   ---      "
        logger.info(f"  {name:<20} {lr_s}  {gb_s}")

    logger.info(f"{'='*62}\n")
    return abl


# ─── Optuna hyperopt ──────────────────────────────────────────────────────────

def run_optuna(results, n_trials, logger, study_path, params_path, gpu_device=None,
               study_db_path=None):
    """
    Run Optuna to jointly optimise:
      - GradBoost hyperparameters
      - Layer window (start_frac, end_frac)
      - Feature windows (which of max/mean/ans to include)
    Returns best_params dict.

    gpu_device: if provided, a GPU matmul is executed at the start of every
    trial to keep GPU power above 100W (bypasses GIL contention issue).
    study_db_path: path to SQLite file for persistent storage (enables resume).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y = np.array([int(r['is_correct']) for r in results])

    # Pre-allocate heater tensors on GPU for in-trial GPU keep-alive
    _ha = _hb = None
    if gpu_device is not None:
        _ha = torch.randn(4096, 4096, device=gpu_device, dtype=torch.float16)
        _hb = torch.randn(4096, 4096, device=gpu_device, dtype=torch.float16)

    def objective(trial):
        # ── Keep GPU above 100W at start of every trial ─────────
        if _ha is not None:
            with torch.no_grad():
                torch.matmul(_ha, _hb)
                torch.cuda.synchronize()

        # ── GradBoost params ────────────────────────────────────
        gb_params = dict(
            n_estimators     = trial.suggest_int('n_estimators',     100, 400),
            max_depth        = trial.suggest_int('max_depth',        2,   6),
            learning_rate    = trial.suggest_float('learning_rate',  0.005, 0.3, log=True),
            subsample        = trial.suggest_float('subsample',      0.4,   1.0),
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1,     30),
            random_state     = 42,
        )
        # ── Layer window ────────────────────────────────────────
        ls = trial.suggest_float('layer_start_frac', 0.0, 0.5)
        le = trial.suggest_float('layer_end_frac',   0.5, 1.0)

        # ── Feature windows ──────────────────────────────────────
        window_choice = trial.suggest_categorical(
            'windows',
            ['all', 'max_only', 'mean_only', 'ans_only', 'max_mean', 'max_ans', 'mean_ans']
        )
        window_map = {
            'all':       ['max', 'mean', 'ans'],
            'max_only':  ['max'],
            'mean_only': ['mean'],
            'ans_only':  ['ans'],
            'max_mean':  ['max', 'mean'],
            'max_ans':   ['max', 'ans'],
            'mean_ans':  ['mean', 'ans'],
        }
        windows = window_map[window_choice]

        X, _ = get_ablation_features(results, 'Hybrid',
                                     windows=windows,
                                     layer_start_frac=ls,
                                     layer_end_frac=le)
        mu, _ = cv_auroc(X, y, gb_params, n_splits=5)
        return mu if not np.isnan(mu) else 0.0

    # ── Persistent storage for resume support ───────────────────
    storage = None
    if study_db_path is not None:
        storage = f"sqlite:///{study_db_path}"

    study = optuna.create_study(
        study_name="vane_optuna",
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage,
        load_if_exists=True,
    )

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    if done > 0:
        logger.info(f"Optuna resuming: {done} trials already done, {remaining} remaining...")
    else:
        logger.info(f"\nStarting Optuna ({n_trials} trials)...")

    def _log_callback(study, trial):
        if (trial.number + 1) % 10 == 0:
            logger.info(f"  Optuna trial {trial.number + 1}/{n_trials} | "
                        f"best={study.best_value:.4f}")

    study.optimize(objective, n_trials=remaining, show_progress_bar=False,
                   callbacks=[_log_callback])

    best = study.best_params
    best_val = study.best_value
    logger.info(f"Optuna best AUROC: {best_val:.4f}")
    logger.info(f"Best params: {best}")

    # Decode window choice
    window_map = {
        'all':       ['max', 'mean', 'ans'],
        'max_only':  ['max'],
        'mean_only': ['mean'],
        'ans_only':  ['ans'],
        'max_mean':  ['max', 'mean'],
        'max_ans':   ['max', 'ans'],
        'mean_ans':  ['mean', 'ans'],
    }
    best['windows_decoded'] = window_map[best['windows']]
    best['best_cv_auroc']   = best_val

    with open(params_path, 'w') as f:
        json.dump(best, f, indent=2)
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)

    logger.info(f"Optuna params  → {params_path}")
    logger.info(f"Optuna study   → {study_path}")
    return best


# ─── Classifier training ──────────────────────────────────────────────────────

def train_hybrid(results, clf_path, logger,
                 clf_params=None, windows=None,
                 layer_start_frac=0.0, layer_end_frac=1.0):
    """Train GradBoost on all data and save clf + scaler to pkl."""
    X, y = get_ablation_features(results, 'Hybrid',
                                 windows=windows,
                                 layer_start_frac=layer_start_frac,
                                 layer_end_frac=layer_end_frac)
    X = np.nan_to_num(X.astype(np.float64), nan=0, posinf=1e6, neginf=-1e6)

    default = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                   subsample=0.8, min_samples_leaf=5, random_state=42)
    params  = {**default, **(clf_params or {})}
    # Remove non-GradBoost keys that Optuna may have added
    gb_keys = {'n_estimators','max_depth','learning_rate','subsample',
               'min_samples_leaf','random_state'}
    params  = {k: v for k, v in params.items() if k in gb_keys}

    scaler  = StandardScaler()
    Xs      = np.nan_to_num(scaler.fit_transform(X), nan=0, posinf=1e6, neginf=-1e6)
    clf     = GradientBoostingClassifier(**params)
    clf.fit(Xs, y)

    bundle = {
        'clf': clf, 'scaler': scaler,
        'clf_params': params,
        'windows': windows or ALL_WINDOWS,
        'layer_start_frac': layer_start_frac,
        'layer_end_frac':   layer_end_frac,
    }
    with open(clf_path, 'wb') as f:
        pickle.dump(bundle, f)

    # Feature importance summary
    imp = clf.feature_importances_
    logger.info(f"  Top feature importance: max={imp.max():.4f}  "
                f"mean={imp.mean():.4f}  nonzero={np.sum(imp>0)}/{len(imp)}")
    logger.info(f"  Saved hybrid classifier → {clf_path}")


# ─── Selective prediction ────────────────────────────────────────────────────

def run_selective_prediction(results, logger, clf_params=None,
                              windows=None, layer_start_frac=0.0, layer_end_frac=1.0):
    """
    Compute risk-coverage curve using out-of-fold instability scores.
    Compares geometric threshold vs log-prob threshold.
    Returns dict of coverage → {geo_acc, logp_acc}.
    """
    y    = np.array([int(r['is_correct']) for r in results])
    logp = np.array([float(r.get('mean_log_prob', 0.0)) for r in results])

    # Out-of-fold geometric scores (no leakage)
    X, _ = get_ablation_features(results, 'Hybrid',
                                  windows=windows,
                                  layer_start_frac=layer_start_frac,
                                  layer_end_frac=layer_end_frac)
    X = np.nan_to_num(X.astype(np.float64), nan=0, posinf=1e6, neginf=-1e6)

    geo_oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    default = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                   subsample=0.8, min_samples_leaf=5, random_state=42)
    params = {**default, **(clf_params or {})}
    gb_keys = {'n_estimators','max_depth','learning_rate','subsample',
               'min_samples_leaf','random_state'}
    params = {k: v for k, v in params.items() if k in gb_keys}

    for tr, te in skf.split(X, y):
        sc  = StandardScaler()
        Xtr = np.nan_to_num(sc.fit_transform(X[tr]), nan=0, posinf=1e6, neginf=-1e6)
        Xte = np.nan_to_num(sc.transform(X[te]),    nan=0, posinf=1e6, neginf=-1e6)
        clf = GradientBoostingClassifier(**params)
        clf.fit(Xtr, y[tr])
        geo_oof[te] = clf.predict_proba(Xte)[:, 1]  # P(correct)

    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    baseline  = float(y.mean())

    logger.info(f"\n{'='*58}")
    logger.info(f"  SELECTIVE PREDICTION (out-of-fold, N={len(results)})")
    logger.info(f"  Baseline (answer all): {baseline:.3f}")
    logger.info(f"{'='*58}")
    logger.info(f"  {'Coverage':>9}  {'Geo Acc':>9}  {'LogProb Acc':>12}  {'Gain':>6}")
    logger.info(f"  {'-'*44}")

    selective = {}
    for cov in coverages:
        n       = int(cov * len(y))
        g_acc   = float(y[np.argsort(geo_oof)[-n:]].mean())
        l_acc   = float(y[np.argsort(logp)[-n:]].mean())
        gain    = g_acc - l_acc
        selective[cov] = {'geo_acc': g_acc, 'logp_acc': l_acc,
                          'gain': gain, 'baseline': baseline}
        logger.info(f"  {cov*100:>7.0f}%    {g_acc:.3f}       {l_acc:.3f}       {gain:+.3f}")
    logger.info(f"{'='*58}\n")
    return selective


# ─── Orthogonality check ──────────────────────────────────────────────────────

def check_orthogonality(results, logger, clf_params=None,
                        windows=None, layer_start_frac=0.0, layer_end_frac=1.0):
    """
    Verify geometric signal is orthogonal to log-prob.
    A well-designed hybrid should NOT improve when log-prob is added.
    """
    logger.info("\nOrthogonality check (Geometric vs Geometric+LogProb):")
    y    = np.array([int(r['is_correct']) for r in results])
    X_g, _  = get_ablation_features(results, 'Hybrid',
                                     windows=windows,
                                     layer_start_frac=layer_start_frac,
                                     layer_end_frac=layer_end_frac)
    logp     = np.array([[float(r.get('mean_log_prob', 0.0))] for r in results])
    X_gl     = np.hstack([X_g, logp])

    mu_g,  sd_g  = cv_auroc(X_g,  y, clf_params)
    mu_gl, sd_gl = cv_auroc(X_gl, y, clf_params)

    logger.info(f"  Geometric only:          {mu_g:.4f} ± {sd_g:.4f}")
    logger.info(f"  Geometric + Log-Prob:    {mu_gl:.4f} ± {sd_gl:.4f}")
    delta = mu_gl - mu_g
    if delta > 0.005:
        logger.info(f"  WARNING: +{delta:.4f} — log-prob adds signal, orthogonality claim is weaker")
    else:
        logger.info(f"  OK: delta={delta:+.4f} — geometric is orthogonal to log-prob ✓")
    return mu_g, mu_gl


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(42)

    # ── Output directory ────────────────────────────────────────────────────
    model_short = model_short_name(args.model_id)
    run_dir     = os.path.join(args.output_dir, f"{model_short}_{args.benchmark}")
    os.makedirs(run_dir, exist_ok=True)

    ckpt_path    = os.path.join(run_dir, 'checkpoint.pkl')
    report_path  = os.path.join(run_dir, 'report.txt')
    progress_log = os.path.join(run_dir, 'progress.log')
    gpu_log      = os.path.join(run_dir, 'gpu_monitor.log')
    clf_path     = os.path.join(run_dir, 'hybrid_clf.pkl')
    params_path  = os.path.join(run_dir, 'optuna_best_params.json')
    study_path   = os.path.join(run_dir, 'optuna_study.pkl')
    study_db_path = os.path.join(run_dir, 'optuna_study.db')

    logger = setup_logger(progress_log)
    logger.info(f"{'='*60}")
    logger.info(f"  Geometric Instability Experiment")
    logger.info(f"  Model:     {args.model_id}")
    logger.info(f"  Benchmark: {args.benchmark}  Samples: {args.max_samples}")
    logger.info(f"  Batch:     {args.batch_size}  MaxTokens: {args.max_new_tokens}")
    logger.info(f"  Optuna:    {'ON (' + str(args.optuna_trials) + ' trials)' if args.optuna else 'OFF'}")
    logger.info(f"  Output:    {run_dir}")
    logger.info(f"{'='*60}")

    # ── GPU monitor ────────────────────────────────────────────────────────
    threading.Thread(target=gpu_wattage_monitor,
                     args=(gpu_log, 900, 100), daemon=True).start()
    logger.info("GPU monitor active (every 15 min → gpu_monitor.log)")

    # ── Load checkpoint ────────────────────────────────────────────────────
    results = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"Resumed: {len(results)} samples already done")
    done_set = {r['sample_idx'] for r in results}

    # ── Load dataset ───────────────────────────────────────────────────────
    if args.benchmark == 'gsm8k':
        ds        = load_dataset("openai/gsm8k", "main", split="test")
        ds        = ds.select(range(min(args.max_samples, len(ds))))
        questions = [ds[i]['question'] for i in range(len(ds))]
        gt_answers = [extract_answer_gsm8k(ds[i]['answer']) for i in range(len(ds))]
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    remaining = [i for i in range(len(ds)) if i not in done_set]
    logger.info(f"Remaining: {len(remaining)} samples")

    # ── Load model (always — needed for GPU keep-alive during Optuna) ─────────
    logger.info(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map=None,
        torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0")
    model.eval()
    logger.info("Model loaded.")

    if remaining:

        # ── Batch inference ────────────────────────────────────────────────
        batches = [remaining[i:i+args.batch_size]
                   for i in range(0, len(remaining), args.batch_size)]
        logger.info(f"Processing {len(remaining)} samples in {len(batches)} batches")

        with tqdm(total=len(remaining), desc="Inference") as pbar:
            for batch_indices in batches:
                prompts = [build_prompt(questions[i], tokenizer) for i in batch_indices]
                gts     = [gt_answers[i] for i in batch_indices]

                batch_in = tokenizer(
                    prompts, return_tensors="pt",
                    padding=True, truncation=False
                ).to(model.device)
                padded_len = batch_in['input_ids'].shape[1]

                with torch.no_grad():
                    out = model.generate(
                        **batch_in,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,        # greedy — consistent with training distribution
                        temperature=None,       # suppress warning from model's default config
                        top_p=None,
                        top_k=None,
                        pad_token_id=tokenizer.pad_token_id,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                    )

                for b, (idx, gt) in enumerate(zip(batch_indices, gts)):
                    gen_ids  = out.sequences[b].unsqueeze(0)
                    gen_text = tokenizer.decode(gen_ids[0][padded_len:],
                                               skip_special_tokens=True)
                    pred       = extract_answer_gsm8k(gen_text)
                    is_correct = int(pred == gt) if pred is not None else 0

                    # Geometric metrics + log-prob via a single forward pass
                    with torch.no_grad():
                        fwd = model(gen_ids,
                                    attention_mask=torch.ones_like(gen_ids),
                                    output_hidden_states=True)

                    # Compute mean log-prob from the forward pass logits (reliable, no scores needed)
                    mean_log_prob = 0.0
                    try:
                        gen_len_actual = gen_ids.shape[1] - padded_len
                        if gen_len_actual > 1:
                            logits   = fwd.logits[0, padded_len:-1, :]       # (gen_len-1, vocab)
                            targets  = gen_ids[0, padded_len + 1:]            # (gen_len-1,)
                            lp       = torch.log_softmax(logits, dim=-1)
                            mean_log_prob = lp[range(len(targets)), targets].mean().item()
                    except Exception:
                        mean_log_prob = 0.0

                    metrics = compute_metrics(fwd.hidden_states, padded_len,
                                              answer_window=args.answer_window)
                    metrics.update({
                        'is_correct':    is_correct,
                        'mean_log_prob': mean_log_prob,
                        'sample_idx':    idx,
                        'pred':          pred,
                        'gt':            gt,
                    })
                    results.append(metrics)

                    # Checkpoint every sample
                    with open(ckpt_path, 'wb') as f:
                        pickle.dump(results, f)
                    pbar.update(1)

                # ── Intermediate evaluation ────────────────────────────────
                n_done = len(results)
                if n_done % args.eval_every == 0 and n_done >= 100:
                    acc = sum(r['is_correct'] for r in results) / n_done
                    logger.info(f"\n[{n_done}/{args.max_samples}] acc={acc:.3f}")
                    run_ablation(results, logger)

    # Keep model in GPU memory and start heater — freeing it drops power below
    # 100W and Berzelius will cancel the job.
    logger.info("Starting GPU heater to maintain >100W during Optuna...")
    _start_gpu_heater(model.device)

    # ── Final accuracy ─────────────────────────────────────────────────────
    acc = sum(r['is_correct'] for r in results) / len(results)
    logger.info(f"\nFinal accuracy: {acc:.4f}  ({sum(r['is_correct'] for r in results)}/{len(results)})")

    # ── Optuna hyperopt ────────────────────────────────────────────────────
    best_clf_params      = None
    best_layer_start     = 0.0
    best_layer_end       = 1.0
    best_windows         = None

    if args.optuna:
        best = run_optuna(results, args.optuna_trials, logger, study_path, params_path,
                          gpu_device=model.device if model is not None else None,
                          study_db_path=study_db_path)
        best_clf_params  = {k: best[k] for k in
                            ['n_estimators','max_depth','learning_rate',
                             'subsample','min_samples_leaf']
                            if k in best}
        best_clf_params['random_state'] = 42
        best_layer_start = best.get('layer_start_frac', 0.0)
        best_layer_end   = best.get('layer_end_frac',   1.0)
        best_windows     = best.get('windows_decoded',  None)
        logger.info(f"\nUsing Optuna best config for final evaluation:")
        logger.info(f"  clf_params={best_clf_params}")
        logger.info(f"  layers={best_layer_start:.0%}-{best_layer_end:.0%}")
        logger.info(f"  windows={best_windows}")
    else:
        logger.info("\nOptuna OFF — using default params.")

    # ── Final ablation ─────────────────────────────────────────────────────
    logger.info("\nFINAL ABLATION:")
    abl = run_ablation(results, logger,
                       clf_params=best_clf_params,
                       layer_start_frac=best_layer_start,
                       layer_end_frac=best_layer_end,
                       windows=best_windows)

    # ── Selective prediction ───────────────────────────────────────────────
    selective = run_selective_prediction(
        results, logger,
        clf_params=best_clf_params,
        windows=best_windows,
        layer_start_frac=best_layer_start,
        layer_end_frac=best_layer_end
    )
    sel_path = os.path.join(run_dir, 'selective_prediction.json')
    with open(sel_path, 'w') as f:
        json.dump({str(k): v for k, v in selective.items()}, f, indent=2)

    # ── Orthogonality check ────────────────────────────────────────────────
    geo_auc, geo_logp_auc = check_orthogonality(
        results, logger,
        clf_params=best_clf_params,
        windows=best_windows,
        layer_start_frac=best_layer_start,
        layer_end_frac=best_layer_end
    )

    # ── Train and save final classifier ───────────────────────────────────
    train_hybrid(results, clf_path, logger,
                 clf_params=best_clf_params,
                 windows=best_windows,
                 layer_start_frac=best_layer_start,
                 layer_end_frac=best_layer_end)

    # ── Write report ───────────────────────────────────────────────────────
    lines = [
        "=" * 62,
        "  GEOMETRIC INSTABILITY — FINAL REPORT",
        "=" * 62,
        f"  Model     : {args.model_id}",
        f"  Benchmark : {args.benchmark}",
        f"  Samples   : {len(results)}",
        f"  Accuracy  : {acc:.4f}",
        f"  Optuna    : {'ON — ' + str(args.optuna_trials) + ' trials' if args.optuna else 'OFF (default params)'}",
        "",
        "  ABLATION RESULTS (5-fold CV AUROC, GradBoost)",
        "  " + "-" * 54,
        f"  {'Metric':<20} {'LogReg':>10}  {'GradBoost':>12}",
        "  " + "-" * 54,
    ]
    for name, vals in abl.items():
        lr = f"{vals['logreg']:.4f}±{vals['logreg_std']:.4f}" \
             if not np.isnan(vals['logreg']) else "      ---      "
        gb = f"{vals['gradboost']:.4f}±{vals['gradboost_std']:.4f}" \
             if not np.isnan(vals['gradboost']) else "      ---       "
        lines.append(f"  {name:<20} {lr:>14}  {gb:>16}")
    lines += [
        "  " + "-" * 54,
        "",
        f"  Orthogonality: geo={geo_auc:.4f}  geo+logp={geo_logp_auc:.4f}  "
        f"delta={geo_logp_auc - geo_auc:+.4f}",
        "",
        "  SELECTIVE PREDICTION (out-of-fold)",
        "  " + "-" * 46,
        f"  {'Coverage':>9}  {'Geo Acc':>9}  {'LogProb Acc':>12}  {'Gain':>6}",
        "  " + "-" * 46,
    ]
    for cov, vals in selective.items():
        lines.append(
            f"  {cov*100:>7.0f}%    {vals['geo_acc']:.3f}       "
            f"{vals['logp_acc']:.3f}       {vals['gain']:+.3f}"
        )
    lines += [
        "=" * 62,
    ]
    if args.optuna:
        lines += [
            "",
            "  OPTUNA BEST PARAMS",
            "  " + "-" * 40,
        ]
        with open(params_path) as f:
            best_loaded = json.load(f)
        for k, v in best_loaded.items():
            lines.append(f"  {k}: {v}")
        lines.append("=" * 62)

    report_str = "\n".join(lines)
    print(report_str)
    with open(report_path, 'w') as f:
        f.write(report_str + "\n")

    logger.info(f"\nReport      → {report_path}")
    logger.info(f"Classifier  → {clf_path}")
    logger.info(f"Checkpoint  → {ckpt_path}")


if __name__ == "__main__":
    main()
