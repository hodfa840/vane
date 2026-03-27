"""
VANE — MATH-500 Benchmark
==========================
Runs the full VANE pipeline on the MATH-500 benchmark
(Lightman et al., 2023 — 500 competition math problems).

Why MATH-500?
  - Harder than GSM8K → models fail more → richer failure signal for VANE
  - Well-known benchmark → reviewers recognise it immediately
  - Free-form generation → long CoT → rich hidden-state trajectory
  - Complements GSM8K: shows VANE generalises across math difficulty levels

Dataset: HuggingFaceH4/MATH-500 (500 problems, all difficulty levels)
Models:  same three as GSM8K (Llama-3-8B, Gemma-3-12B, Ministral-8B)
Output:  new_results_v2/{model_short}_math500/  — same checkpoint format

Outputs per run:
    checkpoint.pkl          per-sample results (resumable)
    hidden_states_ckpt.npz  intermediate hidden-state buffer (resumable)
    hidden_states.npz       final (N, n_layers, hidden_dim) float16
    hidden_states_meta.json model/shape metadata
    optuna_best_params.json best hyperparameters
    optuna_study.pkl        full Optuna study object
    optuna_study.db         SQLite DB for Optuna resume
    progress.log            inference + intermediate ablations
    gpu_monitor.log         GPU wattage every 15 min

Usage:
    python3 scripts/run_math500.py \\
        --model_id  models/llama3-8b-instruct \\
        --output_dir new_results_v2 \\
        --batch_size 8 \\
        --optuna 1

Eval-only (checkpoint already exists):
    python3 scripts/run_math500.py \\
        --model_id  models/llama3-8b-instruct \\
        --output_dir new_results_v2 \\
        --eval_only
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
from collections import defaultdict
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
    compute_metrics, get_ablation_features,
    ABLATION_CONFIGS, ALL_WINDOWS,
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VANE — MATH-500 Experiment")
    p.add_argument("--model_id",       type=str, required=True)
    p.add_argument("--output_dir",     type=str, default="new_results_v2")
    p.add_argument("--batch_size",     type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--answer_window",  type=int, default=20)
    p.add_argument("--optuna",         type=int, default=0, choices=[0, 1])
    p.add_argument("--optuna_trials",  type=int, default=200)
    p.add_argument("--eval_every",     type=int, default=100)
    p.add_argument("--eval_only",      action="store_true",
                   help="Skip inference; run ablation on existing checkpoint only")
    return p.parse_args()


# ─── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(question: str, tokenizer) -> str:
    """
    CoT prompt for competition math.  Instructs the model to reason step by
    step and place the final answer in \\boxed{} — the format expected by the
    answer extractor.  The system prompt primes the model for rigorous math.
    """
    content = (
        "Solve the following competition math problem.\n"
        "Think through the problem carefully, step by step.\n"
        "Show all your working clearly.\n"
        "At the very end, write your final answer inside \\boxed{} like this:\n"
        "\\boxed{<your answer>}\n\n"
        f"Problem:\n{question}"
    )
    messages = [
        {"role": "system",
         "content": (
             "Reason carefully and show every step. "
             "Always end with \\boxed{<answer>}."
         )},
        {"role": "user", "content": content},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return (
            f"Problem:\n{question}\n\n"
            "Solve step by step. Write the final answer as \\boxed{<answer>}.\n"
        )


# ─── Answer extraction ────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """
    Light normalisation for MATH answers (no sympy dependency).
    Strips LaTeX formatting, whitespace, and punctuation so string
    comparison catches common equivalent forms.
    """
    s = s.strip()
    s = re.sub(r'^\$+|\$+$', '', s).strip()   # remove surrounding $
    s = re.sub(r'\\left|\\right', '', s)        # remove \left \right
    s = re.sub(r'\s+', '', s)                   # collapse whitespace
    s = s.rstrip('.')                            # trailing dot
    s = re.sub(r'\{(\w)\}', r'\1', s)           # {x} → x for single chars
    return s.lower()


def extract_boxed(text: str):
    """
    Extract the content of the last \\boxed{...} in the generated text.
    Handles nested braces correctly.
    """
    starts = [m.start() for m in re.finditer(r'\\boxed\s*\{', text)]
    if not starts:
        return None
    pos = starts[-1]
    pos = text.index('{', pos)
    depth, content = 0, []
    for ch in text[pos:]:
        if ch == '{':
            depth += 1
            if depth > 1:
                content.append(ch)
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
            content.append(ch)
        else:
            content.append(ch)
    return ''.join(content).strip() if depth == 0 else None


def extract_answer_math(text: str, gt: str):
    """
    Returns (pred_str, is_correct).
    Strategy:
      1. Last \\boxed{...} in model output → normalise → exact-match with gt
      2. Fallback: last number in output
    """
    gt_norm = _normalize(gt)

    boxed = extract_boxed(text)
    if boxed is not None:
        pred_norm = _normalize(boxed)
        return pred_norm, int(pred_norm == gt_norm)

    nums = re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
    if nums:
        pred_norm = _normalize(nums[-1])
        return pred_norm, int(pred_norm == gt_norm)

    return None, 0


# ─── Utilities ────────────────────────────────────────────────────────────────

def model_short_name(model_id: str) -> str:
    name = os.path.basename(model_id.rstrip('/'))
    return re.sub(r'[^a-zA-Z0-9\-]', '-', name).lower()


def setup_logger(log_path: str):
    logger = logging.getLogger('math500')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')
    fh  = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def gpu_wattage_monitor(log_path, interval=900, warn_threshold=100):
    while True:
        time.sleep(interval)
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=power.draw,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL).decode().strip()
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(log_path, 'a') as f:
                for i, line in enumerate(out.splitlines()):
                    parts = [x.strip() for x in line.split(',')]
                    watts = float(parts[0])
                    warn  = ' *** LOW ***' if watts < warn_threshold else ''
                    f.write(f"[{ts}] GPU{i}: {watts:.0f}W{warn}\n")
        except Exception:
            pass


def get_model_dims(model):
    cfg = model.config
    if hasattr(cfg, 'text_config'):
        cfg = cfg.text_config
    return cfg.num_hidden_layers, cfg.hidden_size


# ─── Hidden-state extraction ──────────────────────────────────────────────────

def extract_mean_hidden(fwd_hidden_states, padded_len: int, n_layers: int):
    """
    Mean-pool hidden states over response tokens for each layer.
    Returns float16 array of shape (n_layers, hidden_dim).
    Skips layer 0 (embedding), uses layers 1..n_layers.
    """
    layer_vecs = []
    for layer_idx in range(1, n_layers + 1):
        hs = fwd_hidden_states[layer_idx]   # (1, seq, D)
        resp_hs = hs[0, padded_len:, :]     # (T_resp, D)
        if resp_hs.shape[0] == 0:
            resp_hs = hs[0, -1:, :]
        mean_vec = resp_hs.float().mean(dim=0).cpu().numpy().astype(np.float16)
        layer_vecs.append(mean_vec)
    return np.stack(layer_vecs, axis=0)     # (n_layers, hidden_dim)


def save_hidden_states_ckpt(path, hidden_buf, labels_buf, logprob_buf,
                             idx_buf, n_done):
    np.savez_compressed(
        path,
        hidden   = hidden_buf,
        labels   = labels_buf,
        logprobs = logprob_buf,
        indices  = idx_buf,
        n_done   = np.array(n_done, dtype=np.int32),
    )


# ─── CV AUROC ─────────────────────────────────────────────────────────────────

def cv_auroc(X, y, clf_params=None, n_splits=5):
    default = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                   subsample=0.8, min_samples_leaf=5, random_state=42)
    params  = {**default, **(clf_params or {})}
    gb_keys = {'n_estimators', 'max_depth', 'learning_rate',
               'subsample', 'min_samples_leaf', 'random_state'}
    params  = {k: v for k, v in params.items() if k in gb_keys}

    X   = np.nan_to_num(X.astype(np.float64), nan=0, posinf=1e6, neginf=-1e6)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        sc  = StandardScaler()
        Xtr = np.nan_to_num(sc.fit_transform(X[tr]), nan=0, posinf=1e6, neginf=-1e6)
        Xte = np.nan_to_num(sc.transform(X[te]),     nan=0, posinf=1e6, neginf=-1e6)
        clf = GradientBoostingClassifier(**params)
        clf.fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


# ─── Ablation ─────────────────────────────────────────────────────────────────

def run_ablation(results, logger, clf_params=None,
                 windows=None, layer_start_frac=0.0, layer_end_frac=1.0):
    y = np.array([int(r['is_correct']) for r in results])
    logger.info(f"\n{'='*62}")
    logger.info(f"  ABLATION ({len(results)} samples)  "
                f"layers={layer_start_frac*100:.0f}%–{layer_end_frac*100:.0f}%  "
                f"windows={windows or ALL_WINDOWS}")
    logger.info(f"{'='*62}")
    logger.info(f"  {'Metric':22s}  {'LogReg AUROC':>12}  {'±':>6}")
    logger.info(f"  {'-'*44}")

    for name, _ in ABLATION_CONFIGS.items():
        X, _ = get_ablation_features(results, name,
                                     windows=windows or ALL_WINDOWS,
                                     layer_start_frac=layer_start_frac,
                                     layer_end_frac=layer_end_frac)
        X   = np.nan_to_num(X.astype(np.float64), nan=0, posinf=1e6, neginf=-1e6)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr, te in skf.split(X, y):
            sc  = StandardScaler()
            Xtr = np.nan_to_num(sc.fit_transform(X[tr]), nan=0, posinf=1e6, neginf=-1e6)
            Xte = np.nan_to_num(sc.transform(X[te]),     nan=0, posinf=1e6, neginf=-1e6)
            lr  = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
            lr.fit(Xtr, y[tr])
            aucs.append(roc_auc_score(y[te], lr.predict_proba(Xte)[:, 1]))
        mu, sd = float(np.mean(aucs)), float(np.std(aucs))
        logger.info(f"  {name:22s}  {mu:.4f}±{sd:.4f}")

    X_h, _ = get_ablation_features(results, 'Hybrid',
                                   windows=windows or ALL_WINDOWS,
                                   layer_start_frac=layer_start_frac,
                                   layer_end_frac=layer_end_frac)
    mu_h, sd_h = cv_auroc(X_h, y, clf_params)
    logger.info(f"  {'VANE (GradBoost)':22s}  ---          {mu_h:.4f}±{sd_h:.4f}")
    logger.info(f"{'='*62}")
    return mu_h


# ─── Optuna ───────────────────────────────────────────────────────────────────

def run_optuna(results, n_trials, params_path, study_path,
               study_db_path, logger):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y = np.array([int(r['is_correct']) for r in results])

    def objective(trial):
        gb_params = dict(
            n_estimators     = trial.suggest_int('n_estimators',     50, 500),
            max_depth        = trial.suggest_int('max_depth',          2,   8),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample        = trial.suggest_float('subsample',      0.4, 1.0),
            min_samples_leaf = trial.suggest_int('min_samples_leaf',   1,  30),
            random_state     = 42,
        )
        ls = trial.suggest_float('layer_start_frac', 0.0, 0.5)
        le = trial.suggest_float('layer_end_frac',   0.5, 1.0)
        wc = trial.suggest_categorical('windows', [
            'all', 'max_only', 'mean_only', 'ans_only',
            'max_mean', 'max_ans', 'mean_ans'])
        window_map = {
            'all': ['max', 'mean', 'ans'], 'max_only': ['max'],
            'mean_only': ['mean'],         'ans_only':  ['ans'],
            'max_mean':  ['max', 'mean'],  'max_ans':   ['max', 'ans'],
            'mean_ans':  ['mean', 'ans'],
        }
        X, _ = get_ablation_features(results, 'Hybrid',
                                     windows=window_map[wc],
                                     layer_start_frac=ls,
                                     layer_end_frac=le)
        mu, _ = cv_auroc(X, y, gb_params)
        return mu if not np.isnan(mu) else 0.0

    study = optuna.create_study(
        study_name="vane_math500",
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///{study_db_path}",
        load_if_exists=True,
    )
    done      = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    logger.info(f"Optuna: {done} done, {remaining} remaining (target {n_trials})")

    def _cb(study, trial):
        if (trial.number + 1) % 20 == 0:
            logger.info(f"  Optuna {trial.number+1}/{n_trials}  "
                        f"best={study.best_value:.4f}")

    study.optimize(objective, n_trials=remaining,
                   show_progress_bar=False, callbacks=[_cb])

    window_map = {
        'all': ['max', 'mean', 'ans'], 'max_only': ['max'],
        'mean_only': ['mean'],         'ans_only':  ['ans'],
        'max_mean':  ['max', 'mean'],  'max_ans':   ['max', 'ans'],
        'mean_ans':  ['mean', 'ans'],
    }
    best = study.best_params
    best['windows_decoded'] = window_map[best['windows']]
    best['best_cv_auroc']   = study.best_value
    logger.info(f"Optuna best AUROC: {study.best_value:.4f}")

    with open(params_path, 'w') as f:
        json.dump(best, f, indent=2)
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    return best


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(42)

    model_short  = model_short_name(args.model_id)
    run_dir      = os.path.join(args.output_dir, f"{model_short}_math500")
    os.makedirs(run_dir, exist_ok=True)

    ckpt_path    = os.path.join(run_dir, 'checkpoint.pkl')
    hs_ckpt_path = os.path.join(run_dir, 'hidden_states_ckpt.npz')
    hs_out_path  = os.path.join(run_dir, 'hidden_states.npz')
    hs_meta_path = os.path.join(run_dir, 'hidden_states_meta.json')
    params_path  = os.path.join(run_dir, 'optuna_best_params.json')
    study_path   = os.path.join(run_dir, 'optuna_study.pkl')
    study_db     = os.path.join(run_dir, 'optuna_study.db')
    progress_log = os.path.join(run_dir, 'progress.log')
    gpu_log      = os.path.join(run_dir, 'gpu_monitor.log')

    logger = setup_logger(progress_log)
    logger.info('='*60)
    logger.info('  VANE — MATH-500 Experiment')
    logger.info(f'  Model:   {args.model_id}')
    logger.info(f'  Output:  {run_dir}')
    logger.info(f'  Optuna:  {"ON (" + str(args.optuna_trials) + " trials)" if args.optuna else "OFF"}')
    logger.info('='*60)

    threading.Thread(target=gpu_wattage_monitor,
                     args=(gpu_log, 900, 100), daemon=True).start()

    # ── Load checkpoint ──────────────────────────────────────────────────────
    results = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"Resumed: {len(results)} samples already done")
    done_set = {r['sample_idx'] for r in results}

    # ── Load MATH-500 dataset ────────────────────────────────────────────────
    logger.info("Loading MATH-500 dataset...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    N = len(ds)
    logger.info(f"Loaded {N} problems")

    questions  = [ds[i]['problem']              for i in range(N)]
    gt_answers = [ds[i]['answer']               for i in range(N)]
    subjects   = [ds[i].get('subject', 'unknown') for i in range(N)]

    remaining = [i for i in range(N) if i not in done_set]
    logger.info(f"Remaining: {len(remaining)} samples")

    if args.eval_only:
        logger.info("--eval_only: skipping inference")
    elif remaining:
        # ── Load model ───────────────────────────────────────────────────────
        logger.info("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, device_map=None,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="eager",
        ).to('cuda:0')
        model.eval()

        n_layers, hidden_dim = get_model_dims(model)
        logger.info(f"Model loaded. Layers: {n_layers}  Hidden dim: {hidden_dim}")

        # ── Hidden-state buffer ──────────────────────────────────────────────
        hidden_buf  = np.zeros((N, n_layers, hidden_dim), dtype=np.float16)
        labels_buf  = np.zeros(N, dtype=np.int8)
        logprob_buf = np.zeros(N, dtype=np.float32)
        idx_buf     = np.arange(N, dtype=np.int32)

        if os.path.exists(hs_ckpt_path):
            ck = np.load(hs_ckpt_path)
            n_hs_done = int(ck['n_done'])
            hidden_buf[:n_hs_done]  = ck['hidden'][:n_hs_done]
            labels_buf[:n_hs_done]  = ck['labels'][:n_hs_done]
            logprob_buf[:n_hs_done] = ck['logprobs'][:n_hs_done]
            logger.info(f"Resumed hidden states: {n_hs_done} samples from ckpt")
        elif results:
            logger.info("No hidden-state ckpt found; backfill will start fresh")

        # ── Batch inference ──────────────────────────────────────────────────
        batches = [remaining[i:i + args.batch_size]
                   for i in range(0, len(remaining), args.batch_size)]
        logger.info(f"Running {len(remaining)} samples in {len(batches)} batches")

        n_correct     = sum(r['is_correct'] for r in results)
        last_eval_n   = len(results)   # track when we last ran ablation

        with tqdm(total=len(remaining), desc="Inference") as pbar:
            for batch_indices in batches:
                prompts = [build_prompt(questions[i], tokenizer)
                           for i in batch_indices]
                gts     = [gt_answers[i] for i in batch_indices]

                batch_in = tokenizer(
                    prompts, return_tensors='pt',
                    padding=True, truncation=False,
                ).to(model.device)
                padded_len = batch_in['input_ids'].shape[1]

                with torch.no_grad():
                    out = model.generate(
                        **batch_in,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=tokenizer.pad_token_id,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                    )

                for b, (idx, gt) in enumerate(zip(batch_indices, gts)):
                    gen_ids  = out.sequences[b].unsqueeze(0)
                    gen_text = tokenizer.decode(
                        gen_ids[0][padded_len:], skip_special_tokens=True)

                    pred, is_correct = extract_answer_math(gen_text, gt)

                    with torch.no_grad():
                        fwd = model(gen_ids,
                                    attention_mask=torch.ones_like(gen_ids),
                                    output_hidden_states=True)

                    mean_log_prob = 0.0
                    try:
                        gen_len = gen_ids.shape[1] - padded_len
                        if gen_len > 1:
                            logits  = fwd.logits[0, padded_len:-1, :]
                            targets = gen_ids[0, padded_len + 1:]
                            lp      = torch.log_softmax(logits, dim=-1)
                            mean_log_prob = lp[
                                range(len(targets)), targets].mean().item()
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
                        'subject':       subjects[idx],
                        'gen_length':    gen_ids.shape[1] - padded_len,
                    })
                    results.append(metrics)
                    n_correct += is_correct

                    hidden_buf[idx]  = extract_mean_hidden(
                        fwd.hidden_states, padded_len, n_layers)
                    labels_buf[idx]  = is_correct
                    logprob_buf[idx] = mean_log_prob

                    pbar.update(1)

                # Save checkpoint after every batch
                with open(ckpt_path, 'wb') as f:
                    pickle.dump(results, f)

                n_done = len(results)
                # Trigger every eval_every samples (robust to batch size)
                if n_done - last_eval_n >= args.eval_every and n_done >= 50:
                    acc = n_correct / n_done
                    logger.info(f"\n[{n_done}/{N}] acc={acc:.3f}")
                    run_ablation(results, logger)
                    max_idx_done = max(r['sample_idx'] for r in results) + 1
                    save_hidden_states_ckpt(
                        hs_ckpt_path, hidden_buf, labels_buf,
                        logprob_buf, idx_buf, max_idx_done)
                    last_eval_n = n_done

        n_correct_final = sum(r['is_correct'] for r in results)
        acc = n_correct_final / len(results)
        logger.info(f"\nFinal accuracy: {acc:.4f}  ({n_correct_final}/{len(results)})")

        # Subject-level accuracy
        subj_correct = defaultdict(int)
        subj_total   = defaultdict(int)
        for r in results:
            subj_total[r['subject']]   += 1
            subj_correct[r['subject']] += r['is_correct']
        logger.info("Subject accuracy:")
        for subj in sorted(subj_total):
            n, c = subj_total[subj], subj_correct[subj]
            logger.info(f"  {subj:35s}: {c/n:.3f}  ({c}/{n})")

        # ── Save final hidden_states.npz ─────────────────────────────────────
        np.savez_compressed(
            hs_out_path,
            hidden   = hidden_buf[:N],
            labels   = labels_buf[:N],
            logprobs = logprob_buf[:N],
            indices  = idx_buf[:N],
        )
        if os.path.exists(hs_ckpt_path):
            os.remove(hs_ckpt_path)
        logger.info(f"Hidden states saved: {hs_out_path}")
        logger.info(f"  Shape: {hidden_buf.shape}  dtype: float16")
        logger.info(f"  Size : {hidden_buf.nbytes / 1e6:.1f} MB uncompressed")

        with open(hs_meta_path, 'w') as f:
            json.dump({
                'model_id':   args.model_id,
                'benchmark':  'MATH-500',
                'n_layers':   n_layers,
                'hidden_dim': hidden_dim,
                'N':          N,
                'accuracy':   float(acc),
            }, f, indent=2)

    # ── Optuna ──────────────────────────────────────────────────────────────
    clf_params = None
    layer_start_frac, layer_end_frac = 0.0, 1.0
    windows = None

    if args.optuna and len(results) >= 50:
        best             = run_optuna(results, args.optuna_trials,
                                      params_path, study_path, study_db, logger)
        clf_params       = {k: best[k] for k in
                            ['n_estimators', 'max_depth', 'learning_rate',
                             'subsample', 'min_samples_leaf']
                            if k in best}
        layer_start_frac = best.get('layer_start_frac', 0.0)
        layer_end_frac   = best.get('layer_end_frac',   1.0)
        windows          = best.get('windows_decoded',  None)

    elif os.path.exists(params_path):
        with open(params_path) as f:
            best = json.load(f)
        logger.info(f"Loaded Optuna params  best_cv_auroc={best.get('best_cv_auroc','?')}")
        clf_params       = {k: best[k] for k in
                            ['n_estimators', 'max_depth', 'learning_rate',
                             'subsample', 'min_samples_leaf']
                            if k in best}
        layer_start_frac = best.get('layer_start_frac', 0.0)
        layer_end_frac   = best.get('layer_end_frac',   1.0)
        windows          = best.get('windows_decoded',  None)

    # ── Final ablation ───────────────────────────────────────────────────────
    if len(results) >= 20:
        logger.info("\nFINAL ABLATION:")
        run_ablation(results, logger, clf_params,
                     windows, layer_start_frac, layer_end_frac)

    logger.info(f"\nCheckpoint saved: {ckpt_path}")
    logger.info("Done.")


if __name__ == '__main__':
    main()
