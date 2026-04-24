"""
Test script for Multimodal 2-Factor model (WSI + LE, logistic regression fusion).

Uses pre-trained WSI GAP MLP (k-fold), LE GMP MLP (k-fold), and a pre-trained
logistic regression that fuses the two soft-average probabilities.

Evaluates two matched test sets:
  - testing   : slides in LE internal test set (le_dataset_split.csv, split='testing')
  - ext_test  : slides in matched external set (external_dataset_split.csv)

Usage:
    python test_multimodal_2factor.py \\
        --wsi_run_dir  <wsi_pooling_run_dir> \\
        --le_run_dir   <le_pooling_run_dir> \\
        --logreg_dir   <dir_containing_logreg_model.pkl> \\
        --output_dir   <output_dir> \\
        [--gpu_id 0]
"""

import os
import sys
import json
import glob
import pickle
import argparse
import joblib

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, balanced_accuracy_score, confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from models import PoolingClassifier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description='Test Multimodal 2-factor fusion (WSI + LE).')
    p.add_argument('--wsi_run_dir', type=str, required=True,
                   help='WSI GAP MLP run directory.')
    p.add_argument('--le_run_dir', type=str, required=True,
                   help='LE GMP MLP run directory.')
    p.add_argument('--logreg_dir', type=str, required=True,
                   help='Directory containing logreg_model.pkl.')
    p.add_argument('--output_dir', type=str, default='./outputs/test_multimodal_2factor')
    p.add_argument('--wsi_test_feats_dir', type=str, default=None,
                   help='Override WSI internal-test feature dir (default: read from WSI run hp).')
    p.add_argument('--wsi_ext_feats_dir', type=str, default=None,
                   help='Override WSI external-test feature dir (default: read from WSI run hp).')
    p.add_argument('--wsi_dataset_csv', type=str, default=None,
                   help='Override WSI internal split CSV (default: read from WSI run hp).')
    p.add_argument('--wsi_ext_test_csv', type=str, default=None,
                   help='Override WSI external split CSV (default: read from WSI run hp).')
    p.add_argument('--le_test_feats_dir', type=str,
                   default='data/le_features/internal_testing',
                   help='Directory of per-slide H5 LE feature files (internal test set).')
    p.add_argument('--le_ext_feats_dir', type=str,
                   default='data/le_features/external_testing',
                   help='Directory of per-slide H5 LE feature files (external test set).')
    p.add_argument('--le_dataset_csv', type=str, default=None,
                   help='Override LE internal split CSV (default: read from LE run hp).')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--k_folds', type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers shared with other test scripts
# ---------------------------------------------------------------------------

def load_fold_info(run_dir, k_folds):
    """Load fold checkpoint paths and optimal thresholds."""
    perf_csv = os.path.join(run_dir, 'all_folds_performance.csv')
    thresholds = {}
    if os.path.exists(perf_csv):
        df = pd.read_csv(perf_csv)
        df.columns = df.columns.str.strip()
        for _, row in df.iterrows():
            fid = int(str(row['fold']).strip())
            bt = row.get('best_threshold', row.get('optimal_threshold', 0.5))
            thresholds[fid] = float(str(bt).strip()) if str(bt).strip() else 0.5
    fold_info = {}
    for i in range(k_folds):
        fid = i + 1
        fold_dir = os.path.join(run_dir, f'fold_{fid}')
        if not os.path.isdir(fold_dir):
            continue
        ckpts = glob.glob(os.path.join(fold_dir, 'best_model*.pth'))
        if not ckpts:
            ckpts = glob.glob(os.path.join(fold_dir, '*.pth'))
        if not ckpts:
            continue
        try:
            ckpts.sort(key=lambda x: float(x.split('score')[1].split('.pth')[0]), reverse=True)
        except Exception:
            ckpts.sort()
        fold_info[fid] = {'ckpt': ckpts[0], 'threshold': thresholds.get(fid, 0.5)}
    return fold_info


def load_pooling_models(fold_info, hp, device):
    models_list = []
    fold_ids = sorted(fold_info.keys())
    for fid in fold_ids:
        m = PoolingClassifier(
            input_dim=hp.get('input_dim', 1536),
            head='mlp',
            dropout=hp.get('dropout', 0.4),
            pooling=hp.get('pooling', 'mean'),
        )
        m.load_state_dict(torch.load(fold_info[fid]['ckpt'], map_location=device, weights_only=False))
        m.to(device).eval()
        models_list.append(m)
    return fold_ids, models_list


def infer_slide_h5(model, h5_path, device, num_patches_sample, pooling):
    """Run PoolingClassifier inference on one H5 feature file."""
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
    features = features.astype(np.float32)
    n = len(features)
    if num_patches_sample is not None and n > num_patches_sample:
        idx = np.random.choice(n, num_patches_sample, replace=False)
        features = features[idx]
    feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(feat_t)
    return torch.sigmoid(logit).item()


def get_slide_probs_pooling(df, feats_dir, fold_ids, models_list, hp):
    """Return dict: sample_id -> mean fold probability (float)."""
    num_patches_sample = hp.get('num_patches_sample', None)
    pooling = hp.get('pooling', 'mean')
    result = {}
    skipped = []
    for _, row in df.iterrows():
        sid = str(row['sample_id'])
        h5_path = os.path.join(feats_dir, f'{sid}.h5')
        if not os.path.exists(h5_path):
            skipped.append(sid)
            continue
        fold_probs = [infer_slide_h5(m, h5_path, device=next(models_list[0].parameters()).device,
                                     num_patches_sample=num_patches_sample, pooling=pooling)
                      for m in models_list]
        result[sid] = float(np.mean(fold_probs))
    if skipped:
        print(f'  Warning: {len(skipped)} slides not found in {feats_dir}: {skipped[:5]}')
    return result


# ---------------------------------------------------------------------------
# Metrics / plots
# ---------------------------------------------------------------------------

def calculate_metrics(labels, probs, threshold=0.5):
    preds = (np.array(probs) > threshold).astype(int)
    m = {}
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        m['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        m['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        m['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        m['PPV'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        m['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        m['F1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    except ValueError:
        m['Accuracy'] = float(np.mean(np.array(preds) == np.array(labels)))
        m['Sensitivity'] = m['Specificity'] = m['PPV'] = m['NPV'] = m['F1'] = 'N/A'
    m['Balanced Accuracy'] = balanced_accuracy_score(labels, preds)
    m['AUC'] = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    return m


def plot_roc(labels, probs, out_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    ba = balanced_accuracy_score(labels, (np.array(probs) > 0.5).astype(int))
    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, color='#0400ff', lw=3, label=f'AUROC = {auc:.3f}')
    plt.plot([], [], ' ', label=f'Bal. Acc = {ba:.3f}')
    plt.plot([0, 1], [0, 1], color='grey', lw=2.5, linestyle='--', label='Random')
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=26); plt.ylabel('True Positive Rate', fontsize=26)
    plt.legend(loc='lower right', fontsize=22)
    plt.xticks(fontsize=22); plt.yticks(fontsize=22)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()


def plot_cm(labels, preds, out_path, title='Confusion Matrix'):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={'size': 28})
    plt.xlabel('Predicted Label', fontsize=24); plt.ylabel('True Label', fontsize=24)
    plt.title(title, fontsize=28, fontweight='bold')
    plt.xticks(fontsize=20); plt.yticks(fontsize=20)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def write_metrics_txt(m, out_path, threshold=0.5):
    with open(out_path, 'w') as f:
        f.write(f'--- Ensemble Performance (Threshold={threshold}) ---\n')
        for k in ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1',
                  'Balanced Accuracy', 'AUC']:
            f.write(f'{k}: {m.get(k, "N/A"):.4f}\n')


# ---------------------------------------------------------------------------
# Save all outputs for one test split
# ---------------------------------------------------------------------------

def save_split_results(slide_ids, labels, wsi_probs, le_probs, final_probs, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    labels_arr = np.array(labels)
    final_arr = np.array(final_probs)
    preds_05 = (final_arr > 0.5).astype(int)
    correct = (preds_05 == labels_arr).astype(int)

    # inputs CSV
    pd.DataFrame({
        'sample_id': slide_ids, 'label': labels,
        'wsi_prob': wsi_probs, 'le_prob': le_probs,
    }).to_csv(os.path.join(out_dir, f'{prefix}_ensemble_inputs.csv'), index=False)

    # predictions CSV
    pd.DataFrame({
        'slide_id': slide_ids, 'ground_truth_label': labels,
        'ensembled_probability': final_probs,
        'predicted_label_threshold0.5': preds_05.tolist(),
        'is_correct': correct.tolist(),
    }).to_csv(os.path.join(out_dir, f'{prefix}_ensemble_predictions.csv'), index=False)

    m = calculate_metrics(labels_arr, final_arr, 0.5)

    # per-slide probs JSON
    per_slide = {str(sid): {'label': int(labels_arr[i]),
                             'wsi_prob': float(wsi_probs[i]),
                             'le_prob': float(le_probs[i]),
                             'final_prob': float(final_arr[i])}
                 for i, sid in enumerate(slide_ids)}
    with open(os.path.join(out_dir, f'{prefix}_with_sd.json'), 'w') as f:
        json.dump({'per_slide': per_slide}, f, indent=4)

    # summary JSON
    with open(os.path.join(out_dir, f'{prefix}_ensemble_summary.json'), 'w') as f:
        json.dump({'test_set_metrics': m, 'optimal_threshold': None}, f, indent=4)

    write_metrics_txt(m, os.path.join(out_dir, f'{prefix}_ensemble_metrics_threshold0.5.txt'))
    plot_roc(labels_arr, final_arr, os.path.join(out_dir, f'{prefix}_ensemble_roc_curve.png'))
    plot_cm(labels_arr, preds_05, os.path.join(out_dir,
            f'{prefix}_ensemble_confusion_matrix_threshold0.5.png'),
            'Ensemble (Threshold=0.5)')
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load hyperparameters ----
    def _hp(run_dir):
        p = os.path.join(run_dir, 'fold_1', 'hyperparameters.json')
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return {}

    wsi_hp = _hp(args.wsi_run_dir)
    le_hp = _hp(args.le_run_dir)

    wsi_test_feats_dir = args.wsi_test_feats_dir or wsi_hp.get('test_feats_dir',
        'data/wsi_features/internal_testing')
    wsi_ext_feats_dir = args.wsi_ext_feats_dir or wsi_hp.get('ext_test_feats_dir',
        'data/wsi_features/external_testing')
    wsi_dataset_csv = args.wsi_dataset_csv or wsi_hp.get('dataset_csv',
        'data/internal_dataset_split.csv')
    wsi_ext_test_csv = args.wsi_ext_test_csv or wsi_hp.get('ext_test_csv',
        'data/external_dataset_split.csv')

    le_test_feats_dir = args.le_test_feats_dir
    le_ext_feats_dir = args.le_ext_feats_dir
    le_dataset_csv = args.le_dataset_csv or le_hp.get('dataset_csv',
        'data/le_dataset_split.csv')

    # ---- Load logreg ----
    logreg_path = os.path.join(args.logreg_dir, 'logreg_model.pkl')
    try:
        logreg = joblib.load(logreg_path)
    except Exception:
        with open(logreg_path, 'rb') as f:
            logreg = pickle.load(f)
    print(f'Loaded logreg from: {logreg_path}')

    # ---- Load fold models ----
    wsi_fold_info = load_fold_info(args.wsi_run_dir, args.k_folds)
    le_fold_info = load_fold_info(args.le_run_dir, args.k_folds)

    print('\nLoading WSI models...')
    wsi_fold_ids, wsi_models = load_pooling_models(wsi_fold_info, wsi_hp, device)
    print(f'  {len(wsi_models)} folds: {wsi_fold_ids}')

    print('Loading LE models...')
    le_fold_ids, le_models = load_pooling_models(le_fold_info, le_hp, device)
    print(f'  {len(le_models)} folds: {le_fold_ids}')

    # ---- TESTING (LE internal test slides) ----
    le_df = pd.read_csv(le_dataset_csv)
    le_test_df = le_df[le_df['split'] == 'testing'].reset_index(drop=True)
    print(f'\nLE test slides: {len(le_test_df)}')

    print('  Getting WSI probs (internal test features)...')
    wsi_probs_test = get_slide_probs_pooling(le_test_df, wsi_test_feats_dir,
                                              wsi_fold_ids, wsi_models, wsi_hp)
    print('  Getting LE probs...')
    le_probs_test = get_slide_probs_pooling(le_test_df, le_test_feats_dir,
                                             le_fold_ids, le_models, le_hp)

    matched_ids_t, matched_labels_t = [], []
    wsi_p_t, le_p_t = [], []
    for _, row in le_test_df.iterrows():
        sid = str(row['sample_id'])
        if sid in wsi_probs_test and sid in le_probs_test:
            matched_ids_t.append(sid)
            matched_labels_t.append(int(row['label']))
            wsi_p_t.append(wsi_probs_test[sid])
            le_p_t.append(le_probs_test[sid])
        else:
            print(f'  Skipping {sid}: WSI={sid in wsi_probs_test}, LE={sid in le_probs_test}')

    X_t = np.column_stack([wsi_p_t, le_p_t])
    final_probs_t = logreg.predict_proba(X_t)[:, 1].tolist()

    m_test = save_split_results(matched_ids_t, matched_labels_t, wsi_p_t, le_p_t,
                                 final_probs_t, args.output_dir, 'testing')
    print(f'\n--- Testing (Internal Matched) ---')
    print(f'  [testing] N={len(matched_ids_t)} AUC={m_test["AUC"]:.4f}'
          f'  BalAcc={m_test["Balanced Accuracy"]:.4f}'
          f'  Sens={m_test["Sensitivity"]:.4f}  Spec={m_test["Specificity"]:.4f}')

    # ---- EXT_TEST (matched external slides) ----
    ext_df = pd.read_csv(wsi_ext_test_csv)
    ext_df = ext_df[ext_df['split'] == 'ext_test'].reset_index(drop=True)
    print(f'\nExternal matched slides: {len(ext_df)}')

    print('  Getting WSI probs (ext features)...')
    wsi_probs_ext = get_slide_probs_pooling(ext_df, wsi_ext_feats_dir,
                                             wsi_fold_ids, wsi_models, wsi_hp)
    print('  Getting LE probs (ext features)...')
    le_probs_ext = get_slide_probs_pooling(ext_df, le_ext_feats_dir,
                                            le_fold_ids, le_models, le_hp)

    matched_ids_e, matched_labels_e = [], []
    wsi_p_e, le_p_e = [], []
    for _, row in ext_df.iterrows():
        sid = str(row['sample_id'])
        if sid in wsi_probs_ext and sid in le_probs_ext:
            matched_ids_e.append(sid)
            matched_labels_e.append(int(row['label']))
            wsi_p_e.append(wsi_probs_ext[sid])
            le_p_e.append(le_probs_ext[sid])
        else:
            print(f'  Skipping {sid}: WSI={sid in wsi_probs_ext}, LE={sid in le_probs_ext}')

    if matched_ids_e:
        X_e = np.column_stack([wsi_p_e, le_p_e])
        final_probs_e = logreg.predict_proba(X_e)[:, 1].tolist()
        m_ext = save_split_results(matched_ids_e, matched_labels_e, wsi_p_e, le_p_e,
                                    final_probs_e, args.output_dir, 'ext_test')
        print(f'\n--- External Test Set ---')
        print(f'  [ext_test] N={len(matched_ids_e)} AUC={m_ext["AUC"]:.4f}'
              f'  BalAcc={m_ext["Balanced Accuracy"]:.4f}'
              f'  Sens={m_ext["Sensitivity"]:.4f}  Spec={m_ext["Specificity"]:.4f}')

    print(f'\nAll outputs saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
