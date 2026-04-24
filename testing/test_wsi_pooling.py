"""
Test script for WSI Global Mean Pooling MLP (WSI GAP MLP) using k-fold ensemble.

Evaluates both the internal test set and the external test set using a 10-fold
ensemble of PoolingClassifier models (1536 -> 768 -> 384 -> 1, LayerNorm + GELU).

Usage:
    python test_wsi_pooling.py --run_dir <path_to_run> --output_dir <output_dir> [--gpu_id 0]
"""

import os
import sys
import json
import glob
import argparse
import datetime

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, confusion_matrix

# Import model from sibling training module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from models import PoolingClassifier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description='Test WSI GAP MLP (k-fold ensemble).')
    p.add_argument('--run_dir', type=str, required=True,
                   help='Path to the k-fold run directory containing fold_1 … fold_10.')
    p.add_argument('--output_dir', type=str, default='./output_test_wsi_pooling',
                   help='Directory to save evaluation outputs.')
    p.add_argument('--gpu_id', type=int, default=0,
                   help='GPU ID (use -1 for CPU).')
    p.add_argument('--k_folds', type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class H5SlideDataset(Dataset):
    def __init__(self, df, feats_dir, num_patches=1700, seed=42):
        self.df = df
        self.feats_dir = feats_dir
        self.num_patches = num_patches
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row['sample_id'])
        label = torch.tensor(row['label'], dtype=torch.float32)
        h5_path = os.path.join(self.feats_dir, slide_id + '.h5')
        try:
            with h5py.File(h5_path, 'r') as f:
                feats = torch.from_numpy(f['features'][:]).float()
        except Exception as e:
            print(f'  Warning: cannot load {h5_path}: {e}')
            return None
        n = feats.shape[0]
        if n >= self.num_patches:
            idx_sel = torch.linspace(0, n - 1, self.num_patches).long()
            feats = feats[idx_sel]
        return feats, label, slide_id


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None
    feats_list, labels, slide_ids = zip(*batch)
    lengths = [f.shape[0] for f in feats_list]
    max_len = max(lengths)
    fdim = feats_list[0].shape[1]
    padded = torch.zeros(len(batch), max_len, fdim)
    mask = torch.zeros(len(batch), max_len)
    for i, f in enumerate(feats_list):
        padded[i, :lengths[i]] = f
        mask[i, :lengths[i]] = 1.0
    return padded, mask, torch.stack(labels), list(slide_ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fold_info(run_dir, k_folds):
    """Return {fold_id: {'ckpt': path, 'threshold': float}}."""
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
            continue
        try:
            ckpts.sort(key=lambda x: float(x.split('score')[1].split('.pth')[0]), reverse=True)
        except Exception:
            ckpts.sort()
        fold_info[fid] = {
            'ckpt': ckpts[0],
            'threshold': thresholds.get(fid, 0.5),
        }
    return fold_info


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


def plot_roc(labels, probs, out_path, bal_acc=None):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    if bal_acc is None:
        yi = np.argmax(tpr - fpr)
        bal_acc = balanced_accuracy_score(labels, (np.array(probs) > _[yi]).astype(int))
    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, color='#0400ff', lw=3, label=f'AUROC = {auc:.3f}')
    plt.plot([], [], ' ', label=f'Bal. Acc = {bal_acc:.3f}')
    plt.plot([0, 1], [0, 1], color='grey', lw=2.5, linestyle='--', label='Random')
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=26)
    plt.legend(loc='lower right', fontsize=22)
    plt.xticks(fontsize=22); plt.yticks(fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cm(labels, preds, out_path, title='Confusion Matrix'):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={'size': 28})
    plt.xlabel('Predicted Label', fontsize=24)
    plt.ylabel('True Label', fontsize=24)
    plt.title(title, fontsize=28, fontweight='bold')
    plt.xticks(fontsize=20); plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_results(slide_ids, labels, per_fold_probs, fold_ids, fold_thresholds, out_dir, prefix):
    """Save all outputs for one evaluation split."""
    os.makedirs(out_dir, exist_ok=True)
    labels = np.array(labels)
    per_fold_probs = np.array(per_fold_probs)   # (n_samples, n_folds)
    ensemble_probs = per_fold_probs.mean(axis=1)

    # ----- per-fold CSVs -----
    rows_05, rows_opt = [], []
    for col, fid in enumerate(fold_ids):
        fp = per_fold_probs[:, col]
        m05 = calculate_metrics(labels, fp, threshold=0.5)
        m05.update({'fold': fid, 'threshold': 0.5})
        rows_05.append(m05)
        t = fold_thresholds.get(fid, 0.5)
        mopt = calculate_metrics(labels, fp, threshold=t)
        mopt.update({'fold': fid, 'threshold': t})
        rows_opt.append(mopt)
        plot_cm(labels, (fp > 0.5).astype(int),
                os.path.join(out_dir, f'{prefix}_fold_{fid}_confusion_matrix_thresh0.5.png'),
                title=f'Fold {fid} (Threshold=0.5)')
        plot_cm(labels, (fp > t).astype(int),
                os.path.join(out_dir, f'{prefix}_fold_{fid}_confusion_matrix_optimal.png'),
                title=f'Fold {fid} (Threshold={t:.4f})')

    def _reorder(df):
        cols = ['fold', 'threshold'] + [c for c in df.columns if c not in ('fold', 'threshold')]
        return df[cols]

    _reorder(pd.DataFrame(rows_05)).to_csv(
        os.path.join(out_dir, f'{prefix}_per_fold_performance_threshold0.5.csv'), index=False)
    _reorder(pd.DataFrame(rows_opt)).to_csv(
        os.path.join(out_dir, f'{prefix}_per_fold_performance_optimal_threshold.csv'), index=False)

    # ----- ensemble metrics -----
    m05 = calculate_metrics(labels, ensemble_probs, threshold=0.5)
    avg_thresh = float(np.mean(list(fold_thresholds.values()))) if fold_thresholds else 0.5

    # Hard voting (per-fold optimal thresholds)
    preds_per_fold = [(per_fold_probs[:, i] > fold_thresholds.get(fid, 0.5)).astype(int)
                      for i, fid in enumerate(fold_ids)]
    vote_frac = np.mean(preds_per_fold, axis=0)
    m_hard = calculate_metrics(labels, vote_frac, threshold=0.5)

    # Soft average (ensemble prob > avg threshold)
    m_soft = calculate_metrics(labels, ensemble_probs, threshold=avg_thresh)

    summary = {
        'average_optimal_threshold': avg_thresh,
        'test_set_metrics': m05,
        'test_set_metrics_optimal': m_hard,
        'test_set_metrics_soft_avg': m_soft,
    }
    with open(os.path.join(out_dir, f'{prefix}_ensemble_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # ----- per-slide SD -----
    per_slide = {}
    for i, sid in enumerate(slide_ids):
        ps = per_fold_probs[i].tolist()
        per_slide[str(sid)] = {
            'label': int(labels[i]),
            'probs': ps,
            'mean_prob': float(np.mean(ps)),
            'std_prob': float(np.std(ps)),
        }
    with open(os.path.join(out_dir, f'{prefix}_with_sd.json'), 'w') as f:
        json.dump({'per_slide': per_slide}, f, indent=4)

    # ----- predictions CSV -----
    pd.DataFrame({
        'slide_id': slide_ids,
        'label': labels.tolist(),
        'ensemble_prob': ensemble_probs.tolist(),
        'pred_05': (ensemble_probs > 0.5).astype(int).tolist(),
        'pred_soft_avg': (ensemble_probs > avg_thresh).astype(int).tolist(),
        'pred_hard_vote': (vote_frac > 0.5).astype(int).tolist(),
    }).to_csv(os.path.join(out_dir, f'{prefix}_ensemble_predictions.csv'), index=False)

    # ----- ROC curves -----
    plot_roc(labels, ensemble_probs,
             os.path.join(out_dir, f'{prefix}_ensemble_roc_curve.png'),
             bal_acc=m_soft['Balanced Accuracy'])
    plot_roc(labels, vote_frac,
             os.path.join(out_dir, f'{prefix}_ensemble_roc_curve_optimal_threshold.png'),
             bal_acc=m_hard['Balanced Accuracy'])

    # ----- confusion matrices -----
    plot_cm(labels, (ensemble_probs > 0.5).astype(int),
            os.path.join(out_dir, f'{prefix}_ensemble_confusion_matrix_threshold0.5.png'),
            title='Ensemble (Threshold=0.5)')
    plot_cm(labels, (vote_frac > 0.5).astype(int),
            os.path.join(out_dir, f'{prefix}_ensemble_confusion_matrix_optimal_threshold.png'),
            title='Ensemble (Hard Voting)')
    plot_cm(labels, (ensemble_probs > avg_thresh).astype(int),
            os.path.join(out_dir, f'{prefix}_ensemble_confusion_matrix_soft_avg_threshold.png'),
            title=f'Ensemble (Soft Avg >{avg_thresh:.4f})')

    return m_soft


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(df, feats_dir, fold_info, model_fn, device, num_patches, seed):
    """Run ensemble inference. Returns (slide_ids, labels, per_fold_probs, fold_ids, thresholds)."""
    dataset = H5SlideDataset(df, feats_dir, num_patches=num_patches, seed=seed)

    def _collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None, None, None, None
        feats_list, labels, slide_ids = zip(*batch)
        lengths = [f.shape[0] for f in feats_list]
        max_len = max(lengths)
        fdim = feats_list[0].shape[1]
        padded = torch.zeros(len(batch), max_len, fdim)
        mask_t = torch.zeros(len(batch), max_len)
        for i, f in enumerate(feats_list):
            padded[i, :lengths[i]] = f
            mask_t[i, :lengths[i]] = 1.0
        return padded, mask_t, torch.stack(labels), list(slide_ids)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=_collate, num_workers=4)

    fold_ids = sorted(fold_info.keys())
    models_list = []
    for fid in fold_ids:
        m = model_fn()
        state = torch.load(fold_info[fid]['ckpt'], map_location=device, weights_only=False)
        m.load_state_dict(state)
        m.to(device).eval()
        models_list.append(m)
        print(f'  Loaded fold {fid}: {os.path.basename(fold_info[fid]["ckpt"])}')

    all_slide_ids, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            if batch[0] is None:
                continue
            feats, mask, label, slide_ids = batch
            feats = feats.to(device)
            mask = mask.to(device)

            fold_probs = []
            for m in models_list:
                logit = m(feats, mask)
                prob = torch.sigmoid(logit).cpu().item()
                fold_probs.append(prob)

            all_slide_ids.extend(slide_ids)
            all_labels.append(label.item())
            all_probs.append(fold_probs)

    thresholds = {fid: fold_info[fid]['threshold'] for fid in fold_ids}
    return all_slide_ids, np.array(all_labels), np.array(all_probs), fold_ids, thresholds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # ----- device -----
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # ----- load hyperparameters -----
    hp_path = os.path.join(args.run_dir, 'fold_1', 'hyperparameters.json')
    hp = {}
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            hp = json.load(f)
        print(f'Loaded hyperparameters from {hp_path}')

    dropout = hp.get('dropout', 0.4)
    pooling = hp.get('pooling', 'mean')
    num_patches = hp.get('num_patches_sample', 1700)
    seed = hp.get('seed', 42)
    dataset_csv = hp.get('dataset_csv', 'data/internal_dataset_split.csv')
    ext_test_csv = hp.get('ext_test_csv', 'data/external_dataset_split.csv')
    test_feats_dir = hp.get('test_feats_dir',
        'data/wsi_features/internal_testing')
    ext_feats_dir = hp.get('ext_test_feats_dir',
        'data/wsi_features/external_testing')

    # ----- fold info -----
    fold_info = load_fold_info(args.run_dir, args.k_folds)
    fold_ids = sorted(fold_info.keys())
    avg_thresh = np.mean([fold_info[fid]['threshold'] for fid in fold_ids])
    print(f'Folds: {fold_ids}')
    print(f'Avg threshold: {avg_thresh:.4f}')

    # Detect head type from a representative checkpoint.
    _probe_ckpt = fold_info[fold_ids[0]]['ckpt']
    _probe_sd = torch.load(_probe_ckpt, map_location='cpu', weights_only=False)
    head = 'linear' if 'classifier.weight' in _probe_sd else 'mlp'
    del _probe_sd
    print(f'Pooling head: {head}')

    def model_fn():
        return PoolingClassifier(input_dim=1536, head=head, dropout=dropout, pooling=pooling)

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- internal test set -----
    full_df = pd.read_csv(dataset_csv)
    test_df = full_df[full_df['split'] == 'testing'].reset_index(drop=True)
    print(f'\nTest set: {len(test_df)} slides')

    print('\nRunning inference on internal test set...')
    slide_ids, labels, probs, f_ids, thresholds = run_inference(
        test_df, test_feats_dir, fold_info, model_fn, device, num_patches, seed)
    m = save_results(slide_ids, labels, probs, f_ids, thresholds,
                     args.output_dir, 'testing')
    print(f'\n--- Internal Test Set ---')
    print(f'  [testing] AUC={m["AUC"]:.4f}  BalAcc={m["Balanced Accuracy"]:.4f}'
          f'  Sens={m["Sensitivity"]:.4f}  Spec={m["Specificity"]:.4f}')

    # ----- external test set -----
    ext_df = pd.read_csv(ext_test_csv)
    ext_df = ext_df[ext_df['split'] == 'ext_test'].reset_index(drop=True)
    print(f'\nExternal test set: {len(ext_df)} slides')

    print('\nRunning inference on external test set...')
    slide_ids_e, labels_e, probs_e, f_ids_e, thresholds_e = run_inference(
        ext_df, ext_feats_dir, fold_info, model_fn, device, num_patches, seed)
    m_e = save_results(slide_ids_e, labels_e, probs_e, f_ids_e, thresholds_e,
                       args.output_dir, 'ext_test')
    print(f'\n--- External Test Set ---')
    print(f'  [ext_test] AUC={m_e["AUC"]:.4f}  BalAcc={m_e["Balanced Accuracy"]:.4f}'
          f'  Sens={m_e["Sensitivity"]:.4f}  Spec={m_e["Specificity"]:.4f}')

    print(f'\nAll outputs saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
