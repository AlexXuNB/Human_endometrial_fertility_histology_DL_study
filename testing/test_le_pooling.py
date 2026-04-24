"""
Test script for LE Global Mean Pooling MLP (LE GMP MLP) using k-fold ensemble.

Evaluates both the internal LE test set and the external LE test set using a
10-fold ensemble of PoolingClassifier models (1536 -> 768 -> 384 -> 1, LayerNorm + GELU).

Usage:
    python test_le_pooling.py --run_dir <path_to_run> --output_dir <output_dir> [--gpu_id 0]
"""

import os
import sys
import json
import glob
import argparse

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from models import PoolingClassifier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description='Test LE GMP MLP (k-fold ensemble).')
    p.add_argument('--run_dir', type=str, required=True,
                   help='Path to the k-fold run directory containing fold_1 … fold_10.')
    p.add_argument('--output_dir', type=str, default='./outputs/test_le_pooling')
    p.add_argument('--test_feats_dir', type=str,
                   default='data/le_features/internal_testing',
                   help='Directory of per-slide H5 LE feature files (internal test set).')
    p.add_argument('--ext_feats_dir', type=str,
                   default='data/le_features/external_testing',
                   help='Directory of per-slide H5 LE feature files (external test set).')
    p.add_argument('--dataset_csv', type=str, default=None,
                   help='Internal split CSV. If unset, taken from the run hyperparameters.')
    p.add_argument('--ext_test_csv', type=str,
                   default='data/external_dataset_split.csv',
                   help='External split CSV (must contain split=="ext_test" rows).')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--k_folds', type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class H5SlideDataset(Dataset):
    def __init__(self, df, feats_dir, num_patches=200, seed=24):
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fold_info(run_dir, k_folds):
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
        fold_info[fid] = {'ckpt': ckpts[0], 'threshold': thresholds.get(fid, 0.5)}
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
    fpr, tpr, thr = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    if bal_acc is None:
        yi = np.argmax(tpr - fpr)
        bal_acc = balanced_accuracy_score(labels, (np.array(probs) > thr[yi]).astype(int))
    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, color='#0400ff', lw=3, label=f'AUROC = {auc:.3f}')
    plt.plot([], [], ' ', label=f'Bal. Acc = {bal_acc:.3f}')
    plt.plot([0, 1], [0, 1], color='grey', lw=2.5, linestyle='--', label='Random')
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=26)
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


def save_results(slide_ids, labels, per_fold_probs, fold_ids, fold_thresholds, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    labels = np.array(labels)
    per_fold_probs = np.array(per_fold_probs)
    ensemble_probs = per_fold_probs.mean(axis=1)

    rows_05, rows_opt = [], []
    for col, fid in enumerate(fold_ids):
        fp = per_fold_probs[:, col]
        m05 = calculate_metrics(labels, fp, 0.5); m05.update({'fold': fid, 'threshold': 0.5})
        rows_05.append(m05)
        t = fold_thresholds.get(fid, 0.5)
        mopt = calculate_metrics(labels, fp, t); mopt.update({'fold': fid, 'threshold': t})
        rows_opt.append(mopt)
        plot_cm(labels, (fp > 0.5).astype(int),
                os.path.join(out_dir, f'{prefix}_fold_{fid}_confusion_matrix_thresh0.5.png'),
                f'Fold {fid} (Threshold=0.5)')
        plot_cm(labels, (fp > t).astype(int),
                os.path.join(out_dir, f'{prefix}_fold_{fid}_confusion_matrix_optimal.png'),
                f'Fold {fid} (Threshold={t:.4f})')

    def _reorder(df):
        cols = ['fold', 'threshold'] + [c for c in df.columns if c not in ('fold', 'threshold')]
        return df[cols]

    _reorder(pd.DataFrame(rows_05)).to_csv(
        os.path.join(out_dir, f'{prefix}_per_fold_performance_threshold0.5.csv'), index=False)
    _reorder(pd.DataFrame(rows_opt)).to_csv(
        os.path.join(out_dir, f'{prefix}_per_fold_performance_optimal_threshold.csv'), index=False)

    m05 = calculate_metrics(labels, ensemble_probs, 0.5)
    avg_thresh = float(np.mean(list(fold_thresholds.values()))) if fold_thresholds else 0.5
    preds_per_fold = [(per_fold_probs[:, i] > fold_thresholds.get(fid, 0.5)).astype(int)
                      for i, fid in enumerate(fold_ids)]
    vote_frac = np.mean(preds_per_fold, axis=0)
    m_hard = calculate_metrics(labels, vote_frac, 0.5)
    m_soft = calculate_metrics(labels, ensemble_probs, avg_thresh)

    with open(os.path.join(out_dir, f'{prefix}_ensemble_summary.json'), 'w') as f:
        json.dump({'average_optimal_threshold': avg_thresh,
                   'test_set_metrics': m05,
                   'test_set_metrics_optimal': m_hard,
                   'test_set_metrics_soft_avg': m_soft}, f, indent=4)

    per_slide = {str(sid): {'label': int(labels[i]),
                             'probs': per_fold_probs[i].tolist(),
                             'mean_prob': float(per_fold_probs[i].mean()),
                             'std_prob': float(per_fold_probs[i].std())}
                 for i, sid in enumerate(slide_ids)}
    with open(os.path.join(out_dir, f'{prefix}_with_sd.json'), 'w') as f:
        json.dump({'per_slide': per_slide}, f, indent=4)

    pd.DataFrame({'slide_id': slide_ids, 'label': labels.tolist(),
                  'ensemble_prob': ensemble_probs.tolist(),
                  'pred_05': (ensemble_probs > 0.5).astype(int).tolist(),
                  'pred_soft_avg': (ensemble_probs > avg_thresh).astype(int).tolist(),
                  'pred_hard_vote': (vote_frac > 0.5).astype(int).tolist()
                  }).to_csv(os.path.join(out_dir, f'{prefix}_ensemble_predictions.csv'), index=False)

    plot_roc(labels, ensemble_probs,
             os.path.join(out_dir, f'{prefix}_ensemble_roc_curve.png'),
             bal_acc=m_soft['Balanced Accuracy'])
    plot_roc(labels, vote_frac,
             os.path.join(out_dir, f'{prefix}_ensemble_roc_curve_optimal_threshold.png'),
             bal_acc=m_hard['Balanced Accuracy'])
    plot_cm(labels, (ensemble_probs > 0.5).astype(int),
            os.path.join(out_dir, f'{prefix}_ensemble_confusion_matrix_threshold0.5.png'),
            'Ensemble (Threshold=0.5)')
    plot_cm(labels, (vote_frac > 0.5).astype(int),
            os.path.join(out_dir, f'{prefix}_ensemble_confusion_matrix_optimal_threshold.png'),
            'Ensemble (Hard Voting)')
    plot_cm(labels, (ensemble_probs > avg_thresh).astype(int),
            os.path.join(out_dir, f'{prefix}_ensemble_confusion_matrix_soft_avg_threshold.png'),
            f'Ensemble (Soft Avg >{avg_thresh:.4f})')
    return m_soft


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(df, feats_dir, fold_info, model_fn, device, num_patches, seed):
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

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate, num_workers=4)

    fold_ids = sorted(fold_info.keys())
    models_list = []
    for fid in fold_ids:
        m = model_fn()
        m.load_state_dict(torch.load(fold_info[fid]['ckpt'], map_location=device, weights_only=False))
        m.to(device).eval()
        models_list.append(m)
        print(f'  Loaded fold {fid}: {os.path.basename(fold_info[fid]["ckpt"])}')

    all_slide_ids, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            if batch[0] is None:
                continue
            feats, mask, label, slide_ids = batch
            feats, mask = feats.to(device), mask.to(device)
            fold_probs = [torch.sigmoid(m(feats, mask)).cpu().item() for m in models_list]
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

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    hp_path = os.path.join(args.run_dir, 'fold_1', 'hyperparameters.json')
    hp = {}
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            hp = json.load(f)

    dropout = hp.get('dropout', 0.35)
    pooling = hp.get('pooling', 'mean')
    num_patches = hp.get('num_patches_sample', 200)
    seed = hp.get('seed', 24)
    dataset_csv = args.dataset_csv or hp.get('dataset_csv',
                                              'data/le_dataset_split.csv')
    test_feats_dir = args.test_feats_dir
    ext_feats_dir = args.ext_feats_dir
    ext_test_csv = hp.get('ext_test_csv', args.ext_test_csv)

    fold_info = load_fold_info(args.run_dir, args.k_folds)
    fold_ids = sorted(fold_info.keys())
    avg_thresh = np.mean([fold_info[fid]['threshold'] for fid in fold_ids])
    print(f'Folds: {fold_ids}')
    print(f'Avg threshold: {avg_thresh:.4f}')

    def model_fn():
        return PoolingClassifier(input_dim=1536, head='mlp', dropout=dropout, pooling=pooling)

    os.makedirs(args.output_dir, exist_ok=True)

    # Internal test
    full_df = pd.read_csv(dataset_csv)
    test_df = full_df[full_df['split'] == 'testing'].reset_index(drop=True)
    print(f'\nTest set: {len(test_df)} slides')
    slide_ids, labels, probs, f_ids, thresholds = run_inference(
        test_df, test_feats_dir, fold_info, model_fn, device, num_patches, seed)
    m = save_results(slide_ids, labels, probs, f_ids, thresholds, args.output_dir, 'testing')
    print(f'\n--- Internal Test Set ---')
    print(f'  [testing] AUC={m["AUC"]:.4f}  BalAcc={m["Balanced Accuracy"]:.4f}'
          f'  Sens={m["Sensitivity"]:.4f}  Spec={m["Specificity"]:.4f}')

    # External test
    ext_df = pd.read_csv(ext_test_csv)
    ext_df = ext_df[ext_df['split'] == 'ext_test'].reset_index(drop=True)
    print(f'\nExternal test set: {len(ext_df)} slides')
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
