"""
Training script for Multimodal 4-Factor OOF Logistic Regression
(WSI + LE + Age + Endometrial Thickness).

Extends the 2-factor model by adding two clinical variables:
  - AgeW_EB:  Patient age at embryo biopsy
  - EnTh:     Endometrial thickness (mm)

The clinical features are loaded from a CSV with columns:
  Sample_no, AgeW_EB, EnTh

Algorithm (Out-Of-Fold logistic regression):
  1. Build WSI and LE OOF predictions (same as 2-factor).
  2. Join with clinical data -> 4-D feature: (wsi_prob, le_prob, AgeW_EB, EnTh).
  4. Train  LogisticRegression(solver='liblinear', class_weight='balanced') on these 4 features.
  4. Evaluate on internal test set and external test set.

Outputs saved to  <output_dir>/<run_name>/:
  logreg_model.pkl              - sklearn Pipeline (joblib-serialized)
  oof_logreg_training_data.csv  - OOF features + predictions
  oof_logreg_metrics.json       - OOF metrics + model coefficients
  testing_ensemble_*.{csv,png,json}   - internal test results
  ext_test_ensemble_*.{csv,png,json}  - external test results

Usage:
  python train_multimodal_4factor.py \\
    --wsi_run_dir outputs/wsi_pooling/run_YYYYMMDD_HHMMSS \\
    --le_run_dir  outputs/le_pooling/run_YYYYMMDD_HHMMSS \\
    --internal_clinical_csv data/internal_clinical.csv \\
    --external_clinical_csv data/external_clinical.csv \\
    --output_dir  outputs/multimodal_4factor \\
    --gpu_id 0
"""

import argparse
import datetime
import json
import os
import re
import warnings

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                              roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, Dataset

from models import PoolingClassifier


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class H5Dataset(Dataset):
    """Load pre-extracted H5 patch features for a list of samples."""

    def __init__(self, df, feats_dir, num_patches=2000, seed=42):
        self.df = df.reset_index(drop=True)
        self.feats_dir = feats_dir
        self.num_patches = num_patches
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row['sample_id'])
        label = torch.tensor(row['label'], dtype=torch.float32)
        h5_path = os.path.join(self.feats_dir, sid + '.h5')
        try:
            with h5py.File(h5_path, 'r') as f:
                features = torch.from_numpy(f['features'][:]).float()
        except Exception as e:
            print(f'  Warning: could not load {h5_path}: {e}')
            return None
        n = features.shape[0]
        if n > self.num_patches:
            indices = torch.linspace(0, n - 1, self.num_patches).long()
            features = features[indices]
        return features, label, sid


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def build_dataloader(df, feats_dir, num_patches, seed, num_workers):
    dataset = H5Dataset(df, feats_dir, num_patches, seed)

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None, None, None, None
        features_list, labels_list, ids_list = zip(*batch)
        lengths = [f.shape[0] for f in features_list]
        max_len = max(lengths)
        feat_dim = features_list[0].shape[1]
        padded = torch.zeros(len(batch), max_len, feat_dim)
        mask = torch.zeros(len(batch), max_len)
        for i, f in enumerate(features_list):
            end = lengths[i]
            padded[i, :end] = f
            mask[i, :end] = 1.0
        return padded, mask, torch.tensor(labels_list), list(ids_list)

    return DataLoader(dataset, batch_size=1, shuffle=False,
                      collate_fn=collate_fn, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Clinical data loading
# ---------------------------------------------------------------------------

def load_clinical_data(csv_path):
    """Load AgeW_EB and EnTh from a CSV with Sample_no, AgeW_EB, EnTh columns.

    Returns {str(sample_id): {'AgeW_EB': float, 'EnTh': float}}
    Entries with non-numeric or missing values are set to NaN.
    """
    print(f'Loading clinical data from: {csv_path}')
    df = pd.read_csv(csv_path)
    required = ['Sample_no', 'AgeW_EB', 'EnTh']
    for col in required:
        if col not in df.columns:
            raise ValueError(f'Missing required column "{col}" in {csv_path}')

    clinical = {}
    for _, row in df.iterrows():
        if pd.isna(row['Sample_no']):
            continue
        try:
            sid = str(int(row['Sample_no']))
        except (ValueError, TypeError):
            sid = str(row['Sample_no']).strip()
        try:
            age = float(row['AgeW_EB'])
        except (ValueError, TypeError):
            age = float('nan')
        try:
            enth = float(str(row['EnTh']).replace(',', '.'))
        except (ValueError, TypeError):
            enth = float('nan')
        clinical[sid] = {'AgeW_EB': age, 'EnTh': enth}

    print(f'Loaded clinical data for {len(clinical)} samples.')
    return clinical


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_hyperparameters(run_dir):
    hp_path = os.path.join(run_dir, 'fold_1', 'hyperparameters.json')
    if not os.path.exists(hp_path):
        return {}
    with open(hp_path) as f:
        return json.load(f)


def _find_best_model(fold_dir):
    files = [f for f in os.listdir(fold_dir)
             if f.startswith('best_model') and f.endswith(('.pt', '.pth'))]
    if not files:
        return None
    scored = []
    for fname in files:
        m = re.search(r'epoch(\d+)', fname)
        if m:
            scored.append((int(m.group(1)), fname))
    if scored:
        scored.sort(reverse=True)
        return os.path.join(fold_dir, scored[0][1])
    files.sort()
    return os.path.join(fold_dir, files[0])


def load_pooling_model(fold_dir, hp, device):
    model_path = _find_best_model(fold_dir)
    if model_path is None:
        return None
    model = PoolingClassifier(
        input_dim=int(hp.get('input_dim', 1536)),
        head='mlp',
        dropout=float(hp.get('dropout', 0.4)),
        pooling=hp.get('pooling', 'mean'),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_probs(model, loader, device):
    probs, labels = {}, {}
    for features, mask, y, sids in loader:
        if features is None:
            continue
        features, mask = features.to(device), mask.to(device)
        logits = model(features, mask)
        prob = torch.sigmoid(logits).cpu().numpy()
        for i, sid in enumerate(sids):
            probs[str(sid)] = float(prob[i])
            labels[str(sid)] = int(y[i].item())
    return probs, labels


# ---------------------------------------------------------------------------
# OOF predictions
# ---------------------------------------------------------------------------

def load_validation_split(fold_dir):
    path = os.path.join(fold_dir, 'data_split_summary.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for col in ('split_type', 'split'):
        if col in df.columns:
            return df[df[col].isin(['validation', 'val'])].reset_index(drop=True)
    return None


def build_oof_predictions(run_dir, hp, feats_dir, k_folds, num_workers, device):
    oof_probs, oof_labels = {}, {}
    num_patches = int(hp.get('num_patches_sample', 2000))
    seed = int(hp.get('seed', 42))

    for i in range(k_folds):
        fold_id = i + 1
        fold_dir = os.path.join(run_dir, f'fold_{fold_id}')
        if not os.path.exists(fold_dir):
            continue
        val_df = load_validation_split(fold_dir)
        if val_df is None or len(val_df) == 0:
            continue
        model = load_pooling_model(fold_dir, hp, device)
        if model is None:
            continue
        loader = build_dataloader(val_df, feats_dir, num_patches, seed, num_workers)
        fold_probs, fold_labels = predict_probs(model, loader, device)
        print(f'  Fold {fold_id}: {len(fold_probs)} OOF samples')
        for sid, p in fold_probs.items():
            oof_probs[sid] = p
            oof_labels[sid] = fold_labels[sid]

    return oof_probs, oof_labels


# ---------------------------------------------------------------------------
# Ensemble (mean across all folds) on test / ext_test
# ---------------------------------------------------------------------------

def ensemble_predict(run_dir, hp, dataset_csv, split_name, feats_dir,
                     k_folds, num_workers, device):
    df = pd.read_csv(dataset_csv)
    actual = split_name
    if split_name not in df['split'].unique() and split_name == 'testing':
        actual = 'test'
    if split_name not in df['split'].unique() and split_name == 'test':
        actual = 'testing'
    split_df = df[df['split'] == actual].reset_index(drop=True)
    if len(split_df) == 0:
        print(f'  No samples found for split "{split_name}" in {dataset_csv}')
        return {}, {}

    num_patches = int(hp.get('num_patches_sample', 2000))
    seed = int(hp.get('seed', 42))
    loader = build_dataloader(split_df, feats_dir, num_patches, seed, num_workers)
    all_fold_probs = []
    labels = {str(r['sample_id']): int(r['label']) for _, r in split_df.iterrows()}

    for i in range(k_folds):
        fold_id = i + 1
        fold_dir = os.path.join(run_dir, f'fold_{fold_id}')
        if not os.path.exists(fold_dir):
            continue
        model = load_pooling_model(fold_dir, hp, device)
        if model is None:
            continue
        fold_probs, _ = predict_probs(model, loader, device)
        if fold_probs:
            all_fold_probs.append(fold_probs)

    if not all_fold_probs:
        return {}, labels

    sids = list(all_fold_probs[0].keys())
    for d in all_fold_probs[1:]:
        sids = [s for s in sids if s in d]

    ensembled = {sid: float(np.mean([d[sid] for d in all_fold_probs])) for sid in sids}
    return ensembled, labels


# ---------------------------------------------------------------------------
# Logistic regression training (4 features)
# ---------------------------------------------------------------------------

def find_best_threshold(y_true, y_score):
    """Sweep thresholds to find the one maximising balanced accuracy."""
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    best_thresh, best_bal = 0.5, -1.0
    for thresh in np.linspace(0.0, 1.0, 1001):
        y_pred = (y_score > thresh).astype(int)
        bal = balanced_accuracy_score(y_true, y_pred)
        if bal > best_bal:
            best_bal, best_thresh = bal, float(thresh)
    return best_thresh, best_bal


def train_logreg(oof_wsi_probs, oof_le_probs, labels, clinical_data):
    rows, skipped = [], 0
    for sid, y in labels.items():
        if sid not in oof_wsi_probs or sid not in oof_le_probs:
            continue
        if sid not in clinical_data:
            skipped += 1
            continue
        c = clinical_data[sid]
        if np.isnan(c['AgeW_EB']) or np.isnan(c['EnTh']):
            skipped += 1
            continue
        rows.append({'sample_id': sid, 'label': y,
                     'wsi_prob': oof_wsi_probs[sid], 'le_prob': oof_le_probs[sid],
                     'AgeW_EB': c['AgeW_EB'], 'EnTh': c['EnTh']})

    if skipped:
        print(f'Excluded {skipped} OOF samples due to missing clinical data.')
    if not rows:
        raise RuntimeError('No valid samples with complete (WSI+LE+Clinical) OOF data.')

    df = pd.DataFrame(rows)
    X = df[['wsi_prob', 'le_prob', 'AgeW_EB', 'EnTh']].values
    y = df['label'].values

    # LogisticRegression with class_weight='balanced' for consistency with the
    # 2-factor and clinical-only multimodal models. Empirically a no-op on this
    # near-balanced OOF set (~47% positive); kept for code-level consistency.
    model = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced')
    model.fit(X, y)

    feat_names = ['wsi_prob', 'le_prob', 'AgeW_EB', 'EnTh']
    coefs = model.coef_[0]
    intercept = model.intercept_[0]
    print('\n' + '=' * 40)
    print('LOGREG MODEL WEIGHTS (WSI + LE + Clinical)')
    print('=' * 40)
    for name, w in zip(feat_names, coefs):
        print(f'  {name}: {w:.4f}')
    print(f'  intercept: {intercept:.4f}')
    print('=' * 40 + '\n')

    return model, df, feat_names, coefs, intercept


# ---------------------------------------------------------------------------
# Metrics + plots
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_score, y_pred):
    metrics = {}
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['PPV'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        metrics['F1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    except ValueError:
        metrics['Accuracy'] = float((np.array(y_pred) == np.array(y_true)).mean())
        for k in ('Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1'):
            metrics[k] = 'N/A'
    metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['AUC'] = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    return metrics


def plot_roc(y_true, y_score, path, bal_acc=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    if bal_acc is None:
        bal_acc = balanced_accuracy_score(y_true,
                                          (np.array(y_score) >= 0.5).astype(int))
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
    plt.savefig(path, dpi=300)
    plt.close()


def plot_cm(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={'size': 28})
    plt.xlabel('Predicted Label', fontsize=24)
    plt.ylabel('True Label', fontsize=24)
    plt.xticks(fontsize=20); plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_results(output_dir, split_name, y_true, y_score, y_pred, sample_ids):
    metrics = calculate_metrics(y_true, y_score, y_pred)
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    plot_roc(y_true, y_score,
             os.path.join(output_dir, f'{split_name}_ensemble_roc_curve.png'),
             bal_acc=metrics['Balanced Accuracy'])
    plot_cm(y_true, y_pred,
            os.path.join(output_dir, f'{split_name}_ensemble_confusion_matrix_threshold0.5.png'))

    pd.DataFrame({
        'slide_id': sample_ids,
        'ground_truth_label': y_true,
        'ensembled_probability': y_score,
        'predicted_label_threshold0.5': y_pred,
        'is_correct': (y_true == y_pred).astype(int),
    }).to_csv(os.path.join(output_dir, f'{split_name}_ensemble_predictions.csv'), index=False)

    with open(os.path.join(output_dir, f'{split_name}_ensemble_metrics_threshold0.5.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f'{k}: {v:.4f}\n' if isinstance(v, float) else f'{k}: {v}\n')

    with open(os.path.join(output_dir, f'{split_name}_ensemble_summary.json'), 'w') as f:
        json.dump({'test_set_metrics': metrics}, f, indent=4)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description='Multimodal 4-Factor OOF Logistic Regression (WSI + LE + Age + EnTh).')
    p.add_argument('--wsi_run_dir', type=str, required=True,
                   help='Directory of a completed WSI PoolingClassifier k-fold run.')
    p.add_argument('--le_run_dir', type=str, required=True,
                   help='Directory of a completed LE PoolingClassifier k-fold run.')
    p.add_argument('--output_dir', type=str, default='outputs/multimodal_4factor')
    p.add_argument('--run_name', type=str, default=None)

    # Clinical data CSVs (columns: Sample_no, AgeW_EB, EnTh)
    p.add_argument('--internal_clinical_csv', type=str,
                   default='data/internal_clinical.csv',
                   help='Clinical CSV for training + internal test samples.')
    p.add_argument('--external_clinical_csv', type=str,
                   default='data/external_clinical.csv',
                   help='Clinical CSV for external test samples.')

    # Feature directories
    p.add_argument('--wsi_train_feats_dir', type=str, default=None)
    p.add_argument('--le_train_feats_dir', type=str, default=None)
    p.add_argument('--wsi_test_feats_dir', type=str,
                   default='data/wsi_features/internal_testing')
    p.add_argument('--wsi_ext_test_feats_dir', type=str,
                   default='data/wsi_features/external_testing')
    p.add_argument('--le_test_feats_dir', type=str,
                   default='data/le_features/internal_testing')
    p.add_argument('--le_ext_test_feats_dir', type=str,
                   default='data/le_features/external_testing')

    # Dataset CSVs
    p.add_argument('--wsi_dataset_csv', type=str, default='data/internal_dataset_split.csv')
    p.add_argument('--wsi_ext_dataset_csv', type=str,
                   default='data/external_dataset_split.csv')
    p.add_argument('--le_dataset_csv', type=str, default='data/le_dataset_split.csv')
    p.add_argument('--le_ext_dataset_csv', type=str,
                   default='data/external_dataset_split.csv')

    p.add_argument('--k_folds', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpu_id', type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    if args.run_name is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f'oof_logreg_clinical_{ts}'

    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f'\nUsing GPU: {args.gpu_id}')
    else:
        device = torch.device('cpu')
        print('\nUsing CPU')

    wsi_hp = load_hyperparameters(args.wsi_run_dir)
    le_hp = load_hyperparameters(args.le_run_dir)

    wsi_train_feats = args.wsi_train_feats_dir or wsi_hp.get('train_feats_dir')
    le_train_feats = args.le_train_feats_dir or le_hp.get('train_feats_dir')
    if not wsi_train_feats or not le_train_feats:
        raise RuntimeError(
            'Training feature directories are required for OOF predictions. '
            'Pass --wsi_train_feats_dir / --le_train_feats_dir, or ensure '
            'train_feats_dir is stored in fold_1/hyperparameters.json.')

    internal_clinical = load_clinical_data(args.internal_clinical_csv)
    external_clinical = load_clinical_data(args.external_clinical_csv)

    # ------------------------------------------------------------------
    # Part 1: OOF predictions + logistic regression training
    # ------------------------------------------------------------------
    print('\n' + '=' * 80)
    print('PART 1: OOF PREDICTIONS FOR LOGISTIC REGRESSION (WSI + LE + Clinical)')
    print('=' * 80)

    print('\nBuilding WSI OOF predictions...')
    wsi_oof_probs, wsi_oof_labels = build_oof_predictions(
        args.wsi_run_dir, wsi_hp, wsi_train_feats,
        args.k_folds, args.num_workers, device)

    print('\nBuilding LE OOF predictions...')
    le_oof_probs, le_oof_labels = build_oof_predictions(
        args.le_run_dir, le_hp, le_train_feats,
        args.k_folds, args.num_workers, device)

    shared_ids = set(wsi_oof_probs) & set(le_oof_probs)
    if not shared_ids:
        raise RuntimeError('No overlapping samples between WSI and LE OOF predictions.')

    labels = {}
    for sid in shared_ids:
        w, l = wsi_oof_labels.get(sid), le_oof_labels.get(sid)
        if w is not None and l is not None and w != l:
            warnings.warn(f'Label mismatch for {sid}: WSI={w}, LE={l}')
        labels[sid] = w if w is not None else l

    logreg, oof_df, feat_names, coefs, intercept = train_logreg(
        wsi_oof_probs, le_oof_probs, labels, internal_clinical)

    X_train = oof_df[['wsi_prob', 'le_prob', 'AgeW_EB', 'EnTh']].values
    oof_df['combined_prob'] = logreg.predict_proba(X_train)[:, 1]
    oof_df['combined_pred'] = (oof_df['combined_prob'] > 0.5).astype(int)
    oof_df.to_csv(os.path.join(output_dir, 'oof_logreg_training_data.csv'), index=False)

    oof_metrics = calculate_metrics(
        oof_df['label'].values, oof_df['combined_prob'].values,
        oof_df['combined_pred'].values)
    # Find optimal threshold by sweeping (matching original training behaviour)
    best_thresh, best_bal = find_best_threshold(
        oof_df['label'].values, oof_df['combined_prob'].values)
    oof_metrics['optimal_threshold'] = best_thresh
    oof_metrics['optimal_threshold_balanced_accuracy'] = best_bal
    oof_metrics['coefficients'] = {
        name: float(w) for name, w in zip(feat_names, coefs)}
    oof_metrics['coefficients']['bias'] = float(intercept)
    with open(os.path.join(output_dir, 'oof_logreg_metrics.json'), 'w') as f:
        json.dump(oof_metrics, f, indent=4)

    joblib.dump(logreg, os.path.join(output_dir, 'logreg_model.pkl'))
    print(f'\nSaved logreg model and OOF data to: {output_dir}')

    # ------------------------------------------------------------------
    # Part 2: Evaluate on test / ext_test sets
    # ------------------------------------------------------------------
    test_splits = [
        ('testing', args.wsi_dataset_csv, args.wsi_test_feats_dir,
         args.le_dataset_csv, args.le_test_feats_dir, internal_clinical),
        ('ext_test', args.wsi_ext_dataset_csv, args.wsi_ext_test_feats_dir,
         args.le_ext_dataset_csv, args.le_ext_test_feats_dir, external_clinical),
    ]

    for split_name, wsi_csv, wsi_feats, le_csv, le_feats, clinical_dict in test_splits:
        print('\n' + '=' * 80)
        print(f'PART 2: ENSEMBLE PERFORMANCE ({split_name})')
        print('=' * 80)

        wsi_probs, wsi_labels = ensemble_predict(
            args.wsi_run_dir, wsi_hp, wsi_csv, split_name, wsi_feats,
            args.k_folds, args.num_workers, device)
        le_probs, le_labels = ensemble_predict(
            args.le_run_dir, le_hp, le_csv, split_name, le_feats,
            args.k_folds, args.num_workers, device)

        shared = sorted(set(wsi_probs) & set(le_probs))
        if not shared:
            print(f'  No overlapping samples for {split_name}. Skipping.')
            continue

        rows, excluded = [], 0
        for sid in shared:
            label = wsi_labels.get(sid, le_labels.get(sid))
            if label is None:
                continue
            if sid not in clinical_dict:
                excluded += 1
                continue
            c = clinical_dict[sid]
            if np.isnan(c['AgeW_EB']) or np.isnan(c['EnTh']):
                excluded += 1
                continue
            rows.append({'sample_id': sid, 'label': label,
                         'wsi_prob': wsi_probs[sid], 'le_prob': le_probs[sid],
                         'AgeW_EB': c['AgeW_EB'], 'EnTh': c['EnTh']})
        if excluded:
            print(f'  Excluded {excluded} samples due to missing clinical data.')
        if not rows:
            print(f'  No valid samples for {split_name} after clinical filtering.')
            continue

        test_df = pd.DataFrame(rows)
        X = test_df[['wsi_prob', 'le_prob', 'AgeW_EB', 'EnTh']].values
        combined_prob = logreg.predict_proba(X)[:, 1]
        combined_pred = (combined_prob > 0.5).astype(int)

        save_results(output_dir, split_name,
                     test_df['label'].values, combined_prob, combined_pred,
                     test_df['sample_id'].values)
        test_df['combined_prob'] = combined_prob
        test_df.to_csv(
            os.path.join(output_dir, f'{split_name}_ensemble_inputs.csv'), index=False)

    print('\n' + '=' * 80)
    print('MULTIMODAL 4-FACTOR OOF LOGREG COMPLETE')
    print('=' * 80)
    print(f'\nResults saved to: {output_dir}')


if __name__ == '__main__':
    main()
