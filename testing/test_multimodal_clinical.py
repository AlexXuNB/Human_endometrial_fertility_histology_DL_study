"""
Test script for Multimodal Clinical-Only model (Age + Endometrial Thickness only).

Loads a pre-trained sklearn Pipeline (StandardScaler + LogisticRegression)
saved as `model_clinical_only_logreg.pkl` and evaluates it on:
  - testing  : slides in LE internal test set (le_dataset_split.csv, split='testing')
  - ext_test : slides in matched external set
                (external_dataset_split.csv)

NO deep learning models are loaded; this is a pure clinical baseline.
Samples missing clinical data (AgeW_EB / EnTh) are skipped with a warning.

Feature ORDER must match training exactly: [AgeW_EB, EnTh].

Usage:
    python test_multimodal_clinical.py \\
        --logreg_dir             <dir_containing_model_clinical_only_logreg.pkl> \\
        --internal_clinical_csv  data/internal_clinical.csv \\
        --external_clinical_csv  data/external_clinical.csv \\
        --internal_dataset_csv   data/internal_dataset_split.csv \\
        --external_dataset_csv   data/external_dataset_split.csv \\
        --output_dir             <output_dir>
"""

import os
import json
import pickle
import argparse
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, balanced_accuracy_score, confusion_matrix,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description='Test clinical-only logistic regression.')
    p.add_argument('--logreg_dir', type=str, required=True,
                   help='Directory containing model_clinical_only_logreg.pkl.')
    p.add_argument('--internal_clinical_csv', type=str,
                   default='data/internal_clinical.csv')
    p.add_argument('--external_clinical_csv', type=str,
                   default='data/external_clinical.csv')
    p.add_argument('--internal_dataset_csv', type=str,
                   default='data/internal_dataset_split.csv',
                   help='Dataset CSV with sample_id, label, split=testing. '
                        'Use the same WSI split CSV that was used for clinical training.')
    p.add_argument('--external_dataset_csv', type=str,
                   default='data/external_dataset_split.csv',
                   help='External dataset CSV with sample_id, label, split=ext_test.')
    p.add_argument('--output_dir', type=str, default='./outputs/test_multimodal_clinical')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Clinical data loading
# ---------------------------------------------------------------------------

def load_clinical_data(csv_path):
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
    print(f'  Loaded {len(clinical)} samples.')
    return clinical


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


# ---------------------------------------------------------------------------
# Per-split inference
# ---------------------------------------------------------------------------

def run_split(clf, dataset_csv, split_value, clinical, out_dir, prefix):
    df = pd.read_csv(dataset_csv)
    df = df[df['split'] == split_value].reset_index(drop=True)
    print(f'\n[{prefix}] {len(df)} slides with split={split_value}')

    ids, labels, ages, enths = [], [], [], []
    skipped = []
    for _, row in df.iterrows():
        sid = str(row['sample_id'])
        cd = clinical.get(sid)
        if cd is None or np.isnan(cd['AgeW_EB']) or np.isnan(cd['EnTh']):
            skipped.append(sid)
            continue
        ids.append(sid)
        labels.append(int(row['label']))
        ages.append(cd['AgeW_EB'])
        enths.append(cd['EnTh'])
    if skipped:
        print(f'  Skipped {len(skipped)} (missing clinical): {skipped[:5]}')
    if not ids:
        print(f'  No usable samples for {prefix}')
        return None

    # IMPORTANT: column ORDER must match training: [AgeW_EB, EnTh].
    X = np.column_stack([ages, enths])
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    labels_arr = np.array(labels)
    correct = (preds == labels_arr).astype(int)

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({
        'sample_id': ids, 'label': labels,
        'AgeW_EB': ages, 'EnTh': enths,
        'prob': probs.tolist(), 'pred': preds.tolist(),
        'is_correct': correct.tolist(),
    }).to_csv(os.path.join(out_dir, f'{prefix}_predictions.csv'), index=False)

    m = calculate_metrics(labels_arr, probs, 0.5)
    with open(os.path.join(out_dir, f'{prefix}_metrics.json'), 'w') as f:
        json.dump(m, f, indent=4)
    plot_roc(labels_arr, probs, os.path.join(out_dir, f'{prefix}_roc.png'))
    plot_cm(labels_arr, preds, os.path.join(out_dir, f'{prefix}_cm.png'),
            f'Clinical-only ({prefix})')

    per_slide = {sid: {'label': int(labels_arr[i]),
                        'AgeW_EB': float(ages[i]),
                        'EnTh': float(enths[i]),
                        'prob': float(probs[i])}
                 for i, sid in enumerate(ids)}
    with open(os.path.join(out_dir, f'{prefix}_with_sd.json'), 'w') as f:
        json.dump({'per_slide': per_slide}, f, indent=4)

    print(f'  N={len(ids)} AUC={m["AUC"]:.4f}'
          f'  BalAcc={m["Balanced Accuracy"]:.4f}'
          f'  Sens={m["Sensitivity"]:.4f}  Spec={m["Specificity"]:.4f}')
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load Pipeline ----
    pkl_path = os.path.join(args.logreg_dir, 'model_clinical_only_logreg.pkl')
    try:
        clf = joblib.load(pkl_path)
    except Exception:
        with open(pkl_path, 'rb') as f:
            clf = pickle.load(f)
    print(f'Loaded clinical-only model from: {pkl_path}')
    print(f'  type={type(clf).__name__}')

    # Sanity: must be a sklearn Pipeline with StandardScaler + LogisticRegression.
    if isinstance(clf, Pipeline):
        steps = list(clf.named_steps.keys())
        print(f'  Pipeline steps: {steps}')
        if 'scaler' in clf.named_steps:
            print('    scaler present ✓')
        else:
            print('    WARNING: no scaler in Pipeline (training used StandardScaler).')
        clf_step = clf.named_steps.get('clf', None)
        if clf_step is not None and hasattr(clf_step, 'coef_'):
            print(f'    LR coef shape: {clf_step.coef_.shape}')
            if clf_step.coef_.shape[1] != 2:
                print(f'    WARNING: expected 2 features (AgeW_EB, EnTh), got {clf_step.coef_.shape[1]}')
    else:
        print('  WARNING: loaded model is not a sklearn Pipeline. '
              'Training script saves Pipeline(StandardScaler + LR).')

    # ---- Clinical data ----
    internal_clinical = load_clinical_data(args.internal_clinical_csv)
    external_clinical = load_clinical_data(args.external_clinical_csv)

    # ---- Internal test ----
    run_split(clf, args.internal_dataset_csv, 'testing',
              internal_clinical, args.output_dir, 'internal_test')

    # ---- External test ----
    run_split(clf, args.external_dataset_csv, 'ext_test',
              external_clinical, args.output_dir, 'external_test')

    print(f'\nAll outputs saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
