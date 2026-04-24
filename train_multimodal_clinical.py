"""
Training script for Clinical-Only Logistic Regression (Age + Endometrial Thickness).

This is a baseline model that uses only two clinical variables to predict
embryo implantation outcome:
  - AgeW_EB:  Patient age at embryo biopsy
  - EnTh:     Endometrial thickness (mm)

No deep learning models are used. The training labels are collected from
the fold data-split files of a pre-trained model run (any k-fold run that
has data_split_summary.csv per fold), used purely to identify which samples
were in the training set.

Algorithm:
  1. Collect training-set sample IDs and labels from the fold split files
     of a reference run (--wsi_run_dir is used for this purpose only).
  2. Join with clinical data -> 2-D feature: (AgeW_EB, EnTh).
  3. Train  Pipeline([StandardScaler, LogisticRegression]) on these 2 features.
  4. Evaluate on internal test set and external test set.

Outputs saved to  <output_dir>/<run_name>/:
  model_clinical_only_logreg.pkl  - sklearn Pipeline (joblib-serialized)
  oof_clinical_training_data.csv  - training features + predictions
  oof_clinical_metrics.json       - training metrics
  internal_test_{roc,cm,metrics,predictions}.*  - internal test results
  external_test_{roc,cm,metrics,predictions}.*  - external test results

Usage:
  python train_multimodal_clinical.py \\
    --wsi_run_dir outputs/wsi_pooling/run_YYYYMMDD_HHMMSS \\
    --internal_clinical_csv data/internal_clinical.csv \\
    --external_clinical_csv data/external_clinical.csv \\
    --output_dir  outputs/multimodal_clinical \\
    --k_folds 10
"""

import argparse
import datetime
import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
# Label collection from fold split files
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


def collect_training_labels(run_dir, k_folds):
    """Gather sample_id -> label for all training-set samples.

    Each fold's validation split represents the held-out portion of the
    training set (i.e. samples that were training samples but not used for
    that fold's training). Together, all folds cover every training sample.
    """
    labels = {}
    for i in range(k_folds):
        fold_id = i + 1
        fold_dir = os.path.join(run_dir, f'fold_{fold_id}')
        if not os.path.exists(fold_dir):
            continue
        val_df = load_validation_split(fold_dir)
        if val_df is None or len(val_df) == 0:
            continue
        if 'sample_id' not in val_df.columns or 'label' not in val_df.columns:
            print(f'  Warning: missing sample_id/label columns in {fold_dir}')
            continue
        for _, row in val_df.iterrows():
            sid = str(row['sample_id'])
            lbl = int(row['label'])
            if sid in labels and labels[sid] != lbl:
                warnings.warn(f'Label mismatch for {sid}')
            labels[sid] = lbl

    print(f'Collected training labels for {len(labels)} samples across all folds.')
    return labels


# ---------------------------------------------------------------------------
# Feature array preparation
# ---------------------------------------------------------------------------

def prepare_clinical_arrays(labels, clinical_data):
    """Build X, y arrays from label dict and clinical dict."""
    rows = []
    for sid, y in labels.items():
        if sid not in clinical_data:
            continue
        c = clinical_data[sid]
        if np.isnan(c['AgeW_EB']) or np.isnan(c['EnTh']):
            continue
        rows.append({'sample_id': sid, 'label': y,
                     'AgeW_EB': c['AgeW_EB'], 'EnTh': c['EnTh']})
    if not rows:
        return None, None, None
    df = pd.DataFrame(rows)
    X = df[['AgeW_EB', 'EnTh']].values
    y = df['label'].values.astype(int)
    return X, y, df


def prepare_test_arrays(dataset_csv, split_name, clinical_data):
    """Build test arrays from a split in a dataset CSV."""
    df = pd.read_csv(dataset_csv)
    actual = split_name
    if split_name not in df['split'].unique() and split_name == 'test':
        actual = 'testing'
    if split_name not in df['split'].unique() and split_name == 'testing':
        actual = 'test'
    split_df = df[df['split'] == actual].reset_index(drop=True)
    if len(split_df) == 0:
        print(f'  No samples for split "{split_name}" in {dataset_csv}')
        return None, None, None
    labels = {str(r['sample_id']): int(r['label']) for _, r in split_df.iterrows()}
    return prepare_clinical_arrays(labels, clinical_data)


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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description='Clinical-Only Logistic Regression (Age + EnTh baseline).')
    p.add_argument('--wsi_run_dir', type=str, required=True,
                   help='Directory of any completed k-fold run. Used only to '
                        'read data_split_summary.csv files for training labels.')
    p.add_argument('--output_dir', type=str, default='outputs/multimodal_clinical')
    p.add_argument('--run_name', type=str, default=None)

    # Clinical data CSVs (columns: Sample_no, AgeW_EB, EnTh)
    p.add_argument('--internal_clinical_csv', type=str,
                   default='data/internal_clinical.csv',
                   help='Clinical CSV for training + internal test samples.')
    p.add_argument('--external_clinical_csv', type=str,
                   default='data/external_clinical.csv',
                   help='Clinical CSV for external test samples.')

    # Dataset CSVs (for test set evaluation)
    p.add_argument('--wsi_dataset_csv', type=str, default='data/internal_dataset_split.csv',
                   help='Dataset CSV with sample_id, label, split columns.')
    p.add_argument('--wsi_ext_dataset_csv', type=str,
                   default='data/external_dataset_split.csv')

    p.add_argument('--k_folds', type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    if args.run_name is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f'clinical_only_{ts}'

    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    internal_clinical = load_clinical_data(args.internal_clinical_csv)
    external_clinical = load_clinical_data(args.external_clinical_csv)

    # ------------------------------------------------------------------
    # Step 1: Collect training labels from fold splits
    # ------------------------------------------------------------------
    print('\n' + '=' * 80)
    print('STEP 1: Collect Training Labels from Fold Splits')
    print('=' * 80)

    train_labels = collect_training_labels(args.wsi_run_dir, args.k_folds)
    if not train_labels:
        raise RuntimeError(
            f'No training labels found in {args.wsi_run_dir}. '
            'Check that data_split_summary.csv files exist per fold.')

    X_train, y_train, train_df = prepare_clinical_arrays(train_labels, internal_clinical)
    if X_train is None:
        raise RuntimeError('Could not build training arrays. '
                           'Check that clinical data covers the training samples.')

    print(f'Training samples after clinical filtering: {len(X_train)}')

    # ------------------------------------------------------------------
    # Step 2: Train logistic regression
    # ------------------------------------------------------------------
    print('\n' + '=' * 80)
    print('STEP 2: Train Logistic Regression (AgeW_EB + EnTh)')
    print('=' * 80)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, solver='liblinear', max_iter=1000,
                                   class_weight='balanced')),
    ])
    clf.fit(X_train, y_train)

    feat_names = ['AgeW_EB', 'EnTh']
    coefs = clf.named_steps['clf'].coef_[0]
    intercept = clf.named_steps['clf'].intercept_[0]
    print('\nLogReg coefficients:')
    for name, w in zip(feat_names, coefs):
        print(f'  {name}: {w:.4f}')
    print(f'  intercept: {intercept:.4f}')

    y_train_prob = clf.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_prob > 0.5).astype(int)
    train_metrics = calculate_metrics(y_train, y_train_prob, y_train_pred)
    print('\nTraining metrics:')
    for k, v in train_metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    train_df['prob'] = y_train_prob
    train_df['pred'] = y_train_pred
    train_df.to_csv(os.path.join(output_dir, 'oof_clinical_training_data.csv'), index=False)
    with open(os.path.join(output_dir, 'oof_clinical_metrics.json'), 'w') as f:
        json.dump(train_metrics, f, indent=4)

    joblib.dump(clf, os.path.join(output_dir, 'model_clinical_only_logreg.pkl'))

    # ------------------------------------------------------------------
    # Step 3: Internal test evaluation
    # ------------------------------------------------------------------
    print('\n' + '=' * 80)
    print('STEP 3: Internal Test Evaluation')
    print('=' * 80)

    X_test, y_test, test_df = prepare_test_arrays(
        args.wsi_dataset_csv, 'testing', internal_clinical)
    if X_test is None:
        # Try 'test' split label
        X_test, y_test, test_df = prepare_test_arrays(
            args.wsi_dataset_csv, 'test', internal_clinical)

    if X_test is not None:
        y_test_prob = clf.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob > 0.5).astype(int)
        test_metrics = calculate_metrics(y_test, y_test_prob, y_test_pred)
        print('Internal Test Metrics:')
        for k, v in test_metrics.items():
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
        test_df['prob'] = y_test_prob
        test_df['pred'] = y_test_pred
        test_df.to_csv(os.path.join(output_dir, 'internal_test_predictions.csv'), index=False)
        plot_roc(y_test, y_test_prob,
                 os.path.join(output_dir, 'internal_test_roc.png'),
                 bal_acc=test_metrics['Balanced Accuracy'])
        plot_cm(y_test, y_test_pred, os.path.join(output_dir, 'internal_test_cm.png'))
        with open(os.path.join(output_dir, 'internal_test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)
    else:
        print('  No internal test samples available after clinical filtering.')

    # ------------------------------------------------------------------
    # Step 4: External test evaluation
    # ------------------------------------------------------------------
    print('\n' + '=' * 80)
    print('STEP 4: External Test Evaluation')
    print('=' * 80)

    X_ext, y_ext, ext_df = prepare_test_arrays(
        args.wsi_ext_dataset_csv, 'ext_test', external_clinical)

    if X_ext is not None:
        y_ext_prob = clf.predict_proba(X_ext)[:, 1]
        y_ext_pred = (y_ext_prob > 0.5).astype(int)
        ext_metrics = calculate_metrics(y_ext, y_ext_prob, y_ext_pred)
        print('External Test Metrics:')
        for k, v in ext_metrics.items():
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
        ext_df['prob'] = y_ext_prob
        ext_df['pred'] = y_ext_pred
        ext_df.to_csv(os.path.join(output_dir, 'external_test_predictions.csv'), index=False)
        plot_roc(y_ext, y_ext_prob,
                 os.path.join(output_dir, 'external_test_roc.png'),
                 bal_acc=ext_metrics['Balanced Accuracy'])
        plot_cm(y_ext, y_ext_pred, os.path.join(output_dir, 'external_test_cm.png'))
        with open(os.path.join(output_dir, 'external_test_metrics.json'), 'w') as f:
            json.dump(ext_metrics, f, indent=4)
    else:
        print('  No external test samples available after clinical filtering.')

    print('\n' + '=' * 80)
    print('CLINICAL-ONLY LOGREG COMPLETE')
    print('=' * 80)
    print(f'\nResults saved to: {output_dir}')


if __name__ == '__main__':
    main()
