"""
Training script for WSI-level UNI2-h ACMIL classification with k-fold CV.

Model variants (selected via --head):
  - linear: ACMIL attention -> Linear(768, 1) slide classifier
  - mlp:    ACMIL attention -> MLP (768 -> 768 -> 384 -> 1) slide classifier

Architecture (ACMIL with AEM regularization):
  1. DimReduction: Linear(1536 -> 768, no bias) + ReLU
  2. Gated Attention: K heads with Tanh/Sigmoid gating
  3. Per-token classifiers: K x Linear(768, 1)
  4. Bag-level classifier: Linear or MLP

Input: Pre-extracted UNI2-h patch features (1536-d) in H5 files.
Loss: BCE (sub-classifiers + bag) + attention diversity + AEM entropy
Model selection: Validation balanced accuracy (no test set leakage)
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, roc_curve, balanced_accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from models import ACMILClassifier


# ---- Learning Rate Scheduler ----

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            alpha = self.current_epoch / self.warmup_epochs
            lr = self.min_lr + (self.base_lr - self.min_lr) * alpha
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]


# ---- Dataset ----

class H5SlideDataset(Dataset):
    """Loads pre-extracted patch features from H5 files (one slide per sample)."""

    def __init__(self, df, feats_dir, num_patches=1700, split='train', seed=42):
        self.df = df
        self.feats_dir = feats_dir
        self.num_patches = num_patches
        self.split = split
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
            print(f'Error loading {h5_path}: {e}')
            return None
        n = features.shape[0]
        if n > self.num_patches:
            if self.split == 'train':
                indices = torch.randperm(n)[:self.num_patches]
            else:
                indices = torch.linspace(0, n - 1, self.num_patches).long()
            features = features[indices]
        return features, label, sid


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, None
    features_list = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch])
    slide_ids = [b[2] for b in batch]
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    feat_dim = features_list[0].shape[1]
    padded = torch.zeros(len(batch), max_len, feat_dim)
    mask = torch.zeros(len(batch), max_len)
    for i, f in enumerate(features_list):
        end = lengths[i]
        padded[i, :end] = f
        mask[i, :end] = 1.0
    return padded, mask, labels, slide_ids


# ---- Metrics ----

def find_optimal_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return thresholds[np.argmax(tpr - fpr)]


# ---- ACMIL Loss Components ----

def compute_acmil_loss(sub_preds, slide_pred, attn_logits, label,
                       criterion, lamda_aem=0.1):
    """Compute ACMIL total loss.

    Components:
        1. Sub-classifier loss: BCE on each per-token prediction
        2. Bag-level loss: BCE on slide prediction
        3. Diversity loss: Cosine similarity between attention heads
        4. AEM loss: Attention entropy maximization
    """
    # Per-token (sub-classifier) loss
    loss_sub = sum(criterion(p.unsqueeze(0), label.unsqueeze(0))
                   for p in sub_preds) / len(sub_preds)

    # Bag-level loss
    loss_bag = criterion(slide_pred, label.unsqueeze(0))

    # Diversity loss: penalize similar attention distributions
    K = attn_logits.shape[1]
    if K > 1:
        attn_softmax = F.softmax(attn_logits.squeeze(0), dim=1)  # (K, N)
        # Pairwise cosine similarity between attention heads
        norm_attn = F.normalize(attn_softmax, p=2, dim=1)
        sim_matrix = torch.mm(norm_attn, norm_attn.T)
        # Average off-diagonal similarity
        mask_diag = ~torch.eye(K, dtype=torch.bool, device=sim_matrix.device)
        diversity_loss = sim_matrix[mask_diag].mean()
    else:
        diversity_loss = torch.tensor(0.0, device=slide_pred.device)

    # AEM loss: maximize attention entropy (encourage uniform attention)
    attn_dist = F.softmax(attn_logits, dim=2)  # (1, K, N)
    aem_loss = torch.sum(attn_dist * F.log_softmax(attn_logits, dim=2))

    total = loss_sub + loss_bag + diversity_loss + lamda_aem * aem_loss
    return total


# ---- Training ----

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs, all_ids = [], [], []
    with torch.no_grad():
        for features, mask, labels, slide_ids in loader:
            if features is None:
                continue
            features, mask = features.to(device), mask.to(device)
            _, slide_pred, _ = model(features, mask)
            prob = torch.sigmoid(slide_pred).item()
            all_labels.append(labels[0].item())
            all_probs.append(prob)
            all_ids.extend(slide_ids)
    return np.array(all_labels), np.array(all_probs), all_ids


def train_fold(args, train_df, val_df, test_df, ext_test_df, device, fold_id):
    run_dir = os.path.join(args.output_dir, args.run_name, f'fold_{fold_id}')
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    with open(os.path.join(run_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    split_df = pd.concat([
        train_df.assign(split_type='training'),
        val_df.assign(split_type='validation')
    ], ignore_index=True)
    split_df[['sample_id', 'label', 'split_type']].to_csv(
        os.path.join(run_dir, 'data_split_summary.csv'), index=False)

    feats = {
        'train': args.train_feats_dir, 'val': args.train_feats_dir,
        'test': args.test_feats_dir, 'ext_test': args.ext_test_feats_dir,
    }
    train_dataset = H5SlideDataset(train_df, feats['train'],
                                   args.num_patches_sample, 'train', args.seed)
    val_dataset = H5SlideDataset(val_df, feats['val'],
                                 args.num_patches_sample, 'val', args.seed)
    test_dataset = H5SlideDataset(test_df, feats['test'],
                                  args.num_patches_sample, 'test', args.seed)
    ext_dataset = H5SlideDataset(ext_test_df, feats['ext_test'],
                                 args.num_patches_sample, 'test', args.seed)

    # ACMIL processes one slide at a time (batch_size=1)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)
    ext_loader = DataLoader(ext_dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    model = ACMILClassifier(
        input_dim=args.input_dim, D_inner=args.D_inner, D_attn=args.D_attn,
        n_token=args.n_token, n_masked_patch=args.n_masked_patch,
        mask_drop=args.mask_drop, head=args.head,
        classifier_dropout=args.classifier_dropout,
        temperature=args.temperature,
    ).to(device)

    print(f'Model: ACMILClassifier (head={args.head}, n_token={args.n_token})')

    pos = train_df['label'].sum()
    neg = len(train_df) - pos
    pos_weight = torch.tensor(neg / max(pos, 1), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs,
                                      args.num_epochs, args.min_lr, args.lr)

    best_val_bal_acc = 0.0
    epochs_no_improve = 0
    best_model_path = None
    best_metrics = None

    log_file = os.path.join(run_dir, 'training_log.csv')
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,val_auc,val_bal_acc,test_auc,test_bal_acc,'
                'ext_auc,ext_bal_acc,lr,optimal_threshold\n')

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for features, mask, labels, _ in train_loader:
            if features is None:
                continue
            features = features.to(device)
            mask = mask.to(device)
            label = labels[0].to(device)

            optimizer.zero_grad()
            sub_preds, slide_pred, attn_logits = model(features, mask)
            loss = compute_acmil_loss(sub_preds, slide_pred, attn_logits,
                                      label, criterion, args.lamda_aem)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        val_labels, val_probs, _ = evaluate(model, val_loader, device)
        test_labels, test_probs, _ = evaluate(model, test_loader, device)
        ext_labels, ext_probs, _ = evaluate(model, ext_loader, device)

        opt_thresh = 0.5
        if len(np.unique(val_labels)) > 1:
            opt_thresh = find_optimal_threshold(val_labels, val_probs)

        val_bal_acc = balanced_accuracy_score(
            val_labels, (val_probs > opt_thresh).astype(int))
        test_bal_acc = balanced_accuracy_score(
            test_labels, (test_probs > opt_thresh).astype(int))
        ext_bal_acc = balanced_accuracy_score(
            ext_labels, (ext_probs > opt_thresh).astype(int))
        val_auc = (roc_auc_score(val_labels, val_probs)
                   if len(np.unique(val_labels)) > 1 else 0.5)
        test_auc = (roc_auc_score(test_labels, test_probs)
                    if len(np.unique(test_labels)) > 1 else 0.5)
        ext_auc = (roc_auc_score(ext_labels, ext_probs)
                   if len(np.unique(ext_labels)) > 1 else 0.5)

        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f'Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.4f} | '
              f'LR: {lr:.6f}')
        print(f'  Val  | AUC: {val_auc:.4f} | BalAcc: {val_bal_acc:.4f}')
        print(f'  Test | AUC: {test_auc:.4f} | BalAcc: {test_bal_acc:.4f}')
        print(f'  Ext  | AUC: {ext_auc:.4f} | BalAcc: {ext_bal_acc:.4f}')

        with open(log_file, 'a') as f:
            f.write(f'{epoch+1},{avg_loss:.6f},{val_auc:.6f},{val_bal_acc:.6f},'
                    f'{test_auc:.6f},{test_bal_acc:.6f},{ext_auc:.6f},'
                    f'{ext_bal_acc:.6f},{lr:.6f},{opt_thresh:.4f}\n')

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        writer.add_scalar('BalAcc/val', val_bal_acc, epoch)

        # ---- Model selection: validation balanced accuracy only ----
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            epochs_no_improve = 0
            best_model_path = os.path.join(
                run_dir,
                f'best_model_epoch{epoch+1}_score{val_bal_acc:.4f}.pth')
            torch.save(model.state_dict(), best_model_path)
            best_metrics = {
                'epoch': epoch + 1, 'val_auc': val_auc,
                'val_bal_acc': val_bal_acc, 'test_auc': test_auc,
                'test_bal_acc': test_bal_acc, 'ext_auc': ext_auc,
                'ext_bal_acc': ext_bal_acc,
                'optimal_threshold': opt_thresh,
            }
            print(f'  -> New best (val_bal_acc={val_bal_acc:.4f})')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f'\nEarly stopping at epoch {epoch+1}.')
            break

    writer.close()
    if best_metrics:
        best_metrics['best_model_path'] = best_model_path
    return best_metrics


# ---- Main ----

def get_args():
    p = argparse.ArgumentParser(
        description='WSI UNI2-h ACMIL+AEM classification with k-fold CV.')
    p.add_argument('--train_feats_dir', type=str,
                   default='data/wsi_features/internal_training')
    p.add_argument('--test_feats_dir', type=str,
                   default='data/wsi_features/internal_testing')
    p.add_argument('--ext_test_feats_dir', type=str,
                   default='data/wsi_features/external_testing')
    p.add_argument('--dataset_csv', type=str, default='data/internal_dataset_split.csv')
    p.add_argument('--ext_test_csv', type=str,
                   default='data/external_dataset_split.csv')
    p.add_argument('--output_dir', type=str,
                   default='./outputs/wsi_acmil')
    p.add_argument('--run_name', type=str, default=None)
    # Model architecture
    p.add_argument('--head', type=str, default='linear',
                   choices=['linear', 'mlp'],
                   help='Slide classifier head type.')
    p.add_argument('--input_dim', type=int, default=1536)
    p.add_argument('--D_inner', type=int, default=768)
    p.add_argument('--D_attn', type=int, default=128)
    p.add_argument('--n_token', type=int, default=5,
                   help='Number of attention heads (5 for linear, 3 for mlp).')
    p.add_argument('--n_masked_patch', type=int, default=10)
    p.add_argument('--mask_drop', type=float, default=0.6)
    p.add_argument('--classifier_dropout', type=float, default=0.4)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--lamda_aem', type=float, default=0.1,
                   help='Weight for AEM entropy regularization loss.')
    # Training
    p.add_argument('--num_epochs', type=int, default=400)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=2e-4)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--num_patches_sample', type=int, default=1700)
    p.add_argument('--k_folds', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--gpu_id', type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.run_name is None:
        args.run_name = f'run_{datetime.datetime.now():%Y%m%d_%H%M%S}'

    device = torch.device(f'cuda:{args.gpu_id}'
                          if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    full_df = pd.read_csv(args.dataset_csv)
    train_data = full_df[full_df['split'] == 'training'].reset_index(drop=True)
    test_df = full_df[full_df['split'] == 'testing'].reset_index(drop=True)
    ext_test_df = pd.read_csv(args.ext_test_csv)
    ext_test_df = ext_test_df[ext_test_df['split'] == 'ext_test'].reset_index(
        drop=True)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True,
                          random_state=args.seed)
    all_fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(train_data, train_data['label']), start=1):
        print(f'\n{"="*60}\nFold {fold_idx}/{args.k_folds}\n{"="*60}')
        fold_train = train_data.iloc[train_idx].reset_index(drop=True)
        fold_val = train_data.iloc[val_idx].reset_index(drop=True)

        metrics = train_fold(args, fold_train, fold_val, test_df,
                             ext_test_df, device, fold_idx)
        if metrics:
            metrics['fold'] = fold_idx
            all_fold_metrics.append(metrics)

    summary_dir = os.path.join(args.output_dir, args.run_name)
    summary = {f'fold_{m["fold"]}': m for m in all_fold_metrics}
    with open(os.path.join(summary_dir, 'kfold_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    print('\nK-fold training complete.')


if __name__ == '__main__':
    main()
