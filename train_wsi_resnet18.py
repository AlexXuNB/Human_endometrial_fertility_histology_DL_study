"""
Training script for WSI-level ResNet-18 patch classification with k-fold CV.

Model: ResNet-18 (ImageNet-pretrained) -> 512-d -> Dropout -> Linear(512, 1)
Input: Raw image patches (3 x 224 x 224) from WSI
Aggregation: Patch-level training; slide-level prediction via mean pooling
Loss: BCEWithLogitsLoss with class-weight balancing
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (roc_auc_score, roc_curve, balanced_accuracy_score,
                             accuracy_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from models import ResNetClassifier


# ---- Learning Rate Scheduler ----

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

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

class PatchDataset(Dataset):
    """Loads individual patches from slide folders for training."""

    def __init__(self, df, patches_dir, transform=None):
        self.records = []
        for _, row in df.iterrows():
            raw = row['sample_id']
            sid = str(int(raw)) if isinstance(raw, (int, float)) else str(raw)
            label = int(row['label'])
            slide_dir = os.path.join(patches_dir, sid)
            if not os.path.isdir(slide_dir):
                continue
            for fname in os.listdir(slide_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.records.append((os.path.join(slide_dir, fname), label, sid))
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label, sid = self.records[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), sid


class SlidePatchDataset(Dataset):
    """Loads all patches for each slide at evaluation time."""

    def __init__(self, df, patches_dir, transform=None):
        self.slides = []
        for _, row in df.iterrows():
            raw = row['sample_id']
            sid = str(int(raw)) if isinstance(raw, (int, float)) else str(raw)
            label = int(row['label'])
            slide_dir = os.path.join(patches_dir, sid)
            if not os.path.isdir(slide_dir):
                continue
            patch_paths = sorted([
                os.path.join(slide_dir, f) for f in os.listdir(slide_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
            ])
            if patch_paths:
                self.slides.append((patch_paths, label, sid))
        self.transform = transform

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        patch_paths, label, sid = self.slides[idx]
        patches = []
        for p in patch_paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            patches.append(img)
        patches = torch.stack(patches)
        return patches, torch.tensor(label, dtype=torch.float32), sid


# ---- Metrics ----

def calculate_metrics(y_true, y_score, y_pred):
    metrics = {}
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        metrics['Sensitivity'] = 'N/A'
        metrics['Specificity'] = 'N/A'
    metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['AUC'] = (roc_auc_score(y_true, y_score)
                      if len(np.unique(y_true)) > 1 else 0.5)
    return metrics


def find_optimal_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return thresholds[np.argmax(j)]


# ---- Training ----

def get_transforms(is_train=True):
    # Match legacy training_resnet18_mil_kfold_linear.py: patches already 224x224,
    # use a plain Resize((224,224)) (default BILINEAR) with no CenterCrop.
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                   saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def evaluate_slides(model, loader, device, threshold=0.5, eval_batch=64):
    """Evaluate model at slide level (mean pooling of patch predictions).

    Iterates a DataLoader over a SlidePatchDataset (batch_size=1). Each
    batch yields the stacked patches for one slide (shape (1, N, 3, H, W)
    after default collation). Patches are processed in sub-batches to
    avoid GPU OOM.
    """
    model.eval()
    all_labels, all_probs, all_ids = [], [], []
    with torch.no_grad():
        for patches, label, sid in loader:
            # default_collate stacks single-item batch -> (1, N, 3, H, W)
            if patches.dim() == 5:
                patches = patches.squeeze(0)
            all_patch_probs = []
            for i in range(0, len(patches), eval_batch):
                batch = patches[i:i + eval_batch].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(batch)
                probs = torch.sigmoid(logits.float())
                all_patch_probs.append(probs.cpu())
            slide_prob = torch.cat(all_patch_probs).mean().item()
            all_labels.append(label.item() if torch.is_tensor(label) else float(label[0]))
            all_probs.append(slide_prob)
            all_ids.append(sid[0] if isinstance(sid, (list, tuple)) else sid)
    return np.array(all_labels), np.array(all_probs), all_ids


def train_fold(args, train_df, val_df, test_df, ext_test_df, device, fold_id):
    run_dir = os.path.join(args.output_dir, args.run_name, f'fold_{fold_id}')
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    # Save hyperparameters
    with open(os.path.join(run_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Save data split
    split_df = pd.concat([
        train_df.assign(split_type='training'),
        val_df.assign(split_type='validation')
    ], ignore_index=True)
    split_df[['sample_id', 'label', 'split_type']].to_csv(
        os.path.join(run_dir, 'data_split_summary.csv'), index=False)

    # Datasets
    train_dataset = PatchDataset(train_df, args.patches_dir,
                                 transform=get_transforms(True))
    val_slide_dataset = SlidePatchDataset(val_df, args.patches_dir,
                                          transform=get_transforms(False))
    test_slide_dataset = SlidePatchDataset(test_df, args.test_patches_dir,
                                           transform=get_transforms(False))
    ext_slide_dataset = SlidePatchDataset(ext_test_df, args.ext_patches_dir,
                                          transform=get_transforms(False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=(args.num_workers > 0))

    eval_workers = min(args.num_workers, 8)
    val_loader = DataLoader(val_slide_dataset, batch_size=1, shuffle=False,
                            num_workers=eval_workers, pin_memory=True)
    test_loader = DataLoader(test_slide_dataset, batch_size=1, shuffle=False,
                             num_workers=eval_workers, pin_memory=True)
    ext_test_loader = DataLoader(ext_slide_dataset, batch_size=1, shuffle=False,
                                 num_workers=eval_workers, pin_memory=True)

    # Model
    model = ResNetClassifier(pretrained=True, dropout=args.dropout).to(device)

    # Loss with class balancing
    pos = train_df['label'].sum()
    neg = len(train_df) - pos
    pos_weight = torch.tensor(neg / max(pos, 1), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs,
                                      args.num_epochs, args.min_lr, args.lr)
    scaler = torch.cuda.amp.GradScaler()

    best_val_bal_acc = 0.0
    epochs_no_improve = 0
    best_model_path = None
    best_metrics = None

    log_file = os.path.join(run_dir, 'training_log.csv')
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,val_auc,val_bal_acc,test_auc,test_bal_acc,'
                'ext_test_auc,ext_test_bal_acc,lr,optimal_threshold\n')

    for epoch in range(args.num_epochs):
        # ---- Train ----
        model.train()
        total_loss = 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)

        # ---- Evaluate ----
        val_labels, val_probs, val_ids = evaluate_slides(
            model, val_loader, device)
        test_labels, test_probs, test_ids = evaluate_slides(
            model, test_loader, device)
        ext_labels, ext_probs, ext_ids = evaluate_slides(
            model, ext_test_loader, device)

        # Optimal threshold from validation set
        opt_thresh = 0.5
        if len(np.unique(val_labels)) > 1:
            opt_thresh = find_optimal_threshold(val_labels, val_probs)

        val_preds = (val_probs > opt_thresh).astype(int)
        test_preds = (test_probs > opt_thresh).astype(int)
        ext_preds = (ext_probs > opt_thresh).astype(int)

        val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
        test_bal_acc = balanced_accuracy_score(test_labels, test_preds)
        ext_bal_acc = balanced_accuracy_score(ext_labels, ext_preds)
        val_auc = (roc_auc_score(val_labels, val_probs)
                   if len(np.unique(val_labels)) > 1 else 0.5)
        test_auc = (roc_auc_score(test_labels, test_probs)
                    if len(np.unique(test_labels)) > 1 else 0.5)
        ext_auc = (roc_auc_score(ext_labels, ext_probs)
                   if len(np.unique(ext_labels)) > 1 else 0.5)

        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f'Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.4f} | '
              f'LR: {lr:.6f} | Thresh: {opt_thresh:.4f}')
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
        writer.add_scalar('AUC/test', test_auc, epoch)
        writer.add_scalar('BalAcc/test', test_bal_acc, epoch)

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
            print(f'  -> New best model (val_bal_acc={val_bal_acc:.4f})')
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
        description='WSI ResNet-18 patch classification with k-fold CV.')
    p.add_argument('--patches_dir', type=str,
                   default='data/wsi_patches_224px/internal_training')
    p.add_argument('--test_patches_dir', type=str,
                   default='data/wsi_patches_224px/internal_testing')
    p.add_argument('--ext_patches_dir', type=str,
                   default='data/wsi_patches_224px/external_testing')
    p.add_argument('--dataset_csv', type=str, default='data/internal_dataset_split.csv')
    p.add_argument('--ext_test_csv', type=str,
                   default='data/external_dataset_split.csv')
    p.add_argument('--output_dir', type=str, default='./outputs/wsi_resnet18')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--dropout', type=float, default=0.4)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_epochs', type=int, default=400)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=150)
    p.add_argument('--k_folds', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=32)
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
    ext_test_df = ext_test_df[ext_test_df['split'] == 'ext_test'].reset_index(drop=True)

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

    # Summary
    summary_dir = os.path.join(args.output_dir, args.run_name)
    summary = {f'fold_{m["fold"]}': m for m in all_fold_metrics}
    with open(os.path.join(summary_dir, 'kfold_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    print('\nK-fold training complete.')


if __name__ == '__main__':
    main()
