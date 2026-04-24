"""
GradCAM visualization for LE ResNet-18 patch classifier.

Generates 2-panel figures: Original H&E | GradCAM Overlay.

Saved as 0000.png, 0001.png, ... (numeric index only).

Uses XGradCAM targeting ResNet-18 layer4[-1].

Usage:
    python gradcam_le_resnet18.py \\
        --run_dir <path_to_run> \\
        --patches_dir <dir_with_slide_subdirs> \\
        --dataset_csv <csv_with_sample_id_label_split_cols> \\
        --output_dir <output_dir> \\
        [--fold_id 1] [--split testing] [--num_samples 50] [--gpu_id 0]
"""

import os
import sys
import json
import glob
import argparse
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from models import ResNetClassifier

try:
    from pytorch_grad_cam import XGradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError(
        'pytorch-grad-cam is required. Install with: pip install grad-cam')


# ---------------------------------------------------------------------------
# Target wrapper for binary classifier
# ---------------------------------------------------------------------------

class _BinaryTarget:
    """Directs XGradCAM to explain the predicted class."""
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        # model_output is a 1-D tensor (batch squeezed) or scalar
        logit = model_output[0] if model_output.dim() > 0 else model_output
        return logit if self.category == 1 else -logit


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((260, 260), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Same spatial crop but no normalization — for visualization as H&E image
DISPLAY_TRANSFORM = transforms.Compose([
    transforms.Resize((260, 260), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description='GradCAM visualization for LE ResNet-18.')
    p.add_argument('--run_dir', type=str, required=True,
                   help='Run directory (contains fold_N/ subdirs with checkpoints).')
    p.add_argument('--patches_dir', type=str, required=True,
                   help='Root directory of slide patch subdirs.')
    p.add_argument('--dataset_csv', type=str, default='data/le_dataset_split.csv')
    p.add_argument('--output_dir', type=str, default='./gradcam_le_resnet18')
    p.add_argument('--fold_id', type=int, default=1,
                   help='Which fold checkpoint to use (default: 1).')
    p.add_argument('--split', type=str, default='testing',
                   help='Dataset split to visualize (testing / training / ext_test).')
    p.add_argument('--num_samples', type=int, default=0,
                   help='Max patches to visualize across all slides. 0 = all.')
    p.add_argument('--gpu_id', type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_best_ckpt(fold_dir):
    ckpts = glob.glob(os.path.join(fold_dir, 'best_model*.pth'))
    if not ckpts:
        ckpts = glob.glob(os.path.join(fold_dir, '*.pth'))
    if not ckpts:
        return None
    try:
        ckpts.sort(key=lambda x: float(x.split('score')[1].split('.pth')[0]), reverse=True)
    except Exception:
        ckpts.sort()
    return ckpts[0]


def load_model(ckpt_path, dropout, device):
    model = ResNetClassifier(pretrained=False, dropout=dropout)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def make_figure(he_img_np, cam_mask):
    """Create 2-panel figure: original H&E | GradCAM overlay.
    No text, no patient info.
    """
    float_img = he_img_np.astype(np.float32) / 255.0
    if cam_mask.shape != float_img.shape[:2]:
        cam_mask = cv2.resize(cam_mask, (float_img.shape[1], float_img.shape[0]))
    overlay = show_cam_on_image(float_img, cam_mask, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.05)
    axes[0].imshow(he_img_np)
    axes[0].set_title('H&E', fontsize=18, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title('GradCAM', fontsize=18, fontweight='bold')
    axes[1].axis('off')
    plt.tight_layout()
    return fig


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

    # Load hyperparameters
    hp_path = os.path.join(args.run_dir, f'fold_{args.fold_id}', 'hyperparameters.json')
    hp = {}
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            hp = json.load(f)
    dropout = hp.get('dropout', 0.4)

    # Load checkpoint
    fold_dir = os.path.join(args.run_dir, f'fold_{args.fold_id}')
    ckpt = find_best_ckpt(fold_dir)
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint found in {fold_dir}')
    print(f'Loading checkpoint: {os.path.basename(ckpt)}')
    model = load_model(ckpt, dropout, device)

    # Setup XGradCAM on layer4[-1]
    target_layers = [model.resnet.layer4[-1]]
    cam = XGradCAM(model=model, target_layers=target_layers)

    # Load dataset
    df = pd.read_csv(args.dataset_csv)
    df = df[df['split'] == args.split].reset_index(drop=True)
    print(f'Slides in split "{args.split}": {len(df)}')

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_rows = []

    global_count = 0
    limit = args.num_samples if args.num_samples > 0 else float('inf')

    for _, row in df.iterrows():
        if global_count >= limit:
            break
        slide_id = str(row['sample_id'])
        slide_dir = os.path.join(args.patches_dir, slide_id)
        if not os.path.isdir(slide_dir):
            print(f'  Skipping {slide_id}: directory not found')
            continue

        exts = ('*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg')
        patch_files = []
        for e in exts:
            patch_files.extend(glob.glob(os.path.join(slide_dir, e)))
        if not patch_files:
            continue

        for img_path in patch_files:
            if global_count >= limit:
                break
            try:
                img_pil = Image.open(img_path).convert('RGB')

                # Tensor for inference/GradCAM
                input_t = EVAL_TRANSFORM(img_pil).unsqueeze(0).to(device)

                # Prediction
                with torch.no_grad():
                    logit = model(input_t)
                    prob = torch.sigmoid(logit).item()
                pred_class = 1 if prob >= 0.5 else 0

                # GradCAM
                targets = [_BinaryTarget(pred_class)]
                cam_map = cam(input_tensor=input_t, targets=targets)[0]  # (H, W)

                # Display image (same spatial crop, no normalization)
                he_np = np.array(DISPLAY_TRANSFORM(img_pil))

                fig = make_figure(he_np, cam_map)
                out_path = os.path.join(args.output_dir, f'{global_count:04d}.png')
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

                manifest_rows.append({
                    'index': global_count,
                    'output_file': f'{global_count:04d}.png',
                    'slide_id': slide_id,
                    'patch_file': os.path.basename(img_path),
                    'pred_prob': float(prob),
                    'pred_class': pred_class,
                })

                global_count += 1
                if global_count % 20 == 0:
                    print(f'  Saved {global_count} images...')

            except Exception as exc:
                warnings.warn(f'Error on {img_path}: {exc}')
                continue

    if manifest_rows:
        manifest_path = os.path.join(args.output_dir, 'manifest.csv')
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f'Manifest saved to: {manifest_path}')
    print(f'Done. {global_count} GradCAM images saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
