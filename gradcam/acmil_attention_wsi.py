"""
ACMIL attention heatmap visualization for WSI slides.

Produces bag-level attention overlay on the full H&E WSI image for each slide.
No patient identifiers or slide IDs in figures or filenames.
Saved as 0000_bag.png, 0001_bag.png, ... (numeric index only).

Usage:
    python acmil_attention_wsi.py \\
        --run_dir <path_to_run> \\
        --test_feats_dir <h5_features_dir> \\
        --wsi_dir <wsi_files_dir> \\
        --dataset_csv <csv_path> \\
        --output_dir <output_dir> \\
        [--fold_id 1] [--n_token 3] [--split testing] [--gpu_id 0]

If openslide is unavailable or WSI files are not found, the script falls back
to plotting attention directly on the patch coordinate grid (no H&E background).
"""

import os
import sys
import json
import glob
import argparse
import warnings

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.collections import PatchCollection
from PIL import Image as _PILImage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from models import ACMILClassifier

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    warnings.warn('openslide not found — H&E overlay unavailable; using coordinate grid.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description='ACMIL bag-level attention visualization for WSI.')
    p.add_argument('--run_dir', type=str, required=True,
                   help='Run directory containing fold_N/ subdirs with checkpoints.')
    p.add_argument('--test_feats_dir', type=str,
                   default='data/wsi_features/internal_testing',
                   help='Directory with per-slide .h5 feature files.')
    p.add_argument('--wsi_dir', type=str, default='',
                   help='Directory containing WSI files (.ndpi/.svs/.tiff etc.). '
                        'Leave empty to skip H&E overlay.')
    p.add_argument('--dataset_csv', type=str, default='data/internal_dataset_split.csv')
    p.add_argument('--output_dir', type=str, default='./acmil_attention_wsi')
    p.add_argument('--fold_id', type=int, default=1,
                   help='Which fold checkpoint to use (default: 1).')
    p.add_argument('--n_token', type=int, default=3,
                   help='Number of ACMIL attention heads (must match model).')
    p.add_argument('--split', type=str, default='testing',
                   help='Dataset split to visualize (testing / ext_test).')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--bag_thumb_size', type=int, default=4000,
                   help='Long-edge pixel size for WSI thumbnail.')
    p.add_argument('--thumbnails_dir', type=str, default='',
                   help='Directory containing pre-generated per-slide thumbnail JPGs '
                        '(auto-detected from test_feats_dir if empty).')
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


def load_model(ckpt_path, hp, device):
    model = ACMILClassifier(
        input_dim=hp.get('input_dim', 1536),
        D_inner=hp.get('D_inner', 768),
        D_attn=hp.get('D_attn', 128),
        n_token=hp.get('n_token', 3),
        n_masked_patch=hp.get('n_masked_patch', 10),
        mask_drop=hp.get('mask_drop', 0.6),
        head='mlp' if 'classifier_dropout' in hp else 'linear',
        classifier_dropout=hp.get('classifier_dropout', 0.4),
    )
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


@torch.no_grad()
def extract_bag_attention(model, features_t, mask_t):
    """Run model forward and extract bag-level attention weights.

    Returns:
        slide_prob (float): sigmoid probability for the slide.
        bag_attn (np.ndarray): (N,) bag-level attention weights (softmax).
    """
    x = features_t.squeeze(0)           # (N, D)
    if mask_t is not None:
        mask = mask_t.squeeze(0)
    else:
        mask = None

    x_r = model.dimreduction(x)         # (N, D_inner)
    A = model.attention(x_r)            # (K, N) raw logits

    if mask is not None:
        pad_mask = (mask < 0.5)
        A = A.masked_fill(pad_mask.unsqueeze(0), -1e9)

    # Bag attention: average of per-token softmax distributions
    A_softmax = F.softmax(A, dim=1)           # (K, N)
    bag_A = A_softmax.mean(0, keepdim=True)   # (1, N)
    bag_feat = torch.mm(bag_A, x_r)           # (1, D_inner)

    slide_logit = model.Slide_classifier(bag_feat)
    slide_prob = torch.sigmoid(slide_logit).item()
    bag_attn = bag_A.squeeze(0).cpu().numpy()  # (N,)
    return slide_prob, bag_attn


def percentile_normalise(attn):
    """Rank-percentile normalisation: each value becomes its fractional rank in [0, 1]."""
    order = attn.argsort().argsort()
    return order.astype(np.float64) / max(order.max(), 1)


def detect_patch_size_l0(coords):
    """Auto-detect patch spacing (= patch size in level-0 pixels)."""
    spacings = []
    for axis in [0, 1]:
        vals = np.sort(np.unique(coords[:, axis]))
        if len(vals) > 1:
            diffs = np.diff(vals)
            spacings.append(int(diffs.min()))
    return min(spacings) if spacings else 512


def find_thumbnails_dir(feats_dir):
    """Auto-detect thumbnails dir: go up 2 levels from features_uni_v2/ and look for thumbnails/."""
    candidate = os.path.join(os.path.dirname(os.path.dirname(feats_dir)), 'thumbnails')
    return candidate if os.path.isdir(candidate) else ''


def find_wsi_path(wsi_dir, slide_id):
    if not wsi_dir:
        return None
    for ext in ('.ndpi', '.svs', '.tiff', '.tif', '.mrxs', '.scn'):
        p = os.path.join(wsi_dir, f'{slide_id}{ext}')
        if os.path.isfile(p):
            return p
    return None


def read_full_wsi_thumbnail(wsi_path, thumbnails_dir, slide_id, long_edge=4000):
    """Return (thumbnail_rgb, scale) or (None, None) if unavailable.

    Tries three methods in order:
    1. openslide — most accurate, generates thumbnail on-the-fly.
    2. Pre-generated thumbnail JPG + PIL header read of WSI file for scale.
    3. Returns (None, None) — caller falls back to coordinate grid.
    """
    # Method 1: openslide
    if HAS_OPENSLIDE and wsi_path and os.path.isfile(wsi_path):
        try:
            wsi = openslide.OpenSlide(wsi_path)
            W, H = wsi.dimensions
            thumb = wsi.get_thumbnail((long_edge, long_edge)).convert('RGB')
            wsi.close()
            scale = max(thumb.size) / max(W, H)
            return np.array(thumb), scale
        except Exception as exc:
            warnings.warn(f'openslide failed for {wsi_path}: {exc}')

    # Method 2: pre-generated thumbnail + PIL header-only read for WSI dims
    if thumbnails_dir:
        thumb_path = os.path.join(thumbnails_dir, f'{slide_id}.jpg')
        if not os.path.isfile(thumb_path):
            thumb_path = os.path.join(thumbnails_dir, f'{slide_id}.png')
        if os.path.isfile(thumb_path) and wsi_path and os.path.isfile(wsi_path):
            try:
                thumb = _PILImage.open(thumb_path).convert('RGB')
                W_thumb, H_thumb = thumb.size
                # Read WSI level-0 dimensions from file header (no pixel decode)
                old_limit = _PILImage.MAX_IMAGE_PIXELS
                _PILImage.MAX_IMAGE_PIXELS = None
                wsi_header = _PILImage.open(wsi_path)
                W_wsi, H_wsi = wsi_header.size
                _PILImage.MAX_IMAGE_PIXELS = old_limit
                # Scale: thumbnail was generated at fixed max-edge resolution
                scale = max(W_thumb, H_thumb) / max(W_wsi, H_wsi)
                print(f'  [thumbnail] {slide_id}: WSI {W_wsi}x{H_wsi}, '
                      f'thumb {W_thumb}x{H_thumb}, scale={scale:.5f}')
                return np.array(thumb), scale
            except Exception as exc:
                warnings.warn(f'PIL thumbnail fallback failed for {slide_id}: {exc}')

    return None, None


# ---------------------------------------------------------------------------
# Figure generation (bag-level only)
# ---------------------------------------------------------------------------

def render_bag_panel(coords, bag_attn, patch_size_l0, he_full, he_scale, out_path):
    """Single-panel bag-level attention heatmap, no patient info."""
    attn_norm = percentile_normalise(bag_attn)
    cmap = plt.cm.jet
    colours = cmap(attn_norm)
    colours[:, 3] = 0.70  # alpha transparency (matches reference heatmap style)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    if he_full is not None and he_scale is not None:
        ax.imshow(he_full)
        tc = (coords * he_scale).astype(np.float64)
        ps = patch_size_l0 * he_scale
        rects = [mpatches.Rectangle((cx, cy), ps, ps) for cx, cy in tc]
    else:
        # Fallback: coordinate grid without H&E
        cw = coords[:, 0].max() + patch_size_l0
        ch = coords[:, 1].max() + patch_size_l0
        rects = [mpatches.Rectangle((cx, cy), patch_size_l0, patch_size_l0)
                 for cx, cy in coords]
        ax.set_xlim(0, cw)
        ax.set_ylim(ch, 0)
        ax.set_aspect('equal')
        ax.set_facecolor('#f0f0f0')

    pc = PatchCollection(rects, facecolor=colours, edgecolor='none', linewidth=0)
    ax.add_collection(pc)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label('Attention percentile', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


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
    print(f'openslide: {"available" if HAS_OPENSLIDE else "NOT available"}')

    # Auto-detect thumbnails directory from feats dir
    thumbnails_dir = args.thumbnails_dir
    if not thumbnails_dir:
        thumbnails_dir = find_thumbnails_dir(args.test_feats_dir)
        if thumbnails_dir:
            print(f'Auto-detected thumbnails dir: {thumbnails_dir}')
        else:
            print('No thumbnails dir found — will use coordinate grid fallback')

    # Load hyperparameters
    hp_path = os.path.join(args.run_dir, f'fold_{args.fold_id}', 'hyperparameters.json')
    hp = {}
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            hp = json.load(f)

    # Load checkpoint
    fold_dir = os.path.join(args.run_dir, f'fold_{args.fold_id}')
    ckpt = find_best_ckpt(fold_dir)
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint found in {fold_dir}')
    print(f'Loading checkpoint: {os.path.basename(ckpt)}')
    model = load_model(ckpt, hp, device)

    # Load dataset
    df = pd.read_csv(args.dataset_csv)
    split_val = 'testing' if args.split == 'testing' else args.split
    df = df[df['split'] == split_val].reset_index(drop=True)
    print(f'Slides in split "{split_val}": {len(df)}')

    os.makedirs(args.output_dir, exist_ok=True)

    count = 0
    for _, row in df.iterrows():
        slide_id = str(row['sample_id'])
        h5_path = os.path.join(args.test_feats_dir, f'{slide_id}.h5')
        if not os.path.isfile(h5_path):
            print(f'  Skipping {slide_id}: H5 not found at {h5_path}')
            continue

        try:
            with h5py.File(h5_path, 'r') as f:
                features = torch.tensor(f['features'][:], dtype=torch.float32)
                coords = f['coords'][:]
        except Exception as exc:
            warnings.warn(f'Could not load {h5_path}: {exc}')
            continue

        features_t = features.unsqueeze(0).to(device)
        mask_t = torch.ones(1, features.shape[0], dtype=torch.float32, device=device)

        slide_prob, bag_attn = extract_bag_attention(model, features_t, mask_t)
        patch_size_l0 = detect_patch_size_l0(coords)

        # Load WSI thumbnail
        wsi_path = find_wsi_path(args.wsi_dir, slide_id)
        he_full, he_scale = read_full_wsi_thumbnail(
            wsi_path, thumbnails_dir, slide_id, args.bag_thumb_size)
        if wsi_path and he_full is None and not HAS_OPENSLIDE:
            print(f'  Note: no H&E background for {slide_id} (pass --wsi_dir for tissue background)')
        elif wsi_path and he_full is None:
            print(f'  Warning: WSI not loaded for {slide_id}')

        out_path = os.path.join(args.output_dir, f'{count:04d}_bag.png')
        render_bag_panel(coords, bag_attn, patch_size_l0, he_full, he_scale, out_path)

        count += 1
        if count % 5 == 0:
            print(f'  Saved {count} attention maps...')

    print(f'Done. {count} bag-level attention maps saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
