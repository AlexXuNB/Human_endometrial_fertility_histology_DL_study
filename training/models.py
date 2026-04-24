"""
Model architecture definitions for WSI and LE classification.

This module defines all model architectures used in the study:
1. ResNetClassifier: ResNet-18 backbone with linear classification head
2. PoolingClassifier: Global mean pooling with Linear or MLP head
3. ACMILClassifier: Attention-Challenging MIL with Linear or MLP head

Feature extractors:
- ResNet-18: Processes raw image patches (3 x 224 x 224), outputs 512-dim features
- UNI2-h: Pre-trained vision foundation model producing 1536-dim patch embeddings (frozen)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


# ---------------------------------------------------------------------------
# 1. ResNet-18 Classifier (patch-level classification)
# ---------------------------------------------------------------------------

class ResNetClassifier(nn.Module):
    """ResNet-18 based patch classifier

    Architecture:
        ResNet-18 (ImageNet-pretrained) -> AdaptiveAvgPool -> 512-d
        -> Dropout -> Linear(512, 1)

    Slide-level prediction is obtained by averaging patch-level sigmoid
    probabilities at inference time.
    """

    def __init__(self, pretrained=True, dropout=0.25):
        super().__init__()
        if pretrained:
            self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)
        # Remove the default fully-connected layer (output remains 512-d)
        self.resnet.fc = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) image patches
        Returns:
            logits: (B,) binary classification logits
        """
        features = self.resnet(x)          # (B, 512)
        features = self.dropout(features)
        logits = self.fc(features).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# 2. Global Mean Pooling Classifier (feature-level classification)
# ---------------------------------------------------------------------------

class PoolingClassifier(nn.Module):
    """Global pooling classifier for pre-extracted patch features.

    Aggregates patch-level features via global mean pooling, then classifies
    with either a single linear layer or a 3-layer MLP.

    Args:
        input_dim: Patch feature dimension (1536 for UNI2-h).
        head: 'linear' for single Linear layer, 'mlp' for 3-layer MLP.
        dropout: Dropout rate (used in MLP head only).
        pooling: 'mean', 'max', or 'both'.
    """

    def __init__(self, input_dim=1536, head='mlp', dropout=0.4, pooling='mean'):
        super().__init__()
        self.pooling = pooling

        if pooling == 'both':
            classifier_input_dim = input_dim * 2
        else:
            classifier_input_dim = input_dim

        if head == 'linear':
            self.classifier = nn.Linear(classifier_input_dim, 1)
        elif head == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(768, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(384, 1),
            )
        else:
            raise ValueError(f"Unknown head type: {head}")

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, D) patch features
            mask: (B, N) binary mask (1 = valid patch, 0 = padding)
        Returns:
            logits: (B,) binary classification logits
        """
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device)

        if self.pooling == 'mean':
            sum_features = torch.sum(x * mask.unsqueeze(-1), dim=1)
            count = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
            aggregated = sum_features / count
        elif self.pooling == 'max':
            masked_x = x.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            aggregated = torch.max(masked_x, dim=1)[0]
        elif self.pooling == 'both':
            sum_features = torch.sum(x * mask.unsqueeze(-1), dim=1)
            count = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
            mean_feat = sum_features / count
            masked_x = x.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            max_feat = torch.max(masked_x, dim=1)[0]
            aggregated = torch.cat([mean_feat, max_feat], dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return self.classifier(aggregated).squeeze(-1)


# ---------------------------------------------------------------------------
# 3. ACMIL Classifier (Attention-Challenging MIL)
# ---------------------------------------------------------------------------

class DimReduction(nn.Module):
    """Linear dimension reduction with ReLU activation.

    Reduces high-dimensional patch embeddings to a lower-dimensional space
    for attention computation.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return F.relu(self.fc(x))


class Attention_Gated(nn.Module):
    """Gated attention mechanism (Ilse et al., 2018).

    Computes K attention distributions over N patches using a gated
    architecture:  A = Linear_c( Tanh(Linear_a(x)) * Sigmoid(Linear_b(x)) )

    Args:
        L: Input feature dimension.
        D: Attention hidden dimension.
        K: Number of attention heads (token branches).
    """

    def __init__(self, L, D, K):
        super().__init__()
        self.attention_a = nn.Linear(L, D)
        self.attention_b = nn.Linear(L, D)
        self.attention_c = nn.Linear(D, K)

    def forward(self, x):
        """
        Args:
            x: (N, L) patch features
        Returns:
            A: (K, N) attention logits
        """
        a = torch.tanh(self.attention_a(x))       # (N, D)
        b = torch.sigmoid(self.attention_b(x))     # (N, D)
        return self.attention_c(a * b).T           # (K, N)


class ACMILClassifier(nn.Module):
    """Attention-Challenging MIL (ACMIL) classifier with AEM regularization.

    Architecture:
        1. DimReduction:  input_dim -> D_inner  (Linear + ReLU)
        2. Gated Attention:  D_inner -> D_attn -> n_token heads
        3. Per-token classifiers:  n_token x Linear(D_inner, 1)
        4. Bag-level classifier:  Linear(D_inner, 1) or MLP

    During training, top-attended patches are randomly masked to encourage
    attention diversity (ACMIL mechanism). AEM entropy regularization further
    promotes uniform attention distributions.

    References:
        - Zhang et al., ECCV 2024 (ACMIL)
        - Zhang et al., MICCAI 2025 (AEM)

    Args:
        input_dim: Patch feature dimension (1536 for UNI2-h).
        D_inner: Intermediate feature dimension after reduction.
        D_attn: Hidden dimension of the attention module.
        n_token: Number of attention heads / per-token classifiers.
        n_masked_patch: Number of top patches considered for masking.
        mask_drop: Fraction of top patches to mask during training.
        head: 'linear' for Linear slide classifier, 'mlp' for MLP.
        classifier_dropout: Dropout rate for MLP slide classifier.
        temperature: Softmax temperature for attention computation.
    """

    def __init__(self, input_dim=1536, D_inner=768, D_attn=128, n_token=5,
                 n_masked_patch=10, mask_drop=0.6, head='linear',
                 classifier_dropout=0.4, temperature=1.0):
        super().__init__()
        self.n_token = n_token
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.temperature = temperature

        # Dimension reduction: input_dim -> D_inner
        self.dimreduction = DimReduction(input_dim, D_inner)

        # Gated attention: D_inner -> D_attn -> n_token branches
        self.attention = Attention_Gated(D_inner, D_attn, n_token)

        # Per-token classifiers
        self.classifier = nn.ModuleList(
            [nn.Linear(D_inner, 1) for _ in range(n_token)]
        )

        # Bag-level (slide) classifier
        if head == 'linear':
            self.Slide_classifier = nn.Linear(D_inner, 1)
        elif head == 'mlp':
            self.Slide_classifier = nn.Sequential(
                nn.Linear(D_inner, D_inner),
                nn.LayerNorm(D_inner),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(D_inner, D_inner // 2),
                nn.LayerNorm(D_inner // 2),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(D_inner // 2, 1),
            )
        else:
            raise ValueError(f"Unknown head type: {head}")

    def forward(self, x, mask=None):
        """
        Args:
            x: (1, N, input_dim) patch features for one slide
            mask: (1, N) binary mask (1 = valid, 0 = padded)
        Returns:
            sub_preds: (n_token,) per-token predictions
            slide_pred: (1,) bag-level prediction
            attn_logits: (1, n_token, N) raw attention logits
        """
        x = x.squeeze(0)                          # (N, input_dim)
        if mask is not None:
            mask = mask.squeeze(0)                 # (N,)

        # Step 1: Dimension reduction
        x = self.dimreduction(x)                   # (N, D_inner)

        # Step 2: Compute attention logits
        A = self.attention(x)                      # (K, N)

        # Step 3: Apply padding mask
        if mask is not None:
            pad_mask = (mask < 0.5)
            A = A.masked_fill(pad_mask.unsqueeze(0), -1e9)

        # Step 4: ACMIL attention masking (training only)
        if self.n_masked_patch > 0 and self.training:
            k, n = A.shape
            n_masked = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked, dim=-1)
            rand_sel = torch.argsort(
                torch.rand(*indices.shape, device=A.device), dim=-1
            )[:, :int(n_masked * self.mask_drop)]
            masked_idx = indices[
                torch.arange(k, device=A.device).unsqueeze(-1), rand_sel
            ]
            rmask = torch.ones(k, n, device=A.device)
            rmask.scatter_(-1, masked_idx, 0)
            A = A.masked_fill(rmask == 0, -1e9)

        A_out = A                                  # (K, N) raw logits

        # Step 5: Softmax -> weighted features -> per-token predictions
        A = F.softmax(A / self.temperature, dim=1)
        afeat = torch.mm(A, x)                     # (K, D_inner)

        outputs = [head(afeat[i]) for i, head in enumerate(self.classifier)]

        # Step 6: Bag feature and bag-level prediction
        bag_A = F.softmax(A_out / self.temperature, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)              # (1, D_inner)
        slide_pred = self.Slide_classifier(bag_feat).squeeze(-1)

        return (torch.stack(outputs, dim=0).squeeze(-1),
                slide_pred,
                A_out.unsqueeze(0))
